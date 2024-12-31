import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import yaml

from src.data import PitchDataset, make_dispatch
from src.net.model import RogersNet
from src.net.extenders import BarlowPretrainer

# Setup -------------------------

# Load config
with open("./cfg/config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.CLoader)
tp = config["training_params"]

# Set a seed
torch.manual_seed(config["split_seed"])

# Create a dataset ---------------

# TKTK data module I guess.

morpher_dispatch = make_dispatch(config["morpher_dispatch"])
inputs = {
    col: (morpher_dispatch[tp], kwargs) for [col, tp, kwargs] in config["features"]
}

ds = PitchDataset(
    parquet_path=config["train_data_path"],
    input_cols=inputs,
)

train_ds, validation_ds, test_ds = torch.utils.data.random_split(
    ds, lengths=[0.75, 0.15, 0.1]
)

train_dl = torch.utils.data.DataLoader(
    dataset=train_ds,
    batch_size=tp["batch_size"],
    num_workers=10,
    shuffle=True,
    drop_last=True,
)

validation_dl = torch.utils.data.DataLoader(
    dataset=validation_ds,
    batch_size=tp["batch_size"],
    num_workers=10,
    drop_last=True,
)

# Train ----------------

# Can I use the tensor cores?
torch.set_float32_matmul_precision("medium")

trainer = Trainer(
    accelerator="gpu",
    max_epochs=tp["epochs"],
    log_every_n_steps=10,
    logger=TensorBoardLogger(
        save_dir="./logs", name=config["experiment_name"], default_hp_metric=False
    ),
    callbacks=[ModelCheckpoint(monitor="validation_loss", save_top_k=1)],
    num_sanity_val_steps=0,
)

# Initialize the network down here, to initialize on GPU with float16
with trainer.init_module():

    core_net_args = config["core_net_params"] | {"morphers": ds.input_morphers}
    extender_args = {
        "n_features": len(ds.input_morphers),
        "embedding_size": core_net_args["embedding_size"],
    } | config["bt_params"]

    net = RogersNet(
        core_net_args=core_net_args,
        extender=BarlowPretrainer,
        extender_args=extender_args,
    )
    net.compile()

trainer.fit(net, train_dataloaders=train_dl, val_dataloaders=validation_dl)
