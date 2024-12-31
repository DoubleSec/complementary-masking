import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import yaml
import polars as pl

from src.data import PitchDataset, make_dispatch
from src.net.model import RogersNet
from src.net.extenders import RPClassifier
from src.utils import SimpleFineTuner, GradNormMonitor

"""This is specifically for ball/strike prediction."""

# Setup -------------------------

# Load config
with open("./cfg/ft_config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.CLoader)
tp = config["training_params"]

# Set a seed
torch.manual_seed(config["split_seed"])

# Load the checkpoint to get morphers
checkpoint = torch.load(config["source_checkpoint"], weights_only=False)
morphers = checkpoint["hyper_parameters"]["core_net_args"]["morphers"]
del checkpoint

# Create a dataset ---------------

morpher_dispatch = make_dispatch(config["morpher_dispatch"])
targets = {
    col: (morpher_dispatch[tp], kwargs) for [col, tp, kwargs] in config["targets"]
}

filter_function = lambda x: x.filter(pl.col("type") != "X")

ds = PitchDataset(
    parquet_path=config["train_data_path"],
    filter_function=filter_function,
    input_morphers=morphers,
    target_cols=targets,
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
    callbacks=[
        ModelCheckpoint(monitor="validation_loss", save_top_k=1),
        SimpleFineTuner(frozen_epochs=tp["frozen_epochs"]),
        GradNormMonitor(),
    ],
    num_sanity_val_steps=0,
)

# Initialize the network down here, to initialize on GPU with float16
with trainer.init_module():

    net = RogersNet.load_with_core_checkpoint(
        ckpt_path=config["source_checkpoint"],
        extender=RPClassifier,
        extender_args=config["predictor_head_params"],
    )
    net.compile()

trainer.fit(net, train_dataloaders=train_dl, val_dataloaders=validation_dl)
