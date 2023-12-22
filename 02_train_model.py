import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import yaml

from src.data import PretrainingDataset
from src.net import RogersNet

# Setup -------------------------

# Load config
with open("./config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.CLoader)

mp = config["model_params"]
tp = config["training_params"]

# Set a seed
torch.manual_seed(config["split_seed"])

# Can I use the tensor cores?
torch.set_float32_matmul_precision("medium")

# Create a dataset ---------------

# TKTK data module I guess.

ds = PretrainingDataset(
    parquet_path=config["train_data_path"],
    cols=config["features"],
    key_cols=config["keys"],
)

train_ds, validation_ds, test_ds = torch.utils.data.random_split(
    ds, lengths=[0.75, 0.15, 0.1]
)

train_dl = torch.utils.data.DataLoader(
    dataset=train_ds,
    batch_size=tp["batch_size"],
    num_workers=6,
    shuffle=True,
)

validation_dl = torch.utils.data.DataLoader(
    dataset=validation_ds,
    batch_size=tp["batch_size"],
    num_workers=6,
)

test_dl = torch.utils.data.DataLoader(
    dataset=test_ds,
    batch_size=tp["batch_size"],
    num_workers=6,
)

# Initialize the network ----------------

net = RogersNet(
    morphers=ds.morphers,
    **mp,
)

# Train ----------------

trainer = Trainer(
    accelerator="gpu",
    max_epochs=tp["epochs"],
    # Default behavior for both, but we're being explicit.
    logger=TensorBoardLogger(save_dir="lightning_logs"),
    callbacks=[ModelCheckpoint()],
    log_every_n_steps=20,
)

trainer.fit(net, train_dataloaders=train_dl, val_dataloaders=validation_dl)
