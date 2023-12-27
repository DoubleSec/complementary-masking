import torch
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import MLFlowLogger
import yaml
import mlflow

# For linear probing
# import polars as pl
# import numpy as np
# from sklearn.linear_model import LogisticRegression
# from sklearn import metrics

from src.data import PretrainingDataset
from src.net import RogersNet
from src.arg_parsers import train_parser, update_config

# Setup -------------------------

# Load config
with open("./cfg/config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.CLoader)

mp = config["model_params"]
# Update the config if any command line arguments are set
mp = update_config(train_parser.parse_args(), mp)
tp = config["training_params"]

# Set a seed
torch.manual_seed(config["split_seed"])

# Can I use the tensor cores?
torch.set_float32_matmul_precision("medium")

# Set mlflow uri: expecting local server
mlflow.set_tracking_uri(uri="http://127.0.0.1:8081")

# Create a dataset ---------------

# TKTK data module I guess.

ds = PretrainingDataset(
    parquet_path=config["train_data_path"],
    cols=config["features"],
    key_cols=config["keys"],
    # aux_cols=config["aux_cols"],
)

train_ds, validation_ds, test_ds = torch.utils.data.random_split(
    ds, lengths=[0.75, 0.15, 0.1]
)

train_dl = torch.utils.data.DataLoader(
    dataset=train_ds,
    batch_size=tp["batch_size"],
    num_workers=10,
    shuffle=True,
)

validation_dl = torch.utils.data.DataLoader(
    dataset=validation_ds,
    batch_size=tp["batch_size"],
    num_workers=10,
)

test_dl = torch.utils.data.DataLoader(
    dataset=test_ds,
    batch_size=tp["batch_size"],
    num_workers=10,
)

# Initialize the network ----------------

net = RogersNet(
    morphers=ds.morphers,
    **mp,
)

# Train ----------------

mlflow.set_experiment(config["mlflow_experiment"])
with mlflow.start_run() as run:
    mlflow.log_dict(config, "config.yaml")
    mlflow.log_params(tp)

    trainer = Trainer(
        accelerator="gpu",
        max_epochs=tp["epochs"],
        # Default behavior for both, but we're being explicit.
        logger=MLFlowLogger(run_id=run.info.run_id, log_model=True),
        log_every_n_steps=10,
    )

    trainer.fit(net, train_dataloaders=train_dl, val_dataloaders=validation_dl)
