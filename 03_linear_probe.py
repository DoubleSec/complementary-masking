import tempfile
from itertools import chain

import torch
from lightning.pytorch import Trainer
import yaml
import mlflow
import polars as pl

# For linear probing
# import polars as pl
# import numpy as np
# from sklearn.linear_model import LogisticRegression
# from sklearn import metrics

from src.data import PretrainingDataset, LinearProbeDataset
from src.net import RogersNet

# Setup -------------------------

# Load config
with open("./lp_config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.CLoader)

# mp = config["model_params"]
# tp = config["training_params"]

# Set a seed
torch.manual_seed(config["split_seed"])

# Can I use the tensor cores?
torch.set_float32_matmul_precision("medium")

# Set mlflow uri: expecting local server
mlflow.set_tracking_uri(uri="http://127.0.0.1:8081")

# Get the checkpoint from mlflow ---------------

with tempfile.TemporaryDirectory() as temp_dir:
    # Model checkpoint
    ckpt_base_path = mlflow.artifacts.download_artifacts(
        run_id=config["run_id"],
        artifact_path=f"model/checkpoints/{config['checkpoint_name']}",
        dst_path=temp_dir,
    )
    ckpt_path = f"{ckpt_base_path}/{config['checkpoint_name']}.ckpt"

    net = RogersNet.load_from_checkpoint(
        checkpoint_path=ckpt_path,
    )
    morpher_states = {feat: m.save_state_dict() for feat, m in net.morphers.items()}


# Create a dataset ---------------

# TKTK data module I guess.

ds = PretrainingDataset(
    parquet_path=config["train_data_path"],
    cols=config["features"],
    key_cols=config["keys"],
    aux_cols=config["aux_cols"],
    morpher_states=morpher_states,  # Does this work?
)

_, _, test_ds = torch.utils.data.random_split(ds, lengths=[0.75, 0.15, 0.1])

test_dl = torch.utils.data.DataLoader(
    dataset=test_ds,
    batch_size=2048,
    num_workers=10,
)

# Get projections
net.predict_cols = config["aux_cols"]

pred_trainer = Trainer(accelerator="gpu", logger=False)
results = pred_trainer.predict(net, test_dl)

# print(results)
print(results)

aux_data = pl.DataFrame(
    {
        col: chain.from_iterable([batch[1][col] for batch in results])
        for col in results[0][1]
    }
)
targets = aux_data.filter(~pl.col("pitch_type").is_null()).select(
    fastball=pl.col("pitch_type").eq("FT").cast(pl.Float32),
    changeup=pl.col("pitch_type").eq("CH").cast(pl.Float32),
    slider=pl.col("pitch_type").eq("SL").cast(pl.Float32),
    cureveball=pl.col("pitch_type").eq("CU").cast(pl.Float32),
)
embeddings = torch.cat([batch[0] for batch in results], dim=0)[
    ~aux_data["pitch_type"].is_null().to_numpy(), :
]

lp_ds = LinearProbeDataset(targets=targets, embeddings=embeddings)

print(lp_ds[30])
