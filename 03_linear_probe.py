import tempfile
from itertools import chain

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import MLFlowLogger
import yaml
import mlflow
import polars as pl

from src.data import PretrainingDataset, LinearProbeDataset
from src.net import RogersNet, LinearProbeNet
from src.arg_parsers import lp_parser, update_config

# Setup -------------------------

# Load config
with open("./cfg/lp_config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.CLoader)


ms = config["model_source"]
# Update the config if any command line arguments are set
ms = update_config(lp_parser.parse_args(), ms)
mp = config["model_params"]
tp = config["training_params"]
tp = update_config(lp_parser.parse_args(), tp)

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
        run_id=ms["run_id"],
        artifact_path=f"model/checkpoints/{ms['checkpoint_name']}",
        dst_path=temp_dir,
    )
    ckpt_path = f"{ckpt_base_path}/{ms['checkpoint_name']}.ckpt"

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
    morpher_states=morpher_states,
    morpher_dispatch=config["morpher_dispatch"],
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

aux_data = pl.DataFrame(
    {
        col: chain.from_iterable([batch[1][col] for batch in results])
        for col in results[0][1]
    }
)
# TKTK should come from config
targets = aux_data.filter(~pl.col("pitch_type").is_null()).select(
    fastball=pl.col("pitch_type").eq("FF").cast(pl.Float32),
    changeup=pl.col("pitch_type").eq("CH").cast(pl.Float32),
    slider=pl.col("pitch_type").eq("SL").cast(pl.Float32),
    curveball=pl.col("pitch_type").eq("CU").cast(pl.Float32),
)
embeddings = torch.cat([batch[0] for batch in results], dim=0)[
    ~aux_data["pitch_type"].is_null().to_numpy(), :
]

lp_ds = LinearProbeDataset(
    targets=targets, embeddings=embeddings, embedding_subset=tp["embedding_subset"]
)
train_ds, valid_ds = torch.utils.data.random_split(lp_ds, lengths=[0.75, 0.25])

train_dl = torch.utils.data.DataLoader(
    dataset=train_ds,
    batch_size=tp["batch_size"],
    num_workers=10,
)
valid_dl = torch.utils.data.DataLoader(
    dataset=valid_ds,
    batch_size=tp["batch_size"],
    num_workers=10,
)

# Create the linear probe model -------

mlflow.set_experiment(config["mlflow_experiment"])
with mlflow.start_run() as run:
    mlflow.log_dict(config, "config.yaml")
    mlflow.log_params(tp)
    mlflow.log_params(mp)
    mlflow.log_params(
        {f"source_{k}": v for k, v in net.hparams.items() if k != "morphers"}
    )

    # We need to train each linear probe separately.
    for target in mp["targets"]:
        lp_net = LinearProbeNet(
            embedding_dim=tp["embedding_subset"],
            **{k: v for k, v in mp.items() if k != "targets"},
            targets=target,
        )

        trainer = Trainer(
            accelerator="gpu",
            max_epochs=tp["epochs"],
            # Default behavior for both, but we're being explicit.
            logger=MLFlowLogger(run_id=run.info.run_id, log_model=tp["log_model"]),
            log_every_n_steps=10,
        )

        trainer.fit(lp_net, train_dataloaders=train_dl, val_dataloaders=valid_dl)
