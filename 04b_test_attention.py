import tempfile
from itertools import chain

import torch
import yaml
import mlflow
import polars as pl

from src.data import PretrainingDataset
from src.net import RogersNet
from src.diag import predict_with_attention

# Setup -------------------------

# Load config
with open("./cfg/lp_config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.CLoader)

ms = config["model_source"]

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
    shuffle=False,
)

projections, attentions = predict_with_attention(net, test_dl, "cuda:0")

# Get a list of the 4 attention maps from the 4 layers, for the first batch.
example_attentions = attentions[0]

result = torch.eye(example_attentions[0].shape[-1], device="cuda:0")

# Going from the lowest (earliest) layer to the highest (latest) layer.
# Just doing mean fusion here, no dropping because the code doesn't make sense.
for att in example_attentions:
    # att is batch x heads x s x s
    # After fusion, batch x s x s
    att_fused = att.mean(dim=1)

    # This is to represent residual connections.
    I = torch.eye(att_fused.shape[-1], device=att_fused.device)
    a = (att_fused + 1.0 * I) / 2

    result = a @ result

untransformed_data = test_ds.dataset.ds[test_ds.indices, :]
print(untransformed_data.filter(pl.col("release_speed").list.get(1) > 0))
print(untransformed_data)

# untransformed_data = test_ds.ds[:2048, :]
# print(untransformed_data)
