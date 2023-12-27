import polars as pl
import torch

import morphers

# So we can re-use easily enough.
DEFAULT_MORPHER_DISPATCH = {
    "numeric": morphers.Normalizer,
    "categorical": morphers.Integerizer,
}


class PretrainingDataset(torch.utils.data.Dataset):
    MORPHER_DISPATCH = DEFAULT_MORPHER_DISPATCH

    def __init__(
        self,
        parquet_path: str,
        cols: dict,
        key_cols: list,
        return_keys: bool = False,
        aux_cols: list = None,
        morpher_states: dict = None,
    ):
        self.key_cols = key_cols
        self.return_keys = return_keys
        self.aux_cols = aux_cols if aux_cols is not None else []

        ds = pl.read_parquet(parquet_path)

        # Set up morphers

        # If there's no prior morpher states, we create them.
        if morpher_states is None:
            self.morphers = {
                feature: self.MORPHER_DISPATCH[ftype].from_data(ds[feature])
                for feature, ftype in cols.items()
            }
        # Otherwise we load their stuff from the state dict.
        else:
            self.morphers = {
                feature: self.MORPHER_DISPATCH[ftype].from_state_dict(
                    morpher_states[feature]
                )
                for feature, ftype in cols.items()
            }

        # Transform the dataset using morphers, and selected required columns
        self.ds = (
            ds.select(
                # keys
                *[pl.col(key) for key in key_cols],
                # morphed inputs
                *[
                    morpher(pl.col(feature))
                    for feature, morpher in self.morphers.items()
                ],
                # Auxiliary columns
                *[pl.col(ac) for ac in self.aux_cols],
            )
            # Only drop nulls based on inputs
            .drop_nulls([feature for feature in self.morphers])
        )

    def __len__(self):
        return self.ds.height

    def __getitem__(self, idx):
        row = self.ds.row(idx, named=True)
        inputs = {
            k: torch.tensor(row[k], dtype=morpher.required_dtype)
            for k, morpher in self.morphers.items()
        }

        return_dict = inputs
        if self.return_keys:
            return_dict = return_dict | {key: row[key] for key in self.key_cols}

        if len(self.aux_cols) > 0:
            return_dict |= {col: row[col] for col in self.aux_cols}

        return return_dict


class LinearProbeDataset(torch.utils.data.Dataset):
    def __init__(self, targets: pl.DataFrame, embeddings: torch.Tensor):
        self.targets = targets
        self.embeddings = embeddings

        assert targets.height == embeddings.shape[0]

    def __len__(self):
        return self.embeddings.shape[0]

    def __getitem__(self, idx):
        row = self.targets.row(idx, named=True)
        return {col: torch.tensor(row[col], dtype=torch.float32) for col in row} | {
            "embeddings": self.embeddings[idx, :]
        }


if __name__ == "__main__":
    import yaml

    with open("./config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.CLoader)

    ds = PretrainingDataset(
        parquet_path=config["train_data_path"],
        cols=config["features"],
        key_cols=config["keys"],
        aux_cols=config["aux_cols"],
    )

    print(len(ds))
    print(ds[200])
