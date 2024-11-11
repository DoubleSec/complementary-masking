import polars as pl
import torch

import morphers

# So we can re-use easily enough.
DEFAULT_MORPHER_DISPATCH = {
    "numeric": morphers.Quantiler,
    "categorical": morphers.Integerizer,
}


def make_dispatch(md):
    return {
        ctype: getattr(morphers, morpher_name) for ctype, morpher_name in md.items()
    }


class PitchDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        parquet_path: str,
        input_cols: dict | None = None,
        input_morphers: dict | None = None,
        target_cols: dict | None = None,
        target_morphers: dict | None = None,
        key_cols: list | None = None,
        aux_cols: list | None = None,
    ):
        self.key_cols = key_cols if key_cols is not None else []
        self.aux_cols = aux_cols if aux_cols is not None else []
        self.target_cols = target_cols if target_cols is not None else []

        ds = pl.read_parquet(parquet_path)

        # Set up morphers

        # If there's no prior morpher states, we create them.
        if input_morphers is None:
            self.input_morphers = {
                feature: morpher_class.from_data(ds[feature], **kwargs)
                for feature, (morpher_class, kwargs) in input_cols.items()
            }
        # Otherwise we load their stuff from the state dict.
        else:
            self.input_morphers = input_morphers

        # Same for targets
        if len(self.target_cols) > 0:
            if target_morphers is None:
                self.target_morphers = {
                    feature: morpher_class.from_data(ds[feature], **kwargs)
                    for feature, (morpher_class, kwargs) in target_cols.items()
                }
            # Otherwise we load their stuff from the state dict.
            else:
                self.target_morphers = target_morphers
        else:
            self.target_morphers = {}

        # Transform the dataset using morphers, and selected required columns
        self.ds = (
            ds.select(
                # keys
                *[pl.col(key) for key in self.key_cols],
                # morphed inputs
                *[
                    morpher(pl.col(feature))
                    for feature, morpher in self.input_morphers.items()
                ],
                # targets
                *[
                    morpher(pl.col(feature))
                    for feature, morpher in self.target_morphers.items()
                ],
                # Auxiliary columns
                *[pl.col(ac) for ac in self.aux_cols],
            )
            # Only drop nulls based on inputs
            .drop_nulls(
                [feature for feature in self.input_morphers | self.target_morphers]
            )
        )
        if len(self.aux_cols) > 0:
            self.ds = self.ds.drop_nulls(aux_cols)

    def __len__(self):
        return self.ds.height

    def __getitem__(self, idx):
        row = self.ds.row(idx, named=True)
        inputs = {
            k: torch.tensor(row[k], dtype=morpher.required_dtype)
            for k, morpher in self.input_morphers.items()
        }
        targets = {
            k: torch.tensor(row[k], dtype=morpher.required_dtype)
            for k, morpher in self.target_morphers.items()
        }

        return_dict = inputs | targets
        # These may be empty
        return_dict = return_dict | {key: row[key] for key in self.key_cols}
        return_dict |= {col: row[col] for col in self.aux_cols}

        return return_dict


class LinearProbeDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        targets: pl.DataFrame,
        embeddings: torch.Tensor,
        embedding_subset: int | None = None,
    ):
        self.targets = targets
        self.embeddings = embeddings[:, :embedding_subset]

        assert targets.height == embeddings.shape[0]

    def __len__(self):
        return self.embeddings.shape[0]

    def __getitem__(self, idx):
        row = self.targets.row(idx, named=True)
        return {col: torch.tensor(row[col], dtype=torch.float32) for col in row} | {
            "embeddings": self.embeddings[idx, :]
        }
