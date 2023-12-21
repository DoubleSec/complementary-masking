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
        morpher_states: dict = None,
    ):
        self.key_cols = key_cols
        self.return_keys = return_keys

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

        return return_dict
