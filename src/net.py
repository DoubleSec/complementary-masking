from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
import torch
from torch import nn
import lightning.pytorch as pl

from .network_layers import (
    FeatureEmbedder,
    FeatureMasker,
    ProjectionHead,
    LearnedPositionEncoding,
)
from .loss import BarlowTwinsLoss


class RogersNet(pl.LightningModule):
    def __init__(
        self,
        morphers: dict,
        embedding_size: int,
        mask_p: float,
        masking_strategy: str,
        projection_size: int,
        tr_nhead: int,
        tr_dim_ff: int,
        tr_n_layers: int,
        proj_n_layers: int,
        bt_lambda: float,
        lr: float,
        weight_decay: float,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Feature embedder
        self.embedding_layer = FeatureEmbedder(
            morphers=morphers,
            output_size=embedding_size,
            gather="stack",
        )

        # Layer norm for features
        self.feature_norm = nn.LayerNorm(embedding_size)

        self.masking_layer = FeatureMasker(
            morphers=morphers,
            input_size=embedding_size,
            p=mask_p,
            masking_strategy=masking_strategy,
            return_complement=True,
        )

        # Positional Encoding
        self.positional_encoding = LearnedPositionEncoding(
            max_length=len(morphers),
            d_model=embedding_size,
        )

        # cls token
        self.register_parameter(
            "cls", nn.Parameter(torch.randn([1, 1, embedding_size]) * 0.02)
        )

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_size,
                nhead=tr_nhead,
                dim_feedforward=tr_dim_ff,
                batch_first=True,
            ),
            num_layers=tr_n_layers,
        )

        self.projection_head = ProjectionHead(
            input_size=embedding_size,
            output_size=projection_size,
            n_layers=proj_n_layers,
        )

        self.projection_norm = nn.BatchNorm1d(projection_size, affine=False)

        # Loss, metrics, etc.
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss = BarlowTwinsLoss(lambda_=bt_lambda)

    def on_train_start(self):
        self.logger.log_hyperparams(
            {k: v for k, v in self.hparams.items() if k != "morphers"}
        )

    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    def forward(self, x):
        x = self.embedding_layer(x)
        x = self.feature_norm(x)
        x1, x2 = self.masking_layer(x)

        normed_cls = self.feature_norm(self.cls)

        # Everything twice now
        x1 = self.positional_encoding(x1)
        x1 = torch.cat([x1, normed_cls.expand([x1.shape[0], -1, -1])], dim=1)
        x1 = self.transformer(x1)
        x1 = self.projection_head(x1[:, -1, :])

        x2 = self.positional_encoding(x2)
        x2 = torch.cat([x2, normed_cls.expand([x2.shape[0], -1, -1])], dim=1)
        x2 = self.transformer(x2)
        x2 = self.projection_head(x2[:, -1, :])

        return x1, x2

    def training_step(self, x):
        proj1, proj2 = self(x)
        np1 = self.projection_norm(proj1)
        np2 = self.projection_norm(proj2)

        loss = self.loss(np1, np2)
        self.log("train_loss", loss)

        return loss

    def validation_step(self, x):
        proj1, proj2 = self(x)
        np1 = self.projection_norm(proj1)
        np2 = self.projection_norm(proj2)

        loss = self.loss(np1, np2)
        self.log("validation_loss", loss)

        return loss

    def inference_forward(self, x):
        """Same as normal forward but it skips masking and only returns once."""
        x = self.embedding_layer(x)
        x = self.feature_norm(x)

        normed_cls = self.feature_norm(self.cls)

        x = self.positional_encoding(x)
        x = torch.cat([x, normed_cls.expand([x.shape[0], -1, -1])], dim=1)
        x = self.transformer(x)
        x = self.projection_head(x[:, -1, :])

        return x
