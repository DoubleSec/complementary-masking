import torch
from torch import nn
import lightning.pytorch as pl
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC

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
        # We'll log these manually later.
        self.save_hyperparameters(logger=False)
        # We'll do morpher saving here for minimal error-possibilities.
        self.morphers = morphers

        # Some behavior for predictions
        self.predict_cols = None

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
        # Custom hyperparameter logging.
        self.logger.log_hyperparams(
            {k: v for k, v in self.hparams.items() if k != "morphers"}
        )
        self.logger.experiment.log_dict(
            run_id=self.logger.run_id,
            dictionary={feat: m.save_state_dict() for feat, m in self.morphers.items()},
            artifact_file="morphers/morphers.yaml",
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

    def predict_step(self, x):
        y_hat = self.inference_forward(x)
        if self.predict_cols is not None:
            extra_cols = {col: x[col] for col in self.predict_cols}
            return y_hat, extra_cols
        else:
            return y_hat


class LinearProbeNet(pl.LightningModule):
    """Small network for linear probing."""

    def __init__(
        self,
        embedding_dim: int,
        n_layers: int,
        targets: str,
        lr: float,
        weight_decay: float,
    ):
        super().__init__()
        # We'll log these manually later.
        self.save_hyperparameters(logger=False)

        # Only one target, please.
        # I'm not modifying the rest yet.
        self.targets = targets if isinstance(targets, list) else [targets]

        if n_layers == 1:
            self.probe = nn.Sequential(
                nn.LayerNorm(embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, len(self.targets)),
            )
        else:
            raise NotImplementedError("I'll get this later.")

        # Loss and metrics

        self.loss = nn.BCEWithLogitsLoss(reduction="none")
        metrics = MetricCollection(
            {
                # "accuracy": BinaryAccuracy(),
                "AUROC": BinaryAUROC(),
            }
        )
        # Make a metric collection for each target.
        # These are just in the target order.
        self.train_metrics = [
            metrics.clone(prefix=f"{target}_train_") for target in self.targets
        ]
        self.validation_metrics = [
            metrics.clone(prefix=f"{target}_validation_") for target in self.targets
        ]

    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

    def on_fit_start(self):
        # This is punishment for my hubris.
        self.train_metrics = [metric.to(self.device) for metric in self.train_metrics]
        self.validation_metrics = [
            metric.to(self.device) for metric in self.validation_metrics
        ]

    def forward(self, x):
        return self.probe(x)

    def _step(self, x):
        y_hat = self(x["embeddings"])
        y = torch.stack([x[target] for target in self.targets], dim=-1)
        loss = self.loss(y_hat, y)
        return loss, y, y_hat

    def training_step(self, x):
        loss, y, y_hat = self._step(x)

        # Log the loss per target.
        self.log_dict(
            {
                f"{target}_train_loss": loss[:, i].mean()
                for i, target in enumerate(self.targets)
            }
        )
        # Log training metrics
        for i, metrics in enumerate(self.train_metrics):
            self.log_dict(metrics(y_hat[:, i], y[:, i].int()))

        return loss.mean()

    def validation_step(self, x):
        loss, y, y_hat = self._step(x)

        # Log the loss per target.
        self.log_dict(
            {
                f"{target}_validation_loss": loss[:, i].mean()
                for i, target in enumerate(self.targets)
            }
        )
        # Log training metrics
        for i, metrics in enumerate(self.validation_metrics):
            self.log_dict(metrics(y_hat[:, i], y[:, i].int()))

        return loss.mean()
