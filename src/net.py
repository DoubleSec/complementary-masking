from abc import ABC, abstractmethod
from itertools import chain

import torch
from torch import nn
import lightning.pytorch as pl
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAUROC

from .network_layers import (
    FeatureEmbedder,
    FeatureMasker,
    ProjectionHead,
    LearnedPositionEncoding,
)
from .loss import (
    BarlowTwinsLoss,
    InfoNCELoss,
    MatroshkaTwinsLoss,
    PositionWeightedBarlowTwins,
)
from .camalambakicken import Transformer, RMSNorm

LOSS_OPTIONS = {
    "Barlow twins": BarlowTwinsLoss,
    "Matroshka twins": MatroshkaTwinsLoss,
    "PW Barlow twins": PositionWeightedBarlowTwins,
    "InfoNCE": InfoNCELoss,
}


class CoreNet(nn.Module):
    """Core of network shared for any task."""

    def __init__(
        self,
        morphers: dict,
        embedding_size: int,
        tr_n_layers: int,
        n_kv_heads: int,
        n_q_heads: int,
        ff_dim: int,
    ):
        super().__init__()
        self.morphers = morphers

        # Feature embedder
        self.embedding_layer = FeatureEmbedder(
            morphers=morphers,
            output_size=embedding_size,
            gather="stack",
        )

        # Layer norm for features
        self.feature_norm = RMSNorm(embedding_size)

        # Positional Encoding
        self.positional_encoding = LearnedPositionEncoding(
            max_length=len(morphers),
            d_model=embedding_size,
        )

        self.transformer = Transformer(
            tr_n_layers,
            layer_args={
                "d_model": embedding_size,
                "n_kv_heads": n_kv_heads,
                "n_q_heads": n_q_heads,
                "ff_dim": ff_dim,
            },
        )

        # cls token
        self.register_parameter(
            "cls", nn.Parameter(torch.randn([1, 1, embedding_size]) * 0.02)
        )

    def forward(self, x):
        """Generic forward for inference or fine-tuning."""
        x = self.embedding_layer(x)
        x = self.feature_norm(x)
        x = self.positional_encoding(x)
        x = self.transformer(x)
        x = torch.cat([x, self.cls.expand([x.shape[0], -1, -1])], dim=1)
        x = x[:, -1, :]
        return x


class Extender(nn.Module, ABC):

    def __init__(self, *args, **kwargs):
        """Initialize anything needed for the extender."""
        super().__init__()

    @abstractmethod
    def forward(self, core_net: CoreNet, x):
        """Generic inference forward"""
        raise NotImplementedError

    def training_forward(self, core_net: CoreNet, x):
        """Forward for training. May be the same as forward."""
        return self.forward(core_net, x)

    @abstractmethod
    def configure_optimizers(self, core_net):
        """Configure optimizers hook to use with lightning."""
        raise NotImplementedError

    @abstractmethod
    def loss(self, *args, **kwargs):
        """Loss calculation for extender."""
        raise NotImplementedError


class BarlowPretrainer(Extender):

    def __init__(
        self,
        n_features: int,
        embedding_size: int,
        mask_p: float,
        masking_strategy: str,
        projection_size: int,
        proj_n_layers: int,
        loss_params: dict,
        lr: float,
        weight_decay: float,
    ):
        super().__init__()

        self.masking_layer = FeatureMasker(
            n_features=n_features,
            input_size=embedding_size,
            p=mask_p,
            masking_strategy=masking_strategy,
            return_complement=True,
        )

        self.projection_head = ProjectionHead(
            input_size=embedding_size,
            output_size=projection_size,
            n_layers=proj_n_layers,
            norm_type=RMSNorm,
            activation_type=nn.GELU,
        )

        # Loss, metrics, etc.
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss = BarlowTwinsLoss(**loss_params)

    def configure_optimizers(self, core_net):
        return torch.optim.Adam(
            params=chain(self.parameters(), core_net.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )


class RogersNet(pl.LightningModule):
    def __init__(
        self,
        core_net_args: dict,
        extender_class,  # I don't know the right type hint.
        extender_args: dict,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.core_net = CoreNet(**core_net_args)
        self.extender = extender_class(**extender_args)

    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )


class RogersNet(pl.LightningModule):
    def __init__(
        self,
        morphers: dict,
        embedding_size: int,
        mask_p: float,
        masking_strategy: str,
        projection_size: int,
        tr_n_layers: int,
        tr_type: str,
        tr_args: dict,
        proj_n_layers: int,
        loss_type: str,
        loss_params: dict,
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
        self.feature_norm = (
            nn.LayerNorm(embedding_size)
            if tr_type == "basic"
            else RMSNorm(embedding_size)
        )

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

        if tr_type == "llama":
            norm_type = RMSNorm
            activation_type = nn.GELU

            layer_args = {"d_model": embedding_size} | tr_args
            self.transformer = Transformer(tr_n_layers, layer_args=layer_args)

        elif tr_type == "basic":
            norm_type = nn.LayerNorm
            activation_type = nn.ReLU

            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=embedding_size,
                    **tr_args,
                    batch_first=True,
                ),
                num_layers=tr_n_layers,
            )

        else:
            raise ValueError("tr_type must be 'llama' or 'basic'")

        self.projection_head = ProjectionHead(
            input_size=embedding_size,
            output_size=projection_size,
            n_layers=proj_n_layers,
            norm_type=norm_type,
            activation_type=activation_type,
        )

        # Loss, metrics, etc.
        self.lr = lr
        self.weight_decay = weight_decay
        loss_class = LOSS_OPTIONS.get(loss_type, "lolwut")
        assert (
            loss_class != "lolwut"
        ), f"Loss class must be one of {', '.join(LOSS_OPTIONS.keys())}"

        self.loss = loss_class(**loss_params)

    def on_train_start(self):
        # Custom hyperparameter logging.
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

        loss = self.loss(proj1, proj2)
        self.log("train_loss", loss)

        return loss

    def validation_step(self, x):
        proj1, proj2 = self(x)

        loss = self.loss(proj1, proj2)
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
