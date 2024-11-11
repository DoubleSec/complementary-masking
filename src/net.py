from abc import ABC, abstractmethod
from itertools import chain
from typing_extensions import Any

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
    def name(self):
        """Name of the extender, for labelling loss."""
        raise NotImplementedError

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
    def loss(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        """Loss calculation for extender."""
        raise NotImplementedError


class BarlowPretrainer(Extender):

    def __init__(
        self,
        name: str,
        n_features: int,
        embedding_size: int,
        mask_p: float,
        masking_strategy: str,
        projection_size: int,
        proj_n_layers: int,
        loss_params: dict,
        lr: float,
        weight_decay: float = 0.01,
    ):
        super().__init__()
        self.extender_name = name

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
        self.bt_loss = BarlowTwinsLoss(**loss_params)

    @property
    def name(self):
        return self.extender_name

    def configure_optimizers(self, core_net):
        return torch.optim.Adam(
            params=chain(self.parameters(), core_net.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    def loss(self, input, x):
        return {self.name: self.bt_loss(x[0], x[1])}

    def training_forward(self, core_net, x):
        x = core_net.embedding_layer(x)
        x = core_net.feature_norm(x)
        x1, x2 = self.masking_layer(x)

        # Everything twice now
        x1 = core_net.positional_encoding(x1)
        x1 = torch.cat([x1, core_net.cls.expand([x1.shape[0], -1, -1])], dim=1)
        x1 = core_net.transformer(x1)
        x1 = self.projection_head(x1[:, -1, :])

        x2 = core_net.positional_encoding(x2)
        x2 = torch.cat([x2, core_net.cls.expand([x2.shape[0], -1, -1])], dim=1)
        x2 = core_net.transformer(x2)
        x2 = self.projection_head(x2[:, -1, :])

        return x1, x2

    def forward(self, core_net, x):
        """Not super useful for this model."""

        x = core_net.embedding_layer(x)
        x = core_net.feature_norm(x)

        x = core_net.positional_encoding(x)
        x = torch.cat([x, core_net.cls.expand([x.shape[0], -1, -1])], dim=1)
        x = core_net.transformer(x)
        x = self.projection_head(x[:, -1, :])

        return x


class Identity(Extender):
    """Simple identity extender, cannot be trained."""

    def __init__(self):
        super().__init__()
        self.extender_name = "identity"

    def name(self):
        return self.extender_name

    def forward(self, core_net: CoreNet, x):
        return core_net(x)

    def training_forward(self, core_net: CoreNet, x):
        raise NotImplementedError

    def configure_optimizers(self, core_net):
        raise NotImplementedError

    def loss(self, *args, **kwargs):
        raise NotImplementedError


class RogersNet(pl.LightningModule):
    def __init__(
        self,
        core_net_args: dict,
        extender: type[Extender],
        extender_args: dict[str, Any],
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.core_net = CoreNet(**core_net_args)
        self.extender = extender(**extender_args)

    @classmethod
    def load_with_core_checkpoint(cls, ckpt_path: str, extender, extender_args):
        ckpt = torch.load(ckpt_path)
        core_args = ckpt["hyper_parameters"]["core_net_args"]
        net = cls(core_args, extender, extender_args)
        missing_keys, unexpected_keys = net.load_state_dict(
            ckpt["state_dict"], strict=False
        )
        if len(missing_keys) > 0:
            raise ValueError(f"Expected keys missing from state dict: {missing_keys}")
        return net

    def on_train_start(self):
        core_params = {
            f"core_{k}": v
            for k, v in self.hparams["core_net_args"].items()
            if not isinstance(v, dict)
        }
        self.logger.log_hyperparams(core_params)
        self.logger.log_hyperparams(self.hparams["extender_args"])

    def configure_optimizers(self):
        return self.extender.configure_optimizers(self.core_net)

    def forward(self, x):
        return self.extender.forward(self.core_net, x)

    def training_step(self, x):

        y = self.extender.training_forward(self.core_net, x)

        losses = self.extender.loss(x, y)
        self.log_dict({f"train_{k}_loss": v for k, v in losses.items()})
        loss = sum(losses.values())
        self.log("train_loss", loss)
        return loss

    def validation_step(self, x):

        y = self.extender.training_forward(self.core_net, x)

        losses = self.extender.loss(x, y)
        self.log_dict({f"validation_{k}_loss": v for k, v in losses.items()})
        loss = sum(losses.values())
        self.log("validation_loss", loss)
        return loss


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
