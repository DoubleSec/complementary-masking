from typing import Any

import torch
import lightning.pytorch as pl
from logzero import logger


from .extenders import Extender
from .core import CoreNet


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
        ckpt = torch.load(ckpt_path, weights_only=False)
        core_args = ckpt["hyper_parameters"]["core_net_args"]
        net = cls(core_args, extender, extender_args)
        missing_keys, _ = net.load_state_dict(ckpt["state_dict"], strict=False)
        missing_keys = [key for key in missing_keys if "core_net" in key]
        if len(missing_keys) > 0:
            raise ValueError(f"Expected keys missing from state dict: {missing_keys}")
        logger.info(f"Loaded core weights from {ckpt_path}")
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

        losses, metrics = self.extender.loss(x, y)
        self.log_dict({f"train_{k}_loss": v for k, v in losses.items()})
        if metrics is not None:
            self.log_dict({f"train_{k}": v for k, v in metrics.items()})
        loss = sum(losses.values())
        self.log("train_loss", loss)
        return loss

    def validation_step(self, x):

        y = self.extender.training_forward(self.core_net, x)

        losses, metrics = self.extender.loss(x, y)
        self.log_dict({f"validation_{k}_loss": v for k, v in losses.items()})
        if metrics is not None:
            self.log_dict({f"validation_{k}": v for k, v in metrics.items()})
        loss = sum(losses.values())
        self.log("validation_loss", loss)
        self.log("hp_metric", loss)
        return loss
