from lightning import LightningModule
from lightning.pytorch.callbacks import BaseFinetuning, Callback
from lightning.pytorch.utilities import grad_norm
from torchmetrics import Metric
import torch
from logzero import logger


# First, uniformity and alignment metrics


def uniformity(x: torch.Tensor) -> torch.Tensor:
    """Function to calculate uniformity metric during pretraining.

    x: tensor of size n x e.

    Returns a scalar."""

    # Squared Euclidean distance between pairs
    sq_pdist = torch.pdist(x, p=2).pow(2)
    # See the paper for details. Roughly, it's smaller as the vectors are more spread out.
    return sq_pdist.mul(-2).exp().mean().log()


def alignment(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Calculates alignment.

    This is literally just average squared distance between matching pairs.
    Smaller = better matches.

    Also returns a scalar."""
    return (x - y).norm(dim=1).pow(2).mean()


class Uniformity(Metric):

    def __init__(self):
        super().__init__()
        self.add_state("running_mean", default=torch.tensor(0), dist_reduce_fx="mean")
        self.add_state("n", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, x) -> None:

        x_size = x.shape[0] ** 2
        sq_pdist = torch.pdist(x, p=2).pow(2)
        val = sq_pdist.mul(-2).exp().sum()

        self.running_mean = (self.running_mean * self.n + val) / (self.n + x_size)
        self.n += x_size

    def compute(self) -> torch.Tensor:
        return self.running_mean.log().float()


class Alignment(Metric):

    def __init__(self):
        super().__init__()
        self.add_state("running_mean", default=torch.tensor(0), dist_reduce_fx="mean")
        self.add_state("n", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, x, y) -> None:

        x_size = x.shape[0]
        val = (x - y).norm(dim=1).pow(2)
        self.running_mean = (self.running_mean * self.n + val) / (self.n + x_size)
        self.n += x_size

    def compute(self) -> torch.Tensor:

        return self.running_mean.float()


class SimpleFineTuner(BaseFinetuning):
    """Class to finetune RogersNet models."""

    def __init__(self, frozen_epochs):
        super().__init__()
        self.frozen_epochs = frozen_epochs

    def freeze_before_training(self, pl_module: LightningModule) -> None:

        logger.info("Freezing core net parameters")
        self.freeze(pl_module.core_net)

    def finetune_function(
        self, pl_module: LightningModule, current_epoch, optimizer
    ) -> None:

        if current_epoch == self.frozen_epochs:
            logger.info(f"Unfreezing core net parameters at epoch {current_epoch}")
            self.make_trainable(pl_module.core_net)


class GradNormMonitor(Callback):

    def on_before_optimizer_step(
        self, trainer, pl_module: LightningModule, optimizer
    ) -> None:
        norms = grad_norm(pl_module, norm_type=2)
        pl_module.log_dict(norms)
