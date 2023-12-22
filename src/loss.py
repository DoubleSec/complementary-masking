import torch
from torch import nn


class BarlowTwinsLoss(nn.Module):
    def __init__(self, lambda_: float = 5e-3):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x1, x2):
        # both are n x e
        # Assuming x1 and x2 have been normalized already
        c = x1.T @ x2

        # Divide by the batch size
        c = c / x1.shape[0]

        on_diag = ((1 - torch.diagonal(c)) ** 2).sum()
        off_diag = ((c * (1 - torch.eye(c.shape[0], device=c.device))) ** 2).sum()
        return on_diag + self.lambda_ * off_diag
