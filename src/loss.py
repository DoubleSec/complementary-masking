import torch
from torch import nn


class BarlowTwinsLoss(nn.Module):
    def __init__(self, lambda_: float = 1):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x1, x2):
        # both are n x e
        # We're normalizing here batch-by-batch, like in the original paper.
        x1 = (x1 - x1.mean(0)) / x1.std(0)
        x2 = (x2 - x2.mean(0)) / x2.std(0)
        c = x1.T @ x2

        # Divide by the batch size
        c = c / x1.shape[0]

        on_diag = ((1 - torch.diagonal(c)) ** 2).sum()
        off_diag = ((c * (1 - torch.eye(c.shape[0], device=c.device))) ** 2).sum()
        # The extra division by embedding_size - 1 is so the total weight
        # of the off-diagonal terms is constant regardless of embedding size.
        return on_diag + (self.lambda_ / (c.shape[0] - 1)) * off_diag


class InfoNCELoss(nn.Module):
    """This is not the most feature-complete implementation of this loss, but
    its signature exactly matches the Barlow twins loss above.

    If you're going to set n_negatives, it probably should be meaningfully smaller
    than the batch size."""

    def __init__(self, n_negatives: int = None, reduction: str = "sum"):
        super().__init__()
        self.n_negatives = n_negatives
        self.reduction = reduction

    def forward(self, x1, x2):
        # both are n x e
        # Assuming both are normalized already.
        if self.n_negatives is None:
            # logits is n x n
            logits = x1 @ x2.T
            labels = torch.arange(x1.shape[0], device=x1.device)
        else:
            # n
            real_labels = torch.arange(x1.shape[0], device=x1.device)
            # n x 1
            positive_logits = (x1 * x2).sum(dim=1, keepdim=True)
            # n x k-1
            negative_indices = torch.randint(
                low=0,
                high=x2.shape[0] - 1,
                size=[x1.shape[0], self.n_negatives],
                device=x1.device,
            )
            # Ooooookay, so if the negative index is less than the target index,
            # we leave it alone.
            # If it's greater than or equal to, we add one to it.
            # Equal to means we can never choose the correct example as a negative.
            # And greater than means our negative indices can cover the whole batch.
            negative_indices += (negative_indices >= real_labels.unsqueeze(-1)).long()

            # n x k-1 x e
            negative_examples = x2[negative_indices]
            # n x k-1:
            negative_logits = (x1.unsqueeze(1) * negative_examples).sum(dim=-1)
            # n x k
            logits = torch.cat([positive_logits, negative_logits], dim=-1)
            labels = torch.zeros([x1.shape[0]], device=x1.device, dtype=torch.long)

        return nn.functional.cross_entropy(logits, labels, reduction=self.reduction)
