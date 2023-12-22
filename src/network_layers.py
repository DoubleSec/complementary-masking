from torch.nn import functional as F
from torch import nn
import torch


class FeatureEmbedder(nn.Module):
    """Class to handle feature embedding.

    Specific embedding logic is the job of the morphers. This just keeps track of everything.

    - morphers: a dictionary of initialized morphers. Should come from the dataset.
    - output_size: size of embedding outputs
    - gather: one of "stack", "cat", or "sum"
        - "stack": creates a new dimension for embeddings. Suitable for transformer-based architecture.
          Output is shaped: batch x features x embedding size.
        - "cat": concatenates along the embedding dimension, for MLP-based architecture.
        - "sum": sums the embeddings. Also for MLP architectures, but much smaller. I'd suggest larger
          embeddings if you're doing this."""

    def __init__(
        self,
        morphers: dict,
        output_size: int,
        gather: str = "stack",
    ):
        super().__init__()

        self.morphers = morphers
        self.output_size = output_size
        self.gather = gather

        self.embedding_layers = nn.ModuleDict(
            {
                feature: morpher.make_embedding(self.output_size)
                for feature, morpher in self.morphers.items()
            }
        )

        if gather == "stack":
            self.gather_step = lambda x: torch.stack(x, dim=1)
        elif gather == "cat":
            self.gather_step = lambda x: torch.cat(x, dim=-1)
        elif gather == "sum":
            self.gather_step = sum
        else:
            raise ValueError("gather must be 'stack', 'cat', or 'sum'")

    def forward(self, x):
        x = [
            embedder(x[feature]) for feature, embedder in self.embedding_layers.items()
        ]
        return self.gather_step(x)


class FeatureMasker(nn.Module):

    """Masks inputs:

    - morphers: just used to identify the number of features
    - input_size: int, the size of embeddings
    - p: masking probability
    - masking_strategy: what we're masking with. One of "learned" or "zero" for now, more later
    - return_complement: if true, return each input twice, with complementary masks
    """

    def __init__(
        self,
        morphers: dict,
        input_size: int,
        p: float = 0.5,
        masking_strategy: str = "zero",
        return_complement: bool = False,
    ):
        super().__init__()

        self.n_features = len(morphers)
        self.input_size = input_size
        # I have ZERO idea of whether this matters or not.
        self.register_buffer("p", torch.tensor(p))
        self.masking_strategy = masking_strategy
        self.return_complement = return_complement

        if masking_strategy == "zero":
            self.mask_vals = nn.Parameter(
                torch.zeros([1, self.n_features, self.input_size], requires_grad=False)
            )
        elif masking_strategy == "learned":
            self.mask_vals = nn.Parameter(
                torch.randn([1, self.n_features, self.input_size], requires_grad=True)
            )

    def forward(self, x):
        # x will be n x s x e

        # Choose the features to mask
        mask = torch.rand([1, self.n_features, 1], device=x.device) < self.p
        # Expand to the data size
        mask = mask.expand([x.shape[0], -1, x.shape[2]])

        if not self.return_complement:
            # Apply the mask
            x = torch.where(mask, self.mask_vals, x)
            return x
        else:
            x1 = torch.where(mask, self.mask_vals, x)
            x2 = torch.where(mask, x, self.mask_vals)
            return x1, x2


class LearnedPositionEncoding(nn.Module):
    def __init__(self, max_length: int, d_model: int):
        super().__init__()
        pe = torch.normal(torch.zeros([1, max_length, d_model]), 0.02)
        self.position_encodings = nn.Parameter(pe)

    def forward(self, x):
        x_len = x.shape[1]
        return x + self.position_encodings[:, :x_len]


class ProjectionHead(nn.Module):
    """Simple projection head for network.

    Used to get an output embedding from the network.

    - input_size: int, what's the input size
    - output_size: int, what's the output size
    - n_layers: int, how many fully-connected layers do we want

    Note each layer is preceded by layer norm + activation (ReLU for now)"""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        n_layers: int = 1,
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.layers = nn.Sequential()

        self.layers.append(nn.LayerNorm(self.input_size))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(self.input_size, self.output_size))

        for _ in range(n_layers - 1):
            self.layers.append(nn.LayerNorm(self.output_size))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(self.output_size, self.output_size))

    def forward(self, x):
        # x should be n x e
        return self.layers(x)
