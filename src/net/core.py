import torch
from torch import nn


from .network_layers import (
    FeatureEmbedder,
    LearnedPositionEncoding,
)
from .camalambakicken import Transformer, RMSNorm


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
        x = torch.cat([x, self.cls.expand([x.shape[0], -1, -1])], dim=1)
        x = self.transformer(x)
        x = x[:, -1, :]
        return x
