from abc import ABC, abstractmethod
from itertools import chain
import math

import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics.classification import BinaryAUROC

from ..utils import alignment, uniformity
from .loss import BarlowTwinsLoss
from .network_layers import (
    FeatureMasker,
    ProjectionHead,
    ComplementaryMasker,
)
from .camalambakicken import RMSNorm
from .core import CoreNet


class Extender(nn.Module, ABC):
    """Extenders specify a task downstream of the pre-trained model."""

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
        # TKTK selectable loss function.
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
        batch_loss = {self.name: self.bt_loss(x[0], x[1])}
        metrics = {
            "alignment": alignment(x[0], x[1]),
            "uniformity": (uniformity(x[0]) + uniformity(x[1])) / 2,
        }
        return batch_loss, metrics

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

    @property
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


class RPClassifier(Extender):
    """ "Representation-preserving" classifier.

    Who knows if this works."""

    def __init__(
        self,
        name: str,
        target_column: str,
        embedding_size: int,
        predictor_dropout: float,
        n_layers: int,
        bt_loss_params: dict,
        alpha: float,
        lr: float,
        weight_decay: float = 0.01,
    ):
        super().__init__()
        self.extender_name = name
        self.target_column = target_column
        self.metric = BinaryAUROC()
        self.alpha = alpha
        self.lr = lr
        self.weight_decay = weight_decay

        # Prediction components
        self.dropout = nn.Dropout(predictor_dropout)
        self.predictor_head = ProjectionHead(
            input_size=embedding_size,
            output_size=1,
            n_layers=n_layers,
            norm_type=RMSNorm,
            activation_type=nn.GELU,
        )
        self.prediction_loss = nn.BCEWithLogitsLoss(reduction="none")

        # Representation Preservation Components
        self.masker = ComplementaryMasker()
        self.bt_projector = ProjectionHead(
            input_size=embedding_size,
            output_size=embedding_size,
            n_layers=n_layers,
            norm_type=RMSNorm,
            activation_type=nn.GELU,
        )
        self.bt_loss = BarlowTwinsLoss(**bt_loss_params)

    @property
    def name(self):
        return self.extender_name

    def forward(self, core_net: CoreNet, x):
        x = core_net(x)
        x = self.predictor_head(x)

    def training_forward(self, core_net: CoreNet, x):

        # Prediction
        x = core_net(x)
        y_hat = self.predictor_head(self.dropout(x))

        # Representation preservation
        x1, x2 = self.masker(x)
        x1 = self.bt_projector(x1)
        x2 = self.bt_projector(x2)
        return y_hat, x1, x2

    def loss(self, x, y):

        y_hat, x1, x2 = y
        pred_target = x[self.target_column]

        pred_loss = self.prediction_loss(y_hat.squeeze(), pred_target.float()).mean()
        pred_metric = self.metric(y_hat.squeeze(), pred_target)
        barlow_loss = self.bt_loss(x1, x2) * self.alpha

        return {"prediction": pred_loss, "barlow": barlow_loss}, {"AUROC": pred_metric}

    def configure_optimizers(self, core_net):
        return torch.optim.Adam(
            params=chain(self.parameters(), core_net.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )


class LoRALinear(nn.Module):
    """Replacement for nn.Linear for LoRA fine-tuning."""

    def __init__(self, input_dim: int, output_dim: int, k: int):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.a = nn.Parameter(torch.randn([input_dim, k]) * 0.02)
        self.b = nn.Parameter(torch.zeros([k, output_dim]))

    def forward(self, x):

        # x should be (whatever) x input_dim
        m = self.a @ self.b
        return x @ m


class LoRAFineTuner(Extender):
    """Implements LoRA fine-tuning."""

    def __init__(
        self,
        name: str,
        target_column: str,
        # Arguments to match the existing transformer
        input_dim: int,
        n_kv_heads: int,
        n_q_heads: int,
        n_transformer_layers: int,
        # LoRA parameters
        k: int,
        scale: float,
        n_predictor_layers: int,
        # Optimizer
        lr: float,
        weight_decay: float = 0.01,
    ):
        super().__init__()
        self.extender_name = name
        self.target_column = target_column
        self.metric = BinaryAUROC()
        self.lr = lr
        self.weight_decay = weight_decay

        self.n_transformer_layers = n_transformer_layers
        self.k = k
        self.scale = scale

        head_dim = input_dim // n_q_heads

        self.layers = nn.ModuleList()
        for _ in range(n_transformer_layers):

            self.layers.append(
                nn.ModuleDict(
                    {
                        "query": LoRALinear(input_dim, input_dim, k),
                        "key": LoRALinear(input_dim, n_kv_heads * head_dim, k),
                        "value": LoRALinear(input_dim, n_kv_heads * head_dim, k),
                        "output": LoRALinear(input_dim, input_dim, k),
                    }
                )
            )

        self.predictor_head = ProjectionHead(
            input_size=input_dim,
            output_size=1,
            n_layers=n_predictor_layers,
            norm_type=RMSNorm,
            activation_type=nn.GELU,
        )
        self.prediction_loss = nn.BCEWithLogitsLoss(reduction="none")

    @property
    def name(self):
        return self.extender_name

    def forward(self, core_net: CoreNet, x):
        x = core_net.embedding_layer(x)
        x = core_net.feature_norm(x)
        x = core_net.positional_encoding(x)
        x = torch.cat([x, core_net.cls.expand([x.shape[0], -1, -1])], dim=1)
        # Transformer layers with LoRA
        for i, tr_layer in enumerate(core_net.transformer.transformer_layers):
            a = x + self._lora_attention(tr_layer, i, tr_layer.attn_norm(x))
            h = tr_layer.swiglu(tr_layer.linear_norm(a))
            h = tr_layer.linear(h)
            x = a + h
        x = x[:, -1, :]
        return self.predictor_head(x)

    def loss(self, x, y):

        pred_target = x[self.target_column]

        pred_loss = self.prediction_loss(y.squeeze(), pred_target.float()).mean()
        pred_metric = self.metric(y.squeeze(), pred_target)

        return {"prediction": pred_loss}, {"AUROC": pred_metric}

    def configure_optimizers(self, core_net: CoreNet):
        # Doesn't include the core_net parameters, which is the whole point.
        return torch.optim.Adam(
            params=self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    def _lora_attention(self, tr_layer, i, x):
        lora_layer = self.layers[i]

        batch_size, seq_len, _ = x.shape
        attn = tr_layer.gq_attn

        # From the transformer layer
        normed_x = tr_layer.attn_norm(x)

        # Get the LoRA modifications
        lxq = lora_layer["query"](normed_x) * self.scale
        lxk = lora_layer["key"](normed_x) * self.scale
        lxv = lora_layer["value"](normed_x) * self.scale

        # Get the original outputs
        xq = attn.wq(normed_x) + lxq
        xk = attn.wk(normed_x) + lxk
        xv = attn.wv(normed_x) + lxv

        xq = xq.view(batch_size, seq_len, attn.n_q_heads, attn.head_dim)
        xk = xk.view(batch_size, seq_len, attn.n_kv_heads, attn.head_dim)
        xv = xv.view(batch_size, seq_len, attn.n_kv_heads, attn.head_dim)

        # Transpose changes (n x s x h x e) to (n x h x s x e)
        xq = xq.transpose(1, 2)
        # Repeats k and v to match number of q heads
        exp_k = torch.repeat_interleave(xk, attn.n_rep, dim=2).transpose(1, 2)
        exp_v = torch.repeat_interleave(xv, attn.n_rep, dim=2).transpose(1, 2)

        # This is Scaled Dot-Product Attention
        attn_scores = torch.matmul(xq, exp_k.transpose(2, 3)) / math.sqrt(attn.head_dim)

        # TKTK Just doesn't support masking at all, yet

        # n x h x s x s
        attn_scores = F.softmax(attn_scores, dim=-1)
        # n x h x s x e
        output = torch.matmul(attn_scores, exp_v)
        # n x s x (h x e)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        # n x s x input_size
        lo = lora_layer["output"](output) * self.scale
        return attn.wo(output) + lo
