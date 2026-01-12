"""PyTorch-compatible replacements for salad.modules.basic.

This implements Linear, MLP, GatedMLP and a few initializer helpers.
The goal is API parity sufficient for incremental porting of the autoencoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable


class Linear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 initializer: Optional[str] = "linear",
                 bias_init: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.initializer = initializer
        self._init_weights(bias_init)

    def _init_weights(self, bias_init: float = 0.0):
        w = self.linear.weight
        b = self.linear.bias
        if callable(self.initializer):
            self.initializer(w)
        else:
            key = (self.initializer or "linear").lower()
            if key == "relu":
                nn.init.kaiming_normal_(w, nonlinearity="relu")
            elif key == "glorot" or key == "xavier":
                nn.init.xavier_uniform_(w)
            elif key == "zeros":
                nn.init.zeros_(w)
            else:
                nn.init.xavier_uniform_(w, gain=1.0)
        if b is not None:
            nn.init.constant_(b, bias_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MLP(nn.Module):
    def __init__(self, 
                 hidden: int,
                 out_size: Optional[int] = None,
                 depth: int = 2,
                 activation: Callable = F.relu,
                 bias: bool = True,
                 init: str = "relu",
                 final_init: str = "linear"):
        super().__init__()
        self.depth = depth
        self.activation = activation
        self.hidden = hidden
        self.out_size = out_size
        self.bias = bias
        self.init = init
        self.final_init = final_init
        self.layers: nn.ModuleList = nn.ModuleList()
        self._built = False

    def _build(self, in_dim: int):
        self.layers = nn.ModuleList()
        for i in range(self.depth):
            if i == 0:
                in_features = in_dim
                out_features = self.hidden if i < self.depth - 1 else (self.out_size or self.hidden)
                initializer = self.init if i < self.depth - 1 else self.final_init
                self.layers.append(Linear(in_features, out_features, bias=self.bias, initializer=initializer))
            else:
                if i < self.depth - 1:
                    self.layers.append(Linear(self.hidden, self.hidden, bias=self.bias, initializer=self.init))
                else:
                    out = self.out_size if self.out_size is not None else self.hidden
                    self.layers.append(Linear(self.hidden, out, bias=self.bias, initializer=self.final_init))
        self._built = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._built:
            self._build(x.shape[-1])
        out = x
        for i, layer in enumerate(self.layers):
            out = layer(out)
            if i < (self.depth - 1):
                out = self.activation(out)
        return out


class GatedMLP(nn.Module):
    def __init__(self,
                 hidden: int,
                 out_size: Optional[int] = None,
                 activation: Callable = F.gelu,
                 init: str = "relu",
                 final_init: str = "zeros"):
        super().__init__()
        self.hidden = hidden
        self.out_size = out_size or hidden
        self.activation = activation
        self.proj = None
        self.final = None
        self._built = False
        self.init = init
        self.final_init = final_init

    def _build(self, in_dim: int):
        self.proj = Linear(in_dim, self.hidden * 2, bias=True, initializer=self.init)
        self.final = Linear(self.hidden, self.out_size, bias=True, initializer=self.final_init)
        self._built = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._built:
            self._build(x.shape[-1])
        h = self.proj(x)
        a, b = h.chunk(2, dim=-1)
        gated = self.activation(a) * b
        return self.final(gated)


# helper initializers (for compatibility)

def init_relu():
    return "relu"


def init_linear():
    return "linear"


def init_zeros():
    return "zeros"


def init_glorot():
    return "glorot"


def small_linear(*args, scale=1e-4, **kwargs):
    def inner(x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            nn.init.normal_(x, mean=0.0, std=scale)
    return inner


def block_stack(depth: int, block_size: int = 1, with_state: bool = False):
    """Return a function that applies `function` repeated `depth` times.

    This is a simplified replacement for the Haiku block_stack + layer_stack.
    It does not implement gradient checkpointing semantics; for that we can
    add torch.utils.checkpoint later when necessary.
    """
    def inner(function: Callable):
        def apply(x, *args, **kwargs):
            out = x
            state = None
            for _ in range(depth):
                if with_state:
                    out, state = function(out, state, *args, **kwargs)
                else:
                    out = function(out, *args, **kwargs)
            return out
        return apply
    return inner
