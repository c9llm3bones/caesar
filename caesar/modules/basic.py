"""PyTorch-compatible replacements for salad.modules.basic.

This implements Linear, MLP, GatedMLP and a few initializer helpers.
The goal is API parity sufficient for incremental porting of the autoencoder.
"""

from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        if bias:
            nn.init.constant_(self.linear.bias, bias_init)
        self._init_weights()

    def _init_weights(self):
        if self.initializer == "relu":
            gain = torch.nn.init.calculate_gain('relu')
            nn.init.kaiming_uniform_(self.linear.weight, a=0, nonlinearity='relu')
        elif self.initializer == "glorot":
            nn.init.xavier_uniform_(self.linear.weight)
        elif self.initializer == "zeros":
            nn.init.constant_(self.linear.weight, 0.0)
        else:  # linear or default
            nn.init.kaiming_uniform_(self.linear.weight, a=0, nonlinearity='linear')

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
        self.layers = nn.ModuleList()
        for i in range(depth):
            if i < depth - 1:
                self.layers.append(Linear(hidden, hidden, bias=bias, initializer=init))
            else:
                out = out_size if out_size is not None else hidden
                self.layers.append(Linear(hidden, out, bias=bias, initializer=final_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        self.activation = activation
        self.hidden = hidden
        self.out_size = out_size
        self.gate = Linear(hidden, hidden, bias=False, initializer=init)
        self.hidden_lin = Linear(hidden, hidden, bias=False, initializer=init)
        self.out_lin = Linear(hidden, out_size or hidden, bias=False, initializer=final_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.activation(self.gate(x))
        hidden = gate * self.hidden_lin(x)
        return self.out_lin(hidden)


# helper initializers (for compatibility)

def init_relu():
    # wrapper returning a callable consistent with previous API 
    return "relu"


def init_linear():
    return "linear"


def init_zeros():
    return "zeros"


def init_glorot():
    return "glorot"


def small_linear(*args, scale=1e-4, **kwargs):
    def inner(x: torch.Tensor) -> torch.Tensor:
        lin = Linear(*args, initializer=init_glorot(), **kwargs)
        y = lin(x)
        return nn.LayerNorm(y.shape[-1])(y)
    return inner


def block_stack(depth: int, block_size: int = 1, with_state: bool = False):
    """Return a function that applies `function` repeated `depth` times.

    This is a simplified replacement for the Haiku block_stack + layer_stack.
    It does not implement gradient checkpointing semantics; for that we can
    add torch.utils.checkpoint later when necessary.
    """
    def inner(function: Callable):
        def apply_fn(x):
            out = x
            for _ in range(depth):
                out = function(out)
            return out
        return apply_fn
    return inner
