"""Explicit Torch-native building blocks for the experimental path.

Unlike ``caesar.modules.basic.Linear``, these layers do not use
``nn.LazyLinear`` and never create parameters from ``forward``. Parameter
names intentionally match the legacy wrappers where practical.
"""

from __future__ import annotations

from typing import Callable, Optional, Union

import torch
import torch.nn as nn

from caesar.modules.basic import gelu_salad, init_linear, init_relu, init_zeros
from caesar.modules.basic import _resolve_init as _resolve_init


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        bias_init: float = 0.0,
        initializer: Union[str, Callable[[torch.Tensor], None]] = "linear",
        name: Optional[str] = "linear",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        del name
        self.bias_init = float(bias_init)
        self.init = _resolve_init(initializer)
        self.lin = nn.Linear(
            int(in_features),
            int(out_features),
            bias=bool(bias),
            device=device,
            dtype=dtype,
        )
        self._apply_init()

    @torch.no_grad()
    def _apply_init(self):
        self.init(self.lin.weight)
        if self.lin.bias is not None:
            nn.init.constant_(self.lin.bias, self.bias_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.lin.weight.device != x.device:
            raise RuntimeError(
                f"Linear module is on {self.lin.weight.device}, input is on {x.device}. "
                "Move the parent model with model.to(device) before forward."
            )
        return self.lin(x)


class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        size: int = 64,
        out_size: Optional[int] = None,
        depth: int = 2,
        activation: Callable = torch.relu,
        bias: bool = True,
        init: Union[str, Callable[[torch.Tensor], None]] = "relu",
        final_init: Union[str, Callable[[torch.Tensor], None]] = "linear",
        name: Optional[str] = "mlp",
    ):
        super().__init__()
        del name
        self.depth = int(depth)
        self.act = activation
        hidden = int(size)
        final = int(out_size if out_size is not None else size)
        dims = [int(in_features)] + [hidden] * max(0, self.depth - 1) + [final]
        self.layers = nn.ModuleList(
            [
                Linear(
                    dims[i],
                    dims[i + 1],
                    bias=bias,
                    initializer=init if i < self.depth - 1 else final_init,
                )
                for i in range(self.depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for i, layer in enumerate(self.layers):
            out = layer(out)
            if i < self.depth - 1:
                out = self.act(out)
        return out


__all__ = [
    "Linear",
    "MLP",
    "gelu_salad",
    "init_linear",
    "init_relu",
    "init_zeros",
]
