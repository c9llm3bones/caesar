"""PyTorch port of salad.modules.basic 
"""

from __future__ import annotations
from typing import Optional, Callable, Union, Any
import math
import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import UninitializedParameter

def gelu_salad(x):
    return F.gelu(x, approximate="tanh")

@torch.no_grad()
def _variance_scaling_(w: torch.Tensor, scale=1.0, mode="fan_in", dist="truncated_normal"):
    fan_in, fan_out = w.shape[1], w.shape[0]  
    denom = fan_in if mode == "fan_in" else fan_out if mode == "fan_out" else (fan_in + fan_out) / 2
    var = float(scale) / max(1.0, float(denom))
    std = math.sqrt(var)

    if dist == "uniform":
        bound = math.sqrt(3.0) * std
        w.uniform_(-bound, bound)
    elif dist == "normal":
        w.normal_(0.0, std)
    else:  
        nn.init.trunc_normal_(w, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)


def init_relu():   return lambda w: _variance_scaling_(w, scale=2.0, mode="fan_in", dist="truncated_normal")
def init_linear(): return lambda w: _variance_scaling_(w, scale=1.0, mode="fan_in", dist="truncated_normal")
def init_glorot(): return lambda w: _variance_scaling_(w, scale=1.0, mode="fan_avg", dist="uniform")
def init_zeros():  return lambda w: nn.init.constant_(w, 0.0)

def init_small(scale=1e-2):
    @torch.no_grad()
    def _init(w): w.uniform_(-scale, scale)
    return _init

def _resolve_init(initializer: Union[str, Callable[[torch.Tensor], None]]):
    if callable(initializer): return initializer
    return {"relu": init_relu(), "linear": init_linear(), "glorot": init_glorot(), "zeros": init_zeros()}.get(initializer, init_linear())


class Linear(nn.Module):
    def __init__(self, size: int = 128, bias: bool = True, bias_init: float = 0.0,
                 initializer: Union[str, Callable[[torch.Tensor], None]] = "linear",
                 name: Optional[str] = "linear"):
        super().__init__()
        self.bias_init = float(bias_init)
        self.init = _resolve_init(initializer)
        self.lin = nn.LazyLinear(int(size), bias=bool(bias))
        self._inited = False

    @torch.no_grad()
    def _apply_init(self):
        self.init(self.lin.weight)
        if self.lin.bias is not None:
            nn.init.constant_(self.lin.bias, self.bias_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.lin.weight.device != x.device:
            self.lin = self.lin.to(x.device)

        y = self.lin(x)

        if not self._inited:
            with torch.no_grad():
                self._apply_init()
            y = self.lin(x)
            self._inited = True

        return y

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)
        self._inited = True


class MLP(nn.Module):
    def __init__(self, size=64, out_size=None, depth=2,
                 activation: Callable = F.relu, bias=True,
                 init="relu", final_init="linear",
                 name: Optional[str] = "mlp"):
        super().__init__()
        self.depth = int(depth)
        self.act = activation
        self.layers = nn.ModuleList([
            Linear(size, bias=bias, initializer=init) if i < depth - 1
            else Linear(out_size or size, bias=bias, initializer=final_init)
            for i in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for i, layer in enumerate(self.layers):
            out = layer(out)
            if i < self.depth - 1:
                out = self.act(out)
        return out


class GatedMLP(nn.Module):
    def __init__(self, size=64, out_size=None,
                 activation: Callable = gelu_salad,
                 init="relu", final_init="zeros",
                 name: Optional[str] = "gated_mlp"):
        super().__init__()
        self.size = int(size)
        self.out_size = out_size
        self.act = activation
        self.gate_lin = Linear(size, bias=False, initializer=init)
        self.hid_lin  = Linear(size, bias=False, initializer=init)
        self.out_lin: Optional[Linear] = Linear(out_size, bias=False, initializer=final_init) if out_size is not None else None
        self.final_init = final_init

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.act(self.gate_lin(x))
        hidden = gate * self.hid_lin(x)
        if self.out_lin is None:
            self.out_lin = Linear(int(x.shape[-1]), bias=False, initializer=self.final_init).to(x.device)
        return self.out_lin(hidden)


class SmallLinear(nn.Module):
    def __init__(self, size: int, scale: float = 1e-4, bias: bool = True, **kwargs: Any):
        super().__init__()
        self.lin = Linear(size, bias=bias, initializer=init_small(scale), **kwargs)
        self.ln = nn.LayerNorm(int(size), elementwise_affine=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(self.lin(x))


def small_linear(*args, scale=1e-4, **kwargs) -> nn.Module:
    size = int(args[0]) if args else int(kwargs.pop("size"))
    return SmallLinear(size=size, scale=scale, **kwargs)


# block_stack 

class _Stack(nn.Module):
    def __init__(self, n: int, factory: Callable, with_state: bool):
        super().__init__()
        self.with_state = with_state
        takes_i = (len(inspect.signature(factory).parameters) >= 1)
        self.blocks = nn.ModuleList([factory(i) if takes_i else factory() for i in range(int(n))])

    def forward(self, x, state=None):
        if self.with_state:
            out, st = x, state
            for b in self.blocks:
                out, st = b(out, st)
            return out, st
        out = x
        for b in self.blocks:
            out = b(out)
        return out


def block_stack(depth: int, block_size: int = 1, with_state: bool = False):

    n = (int(depth) // int(block_size)) * int(block_size)

    def inner(factory: Callable):
        return _Stack(n=n, factory=factory, with_state=with_state)

    return inner
