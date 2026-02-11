from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Dict, List, Union, Optional, Tuple

from torch.nn.parameter import UninitializedParameter
import numpy as np
import torch


def load_salad_flat_from_pickle(jax_path: str) -> Dict[str, np.ndarray]:
    """
    salad .jax: trusted pickle with structure:
      { "module/path": { "w": ndarray, "b": ndarray, "scale": ndarray, ... }, ... }
    We flatten it into:
      { "module/path/w": ndarray, ... }
    """
    import pickle
    with open(jax_path, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, (tuple, list)) and len(obj) == 2:
        obj = obj[0]

    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict (or (params,state)) at top-level, got {type(obj)}")
    flat: Dict[str, np.ndarray] = {}
    for mod_path, pdict in obj.items():
        if not isinstance(mod_path, str) or not isinstance(pdict, dict):
            continue
        for pname, pval in pdict.items():
            if not isinstance(pname, str):
                continue
            arr = np.asarray(pval)
            if isinstance(arr, np.ndarray) and arr.dtype != object:
                flat[f"{mod_path}/{pname}"] = arr
    return flat


# Pattern mirrors OpenFold: ParamType/Param/process_translation_dict/stacked/assign.
class ParamType(Enum):
    # Dense (in,out) -> torch Linear.weight (out,in)
    LinearWeight = "linear_weight"
    Other = "other"

    @property
    def transformation(self):
        if self is ParamType.LinearWeight:
            return lambda w: w.unsqueeze(-1) if len(w.shape) == 1 else w.transpose(-1, -2)
        return lambda w: w



@dataclass
class Param:
    param: Union[torch.Tensor, List[torch.Tensor], None]
    param_type: ParamType = ParamType.Other
    stacked: bool = False
    swap: bool = False
    check_zero: bool = False
    atol: float = 0.0


def process_translation_dict(d: dict, prefix: str = "", top_layer: bool = True) -> Dict[str, Param]:
    """
    Flatten a nested translation dict into:
      "a/b/c" -> Param(...)
    """
    flat: Dict[str, Param] = {}
    for k, v in d.items():
        if top_layer:
            key = k
        else:
            key = f"{prefix}/{k}" if prefix else k

        if isinstance(v, dict):
            flat.update(process_translation_dict(v, prefix=key, top_layer=False))
        else:
            if not isinstance(v, Param):
                raise TypeError(f"Leaf must be Param, got {type(v)} at key={key}")
            flat[key] = v
    return flat


def stacked(param_dict_list: List[dict], out: Optional[dict] = None) -> dict:
    """
    Stack same-structured dicts into Param(stacked=True).
    Structure must match across all dicts.
    """
    if len(param_dict_list) == 0:
        raise ValueError("stacked() requires at least one element")
    if out is None:
        out = {}

    template = param_dict_list[0]
    for k in template.keys():
        v = [d[k] for d in param_dict_list]
        if isinstance(v[0], dict):
            out[k] = {}
            stacked(v, out=out[k])
        elif isinstance(v[0], Param):
            refs = [p.param for p in v]
            out[k] = Param(
                param=[r for r in refs],  
                param_type=v[0].param_type,
                stacked=True,
                swap=v[0].swap,
                check_zero=v[0].check_zero,
                atol=v[0].atol,
            )
        else:
            raise TypeError(f"Unsupported value type in stacked(): {type(v[0])}")
    return out


def _maxabs_np(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.max(np.abs(x)))


def _squeeze_leading_ones_nonstack(x: np.ndarray) -> np.ndarray:
    while x.ndim >= 1 and x.shape[0] == 1:
        x = x[0]
    return x

def _build_state_dict_tensor_index(model) -> tuple[dict[int, str], dict[tuple[int, tuple, torch.dtype], list[str]], list[str]]:
    """
      - by_id: id(tensor) -> key
      - by_ptr: (data_ptr, shape, dtype) -> [keys]  (на случай если id не совпал)
    """
    sd = model.state_dict()
    by_id: dict[int, str] = {}
    by_ptr: dict[tuple[int, tuple, torch.dtype], list[str]] = {}
    keys = list(sd.keys())

    for k, t in sd.items():
        by_id[id(t)] = k
        try:
            ptr_key = (int(t.data_ptr()), tuple(t.shape), t.dtype)
            by_ptr.setdefault(ptr_key, []).append(k)
        except Exception:
            pass

    return by_id, by_ptr, keys


def _torch_keys_touched_by_translation(model, flat_map: Dict[str, Param]) -> set[str]:
    by_id, by_ptr, _ = _build_state_dict_tensor_index(model)
    touched: set[str] = set()

    def resolve_name(t: torch.Tensor) -> Optional[str]:
        name = by_id.get(id(t))
        if name is not None:
            return name
        try:
            ptr_key = (int(t.data_ptr()), tuple(t.shape), t.dtype)
            names = by_ptr.get(ptr_key)
            if names:
                return names[0]
        except Exception:
            pass
        return None

    for _, p in flat_map.items():
        ref = p.param
        if ref is None:
            continue
        refs = ref if isinstance(ref, list) else [ref]
        for r in refs:
            if r is None:
                continue
            name = resolve_name(r)
            if name is not None:
                touched.add(name)

    return touched

def _materialize_if_uninitialized(p: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
    """
    Materialize Lazy/LazyLinear parameters before accessing shape/copying.
    Works for both UninitializedParameter itself and parameters that may wrap it.
    """
    if isinstance(p, UninitializedParameter):
        p.materialize(shape)
        return p
    return p

def assign(translation_dict: Dict[str, Param], orig_weights: Dict[str, np.ndarray], *, verbose: bool = False):
    """
      - stacked handling via unbind(0)
      - "Param(None, check_zero=True)" to require a key exists but is all-zeros (e.g. bias absent in torch)
    """
    for k, param in translation_dict.items():
        if k not in orig_weights:
            raise KeyError(f"Missing salad key: {k}")

        w_np = orig_weights[k]
       
        if param.param is None: 
            if param.check_zero:
                vmax = _maxabs_np(w_np)
                if vmax > param.atol:
                    raise ValueError(f"{k}: expected ~0 (atol={param.atol}), got maxabs={vmax}")
            continue

        with torch.no_grad():
            ref = param.param
            param_type = param.param_type

            if param.stacked:
                if not isinstance(ref, list):
                    raise TypeError(f"{k}: stacked Param expects list[Tensor] refs")
                n = len(ref)

                if n == 1 and (w_np.ndim >= 1) and (w_np.shape[0] != 1):
                    w_np = np.expand_dims(w_np, axis=0)

                weights = torch.as_tensor(np.array(w_np, copy=True))
                if weights.ndim < 1:
                    raise ValueError(f"{k}: stacked weights must have ndim>=1, got {tuple(weights.shape)}")

                weights_list = list(torch.unbind(weights, 0))
                if len(weights_list) != n:
                    raise ValueError(f"{k}: stacked length mismatch: salad={len(weights_list)} torch={n}")

                weights_list = list(map(param_type.transformation, weights_list))

                for p, w in zip(ref, weights_list):
                    if p is None:
                        continue
                    _materialize_if_uninitialized(p, tuple(w.shape))
                    if param.swap:
                        index = p.shape[0] // 2
                        p[:index].copy_(w[index:])
                        p[index:].copy_(w[:index])
                    else:
                        if tuple(p.shape) != tuple(w.shape):
                            raise ValueError(f"{k}: shape mismatch torch={tuple(p.shape)} vs salad={tuple(w.shape)}")
                        p.copy_(w.to(dtype=p.dtype, device=p.device))

            else:
                if isinstance(ref, list):
                    raise TypeError(f"{k}: non-stacked Param expects Tensor ref, got list")

                w_np2 = _squeeze_leading_ones_nonstack(w_np)
                weights = torch.as_tensor(np.array(w_np2, copy=True))
                weights = param_type.transformation(weights)
                _materialize_if_uninitialized(ref, tuple(weights.shape))
                if param.swap:
                    index = ref.shape[0] // 2
                    ref[:index].copy_(weights[index:])
                    ref[index:].copy_(weights[:index])
                else:
                    if tuple(ref.shape) != tuple(weights.shape):
                        raise ValueError(f"{k}: shape mismatch torch={tuple(ref.shape)} vs salad={tuple(weights.shape)}")
                    ref.copy_(weights.to(dtype=ref.dtype, device=ref.device))

        if verbose:
            print("assigned", k)


### Templates 
LinearW = lambda t: Param(t, param_type=ParamType.LinearWeight)
Other = lambda t: Param(t, param_type=ParamType.Other)


def LinearParams(lin: torch.nn.Module, *, has_bias: bool) -> dict:
    d = {"w": LinearW(lin.weight)}
    if has_bias:
        if lin.bias is None:
            raise ValueError("Expected bias, found None")
        d["b"] = Other(lin.bias)
    else:
        d["b"] = Param(None, check_zero=True, atol=0.0)
    return d


def LayerNormParams(ln: torch.nn.LayerNorm) -> dict:
    return {
        "scale": Other(ln.weight),
        "offset": Other(ln.bias),
    }


def BiasModuleParams(bias_lin: torch.nn.Module, *, has_bias: bool) -> dict:
    # jax: bias/{w,b} where w is (C, heads) => torch Linear weight (heads, C)
    return LinearParams(bias_lin, has_bias=has_bias)


def MLP2Params(mlp, *, has_bias: bool) -> dict:
    # expects mlp.layers[0].lin and mlp.layers[1].lin
    return {
        "linear": LinearParams(mlp.layers[0].lin, has_bias=has_bias),
        "linear_1": LinearParams(mlp.layers[1].lin, has_bias=has_bias),
    }


### Shared submodules

def _AAPointAttnParams(attn) -> dict:
    # torch:
    #   bias.lin.weight (no bias)
    #   gamma
    #   _ln_local, ln_q, ln_k
    #   qkv.lin.weight (no bias)
    #   qkv_global.proj.lin.weight (no bias)
    #   _project_out.lin.{weight,bias}
    return {
        "bias": BiasModuleParams(attn.bias.lin, has_bias=False),
        "gamma": Other(attn.gamma),
        "layer_norm": LayerNormParams(attn._ln_local),
        "layer_norm_1": LayerNormParams(attn.ln_q),
        "layer_norm_2": LayerNormParams(attn.ln_k),
        "qkv": LinearParams(attn.qkv.lin, has_bias=False),
        "qkv_global": {
            "linear": LinearParams(attn.qkv_global.proj.lin, has_bias=False),
        },
        "project_out": LinearParams(attn._project_out.lin, has_bias=True),
    }


def _AAUpdateParams(upd) -> dict:
    return {
        "linear": LinearParams(upd.update_proj.lin, has_bias=False),
        "linear_1": LinearParams(upd.gate_proj.lin, has_bias=False),
        "linear_2": LinearParams(upd.out_proj.lin, has_bias=True),
        "mlp": {
            "linear": LinearParams(upd.localpos_mlp.layers[0].lin, has_bias=True),
            "linear_1": LinearParams(upd.localpos_mlp.layers[1].lin, has_bias=True),
        },
    }


# decoder (structure_autoencoder/diffusion/*)

def _DecoderPrepareFeaturesParams(dec) -> dict:
    # salad: structure_autoencoder/diffusion/~prepare_features/*
    return {
        "layer_norm": LayerNormParams(dec.prev_local_ln),
        "mlp": {
            "linear": LinearParams(dec.local_mlp.layers[0].lin, has_bias=False),
            "linear_1": LinearParams(dec.local_mlp.layers[1].lin, has_bias=False),
        },
        "layer_norm_1": LayerNormParams(dec.local_ln),
    }


def _AnglePosMLPParams(dec) -> dict:
    # salad: structure_autoencoder/diffusion/mlp/*
    # torch: decoder.angle_pos.mlp layers have no bias
    return {
        "linear": LinearParams(dec.angle_pos.mlp.layers[0].lin, has_bias=False),
        "linear_1": LinearParams(dec.angle_pos.mlp.layers[1].lin, has_bias=False),
    }


def _AADecoderBlockParams(block) -> dict:
    # salad: diffusion/aa_decoder/aa_decoder_stack/__layer_stack_no_state/aa_decoder_block/*
    return {
        "layer_norm": LayerNormParams(block.pair_features.ln),
        "layer_norm_1": LayerNormParams(block.ln_aa),
        "layer_norm_2": LayerNormParams(block.ln_attn_in),
        "layer_norm_3": LayerNormParams(block.ln_update_in),

        # projections (torch has no bias here)
        "linear": LinearParams(block.pair_features.p_relpos.lin, has_bias=False),
        "linear_1": LinearParams(block.pair_features.p_dist.lin, has_bias=False),
        "linear_2": LinearParams(block.pair_features.p_dir.lin, has_bias=False),
        "linear_3": LinearParams(block.pair_features.p_rot.lin, has_bias=False),
        "linear_4": LinearParams(block.pair_features.p_vec.lin, has_bias=False),
        "linear_5": LinearParams(block.aa_linear.lin, has_bias=False),

        # pair_mlp biasful
        "mlp": MLP2Params(block.pair_mlp, has_bias=True),

        "sparse_structure_attn": {
            "ada_point_attention": _AAPointAttnParams(block.attn.attn),
        },

        "light_global_update": _AAUpdateParams(block.update),
    }


def _AADecoderParams(dec) -> dict:
    blocks = list(dec.aa_decoder.stack.blocks)
    aa_stack = stacked([_AADecoderBlockParams(b) for b in blocks])

    return {
        "layer_norm": LayerNormParams(dec.aa_decoder.norm),
        # salad has bias for this linear but it is zero; torch has no bias
        "linear": LinearParams(dec.aa_decoder.proj, has_bias=False),
        "aa_decoder_stack": {
            "layer_norm": LayerNormParams(dec.aa_decoder.stack.ln),
            "__layer_stack_no_state": {
                "aa_decoder_block": aa_stack,
            },
        },
    }


def _DistogramParams(dist) -> dict:
    return {
        "inner_weight": Other(dist.inner_weight),
        "linear": LinearParams(dist.code_proj.lin, has_bias=False),
        "linear_1": LinearParams(dist.gate_proj.lin, has_bias=False),
    }


def _DecoderUpdateParams(upd) -> dict:
    return {
        "linear": LinearParams(upd.local_update.lin, has_bias=False),
        "linear_1": LinearParams(upd.local_gate.lin, has_bias=False),
        "linear_2": LinearParams(upd.chain_gate.lin, has_bias=False),
        "linear_3": LinearParams(upd.batch_gate.lin, has_bias=False),
        "linear_4": LinearParams(upd.out.lin, has_bias=True),
        "mlp": {
            "linear": LinearParams(upd.pos_mlp.layers[0].lin, has_bias=True),
            "linear_1": LinearParams(upd.pos_mlp.layers[1].lin, has_bias=True),
        },
    }


def _DecoderBlockParams(block) -> dict:
    out = {
        "layer_norm":   LayerNormParams(block.pair_features_main.ln),
        "layer_norm_1": LayerNormParams(block.ln_attn_main),
        "layer_norm_4": LayerNormParams(block.ln_update),
        "layer_norm_5": LayerNormParams(block.ln_pos),

        "linear":   LinearParams(block.pair_features_main.p_relpos.lin, has_bias=False),
        "linear_2": LinearParams(block.pair_features_main.p_dist.lin,  has_bias=False),
        "linear_3": LinearParams(block.pair_features_main.p_dir.lin,   has_bias=False),
        "linear_4": LinearParams(block.pair_features_main.p_rot.lin,   has_bias=False),
        "linear_5": LinearParams(block.pair_features_main.p_vec.lin,   has_bias=False),

        "linear_12": LinearParams(block.pos_update.proj.lin, has_bias=False),

        "mlp": MLP2Params(block.pair_features_main.mlp, has_bias=True),

        "sparse_structure_attn": {
            "ada_point_attention": _AAPointAttnParams(block.attn_main.attn),
        },
        "light_global_update": _DecoderUpdateParams(block.update),
    }

    if getattr(block.pair_features_main, "p_dmap", None) is not None:
        out["linear_1"] = LinearParams(block.pair_features_main.p_dmap.lin, has_bias=False)

    if getattr(block, "pair_features_dist", None) is not None:
        out.update({
            "layer_norm_2": LayerNormParams(block.pair_features_dist.ln),
            "layer_norm_3": LayerNormParams(block.ln_attn_dist),

            "linear_6":  LinearParams(block.pair_features_dist.p_relpos.lin, has_bias=False),
            "linear_7":  LinearParams(block.pair_features_dist.p_dmap.lin,   has_bias=False),
            "linear_8":  LinearParams(block.pair_features_dist.p_dist.lin,   has_bias=False),
            "linear_9":  LinearParams(block.pair_features_dist.p_dir.lin,    has_bias=False),
            "linear_10": LinearParams(block.pair_features_dist.p_rot.lin,    has_bias=False),
            "linear_11": LinearParams(block.pair_features_dist.p_vec.lin,    has_bias=False),

            "mlp_1": MLP2Params(block.pair_features_dist.mlp, has_bias=True),

            "sparse_structure_attn_1": {
                "ada_point_attention": _AAPointAttnParams(block.attn_dist.attn),
            },
        })

    if getattr(block, "distogram", None) is not None:
        out["inner_distogram"] = _DistogramParams(block.distogram)

    return out

# encoder (structure_autoencoder/encoder/*)

def _EncoderInitLocalParams(enc) -> dict:
    # salad: structure_autoencoder/encoder/~prepare_features/*
    # torch: encoder.init_local.*, encoder.local_ln/local_mlp are separate 
    return {
        "layer_norm": LayerNormParams(enc.init_local.ln),
        "linear": LinearParams(enc.init_local.p_relpos.lin, has_bias=False),
        "linear_1": LinearParams(enc.init_local.p_dist.lin, has_bias=False),
        "linear_2": LinearParams(enc.init_local.p_dir.lin, has_bias=False),
        "linear_3": LinearParams(enc.init_local.p_rot.lin, has_bias=False),
        "linear_4": LinearParams(enc.init_local.p_vec.lin, has_bias=False),
        "linear_5": LinearParams(enc.init_local.to_local.lin, has_bias=False),
        "mlp": MLP2Params(enc.init_local.mlp, has_bias=True),
    }


def _EncoderLocalMixParams(enc) -> dict:
    # salad: structure_autoencoder/encoder/~prepare_features/layer_norm_1 + mlp_1/*
    # torch: encoder.local_ln + encoder.local_mlp (biasless)
    return {
        "layer_norm_1": LayerNormParams(enc.local_ln),
        "mlp_1": {
            "linear": LinearParams(enc.local_mlp.layers[0].lin, has_bias=False),
            "linear_1": LinearParams(enc.local_mlp.layers[1].lin, has_bias=False),
        },
    }


def _EncoderBlockParams(block) -> dict:
    # salad: structure_autoencoder/encoder/encoder_stack/__layer_stack_no_state/encoder_block/*
    return {
        "layer_norm": LayerNormParams(block.pair_features.ln),
        "layer_norm_1": LayerNormParams(block.ln_attn),
        "layer_norm_2": LayerNormParams(block.ln_update),

        "linear": LinearParams(block.pair_features.p_relpos.lin, has_bias=False),
        "linear_1": LinearParams(block.pair_features.p_dist.lin, has_bias=False),
        "linear_2": LinearParams(block.pair_features.p_dir.lin, has_bias=False),
        "linear_3": LinearParams(block.pair_features.p_rot.lin, has_bias=False),
        "linear_4": LinearParams(block.pair_features.p_vec.lin, has_bias=False),

        "mlp": MLP2Params(block.pair_mlp, has_bias=True),

        "sparse_structure_attn": {
            "ada_point_attention": _AAPointAttnParams(block.attn.attn),
        },

        "light_global_update": _AAUpdateParams(block.update),
    }


def _EncoderParams(ae) -> dict:
    enc = ae.encoder
    blocks = list(enc.stack.blocks)
    enc_stack = stacked([_EncoderBlockParams(b) for b in blocks])  # len==1 works because assign() keeps stack dim

    # encoder/linear has b in salad but it is zero; torch has no bias
    return {
        "~prepare_features": {
            **_EncoderInitLocalParams(enc),
            **_EncoderLocalMixParams(enc),
        },
        "layer_norm": LayerNormParams(enc.ln_final),
        "linear": LinearParams(enc.to_latent.lin, has_bias=False),
        "encoder_stack": {
            "layer_norm": LayerNormParams(enc.stack.ln_final),
            "__layer_stack_no_state": {
                "encoder_block": enc_stack,
            },
        },
    }


# Top-level translation dict

def generate_translation_dict_salad_structure_autoencoder(model) -> dict:
    """
    Build translation dict keyed exactly like salad flat keys:
      structure_autoencoder/{diffusion,encoder}/...
    Expects torch model to be StructureAutoencoder with .decoder and .encoder.
    """
    if not hasattr(model, "decoder") or not hasattr(model, "encoder"):
        raise TypeError(
            "Expected a StructureAutoencoder-like torch model with attributes .decoder and .encoder "
            "(so we can import both diffusion and encoder weights)."
        )

    dec = model.decoder
    dec_blocks = list(dec.decoder_stack.blocks)
    dec_stack = stacked([_DecoderBlockParams(b) for b in dec_blocks])

    translations = {
        "structure_autoencoder": {
            "diffusion": {
                "~prepare_features": _DecoderPrepareFeaturesParams(dec),
                "mlp": _AnglePosMLPParams(dec),
                "aa_decoder": _AADecoderParams(dec),
                "decoder_stack": {
                    "__layer_stack_with_state": {
                        "decoder_block": dec_stack,
                    },
                },
            },
            "encoder": _EncoderParams(model),
        },
    }
    return translations

def import_salad_weights_(
    model,
    salad_jax_path: str,
    *,
    verbose: bool = False,
    strict_missing: bool = True,
    report: bool = True,
) -> Tuple[List[str], List[str]]:
    """
    Loads salad weights into torch model in-place.

    Returns:
      (unused_salad_keys, used_salad_keys)
    """
    orig = load_salad_flat_from_pickle(salad_jax_path)
    translations = generate_translation_dict_salad_structure_autoencoder(model)
    flat_map = process_translation_dict(translations)

    used = set(flat_map.keys())
    all_keys = set(orig.keys())
    missing_in_ckpt = sorted(list(used - all_keys))
    unused_in_ckpt = sorted(list(all_keys - used))

    _, _, torch_all = _build_state_dict_tensor_index(model)
    touched_torch = _torch_keys_touched_by_translation(model, flat_map)
    untouched_torch = sorted([k for k in torch_all if k not in touched_torch])

    if report:
        print("salad tensors:", len(orig))
        print("mapped keys:", len(flat_map))

        if missing_in_ckpt:
            print("MISSING in salad checkpoint :", missing_in_ckpt)

        print("UNUSED salad keys ", unused_in_ckpt)

        print(f"torch state_dict tensors: {len(torch_all)}")
        print(f"touched torch tensors by mapping: {len(touched_torch)}")
        print(f"UNMAPPED/UNTOUCHED torch tensors :", untouched_torch)

    if strict_missing and missing_in_ckpt:
        raise KeyError(f"Missing {len(missing_in_ckpt)} salad keys (see first ones above)")

    assign(flat_map, orig, verbose=verbose)

    return unused_in_ckpt, sorted(list(used))
