#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np


def load_npz(path: Path) -> dict:
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def compare(a: dict, b: dict, atol: float, rtol: float):
    keys_a = set(a.keys())
    keys_b = set(b.keys())
    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)
    shared = sorted(keys_a & keys_b)

    diffs = []
    for k in shared:
        va = a[k]
        vb = b[k]
        if va.shape != vb.shape:
            diffs.append((k, "shape", va.shape, vb.shape))
            continue
        if va.dtype != vb.dtype:
            # allow dtype mismatch if numeric compare passes
            pass
        if not np.allclose(va, vb, atol=atol, rtol=rtol):
            diff = np.abs(va - vb)
            diffs.append((k, "value", float(diff.max()), float(diff.mean())))
    return only_a, only_b, diffs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", required=True, help="dir with trace_*.npz (torch)")
    ap.add_argument("--right", required=True, help="dir with trace_*.npz (jax)")
    ap.add_argument("--atol", type=float, default=1e-4)
    ap.add_argument("--rtol", type=float, default=1e-4)
    args = ap.parse_args()

    left = Path(args.left)
    right = Path(args.right)
    left_files = sorted(left.glob("trace_*.npz"))
    if not left_files:
        raise SystemExit(f"No trace_*.npz in {left}")

    for lf in left_files:
        rf = right / lf.name
        if not rf.exists():
            print(f"[MISSING] {rf}")
            continue
        a = load_npz(lf)
        b = load_npz(rf)
        only_a, only_b, diffs = compare(a, b, args.atol, args.rtol)
        print(f"\n== {lf.name} ==")
        if only_a:
            print("only in left:", only_a)
        if only_b:
            print("only in right:", only_b)
        if not diffs:
            print("OK")
            continue
        for k, kind, v1, v2 in diffs:
            if kind == "shape":
                print(f"{k}: shape {v1} vs {v2}")
            else:
                print(f"{k}: max_abs={v1:.6g} mean_abs={v2:.6g}")


if __name__ == "__main__":
    main()
