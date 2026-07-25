#!/usr/bin/env python3
"""
Convert a raw training checkpoint (training/checkpoints_EgoExo/*.pth, a plain state_dict
saved by training/trainer.py, with DDP's "module." prefix on every key) into the format
`sam2.build_sam._load_checkpoint` expects for inference: a dict with a top-level "model" key
holding the state_dict, with "module." stripped.

Usage:
    python tools/prepare_inference_checkpoint.py \
      --in training/checkpoints_EgoExo/tesco_egoexo4d_epoch59.pth \
      --out training/checkpoints_EgoExo/tesco_egoexo4d_epoch59_for_inference.pt
"""
from __future__ import annotations

import argparse

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_path", required=True, help="raw training checkpoint (state_dict)")
    parser.add_argument("--out", dest="out_path", required=True, help="output path, e.g. *_for_inference.pt")
    args = parser.parse_args()

    sd = torch.load(args.in_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        # Already in the wrapped format (e.g. the official SAM2.1 init checkpoint) -- nothing to do.
        print(f"{args.in_path} already has a top-level 'model' key; copying as-is.")
        torch.save(sd, args.out_path)
        return

    if not isinstance(sd, dict):
        raise SystemExit(f"Unexpected checkpoint format in {args.in_path}: top-level type {type(sd)}")

    stripped = {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in sd.items()
    }
    torch.save({"model": stripped}, args.out_path)
    print(f"Wrote {args.out_path}: {len(stripped)} keys (stripped 'module.' prefix, wrapped under 'model').")


if __name__ == "__main__":
    main()
