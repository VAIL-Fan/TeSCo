#!/usr/bin/env python3
"""
Build a merged GT JSON in the same shape as `merge_results.py` output so that
`tools/visualize_predictions.py` can overlay ground-truth exo masks.

Reads each take's `annotation.json` under --image_dir and writes:
  { "ego-exo": { "results": { <take_id>: { "masks": { ... } } } } }

Camera keys are `egoCam_exoCam` (single underscore between names), matching merge_results.
Each entry uses the **exo** camera RLE as `pred_mask` (field name matches pred JSON).
"""

from __future__ import annotations

import argparse
import json
import os
from pycocotools import mask as mask_utils


def _rle_to_jsonable(rle):
    """Ensure RLE dict is JSON-serializable (counts as utf-8 str)."""
    if rle is None:
        return None
    if isinstance(rle, dict):
        out = dict(rle)
        c = out.get("counts")
        if isinstance(c, bytes):
            out["counts"] = c.decode("utf-8")
        elif isinstance(c, str):
            pass
        return out
    return rle


def build_take_gt(annotation_path: str) -> dict:
    with open(annotation_path, "r") as fp:
        ann = json.load(fp)
    masks_in = ann["masks"]
    masks_out: dict = {}

    for obj_name, cam_dict in masks_in.items():
        ego_cams = [c for c in cam_dict if "aria" in c]
        exo_cams = [c for c in cam_dict if c not in ego_cams]
        if not ego_cams or not exo_cams:
            continue
        masks_out[obj_name] = {}
        for ego_cam in ego_cams:
            for exo_cam in exo_cams:
                if exo_cam not in cam_dict:
                    continue
                cam_key = f"{ego_cam}_{exo_cam}"
                ego_frames = set(cam_dict[ego_cam].keys())
                exo_masks = cam_dict[exo_cam]
                masks_out[obj_name][cam_key] = {}
                for fid in ego_frames:
                    if fid not in exo_masks:
                        continue
                    rle = exo_masks[fid]
                    if not rle:
                        continue
                    # Decode/re-encode to normalize counts to str for JSON
                    try:
                        m = mask_utils.decode(rle)
                        enc = mask_utils.encode(m.astype("uint8", order="F"))
                        enc["counts"] = enc["counts"].decode("ascii")
                    except Exception:
                        enc = _rle_to_jsonable(rle)
                    masks_out[obj_name][cam_key][fid] = {"pred_mask": enc, "confidence": 1}

    return {"masks": masks_out, "subsample_idx": ann.get("subsample_idx", [])}


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--image_dir",
        required=True,
        help="EgoExo split root: <take_id>/annotation.jpg layout",
    )
    p.add_argument(
        "--output_json",
        required=True,
        help="Output path, e.g. egoexo_val_gt_viz.json",
    )
    p.add_argument(
        "--video_list_file",
        default=None,
        help="Optional: one take UUID per line (default: all subdirs of image_dir)",
    )
    args = p.parse_args()

    if args.video_list_file:
        with open(args.video_list_file, "r") as fp:
            take_ids = [ln.strip() for ln in fp if ln.strip()]
    else:
        take_ids = sorted(
            d
            for d in os.listdir(args.image_dir)
            if os.path.isdir(os.path.join(args.image_dir, d)) and not d.startswith(".")
        )

    results = {}
    for tid in take_ids:
        ap = os.path.join(args.image_dir, tid, "annotation.json")
        if not os.path.isfile(ap):
            continue
        results[tid] = build_take_gt(ap)

    out = {"ego-exo": {"version": "viz-gt", "challenge": "ego-exo", "results": results}}
    out_dir = os.path.dirname(os.path.abspath(args.output_json))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_json, "w") as fp:
        json.dump(out, fp)
    print(f"Wrote {args.output_json} with {len(results)} takes.")


if __name__ == "__main__":
    main()
