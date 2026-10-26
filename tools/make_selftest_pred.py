"""Write a prediction file that is a copy of the ground truth, for N takes.

A scorer should be checked on an input whose answer is known before it is used
to rank methods. Feeding the GT back as its own prediction forces two numbers:
the skip convention must report a perfect 100.00 IoU (anything less is broken
metric plumbing, not a weak model), and the full-coverage mean must equal the
covered fraction of scoreable rows.

    python tools/make_selftest_pred.py \
        --gt data/gt_results/egoexo_test_gt.json --task ego-exo \
        --takes 20 --out /tmp/selftest_480.json

``--upscale 700`` re-encodes the same masks at 700x700, which exercises the
resize path that a method predicting off-grid (e.g. O-MaMa) needs.
"""
import argparse
import json

import cv2
import numpy as np
from pycocotools import mask as mask_utils


def decode(entry):
    rle = entry.get("pred_mask") if entry else None
    if not rle:
        return None
    counts = rle["counts"]
    return mask_utils.decode(
        {"size": rle["size"],
         "counts": counts.encode("utf-8") if isinstance(counts, str) else counts})


def encode(mask):
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return {"pred_mask": rle, "confidence": 1.0}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gt", required=True)
    parser.add_argument("--task", default="ego-exo", choices=["ego-exo", "exo-ego"])
    parser.add_argument("--takes", type=int, default=20,
                        help="how many takes to cover; 0 covers all of them")
    parser.add_argument("--upscale", type=int, default=0,
                        help="re-encode masks at this square size instead of the GT size")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    gt = json.load(open(args.gt))[args.task]
    results = gt["results"]
    takes = sorted(results)
    if args.takes:
        takes = takes[:args.takes]

    out = {args.task: {"version": gt.get("version", "1.0"),
                       "challenge": gt.get("challenge", "correspondence"),
                       "results": {}}}
    rows = 0
    for take in takes:
        take_out = {"masks": {}, "subsample_idx": results[take].get("subsample_idx")}
        for obj, obj_value in results[take].get("masks", {}).items():
            take_out["masks"][obj] = {}
            for cam, frames in obj_value.items():
                entries = {}
                for frame, entry in frames.items():
                    mask = decode(entry)
                    if mask is None:
                        continue
                    if args.upscale:
                        mask = cv2.resize(mask, (args.upscale, args.upscale),
                                          interpolation=cv2.INTER_NEAREST)
                    entries[frame] = encode(mask)
                    rows += 1
                take_out["masks"][obj][cam] = entries
        out[args.task]["results"][take] = take_out

    json.dump(out, open(args.out, "w"))
    size = f"{args.upscale}x{args.upscale}" if args.upscale else "GT size (480x480)"
    print(f"wrote {args.out}")
    print(f"  takes covered : {len(takes)} of {len(results)}")
    print(f"  rows written  : {rows}   at {size}")
    print("\nExpected from unified_eval.py --coverage both:")
    if args.upscale:
        # A round trip at a non-integer scale (700/480 ~= 1.458) misaligns the
        # sampling grids and shifts every boundary by about a pixel, which costs
        # far more IoU than it looks like it should: measured 76.64 at 700.
        # This checks that the resize path runs and that `resized` counts
        # correctly -- it is NOT an estimate of what any real method pays, since
        # a method predicting natively at 700 is resized once, not twice.
        print("  skip missing  : IoU well below 100 (76.64 at --upscale 700) --")
        print("                  the point of this run is that `resized` equals the row count,")
        print("                  not the IoU value. Do not read it as a method's resize cost.")
    else:
        print("  skip missing  : IoU 100.00 exactly")
    print("  full coverage : IoU = 100 * (scored non-empty rows of these takes) / (all scoreable rows)")


if __name__ == "__main__":
    main()
