"""Summarise a correspondence GT file: takes, rows, scoreable rows, RLE size.

Scoreable = the target-view mask is non-empty. IoU / contour accuracy /
location error are defined only on those rows, so this count is what a reported
mean is actually averaged over.

    python tools/check_gt.py data/gt_results/egoexo_test_gt.json ego-exo
"""
import argparse
import json
from collections import Counter

import numpy as np
from pycocotools import mask as mask_utils

EXPECTED = {
    "ego-exo": dict(takes=291, rows=163947, scoreable=132503),
    "exo-ego": dict(takes=291, rows=265358, scoreable=132556),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("gt")
    parser.add_argument("task", choices=["ego-exo", "exo-ego"])
    args = parser.parse_args()

    results = json.load(open(args.gt))[args.task]["results"]
    rows = scoreable = 0
    sizes = Counter()
    for take_value in results.values():
        for obj_value in take_value.get("masks", {}).values():
            for frames in obj_value.values():
                for entry in frames.values():
                    rows += 1
                    rle = entry.get("pred_mask")
                    if not rle:
                        continue
                    sizes[tuple(rle["size"])] += 1
                    counts = rle["counts"]
                    decoded = mask_utils.decode(
                        {"size": rle["size"],
                         "counts": counts.encode("utf-8") if isinstance(counts, str) else counts})
                    if decoded.any():
                        scoreable += 1

    found = dict(takes=len(results), rows=rows, scoreable=scoreable)
    want = EXPECTED[args.task]
    print(f"task key   : {args.task}")
    for field in ("takes", "rows", "scoreable"):
        mark = "ok" if found[field] == want[field] else f"MISMATCH (expected {want[field]})"
        print(f"{field:11s}: {found[field]:>8}   {mark}")
    print(f"RLE sizes  : {dict(sizes)}")
    if list(sizes) != [(480, 480)]:
        print("WARNING: evaluation assumes a 480x480 GT grid; this file is not uniformly 480x480.")


if __name__ == "__main__":
    main()
