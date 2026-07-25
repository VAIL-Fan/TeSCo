"""
One-pass sweep of pred mask area fraction for existence balanced accuracy.

Loads GT and prediction JSON once, decodes each predicted mask the same way as
compute_metrics.evaluate_take, stores (gt_exists, pred_area_frac), then scores
many area thresholds in memory.

IoU / shape / location in compute_metrics are computed only when GT says the
object exists; they do not depend on this threshold — use the best threshold
here then run compute_metrics.py with --pred-exists-min-area-frac <t> to match.
"""

from __future__ import annotations

import argparse
import json
from typing import List, Tuple

import numpy as np
import tqdm
from pycocotools import mask as mask_utils
from sklearn.metrics import balanced_accuracy_score


def collect_gt_and_pred_area_frac(gt_take, pred_take) -> List[Tuple[int, float]]:
    """Mirror compute_metrics.evaluate_take; return (gt_exists, pred_area_frac)."""
    pairs: List[Tuple[int, float]] = []
    for object_id in pred_take["masks"].keys():
        cams = [x for x in pred_take["masks"][object_id].keys() if "aria" in x]
        if object_id not in gt_take["masks"].keys():
            continue
        for cam in cams:
            if cam not in gt_take["masks"][object_id].keys():
                continue
            gt_masks_exo = gt_take["masks"][object_id][cam]
            pred_masks_exo = pred_take["masks"][object_id][cam]

            for frame_idx in pred_masks_exo.keys():
                if frame_idx not in gt_masks_exo:
                    gt_mask = None
                    gt_obj_exists = 0
                else:
                    gt_mask = mask_utils.decode(gt_masks_exo[frame_idx]["pred_mask"])
                    unique_values = np.unique(gt_mask)
                    if np.array_equal(unique_values, [0]):
                        gt_obj_exists = 0
                    else:
                        gt_obj_exists = 1

                try:
                    pred_mask = mask_utils.decode(pred_masks_exo[frame_idx]["pred_mask"])
                except Exception:
                    continue

                area_frac = float(np.sum(pred_mask > 0)) / float(pred_mask.size)
                pairs.append((gt_obj_exists, area_frac))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-file", type=str, required=True)
    parser.add_argument("--pred-file", type=str, required=True)
    parser.add_argument(
        "--thresholds",
        type=str,
        default="",
        help="Comma-separated area fractions (e.g. 0.0005,0.001,0.002). "
        "If empty, uses --thresh-min/--thresh-max/--thresh-num linspace.",
    )
    parser.add_argument("--thresh-min", type=float, default=0.0002)
    parser.add_argument("--thresh-max", type=float, default=0.01)
    parser.add_argument("--thresh-num", type=int, default=40)
    args = parser.parse_args()

    with open(args.gt_file, "r") as fp:
        gt = json.load(fp)
    with open(args.pred_file, "r") as fp:
        preds = json.load(fp)

    preds = preds["ego-exo"]
    gt = gt["ego-exo"]

    all_pairs: List[Tuple[int, float]] = []
    for take_id in tqdm.tqdm(sorted(preds["results"].keys())):
        pairs = collect_gt_and_pred_area_frac(
            gt["results"][take_id], preds["results"][take_id]
        )
        all_pairs.extend(pairs)

    gt_arr = np.array([p[0] for p in all_pairs], dtype=np.int64)
    area_frac = np.array([p[1] for p in all_pairs], dtype=np.float64)
    n = len(all_pairs)
    pos_rate = float(gt_arr.sum()) / max(n, 1)
    print(f"Frames evaluated: {n}; GT positive rate (object exists): {pos_rate:.4f}")

    if args.thresholds.strip():
        thresh_list = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    else:
        thresh_list = list(
            np.linspace(args.thresh_min, args.thresh_max, args.thresh_num, dtype=np.float64)
        )

    best_t, best_ba = None, -1.0
    rows = []
    for t in thresh_list:
        pred_exists = (area_frac > float(t)).astype(np.int64)
        ba = balanced_accuracy_score(gt_arr, pred_exists)
        rows.append((t, ba))
        if ba > best_ba:
            best_ba = ba
            best_t = t

    print("\nThreshold (area frac)  |  Existence balanced accuracy")
    print("-" * 52)
    for t, ba in rows:
        mark = " *" if t == best_t else ""
        print(f"  {t:.8f}              {ba:.9f}{mark}")
    print(
        f"\nBest: pred_exists iff area_frac > {best_t:.8f}  ->  balanced_acc={best_ba:.9f}"
    )
    print(
        "\nRe-run full metrics (IoU/shape/location unchanged vs default; EL uses threshold):\n"
        f"  python evaluation/compute_metrics.py \\\n"
        f"    --gt-file {args.gt_file} \\\n"
        f"    --pred-file {args.pred_file} \\\n"
        f"    --pred-exists-min-area-frac {best_t:.8f}"
    )


if __name__ == "__main__":
    main()
