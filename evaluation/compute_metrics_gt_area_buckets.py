"""
Aggregate IoU / shape / location / existence metrics by GT foreground area ratio per frame.

Ratio (percent): 100 * (number of GT foreground pixels) / (H * W) for that decoded GT mask.

Buckets (10): 0-10, 10-20, ..., 90-100 (percent).

Mirrors ``compute_metrics.evaluate_take`` traversal and thresholds.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Tuple

import numpy as np
from pycocotools import mask as mask_utils
from sklearn.metrics import balanced_accuracy_score

import utils
from compute_metrics import CONF_THRESH
from compute_metrics_scenarios import normalize_preds_exoego_cam_keys

H_DEF, W_DEF = 480, 480


def bucket_label(ratio_pct: float) -> str:
    idx = int(np.floor(float(ratio_pct) / 10.0))
    idx = min(max(idx, 0), 9)
    low = idx * 10
    return f"{low}-{low + 10}"


def accumulate_buckets(
    gt_take: dict,
    pred_take: dict,
    existence_conf_thresh: float = CONF_THRESH,
    pred_exists_min_area_frac: Optional[float] = None,
    location_hw: Tuple[int, int] = (H_DEF, W_DEF),
) -> Tuple[
    DefaultDict[str, Dict[str, List]],
    DefaultDict[str, List[int]],
    DefaultDict[str, List[int]],
]:
    """
    Returns per bucket:
      geo[bucket]['iou'|'shape'|'loc'|'exist_acc'] lists (GT-present frames only)
      exist_gt[bucket], exist_pred[bucket] for ALL frames (present bucket by GT ratio bin when present,
      ``gt_absent`` when GT empty).
    """
    geo: DefaultDict[str, Dict[str, List]] = defaultdict(lambda: defaultdict(list))
    exist_gt: DefaultDict[str, List[int]] = defaultdict(list)
    exist_pred: DefaultDict[str, List[int]] = defaultdict(list)

    gt_masks_root = gt_take.get("masks") or {}
    pred_masks_root = pred_take.get("masks") or {}

    for object_id in pred_masks_root.keys():
        cams = [x for x in pred_masks_root[object_id].keys() if "aria" in x]
        if object_id not in gt_masks_root:
            continue

        for cam in cams:
            if cam not in gt_masks_root[object_id]:
                continue
            gt_masks_exo = gt_masks_root[object_id][cam]
            pred_masks_exo = pred_masks_root[object_id][cam]

            for frame_idx in pred_masks_exo.keys():
                if frame_idx not in gt_masks_exo:
                    gt_mask = None
                    gt_obj_exists = 0
                else:
                    gt_mask = mask_utils.decode(gt_masks_exo[frame_idx]["pred_mask"])
                    gt_obj_exists = 0 if np.array_equal(np.unique(gt_mask), [0]) else 1

                try:
                    pred_mask = mask_utils.decode(pred_masks_exo[frame_idx]["pred_mask"])
                except Exception:
                    continue

                if pred_exists_min_area_frac is not None:
                    pred_obj_exists = int(
                        np.sum(pred_mask > 0)
                        > float(pred_exists_min_area_frac) * pred_mask.size
                    )
                else:
                    pred_obj_exists = int(
                        pred_masks_exo[frame_idx]["confidence"] > existence_conf_thresh
                    )

                if gt_obj_exists and gt_mask is not None:
                    ratio_pct = 100.0 * float(np.sum(gt_mask > 0)) / float(gt_mask.size)
                    eb = bucket_label(ratio_pct)
                    exist_gt[eb].append(1)
                    exist_pred[eb].append(pred_obj_exists)

                    try:
                        iou, shape_acc = utils.eval_mask(gt_mask, pred_mask)
                    except Exception:
                        continue

                    loc = utils.location_score(gt_mask, pred_mask, size=location_hw)
                    ex_acc = utils.existence_accuracy(gt_mask, pred_mask)

                    geo[eb]["iou"].append(float(iou))
                    geo[eb]["shape"].append(float(shape_acc))
                    geo[eb]["loc"].append(float(loc))
                    geo[eb]["exist_acc"].append(float(ex_acc))
                else:
                    exist_gt["gt_absent"].append(0)
                    exist_pred["gt_absent"].append(pred_obj_exists)

    return geo, exist_gt, exist_pred


def summarize_geo(geo_vals: List[float]) -> Optional[float]:
    return float(np.mean(geo_vals)) if geo_vals else None


def summarize_bucket_metrics(
    geo: DefaultDict[str, Dict[str, List]],
    exist_gt: DefaultDict[str, List[int]],
    exist_pred: DefaultDict[str, List[int]],
) -> Dict[str, dict]:
    order = [f"{i}-{i + 10}" for i in range(0, 100, 10)]
    out: Dict[str, dict] = {}

    for lab in order:
        g = geo[lab]
        out[lab] = {
            "gt_present_frames": len(g["iou"]),
            "TOTAL_IOU": summarize_geo(g["iou"]),
            "TOTAL_SHAPE_ACC": summarize_geo(g["shape"]),
            "TOTAL_LOCATION_SCORE": summarize_geo(g["loc"]),
            "EXISTENCE_ACC_MEAN": summarize_geo(g["exist_acc"]),
            "TOTAL_EXISTENCE_BALANCED_ACC": (
                float(balanced_accuracy_score(exist_gt[lab], exist_pred[lab]))
                if len(set(exist_gt[lab])) > 1 and len(exist_gt[lab])
                else None
            ),
            "existence_pairs_in_bucket": len(exist_gt[lab]),
        }

    ga_gt = exist_gt["gt_absent"]
    ga_pr = exist_pred["gt_absent"]
    out["gt_absent"] = {
        "frames": len(ga_gt),
        "TOTAL_EXISTENCE_BALANCED_ACC": (
            float(balanced_accuracy_score(ga_gt, ga_pr))
            if len(set(ga_gt)) > 1 and len(ga_gt)
            else None
        ),
    }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-file", type=str, required=True)
    parser.add_argument("--pred-file", type=str, required=True)
    parser.add_argument("--task-key", type=str, required=True, choices=["ego-exo", "exo-ego"])
    parser.add_argument("--existence-conf-thresh", type=float, default=CONF_THRESH)
    parser.add_argument("--pred-exists-min-area-frac", type=float, default=None)
    parser.add_argument("--out-json", type=str, required=True)
    parser.add_argument(
        "--note",
        type=str,
        default="",
        help="Recording which GT file / pipeline this run corresponds to.",
    )
    parser.add_argument(
        "--normalize-exoego-cam-keys",
        action="store_true",
        help="Rewrite prediction camera keys to {exo}_{aria} (see compute_metrics_scenarios).",
    )
    args = parser.parse_args()

    with open(args.gt_file) as f:
        gt = json.load(f)
    with open(args.pred_file) as f:
        preds = json.load(f)

    if args.normalize_exoego_cam_keys:
        preds = normalize_preds_exoego_cam_keys(preds, args.task_key)

    gt_b = gt[args.task_key]
    pr_b = preds[args.task_key]

    agg_geo: DefaultDict[str, Dict[str, List]] = defaultdict(lambda: defaultdict(list))
    agg_ex_gt: DefaultDict[str, List[int]] = defaultdict(list)
    agg_ex_pr: DefaultDict[str, List[int]] = defaultdict(list)

    for take_id in pr_b["results"]:
        if take_id not in gt_b["results"]:
            continue
        g, eg, ep = accumulate_buckets(
            gt_b["results"][take_id],
            pr_b["results"][take_id],
            existence_conf_thresh=args.existence_conf_thresh,
            pred_exists_min_area_frac=args.pred_exists_min_area_frac,
        )
        for b, metrics in g.items():
            for k, vals in metrics.items():
                agg_geo[b][k].extend(vals)
        for b in eg:
            agg_ex_gt[b].extend(eg[b])
            agg_ex_pr[b].extend(ep[b])

    buckets = summarize_bucket_metrics(agg_geo, agg_ex_gt, agg_ex_pr)

    payload = {
        "note": args.note,
        "gt_file": args.gt_file,
        "pred_file": args.pred_file,
        "task_key": args.task_key,
        "normalize_exoego_cam_keys": bool(args.normalize_exoego_cam_keys),
        "ratio_definition_percent": "100 * gt_foreground_pixels / (H * W) for decoded GT mask",
        "buckets_percent_intervals": [f"{i}-{i + 10}" for i in range(0, 100, 10)],
        "gt_absent_bucket_note": "Frames with empty GT mask for matched pred frames (existence-only)",
        "buckets": buckets,
    }

    with open(args.out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print("wrote", args.out_json)


if __name__ == "__main__":
    main()
