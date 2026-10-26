"""One scorer for every method compared on Ego-Exo4D correspondence.

``compute_metrics.py`` walks the *prediction* file, so a take, object, camera
pair or frame that a method never emitted is silently absent from the mean.
That is fine for a submission validated to cover the whole test set, but it
makes two methods with different coverage incomparable: the one that answers
only where it is confident is rewarded for abstaining.

This script walks the *ground truth* instead and scores a missing prediction
as an empty mask (IoU 0), which is what the official EGO4D driver does
(``XMem/scripts/evaluation.py`` iterates ``gt["annotation"].keys()``). It also
resizes a prediction onto the GT grid with INTER_NEAREST when the two differ,
matching ``utils.reshape_img_nopad_square``, so methods that predict at their
own resolution can be scored without editing their output.

The metric functions themselves are untouched -- ``utils.eval_mask``,
``utils.location_score`` and ``utils.existence_accuracy`` are the same ones
``compute_metrics.py`` calls.

    python evaluation/unified_eval.py \
        --gt  /path/to/egoexo_test_gt.json \
        --pred outputs/tesco_ego2exo/merged.json \
        --task ego-exo

Pass ``--coverage pred`` to reproduce the prediction-driven convention, or
``--coverage both`` to print the two side by side and see what coverage costs.
"""
import argparse
import json
from collections import defaultdict

import cv2
import numpy as np
import tqdm
from pycocotools import mask as mask_utils

import utils

EVAL_SIZE = (480, 480)   # the benchmark grid; GT is distributed at this size


def decode(entry):
    """RLE -> binary mask, or None when the entry carries no usable mask."""
    if not entry:
        return None
    rle = entry.get("pred_mask")
    if not rle:
        return None
    counts = rle["counts"]
    try:
        return mask_utils.decode(
            {"size": rle["size"],
             "counts": counts.encode("utf-8") if isinstance(counts, str) else counts})
    except Exception:
        return None


def to_grid(mask, shape):
    """Resize a binary mask onto `shape` with nearest neighbour (official rule)."""
    if mask.shape[:2] == tuple(shape):
        return mask
    h, w = shape
    return cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)


def iter_gt(gt_results):
    """Every (take, object, cam, frame, entry) the GT defines."""
    for take, take_value in gt_results.items():
        for obj, obj_value in take_value.get("masks", {}).items():
            for cam, frames in obj_value.items():
                for frame, entry in frames.items():
                    yield take, obj, cam, frame, entry


def lookup(pred_results, take, obj, cam, frame):
    try:
        return pred_results[take]["masks"][obj][cam][frame]
    except (KeyError, TypeError):
        return None


def score(gt, preds, task, coverage, conf_thresh=0.5):
    gt_results = gt[task]["results"]
    pred_results = preds[task]["results"] if task in preds else preds["results"]

    acc = defaultdict(list)
    stats = dict(gt_rows=0, scoreable=0, scored=0, missing=0, empty_gt=0, resized=0)

    for take, obj, cam, frame, gt_entry in tqdm.tqdm(
            list(iter_gt(gt_results)), desc=f"{task}/{coverage}"):
        stats["gt_rows"] += 1
        gt_mask = decode(gt_entry)
        # Official convention: IoU/shape/location are defined only where the
        # object is actually visible in the target view.
        if gt_mask is None or not gt_mask.any():
            stats["empty_gt"] += 1
            continue
        stats["scoreable"] += 1

        pred_entry = lookup(pred_results, take, obj, cam, frame)
        pred_mask = decode(pred_entry)

        if pred_mask is None:
            stats["missing"] += 1
            if coverage == "pred":
                continue                      # the skip convention
            pred_mask = np.zeros_like(gt_mask)  # full coverage: a miss scores 0
        elif pred_mask.shape[:2] != gt_mask.shape[:2]:
            stats["resized"] += 1
            pred_mask = to_grid(pred_mask, gt_mask.shape[:2])

        # Existence accuracy here is the per-row agreement on "is the object
        # present", averaged over scoreable rows only -- not compute_metrics.py's
        # balanced accuracy over every row including empty-GT ones.
        iou, shape_acc = utils.eval_mask(gt_mask, pred_mask)
        acc["IoU"].append(iou)
        acc["ShapeAcc"].append(shape_acc)
        acc["ExistAcc"].append(utils.existence_accuracy(gt_mask, pred_mask))
        acc["LocE"].append(utils.location_score(gt_mask, pred_mask, size=EVAL_SIZE))
        stats["scored"] += 1

    summary = {k: float(np.mean(v)) if v else float("nan") for k, v in acc.items()}
    # IoU and existence accuracy are reported as percentages, contour accuracy
    # (ShapeAcc) and location error as fractions -- matching how the paper's
    # table prints them.
    summary["IoU"] *= 100
    summary["ExistAcc"] *= 100
    return summary, stats


def load(paths):
    """Load one prediction file, or merge several shards into one.

    Shards are not necessarily split by take: O-MaMa's four shards each carry
    all 291 takes and divide the frames between them, so merging at the take
    level with dict.update() would silently keep only the last shard's rows
    (66,339 of 265,358). The merge therefore has to go all the way down to the
    frame.
    """
    merged = None
    for path in paths:
        part = json.load(open(path))
        if merged is None:
            merged = part
            continue
        for task, task_value in part.items():
            if task not in merged:
                merged[task] = task_value
                continue
            dst = merged[task]["results"]
            for take, take_value in task_value["results"].items():
                if take not in dst:
                    dst[take] = take_value
                    continue
                for obj, obj_value in take_value.get("masks", {}).items():
                    dst_obj = dst[take].setdefault("masks", {}).setdefault(obj, {})
                    for cam, frames in obj_value.items():
                        dst_obj.setdefault(cam, {}).update(frames)
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gt", required=True)
    parser.add_argument("--pred", required=True, nargs="+",
                        help="one prediction file, or several shards to merge")
    parser.add_argument("--task", default="ego-exo", choices=["ego-exo", "exo-ego"])
    parser.add_argument("--coverage", default="gt", choices=["gt", "pred", "both"],
                        help="gt = missing predictions score 0 (default, comparable); "
                             "pred = skip them (reproduces compute_metrics.py)")
    parser.add_argument("--out", help="write the summary as JSON here")
    args = parser.parse_args()

    gt = json.load(open(args.gt))
    preds = load(args.pred)

    modes = ["gt", "pred"] if args.coverage == "both" else [args.coverage]
    results = {}
    for mode in modes:
        summary, stats = score(gt, preds, args.task, mode)
        results[mode] = {**summary, **stats}

    label = {"gt": "full coverage (missing = 0)", "pred": "skip missing"}
    print(f"\ntask: {args.task}   eval grid: {EVAL_SIZE[0]}x{EVAL_SIZE[1]} (nearest)")
    print(f"{'convention':28s} {'IoU':>7} {'Cont.A':>7} {'Loc.E':>7} {'Exist':>7} {'scored':>8} {'missing':>8}")
    print("-" * 80)
    for mode in modes:
        r = results[mode]
        print(f"{label[mode]:28s} {r['IoU']:7.2f} {r['ShapeAcc']:7.3f} "
              f"{r['LocE']:7.4f} {r['ExistAcc']:7.2f} {r['scored']:8d} {r['missing']:8d}")
    s = results[modes[0]]
    print(f"\nGT rows {s['gt_rows']}, of which {s['scoreable']} have a non-empty "
          f"target mask; {s['resized']} predictions needed resizing.")
    if s["missing"]:
        print(f"WARNING: {s['missing']} scoreable GT rows have no prediction "
              f"({100 * s['missing'] / max(s['scoreable'], 1):.1f}%). Under "
              f"'skip missing' these vanish from the mean.")

    if args.out:
        json.dump(results, open(args.out, "w"), indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
