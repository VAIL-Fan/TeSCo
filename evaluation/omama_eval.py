"""The O-MaMa evaluation protocol, reimplemented so our numbers can be read on it.

Published Ego-Exo4D correspondence results do not all come from the same scorer.
This repository's own `compute_metrics.py` descends from LM-EEC; O-MaMa ships a
different driver, and its published figures are on that one -- we confirmed this
by scoring O-MaMa's released predictions with its own code (IoU 44.31 against a
published 44.10, identical Loc.E). Reporting our method on both is the only way
to make the comparison legible, so this file implements O-MaMa's convention.

The metric functions are not reimplemented: `utils.eval_mask`,
`utils.location_score` and `utils.existence_accuracy` here are bit-identical to
O-MaMa's (verified by AST comparison), as are the underlying `db_eval_iou`,
`db_eval_boundary`, `f_measure` and `crop_mask`. Only the driver differs, and
the driver is what moves the number.

How this differs from `compute_metrics.py` (the LM-EEC convention):

  * it walks the GROUND TRUTH, not the prediction file;
  * it keeps only frames listed in the GT's `annotated_frames` for the target
    camera, which is roughly half the rows the GT otherwise defines;
  * it resizes the GROUND TRUTH DOWN to the prediction's size, rather than
    leaving both alone -- so the GT precision a method is judged against depends
    on that method's own output resolution;
  * it reports existence as a balanced accuracy over every row, including those
    whose target mask is empty.

One deliberate departure from O-MaMa's code. Theirs reads

    try:    pred_mask = mask_utils.decode(pred[frame_idx]["pred_mask"])
    except: breakpoint()

which, with breakpoints disabled, falls through and scores the PREVIOUS row's
mask. That is silent and wrong on any incomplete prediction file, so this
implementation verifies coverage up front and refuses to score instead. Pass
`--allow_missing` to score the covered subset and have the rest reported.

It needs the GT in O-MaMa's layout -- `{"annotations": {take: {"masks": ...,
"annotated_frames": ...}}}`, masks at their native resolution -- which is a
different file from the 480x480 `data/gt_results/*_test_gt.json` this repo's
other scorers use. See docs/METRIC_SETTINGS.md.

    python evaluation/omama_eval.py \
        --gt   /path/to/exo2egoGT.json \
        --pred outputs/<run>/merged.json \
        --task exo-ego
"""
import argparse
import json
from collections import defaultdict

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from sklearn.metrics import balanced_accuracy_score

import utils

CONF_THRESH = 0.5


def decode(rle):
    if not rle:
        return None
    counts = rle["counts"]
    return mask_utils.decode(
        {"size": rle["size"],
         "counts": counts.encode("utf-8") if isinstance(counts, str) else counts})


def confidence_of(entry):
    """Prediction files store this as a float or as a one-element list."""
    c = entry.get("confidence")
    if isinstance(c, (list, tuple)):
        return float(c[0]) if len(c) == 1 else float(np.mean(c))
    return float(c) if c is not None else 0.0


def iter_rows(gt_annotations, task):
    """Every row O-MaMa's driver visits: GT frames of the SOURCE camera, kept
    only when the TARGET camera's annotated_frames lists them."""
    for take, take_value in gt_annotations.items():
        masks = take_value.get("masks", {})
        annotated = take_value.get("annotated_frames", {})
        for obj, obj_value in masks.items():
            ego_cams = [c for c in obj_value if "aria" in c]
            exo_cams = [c for c in obj_value if "aria" not in c]
            if not ego_cams or not exo_cams:
                continue
            ego = ego_cams[0]
            for exo in exo_cams:
                if task == "exo-ego":
                    src_cam, tgt_cam, pair = exo, ego, f"{exo}_{ego}"
                else:
                    src_cam, tgt_cam, pair = ego, exo, f"{ego}_{exo}"
                allowed = set(annotated.get(obj, {}).get(tgt_cam, []))
                for frame in sorted(obj_value.get(src_cam, {})):
                    if int(frame) in allowed:
                        yield take, obj, pair, tgt_cam, str(frame)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gt", required=True, help="GT in O-MaMa's layout")
    parser.add_argument("--pred", required=True, nargs="+")
    parser.add_argument("--task", default="exo-ego", choices=["exo-ego", "ego-exo"])
    parser.add_argument("--allow_missing", action="store_true",
                        help="score the covered subset instead of refusing")
    parser.add_argument("--out")
    args = parser.parse_args()

    gt = json.load(open(args.gt))["annotations"]
    preds = None
    for path in args.pred:
        part = json.load(open(path))
        part = part[args.task]["results"] if args.task in part else part["results"]
        if preds is None:
            preds = part
            continue
        for take, tv in part.items():
            if take not in preds:
                preds[take] = tv
                continue
            for obj, ov in tv.get("masks", {}).items():
                dst = preds[take].setdefault("masks", {}).setdefault(obj, {})
                for cam, frames in ov.items():
                    dst.setdefault(cam, {}).update(frames)

    rows = list(iter_rows(gt, args.task))
    missing = [r for r in rows
               if not (preds.get(r[0], {}).get("masks", {}).get(r[1], {})
                       .get(r[2], {}).get(r[4], {}) or {}).get("pred_mask")]
    print(f"rows this protocol visits : {len(rows)}")
    print(f"rows with no prediction   : {len(missing)}")
    if missing and not args.allow_missing:
        for r in missing[:5]:
            print(f"   missing: take={r[0]} obj={r[1]} cam={r[2]} frame={r[4]}")
        raise SystemExit(
            f"\nREFUSING to score: {len(missing)} of {len(rows)} rows have no prediction.\n"
            "O-MaMa's own code would reuse the previous row's mask for these and report a\n"
            "number that is neither yours nor meaningful. Either predict those frames (see\n"
            "--restrict_frames_json on tools/correspondence_inference.py, which pins the frame\n"
            "list to another method's sampling) or pass --allow_missing to score the overlap.")

    acc = defaultdict(list)
    exist_gt, exist_pred = [], []
    resized = 0
    for take, obj, pair, tgt_cam, frame in rows:
        entry = preds[take]["masks"][obj][pair].get(frame)
        if not entry or not entry.get("pred_mask"):
            continue
        pred_mask = decode(entry["pred_mask"])
        gt_entry = gt[take]["masks"][obj].get(tgt_cam, {}).get(frame)
        gt_mask = decode(gt_entry) if gt_entry else None
        gt_exists = int(gt_mask is not None and gt_mask.any())
        exist_gt.append(gt_exists)
        exist_pred.append(int(confidence_of(entry) > CONF_THRESH))
        if not gt_exists:
            continue
        # O-MaMa's rule: the GT moves onto the prediction's grid, not the reverse.
        if gt_mask.shape != pred_mask.shape:
            gt_mask = cv2.resize(gt_mask, (pred_mask.shape[1], pred_mask.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)
            resized += 1
        iou, shape_acc = utils.eval_mask(gt_mask, pred_mask)
        acc["iou"].append(iou)
        acc["shape_acc"].append(shape_acc)
        acc["location_score"].append(
            utils.location_score(gt_mask, pred_mask, size=gt_mask.shape[:2]))

    out = {k: float(np.mean(v)) for k, v in acc.items()}
    out["balanced_acc"] = float(balanced_accuracy_score(exist_gt, exist_pred))
    out["scored_rows"] = len(acc["iou"])
    out["gt_resized"] = resized

    print(f"\ntask: {args.task}   protocol: O-MaMa (GT -> prediction grid, annotated_frames)")
    print(f"  IoU          : {100 * out['iou']:.2f}")
    print(f"  Cont.A       : {out['shape_acc']:.4f}")
    print(f"  Loc.E        : {out['location_score']:.4f}")
    print(f"  Exist (bal.) : {out['balanced_acc']:.4f}")
    print(f"  scored rows  : {out['scored_rows']}   (GT resized on {resized})")
    print("\nNot comparable to compute_metrics.py row for row: different row set, different\n"
          "grid, and a different existence definition. See docs/METRIC_SETTINGS.md.")
    if args.out:
        json.dump(out, open(args.out, "w"), indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
