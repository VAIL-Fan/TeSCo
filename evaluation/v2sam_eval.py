"""V2-SAM's SegMetric, so the convention this paper follows can be run here.

The paper states "Following [V2-SAM], we use IoU, Loc.E and Cont.A provided in
[Ego-Exo4D]", and Table 1's comparison rows are quoted through V2-SAM's table.
Its repository ships exactly one evaluator, `SegMetric` in
`projects/v2sam_*/evaluation/seg_metric.py`, reproduced here:

    inter = (pred * gt).sum()
    union = (pred + gt - pred * gt).sum()
    iou   = inter / (union + 1e-6)
    dice  = 2 * inter / (pred.sum() + gt.sum() + 1e-6)

averaged over every object instance it is handed.

Two things follow, and both matter when reading a comparison table:

  * It ships **no contour-accuracy and no location-error metric.** The Cont.A
    and Loc.E columns of its own table therefore came from code it did not
    release; ours come from the official Ego-Exo4D implementations, which is
    what "provided in [Ego-Exo4D]" means. Run `unified_eval.py` for those two.
  * It does **not** skip rows whose ground truth is empty, so a frame where the
    object is genuinely absent and the model correctly predicts nothing scores
    0, not 1. `--skip_empty_gt` isolates that choice; on our Exo->Ego
    predictions it is worth 0.04 IoU, because only 6 of 13,866 rows are empty.

It also does no resizing: masks are compared as stored. That is harmless here
only because this model predicts at the ground truth's own 480x480.

    python evaluation/v2sam_eval.py \
        --gt   data/gt_results/exoego_test_gt.json \
        --pred outputs/<run>/merged.json --task exo-ego
"""
import argparse
import json

import numpy as np
from pycocotools import mask as mask_utils


def decode(entry):
    """Accept a bare RLE or one wrapped under "pred_mask"."""
    if not entry:
        return None
    rle = entry if "counts" in entry else entry.get("pred_mask")
    if not rle:
        return None
    c = rle["counts"]
    return mask_utils.decode({"size": rle["size"],
                              "counts": c.encode("utf-8") if isinstance(c, str) else c})


def seg_metric(pred, gt):
    """V2-SAM's SegMetric, on float arrays, element for element."""
    p = pred.astype(np.float64)
    g = gt.astype(np.float64)
    inter = (p * g).sum()
    union = (p + g - p * g).sum()
    return inter / (union + 1e-6), (2 * inter) / (p.sum() + g.sum() + 1e-6)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True, nargs="+")
    ap.add_argument("--task", default="exo-ego", choices=["exo-ego", "ego-exo"])
    ap.add_argument("--skip_empty_gt", action="store_true",
                    help="drop rows with an empty target mask; V2-SAM keeps them")
    ap.add_argument("--score_missing_as_zero", action="store_true",
                    help="also score GT rows your file does not predict. V2-SAM's evaluator has "
                         "no notion of a missing prediction -- it only sees what its dataloader "
                         "yields -- so by default those rows are left out and reported instead. "
                         "Turn this on only to measure coverage deliberately.")
    ap.add_argument("--out")
    args = ap.parse_args()

    gt = json.load(open(args.gt))[args.task]["results"]

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

    ious, dices = [], []
    rows = empty_gt = missing = 0
    for take, tv in gt.items():
        for obj, ov in tv.get("masks", {}).items():
            for cam, frames in ov.items():
                for frame, entry in frames.items():
                    g = decode(entry)
                    if g is None:
                        continue
                    rows += 1
                    if not g.any():
                        empty_gt += 1
                        if args.skip_empty_gt:
                            continue
                    try:
                        p = decode(preds[take]["masks"][obj][cam][frame])
                    except (KeyError, TypeError):
                        p = None
                    if p is None:
                        missing += 1
                        if not args.score_missing_as_zero:
                            continue
                        p = np.zeros_like(g)
                    elif p.shape != g.shape:
                        raise SystemExit(
                            f"shape mismatch {p.shape} vs {g.shape}: V2-SAM's SegMetric does no "
                            "resizing, so the two must already be on one grid. Use unified_eval.py "
                            "if your predictions are not at the ground truth's resolution.")
                    i, d = seg_metric(p, g)
                    ious.append(i); dices.append(d)

    print(f"\ntask: {args.task}   scorer: V2-SAM SegMetric"
          f"{' (empty GT skipped)' if args.skip_empty_gt else ''}")
    print(f"  mean IoU     : {100 * float(np.mean(ious)):.2f}")
    print(f"  mean Dice    : {100 * float(np.mean(dices)):.2f}")
    print(f"  rows scored  : {len(ious)}  of {rows} GT rows")
    print(f"  empty GT     : {empty_gt}" + ("  (excluded)" if args.skip_empty_gt else "  (scored 0 unless the prediction is also empty)"))
    print(f"  no prediction: {missing}" + ("  (scored 0)" if args.score_missing_as_zero else "  (left out)"))
    if missing and not args.score_missing_as_zero:
        print(f"\n  NOTE: this file predicts {100 * (rows - missing) / max(rows, 1):.1f}% of the GT. "
              "The mean above is over that part only,\n        which is comparable to a published "
              "number only if that number covers the same rows.")
    print("\nNo contour accuracy or location error: V2-SAM ships neither. Use unified_eval.py.")
    if args.out:
        json.dump({"mean_IoU": float(np.mean(ious)), "mean_Dice": float(np.mean(dices)),
                   "rows": len(ious), "empty_gt": empty_gt, "missing": missing},
                  open(args.out, "w"), indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
