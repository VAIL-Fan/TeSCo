"""
Per-scenario metrics for exo-ego GT/pred JSONs using the same logic as compute_metrics.evaluate_take.

Scenarios are assigned from object id strings (e.g. ``white bowl_0``) via keyword rules.
Objects that match no scenario are skipped (reported as unmapped).
"""
from __future__ import annotations

import argparse
import copy
import json
import re
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm

from compute_metrics import CONF_THRESH, evaluate_take


GROUP_PATTERNS: Dict[str, List[str]] = {
    "Cooking": [
        r"cook",
        r"kitchen",
        r"pan",
        r"pot",
        r"bowl",
        r"plate",
        r"spatula",
        r"spoon",
        r"fork",
        r"knife",
        r"chop",
        r"egg",
        r"onion",
        r"celery",
        r"carrot",
        r"cucumber",
        r"tomato",
        r"pepper",
        r"sauce",
        r"oil",
        r"salt",
        r"sugar",
        r"honey",
        r"milk",
        r"noodle",
        r"omelet",
        r"board",
        r"kettle",
        r"tea",
        r"coffee",
        r"recipe",
        r"cilantro",
        r"coriander",
        r"garlic",
    ],
    "Music": [
        r"piano",
        r"guitar",
        r"violin",
        r"viola",
        r"cello",
        r"flute",
        r"sax",
        r"trumpet",
        r"drum",
        r"sheet music",
        r"bow",
    ],
    "Health": [
        r"covid",
        r"test",
        r"cassette",
        r"swab",
        r"sterile",
        r"buffer",
        r"cpr",
        r"dummy",
        r"device",
        r"kit",
        r"nasal",
    ],
    "Basketball": [r"basketball", r"hoop"],
    "Bike Repair": [
        r"bicycle",
        r"bike",
        r"wheel",
        r"pedal",
        r"chain",
        r"tire",
        r"brake",
        r"seat",
        r"tube",
        r"pump",
        r"repair stand",
        r"inner tube",
        r"frame",
    ],
}

# First match wins — put more specific / less ambiguous groups first.
_SCENARIO_ORDER = ["Basketball", "Bike Repair", "Music", "Health", "Cooking"]


def _compile_patterns() -> Dict[str, List[re.Pattern]]:
    return {k: [re.compile(p, re.I) for p in v] for k, v in GROUP_PATTERNS.items()}


_COMPILED = _compile_patterns()


def canonical_exoego_cam_key(cam: str) -> str:
    """
    Map prediction camera strings to GT-style ``{exo}_{aria}`` used in ``exoego_test_gt.json``
    (same convention as ``merge_results.py`` with ``--swap``).

    Some exports use ``__`` folders merged to single underscores, or ``{aria}_{exo}`` instead
    of ``{exo}_{aria}`` — this normalizes both.
    """
    if "__" in cam:
        a, b = cam.split("__", 1)
        if "aria" in a.lower():
            aria, exo = a, b
        else:
            exo, aria = a, b
        return f"{exo}_{aria}"
    if cam.startswith("cam") or cam.startswith("gp"):
        return cam
    i = cam.rfind("_")
    if i <= 0:
        return cam
    tail = cam[i + 1 :]
    head = cam[:i]
    if (tail.startswith("cam") or tail.startswith("gp")) and "aria" in head.lower():
        return f"{tail}_{head}"
    return cam


def normalize_preds_exoego_cam_keys(preds: dict, task_key: str) -> dict:
    """Deep-copy preds and rewrite ``masks[*][camera]`` keys to canonical exo-ego form."""
    out = copy.deepcopy(preds)
    res = out[task_key]["results"]
    for tid in res:
        masks = res[tid].get("masks") or {}
        new_masks: Dict[str, Dict] = {}
        for oid, cams in masks.items():
            nc: Dict[str, dict] = {}
            for ck, frames in cams.items():
                nk = canonical_exoego_cam_key(ck)
                if nk in nc:
                    nc[nk].update(frames)
                else:
                    nc[nk] = frames
            new_masks[oid] = nc
        res[tid]["masks"] = new_masks
    return out


def scenario_for_object(object_id: str) -> Optional[str]:
    for scen in _SCENARIO_ORDER:
        if any(p.search(object_id) for p in _COMPILED[scen]):
            return scen
    return None


def _filter_take_masks(
    take: dict, allowed: Callable[[str], bool]
) -> Tuple[dict, List[str]]:
    """Return a shallow copy of take with ``masks`` filtered to allowed object ids."""
    masks = take.get("masks") or {}
    kept = {k: v for k, v in masks.items() if allowed(k)}
    dropped = [k for k in masks if k not in kept]
    return {"masks": kept}, dropped


def evaluate_scenarios(
    gt: dict,
    preds: dict,
    task_key: str,
    existence_conf_thresh: float = CONF_THRESH,
    pred_exists_min_area_frac: Optional[float] = None,
) -> Dict[str, dict]:
    gt_b = gt[task_key]
    pr_b = preds[task_key]

    scenario_keys = list(GROUP_PATTERNS.keys())
    out: Dict[str, dict] = {}

    all_objects_seen = set()
    for tid in pr_b["results"]:
        all_objects_seen.update((pr_b["results"][tid].get("masks") or {}).keys())

    unmapped_objects = sorted(
        oid for oid in all_objects_seen if scenario_for_object(oid) is None
    )

    for scen in scenario_keys:
        total_iou: List[float] = []
        total_shape: List[float] = []
        total_exist_acc: List[float] = []
        total_loc: List[float] = []
        obj_exist_gt: List[int] = []
        obj_exist_pred: List[int] = []
        takes_with_hits = 0
        object_hits = 0

        for take_id in tqdm(pr_b["results"], desc=f"scenario:{scen}"):
            if take_id not in gt_b["results"]:
                continue

            gt_masks = (gt_b["results"][take_id].get("masks") or {})

            def allow(oid: str) -> bool:
                return scenario_for_object(oid) == scen and oid in gt_masks

            pred_take_f, _ = _filter_take_masks(pr_b["results"][take_id], allow)
            gt_take_f, _ = _filter_take_masks(gt_b["results"][take_id], allow)

            if not pred_take_f["masks"]:
                continue

            takes_with_hits += 1
            object_hits += len(pred_take_f["masks"])

            ious, shape_accs, existence_accs, location_scores, o_gt, o_pred, *_ = evaluate_take(
                gt_take_f,
                pred_take_f,
                existence_conf_thresh=existence_conf_thresh,
                pred_exists_min_area_frac=pred_exists_min_area_frac,
            )

            total_iou += ious
            total_shape += shape_accs
            total_exist_acc += existence_accs
            total_loc += location_scores
            obj_exist_gt += o_gt
            obj_exist_pred += o_pred

        bal = None
        if len(set(obj_exist_gt)) > 1:
            bal = float(balanced_accuracy_score(obj_exist_gt, obj_exist_pred))

        out[scen] = {
            "TOTAL_EXISTENCE_BALANCED_ACC": bal,
            "TOTAL_IOU": float(np.mean(total_iou)) if total_iou else None,
            "TOTAL_LOCATION_SCORE": float(np.mean(total_loc)) if total_loc else None,
            "TOTAL_SHAPE_ACC": float(np.mean(total_shape)) if total_shape else None,
            "EXISTENCE_ACC_MEAN": float(np.mean(total_exist_acc)) if total_exist_acc else None,
            "takes_with_objects_in_scenario": takes_with_hits,
            "object_keys_counted_in_pred": object_hits,
            "frame_metric_count_iou": len(total_iou),
            "existence_pair_count": len(obj_exist_gt),
        }

    out["_meta"] = {
        "unmapped_object_names_in_pred": unmapped_objects,
        "unmapped_count": len(unmapped_objects),
    }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-file", type=str, required=True)
    parser.add_argument("--pred-file", type=str, required=True)
    parser.add_argument("--task-key", type=str, default="exo-ego")
    parser.add_argument("--existence-conf-thresh", type=float, default=CONF_THRESH)
    parser.add_argument("--pred-exists-min-area-frac", type=float, default=None)
    parser.add_argument("--out-json", type=str, default=None)
    parser.add_argument(
        "--normalize-exoego-cam-keys",
        action="store_true",
        help="Rewrite prediction camera keys to {exo}_{aria} to match GT (fixes aria-first exports).",
    )
    args = parser.parse_args()

    with open(args.gt_file) as f:
        gt = json.load(f)
    with open(args.pred_file) as f:
        preds = json.load(f)

    if args.normalize_exoego_cam_keys:
        preds = normalize_preds_exoego_cam_keys(preds, args.task_key)

    results = evaluate_scenarios(
        gt,
        preds,
        task_key=args.task_key,
        existence_conf_thresh=args.existence_conf_thresh,
        pred_exists_min_area_frac=args.pred_exists_min_area_frac,
    )

    meta = results.pop("_meta", {})
    for scen in GROUP_PATTERNS:
        r = results[scen]
        print(scen, r)

    print("UNMAPPED_OBJECT_COUNT:", meta.get("unmapped_count"))
    if args.out_json:
        payload = {
            "meta": meta,
            "results": results,
            "task_key": args.task_key,
            "normalize_exoego_cam_keys": bool(args.normalize_exoego_cam_keys),
        }
        with open(args.out_json, "w") as f:
            json.dump(payload, f, indent=2)
        print("wrote", args.out_json)


if __name__ == "__main__":
    main()
