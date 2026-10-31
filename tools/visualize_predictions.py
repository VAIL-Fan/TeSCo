"""
Visualize GT vs Predicted masks for EgoExo4D ego→exo correspondence.

For the **text-conditioned** LM-EEC model, each take can include
`object_descriptions_ablation.json` (same as training). When present, the object
description used for RoBERTa is shown in panel titles.

For each selected video:
  - Picks up to 3 annotated frames
  - Shows: Ego frame | Exo + GT mask | Exo + Pred mask
  - Saves one PNG per video to --output_dir
"""

import argparse
import json
import os
import random
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pycocotools import mask as mask_utils


# ── RLE helpers ────────────────────────────────────────────────────────────────

def decode_rle(rle_dict, target_h, target_w):
    """Decode a pycocotools-style RLE dict to a uint8 binary mask."""
    try:
        m = mask_utils.decode(rle_dict)          # H x W uint8
    except Exception:
        return np.zeros((target_h, target_w), dtype=np.uint8)
    if m.shape != (target_h, target_w):
        m = cv2.resize(m.astype(np.uint8), (target_w, target_h),
                       interpolation=cv2.INTER_NEAREST)
    return m.astype(np.uint8)


def overlay_mask(img_rgb, mask, color, alpha=0.45):
    """Return a copy of img_rgb with a semi-transparent coloured mask overlay."""
    out = img_rgb.copy().astype(float)
    for c, v in enumerate(color):
        out[:, :, c] = np.where(mask, out[:, :, c] * (1 - alpha) + v * alpha, out[:, :, c])
    return np.clip(out, 0, 255).astype(np.uint8)


# ── Camera name helpers ────────────────────────────────────────────────────────

def parse_cam_key(cam_key):
    """
    cam_key format: '<ego_cam>_<exo_cam>'  e.g. 'aria01_214-1_cam03'
    ego cameras start with 'aria', exo cameras start with 'cam'/'gp'/etc.
    Split on last '_' that gives a known exo-cam prefix.
    """
    parts = cam_key.rsplit("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1]   # ego, exo
    return cam_key, cam_key


def find_image(video_dir, cam_folder, frame_id):
    """Try <frame_id>.jpg; return None if missing."""
    path = os.path.join(video_dir, cam_folder, f"{frame_id}.jpg")
    return path if os.path.exists(path) else None


def load_object_descriptions_map(video_dir):
    """Return { object_name: description } from take folder, or {}."""
    path = os.path.join(video_dir, "object_descriptions_ablation.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


# ── Per-video visualisation ────────────────────────────────────────────────────

DISPLAY_H = 480          # resize all images to this height for display


def resize_to_h(img, h):
    ratio = h / img.shape[0]
    w = int(img.shape[1] * ratio)
    return cv2.resize(img, (w, h))


def _short_text(s, max_len=120):
    s = (s or "").strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def visualize_video(vid, pred_masks, gt_masks, video_dir, output_path, show_object_text=True):
    """
    pred_masks / gt_masks: dict[obj_name][cam_key][frame_id] -> {'pred_mask': rle, ...}
    """
    desc_map = load_object_descriptions_map(video_dir) if show_object_text else {}
    # Collect (obj, cam_key, frame_id) triples that appear in BOTH pred and GT
    candidates = []
    for obj in pred_masks:
        if obj not in gt_masks:
            continue
        for cam_key in pred_masks[obj]:
            if cam_key not in gt_masks[obj]:
                continue
            shared_frames = sorted(
                set(pred_masks[obj][cam_key]) & set(gt_masks[obj][cam_key]),
                key=lambda x: int(x)
            )
            for fid in shared_frames:
                candidates.append((obj, cam_key, fid))

    if not candidates:
        print(f"  [skip] No overlapping frames for {vid}")
        return False

    # Pick up to 3 evenly-spaced frames
    step = max(1, len(candidates) // 3)
    chosen = candidates[::step][:3]

    n_frames = len(chosen)
    fig, axes = plt.subplots(n_frames, 3,
                             figsize=(18, 6 * n_frames),
                             squeeze=False)

    fig.suptitle(
        f"Video: {vid}\n(Text-conditioned model — descriptions from object_descriptions_ablation.json when available)",
        fontsize=10,
        fontweight="bold",
        y=1.02,
    )

    GT_COLOR   = (0,   220,  50)   # green
    PRED_COLOR = (255, 80,   30)   # orange-red

    for row, (obj, cam_key, fid) in enumerate(chosen):
        ego_cam, exo_cam = parse_cam_key(cam_key)
        obj_desc = desc_map.get(obj, "")
        if not obj_desc:
            for k, v in desc_map.items():
                if k in obj or obj in k:
                    obj_desc = v
                    break
        desc_line = _short_text(obj_desc, 100) if obj_desc else "(no text in object_descriptions_ablation.json — inference uses zero text like empty training rows)"

        # ── load images ────────────────────────────────────────────────────────
        ego_path = find_image(video_dir, ego_cam, fid)
        exo_path = find_image(video_dir, exo_cam, fid)

        if ego_path is None or exo_path is None:
            # try frame_id as integer without leading zeros
            ego_path = find_image(video_dir, ego_cam, str(int(fid)))
            exo_path = find_image(video_dir, exo_cam, str(int(fid)))

        if ego_path is None:
            ego_img = np.zeros((DISPLAY_H, DISPLAY_H, 3), dtype=np.uint8)
        else:
            ego_img = cv2.cvtColor(cv2.imread(ego_path), cv2.COLOR_BGR2RGB)

        if exo_path is None:
            exo_img = np.zeros((DISPLAY_H, DISPLAY_H, 3), dtype=np.uint8)
        else:
            exo_img = cv2.cvtColor(cv2.imread(exo_path), cv2.COLOR_BGR2RGB)

        H, W = exo_img.shape[:2]

        # ── decode masks ───────────────────────────────────────────────────────
        gt_rle   = gt_masks[obj][cam_key][fid].get("pred_mask", {})
        pred_rle = pred_masks[obj][cam_key][fid].get("pred_mask", {})

        gt_mask   = decode_rle(gt_rle,   H, W)
        pred_mask = decode_rle(pred_rle, H, W)

        # ── overlays ───────────────────────────────────────────────────────────
        exo_gt   = overlay_mask(exo_img, gt_mask,   GT_COLOR)
        exo_pred = overlay_mask(exo_img, pred_mask, PRED_COLOR)

        # resize for display
        ego_disp      = resize_to_h(ego_img,  DISPLAY_H)
        exo_gt_disp   = resize_to_h(exo_gt,   DISPLAY_H)
        exo_pred_disp = resize_to_h(exo_pred, DISPLAY_H)

        # ── plot ───────────────────────────────────────────────────────────────
        axes[row][0].imshow(ego_disp)
        axes[row][0].set_title(
            f"Ego cam ({ego_cam})\nframe {fid}\nobj: {obj}\nText: {desc_line}",
            fontsize=8,
        )

        axes[row][1].imshow(exo_gt_disp)
        gt_patch = mpatches.Patch(color=np.array(GT_COLOR)/255, label="GT mask")
        axes[row][1].legend(handles=[gt_patch], loc="lower right", fontsize=8)
        axes[row][1].set_title(f"Exo + GT\n{exo_cam}", fontsize=9)

        axes[row][2].imshow(exo_pred_disp)
        pred_patch = mpatches.Patch(color=np.array(PRED_COLOR)/255, label="Pred mask")
        axes[row][2].legend(handles=[pred_patch], loc="lower right", fontsize=8)
        axes[row][2].set_title(f"Exo + Pred\n{exo_cam}", fontsize=9)

        for ax in axes[row]:
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {output_path}")
    return True


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_file",   default="data/EgoExo4D/gt_results/egoexo_test_gt.json")
    parser.add_argument("--pred_file", default="output/egoexo_test_results.json")
    parser.add_argument("--image_dir", default="data/EgoExo4D/test")
    parser.add_argument("--output_dir", default="output/visualizations")
    parser.add_argument("--num_videos", type=int, default=20)
    parser.add_argument(
        "--video_list_file",
        type=str,
        default=None,
        help="Optional text file with one take UUID per line; only these videos are visualized (must appear in both GT and pred).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no_show_object_text",
        action="store_true",
        help="Do not load object_descriptions_ablation.json for subtitles",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading GT ...")
    with open(args.gt_file)   as f: gt   = json.load(f)
    print("Loading Pred ...")
    with open(args.pred_file) as f: pred = json.load(f)

    gt_results   = gt["ego-exo"]["results"]
    pred_results = pred["ego-exo"]["results"]

    common_vids = sorted(set(gt_results) & set(pred_results))
    print(f"Common videos: {len(common_vids)}")

    if args.video_list_file:
        with open(args.video_list_file, "r") as f:
            requested = [ln.strip() for ln in f.readlines() if ln.strip()]
        selected = [v for v in requested if v in common_vids]
        missing = [v for v in requested if v not in common_vids]
        for v in missing:
            print(f"  [warn] video not in GT∩pred, skip: {v}")
    else:
        random.seed(args.seed)
        selected = random.sample(common_vids, min(args.num_videos, len(common_vids)))

    for i, vid in enumerate(selected):
        print(f"\n[{i+1}/{len(selected)}] {vid}")
        video_dir = os.path.join(args.image_dir, vid)
        if not os.path.isdir(video_dir):
            print(f"  [skip] image dir not found: {video_dir}")
            continue

        pred_masks = pred_results[vid]["masks"]
        gt_masks   = gt_results[vid]["masks"]

        output_path = os.path.join(args.output_dir, f"{i+1:02d}_{vid[:8]}.png")
        visualize_video(
            vid,
            pred_masks,
            gt_masks,
            video_dir,
            output_path,
            show_object_text=not args.no_show_object_text,
        )

    print(f"\nDone. Visualisations saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
