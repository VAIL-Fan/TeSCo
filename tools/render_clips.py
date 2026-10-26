"""Render side-by-side correspondence clips for the project page.

Each clip shows what the method is actually given and what it returns: the
query view with the object marked, the search view with the predicted mask, and
underneath, the sentence the model was conditioned on. Ground truth is drawn as
a contour rather than a fill so the prediction stays readable against it.

Predictions are per-frame RLE JSON as correspondence_inference.py writes them:
  <pred_root>/<take>/<cam_a>__<cam_b>/<object>/<frame>.json
"""
import argparse, json, os, re, subprocess, sys
import cv2
import numpy as np
from pycocotools import mask as mask_utils

PRED_RGB = (64, 220, 255)    # cyan   - prediction
SRC_RGB  = (120, 255, 150)   # green  - the mask given in the query view
GT_RGB   = (255, 170, 60)    # amber  - ground-truth contour


def decode(rle):
    """Accept either a bare RLE or one wrapped in {"pred_mask": ...}.

    The two stages of this pipeline disagree: correspondence_inference.py writes
    per-frame files that ARE the RLE, while merge_pred.py and the GT files wrap
    it under "pred_mask". Reading only the wrapped form silently yields None and
    draws nothing -- which looks like a model that predicts nothing.
    """
    if not rle:
        return None
    if "counts" not in rle:
        rle = rle.get("pred_mask")
        if not rle:
            return None
    c = rle["counts"]
    return mask_utils.decode({"size": rle["size"],
                              "counts": c.encode("utf-8") if isinstance(c, str) else c})


def overlay(img, mask, rgb, alpha=0.45, contour=False, thickness=3):
    if mask is None or not mask.any():
        return img
    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(mask.astype(np.uint8), (img.shape[1], img.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    bgr = rgb[::-1]
    if contour:
        cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cs, -1, bgr, thickness, lineType=cv2.LINE_AA)
        return img
    tint = np.zeros_like(img); tint[:] = bgr
    m = mask.astype(bool)
    img[m] = (img[m] * (1 - alpha) + tint[m] * alpha).astype(np.uint8)
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cs, -1, bgr, 2, lineType=cv2.LINE_AA)
    return img


def inset(img, masks, rgbs, frac=0.34, pad=0.9):
    """Magnify the region the masks occupy and pin it to the panel's corner.

    Targets in the exo view are often under half a percent of the frame -- about
    25x25 px once scaled for the page -- so the result is correct but invisible.
    The inset makes the comparison legible without cropping the context away.
    """
    present = [m for m in masks if m is not None and m.any()]
    if not present:
        return img
    H, W = img.shape[:2]
    ys, xs = [], []
    for m in present:
        if m.shape[:2] != (H, W):
            m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        y, x = np.where(m > 0)
        ys.append(y); xs.append(x)
    y0, y1 = int(np.concatenate(ys).min()), int(np.concatenate(ys).max())
    x0, x1 = int(np.concatenate(xs).min()), int(np.concatenate(xs).max())
    cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
    half = max((y1 - y0), (x1 - x0)) * (1 + pad) / 2
    half = max(half, 26)
    y0, y1 = int(max(0, cy - half)), int(min(H, cy + half))
    x0, x1 = int(max(0, cx - half)), int(min(W, cx + half))
    if y1 - y0 < 8 or x1 - x0 < 8:
        return img
    side = int(min(H, W) * frac)
    crop = cv2.resize(img[y0:y1, x0:x1], (side, side), interpolation=cv2.INTER_CUBIC)
    cv2.rectangle(crop, (0, 0), (side - 1, side - 1), (230, 230, 230), 2)
    my, mx = 10, W - side - 10
    img[my:my + side, mx:mx + side] = crop
    # tie the inset back to where it came from
    cv2.rectangle(img, (x0, y0), (x1, y1), (230, 230, 230), 1, cv2.LINE_AA)
    return img


def label(img, text, rgb, origin=(14, 34), scale=0.8):
    cv2.putText(img, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, rgb[::-1], 2, cv2.LINE_AA)
    return img


def wrap(text, width):
    out, line = [], ""
    for w in text.split():
        if len(line) + len(w) + 1 <= width:
            line = f"{line} {w}".strip()
        else:
            out.append(line); line = w
    if line:
        out.append(line)
    return out[:2]


def caption_bar(w, lines, height=74):
    bar = np.full((height, w, 3), 18, np.uint8)
    for i, ln in enumerate(lines):
        cv2.putText(bar, ln, (18, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.68, (235, 235, 235), 2, cv2.LINE_AA)
    return bar


def load_captions(root, take, subdir, obj):
    """{"frames": {"<frame>": {"concise": ..., "detail": ...}}, "_meta": ...}

    Captions are per frame, not per take -- the paper refreshes the description
    every frame -- so this returns the whole map and the caller picks by frame.
    `_meta` holds the generating model's name, not a caption; walking values
    blindly picks it up instead.
    """
    for name in (f"{obj}.json", f"{obj.replace('/', '-')}.json"):
        p = os.path.join(root, take, subdir, name)
        if os.path.isfile(p):
            try:
                frames = json.load(open(p)).get("frames", {})
            except Exception:
                return {}
            return {k: (v.get("concise") or v.get("detail") or "").strip()
                    for k, v in frames.items() if isinstance(v, dict)}
    return {}


def source_masks(root, take, obj, cam):
    """Source-view masks live in the take's annotation.json, not in the GT file:
    the correspondence GT only carries the target view, keyed by camera pair."""
    p = os.path.join(root, take, "annotation.json")
    if not os.path.isfile(p):
        return {}
    try:
        return json.load(open(p)).get("masks", {}).get(obj, {}).get(cam, {})
    except Exception:
        return {}


def build(args):
    gt_all = json.load(open(args.gt))[args.task]["results"]
    src_cam_is_ego = args.task == "ego-exo"
    made = []
    for take in sorted(os.listdir(args.pred_root)):
        tdir = os.path.join(args.pred_root, take)
        if not os.path.isdir(tdir) or take not in gt_all:
            continue
        for pair in sorted(os.listdir(tdir)):
            pdir = os.path.join(tdir, pair)
            if not os.path.isdir(pdir) or "__" not in pair:
                continue
            a, b = pair.split("__", 1)
            ego, exo = (a, b) if "aria" in a else (b, a)
            src_cam, tgt_cam = (ego, exo) if src_cam_is_ego else (exo, ego)
            for obj in sorted(os.listdir(pdir)):
                odir = os.path.join(pdir, obj)
                if not os.path.isdir(odir):
                    continue
                frames = sorted((f[:-5] for f in os.listdir(odir) if f.endswith(".json")),
                                key=lambda x: int(x) if x.isdigit() else 0)
                if len(frames) < args.min_frames:
                    continue
                step = max(1, len(frames) // args.max_frames)
                frames = frames[::step][:args.max_frames]
                out = os.path.join(args.out, f"{take[:8]}_{re.sub(r'[^A-Za-z0-9]+','-',obj)}.mp4")
                if render(args, take, src_cam, tgt_cam, pair, obj, frames, gt_all[take], out):
                    made.append(out)
                    print(f"  {os.path.basename(out)}  ({len(frames)} frames)", flush=True)
                if len(made) >= args.limit:
                    return made
    return made


def render(args, take, src_cam, tgt_cam, pair, obj, frames, gt_take, out):
    cap_sub = "dam_frame_captions_v2" if args.task == "ego-exo" else "dam_frame_captions_v2_exo"
    caps = load_captions(args.frames_root, take, cap_sub, obj)
    src_rles = source_masks(args.frames_root, take, obj, src_cam)
    gt_frames = gt_take.get("masks", {}).get(obj, {}).get(pair.replace("__", "_"), {})
    if not gt_frames:
        for k, v in gt_take.get("masks", {}).get(obj, {}).items():
            if tgt_cam in k:
                gt_frames = v
                break

    writer, size, written = None, None, 0
    for f in frames:
        pa = os.path.join(args.frames_root, take, src_cam, f"{f}.jpg")
        pb = os.path.join(args.frames_root, take, tgt_cam, f"{f}.jpg")
        if not (os.path.isfile(pa) and os.path.isfile(pb)):
            continue
        ia, ib = cv2.imread(pa), cv2.imread(pb)
        if ia is None or ib is None:
            continue
        try:
            pred = decode(json.load(open(os.path.join(
                args.pred_root, take, pair, obj, f"{f}.json"))))
        except Exception:
            pred = None
        src_m = decode(src_rles.get(f))
        gt_m = decode(gt_frames.get(f))

        h = args.height
        ia = cv2.resize(ia, (int(ia.shape[1] * h / ia.shape[0]), h))
        ib = cv2.resize(ib, (int(ib.shape[1] * h / ib.shape[0]), h))
        overlay(ia, src_m, SRC_RGB)
        overlay(ib, gt_m, GT_RGB, contour=True)
        overlay(ib, pred, PRED_RGB)
        # Label only what is actually on screen: an unconditional label is easy to
        # mistake for evidence that the mask rendered.
        label(ia, "query view" + ("" if src_m is not None and src_m.any() else "  (no mask)"),
              SRC_RGB, scale=0.62)
        y = 34
        if pred is not None and pred.any():
            label(ib, "TeSCo prediction", PRED_RGB, origin=(14, y), scale=0.62); y += 24
        if gt_m is not None and gt_m.any():
            label(ib, "ground truth", GT_RGB, origin=(14, y), scale=0.52)

        inset(ib, [pred, gt_m], [PRED_RGB, GT_RGB])
        gap = np.full((h, 8, 3), 18, np.uint8)
        row = np.hstack([ia, gap, ib])
        text = caps.get(str(f)) or caps.get(str(int(f)) if str(f).isdigit() else "") or obj
        bar = caption_bar(row.shape[1], wrap(text, max(46, row.shape[1] // 12)))
        frame = np.vstack([row, bar])
        if writer is None:
            size = (frame.shape[1], frame.shape[0])
            writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, size)
        if (frame.shape[1], frame.shape[0]) == size:
            writer.write(frame); written += 1
    if writer:
        writer.release()
    if written < args.min_frames:
        if os.path.exists(out):
            os.remove(out)
        return False
    tmp = out + ".h264.mp4"
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", out,
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "26",
                        "-movflags", "+faststart", tmp], capture_output=True)
    if r.returncode == 0 and os.path.getsize(tmp) > 0:
        os.replace(tmp, out)
    elif os.path.exists(tmp):
        os.remove(tmp)
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pred_root", required=True)
    ap.add_argument("--frames_root", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--task", default="exo-ego", choices=["exo-ego", "ego-exo"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--max_frames", type=int, default=90)
    ap.add_argument("--min_frames", type=int, default=12)
    ap.add_argument("--limit", type=int, default=12)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    got = build(a)
    print(f"\nwrote {len(got)} clips to {a.out}")
