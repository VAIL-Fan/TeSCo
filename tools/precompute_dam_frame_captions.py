#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Precompute Describe Anything (image DAM) captions for EgoExo4D clips (ego and/or exo camera).

**Ego (default):** {dataset_root}/{take_id}/{captions_subdir}/{object_safe}.json
**Exo (optional `--include_exo`):** {dataset_root}/{take_id}/{captions_subdir_exo}/{object_safe}.json
Same JSON schema (`frames` with concise/detail); use `camera_view` in `_meta` to distinguish.

Per masked frame, DAM runs **two separate generations** on the **same** RGB + mask:
  1) `get_description(image, mask, CONCISE_QUERY)` — short one-sentence caption
  2) `get_description(image, mask, DETAIL_QUERY)` — longer one–two sentence caption

The **text prompts are different** (`CONCISE_QUERY_TEMPLATE` vs `DETAIL_QUERY_TEMPLATE`). Outputs are
post-processed with `finalize_concise_caption` / `finalize_detail_caption`. Optional files
`--concise_prompt_path` / `--detail_prompt_path` override those templates (each must contain `<image>`).

**DAM conditioning is only:** RGB + binary union mask + instruction text (ego or exo view). No object name from paths,
no `object_descriptions.json`, no other dataset metadata in the prompt.

No object_descriptions.json. Frames without a usable mask are skipped.

Requires: local clone of describe-anything at describe-anything/describe-anything (or pip install dam from NVlabs).
Also: pip install -r tools/requirements_dam_precompute.txt
(transformers<5, sentencepiece, tiktoken — see INTEGRATION_DAM_PER_FRAME_TEXT.md).

Example:
  python tools/precompute_dam_frame_captions.py \\
    --dataset_root /path/to/EgoExo4D/train \\
    --dam_model_path nvidia/DAM-3B \\
    --device cuda:0

  # Only the first two distinct takes (by dataset index order), then visualize:
  python tools/precompute_dam_frame_captions.py ... --only_first_n_takes 2 --captions_subdir dam_frame_captions_v2
"""

from __future__ import annotations

import argparse
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from PIL import Image

try:
    import fcntl  # Unix: safe concurrent writes to the same JSON
except ImportError:
    fcntl = None

# Repo root (TeSCo)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Vendored NVIDIA Describe Anything (DAM) package -- just the `dam/` module (860K), not the
# full upstream repo with demo assets/.git (~224M). See third_party/describe_anything/README.md.
DAM_PKG = ROOT / "third_party" / "describe_anything"
if DAM_PKG.is_dir() and str(DAM_PKG) not in sys.path:
    sys.path.insert(0, str(DAM_PKG))


def _ensure_transformers_compat() -> None:
    """DAM imports no_init_weights from transformers; some newer versions removed it."""
    try:
        import transformers.modeling_utils as mu
    except ImportError:
        return
    if hasattr(mu, "no_init_weights"):
        return
    from contextlib import contextmanager

    @contextmanager
    def no_init_weights(_enable=True):
        yield

    mu.no_init_weights = no_init_weights


_ensure_transformers_compat()

from training.dataset.vos_raw_dataset import EgoExoRawDataset
# NOTE: this repo's training/inference code (sam2/utils/dam_frame_captions.py) only has a
# *read* API for the dam_frame_captions_v2 format. The *write* API this generator script needs
# (save_dam_frame_captions, finalize_*_caption, etc.) is kept separately in
# tools/dam_caption_writer_utils.py (ported from the snapshot that introduced this generator)
# so it doesn't collide with the read-only module the model actually imports at train/eval time.
# Imported as a flat module (not `tools.dam_caption_writer_utils`) because a `tools` package
# from an unrelated project can shadow this one on some machines' PYTHONPATH.
_TOOLS_DIR = str(Path(__file__).resolve().parent)
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)
from dam_caption_writer_utils import (
    dam_frame_caption_json_path,
    finalize_concise_caption,
    finalize_detail_caption,
    frame_caption_entry_complete,
    load_dam_frame_captions,
    save_dam_frame_captions,
)


def _egoexo_take_id_from_clip_video_name(video_name: str) -> str | None:
    """Match EgoExoRawDataset.get_video: take_id is the first of the last four path segments."""
    parts = video_name.split("/")
    if len(parts) < 4:
        return None
    return parts[-4]


def _first_n_distinct_take_ids(ds: EgoExoRawDataset, n: int) -> set[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for j in range(len(ds)):
        tid = _egoexo_take_id_from_clip_video_name(ds.video_names[j])
        if tid is None:
            continue
        if tid not in seen:
            seen.add(tid)
            ordered.append(tid)
            if len(ordered) >= n:
                break
    return set(ordered)


def _read_take_ids_from_text_file(path: str) -> set[str]:
    """Read take IDs (one per line); ignore empty lines and # comments."""
    take_ids: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            take_ids.add(line)
    return take_ids


# Two different user-query strings → two DAM forwards per frame (never share one prompt for both).
# Placeholders: {concise_max_words} {detail_max_words} via _fill_prompt.
# Legacy custom files may still contain {object_label}; it is stripped (empty) — do not use dataset names in prompts.
CONCISE_QUERY_TEMPLATE = (
    "<image>\n"
    "The user highlighted exactly one region with a mask in this egocentric image.\n"
    "Write exactly ONE complete English sentence describing ONLY what is inside the mask, at most {concise_max_words} words "
    "(fewer than 10). Include: (1) main color(s), (2) what the object or substance is, (3) where it sits in the frame. "
    "Infer identity from pixels only. One sentence only; end with a period. No quotes or preamble.\n"
)

DETAIL_QUERY_TEMPLATE = (
    "<image>\n"
    "The user highlighted exactly one region with a mask in this egocentric image.\n"
    "Write one or two complete English sentences describing ONLY the masked region, at most {detail_max_words} words total. "
    "Cover: (1) colors and materials, (2) what it is (from appearance), (3) location, pose, and visibility. "
    "Infer from pixels only — you are not given a class name. Each sentence must be complete; end with periods. "
    "Add richer visual detail than a one-line summary would. Do not describe background outside the mask.\n"
)

# Exocentric (fixed third-person) camera — same JSON format as ego; different wording only.
CONCISE_QUERY_TEMPLATE_EXO = (
    "<image>\n"
    "The user highlighted exactly one region with a mask in this exocentric (third-person) camera image.\n"
    "Write exactly ONE complete English sentence describing ONLY what is inside the mask, at most {concise_max_words} words "
    "(fewer than 10). Include: (1) main color(s), (2) what the object or substance is, (3) where it sits in the frame. "
    "Infer identity from pixels only. One sentence only; end with a period. No quotes or preamble.\n"
)

DETAIL_QUERY_TEMPLATE_EXO = (
    "<image>\n"
    "The user highlighted exactly one region with a mask in this exocentric (third-person) camera image.\n"
    "Write one or two complete English sentences describing ONLY the masked region, at most {detail_max_words} words total. "
    "Cover: (1) colors and materials, (2) what it is (from appearance), (3) location, pose, and visibility. "
    "Infer from pixels only — you are not given a class name. Each sentence must be complete; end with periods. "
    "Add richer visual detail than a one-line summary would. Do not describe background outside the mask.\n"
)


@contextmanager
def _file_lock(lock_path: str):
    """Exclusive lock so multiple GPU workers can merge into the same caption JSON."""
    if fcntl is None:
        yield
        return
    os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
    lf = open(lock_path, "w")
    try:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
        lf.close()


def _union_mask_from_segments(segments: dict) -> np.ndarray | None:
    """Binary uint8 mask HxW in {0,255}, or None if empty."""
    if not segments:
        return None
    union = None
    for seg in segments.values():
        if seg is None:
            continue
        m = seg
        if not isinstance(m, torch.Tensor):
            m = torch.as_tensor(m)
        m = m.bool()
        if union is None:
            union = m.clone()
        else:
            union |= m
    if union is None or not union.any():
        return None
    return (union.cpu().numpy().astype(np.uint8) * 255)


def _nearest_resample():
    try:
        return Image.Resampling.NEAREST
    except AttributeError:
        return Image.NEAREST


def _resize_mask_to_match_image(mask_pil: Image.Image, pil_img: Image.Image) -> Image.Image:
    if mask_pil.size == pil_img.size:
        return mask_pil
    return mask_pil.resize(pil_img.size, resample=_nearest_resample())


def _fill_prompt(template: str, concise_max_words: int, detail_max_words: int) -> str:
    """Substitute word limits; remove legacy {object_label} so no dataset string leaks into DAM."""
    return (
        template.replace("{concise_max_words}", str(concise_max_words))
        .replace("{detail_max_words}", str(detail_max_words))
        .replace("{object_label}", "")
    )


def _load_prompt_template(path: str | None, default: str) -> str:
    if not path:
        return default
    with open(path, encoding="utf-8") as f:
        t = f.read()
    if "<image>" not in t:
        raise SystemExit(f"Prompt file must contain '<image>': {path}")
    return t


def _run_dam_concise_and_detail(
    dam,
    pil_img: Image.Image,
    mask_pil: Image.Image,
    q_concise: str,
    q_detail: str,
    args: argparse.Namespace,
) -> tuple[str, str]:
    """
    Two independent DAM calls: same pixels, **different** text prompts (q_concise ≠ q_detail).
    """
    with torch.inference_mode():
        raw_c = dam.get_description(
            pil_img,
            mask_pil,
            q_concise,
            streaming=False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=1,
            max_new_tokens=args.max_new_tokens_concise,
        )
        raw_d = dam.get_description(
            pil_img,
            mask_pil,
            q_detail,
            streaming=False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=1,
            max_new_tokens=args.max_new_tokens_detail,
        )
    text_c = finalize_concise_caption((raw_c or "").strip(), args.concise_max_words)
    text_d = finalize_detail_caption((raw_d or "").strip(), args.detail_max_words)
    return text_c, text_d


def _frame_has_valid_mask_for_dam(segment_loader, frame_idx: int, img_path: str) -> bool:
    """True if JPEG exists and decoded+resized mask has at least one foreground pixel."""
    if not os.path.isfile(img_path):
        return False
    segments = segment_loader.load(frame_idx)
    mask_np = _union_mask_from_segments(segments)
    if mask_np is None:
        return False
    pil_img = Image.open(img_path).convert("RGB")
    mask_pil = _resize_mask_to_match_image(Image.fromarray(mask_np), pil_img)
    return bool((np.asarray(mask_pil) > 0).any())


def _clip_fully_cached_for_view(
    args: argparse.Namespace,
    frames: list,
    segment_loader,
    captions: dict,
) -> bool:
    """True if every frame with a valid mask already has concise+detail in captions."""
    for frame in frames:
        fid = str(int(frame.frame_idx))
        img_path = frame.image_path
        if not _frame_has_valid_mask_for_dam(segment_loader, frame.frame_idx, img_path):
            continue
        if not frame_caption_entry_complete(captions.get(fid)):
            return False
    return True


def _process_clip_dam_view(
    dam,
    args: argparse.Namespace,
    *,
    view_tag: str,
    frames: list,
    segment_loader,
    take_id: str,
    object_name: str,
    idx: int,
    start_idx: int,
    end: int,
    captions_subdir: str,
    concise_tpl: str,
    detail_tpl: str,
    meta: dict,
) -> None:
    """Run DAM for one camera (ego or exo); merge into JSON under captions_subdir."""
    out_path = dam_frame_caption_json_path(
        args.dataset_root, take_id, object_name, captions_subdir=captions_subdir
    )
    captions = load_dam_frame_captions(
        args.dataset_root,
        take_id,
        object_name,
        captions_subdir=captions_subdir,
        concise_max_words=args.concise_max_words,
        detail_max_words=args.detail_max_words,
    )

    if args.skip_existing and _clip_fully_cached_for_view(
        args, frames, segment_loader, captions
    ):
        print(
            f"Skip [{idx}] {take_id}/{object_name} [{view_tag}] (all masked frames cached)"
        )
        return

    q_concise = _fill_prompt(concise_tpl, args.concise_max_words, args.detail_max_words)
    q_detail = _fill_prompt(detail_tpl, args.concise_max_words, args.detail_max_words)

    updates: dict[str, dict[str, str]] = {}
    for frame in frames:
        fid = str(int(frame.frame_idx))
        if args.skip_existing and frame_caption_entry_complete(captions.get(fid)):
            continue

        img_path = frame.image_path
        if not os.path.isfile(img_path):
            print(
                f"  [{view_tag}] Missing image {img_path}, skip frame {fid} (no DAM)"
            )
            continue

        segments = segment_loader.load(frame.frame_idx)
        mask_np = _union_mask_from_segments(segments)
        if mask_np is None:
            print(
                f"  [{view_tag}] No mask take={take_id} obj={object_name} frame={fid}, skip (no JSON entry)"
            )
            continue

        pil_img = Image.open(img_path).convert("RGB")
        mask_pil = _resize_mask_to_match_image(Image.fromarray(mask_np), pil_img)

        if not (np.asarray(mask_pil) > 0).any():
            print(
                f"  [{view_tag}] Empty foreground take={take_id} obj={object_name} frame={fid}, skip (no JSON entry)"
            )
            continue

        text_concise, text_detail = _run_dam_concise_and_detail(
            dam, pil_img, mask_pil, q_concise, q_detail, args
        )
        if not text_concise or not text_detail:
            print(
                f"  [{view_tag}] Empty DAM output take={take_id} obj={object_name} frame={fid}, skip (no JSON entry)"
            )
            continue
        updates[fid] = {"concise": text_concise, "detail": text_detail}
        d_prev = text_detail if len(text_detail) <= 100 else text_detail[:100] + "..."
        print(
            f"  [{idx}/{end - start_idx}] [{view_tag}] {take_id}/{object_name} f={fid} :: "
            f"c={text_concise} | d={d_prev}"
        )

        if args.write_every_frame:
            lock_p = out_path + ".lock"
            with _file_lock(lock_p):
                merged = load_dam_frame_captions(
                    args.dataset_root,
                    take_id,
                    object_name,
                    captions_subdir=captions_subdir,
                    concise_max_words=args.concise_max_words,
                    detail_max_words=args.detail_max_words,
                )
                merged.update(updates)
                save_dam_frame_captions(
                    args.dataset_root,
                    take_id,
                    object_name,
                    merged,
                    meta=meta,
                    captions_subdir=captions_subdir,
                    concise_max_words=args.concise_max_words,
                    detail_max_words=args.detail_max_words,
                )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not updates:
        return

    if not args.write_every_frame:
        lock_p = out_path + ".lock"
        with _file_lock(lock_p):
            merged = load_dam_frame_captions(
                args.dataset_root,
                take_id,
                object_name,
                captions_subdir=captions_subdir,
                concise_max_words=args.concise_max_words,
                detail_max_words=args.detail_max_words,
            )
            merged.update(updates)
            save_dam_frame_captions(
                args.dataset_root,
                take_id,
                object_name,
                merged,
                meta=meta,
                captions_subdir=captions_subdir,
                concise_max_words=args.concise_max_words,
                detail_max_words=args.detail_max_words,
            )
    print(f"Merged {len(updates)} frame(s) [{view_tag}] -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="EgoExo img_folder (contains take ids with annotation.json)",
    )
    parser.add_argument(
        "--dam_model_path",
        type=str,
        default="nvidia/DAM-3B",
        help="Hugging Face id or local path for image DAM",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--num_frames",
        type=int,
        default=8,
        help="Min overlapping frames per clip (must match training EgoExoRawDataset)",
    )
    parser.add_argument("--conv_mode", type=str, default="v1")
    parser.add_argument(
        "--concise_max_words",
        type=int,
        default=9,
        help="Hard cap on concise caption (default 9 = strictly fewer than 10 words); enforced after DAM.",
    )
    parser.add_argument(
        "--detail_max_words",
        type=int,
        default=99,
        help="Hard cap on detail caption length (< 100 words => default 99).",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument(
        "--max_new_tokens_concise",
        type=int,
        default=40,
        help="DAM max_new_tokens for the concise pass (keep low to reduce rambling).",
    )
    parser.add_argument(
        "--max_new_tokens_detail",
        type=int,
        default=384,
        help="DAM max_new_tokens for the detail pass.",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Shard: first dataset index (inclusive)",
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=-1,
        help="Shard: last dataset index (exclusive); -1 = len(dataset)",
    )
    parser.add_argument(
        "--shard_id",
        type=int,
        default=None,
        help="Parallel worker id in [0, num_shards). Sets start/end from dataset length.",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=None,
        help="Number of parallel workers (e.g. 4 for 4 GPUs). Use with --shard_id.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip frame ids that already have a non-empty caption in the JSON",
    )
    parser.add_argument(
        "--write_every_frame",
        action="store_true",
        help="Save JSON after each new caption (slower, safer for crashes)",
    )
    parser.add_argument(
        "--captions_subdir",
        type=str,
        default="dam_frame_captions_v2",
        help=(
            "Output directory name under each take (default: dam_frame_captions_v2, separate from "
            "legacy dam_frame_captions). Set dataset.dam_captions_subdir in training YAML to the same value."
        ),
    )
    parser.add_argument(
        "--only_first_n_takes",
        type=int,
        default=None,
        help=(
            "If set, only process clips whose take_id is among the first N distinct takes "
            "(by ascending dataset index). Skips get_video for other clips. Prefer running "
            "without --shard_id so those takes are fully covered."
        ),
    )
    parser.add_argument(
        "--take_ids_path",
        type=str,
        default=None,
        help=(
            "Optional text file containing explicit take IDs (one per line). "
            "Empty lines and lines starting with '#' are ignored."
        ),
    )
    parser.add_argument(
        "--concise_prompt_path",
        type=str,
        default=None,
        help=(
            "Optional UTF-8 text file: full user prompt for the **concise** DAM pass only "
            "(must contain '<image>'; placeholders: {concise_max_words} {detail_max_words})."
        ),
    )
    parser.add_argument(
        "--detail_prompt_path",
        type=str,
        default=None,
        help=(
            "Optional UTF-8 text file: full user prompt for the **detail** DAM pass only "
            "(must contain '<image>'; placeholders: {concise_max_words} {detail_max_words})."
        ),
    )
    parser.add_argument(
        "--include_exo",
        action="store_true",
        help=(
            "Also run DAM on each exocentric RGB frame and union exo mask; write JSON under "
            "--captions_subdir_exo (same schema as ego, separate folder per take)."
        ),
    )
    parser.add_argument(
        "--exo_only",
        action="store_true",
        help="Only exocentric pass (skip ego). Use after ego JSON is complete to save time.",
    )
    parser.add_argument(
        "--captions_subdir_exo",
        type=str,
        default="dam_frame_captions_v2_exo",
        help="Per-take subfolder for exo-view DAM captions (default: dam_frame_captions_v2_exo).",
    )
    parser.add_argument(
        "--concise_prompt_path_exo",
        type=str,
        default=None,
        help="Optional prompt file for exo concise pass (default: built-in exocentric template).",
    )
    parser.add_argument(
        "--detail_prompt_path_exo",
        type=str,
        default=None,
        help="Optional prompt file for exo detail pass (default: built-in exocentric template).",
    )
    args = parser.parse_args()

    if args.exo_only:
        do_ego = False
        do_exo = True
    else:
        do_ego = True
        do_exo = bool(args.include_exo)
    if not do_ego and not do_exo:
        raise SystemExit("Nothing to run: enable ego (default) or pass --include_exo / --exo_only")

    concise_tpl = (
        _load_prompt_template(args.concise_prompt_path, CONCISE_QUERY_TEMPLATE) if do_ego else ""
    )
    detail_tpl = (
        _load_prompt_template(args.detail_prompt_path, DETAIL_QUERY_TEMPLATE) if do_ego else ""
    )
    concise_tpl_exo = (
        _load_prompt_template(args.concise_prompt_path_exo, CONCISE_QUERY_TEMPLATE_EXO)
        if do_exo
        else ""
    )
    detail_tpl_exo = (
        _load_prompt_template(args.detail_prompt_path_exo, DETAIL_QUERY_TEMPLATE_EXO) if do_exo else ""
    )

    try:
        from dam import DescribeAnythingModel, disable_torch_init
    except ImportError as e:
        raise SystemExit(
            "Could not import DescribeAnythingModel. Install DAM or add describe-anything/describe-anything to PYTHONPATH.\n"
            f"  Tried package path: {DAM_PKG}\n"
            f"  Original error: {e}"
        ) from e

    ds = EgoExoRawDataset(
        img_folder=args.dataset_root,
        gt_folder=args.dataset_root,
        num_frames=args.num_frames,
    )
    allowed_takes: set[str] | None = None
    if args.take_ids_path:
        allowed_takes = _read_take_ids_from_text_file(args.take_ids_path)
        print(
            f"--take_ids_path={args.take_ids_path}: loaded {len(allowed_takes)} explicit take ID(s)"
        )
    if args.only_first_n_takes is not None:
        if args.only_first_n_takes < 1:
            raise SystemExit("--only_first_n_takes must be >= 1")
        first_n_takes = _first_n_distinct_take_ids(ds, args.only_first_n_takes)
        if allowed_takes is None:
            allowed_takes = first_n_takes
        else:
            allowed_takes &= first_n_takes
        print(
            f"--only_first_n_takes={args.only_first_n_takes}: allowed take_id set ({len(allowed_takes)}): "
            f"{sorted(allowed_takes)}"
        )
    if allowed_takes is not None:
        print(f"Allowed take IDs count: {len(allowed_takes)}")
    n = len(ds)
    if args.shard_id is not None:
        if args.num_shards is None or args.num_shards < 1:
            raise SystemExit("--num_shards is required and must be >= 1 when --shard_id is set")
        if not (0 <= args.shard_id < args.num_shards):
            raise SystemExit(f"--shard_id must be in [0, {args.num_shards})")
        chunk = (n + args.num_shards - 1) // args.num_shards
        args.start_idx = args.shard_id * chunk
        args.end_idx = min(args.start_idx + chunk, n)
    end = args.end_idx if args.end_idx >= 0 else n
    end = min(end, n)
    print(
        f"Dataset clips: {n}; processing indices [{args.start_idx}, {end})"
        + (
            f" (shard {args.shard_id}/{args.num_shards})"
            if args.shard_id is not None
            else ""
        )
    )
    _vinfo = []
    if do_ego:
        _vinfo.append("ego -> " + args.captions_subdir)
    if do_exo:
        _vinfo.append("exo -> " + args.captions_subdir_exo)
    print("Views: " + " | ".join(_vinfo))

    disable_torch_init()
    # DAM expects "first_view+second_view" (e.g. full+focal_crop), not the demo alias "focal_prompt".
    dam = DescribeAnythingModel(
        model_path=args.dam_model_path,
        conv_mode=args.conv_mode,
        prompt_mode="full+focal_crop",
    ).to(args.device)
    dam.eval()

    sub = args.captions_subdir
    meta_base: dict = {
        "dam_model": args.dam_model_path,
        "no_object_description_fallback": True,
        "caption_format": "concise_detail",
        "concise_max_words": args.concise_max_words,
        "detail_max_words": args.detail_max_words,
        "dam_two_prompts": True,
        "dam_prompt_uses_only_image_mask_and_instruction": True,
    }
    if args.concise_prompt_path:
        meta_base["concise_prompt_path"] = args.concise_prompt_path
    if args.detail_prompt_path:
        meta_base["detail_prompt_path"] = args.detail_prompt_path
    if args.concise_prompt_path_exo:
        meta_base["concise_prompt_path_exo"] = args.concise_prompt_path_exo
    if args.detail_prompt_path_exo:
        meta_base["detail_prompt_path_exo"] = args.detail_prompt_path_exo

    for idx in range(args.start_idx, end):
        if allowed_takes is not None:
            tid0 = _egoexo_take_id_from_clip_video_name(ds.video_names[idx])
            if tid0 is None or tid0 not in allowed_takes:
                continue
        (
            ego_video,
            exo_video,
            ego_segment_loader,
            exo_segment_loader,
            _global_description,
            _fc,
        ) = ds.get_video(idx)
        video_name = ego_video.video_name
        parts = video_name.split("/")
        if len(parts) < 4:
            print(f"Skip bad video_name: {video_name}")
            continue
        take_id, ego_cam_name, _cam_name, object_name = parts[-4:]

        if do_ego:
            meta_ego = {
                **meta_base,
                "captions_subdir": sub,
                "camera_view": "ego",
            }
            _process_clip_dam_view(
                dam,
                args,
                view_tag="ego",
                frames=ego_video.frames,
                segment_loader=ego_segment_loader,
                take_id=take_id,
                object_name=object_name,
                idx=idx,
                start_idx=args.start_idx,
                end=end,
                captions_subdir=sub,
                concise_tpl=concise_tpl,
                detail_tpl=detail_tpl,
                meta=meta_ego,
            )
        if do_exo:
            meta_exo = {
                **meta_base,
                "captions_subdir": args.captions_subdir_exo,
                "camera_view": "exo",
            }
            _process_clip_dam_view(
                dam,
                args,
                view_tag="exo",
                frames=exo_video.frames,
                segment_loader=exo_segment_loader,
                take_id=take_id,
                object_name=object_name,
                idx=idx,
                start_idx=args.start_idx,
                end=end,
                captions_subdir=args.captions_subdir_exo,
                concise_tpl=concise_tpl_exo,
                detail_tpl=detail_tpl_exo,
                meta=meta_exo,
            )

    print("Done.")


if __name__ == "__main__":
    main()
