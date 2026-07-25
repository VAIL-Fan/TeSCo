# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Paths and I/O for precomputed per-frame DAM captions (ego view)."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Union

FrameCaptionEntry = Dict[str, str]  # keys: "concise", "detail"

# Match precompute defaults: concise strictly < 10 words, detail strictly < 100 words.
DEFAULT_CONCISE_MAX_WORDS = 9
DEFAULT_DETAIL_MAX_WORDS = 99


def cap_caption_words(text: str, max_words: int) -> str:
    """Truncate to the first max_words whitespace-separated tokens (hard cap)."""
    text = (text or "").strip()
    if max_words <= 0:
        return ""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).strip()


def split_into_sentences(text: str) -> List[str]:
    """Split on . ? ! followed by whitespace; if none, one segment."""
    text = (text or "").strip().strip('"').strip("'")
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    out = [p.strip() for p in parts if p.strip()]
    return out if out else [text]


def _ensure_sentence_punct(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    if s[-1] not in ".!?":
        return s + "."
    return s


def finalize_concise_caption(raw: str, max_words: int) -> str:
    """
    One complete sentence, <= max_words words. Uses the model's first sentence when possible;
    does not truncate detail text in place of a real concise line.
    """
    raw = (raw or "").strip().strip('"').strip("'")
    if not raw:
        return ""
    sents = split_into_sentences(raw)
    first = sents[0]
    words = first.split()
    if len(words) > max_words:
        first = " ".join(words[:max_words])
    return _ensure_sentence_punct(first)


def finalize_detail_caption(raw: str, max_words: int) -> str:
    """One or more complete sentences, total word count <= max_words (sentence boundaries preferred)."""
    raw = (raw or "").strip().strip('"').strip("'")
    if not raw:
        return ""
    sents = split_into_sentences(raw)
    picked: List[str] = []
    wc = 0
    for s in sents:
        sw = len(s.split())
        gap = 1 if picked else 0
        if wc + gap + sw <= max_words:
            picked.append(s)
            wc += gap + sw
        else:
            break
    if picked:
        out = " ".join(picked).strip()
        return _ensure_sentence_punct(out)
    # Single run-on without clear splits, or first sentence alone exceeds budget
    blob = sents[0] if sents else raw
    out = cap_caption_words(blob, max_words)
    return _ensure_sentence_punct(out)


def object_name_to_filename_safe(object_name: str) -> str:
    return object_name.replace("/", "-")


def dam_frame_captions_dir(
    dataset_img_folder: str, take_id: str, captions_subdir: str = "dam_frame_captions"
) -> str:
    return os.path.join(dataset_img_folder, take_id, captions_subdir)


def dam_frame_caption_json_path(
    dataset_img_folder: str,
    take_id: str,
    object_name: str,
    captions_subdir: str = "dam_frame_captions",
) -> str:
    safe = object_name_to_filename_safe(object_name)
    return os.path.join(
        dam_frame_captions_dir(dataset_img_folder, take_id, captions_subdir), f"{safe}.json"
    )


def frame_caption_file_exists(
    dataset_img_folder: str,
    take_id: str,
    object_name: str,
    captions_subdir: str = "dam_frame_captions",
) -> bool:
    path = dam_frame_caption_json_path(
        dataset_img_folder, take_id, object_name, captions_subdir=captions_subdir
    )
    return os.path.isfile(path)


def normalize_frame_caption_entry(
    v: Any,
    concise_max_words: int = DEFAULT_CONCISE_MAX_WORDS,
    detail_max_words: int = DEFAULT_DETAIL_MAX_WORDS,
) -> FrameCaptionEntry:
    """Map JSON value to {concise, detail}. Never copies the start of detail into concise."""
    if isinstance(v, dict):
        c_raw = str(v.get("concise", "") or "").strip()
        d_raw = str(v.get("detail", "") or "").strip()
        c = finalize_concise_caption(c_raw, concise_max_words) if c_raw else ""
        d = finalize_detail_caption(d_raw, detail_max_words) if d_raw else ""
        if not d and c:
            d = finalize_detail_caption(c_raw, detail_max_words)
        return {"concise": c, "detail": d}
    s = str(v or "").strip()
    return {
        "concise": finalize_concise_caption(s, concise_max_words),
        "detail": finalize_detail_caption(s, detail_max_words),
    }


def load_dam_frame_captions(
    dataset_img_folder: str,
    take_id: str,
    object_name: str,
    captions_subdir: str = "dam_frame_captions",
    concise_max_words: int = DEFAULT_CONCISE_MAX_WORDS,
    detail_max_words: int = DEFAULT_DETAIL_MAX_WORDS,
) -> Dict[str, FrameCaptionEntry]:
    """Load frame_id (str) -> {concise, detail}. Missing file returns {}. Word caps applied on read."""
    path = dam_frame_caption_json_path(
        dataset_img_folder, take_id, object_name, captions_subdir=captions_subdir
    )
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {}
    if "frames" in data and isinstance(data["frames"], dict):
        raw = data["frames"]
    else:
        raw = data
    out: Dict[str, FrameCaptionEntry] = {}
    for k, v in raw.items():
        if str(k).startswith("_"):
            continue
        if v is None:
            continue
        out[str(k)] = normalize_frame_caption_entry(
            v, concise_max_words=concise_max_words, detail_max_words=detail_max_words
        )
    return out


def caption_text_for_training(entry: Union[str, FrameCaptionEntry, None]) -> str:
    """Prefer detail, then concise, for a single RoBERTa string."""
    if entry is None:
        return ""
    if isinstance(entry, str):
        return entry.strip()
    if isinstance(entry, dict):
        d = str(entry.get("detail", "") or "").strip()
        if d:
            return d
        return str(entry.get("concise", "") or "").strip()
    return ""


def frame_caption_entry_complete(entry: Union[str, FrameCaptionEntry, None]) -> bool:
    """True if both concise and detail are non-empty (dual-caption precompute)."""
    if entry is None:
        return False
    if isinstance(entry, str):
        return bool(entry.strip())
    if isinstance(entry, dict):
        return bool(str(entry.get("concise", "") or "").strip()) and bool(
            str(entry.get("detail", "") or "").strip()
        )
    return False


def save_dam_frame_captions(
    dataset_img_folder: str,
    take_id: str,
    object_name: str,
    frame_id_to_caption: Dict[str, Union[str, FrameCaptionEntry]],
    meta: Optional[dict] = None,
    captions_subdir: str = "dam_frame_captions",
    concise_max_words: int = DEFAULT_CONCISE_MAX_WORDS,
    detail_max_words: int = DEFAULT_DETAIL_MAX_WORDS,
) -> str:
    """Write captions JSON; frame values are {concise, detail} objects (or coerced from str)."""
    out_dir = dam_frame_captions_dir(dataset_img_folder, take_id, captions_subdir)
    os.makedirs(out_dir, exist_ok=True)
    path = dam_frame_caption_json_path(
        dataset_img_folder, take_id, object_name, captions_subdir=captions_subdir
    )
    frames_out: Dict[str, Dict[str, str]] = {}
    for k, v in frame_id_to_caption.items():
        norm = normalize_frame_caption_entry(
            v, concise_max_words=concise_max_words, detail_max_words=detail_max_words
        )
        frames_out[str(k)] = {"concise": norm["concise"], "detail": norm["detail"]}
    payload: dict = {"frames": frames_out}
    if meta:
        payload["_meta"] = meta
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path
