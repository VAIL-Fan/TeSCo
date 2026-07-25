# Per-frame caption files under <take>/<DAM_FRAME_CAPTIONS_SUBDIR>/

from __future__ import annotations

import json
import os

# Subdirectory under each EgoExo take UUID (train/val/test) holding per-frame .txt/.json captions.
DAM_FRAME_CAPTIONS_SUBDIR = "dam_frame_captions_v2"


def read_dam_frame_caption(captions_dir: str, frame_token: str) -> str:
    """
    Load text for one frame. ``frame_token`` is the stem used in the dataset (e.g. ``video_frames[i]``).

    Tries, in order:
    - ``{token}.txt``
    - ``{token}.json`` (expects a JSON object with key ``text`` or ``caption``, else raw read)
    - zero-padded 6-digit ``.txt`` if token is numeric
    """
    if not captions_dir or not os.path.isdir(captions_dir):
        return ""

    token = str(frame_token).strip()
    if not token:
        return ""

    candidates = [f"{token}.txt", f"{token}.json"]
    if token.isdigit():
        z = token.zfill(6)
        candidates.extend([f"{z}.txt", f"{z}.json"])

    for name in candidates:
        path = os.path.join(captions_dir, name)
        if not os.path.isfile(path):
            continue
        try:
            if name.endswith(".json"):
                with open(path, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                if isinstance(data, str):
                    return data.strip()
                if isinstance(data, dict):
                    for k in ("text", "caption", "description", "utterance"):
                        if k in data and data[k] is not None:
                            return str(data[k]).strip()
                return ""
            with open(path, "r", encoding="utf-8") as fp:
                return fp.read().strip()
        except (OSError, UnicodeDecodeError, ValueError):
            continue
    return ""


def _normalize_object_key(name: str) -> str:
    return str(name).strip().replace("/", "-").replace("_", "").replace(" ", "").lower()


def _load_frame_caption_map_from_object_json(captions_dir: str, object_key: str) -> dict[int, str]:
    """
    Read captions from object-level DAM files like `<object>_0.json` that store:
    {
      "frames": { "2550": {"concise": "...", "detail": "..."} , ... }
    }
    """
    if not object_key:
        return {}

    normalized_target = _normalize_object_key(object_key)
    candidates = []
    try:
        for name in os.listdir(captions_dir):
            if not name.endswith(".json") or name.endswith(".json.lock"):
                continue
            stem = name[:-5]
            if stem.endswith("_0"):
                stem = stem[:-2]
            if _normalize_object_key(stem) == normalized_target:
                candidates.append(os.path.join(captions_dir, name))
    except OSError:
        return {}

    out: dict[int, str] = {}
    for path in candidates:
        try:
            with open(path, "r", encoding="utf-8") as fp:
                data = json.load(fp)
        except (OSError, UnicodeDecodeError, ValueError):
            continue
        frames = data.get("frames") if isinstance(data, dict) else None
        if not isinstance(frames, dict):
            continue
        for fid_str, payload in frames.items():
            try:
                fid = int(str(fid_str).strip())
            except ValueError:
                continue
            txt = ""
            if isinstance(payload, dict):
                txt = str(payload.get("concise") or payload.get("text") or payload.get("caption") or "").strip()
                if not txt:
                    txt = str(payload.get("detail") or payload.get("description") or "").strip()
            elif isinstance(payload, str):
                txt = payload.strip()
            if txt and not out.get(fid):
                out[fid] = txt
    return out


def _load_frame_caption_map_from_any_object_json(captions_dir: str) -> dict[int, str]:
    """Fallback: for each frame id, use the first non-empty caption from any object-level JSON."""
    out: dict[int, str] = {}
    try:
        names = os.listdir(captions_dir)
    except OSError:
        return out
    for name in names:
        if not name.endswith(".json") or name.endswith(".json.lock"):
            continue
        path = os.path.join(captions_dir, name)
        try:
            with open(path, "r", encoding="utf-8") as fp:
                data = json.load(fp)
        except (OSError, UnicodeDecodeError, ValueError):
            continue
        frames = data.get("frames") if isinstance(data, dict) else None
        if not isinstance(frames, dict):
            continue
        for fid_str, payload in frames.items():
            if isinstance(payload, dict):
                txt = str(payload.get("concise") or payload.get("text") or payload.get("caption") or "").strip()
                if not txt:
                    txt = str(payload.get("detail") or payload.get("description") or "").strip()
            elif isinstance(payload, str):
                txt = payload.strip()
            else:
                txt = ""
            if not txt:
                continue
            try:
                fid = int(str(fid_str).strip())
            except ValueError:
                continue
            if fid not in out:
                out[fid] = txt
    return out


def load_frame_caption_map_for_take(
    captions_dir: str, frame_tokens: list, object_key: str | None = None
) -> dict[int, str]:
    """Map int(frame_id) -> caption for each token in the video's frame list."""
    out: dict[int, str] = {}
    if not captions_dir or not os.path.isdir(captions_dir):
        return out
    object_level_map = _load_frame_caption_map_from_object_json(captions_dir, object_key)
    any_object_level_map = _load_frame_caption_map_from_any_object_json(captions_dir)
    for tok in frame_tokens:
        try:
            fid = int(str(tok).strip())
        except ValueError:
            continue
        txt = read_dam_frame_caption(captions_dir, str(tok).strip())
        if not txt and object_level_map:
            txt = object_level_map.get(fid, "")
        if not txt and any_object_level_map:
            txt = any_object_level_map.get(fid, "")
        out[fid] = txt
    return out
