#!/usr/bin/env python3
"""
Build a temporal-paired HANDAL dataset in LM-EEC layout:
  key frame i  ->  exo view uses frame i, aria view uses frame i+offset.

Input is an existing LM-EEC-style HANDAL root (with per-take annotation.json and
aria01/exo01 image folders). Output is a new LM-EEC-style root that keeps frame
tokens as i while shifting only aria-side images/masks by +offset.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def read_caption_text(cap_dir: Path, token: str) -> str:
    txt_path = cap_dir / f"{token}.txt"
    if txt_path.is_file():
        try:
            return txt_path.read_text(encoding="utf-8").strip()
        except OSError:
            return ""
    return ""


def write_caption_text(cap_dir: Path, token: str, text: str) -> None:
    cap_dir.mkdir(parents=True, exist_ok=True)
    (cap_dir / f"{token}.txt").write_text((text or "").strip() + "\n", encoding="utf-8")


def linked_or_copied(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def process_take(src_take: Path, dst_take: Path, offset: int) -> bool:
    ann_path = src_take / "annotation.json"
    if not ann_path.is_file():
        return False

    with ann_path.open("r", encoding="utf-8") as fp:
        ann = json.load(fp)

    masks = ann.get("masks", {})
    if not masks:
        return False

    src_aria = src_take / "aria01"
    src_exo = src_take / "exo01"
    if not src_aria.exists() or not src_exo.exists():
        return False

    dst_aria = dst_take / "aria01"
    dst_exo = dst_take / "exo01"
    dst_cap_aria = dst_take / "dam_frame_captions_v2"
    dst_cap_exo = dst_take / "dam_frame_captions_v2_exo"
    dst_aria.mkdir(parents=True, exist_ok=True)
    dst_exo.mkdir(parents=True, exist_ok=True)
    dst_cap_aria.mkdir(parents=True, exist_ok=True)
    dst_cap_exo.mkdir(parents=True, exist_ok=True)

    src_cap_aria = src_take / "dam_frame_captions_v2"
    src_cap_exo = src_take / "dam_frame_captions_v2_exo"

    out_masks = {}
    global_tokens = set()

    for obj_name, cams in masks.items():
        aria_map = cams.get("aria01", {})
        exo_map = cams.get("exo01", {})
        if not aria_map or not exo_map:
            continue

        out_aria = {}
        out_exo = {}
        exo_tokens = sorted(exo_map.keys())
        aria_keys = set(aria_map.keys())

        for tok in exo_tokens:
            try:
                shifted = str(int(tok) + offset).zfill(len(tok))
            except ValueError:
                continue
            if shifted not in aria_keys:
                continue

            # Keep output token as original exo token i.
            out_exo[tok] = exo_map[tok]
            out_aria[tok] = aria_map[shifted]
            global_tokens.add(tok)

            # exo image at i
            src_exo_img = src_exo / f"{tok}.jpg"
            # aria image at i+offset, renamed as i.jpg in new dataset
            src_aria_img = src_aria / f"{shifted}.jpg"
            if src_exo_img.is_file() and src_aria_img.is_file():
                linked_or_copied(src_exo_img, dst_exo / f"{tok}.jpg")
                linked_or_copied(src_aria_img, dst_aria / f"{tok}.jpg")

            # captions aligned to each branch's actual frame source
            exo_text = read_caption_text(src_cap_exo, tok) if src_cap_exo.exists() else ""
            aria_text = read_caption_text(src_cap_aria, shifted) if src_cap_aria.exists() else ""
            if not exo_text:
                exo_text = obj_name.replace("_", " ")
            if not aria_text:
                aria_text = obj_name.replace("_", " ")
            write_caption_text(dst_cap_exo, tok, exo_text)
            write_caption_text(dst_cap_aria, tok, aria_text)

        if out_aria and out_exo:
            out_masks[obj_name] = {"aria01": out_aria, "exo01": out_exo}

    if not out_masks:
        return False

    out_ann = {
        "masks": out_masks,
        "subsample_idx": sorted(int(t) for t in global_tokens),
    }
    dst_take.mkdir(parents=True, exist_ok=True)
    with (dst_take / "annotation.json").open("w", encoding="utf-8") as fp:
        json.dump(out_ann, fp)
    return True


def process_split(src_split: Path, dst_split: Path, offset: int) -> tuple[int, int]:
    total = 0
    kept = 0
    for take in sorted(src_split.iterdir()):
        if not take.is_dir():
            continue
        total += 1
        ok = process_take(take, dst_split / take.name, offset)
        if ok:
            kept += 1
    return total, kept


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_root", required=True, help="Existing handal_lmeec root containing train/test.")
    parser.add_argument("--dst_root", required=True, help="Output root for temporal-paired dataset.")
    parser.add_argument("--offset", type=int, default=100, help="Temporal offset (target=source+offset).")
    parser.add_argument("--overwrite", action="store_true", help="Delete destination root if it exists.")
    args = parser.parse_args()

    src_root = Path(args.src_root).resolve()
    dst_root = Path(args.dst_root).resolve()
    if not src_root.is_dir():
        raise FileNotFoundError(f"src_root not found: {src_root}")

    if dst_root.exists() and args.overwrite:
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    for split in ("train", "test"):
        src_split = src_root / split
        if not src_split.is_dir():
            continue
        dst_split = dst_root / split
        dst_split.mkdir(parents=True, exist_ok=True)
        total, kept = process_split(src_split, dst_split, args.offset)
        print(f"[{split}] processed takes: {total}, kept: {kept}")

    print(f"Done. Temporal-paired dataset written to: {dst_root}")


if __name__ == "__main__":
    main()

