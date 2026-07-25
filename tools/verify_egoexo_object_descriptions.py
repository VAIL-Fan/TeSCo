#!/usr/bin/env python3
"""
Verify that every EgoExo training sample (same indexing as EgoExoRawDataset)
has ``<take>/<DAM_FRAME_CAPTIONS_SUBDIR>/`` and non-empty per-frame text for frames used in the index.

Usage:
  python tools/verify_egoexo_object_descriptions.py --img_folder /path/to/EgoExo4D/train [--num_frames 8]
"""
from __future__ import annotations

import argparse
import os
import sys

# Repo root
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from sam2.utils.dam_frame_captions import DAM_FRAME_CAPTIONS_SUBDIR
from training.dataset.vos_raw_dataset import EgoExoRawDataset, MissingDamFrameCaptionsError


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--img_folder",
        type=str,
        default=None,
        help="EgoExo train (or val) root containing take UUID folders",
    )
    p.add_argument("--num_frames", type=int, default=8)
    args = p.parse_args()

    if not args.img_folder:
        print("ERROR: pass --img_folder (e.g. path from dataset.img_folder in training YAML)")
        sys.exit(2)

    img_folder = os.path.expanduser(args.img_folder)
    if not os.path.isdir(img_folder):
        print(f"ERROR: not a directory: {img_folder}")
        sys.exit(2)

    ds = EgoExoRawDataset(
        img_folder=img_folder,
        gt_folder=None,
        file_list_txt=None,
        num_frames=args.num_frames,
        excluded_videos_list_txt=None,
    )

    n = len(ds)
    print(f"img_folder: {img_folder}")
    print(f"num_frames threshold: {args.num_frames}")
    print(f"Indexed training samples (ego-exo-object tuples): {n}")
    print(f"DAM captions subdirectory: {DAM_FRAME_CAPTIONS_SUBDIR}/")

    missing_dam_dir = []  # take has no DAM captions directory
    empty_caption_frames = []  # (idx, video_name, take_id) with some frame caption empty

    takes_seen = set()
    for i in range(n):
        video_name = ds.video_names[i]
        parts = video_name.split("/")
        take_id = parts[-4] if len(parts) >= 4 else parts[0]
        takes_seen.add(take_id)

        dam_dir = os.path.join(img_folder, take_id, DAM_FRAME_CAPTIONS_SUBDIR)
        if not os.path.isdir(dam_dir):
            missing_dam_dir.append((i, video_name, take_id))
            continue

        try:
            _, _, _, _, _desc, frame_caption_by_fid = ds.get_video(i)
        except MissingDamFrameCaptionsError:
            missing_dam_dir.append((i, video_name, take_id))
            continue

        frames = ds.frames[video_name]
        bad = False
        for fid_s in frames:
            try:
                fid = int(str(fid_s).strip())
            except ValueError:
                continue
            cap = frame_caption_by_fid.get(fid, "")
            if not str(cap).strip():
                bad = True
                break
        if bad:
            empty_caption_frames.append((i, video_name, take_id))

    print(f"Unique take_ids referenced by index: {len(takes_seen)}")

    if missing_dam_dir:
        print(f"\nFAIL: {len(missing_dam_dir)} samples in takes with NO {DAM_FRAME_CAPTIONS_SUBDIR}/ directory")
        for row in missing_dam_dir[:15]:
            print(f"  idx={row[0]} take={row[2]} vid={row[1]}")
        if len(missing_dam_dir) > 15:
            print(f"  ... and {len(missing_dam_dir) - 15} more")

    if empty_caption_frames:
        print(f"\nFAIL: {len(empty_caption_frames)} samples with empty DAM caption for at least one frame")
        for row in empty_caption_frames[:20]:
            print(f"  idx={row[0]} take={row[2]} vid={row[1]}")
        if len(empty_caption_frames) > 20:
            print(f"  ... and {len(empty_caption_frames) - 20} more")

    takes_no_dam = []
    for tid in sorted(takes_seen):
        p = os.path.join(img_folder, tid, DAM_FRAME_CAPTIONS_SUBDIR)
        if not os.path.isdir(p):
            takes_no_dam.append(tid)

    if takes_no_dam:
        print(f"\nWARN: {len(takes_no_dam)} take_ids have no {DAM_FRAME_CAPTIONS_SUBDIR}/ at all:")
        for tid in takes_no_dam[:20]:
            print(f"  {tid}")
        if len(takes_no_dam) > 20:
            print(f"  ... and {len(takes_no_dam) - 20} more")

    ok = len(missing_dam_dir) == 0 and len(empty_caption_frames) == 0
    if ok:
        print(f"\nOK: all indexed samples have {DAM_FRAME_CAPTIONS_SUBDIR}/ and non-empty text per frame.")
        sys.exit(0)
    print(f"\nNOT OK: add {DAM_FRAME_CAPTIONS_SUBDIR} or fix per-frame files before text-conditioned training.")
    sys.exit(1)


if __name__ == "__main__":
    main()
