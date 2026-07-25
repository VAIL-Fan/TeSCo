# Copyright (c) Meta Platforms, Inc. and affiliates.
# Single source for reading per-object text from object_descriptions_ablation.json (EgoExo takes).

from __future__ import annotations

import json
import os


def load_object_description_ablation(base_video_dir: str, video_name: str, object_name: str) -> str:
    """
    Read per-take `object_descriptions_ablation.json` and return the description for this object
    (empty string if missing). Matches training key lookup in `vos_raw_dataset.EgoExoRawDataset`.
    """
    description_path = os.path.join(base_video_dir, video_name, "object_descriptions_ablation.json")
    if not os.path.exists(description_path):
        return ""
    try:
        with open(description_path, "r") as fp:
            desc_data = json.load(fp)
    except (json.JSONDecodeError, OSError):
        return ""
    if object_name in desc_data:
        return desc_data[object_name] or ""
    for key in desc_data:
        if key in object_name or object_name in key:
            return desc_data[key] or ""
    return ""
