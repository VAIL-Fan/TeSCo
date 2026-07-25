import os
import json
from tqdm import tqdm
from pycocotools import mask as mask_utils
import numpy as np
import argparse


def check_pred_format(input_dir, pred_dir, split, swap, json_path):
    from copy import deepcopy

    input_dir = os.path.join(input_dir, split)
    videos = os.listdir(input_dir)
    task_name = "ego-exo" if not swap else "exo-ego"
    annotations = {task_name: {"version": "xx", "challenge": "xx", "results": {}}}
    for vid in tqdm(videos):
        if not os.path.isdir(f"{pred_dir}/{vid}"):
            continue
        with open(f"{pred_dir}/{vid}/annotations.json", "r") as fp:
            vid_anno = json.load(fp)

        correct_anno = deepcopy(vid_anno)

        for obj in vid_anno["masks"]:
            correct_anno["masks"][obj] = {}
            #
            for cam in vid_anno["masks"][obj]:

                for frame_idx in vid_anno["masks"][obj][cam]:
                    mask_dict = vid_anno["masks"][obj][cam][frame_idx]
                    if mask_dict:
                        mask = mask_utils.decode(mask_dict)
                    else:
                        continue
                    encoded_mask = mask_utils.encode(np.asfortranarray(mask))
                    encoded_mask["counts"] = encoded_mask["counts"].decode("ascii")
                    vid_anno["masks"][obj][cam][frame_idx] = {
                        "pred_mask": encoded_mask,
                        "confidence": int(
                            mask.sum() > (0.001 * mask.shape[0] * mask.shape[1])
                        ),
                    }
                # Folder is stored as "{exo_cam}__{aria_cam}" or "{aria_cam}__{exo_cam}".
                _parts = cam.split("__")
                _aria = _parts[0] if "aria" in _parts[0] else _parts[1]
                _exo = _parts[1] if "aria" in _parts[0] else _parts[0]
                # ego-exo GT uses "{aria}_{exo}"; exo-ego GT uses "{exo}_{aria}".
                cam_key = f"{_exo}_{_aria}" if swap else f"{_aria}_{_exo}"
                correct_anno["masks"][obj][cam_key] = vid_anno["masks"][obj][cam]

        annotations[task_name]["results"][vid] = correct_anno

    # output_dir = os.path.dirname(pred_dir)
    # print(f"{output_dir}/final_results.json")
    with open(json_path, "w+") as fp:
        json.dump(annotations, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        help="The input path",
        default="/data/seg/EgoExo4D",
    )
    parser.add_argument(
        "--split",
        help="The split name",
        default="val",
    )
    parser.add_argument(
        "--pred_dir",
        help="The predicted path",
        default="/data/seg/LM-EEC/egoexo_val",
    )
    parser.add_argument(
        "--json_path",
        help="The merged json file path",
        default="/data/seg/LM-EEC/egoexo_val_results.json",
    )

    parser.add_argument(
        "--swap",
        action="store_true",
        help="Build submission under task exo-ego with camera keys {exo}_{aria}.",
    )
    args = parser.parse_args()
    check_pred_format(args.input_dir, args.pred_dir, args.split, args.swap, args.json_path)
