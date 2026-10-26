"""Report DAM caption coverage per take, before a long inference run.

A take with no captions does not raise -- the text branch just receives empty
strings, and the resulting prediction is neither the text-conditioned model's
nor a clean ablation. So coverage has to be checked up front, not after.

    python tools/check_captions.py --split data_segswap_test \
        --field dam_frame_captions_v2_exo
"""
import argparse
import json
import os

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GT_FOR_FIELD = {
    "dam_frame_captions_v2": ("egoexo_test_gt.json", "ego-exo"),
    "dam_frame_captions_v2_exo": ("exoego_test_gt.json", "exo-ego"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", default="data_segswap_test")
    parser.add_argument("--field", default="dam_frame_captions_v2_exo",
                        choices=sorted(GT_FOR_FIELD))
    parser.add_argument("--text_labels", default=os.path.join(ROOT, "text_labels", "EgoExo4D"))
    parser.add_argument("--gt", help="defaults to the GT file matching --field")
    args = parser.parse_args()

    gt_name, task = GT_FOR_FIELD[args.field]
    gt_path = args.gt or os.path.join(ROOT, "data", "gt_results", gt_name)
    if not os.path.isfile(gt_path):
        raise SystemExit(f"GT file not found: {gt_path} (pass --gt explicitly)")
    needed = set(json.load(open(gt_path))[task]["results"])

    split_dir = os.path.join(args.text_labels, args.split)
    present = set(os.listdir(split_dir)) if os.path.isdir(split_dir) else set()

    no_dir = sorted(needed - present)
    no_field, files = [], 0
    for take in sorted(needed & present):
        field_dir = os.path.join(split_dir, take, args.field)
        if not os.path.isdir(field_dir) or not os.listdir(field_dir):
            no_field.append(take)
        else:
            files += len(os.listdir(field_dir))

    covered = len(needed) - len(no_dir) - len(no_field)
    print(f"field            : {args.field}")
    print(f"GT takes         : {len(needed)}")
    print(f"covered          : {covered}")
    print(f"no take dir      : {len(no_dir)}")
    print(f"dir but no field : {len(no_field)}")
    print(f"caption files    : {files}")
    if covered < len(needed):
        print(f"\nWARNING: {len(needed) - covered} takes lack '{args.field}'. Running inference "
              f"on them yields a confounded result, not an error. Generate their captions first.")
        for take in (no_dir + no_field)[:10]:
            print(f"  {take}")
        if len(no_dir) + len(no_field) > 10:
            print(f"  ... and {len(no_dir) + len(no_field) - 10} more")
    else:
        print("\nAll GT takes have captions for this field.")


if __name__ == "__main__":
    main()
