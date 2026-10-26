# data/

Where the Ego-Exo4D dataset goes. The image/video data is not in this repo (large and
third-party-licensed); see [`../docs/SETUP.md`](../docs/SETUP.md) for where to obtain it and the
exact layout expected here.

What this directory does carry:

```
data/
├── EgoExo4D -> ...            # symlink to the image splits (train/val/data_segswap_test)
└── gt_results/
    ├── egoexo_test_gt.json    # Ego->Exo ground truth, task key "ego-exo"
    └── exoego_test_gt.json    # Exo->Ego ground truth, task key "exo-ego"
```

`gt_results/` is what `evaluation/unified_eval.py` scores against, so these two files are
required even if you only evaluate. Both are 291 takes at 480x480; check a file reproduces the
documented row counts before trusting a score:

```bash
python tools/check_gt.py data/gt_results/egoexo_test_gt.json ego-exo
python tools/check_gt.py data/gt_results/exoego_test_gt.json exo-ego
```

Note that the symlinks above point outside this folder, so a copy of the folder alone will not
resolve them — repoint them at your own dataset location. `.gitignore` excludes everything here
except this file, so local dataset paths never get committed.
