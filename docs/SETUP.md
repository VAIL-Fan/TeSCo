# Setup, data and running TeSCo

Everything needed to go from a clean checkout to a reproduced number. The
README keeps only the short version; this is the full detail, moved here
unchanged so nothing was lost in shortening it.

For which metric protocol to use and what our numbers mean, see
[`METRIC_SETTINGS.md`](METRIC_SETTINGS.md) and [`RESULTS.md`](RESULTS.md).

## Folder structure


```
TeSCo/
├── README.md                    # this file
├── environment.yml              # conda env spec (env name: tesco)
├── setup.py                     # pip-installs this repo's sam2/training/evaluation packages
├── .gitattributes                # Git LFS tracking for *.pt/*.pth (see "Notes for a git release")
├── sam2/                        # model code: backbone, memory attention, TCF (FiLM), ITFR,
│   └── configs/sam2.1_training/ #   mask decoder, configs/ (Hydra yaml), predictors
│       ├── sam2.1_hiera_b+_EgoExo_finetune.yaml   # Ego-Exo4D training
│       └── sam2.1_hiera_b+_Handal_finetune.yaml   # HANDAL-X training ("retraining" setting)
├── training/                    # training entrypoint (train.py), dataset loaders, trainer,
│   ├── checkpoints_EgoExo/      #   loss functions
│   │   ├── tesco_egoexo4d_epoch59.pth                 # <- included: raw, for training resume
│   │   ├── tesco_egoexo4d_epoch59_for_inference.pt     # <- included: use this for inference
│   │   ├── tesco_egoexo4d_epoch79.pth                 # <- included: raw, for training resume
│   │   └── tesco_egoexo4d_epoch79_for_inference.pt     # <- included: use this for inference
│   └── checkpoints_Handal/      # created by tools/run_train_handal.sh (empty here — no
│                                 #   pretrained HANDAL-X checkpoint is included, see below)
├── evaluation/                  # three scorers, deliberately kept side by side:
│                                 #   compute_metrics.py  - the LM-EEC protocol (this repo's
│                                 #     lineage); prediction-driven, skips unpredicted rows
│                                 #   omama_eval.py       - the O-MaMa protocol; GT-driven,
│                                 #     annotated_frames filter, GT resized to the prediction
│                                 #   unified_eval.py     - the official benchmark protocol;
│                                 #     GT-driven, full coverage, both sides at 480x480
│                                 #   plus compute_metrics_scenarios.py and
│                                 #   merge_pred.py / merge_results.py (assemble predictions)
├── tools/                       # correspondence_inference.py (run the model on a video split),
│   │                             #   precompute_dam_frame_captions.py (regenerate DAM captions),
│   │                             #   prepare_inference_checkpoint.py (convert a raw training
│   │                             #   checkpoint to the format correspondence_inference.py needs),
│   │                             #   build_handal_i100_lmeec.py (HANDAL-X data prep helper),
│   │                             #   diff_checkpoints.py (is this checkpoint this architecture?),
│   │                             #   check_gt.py (GT file integrity / row counts),
│   │                             #   check_captions.py (DAM caption coverage before a long run),
│   │                             #   make_selftest_pred.py (GT-as-prediction, to sanity-check
│   │                             #     the scorer on an input whose answer is known),
│   └── run_*.sh                 #   end-to-end train/infer/eval pipelines
├── third_party/describe_anything/  # vendored NVIDIA DAM package (840K, `dam/` module only —
│                                 #   needed only to regenerate captions, not to use existing ones)
├── checkpoint/
│   └── sam2.1_hiera_base_plus.pt   # official SAM2.1 base+ init weights, for training from scratch
├── text_labels/                 # <-- ALL already-generated DAM text captions (447MB EgoExo4D +
│   ├── EgoExo4D/{split}/{take_id}/dam_frame_captions_v2[_exo]/{object}.json    (1,254 takes)
│   └── HANDAL-X/
│       ├── handal_lmeec/{train,test}/{take_id}/dam_frame_captions_v2[_exo]/*.txt       (full: 1684 train / 520 test takes)
│       └── handal_lmeec_i100/{train,test}/{take_id}/dam_frame_captions_v2[_exo]/*.txt  (1681 train / 518 test takes — 3 fewer, used for HANDAL-X retraining, see "HANDAL-X usage")
├── data/                         # empty here — place/symlink your own dataset copies (see "Datasets")
└── docs/
    ├── ARCHITECTURE.md                    # detailed architecture <-> code mapping
    ├── ANNOTATION_FORMATS.md              # exact annotation.json / GT json schemas
    ├── ego2exo_gt_pred_metrics_howto.md   # exact GT-matching/resize protocol used by the metrics scripts
    ├── REPRODUCE.md                       # READ FIRST to reproduce reported numbers: which
    │                                      #   checkpoint goes with which direction, and which
    │                                      #   scorer to use
    ├── METRIC_SETTINGS.md                 # the two metric settings, their difference, our
    │                                      #   results under both, and which the paper reports
    ├── RESULTS.md                         # Table 1 row by row, with per-row provenance
    └── RELEASE_CHECKLIST.md               # what must be decided before this goes public
```

## Environment setup


```bash
conda env create -f environment.yml   # creates env "tesco"
conda activate tesco
pip install -e .                      # installs sam2/, training/, evaluation/ as packages
```

`transformers`/`nvidia/DAM-3B` are only needed if you run `tools/precompute_dam_frame_captions.py`
to regenerate/extend captions — not needed for training or evaluating against the included
`text_labels/`.

## Datasets (not included)


The video/image data for **Ego-Exo4D** and **HANDAL-X** are not included in this repo (too
large, and third-party-licensed). `data/` holds only pointers: a `README.md`, symlinks to the
image splits, and `data/gt_results/` with the two correspondence test GT files the scorer needs.
You need to obtain the image data separately and place it so the folder layout below matches; the
included `text_labels/` and checkpoints then let you reproduce evaluation without regenerating
captions or retraining from scratch.

**Ground truth.** `evaluation/unified_eval.py` reads one GT file per direction:

| Path | Task key | Takes | GT rows | Scoreable rows | RLE size |
|---|---|---|---|---|---|
| `data/gt_results/egoexo_test_gt.json` | `ego-exo` | 291 | 163,947 | 132,503 | 480 × 480 |
| `data/gt_results/exoego_test_gt.json` | `exo-ego` | 291 | 265,358 | 132,556 | 480 × 480 |

"Scoreable" = the target-view mask is non-empty; IoU, contour accuracy and location error are
averaged over exactly those rows. Verify a GT file reproduces these counts before trusting a
score: `python tools/check_gt.py data/gt_results/egoexo_test_gt.json ego-exo`.

**DAM caption coverage is incomplete for Exo→Ego.** `text_labels/` (2.9 GB, included) covers
`dam_frame_captions_v2` for 290/291 test takes, but `dam_frame_captions_v2_exo` for only
**201/291**. A take with no captions does not raise — the text branch receives empty strings and
the prediction is neither the text-conditioned model's nor a clean ablation — so a full-coverage
Exo→Ego run needs the 90 missing takes' captions generated first (DAM-3B, roughly 4 h on one GPU
for ~53 k captions). Always check first:

```bash
python tools/check_captions.py --field dam_frame_captions_v2_exo
```

### Ego-Exo4D

- **Source**: Grauman et al., *"Ego-Exo4D: Understanding Skilled Human Activity from First- and
  Third-Person Perspectives"*, CVPR 2024. Official dataset portal:
  [ego-exo4d-data.org](https://ego-exo4d-data.org) — requires signing a license agreement to
  download.
- **Expected layout** once downloaded/extracted (see `docs/ANNOTATION_FORMATS.md` for the exact
  `annotation.json` and `gt_results/*.json` schemas):
  ```
  data/EgoExo4D/
  ├── train/<take_id>/{aria_camera}/<frame>.jpg, {exo_camera}/<frame>.jpg, annotation.json
  ├── val/...                                    (same layout)
  ├── test/...                                    (same layout)
  └── gt_results/{ego2exo_test,exo2ego_test,...}.json   (official relation/GT files)
  ```
- Frames are expected pre-extracted at 1-in-30 sampling (see `docs/ANNOTATION_FORMATS.md` §1 for
  details on how camera indices line up across ego/exo).

### HANDAL-X

- **Source**: HANDAL-X is an ego-exo correspondence variant of the HANDAL dataset (Guo et al.,
  *"HANDAL: A Dataset of Real-World Manipulable Object Categories with Pose Annotations,
  Affordances, and Reconstructions"*, IROS 2023), with the ego-exo pairing/split introduced by
  Fu et al., *"ObjectRelator: Enabling Cross-View Object Relation Understanding Across
  Ego-Centric and Exo-Centric Perspectives"*, ICCV 2025. Obtain the base HANDAL data and the
  HANDAL-X ego-exo split from the official sources referenced in those two papers (search by
  title/authors above; a specific download URL isn't reproduced here to avoid linking a
  possibly-stale mirror).
- **Expected layout** (same `annotation.json` schema as EgoExo4D — `docs/ANNOTATION_FORMATS.md`
  applies to both):
  ```
  data/handal_lmeec/{train,test}/<take_id>/{aria01,exo01}/<frame>.jpg, annotation.json
  data/handal_lmeec_i100/{train,test}/...          (same layout, see "HANDAL-X usage" below
                                                      for why there are two copies)
  ```
- `tools/build_handal_i100_lmeec.py` is included as a reference for one preprocessing step used
  in this project (temporal frame-offset pairing between the aria and exo views, given data
  that's *already* in the `annotation.json` + `aria01`/`exo01` layout above) — it is **not** a
  full raw-HANDAL-to-this-layout converter; you may need to adapt it or write your own
  conversion step depending on the format the HANDAL-X release you obtain is in.

## Text labels (the DAM-generated captions)


**All already-generated DAM captions are included in `text_labels/`** as a standalone, portable
archive (447MB for EgoExo4D across 1,254 takes + 2.5GB for HANDAL-X — mostly filesystem block
overhead from many small per-frame `.txt` files, not real content size). This is what lets you
skip caption generation entirely and go straight to training/evaluating once you have the raw
video data in place.

Two important things to know:

1. **Format differs by dataset.** EgoExo4D captions are one JSON per object per take
   (`dam_frame_captions_v2/{object}.json`, containing `{"frames": {"<frame_idx>": {"concise":
   ..., "detail": ...}}}`). HANDAL-X captions are one `.txt` file per frame
   (`dam_frame_captions_v2_exo/<frame_idx>.txt`, plain text). The loader code
   (`sam2/utils/dam_frame_captions.py`) handles both transparently.
2. **The dataloader expects `dam_frame_captions_v2*/` to sit *inside* each take's own folder**,
   next to its RGB frames — e.g. `data/EgoExo4D/<split>/<take_id>/dam_frame_captions_v2/...`.
   So after you place the raw Ego-Exo4D / HANDAL-X video data under `data/` (see "Datasets"
   below), **copy the matching subtree from `text_labels/<dataset>/...` into each take's folder**
   (same `<split>/<take_id>/` path, just merge the `dam_frame_captions_v2*/` subfolder in) before
   training or running inference. A one-liner to do this for EgoExo4D once `data/EgoExo4D/train`
   exists:
   ```bash
   rsync -a text_labels/EgoExo4D/ data/EgoExo4D/
   ```
   (and similarly `rsync -a text_labels/HANDAL-X/handal_lmeec/ data/handal_lmeec/`, etc.)

To regenerate or extend captions for new data instead of using the included ones, use
`tools/precompute_dam_frame_captions.py` (needs `third_party/describe_anything` on
`PYTHONPATH`, done automatically by the script, plus `transformers>=4.46,<5` and `nvidia/DAM-3B`
downloaded from HuggingFace):

```bash
python tools/precompute_dam_frame_captions.py \
  --dataset_root data/EgoExo4D/train \
  --dam_model_path nvidia/DAM-3B \
  --captions_subdir dam_frame_captions_v2
```

## Checkpoints included


| File | Epoch | Notes |
|---|---|---|
| `checkpoint/sam2.1_hiera_base_plus.pt` | — | Official SAM2.1 base+ release weights; init point for training from scratch. Already in the `{"model": state_dict}` format `correspondence_inference.py` expects. |
| `training/checkpoints_EgoExo/tesco_egoexo4d_epoch59.pth` | 59 (0-indexed → 60 training epochs) | Trained on Ego-Exo4D at 480 resolution; use this as the default checkpoint for Ego-Exo4D evaluation. Raw training-format state_dict — for **training resume**, not for `correspondence_inference.py` directly (see note below). |
| `training/checkpoints_EgoExo/tesco_egoexo4d_epoch59_for_inference.pt` | 59 | Same weights, converted to inference format (see below). **Use this one for testing/inference.** |
| `training/checkpoints_EgoExo/tesco_egoexo4d_epoch79.pth` | 79 | Same training run, 20 epochs further; used for HANDAL-X zero-shot generalization. Raw training format. |
| `training/checkpoints_EgoExo/tesco_egoexo4d_epoch79_for_inference.pt` | 79 | Same weights, inference format. **Use this one for testing/inference.** |

> **The checkpoint and the evaluation direction are not independent — do not pick by recency.**
> `epoch59_for_inference.pt` reproduces **Exo→Ego 70.48** (paper 70.30) and was verified
> bit-identical to the checkpoint that produced it (844/844 keys, 400/400 sampled tensors equal).
> The same checkpoint on **Ego→Exo gives 52.34**, not the paper's 60.40: that number came from a
> *different model variant* (a `text_filter_gate` MLP in place of this code's second ITFR block,
> `text_refine_attn2`/`text_refine_ln2`), whose weights this code cannot load. **This folder
> reproduces Exo→Ego only.** Read `docs/REPRODUCE.md` §2 before citing any Ego→Exo number, and
> use `tools/diff_checkpoints.py` to check any checkpoint against this code's architecture —
> `strict=False` loading will not tell you.

Both Ego-Exo4D checkpoints come from the same continuous 480-resolution training run, saved at
two different points. No checkpoint for a HANDAL-X-retrained model is included — see "HANDAL-X
usage" below and `tools/run_train_handal.sh` to train one yourself.

**Why two formats:** `training/train.py` saves/resumes from a plain `state_dict` with a DDP
`module.` prefix on every key. `sam2/build_sam.py`'s `_load_checkpoint` (used by
`correspondence_inference.py` and the video predictor) instead expects
`torch.load(ckpt)["model"]` with no `module.` prefix — the same format as the official
`sam2.1_hiera_base_plus.pt` release. Loading a raw training checkpoint directly into the
inference path **fails on `KeyError: 'model'`**, or — if you patch around that — silently loads
**zero** real weights, since `_load_checkpoint` uses `strict=False` and every `module.`-prefixed
key would mismatch (verified: both failure modes were reproduced while testing this repo). The
`_for_inference.pt` files were produced with `tools/prepare_inference_checkpoint.py` (included);
if you train your own checkpoint, convert it the same way before running inference:

```bash
python tools/prepare_inference_checkpoint.py \
  --in training/checkpoints_EgoExo/checkpoint.pt \
  --out training/checkpoints_EgoExo/checkpoint_for_inference.pt
```

This repo's checkpoint-loading was verified end-to-end (see "Verification performed" at the
bottom of this file): both `_for_inference.pt` files load into the model built from
`sam2/configs/sam2.1/sam2.1_hiera_b+_egoexo_text.yaml` with zero missing/unexpected keys, and a
real inference run (`correspondence_inference.py`) and a real training step
(`training/train.py`) both completed successfully against this exact code+config+checkpoint
combination on synthetic data matching the documented data format.

## Training configuration


`sam2/configs/sam2.1_training/sam2.1_hiera_b+_EgoExo_finetune.yaml`:

| Setting | Value |
|---|---|
| Resolution | 480×480 |
| Batch size | 3/GPU (12 total on 4 GPUs) |
| `base_lr` (new modules/decoder) | 1.0e-5, cosine decay to `base_lr/10` |
| `vision_lr` (image encoder) | 6.0e-6, cosine decay to `vision_lr/10` |
| `num_epochs` | 60 |
| `num_frames` per clip | 8 |
| Loss weights | `loss_mask: 20, loss_dice: 1, loss_iou: 1, loss_class: 1` |
| Optimizer | AdamW, weight decay 0.1 (0 for biases/LayerNorm), grad clip max-norm 0.1 |
| Precision | AMP bfloat16 |
| Init weights | `checkpoint/sam2.1_hiera_base_plus.pt` (official SAM2.1, `strict=False`) |

The inference config (`sam2/configs/sam2.1/sam2.1_hiera_b+_egoexo_text.yaml`) is also at 480
resolution, matching the included checkpoints.

## How to run: training


```bash
cd TeSCo
./tools/run_train.sh 4        # 4 GPUs; set CONDA_ENV=<name> to override the "tesco" env name
# equivalent to:
#   conda run -n tesco python training/train.py \
#     -c configs/sam2.1_training/sam2.1_hiera_b+_EgoExo_finetune.yaml \
#     --use-cluster 0 --num-gpus 4
```

- Trains **exo→ego** by default (`correspondence_direction: exo_to_ego`, reading
  `dam_frame_captions_v2_exo` for the search-view exo frames used as the query). Flip to
  `ego_to_exo` + `caption_subdir: dam_frame_captions_v2` in the YAML to train the other
  direction.
- Initializes from `checkpoint/sam2.1_hiera_base_plus.pt` (official SAM2.1, not a fine-tune).
- Checkpoints save to `training/checkpoints_EgoExo/checkpoint.pt` (last epoch only,
  `save_freq: 0`); the epoch-tagged `*.pth` files already included let you skip straight to
  evaluation without retraining.
- Requires ~4×48GB GPUs at the default batch=3/GPU, 480 resolution, 8 frames/clip.

To train on **HANDAL-X** instead:

```bash
./tools/run_train_handal.sh 4
# equivalent to:
#   conda run -n tesco python training/train.py \
#     -c configs/sam2.1_training/sam2.1_hiera_b+_Handal_finetune.yaml \
#     --use-cluster 0 --num-gpus 4
```

Saves to `training/checkpoints_Handal/` (deliberately a **separate** directory from
`checkpoints_EgoExo/` so the two runs can never accidentally resume from / overwrite each
other's checkpoints). Starts from `checkpoint/sam2.1_hiera_base_plus.pt`, same as the Ego-Exo4D
config — no HANDAL-X-pretrained checkpoint is included, so this is provided as a config to run,
not a checkpoint you can skip straight to.

## HANDAL-X usage — two settings, two dataset copies


There are two distinct ways to evaluate on HANDAL-X, and correspondingly two similarly-named
HANDAL-X data folders — using the wrong one for the wrong setting will silently give you a
different number, so this is worth getting right:

| Setting | Train on | Test on | Checkpoint |
|---|---|---|---|
| **Zero-shot** (generalization) | Ego-Exo4D only | `data/handal_lmeec/test` (full: 520 takes) | `training/checkpoints_EgoExo/tesco_egoexo4d_epoch79.pth` |
| **Retraining** (in-domain) | `data/handal_lmeec_i100/train` | `data/handal_lmeec_i100/test` (518 takes) | train your own via `tools/run_train_handal.sh` |

`handal_lmeec_i100` is not a small subsample despite the name — it's the same data as
`handal_lmeec` minus 3 excluded takes (`hammers__008005`, `hammers__010006`,
`screwdrivers__002001`). Both copies' captions are included under `text_labels/HANDAL-X/`.

## How to run: testing / inference


Run the model over a video split to produce prediction JSONs:

```bash
cd TeSCo
# Ego→Exo direction:
python tools/correspondence_inference.py \
  --sam2_cfg sam2.1/sam2.1_hiera_b+_egoexo_text.yaml \
  --sam2_checkpoint training/checkpoints_EgoExo/tesco_egoexo4d_epoch59_for_inference.pt \
  --base_video_dir data/EgoExo4D/test \
  --input_mask_dir data/EgoExo4D/test \
  --output_mask_dir output/egoexo_test_ego2exo \
  --caption_subdir dam_frame_captions_v2

# Exo→Ego direction -- add --exo_to_ego:
python tools/correspondence_inference.py \
  --sam2_cfg sam2.1/sam2.1_hiera_b+_egoexo_text.yaml \
  --sam2_checkpoint training/checkpoints_EgoExo/tesco_egoexo4d_epoch59_for_inference.pt \
  --base_video_dir data/EgoExo4D/test \
  --input_mask_dir data/EgoExo4D/test \
  --output_mask_dir output/egoexo_test_exo2ego \
  --exo_to_ego --caption_subdir dam_frame_captions_v2_exo

# HANDAL-X zero-shot: Ego-Exo4D-trained checkpoint (epoch 79), full handal_lmeec test set:
python tools/correspondence_inference.py \
  --sam2_cfg sam2.1/sam2.1_hiera_b+_egoexo_text.yaml \
  --sam2_checkpoint training/checkpoints_EgoExo/tesco_egoexo4d_epoch79_for_inference.pt \
  --base_video_dir data/handal_lmeec/test \
  --input_mask_dir data/handal_lmeec/test \
  --output_mask_dir output/handal_zeroshot

# HANDAL-X retraining: convert your trained checkpoint first (see "Checkpoints included"), then:
python tools/prepare_inference_checkpoint.py \
  --in training/checkpoints_Handal/checkpoint.pt \
  --out training/checkpoints_Handal/checkpoint_for_inference.pt
python tools/correspondence_inference.py \
  --sam2_cfg sam2.1/sam2.1_hiera_b+_egoexo_text.yaml \
  --sam2_checkpoint training/checkpoints_Handal/checkpoint_for_inference.pt \
  --base_video_dir data/handal_lmeec_i100/test \
  --input_mask_dir data/handal_lmeec_i100/test \
  --output_mask_dir output/handal_retrained
```

`--input_mask_dir` isn't actually read on the `annotation.json`-driven EgoExo/HANDAL-X path
above (masks come from `annotation.json`, not a separate PNG directory) but is a required CLI
argument — any existing directory works; pointing it at the same video dir is the simplest
choice and is what was used for the verification run below.

For multi-GPU sharding, use `tools/run_inference_4gpu.sh` (round-robins a split across 4 GPUs
via `--gpu_id`/`--num_gpus`; override `CKPT`, `EEO_DATA_ROOT`, `OUTPUT` as env vars).
`tools/run_egoexo_test_inference_dam.sh` and `tools/run_exo_to_ego_test_eval.sh` wrap this plus
the merge+metrics steps below into one script — read the comments at the top of each for the
exact env vars they expect.

## How to run: computing metrics


Predictions from `correspondence_inference.py` are per-frame RLE JSONs; merge them into one
submission file, then score against ground truth:

```bash
cd TeSCo/evaluation

# 1) Merge per-take predictions into one JSON (see merge_pred.py / merge_results.py --help)
python merge_results.py --input_dir ../output/egoexo_test_ego2exo --task ego-exo \
  --out_json ../output/ego2exo_test_results.json

# 2) Aggregate metrics: mean IoU, Contour Accuracy, Location Error, Existence Accuracy
python unified_eval.py \
  --gt   ../data/gt_results/egoexo_test_gt.json \
  --pred ../output/ego2exo_test_results.json \
  --task ego-exo --coverage both
```

**Use `unified_eval.py`, not `compute_metrics.py`, for anything you intend to compare against
another method.** Both call the same official metric functions, but `compute_metrics.py` iterates
the *prediction* file at every level (take, object, frame), so a row the method never emitted is
dropped from the mean rather than scored as a miss — which silently rewards a method for
abstaining. `unified_eval.py` iterates the *ground truth* instead (what the official EGO4D driver
does) and scores a missing prediction as IoU 0. It also resizes a prediction onto the GT grid
when the two differ, so a method predicting at its own resolution can be scored unmodified.

`--coverage both` prints the two conventions side by side. **If they disagree, your prediction
file is incomplete and the higher number is not comparable to anything.** Measured on this repo:
a prediction file covering 201 of 291 takes scored 70.69 (skip) vs 54.31 (full coverage); once the
90 missing takes were actually run, the conventions agreed (70.69 vs 70.48).

Evaluation is at **480 × 480, nearest neighbour** — the grid the GT files are distributed at, not
a free parameter. See `docs/REPRODUCE.md` §1 and `docs/ego2exo_gt_pred_metrics_howto.md` for the
GT-matching protocol (object matched by `prompt.first_frame_anns[*].text`, not
`objects[*].formated` — getting this wrong is the most common cause of near-zero metrics).

`evaluation/compute_metrics_scenarios.py` can additionally break results down by EgoExo4D
scenario (Cooking/Music/Health/Basketball/Bike Repair/Soccer), given a GT file that maps each
take to a scenario — the official Ego-Exo4D metadata (`takes.json` from the dataset portal)
contains this mapping; see the script's `--help` for the expected GT format.

For HANDAL-X, `evaluation/metrics.py`'s standalone functions (`db_eval_iou`, `db_eval_boundary`,
`location_score`, `existence_accuracy`) can be called directly per prediction/GT mask pair.

## Verification performed


Before release, this repo's code/config/checkpoints were smoke-tested end-to-end against a
tiny synthetic take built to match the exact `annotation.json` + RLE mask + JPEG frame +
`dam_frame_captions_v2*` layout documented in `docs/ANNOTATION_FORMATS.md` (no real dataset
access was used for this — it's a correctness check of the pipeline wiring, not of model
quality):

- **Model/checkpoint compatibility**: built the model from `sam2/configs/sam2.1/sam2.1_hiera_b+_egoexo_text.yaml`
  and loaded both `tesco_egoexo4d_epoch59_for_inference.pt` and `..._epoch79_for_inference.pt`
  — **zero missing keys, zero unexpected keys** for both.
- **Inference**: ran `tools/correspondence_inference.py` against the synthetic take with the
  epoch-59 checkpoint — completed successfully, produced real per-frame prediction files, and
  logged `"Using object text ... for RoBERTa conditioning"`, confirming captions were read and
  used.
- **Training**: ran `training/train.py` with the real `sam2.1_hiera_b+_EgoExo_finetune.yaml`
  config (only `img_folder`/batch-size/epoch-count overridden to point at the synthetic take)
  — completed a full forward → loss → backward → optimizer step → checkpoint-save cycle, with
  `Text/non_empty_frame_descriptions_count: 8` in the logged metrics confirming the DAM
  captions were loaded and consumed by the model during training.
- **Bugs found and fixed as a result**: (1) the `--sam2_cfg` value must **not** include a
  `configs/` prefix (`sam2.1/sam2.1_hiera_b+_egoexo_text.yaml`, not
  `configs/sam2.1/sam2.1_hiera_b+_egoexo_text.yaml`) — the wrong prefix was in an earlier draft
  of this README and has been corrected throughout; (2) the raw training checkpoints needed a
  format conversion before `correspondence_inference.py` could load them — see "Checkpoints
  included" above and `tools/prepare_inference_checkpoint.py`, which didn't exist until this
  was caught.

**Not yet verified**: an end-to-end run against the *real* Ego-Exo4D/HANDAL-X datasets (not
available in this environment), and the evaluation/metrics scripts (`compute_metrics.py` etc.)
— those were checked for syntax/logic but not run against real predictions here.
