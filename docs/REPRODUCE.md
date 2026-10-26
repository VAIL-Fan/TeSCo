# Reproducing the Ego-Exo4D correspondence numbers

This document is the single source of truth for the numbers this folder can and cannot
reproduce. It exists because two things that look like details decide the score by tens of
IoU points: **which coverage convention the scorer uses**, and **which checkpoint goes with
which direction**. Both are stated explicitly below, with the commands to check them.

Everything here was measured on the Ego-Exo4D correspondence **test** split, 291 takes.

---

## 1. The metric: use `unified_eval.py`, not `compute_metrics.py`

Both scripts call the same metric functions (`utils.eval_mask`, `utils.location_score`,
`utils.existence_accuracy`, which are the official EGO4D/DAVIS implementations). They differ
in what they iterate over:

| | `compute_metrics.py` | `unified_eval.py` |
|---|---|---|
| Outer loop | `preds["results"]` | `gt["results"]` |
| A take/object/camera/frame with no prediction | **omitted from the mean** | scored as an empty mask (IoU 0) |
| Prediction at a different resolution | not handled (shape assert fails) | resized to the GT grid, `INTER_NEAREST` |

`compute_metrics.py` is prediction-driven at all three levels (take, object, frame). For a
submission validated to cover the whole test set the two agree, but they diverge the moment a
method's output is incomplete — and then the method that answers only where it is confident is
rewarded for abstaining. **Any cross-method comparison must use `unified_eval.py`** (or
`--coverage gt`, its default), which is what the official EGO4D driver does: upstream
`XMem/scripts/evaluation.py` iterates `gt["annotation"].keys()`.

How much this matters, measured: on a prediction file covering 201 of 291 takes, Exo→Ego scored
**70.69** under skip and **54.31** under full coverage. After the 90 missing takes were actually
run, the two conventions agreed (**70.69** vs **70.48**) — the gap had been pure coverage, not a
metric quirk.

```bash
cd /path/to/TeSCo
python evaluation/unified_eval.py \
    --gt   data/gt_results/egoexo_test_gt.json \
    --pred outputs/<run>/merged.json \
    --task ego-exo \
    --coverage both          # prints skip and full coverage side by side
```

`--coverage both` is the recommended way to report: if the two numbers differ, the prediction
file is incomplete and the skip number is not comparable to anything.

**What each column means.** IoU and existence accuracy are percentages; contour accuracy
(`Cont.A`, the boundary F-measure that `utils.eval_mask` returns as its second value) and
location error are fractions, matching how the paper's table prints them. Existence accuracy here
is the per-row agreement on "is the object present", averaged over scoreable rows only — it is
*not* `compute_metrics.py`'s balanced accuracy over every row including empty-GT ones, so the two
scripts' existence numbers are not interchangeable even when coverage is complete. The IoU,
contour and location columns are directly comparable between the two.

### Evaluation resolution

**480 × 480, nearest neighbour** — but this is a *choice*, not the only ground truth available,
and an earlier version of this document was wrong to call it "the benchmark grid, not a free
parameter". Every RLE in the two GT files shipped here is `size == [480, 480]`,
`utils.reshape_img_nopad` defaults to `max_dim=480` and `utils.location_score` defaults to
`size=(480, 480)`, so 480² is what this code evaluates at. A **1408 × 1408** ground truth for the
same annotations also exists (O-MaMa's `exo2egoGT.json`); the 480² files are a downscaled export
of it. Rescaled to a common grid the two agree closely but not exactly — mean IoU 0.9262, median
0.9436, 79.7% of rows above 0.9 over all 33,154 shared rows — the residual being resampling loss.
Pick one and state it; do not compare a 480²-scored number with a 1408²-scored one.

`unified_eval.py` resizes a prediction onto the GT grid only when the shapes differ, so a method
that already predicts at 480² is untouched. This is not a courtesy to any one method — it is the
benchmark grid, applied identically — but it does have an uneven *effect*, which is worth stating
when comparing:

| Method | Native prediction grid | Treatment |
|---|---|---|
| TeSCo | 480 × 480 | no resize (identity) |
| O-MaMa | 700 × 700 | nearest downscale to 480² |

Resolution is only one of three things that differ, and **not the dominant one**. Scoring
O-MaMa's own predictions under both protocols gives IoU **44.31** (its scorer) vs **42.81**
(`unified_eval.py`) — a 1.50-point gap, with `missing = 0` and the skip and full-coverage
conventions agreeing exactly, so coverage contributes nothing. The gap decomposes into:

Scoring the same predictions a third way separates the two candidate causes. The row set is
whatever O-MaMa's loop reaches; the grid is whichever side gets resized:

| scored how | rows | IoU |
|---|---|---|
| A — unified convention, all non-empty GT rows, 480² grid | 132,556 | 42.81 |
| B — unified convention, **only O-MaMa's rows**, 480² grid | 33,143 | **42.67** |
| C — O-MaMa's own scorer (1408² GT → the prediction's 700² grid) | 33,143 | **44.31** |

| cause | effect on IoU |
|---|---|
| restricting to O-MaMa's row set (B − A) | **−0.14** |
| GT scale and evaluation grid (C − B) | **+1.64** |

**It is the ground truth's provenance, not the sample, that moves the number.** O-MaMa's loop does
reach only about half the rows — it iterates the exo camera's frames but filters them by the *ego*
camera's `annotated_frames` list, and `annotated_frames` carries an entry for both cameras — but
averaging over that half instead of all of it is worth 0.14 IoU. What costs 1.64 is that one
protocol scores against a GT exported at 480² while the other scores against the 1408² GT
downscaled at evaluation time; rescaled to a common grid those two masks agree at mean IoU 0.9262,
and that residual is the gap. The same holds for TeSCo's predictions, where restricting to matched
rows moved IoU from 74.18 to 74.16.

Contour accuracy and location error barely move under either change (0.856 vs 0.8560, 0.0222 vs
0.0225 on matched TeSCo rows): the boundary F-measure's ~5 px tolerance absorbs a resampling
difference of about a pixel, and a centroid is robust to it. **IoU is the only one of the four
metrics sensitive to which GT export you score against.**

> `docs/ego2exo_gt_pred_metrics_howto.md` previously used 1024 × 1024 in its worked example.
> That was illustrative only and did not match this code or the benchmark; it now says 480.

---

## 2. Checkpoint ↔ direction: they are not interchangeable

This is the single easiest way to get a wrong number from this folder. **Do not pick the
checkpoint by recency.**

| Direction | Checkpoint | Full-coverage IoU | Paper |
|---|---|---|---|
| Exo→Ego | `training/checkpoints_EgoExo/tesco_egoexo4d_epoch59_for_inference.pt` | **70.48** | 70.30 |
| Ego→Exo | *not reproducible from this folder* — see below | — | 60.40 |

`tesco_egoexo4d_epoch59_for_inference.pt` was verified bit-identical to the checkpoint that
produced 70.48 (844/844 keys present in both, 400/400 sampled tensors exactly equal). **Exo→Ego
is fully reproducible from this folder.**

### The Ego→Exo gap — read this before trying to reproduce 60.40

The 60.40 / 61.48 Ego→Exo number was produced by a **different model variant than the one this
folder ships**, so the bundled code cannot load its weights. The two state dicts differ
structurally, not just numerically:

```
Ego→Exo checkpoint (842 params)        this folder's checkpoints (844 params)
  text_filter_gate.0.{weight,bias}       text_refine_attn2.in_proj_{weight,bias}
  text_filter_gate.2.{weight,bias}       text_refine_attn2.out_proj.{weight,bias}
                                         text_refine_ln2.{weight,bias}
```

That is a gating MLP on one side and a **second ITFR refinement block** on the other — two
architectures, 838 keys in common. Reproduce this with:

```bash
python tools/diff_checkpoints.py <ego2exo_checkpoint.pt> \
    training/checkpoints_EgoExo/tesco_egoexo4d_epoch59_for_inference.pt
```

Running this folder's `epoch59` on Ego→Exo gives **52.34**, not 60.40. Two explanations were
possible; the first has now been tested and ruled out.

**`epoch79` is not the Ego→Exo checkpoint — measured, not assumed.** Both bundled checkpoints
were run over the same 24 takes (seed 42 of the 290 test takes that have ego captions) and scored
on the same 23,363 rows with the same scorer:

| checkpoint | IoU | Cont.A | Loc.E |
|---|---|---|---|
| `tesco_egoexo4d_epoch79_for_inference.pt` | **10.45** | 0.133 | 0.1989 |
| `tesco_egoexo4d_epoch59_for_inference.pt` | **42.38** | 0.647 | 0.0282 |

epoch79 is four times worse, and all three metrics agree, so this is not one metric misbehaving.
Reproduce with `probe_ego2exo.sh`-style runs: inference restricted by `--video_list_file`, then
`merge_pred.py` → `merge_results.py` → `unified_eval.py --coverage pred`. Skip-missing is correct
*for a probe*, because it ranks checkpoints on an identical row set; the 24-take IoU is not an
estimate of the full-set number (epoch59 scores 42.38 here but 52.34 over all 291 takes, which is
ordinary subset variance at n=24).

That leaves the second explanation: **Ego→Exo needs the `text_filter_gate` variant's code, which
is not in this folder.** So **this folder reproduces Exo→Ego only.** Do not cite an Ego→Exo number
from these checkpoints — neither of them is the one behind the published 60.40.

---

## 3. Ground truth files

The scorer needs the correspondence test GT, one file per direction. They are not part of the
Ego-Exo4D release as shipped and must be placed at:

```
data/gt_results/egoexo_test_gt.json      # Ego→Exo
data/gt_results/exoego_test_gt.json      # Exo→Ego
```

Verified contents, so a substituted file can be checked:

| File | Task key | Takes | GT rows | Scoreable rows | RLE size |
|---|---|---|---|---|---|
| `egoexo_test_gt.json` | `ego-exo` | 291 | 163,947 | **132,503** | 480 × 480 |
| `exoego_test_gt.json` | `exo-ego` | 291 | 265,358 | **132,556** | 480 × 480 |

"Scoreable" means the target-view mask is non-empty. IoU, contour accuracy and location error are
defined only on those rows — the official convention, and what `unified_eval.py` counts. Existence
accuracy is the one metric that also uses empty-GT rows.

Check a GT file matches these counts:

```bash
python tools/check_gt.py data/gt_results/egoexo_test_gt.json ego-exo
```

---

## 4. DAM captions: `_exo` coverage is incomplete

`text_labels/` (2.9 GB, included) holds the DAM-3B per-frame captions the text branch consumes.
Coverage over the 291 test takes is **not uniform across directions**:

| Field | Used by | Takes covered |
|---|---|---|
| `dam_frame_captions_v2` | Ego→Exo | 290 / 291 |
| `dam_frame_captions_v2_exo` | Exo→Ego | **201 / 291** |

Take `22568049-4087-4ca2-8857-a834a9fc7f28` has no caption directory at all; 89 further takes
have the ego field but not `_exo`.

**This matters because running inference on a take with no captions silently produces a
confounded result** rather than an error — the text branch receives empty strings and the
prediction is neither the text-conditioned model's nor a clean ablation. A full-coverage Exo→Ego
run therefore needs the 90 missing takes' `_exo` captions generated first (DAM-3B, ~4 h on one
GPU for ~53 k captions). Verify coverage before any long run:

```bash
python tools/check_captions.py --split data_segswap_test --field dam_frame_captions_v2_exo
```

---

## 5. End-to-end: Exo→Ego, the reproducible path

```bash
conda activate tesco

# 1. Inference (see README "How to run: testing / inference" for the full flag list)
python correspondence_inference.py \
    --ckpt training/checkpoints_EgoExo/tesco_egoexo4d_epoch59_for_inference.pt \
    --exo_to_ego \
    --output outputs/exo2ego_epoch59

# 2. Merge per-take predictions into one JSON
python evaluation/merge_pred.py --pred outputs/exo2ego_epoch59 --gt <annotation json>
python evaluation/merge_results.py --pred outputs/exo2ego_epoch59 \
    --out outputs/exo2ego_epoch59/merged.json

# 3. Score, under both conventions
python evaluation/unified_eval.py \
    --gt   data/gt_results/exoego_test_gt.json \
    --pred outputs/exo2ego_epoch59/merged.json \
    --task exo-ego --coverage both \
    --out  outputs/exo2ego_epoch59/metrics.json
```

Expected: **IoU 70.48**, contour accuracy ≈ 0.85, location error ≈ 0.011, and `missing = 0`. A
non-zero `missing` means step 1 or 2 dropped takes and the number is not comparable.

---

## 6. Sanity-checking the scorer itself

Before trusting a score, check the scorer on an input whose answer is known: feed the GT back as
its own prediction, for a subset of takes. A perfect prediction must score perfectly, and the
full-coverage mean must be exactly the covered fraction.

```bash
python tools/make_selftest_pred.py \
    --gt data/gt_results/egoexo_test_gt.json --task ego-exo \
    --takes 20 --out /tmp/selftest_480.json

python evaluation/unified_eval.py \
    --gt data/gt_results/egoexo_test_gt.json \
    --pred /tmp/selftest_480.json --task ego-exo --coverage both
```

Measured, with 20 of 291 takes covered (8,215 of 132,503 scoreable rows):

| convention | IoU | Cont.A | Loc.E | Exist | scored | missing |
|---|---|---|---|---|---|---|
| skip missing | **100.00** | 1.000 | 0.0000 | 100.00 | 8,215 | 124,288 |
| full coverage | **6.20** | 0.062 | 0.1465 | 6.20 | 132,503 | 124,288 |

Both numbers are forced. The skip row must be a perfect 100.00 — anything less means the metric
plumbing is broken, not the model. The full-coverage row must equal the covered fraction:
8,215 / 132,503 = 6.1998%. This is also the clearest possible demonstration of what the two
conventions do to the same prediction file.

Passing `--upscale 700` writes the same masks at 700 x 700 instead, which exercises the resize
path. Measured: `resized` equals all 8,215 rows (the path runs), `Cont.A` stays at 1.000 and
`Loc.E` at 0.0020 — but IoU drops to **76.64**.

That 23-point drop is an artefact of the test, not a property of the benchmark, and it is worth
understanding before anyone cites it. The helper encodes a 480² GT mask up to 700² and
`unified_eval.py` brings it back down, so the mask makes a **round trip** at a non-integer scale
(700/480 ≈ 1.458). The two nearest-neighbour sampling grids do not align, every boundary shifts
by roughly a pixel, and across many small objects that is worth tens of IoU points. Contour
accuracy is unaffected because the DAVIS boundary F-measure applies a tolerance of about 0.008 of
the image diagonal (≈5 px at 480²), which absorbs a one-pixel shift entirely.

**A real method pays one resize, not two.** O-MaMa predicts natively at 700² and is downscaled
once, which costs it about 1.3 IoU points (§1), not 23. Use this run to confirm the resize path
executes and that `resized` matches the row count; do not read its IoU as any method's resize
cost.

---

## 7. How much do the protocol choices actually matter?

Four things can be varied without touching the model: which GT export you score against, which
side gets resized, which rows you average over, and — because this repo's method is a
memory-based tracker — how densely it is allowed to see frames. All four were measured on one
set of TeSCo Exo→Ego predictions (epoch59, 40 takes, the 3,439 rows both pipelines can score):

| condition | IoU |
|---|---|
| native frame sampling, 480² GT export, prediction native 480² | **74.16** |
| frames pinned to O-MaMa's sampling (`--restrict_frames_json`) | **73.60** |
| 1408² GT used natively, prediction upsampled to 1408² | **73.48** |
| 1408² GT downscaled to 480² (O-MaMa's own scorer) | **72.92** |

**Every one of these choices is worthabout 1 IoU point or less, and the full spread is 1.24.**
O-MaMa's predictions score 42.67–44.31 across the same kinds of variation. The gap between the
two methods is roughly 30 points, an order of magnitude larger than anything the protocol moves,
so none of these decisions changes the ranking — but all of them have to be stated, because a
1-point claim cannot survive them.

### Frame density is not where TeSCo's margin comes from

This repo predicts on every frame where the object is annotated in the source view — usually
stride 30 — while O-MaMa's ground truth is annotated every 120. A tracker with a memory bank gets
easier propagation from closely spaced frames; a per-frame matcher such as O-MaMa gains nothing
from them, so the two are not solving an equally hard version of the task. `--restrict_frames_json`
pins the frame list that the model both *sees* and predicts, which turns that into a measurement
rather than an argument: **pinning TeSCo to O-MaMa's sampling costs 0.56 IoU** (74.16 → 73.60, and
the run takes 8 minutes instead of 31). Temporal density is a real advantage and a small one.

Note what the frame lists are *not*: TeSCo's are neither a superset nor a clean subsample of
O-MaMa's. Nominal strides are 30 and 120, so TeSCo's ought to contain O-MaMa's, but only 3,439 of
the 6,916 rows O-MaMa's scorer wants are present — 2,917 of the misses fall *inside* TeSCo's frame
range. The reason is that TeSCo's list comes from `annotation.json`'s per-object annotated frames,
which is sparse and irregular (one triple had 257 frames where a dense stride-30 grid over its
range would hold 604). So "just subsample TeSCo to the coarser stride" is not available; the
intersection is.

### The resize direction is not neutral between methods

O-MaMa's scorer resizes the **ground truth to the prediction's size**, so the GT precision a method
is judged against depends on that method's own output resolution: O-MaMa's 700² predictions are
compared against a 2×-downscaled GT, this repo's 480² ones against a ~3×-downscaled GT. That
systematically favours whichever method emits larger masks, independently of whether it is better.
`unified_eval.py` resizes the other way — predictions onto a fixed GT grid — so every method is
judged against the same ground truth. Prefer that for cross-method tables, and say which GT export
it is.

---

## 8. Cross-method comparison table

All rows below were scored with `unified_eval.py` defaults — same GT file per direction, same
480² nearest grid, full coverage, missing predictions scored 0. The only variable is the
predictions.

| Method | Direction | Paper | Reproduced here |
|---|---|---|---|
| TeSCo | Exo→Ego | 70.30 | **70.48** |
| TeSCo | Ego→Exo | 60.40 | 60.01 (official script, 132,591 rows) — see §2 for why this folder does not reproduce it |
| O-MaMa | Exo→Ego | 44.10 | **42.81** (nearest resize; ≈43.7 with smooth resize — see §1) |

The Ego→Exo 60.01 came from running the official scorer on predictions from the
`text_filter_gate` variant, not from this folder's checkpoints.
