# Two metric protocols, and our results under both

Published Ego-Exo4D correspondence numbers are not all produced by the same scorer. Two drivers
are in circulation, they disagree by over an IoU point on identical predictions, and papers cite
whichever their own lineage ships. This repository therefore carries **both**, so our results can
be read on either.

**Short answer.** The paper states that it uses the metrics "provided in" the Ego-Exo4D
benchmark, i.e. the official protocol, and does not single out a driver. That is unambiguous for
our own rows because our predictions cover the test split: full coverage and the
prediction-driven variant this codebase inherits agree to **0.21 IoU** (70.48 vs 70.69 on
Exo→Ego), so the paper's numbers are the same either way. Under the **O-MaMa protocol** the same
predictions score about 1.2 IoU lower. Neither is wrong; they are different conventions, and the
difference is small next to the ~18–26 IoU gap between methods in Table 1 — but it is larger than
many of the margins people quote, so it has to be stated.

---

## The two protocols

| | **LM-EEC protocol** | **O-MaMa protocol** |
|---|---|---|
| in this repo | `evaluation/compute_metrics.py` | `evaluation/omama_eval.py` |
| origin | LM-EEC (NeurIPS'25); TeSCo inherits it | O-MaMa (ICCV'25) |
| outer loop | the **prediction** file | the **ground truth** |
| row filter | none | keeps only the target camera's `annotated_frames` |
| row with no prediction | omitted from the mean | *(their code reuses the previous row's mask; ours refuses — see below)* |
| resizing | none; both sides as stored | **ground truth → the prediction's size** |
| GT file | 480×480 export (`data/gt_results/*_test_gt.json`) | native-resolution export (1408² ego, 3840×2160 exo) |
| existence | balanced accuracy over all visited rows | balanced accuracy over all visited rows |

The metric functions are the same in both — `eval_mask`, `location_score`,
`existence_accuracy`, and under them `db_eval_iou`, `db_eval_boundary`, `f_measure`, `crop_mask`
are bit-identical between the two projects (verified by AST comparison; the only textual
differences are an apostrophe in an assert message and the indentation of a `return` whose two
branches both return). **The drivers are what move the number, not the maths.**

### Three consequences worth knowing

**The row sets differ by a factor of two.** O-MaMa's driver iterates the source camera's GT
frames but filters them by the *target* camera's `annotated_frames` list, and since
`annotated_frames` carries an entry for both cameras it reaches 66,339 rows where a full-coverage
pass scores 132,556. Measured, averaging over that half instead of all of it is worth only
**−0.14 IoU**.

**The resize direction is not neutral between methods.** O-MaMa's driver moves the GT onto the
prediction's grid, so the GT precision a method is judged against depends on its own output
resolution: O-MaMa's 700² predictions are compared against a 2×-downscaled GT, this repo's 480²
predictions against a ~3×-downscaled one. Combined with the GT export difference, that is worth
**+1.64 IoU** — the dominant term. The two GT exports agree at mean IoU 0.9262 once rescaled onto
a common grid, and that residual is the gap.

**Both original drivers share a latent bug.** LM-EEC's and O-MaMa's both contain

```python
try:    pred_mask = mask_utils.decode(pred[frame_idx]["pred_mask"])
except: breakpoint()
```

With breakpoints disabled this falls through and the next lines score the **previous row's mask**.
`compute_metrics.py` here already replaces it with a warning and `continue` (so a failed row is
dropped rather than mis-scored), and `omama_eval.py` goes further: it verifies coverage up front
and **refuses to score** rather than silently reusing a mask, with `--allow_missing` to score the
overlap deliberately.

### Is `compute_metrics.py` still the LM-EEC metric?

Yes. It differs from LM-EEC's file in three ways, all additive: optional
`existence_conf_thresh` / `pred_exists_min_area_frac` arguments (whose defaults reproduce
LM-EEC's `int(confidence > 0.5)` exactly), the `except: breakpoint()` replacement described
above, and an explicit `--task-key` in place of LM-EEC's auto-detecting `get_task_data`. The
scoring loop is untouched.

Only the second can change a number, and only when a prediction fails to decode: LM-EEC scores
the stale mask, this version drops the row from both the IoU and the existence totals. Run
side by side on the same predictions, the two agree to the last digit:

| | LM-EEC's `compute_metrics.py` | this repo's |
|---|---|---|
| IoU | 0.7417545410142025 | 0.7417545410142025 |
| Cont.A | 0.8547713787705662 | 0.8547713787705662 |
| Loc.E | 0.022154000734943623 | 0.022154000734943623 |
| Existence (balanced) | 0.5418109668109669 | 0.5418109668109669 |

Identical, which also confirms that no prediction in that file failed to decode — the one path
on which the two implementations could diverge was never taken. `unified_eval.py --coverage pred`
independently reproduces the same IoU (74.18) from a separate implementation.

---

## Our results under both

### Like-for-like, same predictions, same rows

TeSCo Exo→Ego, `tesco_egoexo4d_epoch59_for_inference.pt`, 40 takes, restricted to the 3,439 rows
both protocols can score (the two pipelines enumerate different frames — see below):

| protocol | IoU | Cont.A | Loc.E |
|---|---|---|---|
| **LM-EEC** | **74.16** | 0.856 | 0.0222 |
| **O-MaMa** | **72.92** | 0.8560 | 0.0225 |

Only IoU moves. Contour accuracy and location error agree to three decimals, because the boundary
F-measure's ~5 px tolerance absorbs a resampling difference of about a pixel and a centroid is
robust to it. **IoU is the only one of the metrics sensitive to which protocol you use.**

Scoring at the GT's native resolution instead — upsampling the prediction to 1408² so O-MaMa's
driver has nothing to downscale — recovers about half the gap, at **73.48**.

### Full test split

| Direction | IoU | Paper Table 1 |
|---|---|---|
| Exo→Ego | **70.69** | 70.30 |
| Ego→Exo | **60.01** | 60.40 |

A full-split O-MaMa-protocol number is not available for TeSCo, and the reason is structural
rather than a missing run: O-MaMa's GT is annotated every 120 frames while this repo predicts on
every frame where the object is annotated in the source view (nominal stride 30, but sparse and
irregular — one triple had 257 frames where a dense stride-30 grid over its range would hold 604).
Only 3,439 of the 6,916 rows O-MaMa's driver wants are present, and 2,917 of the misses fall
*inside* the predicted frame range, so subsampling to the coarser stride is not available either.
The intersection is what can be scored, which is what the table above reports.

### Does the frame density explain our margin? No — 0.56 IoU

A memory-based tracker gets easier propagation from closely spaced frames, while a per-frame
matcher such as O-MaMa gains nothing, so denser sampling is a real advantage. To size it,
`tools/correspondence_inference.py --restrict_frames_json` pins the frame list the model both
*sees* and predicts. Pinned to O-MaMa's sampling, TeSCo scores **73.60** against **74.16**
unrestricted: **0.56 IoU**, with the run taking 8 minutes instead of 31.

### Everything together

| condition | IoU |
|---|---|
| LM-EEC protocol, native frames | **74.16** |
| LM-EEC protocol, frames pinned to O-MaMa's sampling | 73.60 |
| O-MaMa protocol, prediction upsampled to the GT's 1408² | 73.48 |
| O-MaMa protocol, as its own code scores it | **72.92** |

The full spread is **1.24 IoU**. Table 1's gap between TeSCo and O-MaMa is 17.8 on Ego→Exo
(60.40 vs 42.60) and 26.2 on Exo→Ego (70.30 vs 44.10), so no protocol choice here changes a
ranking — but a one- or two-point claim cannot survive them, which is why the protocol has to be
stated rather than assumed.

---

## Which protocol produced which published number

We re-measured two rows of Table 1 and can state their protocol with evidence; the rest are the
figures reported in their own papers. See [`RESULTS.md`](RESULTS.md) for the full provenance.

* **O-MaMa's published 44.10 / 0.524 / 0.082 (Exo→Ego) is on the O-MaMa protocol.** Scoring its
  released predictions with its own driver gives 44.31 / 0.5150 / 0.0821 — Loc.E identical, IoU
  within 0.21. The same predictions score **42.81** under a full-coverage 480² pass.
* **Our Table 1 numbers follow the official Ego-Exo4D metrics**, which is what the paper states.
  Because our predictions cover the test split, the prediction-driven scorer this codebase
  inherits from LM-EEC gives the same answer to within 0.21 IoU, so the distinction does not
  change our row.

So Table 1 compares a TeSCo row on one protocol against an O-MaMa row on another. The difference
is ~1.3 IoU and does not affect the ordering, but it is real and is recorded here rather than
left for a reader to discover.

## Running both

```bash
# LM-EEC protocol (this codebase's inherited scorer)
python evaluation/compute_metrics.py \
    --gt-file   data/gt_results/exoego_test_gt.json \
    --pred-file outputs/<run>/merged.json --task-key exo-ego

# O-MaMa protocol (needs their native-resolution GT export)
python evaluation/omama_eval.py \
    --gt   /path/to/exo2egoGT.json \
    --pred outputs/<run>/merged.json --task exo-ego

# Official benchmark protocol: GT-driven, full coverage, both sides at 480x480.
# Use this one for cross-method tables -- it is the only one of the three that
# neither rewards abstaining nor varies the GT with the prediction's resolution.
python evaluation/unified_eval.py \
    --gt   data/gt_results/exoego_test_gt.json \
    --pred outputs/<run>/merged.json --task exo-ego --coverage both
```
