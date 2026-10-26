# Table 1: where every number comes from, and what metric produced it

The paper's comparison table follows the convention established by
**V²-SAM** [CVPR'26] — the paper states this ("Following [V²-SAM], we use IoU, Loc.E and Cont.A
provided in [Ego-Exo4D]"), and we traced each row to confirm it. This file records that trace,
and reports our own method under every scorer we could run, so that a reader comparing against
us knows exactly which number to compare with.

**The short version.** Seven of the ten comparison rows are quoted verbatim from V²-SAM's Table 1,
which in turn quotes five of them from O-MaMa's Table 1. The chain is internally consistent, and
our rows sit on the same convention. The one thing worth knowing is that the convention is not
the full-coverage official one: O-MaMa's published 44.10 was produced by O-MaMa's own scorer, and
its released predictions score **42.81** under a full-coverage 480×480 pass. That difference is
inherited through the whole citation chain, not introduced here.

---

## 1. Table 1 as published

| Method | Ego→Exo IoU | Cont.A | Loc.E | Exo→Ego IoU | Cont.A | Loc.E | mIoU |
|---|---|---|---|---|---|---|---|
| XSegTx [CVPR'24] | 18.90 | 0.386 | 0.070 | 27.10 | 0.358 | 0.104 | 23.00 |
| XMem [CVPR'24] | 19.30 | 0.262 | 0.151 | 16.60 | 0.240 | 0.160 | 17.95 |
| XMem+XSegTx [CVPR'24] | 34.90 | 0.559 | 0.038 | 25.00 | 0.237 | 0.117 | 29.95 |
| PCC [CVPR'25] | 39.01 | 0.600 | 0.033 | 47.06 | 0.590 | 0.054 | 43.04 |
| ObjectRelator [ICCV'25] | 35.30 | 0.540 | 0.036 | 40.30 | 0.500 | 0.068 | 37.80 |
| O-MaMa (k-NN) [ICCV'25] | 31.90 | 0.414 | 0.195 | 30.90 | 0.373 | 0.127 | 31.40 |
| O-MaMa [ICCV'25] | 42.60 | 0.590 | 0.033 | 44.10 | 0.524 | 0.082 | 43.35 |
| CCMP [CVPR'26] | 41.95 | 0.669 | 0.038 | 47.18 | 0.591 | 0.081 | 44.57 |
| V²-SAM [CVPR'26] | 46.30 | 0.616 | 0.056 | 49.60 | 0.575 | 0.078 | 47.95 |
| LM-EEC [NeurIPS'25] | 54.98 | 0.778 | 0.017 | 65.77 | 0.774 | 0.031 | 60.38 |
| Baseline (ours) | 52.70 | 0.719 | 0.027 | 58.30 | 0.664 | 0.043 | 55.50 |
| **TeSCo (ours)** | **60.40** | **0.847** | **0.011** | **70.30** | **0.808** | **0.023** | **65.35** |

## 2. Provenance, row by row

Each comparison row was checked against the source paper's own table. "exact" means every value
matched digit for digit, including the column reordering (V²-SAM prints IoU / Cont.A / Loc.E,
O-MaMa prints IoU / Loc.E / Cont.A).

| Row | Traced to | Match |
|---|---|---|
| XSegTx | V²-SAM Tab. 1 (itself quoting O-MaMa Tab. 1) | **exact** |
| XMem | V²-SAM Tab. 1 (quoting O-MaMa) | **exact** |
| XMem+XSegTx | V²-SAM Tab. 1 (quoting O-MaMa) | **exact** |
| PCC | *not located* — absent from the V²-SAM, CCMP and LM-EEC tables; the CVF PDF is not fetchable | — |
| ObjectRelator | **V²-SAM Tab. 1** | **exact** |
| O-MaMa (k-NN) | V²-SAM Tab. 1 (quoting O-MaMa) | **exact** |
| O-MaMa | V²-SAM Tab. 1 (quoting O-MaMa) | **exact** |
| CCMP | CCMP Tab. 1 | **exact** |
| V²-SAM | V²-SAM Tab. 1, Multi-Experts row | **exact** |
| LM-EEC | LM-EEC Tab. 1 | **exact** |

Three things this trace turned up, recorded because they are not obvious from the table:

**ObjectRelator's row is not ObjectRelator's own number.** Its paper reports the *validation*
split only (Ego→Exo 45.4, Exo→Ego 50.9) and states that "the Test set lacks ground truth". The
35.30 / 40.30 quoted here is V²-SAM's measurement of it on test.

**The same method differs between papers.** O-MaMa is 42.6 / 44.1 in its own paper and in
V²-SAM's, but 42.57 / 44.08 in CCMP's; XSegTx is 18.9 in V²-SAM's and 18.99 in LM-EEC's and
CCMP's; XMem+XSegTx's Exo→Ego Cont.A is 0.237 in V²-SAM's and 0.327 in CCMP's. Table 1 takes
V²-SAM's values throughout, so it is internally consistent.

**V²-SAM's released code does not contain the scorer behind its own Table 1.** The repository
ships one evaluator, `projects/v2sam_*/evaluation/seg_metric.py`, and its configs use it
(`val_evaluator = dict(type='SegMetric', iou_metrics=['IoU', 'Dice'])`). It computes a flat mean
of `inter / (union + 1e-6)` and Dice, and **nothing else** — there is no contour or location
metric anywhere in the repository, so the Cont.A and Loc.E columns were produced by code that was
not released.

---

## 3. Our method under every scorer we could run

One set of TeSCo Exo→Ego predictions (`tesco_egoexo4d_epoch59_for_inference.pt`), scored five
ways. Only the scorer changes.

| Scorer | Where it comes from | What it does differently | IoU |
|---|---|---|---|
| **V²-SAM** `SegMetric` | V²-SAM repo, the only evaluator it ships | flat `inter/(union+1e-6)`; no resize; does **not** skip empty-GT rows | **74.14** |
| V²-SAM `SegMetric`, empty GT excluded | same formula, official row set | isolates the empty-GT choice above | 74.18 |
| **LM-EEC** `compute_metrics.py` | this repo's inherited scorer | walks the prediction file; omits rows you did not predict | **74.18** |
| **Official benchmark** `unified_eval.py` | `XMem/scripts/evaluation.py` in the Ego-Exo4D release | walks the GT; a missing prediction scores 0; both sides on a 480×480 grid | **74.18** |
| **O-MaMa** `omama_eval.py` | O-MaMa repo's `evaluate_exoego.py` | walks the GT, filters by `annotated_frames`, resizes the **GT** onto the prediction's grid, uses the 1408² GT export | **72.92** |

The first four agree to within 0.04 IoU. They differ only in ways that do not bite on this data:
TeSCo predicts at the GT's own 480×480 so no resize happens, the `1e-6` is negligible, and only
**6 of 13,866** rows have an empty ground truth, so V²-SAM's choice of scoring those 0 — even when
the model correctly predicts nothing — costs just 0.04.

**O-MaMa's scorer is the one that differs**, by 1.24 IoU, and we measured why rather than
guessing. Restricted to the 3,439 rows both pipelines can score:

| condition | IoU |
|---|---|
| official / LM-EEC convention, 480² GT export | **74.16** |
| frames pinned to O-MaMa's sampling (`--restrict_frames_json`) | 73.60 |
| 1408² GT used natively, prediction upsampled to 1408² | 73.48 |
| O-MaMa's scorer as its own code runs it | **72.92** |

and decomposing that gap on O-MaMa's own predictions:

| cause | effect on IoU |
|---|---|
| averaging over O-MaMa's row subset instead of all GT rows | **−0.14** |
| ground-truth export (480² vs 1408²) and resize direction | **+1.64** |

It is the ground truth's provenance, not the sample, that moves the number. Contour accuracy and
location error barely move under any of this (0.856 vs 0.8560, 0.0222 vs 0.0225): the boundary
F-measure's ~5 px tolerance absorbs a resampling difference of about a pixel, and a centroid is
robust to it. **IoU is the only one of the four metrics sensitive to which scorer you use.**

## 4. Full test split

| Direction | Official / full coverage | Prediction-driven (skip missing) | Paper |
|---|---|---|---|
| Exo→Ego | **70.48** | **70.69** | 70.30 |
| Ego→Exo | **60.01** (Cont.A 0.853, Loc.E 0.0118) | — | 60.40 (0.847, 0.011) |

Exo→Ego was scored from a checkpoint verified bit-identical to the one behind the published
number (844/844 keys, 400/400 sampled tensors equal). Ego→Exo 60.01 came from running the
official benchmark scorer end to end over 132,591 rows.

> **The Ego→Exo direction is not reproducible from this repository.** Neither bundled checkpoint
> is the one behind the published 60.40: `epoch59` scores 52.34 there and `epoch79` scores 10.45
> against `epoch59`'s 42.38 on a shared 24-take probe. The published weights belong to a
> different model variant — a `text_filter_gate` MLP where this code has a second ITFR block —
> that this code cannot load. See [`REPRODUCE.md`](REPRODUCE.md) §2.

## 5. If you are comparing against us

Quote our Table 1 row if you are building a table on the same convention everyone else uses.
If you want a number you can regenerate yourself with a scorer in this repository, use

```bash
python evaluation/unified_eval.py \
    --gt   data/gt_results/exoego_test_gt.json \
    --pred <your predictions>.json \
    --task exo-ego --coverage both
```

and compare against **70.48 (Exo→Ego)** and **60.01 (Ego→Exo)**. Report `--coverage both`, so
that an incomplete prediction file shows up instead of being silently rewarded.

Sources for the trace: [O-MaMa](https://arxiv.org/html/2506.06026v2),
[ObjectRelator](https://arxiv.org/html/2411.19083v2), [LM-EEC](https://arxiv.org/html/2510.11417v1),
[V²-SAM](https://arxiv.org/html/2511.20886v2) and [its code](https://github.com/jaychempan/V2-SAM),
[CCMP](https://arxiv.org/html/2602.18996v2).
