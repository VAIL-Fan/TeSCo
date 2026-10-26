# Table 1, row by row: where each number comes from

Table 1 of the paper compares TeSCo against ten prior methods on the Ego-Exo4D correspondence
test split. This file records, for every row, **which evaluation setting the number corresponds
to and whether we verified it ourselves** — because the rows do not all come from the same
protocol, and a reader comparing against us should know which ones we re-measured and which we
cite.

Settings referred to below are defined in [`METRIC_SETTINGS.md`](METRIC_SETTINGS.md):
**Setting A** is full coverage, GT-driven, both sides resized to 480² (the official benchmark
convention); **Setting B** skips rows a method did not predict.

---

## Table 1 as published

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

## Provenance

| Method | Source of the number | Setting | Code available to us | Verified by re-running? |
|---|---|---|---|---|
| XSegTx | Ego-Exo4D benchmark paper | official (A-like) | scorer only | no |
| XMem | Ego-Exo4D benchmark paper | official (A-like) | scorer only | no |
| XMem+XSegTx | Ego-Exo4D benchmark paper | official (A-like) | scorer only | no |
| PCC | as published | not verified | no | no |
| ObjectRelator | as published | not verified | yes (not run) | no |
| O-MaMa (k-NN) | as published | O-MaMa's own scorer | yes | no |
| **O-MaMa** | as published | **O-MaMa's own scorer — confirmed** | yes | **yes, see below** |
| CCMP | as published | not verified | no | no |
| V²-SAM | as published | not verified | no | no |
| LM-EEC | as published | not verified | yes (not run) | no |
| Baseline (ours) | our run | A | yes | — |
| **TeSCo (ours)** | our run | **A** | yes | **yes, see below** |

**Only the two rows marked "yes" have been re-measured by us.** The rest are the numbers reported
in the cited papers, reproduced as published. That is normal practice for a comparison table, but
it means Table 1 is not guaranteed to be internally uniform in protocol — and for O-MaMa we were
able to show that it is not (below). We state this rather than leave it implicit.

---

## O-MaMa: verified, and it is *not* Setting A

We obtained O-MaMa's released Exo→Ego predictions (265,358 rows, complete coverage of the GT) and
scored them two ways:

| scored with | IoU | Cont.A / ShapeAcc | Loc.E |
|---|---|---|---|
| **O-MaMa's own `evaluate_exoego.py`** | **44.31** | 0.5150 | 0.0821 |
| `unified_eval.py` (Setting A) | 42.81 | 0.522 | 0.0819 |
| `unified_eval.py`, restricted to O-MaMa's row set | 42.67 | 0.521 | 0.0825 |

The published **44.10 / 0.524 / 0.082** matches O-MaMa's own scorer (44.31 / 0.5150 / 0.0821 —
Loc.E identical, IoU within 0.21), not Setting A. **So the O-MaMa row in Table 1 is on a different
protocol from the TeSCo row.** Under Setting A, O-MaMa's released predictions score **42.81**,
about 1.3 IoU below its published figure.

Why the two differ — measured, not estimated:

| cause | effect on IoU |
|---|---|
| averaging over O-MaMa's row subset instead of all GT rows | **−0.14** |
| ground-truth export and resize direction | **+1.64** |

Its scorer reaches 66,339 rows where Setting A scores 132,556, but averaging over half the sample
is worth only 0.14 IoU. What costs 1.64 is that its scorer resizes the **1408² ground truth down
to the prediction's size**, while Setting A resizes predictions onto the fixed 480² grid of the
480² GT export; the two GT exports agree at mean IoU 0.9262 once rescaled, and that residual is
the gap. Full derivation in [`REPRODUCE.md`](REPRODUCE.md) §7.

Two incidental findings, recorded because anyone re-running O-MaMa will hit them: its scorer's
`validate_predictions` asserts equal per-take object *counts* and fails on its own released
predictions (they carry 6 extra objects across 5 of 291 takes; no take is missing one), and it
compares `confidence` to a float while the prediction files store it as a one-element list.
Neither affects IoU, contour accuracy or location error.

## TeSCo: verified

| Direction | Setting A | Setting B | Published |
|---|---|---|---|
| Exo→Ego | **70.48** | **70.69** | 70.30 |
| Ego→Exo | **60.01** (Cont.A 0.853, Loc.E 0.0118) | — | 60.40 (0.847, 0.011) |

Exo→Ego was scored from `tesco_egoexo4d_epoch59_for_inference.pt`, verified bit-identical to the
checkpoint that produced the published number (844/844 keys, 400/400 sampled tensors equal).
Ego→Exo 60.01 came from running the official benchmark scorer end to end over 132,591 rows.

The paper's numbers are **Setting A**. For TeSCo the choice barely matters — 70.48 vs 70.69,
because our predictions cover the test set — but it matters for the comparison, since at least one
competing row (O-MaMa) is on a different protocol.

> **The Ego→Exo direction is not reproducible from this repository.** Neither bundled checkpoint
> is the one behind the published 60.40: `epoch59` scores 52.34 there, and `epoch79` scores 10.45
> against `epoch59`'s 42.38 on a shared 24-take probe. The published Ego→Exo weights belong to a
> different model variant — a `text_filter_gate` MLP where this code has a second ITFR block
> (`text_refine_attn2`, `text_refine_ln2`) — which this code cannot load. See
> [`REPRODUCE.md`](REPRODUCE.md) §2.

## If you are comparing against us

Score your predictions with `evaluation/unified_eval.py` (Setting A) against
`data/gt_results/*_test_gt.json`, and report `--coverage both` so that incomplete coverage is
visible rather than silently rewarded:

```bash
python evaluation/unified_eval.py \
    --gt   data/gt_results/exoego_test_gt.json \
    --pred <your predictions>.json \
    --task exo-ego --coverage both
```

Compare against **70.48 (Exo→Ego)** and **60.01 (Ego→Exo)** — our Setting A numbers — rather than
against the paper's 70.30 / 60.40, if you want the comparison to be exactly like-for-like with a
scorer you can run yourself.
