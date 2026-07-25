# Ego2exo GT + prediction: how to compute IoU and the other three metrics

This document is a **reusable recipe** for evaluating ego-to-exo (or the same structure) when:

- **Ground truth** comes from the V2-SAM–style file `ego2exo_test.json` (Ego-Exo4D relation test, ego2exo split).
- **Predictions** come from a directory tree of per-frame masks (RLE in JSON and/or binary PNG) produced by LM-EEC (or a compatible layout).

The **matching rule** below is the one that was validated to give correct IoU. Earlier bugs were caused by using the wrong object name field in the GT JSON, or by matching on normalized names without the instance suffix, which pairs the wrong pred folder to the wrong GT object.

---

## 1) GT file: `ego2exo_test.json`

- **Path example (this project):**  
  `V2-SAM/.../Ego-Exo4D-Relation-Test/ego2exo_test.json`
- **Top level:** a dictionary keyed by a **sample id** (string, often `{take_uuid}_{frame_id}`). Each value is one benchmark row: one (ego source, exo target, frame) with one or more objects.

### 1.1 Per-sample fields you need

| Role | JSON path | Use |
|------|-----------|-----|
| **Take (UUID)** | `video_path[0]` → first folder in path, e.g. `4764e94b-.../cam04/2250.jpg` | `take` = `4764e94b-...` |
| **Target (exo) view** | `video_path[0]` → second folder | `target_view` = e.g. `cam04` |
| **Frame id** | `video_path[0]` → filename without extension | `frame_id` = e.g. `2250` (string) |
| **Source (ego) view** | `prompt.first_frame_image` → second folder in path | `source_view` = e.g. `aria02_214-1` |
| **Object name for matching** | `prompt.first_frame_anns` → **only** the `.text` field for each object id | **This is the folder name in predictions.** Do not use `objects[*].formated` or other keys for matching. |
| **GT mask (RLE) for that object** | `objects` → same string key as in `first_frame_anns` → `segmentation` | COCO RLE: `{ "size": [H, W], "counts": "..." }` |

**Important:**

- The **index between prompt and objects must align**: for object id `"0"`, the prompt text is `prompt.first_frame_anns["0"].text` and the GT exo mask is `objects["0"].segmentation` (or integer keys serialized as string in JSON).
- The segmentation under `prompt.first_frame_anns[*].segmentation` is the **input/prompt** side (aria); evaluation IoU is against the **target exo** mask in `objects[*].segmentation` for the same id.

### 1.2 Building the evaluation key

For each (sample × object id) pair you will evaluate, store:

- `(take_id, target_view, source_view, frame_id, object_id, object_name_from_prompt_text)`

`object_name_from_prompt_text` must match prediction directories **exactly** (including instance suffixes like `white bowl_0`).

---

## 2) Prediction layout (LM-EEC default)

Under a **prediction root** directory:

```text
{take_id}/
  {target_view}__{source_view}/
    {object_name_exactly_as_prompt_text}/
      {frame_id}.png
```

Some runs may use the reverse order; accept either:

- `{target_view}__{source_view}` **or** `{source_view}__{target_view}`

**Lookup algorithm:**

1. `take_dir = pred_root / take_id` — must exist.
2. `cam_pair_dir = take_dir / f"{target_view}__{source_view}"` if it exists, else `take_dir / f"{source_view}__{target_view}"`.
3. `object_dir = cam_pair_dir / object_name` — the string from `first_frame_anns[id].text` only.
4. `mask_file = object_dir / f"{frame_id}.png"` (or `.json` with RLE if you did not convert to PNG).

If any step is missing, that GT row has **no prediction** for that run (count as skip or 0 match depending on your protocol).

---

## 3) Decoding masks

### 3.1 GT (RLE)

- Decode with COCO RLE, e.g. `pycocotools.mask.decode(segmentation)`.
- Result is `H x W` (or `H x W x 1`); binarize: foreground `> 0`.

### 3.2 Prediction (PNG)

- Read grayscale or RGB; binarize: `> 0` for foreground.

### 3.3 Optional: RLE in prediction JSON

- If the file is a dict with `pred_mask` nested, unwrap to the inner RLE before decode (same as official merge JSON in LM-EEC).

---

## 4) Same resolution before metrics (critical)

Native GT and pred can differ in **height × width**. For comparable IoU and for boundary metrics, **resize both** to a common size with **nearest neighbor** (preserves {0,1} masks):

- Example: **1024 × 1024** (used in the V2-SAM–style and full-dataset reports in this project).

Pseudocode:

```text
G = resize_binary(gt_rle_decoded, (1024, 1024), NEAREST)
P = resize_binary(pred_png,     (1024, 1024), NEAREST)
```

Then compute all four per-sample metrics on `G` and `P` of the same shape.

---

## 5) IoU (Intersection over Union)

On equal-shaped binary `G` and `P` (bool or 0/1):

- `intersection = (G & P).sum()`
- `union = (G | P).sum()`
- `IoU = intersection / (union + eps)`  
- Convention used in `evaluation/metrics.py` `db_eval_iou`: if `union == 0`, set IoU to **1** (both empty).

**Dataset mean IoU:** average IoU over all **successfully matched** (GT row + pred file) samples you include.

**Reference:** `evaluation/metrics.py` → `db_eval_iou`

---

## 6) The other three metrics (same G, P as IoU)

These definitions match **this project**’s `evaluation/utils.py` + `evaluation/metrics.py`. Import path when running from repo root: add `evaluation` to `sys.path` and `import utils` (which wraps `metrics`).

### 6.1 Shape ACC (boundary F-measure)

- Implemented as **boundary F-measure** (not a generic “shape” classifier).
- **API:** `iou, shape_acc = utils.eval_mask(G, P)` where `G`, `P` are 2D binary arrays (same as IoU).
- **Internally:** `metrics.db_eval_boundary(G, P)` with default `bound_th=0.008` (relative to image diagonal in `f_measure`).

**Higher is better** (1 = perfect boundary agreement under the F-measure).

**Dependencies:** `scikit-image` (for `disk` structuring element) is required by the boundary code path.

**Reference:** `evaluation/utils.py` → `eval_mask`; `evaluation/metrics.py` → `db_eval_boundary`, `f_measure`

### 6.2 Location Score

- **API:** `location_score = utils.location_score(G, P, size=(H, W))`  
- Uses centroids from `metrics.crop_mask` / `compute_size_boundry_centroid`, then:
  - `d = euclidean(centroid_gt, centroid_pred)`
  - `LocationScore = d / sqrt(H^2 + W^2)` with `H, W` from the `size=` argument.
- If you evaluated at **1024×1024**, pass `size=(1024, 1024)`.

**Lower is better** (0 = same centroid).

**Reference:** `evaluation/utils.py` → `location_score`; `evaluation/metrics.py` → `crop_mask`, `compute_size_boundry_centroid`

### 6.3 Existence ACC (per-sample)

- **API:** `existence_accuracy(G, P)` in `utils`.
- `True` if both masks are all-zero **or** both have at least one foreground pixel; `False` if one is empty and the other is not.
- This is a **0/1** score per frame/object; **mean** over samples is reported as “Existence ACC” in the custom full-dataset script.

**Reference:** `evaluation/utils.py` → `existence_accuracy`

### 6.4 Optional: balanced “existence” over the dataset

The official `compute_metrics.py` also reports **balanced accuracy** on binary existence flags (GT present vs pred present), using `sklearn.metrics.balanced_accuracy_score` over all (frame, object) rows. That is a **different number** from mean per-sample existence agreement on masks; you can report both with clear names.

**Reference:** `evaluation/compute_metrics.py` → `evaluate`, `evaluate_take`

---

## 7) End-to-end algorithm (pseudocode)

For each top-level entry in `ego2exo_test.json`:

1. Parse `take_id`, `target_view`, `source_view`, `frame_id` (Section 1).
2. For each `object_id` in `objects`:
   - `name = prompt["first_frame_anns"][str(object_id)]["text"]`
   - `gt_rle = objects[str(object_id)]["segmentation"]`
3. Load prediction at `pred_root / take / cam_pair / name / {frame_id}.png` (try both cam_pair orders).
4. Decode `gt_rle` and pred to binary; resize both to e.g. 1024×1024.
5. Compute: IoU, `eval_mask` (IoU + Shape), `existence_accuracy`, `location_score` (with matching `size=`).
6. Aggregate means over all valid pairs.

---

## 8) Pitfalls that previously broke IoU

1. **Wrong object key:** using `objects[*].formated` (or any field other than `prompt.first_frame_anns[*].text`) to find the pred folder name.
2. **Stripping instance ids:** e.g. mapping `basketball_0` and `basketball_1` to the same string and then picking the wrong folder.
3. **Evaluating on different frame sets** than the GT file (e.g. full video vs benchmark rows only) without stating it.
4. **Resizing GT and pred with different rules** (e.g. one bilinear, one nearest) or comparing without resizing.

---

## 9) Files in this repo for reference

| What | Path |
|------|------|
| IoU, boundary, centroid helpers | `evaluation/metrics.py` |
| `eval_mask`, `location_score`, `existence_accuracy` | `evaluation/utils.py` |
| Full metrics on official merged JSON (ego-exo format) | `evaluation/compute_metrics.py` |
| Per-scenario breakdown | `evaluation/compute_metrics_scenarios.py` |

---

*This document was written to copy into other codebases: keep the **matching** section (1–2) and **resolution** (4) in sync with your GT and pred formats.*
