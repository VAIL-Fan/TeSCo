# TeSCo: Architecture and Implementation Reference

A technical reference for the model architecture, hyperparameters, and reproducibility details
implemented in this repo (`sam2/modeling/sam2_base.py`, configs, training, inference). Where
comments elsewhere disagree with the code, **the repository code is ground truth**.

**Primary code references:** `sam2/modeling/sam2_base.py`, `training/model/sam2.py`, `sam2/sam2_correspondence_predictor.py`, `sam2/build_sam.py`, `sam2/configs/sam2.1/sam2.1_hiera_b+_egoexo_text.yaml`, `sam2/configs/sam2.1_training/sam2.1_hiera_b+_EgoExo_finetune.yaml`.

---

## 1. Problem and high-level goal

The system addresses **ego–exo visual correspondence** on EgoExo4D-style data: a **wearable (ego / Aria)** view and one or more **fixed (exo)** cameras observe the same scene. Given **per-frame natural-language descriptions** of a target object (from DAM-style captions) and **ego-side mask supervision** (and memory from past frames), the model predicts **segmentation masks on the exocentric view** while using a **shared** vision backbone and **long-term memory** from both views.

At a high level the implementation is a **text-conditioned extension of SAM 2.1** (Hiera-B+) with:

- Shared image encoder for ego and exo frames.
- **Dual memory cross-attention** on the exo query map: (a) current-frame ego mask memory, (b) fused temporal bank of past ego + exo mask memories (plus optional object-pointer tokens).
- **FiLM-style conditioning** of memory tokens with **exorefined** text features.
- **Optional second-stage fusion** (`use_text_cross_attn`): exo features attend to projected **ego feature map + broadcast text** via the same `MemoryAttention` module, with a residual; otherwise a simple **additive** merge of exo, ego, and text in feature space.

There is **no MoE** fusion in `sam2_base.py`; the two memory branches are combined by **element-wise addition** of their outputs in `hidden_dim` space before the frame stage.

---

## 2. Notation and tensor shapes

| Symbol | Meaning | Typical value (this repo) |
|--------|---------|---------------------------|
| \(B\) | Batch size | 1 in many inference paths; training `train_batch_size: 1` |
| \(T\) | Frames per training clip | `num_frames: 8` |
| `hidden_dim` / \(C\) | Neck / SAM embedding dim | **256** (`image_encoder.neck.d_model`) |
| `mem_dim` | Memory token channel dim after encoder compression | **64** (`memory_encoder.out_dim`); `text_proj_to_mem: Linear(256→64)` when dims differ |
| `film_text_tokens` | Learned “text token” count for FiLM MHA | **4** (fixed in `SAM2Base.__init__`) |
| `num_maskmem` | Memory bank slots (incl. conditioning / temporal design) | **7** |
| `image_size` | Input square resolution | **480** (training `scratch.resolution`) |

Token layout follows SAM2: backbone features are flattened to sequences `[HW, B, C]` for `MemoryAttention` and FiLM MHA (PyTorch MHA with `batch_first=False`).

---

## 3. Vision backbone: SAM2.1 Hiera-B+ and FPN

**Config:** `sam2.1_hiera_b+_egoexo_text.yaml` and training YAML.

- **Trunk:** `Hiera` with `embed_dim: 112`, `num_heads: 2`, `drop_path_rate: 0.1`, `scalp: 1` on the image encoder.
- **Neck:** `FpnNeck` with `d_model: 256`, `backbone_channel_list: [896, 448, 224, 112]`, `fpn_top_down_levels: [2, 3]`, `fpn_interp_model: nearest`.
- **Positional encoding:** `PositionEmbeddingSine` for neck features (`num_pos_feats: 256`, `normalize: true`, `temperature: 10000`).

Ego and exo frames are processed through the **same** `image_encoder` weights.

When `use_high_res_features_in_sam: true`, additional higher-resolution FPN levels feed the **SAM mask decoder** (three feature levels in `SAM2Base`).

**Backbone stride** for prompt/SAM head sizing defaults to **16** (`backbone_stride=16` in `SAM2Base`), consistent with the coarsest map used for the prompt grid.

---

## 4. Text encoding (frozen RoBERTa + projection)

**Implementation:** `SAM2Base._get_text_features`, `SAM2Base.__init__`.

- **Tokenizer / encoder:** `RobertaTokenizer` and `RobertaModel` from HuggingFace, default `text_encoder_name="roberta-base"`.
- All **RoBERTa parameters are frozen** (`requires_grad=False`).
- The **[CLS] token** last hidden state (dim **768**) is taken and passed through `text_proj: Linear(768, hidden_dim)` → **`text_feat` ∈ ℝ\(^{B \times 256}\)**.
- **Padding / truncation** use the tokenizer defaults in batch mode.
- If all strings are empty or the list is empty, **`text_feat` is all zeros**; downstream refinement and FiLM can become **no-ops** (refinement returns early if `text_feat` is all zero).

**Training:** The training wrapper can supply either:

- `frame_descriptions`: one string per frame → `text_feat_per_frame` shaped `[T, B, hidden_dim]`, or  
- a single `descriptions` list → legacy `text_feat` per object/batch.

See `training/model/sam2.py` `forward()`.

**Inference:** `SAM2VideoPredictor.init_state` loads per-frame strings from the take’s DAM caption directory (see §9). `object_descriptions.json` and similar are **not** used for the forward pass in the current pipeline (`load_object_description` in `tools/correspondence_inference.py` is explicitly deprecated and returns `""`).

---

## 5. Exo-conditioned text refinement (pre-FiLM)

**Implementation:** `_refine_text_with_exo_backbone(text_feat, exo_vision_tokens)`.

- **Input `exo_vision_tokens`:** the flattened exo view at the **neck resolution**, shape `[N, B, hidden_dim]`, i.e. the same token layout as the **query** `curr` passed to `memory_attention` for the exo stream.
- **Two stacked blocks** (not a gating “filter” in current code):
  1. `q = text_feat.unsqueeze(0)`; `MHA1(q, exo, exo)`; residual + `text_refine_ln` → `out1`.
  2. `q2 = out1.unsqueeze(0)`; `MHA2(q2, exo, exo)`; residual + `text_refine_ln2` → **`film_text_feat`** (output of block 2).

- Each `text_refine_attn` / `text_refine_attn2` is `nn.MultiheadAttention` with `embed_dim=hidden_dim`, **`num_heads=8`**, `dropout=0.0`, `batch_first=False`.
- If `text_feat` is all zeros, refinement **returns without change** (no visual conditioning from empty text).

**Purpose (for the paper):** ground the language embedding in the **current exo frame’s spatial features** before using it to modulate **memory** tokens, so text interacts with *what the exo camera sees now* in addition to lexical content.

---

## 6. Memory encoding (mask → memory tokens)

**Module:** `MemoryEncoder` (config: `out_dim: 64`, `MaskDownSampler` with `kernel_size: 3`, `stride: 2`, `padding: 1`, `Fuser` with 2× `CXBlock` layers, etc.).

**High-level flow (`_encode_new_memory`):**

1. Take top-level **pixel features** and **predicted or GT high-res mask** (logits or GT depending on stream).
2. Apply **sigmoid** to obtain soft masks; optional **scale/bias** (`sigmoid_scale_for_mem_enc: 20.0`, `sigmoid_bias_for_mem_enc: -10.0` in config) before feeding the memory encoder.
3. **Memory tokens** and positional encodings are produced at **`mem_dim`**.
4. If `no_obj_embed_spatial` is enabled, a learned spatial embedding is added when the object is predicted absent (from `object_score_logits` during exo memory encoding; ego bank uses a fixed positive score in code paths that force “object present” for ego mask memory when encoding from GT).

**Text-to-memory channel adapter:** `text_proj_to_mem` is `Linear(256, 64)` when `mem_dim != hidden_dim`, or `Identity` otherwise. It is used where text must be added into memory-signal paths that live in 64-dim (see FiLM and cross-attention K/V design).

---

## 7. FiLM-style modulation on memory tokens

**Implementation:** `_apply_film_to_memory(feature_1, feature_2)`.

- `feature_1`: memory tokens `[N, B, mem_dim]`.
- `feature_2`: **refined** text `film_text_feat` `[B, hidden_dim]`, or `None` to skip.

**Steps:**

1. `film_hidden = ReLU(Linear(hidden_dim → hidden_dim)(feature_2))`.
2. `alpha = Linear(hidden_dim → mem_dim)(film_hidden)`, `beta = Linear(hidden_dim → mem_dim)(film_hidden)`.
3. **Synthetic KV sequence from text:** `film_text_kv_proj: Linear(hidden_dim → mem_dim * film_text_tokens)` → reshape to **4 tokens** per batch → `[T_txt, B, mem_dim]`.
4. `cross_out = MHA(q=feature_1, k=kv, v=kv)` with `film_cross_attn` (embed `mem_dim`, **8 heads** or fewer if not divisible).
5. **Output:** `feature_1 + alpha.unsqueeze(0) * cross_out + beta.unsqueeze(0)`.

**Where it is applied (dual memory, training path `_prepare_memory_conditioned_features_wo_prompt`):**

1. **Current ego mask memory** (from GT ego mask on the frame) — FiLM with `film_text_feat`, then `memory_attention(curr=exo_tokens, memory=current_ego_maskmem, ...)`.
2. **Fused past bank** — concatenate **past exo and ego** mask memories along the token dimension (and their pos encodings), **optionally** with **object pointer** tokens from past exo frames; apply FiLM to the **fused** tensor; then second `memory_attention` with the same exo `curr` query.

The two `memory_attention` outputs are **added** in feature-map form:  
`pix_feat_with_mem = out_bank + out_current_ego` (permutation to `[B,C,H,W]`).

**Object pointers:** When `use_obj_ptrs_in_encoder: true`, past-frame SAM **object pointer** tokens (projected, possibly split across mem_dim) are **concatenated** into the fused memory sequence; temporal sine PE and optional `obj_ptr_tpos_proj` are applied as in SAM2. Eval-only options such as `only_obj_ptrs_in_the_past_for_eval` restrict which pointers are used.

**Evaluation path:** A parallel implementation `_prepare_memory_conditioned_features_compress_wo_prompt` follows the same **FiLM → dual MHA → add** pattern with compressed memory selection (different frame indexing when `len(non_cond_frame_outputs) < num_maskmem`); semantics align with training but handle cold-start and stride.

---

## 8. Frame stage: `use_text_cross_attn` vs additive merge

**Flag:** `use_text_cross_attn` (both inference and training configs set it **`true`** for the text+FiLM checkpoint path).

**When `True` (see `_track_step` in `sam2_base.py`):**

1. `egff`, `exff` are ego/exo top-level feature maps `[B, C, H, W]`.
2. `egtxtff = egff + text_feat.view(B, C, 1, 1)` (broadcast add in channel space).
3. `ego_frame_to_mem_proj: Linear(hidden_dim, mem_dim)` projects **spatial channels** of ego+text to `mem_dim`, both the map and, critically, the **ego positional sequence** is projected: `egtxtff_mem_pos = self.ego_frame_to_mem_proj(ego_pos_seq)` so keys live in 64-d matching cross-attn `kv_in_dim`.
4. Third call: `memory_attention(curr=exo_flat, memory=ego+text memory, ...)`; output is **residual-added** to exo sequence: `exo ← exo + frame_out` (in sequence form, then reshape).

**When `False`:** `exo_pix_feat = exff + egff + broadcast(text_feat)`.

**Role in a paper:** this is a **shallow "ego+language → exo" alignment** using the *same* transformer stack as memory fusion, without introducing a new module class—only a third forward through `MemoryAttention`.

---

## 9. Text data source (DAM per-frame captions)

**Constant in code:** `sam2/utils/dam_frame_captions.py`  
`DAM_FRAME_CAPTIONS_SUBDIR = "dam_frame_captions_v2"`.

**Layout:** Under each take root:  
`<dataset>/<take_id>/dam_frame_captions_v2/`  
Per-frame text is resolved by `read_dam_frame_caption` (tries `frame_id.txt`, `frame_id.json`, zero-padded variants, etc.) and optional **object JSON** files (`load_frame_caption_map_from_dam_v2_object_json`) keyed by `frames[frame_id]` with fields such as `concise`, `caption`, `text`, etc.

**Inference:** `init_state` sets `text_feat_per_frame` from the caption directory. If the directory is **missing**, a **warning** is logged and **empty** text features are used (no automatic JSON fallback from other files).

**Optional reuse:** `dam_text_reuse_if_abs_frame_distance_lt` in `init_state` reuses a caption for neighboring frames when \(|f - f_0| < L\) (stability / missing-file tolerance).

**Visualization-only:** `object_descriptions_ablation.json` may be used in `tools/visualize_predictions.py` for **figure titles**—not the model’s conditioning unless you wire it separately.

**Doc inconsistency:** Some markdown referred to `dam_frame_captions_v2_exo/`; the **code** uses **`dam_frame_captions_v2`**. Use the code name for the paper’s reproducibility section.

---

## 10. SAM prompt encoder and mask decoder (dual view)

**Dual-view design:** `ViewPromptEncoder` / `_forward_sam_heads` in `sam2_base.py`:

- **Ego** mask is passed as the **mask prompt** into `PromptEncoder` (shape aligned to `mask_input_size` via interpolation as needed).
- **Exo** image features (and high-res exo FPN features) go through **`MaskDecoder`**; points default to a dummy “empty” click if not provided.
- `MaskDecoder` uses `TwoWayTransformer` depth **2**, `mlp_dim: 2048`, **8 heads**, multimask head with **3** candidates when `multimask_output` is True on first clicks; `iou_head_depth: 3`, `iou_head_hidden_dim: 256`, optional **object score** and **pointer** heads when configured.

**Training note:** with `use_mask_input_as_output_without_sam: true` and mask-based conditioning, some branches bypass full SAM and copy inputs—your fine-tune config enables mask-as-output on frames with mask inputs; actual `_track_step` uses memory fusion then `_forward_sam_heads` for the exo head when not short-circuited.

---

## 11. Training configuration (EgoExo fine-tune)

**Config file:** `sam2/configs/sam2.1_training/sam2.1_hiera_b+_EgoExo_finetune.yaml`

| Item | Value |
|------|--------|
| Model wrapper | `training.model.sam2.SAM2Train` |
| Resolution | 480 |
| `num_frames` | 8 |
| `max_num_objects` | 3 |
| `train_batch_size` | 1 |
| `num_train_workers` | 8 |
| `num_epochs` | 150 |
| `base_lr` | 1e-5 (cosine to `base_lr/10`) |
| `vision_lr` | 6e-6 (cosine to `vision_lr/10`) |
| `weight_decay` | 0.1 (0 for biases / LayerNorm) |
| `layer_decay` (image trunk) | 0.9 (pos_embed override 1.0) |
| `gradient_clip` | max_norm 0.1 (L2) |
| AMP | enabled, `bfloat16` |
| Optimizer | AdamW |
| `find_unused_parameters` | True (DDP) |
| Initial weights | `checkpoint/sam2.1_hiera_base_plus.pt` (SAM2.1 Hiera B+), `strict: False` |
| Checkpoint dir | `./training/checkpoints_EgoExo` |

**Augmentations (EgoExo transforms):** horizontal flip, affine (degrees 25, shear 20) when enabled, resize to square 480, color jitter, rare grayscale, per-frame or paired jitter, normalize ImageNet mean/std.

**Data:** `EgoExoVOSDataset` + `EgoExoRawDataset` reading `img_folder` (each take with `annotation.json`, ego Aria + exo cameras). Training **requires** non-empty per-frame captions for sampled frames (otherwise resamples). Missing DAM directory raises `MissingDamFrameCaptionsError` and triggers resampling.

**Loss:** `MultiStepMultiMasksAndIous` with weights: `loss_mask: 20`, `loss_dice: 1`, `loss_iou: 1`, `loss_class: 1`, `supervise_all_iou: true`, `iou_use_l1_loss: true`, `pred_obj_scores: true`, `focal_alpha_obj_score: 0.25`, `focal_gamma_obj_score: 0.0`, `empty_gt_mask_weight: 1.0` (supervises mask/dice/iou on empty GT frames for existence performance).

**Interaction sampling:** EgoExo config sets `prob_to_use_pt_input_*` and `prob_to_use_box_input_*` to **0** (mask-only), `num_init_cond_frames_for_train: 0`, etc.—aligning training with **mask-driven** correspondence rather than point refinement.

**Logging:** optional `text_film_viz_every_epochs` for rank-0 diagnostics under `logs/text_film_viz/`.

**Note on `video_dataset.swap`:** An older `VOSRawDataset` subclass supports swapping ego/exo dict keys for augmentation experiments; the **EgoExo fine-tune YAML** wires **`EgoExoRawDataset`** without that flag—**ego and exo roles are not swapped in the default training config**. Describe your exact training recipe if you used a custom YAML.

---

## 12. Documentation vs. code notes

1. **Caption subdirectory:** Code = `dam_frame_captions_v2` (not `..._v2_exo` unless you change the constant).
2. **Text refinement:** Current code = **two residual MHA + LayerNorm blocks**; an older guide mentioned a per-channel “gate” (`text_filter_gate`)—**that is not in the current `sam2_base.py`**.
3. **`--swap` default:** In `tools/correspondence_inference.py`, `default=False` with help text about predicting exo view. Older runbooks may describe different defaults; **cite the code** for the paper’s supplemental.
4. **`configs/` prefix:** `build_sam.py` composes from `sam2/configs/`. Pass names like `sam2.1/sam2.1_hiera_b+_egoexo_text` (no leading `configs/` in the name passed to Hydra from this tree).

---

## 13. Inference

**Build:** `build_sam2_video_predictor_ego` (see `tools/correspondence_inference.py`).

**Typical command shape:**

```bash
python tools/correspondence_inference.py \
  --sam2_cfg sam2.1/sam2.1_hiera_b+_egoexo_text.yaml \
  --sam2_checkpoint /path/to/checkpoint.pt \
  --base_video_dir /path/to/EgoExo4D/test \
  --output_mask_dir /path/to/output \
  # optional: --swap True/False, --dam_text_reuse_if_abs_frame_distance_lt N, --gpu_id, --num_gpus
```

**`--swap`:** Controls which annotation/camera path drives frame lists and `vos_inference` call pattern (`swap=True` vs `False` branches around lines 653–717). It does not change the neural architecture; it changes **data routing** to match exo/ego directory layout for evaluation.

**Multi-GPU:** Shard `video_names` by `gpu_id` and `num_gpus` (round-robin).

---

## 14. Evaluation pipeline

1. **`evaluation/merge_pred.py`** — merge per-frame predictions with GT `relations_*.json`.
2. **`evaluation/merge_results.py`** — build challenge JSON (`--swap True` if you evaluated with swap).
3. **`evaluation/compute_metrics.py`** — IoU, **Shape Accuracy**, **Location Score**, **Existence Balanced Accuracy** (and related aggregates; see script for `CONF_THRESH` and optional existence confidence thresholding).

Default eval mask resolution in metrics code: **480×480** (see `compute_metrics.py`).

---

## 15. What is *not* in this workspace

- **No `MV_MoE`** (mixture-of-experts) fusion in `sam2_base.py` — the two memory branches are combined by simple element-wise addition (see §1).
- **No gating** module for text refinement in the current file (MHA+LN only).
- Historical / alternate configs may exist under `sam2/configs/..._continue_*.yaml`—treat the **inference** YAML above as the reference for the text+FiLM release unless your checkpoint was trained with another file.

---

## 16. Related prior work (BibTeX)

This project's dual-memory correspondence backbone builds on:

```bibtex
@article{hu2025robust,
  title={Robust Ego-Exo Correspondence with Long-Term Memory},
  author={Hu, Yijun and Fan, Bing and Gu, Xin and Ren, Haiqing and Liu, Dongfang and Fan, Heng and Zhang, Libo},
  journal={arXiv preprint arXiv:2510.11417},
  year={2025}
}
```

**Backbone / SAM2:** cite Meta's SAM 2 / SAM2.1 and Hiera as appropriate from official references.

---

## 17. File index (implementation map)

| Component | File(s) |
|-----------|---------|
| Model core, FiLM, refinement, track step | `sam2/modeling/sam2_base.py` |
| Training forward, tracking loop | `training/model/sam2.py` |
| Video predictor, `init_state`, caption loading | `sam2/sam2_correspondence_predictor.py` |
| Hydra build | `sam2/build_sam.py` |
| Memory stack | `sam2/modeling/memory_attention.py`, `sam2/modeling/memory_encoder.py` |
| Captions | `sam2/utils/dam_frame_captions.py` |
| Training data | `training/dataset/vos_raw_dataset.py`, `vos_dataset.py`, `EgoExo_transforms.py`, `training/utils/data_utils.py` (collate) |
| Losses | `training/loss_fns.py` |
| Inference CLI | `tools/correspondence_inference.py` |
| Eval | `evaluation/merge_pred.py`, `merge_results.py`, `compute_metrics.py` |
