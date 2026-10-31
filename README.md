# TeSCo

**Exploiting Textual Semantics for Robust Cross-View Object Correspondence** — NeurIPS 2026

[**Project page**](https://vail-fan.github.io/TeSCo/page/) · [Paper](#citation) · [Results](#results) · [Video results](https://vail-fan.github.io/TeSCo/page/#clips)

Given an object marked in one camera's video, segment the same object in another camera's video
of the same scene. TeSCo describes the object in words first, then uses that description to find
it in the other view — which survives the viewpoint and appearance changes that defeat purely
visual matching.

<p align="center"><img src="assets/framework.png" width="100%"></p>

The query view's object is captioned by a vision-language model. **ITFR** refines that text against
the search view, and **TCF** fuses the refined text into the visual features that SAM 2's decoder
sees. Everything else is SAM 2.1.

## Results

Ego-Exo4D correspondence test split:

| Method | Ego→Exo IoU | Exo→Ego IoU | mIoU |
|---|---|---|---|
| ObjectRelator [ICCV'25] | 35.30 | 40.30 | 37.80 |
| O-MaMa [ICCV'25] | 42.60 | 44.10 | 43.35 |
| V²-SAM [CVPR'26] | 46.30 | 49.60 | 47.95 |
| LM-EEC [NeurIPS'25] | 54.98 | 65.77 | 60.38 |
| Baseline (ours, no text) | 52.70 | 58.30 | 55.50 |
| **TeSCo** | **60.40** | **70.30** | **65.35** |

Text is worth **9.85 mIoU** over the same model without it.

## Quick start

```bash
conda env create -f environment.yml && conda activate tesco
pip install -e .
```

Obtain Ego-Exo4D and place it under `data/` — [`docs/SETUP.md`](docs/SETUP.md) gives the exact
layout. Then:

```bash
# 1. predict  (Ego->Exo; add --exo_to_ego for the other direction)
python tools/correspondence_inference.py \
  --sam2_cfg sam2.1/sam2.1_hiera_b+_egoexo_text.yaml \
  --sam2_checkpoint training/checkpoints_EgoExo/tesco_egoexo4d_epoch59_for_inference.pt \
  --base_video_dir data/EgoExo4D/data_segswap_test \
  --input_mask_dir data/EgoExo4D/data_segswap_test \
  --output_mask_dir output/ego2exo \
  --caption_subdir dam_frame_captions_v2

# 2. assemble per-take PNGs into one submission JSON
python evaluation/merge_pred.py --input output/ego2exo --pred output/ego2exo \
  --gt data/gt_results/relations_test_from_annotationjson.json
python evaluation/merge_results.py --input_dir data/EgoExo4D --split data_segswap_test \
  --pred_dir output/ego2exo --json_path output/ego2exo.json

# 3. score
python evaluation/unified_eval.py --gt data/gt_results/egoexo_test_gt.json \
  --pred output/ego2exo.json --task ego-exo --coverage both
```

Training and every flag: [`docs/SETUP.md`](docs/SETUP.md).


## Citation

```bibtex
@inproceedings{tesco2026,
  author  = {Fan, Bing and Feng, Yunhe and Huang, Yan and Fan, Heng},
  title     = {Exploiting Textual Semantics for Robust Cross-View Object Correspondence},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
