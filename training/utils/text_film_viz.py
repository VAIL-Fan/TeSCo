# Copyright (c) Meta Platforms, Inc. and affiliates.
# Periodic training diagnostics for text refinement + FiLM on memory.

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np


def save_text_film_training_viz(
    epoch: int,
    out_dir: str,
    descriptions: Optional[List[str]],
    acc: Dict[str, Any],
) -> str:
    """
    Write per-epoch text/FiLM stats and a figure under ``out_dir/epoch_XXXX/``.

    Returns the subdirectory path written.
    """
    os.makedirs(out_dir, exist_ok=True)
    sub = os.path.join(out_dir, f"epoch_{epoch:04d}")
    os.makedirs(sub, exist_ok=True)

    descs = descriptions or []
    with open(os.path.join(sub, "descriptions_sample.txt"), "w", encoding="utf-8") as f:
        for i, d in enumerate(descs):
            f.write(f"[{i}] {(d if d else '(empty)')}\n\n")

    cos = acc.get("text_cos") or []
    l2d = acc.get("text_l2_delta") or []
    fnorm = acc.get("film_delta_norm") or []
    frel = acc.get("film_relative") or []

    summary_lines = [
        f"epoch={epoch}",
        f"text_refine_samples={len(cos)}",
        f"film_apply_samples={len(fnorm)}",
    ]
    if cos:
        summary_lines.append(f"text_cos_mean={float(np.mean(cos)):.6f}")
        summary_lines.append(f"text_cos_min={float(np.min(cos)):.6f}")
    if l2d:
        summary_lines.append(f"text_l2_delta_mean={float(np.mean(l2d)):.6f}")
    if fnorm:
        summary_lines.append(f"film_delta_l2_mean={float(np.mean(fnorm)):.6f}")
    if frel:
        summary_lines.append(f"film_delta_relative_to_memory_mean={float(np.mean(frel)):.6f}")

    with open(os.path.join(sub, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax = axes[0, 0]
    if cos:
        ax.plot(cos, marker="o", ms=2, linewidth=1)
        ax.axhline(float(np.mean(cos)), color="r", linestyle="--", alpha=0.6, label="mean")
        ax.legend(fontsize=8)
    ax.set_title("Text refine: cosine(text, refined) (batch mean per step)")
    ax.set_xlabel("capture index")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    if l2d:
        ax.plot(l2d, marker="o", ms=2, linewidth=1, color="green")
        ax.axhline(float(np.mean(l2d)), color="r", linestyle="--", alpha=0.6)
    ax.set_title("Text refine: L2(refined - raw) batch-mean")
    ax.set_xlabel("capture index")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    if fnorm:
        ax.plot(fnorm, marker="o", ms=2, linewidth=1, color="purple")
        ax.axhline(float(np.mean(fnorm)), color="r", linestyle="--", alpha=0.6)
    ax.set_title("FiLM: ||memory_out - memory_in||_F")
    ax.set_xlabel("capture index")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    if frel:
        ax.plot(frel, marker="o", ms=2, linewidth=1, color="orange")
        ax.axhline(float(np.mean(frel)), color="r", linestyle="--", alpha=0.6)
    ax.set_title("FiLM: delta norm / memory norm")
    ax.set_xlabel("capture index")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Text + FiLM diagnostics (epoch {epoch})")
    fig.tight_layout()
    png_path = os.path.join(sub, "text_film_diagnostics.png")
    fig.savefig(png_path, dpi=120)
    plt.close(fig)

    return sub
