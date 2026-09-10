"""
Plotting utilities for Surface Code QEC benchmarking with Stim and Sinter.

Provides publication-quality visualizations for QEC threshold curves and
noise model comparisons (Pauli vs Coherent noise) for Maestro 0.3.1.
"""

from __future__ import annotations

import math
from typing import Sequence, Any
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _wilson_score_interval(errors: int, shots: int, z: float = 1.96) -> tuple[float, float, float]:
    """Compute center, low, high of Wilson score interval for binomial proportion."""
    if shots <= 0:
        return 0.0, 0.0, 0.0
    p = errors / shots
    denom = 1.0 + z * z / shots
    center = (p + z * z / (2.0 * shots)) / denom
    half_width = (z * math.sqrt(p * (1.0 - p) / shots + z * z / (4.0 * shots * shots))) / denom
    low = max(0.0, center - half_width)
    high = min(1.0, center + half_width)
    return p, low, high


def plot_threshold_curves(
    sinter_stats: Sequence[Any],
    save_path: str,
    title: str = "Rotated Surface Code Memory: Logical Error Rate vs Physical Noise",
) -> str:
    """
    Plot logical error rate vs physical error rate curves from Sinter TaskStats.

    Groups stats by (sampler_name, distance d), displaying threshold crossing.
    """
    fig, ax = plt.subplots(figsize=(10, 6.5))

    # Group stats by (sampler_label, distance)
    grouped: dict[tuple[str, int], list[tuple[float, int, int]]] = {}
    for s in sinter_stats:
        meta = s.json_metadata if hasattr(s, "json_metadata") else {}
        d = meta.get("d", 3)
        p = meta.get("p", 0.01)
        sampler_name = meta.get("sampler", "maestro")
        key = (sampler_name, d)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append((p, s.errors, s.shots))

    color_cycle = {
        3: "#1565C0",  # Blue for d=3
        5: "#E65100",  # Orange for d=5
        7: "#2E7D32",  # Green for d=7
    }

    markers = {"stim": "o", "maestro": "s", "coherent": "^"}

    for (sampler_name, d), data in sorted(grouped.items(), key=lambda x: (x[0][1], x[0][0])):
        data.sort(key=lambda item: item[0])
        p_vals = np.array([item[0] for item in data])
        errors_arr = np.array([item[1] for item in data])
        shots_arr = np.array([item[2] for item in data])

        rates = []
        err_low = []
        err_high = []
        for e, n in zip(errors_arr, shots_arr):
            p_center, low, high = _wilson_score_interval(e, n)
            rates.append(p_center)
            err_low.append(max(0.0, p_center - low))
            err_high.append(max(0.0, high - p_center))

        rates = np.array(rates)
        yerr = [err_low, err_high]

        color = color_cycle.get(d, "#9C27B0")
        marker = markers.get(sampler_name.lower(), "o")
        linestyle = "--" if sampler_name.lower() == "stim" else "-"

        label = f"{sampler_name.capitalize()} (d={d})"
        ax.errorbar(
            p_vals,
            rates,
            yerr=yerr,
            fmt=f"{marker}{linestyle}",
            color=color,
            label=label,
            capsize=3,
            linewidth=2,
            markersize=7,
            alpha=0.9,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Physical Error Rate (p)", fontsize=12)
    ax.set_ylabel("Logical Error Rate (P_L)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(fontsize=10, loc="lower right")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_noise_comparison(
    pauli_stats: Sequence[Any],
    coherent_stats: Sequence[Any],
    save_path: str,
    distance: int = 3,
) -> str:
    """
    Plot comparison between Pauli depolarizing noise and coherent rotation noise
    evaluated on the surface code decoder using PyMatching.
    """
    fig, ax = plt.subplots(figsize=(10, 6.5))

    def extract_curve(stats_list):
        points = []
        for s in stats_list:
            meta = s.json_metadata if hasattr(s, "json_metadata") else {}
            p = meta.get("p", 0.01)
            p_center, low, high = _wilson_score_interval(s.errors, s.shots)
            points.append((p, p_center, p_center - low, high - p_center))
        points.sort(key=lambda x: x[0])
        return points

    p_pauli = extract_curve(pauli_stats)
    p_coh = extract_curve(coherent_stats)

    if p_pauli:
        p_vals = [pt[0] for pt in p_pauli]
        rates = [pt[1] for pt in p_pauli]
        yerr = [[pt[2] for pt in p_pauli], [pt[3] for pt in p_pauli]]
        ax.errorbar(
            p_vals,
            rates,
            yerr=yerr,
            fmt="o--",
            color="#2196F3",
            linewidth=2,
            markersize=7,
            capsize=3,
            label=f"Pauli Depolarizing (d={distance}) — Stim / Maestro MPS",
        )

    if p_coh:
        p_vals = [pt[0] for pt in p_coh]
        rates = [pt[1] for pt in p_coh]
        yerr = [[pt[2] for pt in p_coh], [pt[3] for pt in p_coh]]
        ax.errorbar(
            p_vals,
            rates,
            yerr=yerr,
            fmt="s-",
            color="#E91E63",
            linewidth=2.5,
            markersize=7,
            capsize=3,
            label=f"Coherent + Idle Noise (d={distance}) — Maestro MPS Only",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Physical Error Parameter (p / ε)", fontsize=12)
    ax.set_ylabel("Logical Error Rate (P_L)", fontsize=12)
    ax.set_title(
        f"Surface Code Memory (d={distance}): Pauli vs Coherent Noise\n"
        f"Stim models Pauli noise; Maestro MPS captures coherent error accumulation",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(fontsize=10, loc="upper left")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path
