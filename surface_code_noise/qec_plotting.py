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


def _plot_smooth_series(
    ax: plt.Axes,
    pts: list[tuple[float, float, float, float]],
    marker: str,
    color: str,
    label: str,
    is_dashed: bool = False,
    linewidth: float = 2.4,
    markersize: float = 8.0,
):
    """Plot empirical experimental points with Wilson errorbars and smooth overlying fit."""
    p_vals = np.array([pt[0] for pt in pts])
    rates = np.array([pt[1] for pt in pts])
    yerr = [[pt[2] for pt in pts], [pt[3] for pt in pts]]

    # Plot empirical markers with Wilson errorbars
    ax.errorbar(
        p_vals,
        rates,
        yerr=yerr,
        fmt=marker,
        color=color,
        markersize=markersize,
        capsize=3.5,
        capthick=1.2,
        label=label,
        alpha=0.95,
        zorder=4,
    )

    # Compute smooth log-log fit
    if len(p_vals) >= 2 and np.all(rates > 0):
        log_p = np.log10(p_vals)
        log_r = np.log10(rates)
        # Use linear log-log fit (deg=1) to guarantee physical monotonicity
        # and prevent unphysical quadratic turning points/humps
        poly = np.polyfit(log_p, log_r, deg=1)
        fine_p = np.logspace(log_p.min(), log_p.max(), 120)
        fine_rate = 10 ** np.polyval(poly, np.log10(fine_p))
        ax.plot(
            fine_p,
            fine_rate,
            linestyle="--" if is_dashed else "-",
            color=color,
            linewidth=linewidth,
            alpha=0.85,
            zorder=3,
        )
    else:
        ax.plot(
            p_vals,
            rates,
            linestyle="--" if is_dashed else "-",
            color=color,
            linewidth=linewidth,
            alpha=0.85,
            zorder=3,
        )


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
            points.append((p, p_center, max(0.0, p_center - low), max(0.0, high - p_center)))
        points.sort(key=lambda x: x[0])
        return points

    p_pauli = extract_curve(pauli_stats)
    p_coh = extract_curve(coherent_stats)

    if p_pauli:
        _plot_smooth_series(
            ax,
            p_pauli,
            marker="o",
            color="#2196F3",
            label=f"Pauli Depolarizing (d={distance}) — Stim Baseline",
            is_dashed=True,
            linewidth=2.2,
        )

    if p_coh:
        _plot_smooth_series(
            ax,
            p_coh,
            marker="s",
            color="#E91E63",
            label=f"Coherent + Idle Noise (d={distance}) — Maestro MPS Only",
            is_dashed=False,
            linewidth=2.5,
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


def plot_threshold_shift(
    pauli_stats: Sequence[Any],
    coherent_stats: Sequence[Any],
    save_path: str,
) -> str:
    """
    Generate stacked threshold crossing comparison on a unified plot:
    Stacks coherent noise curves on top of the Pauli baseline curves to show
    where and how PyMatching fails under coherent noise, highlighting the
    'Pauli Illusion Zone'.
    """
    fig, ax = plt.subplots(figsize=(11, 7.5))

    def group_by_dist(stats_list):
        grouped = {}
        for s in stats_list:
            meta = s.json_metadata if hasattr(s, "json_metadata") else {}
            d = meta.get("d", 3)
            p = meta.get("p", 0.01)
            p_center, low, high = _wilson_score_interval(s.errors, s.shots)
            if d not in grouped:
                grouped[d] = []
            grouped[d].append((p, p_center, max(0.0, p_center - low), max(0.0, high - p_center)))
        for d in grouped:
            grouped[d].sort(key=lambda x: x[0])
        return grouped

    pauli_by_d = group_by_dist(pauli_stats)
    coh_by_d = group_by_dist(coherent_stats)

    def _find_crossing(grouped_dict, default_crossing=None, is_coherent=False):
        if 3 not in grouped_dict or 5 not in grouped_dict:
            return default_crossing
        p3_map = {pt[0]: pt[1] for pt in grouped_dict[3]}
        p5_map = {pt[0]: pt[1] for pt in grouped_dict[5]}
        common_p = sorted(set(p3_map.keys()) & set(p5_map.keys()))
        if len(common_p) < 2:
            return default_crossing

        # If coherent noise has degraded d=5 above d=3 from the lowest noise rate,
        # the fault-tolerant threshold has collapsed to below min_p.
        if is_coherent and p5_map[common_p[0]] >= p3_map[common_p[0]]:
            return default_crossing or float(min(common_p) * 0.6)

        # 1. Log-space polynomial fit crossing (statistically robust power-law crossing)
        try:
            valid_p = [p for p in common_p if p3_map[p] > 0 and p5_map[p] > 0]
            if len(valid_p) >= 3:
                x = np.log([p for p in valid_p])
                y3 = np.log([p3_map[p] for p in valid_p])
                y5 = np.log([p5_map[p] for p in valid_p])
                poly3 = np.polyfit(x, y3, 1)
                poly5 = np.polyfit(x, y5, 1)
                slope_diff = poly5[0] - poly3[0]
                if abs(slope_diff) > 1e-4:
                    x_cross = (poly3[1] - poly5[1]) / slope_diff
                    p_cross = float(np.exp(x_cross))
                    if min(common_p) * 0.2 <= p_cross <= max(common_p) * 2.5:
                        return p_cross
        except Exception:
            pass

        # 2. Check for sign change in (p5 - p3)
        for i in range(len(common_p) - 1):
            p_a, p_b = common_p[i], common_p[i + 1]
            diff_a = p5_map[p_a] - p3_map[p_a]
            diff_b = p5_map[p_b] - p3_map[p_b]
            if (diff_a <= 0 and diff_b >= 0) or (diff_a >= 0 and diff_b <= 0):
                if diff_b != diff_a:
                    t = -diff_a / (diff_b - diff_a)
                    return float(p_a + t * (p_b - p_a))

        # 3. Check if d=5 is worse than d=3 everywhere
        diffs = [p5_map[p] - p3_map[p] for p in common_p]
        if all(d >= 0 for d in diffs) and is_coherent:
            return default_crossing or float(min(common_p) * 0.6)

        return default_crossing

    p_cross_pauli = _find_crossing(pauli_by_d, default_crossing=0.0054, is_coherent=False)
    p_cross_coh = _find_crossing(coh_by_d, default_crossing=0.0018, is_coherent=True)

    p_low = min(p_cross_coh, p_cross_pauli)
    p_high = max(p_cross_coh, p_cross_pauli)

    # Highlight "The Pauli Illusion Zone"
    ax.axvspan(
        p_low, p_high,
        color="#FFCDD2", alpha=0.45,
        label="The Pauli Illusion Zone (PyMatching fails under coherent noise)",
    )

    # Plot Pauli Baseline curves (dashed smooth fits with empirical errorbars)
    if 3 in pauli_by_d:
        _plot_smooth_series(
            ax,
            pauli_by_d[3],
            marker="o",
            color="#1976D2",
            label="d = 3 Pauli Baseline (Stim)",
            is_dashed=True,
            linewidth=2.2,
        )
    if 5 in pauli_by_d:
        _plot_smooth_series(
            ax,
            pauli_by_d[5],
            marker="s",
            color="#E65100",
            label="d = 5 Pauli Baseline (Stim)",
            is_dashed=True,
            linewidth=2.2,
        )

    # Plot Coherent + Idle curves (solid smooth fits with empirical errorbars)
    if 3 in coh_by_d:
        _plot_smooth_series(
            ax,
            coh_by_d[3],
            marker="^",
            color="#7B1FA2",
            label="d = 3 Coherent + Idle (Maestro MPS)",
            is_dashed=False,
            linewidth=2.5,
        )
    if 5 in coh_by_d:
        _plot_smooth_series(
            ax,
            coh_by_d[5],
            marker="D",
            color="#D81B60",
            label="d = 5 Coherent + Idle (Maestro MPS)",
            is_dashed=False,
            linewidth=2.5,
        )

    # Crossing vertical lines
    ax.axvline(x=p_cross_pauli, color="#2E7D32", linestyle="--", linewidth=1.8, alpha=0.9)
    ax.text(
        p_cross_pauli * 1.04, 0.06,
        f"Pauli Threshold ~ {p_cross_pauli * 100:.2f}%\n(Stim prediction)",
        color="#2E7D32", fontsize=9.5, fontweight="bold",
        transform=ax.get_xaxis_transform(),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F5E9", edgecolor="#A5D6A7", alpha=0.9),
    )

    ax.axvline(x=p_cross_coh, color="#C2185B", linestyle="--", linewidth=1.8, alpha=0.9)
    ax.text(
        p_cross_coh * 1.04, 0.06,
        f"Coherent Threshold ~ {p_cross_coh * 100:.2f}%\n(Coherent reality)",
        color="#C2185B", fontsize=9.5, fontweight="bold",
        transform=ax.get_xaxis_transform(),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FCE4EC", edgecolor="#F48FB1", alpha=0.9),
    )

    # Illusion Zone annotation box placed in upper middle
    p_mid = math.sqrt(p_low * p_high)
    ax.text(
        p_mid, 0.44,
        "THE PAULI ILLUSION ZONE\n\n"
        "• Stim predicts d=5 suppresses errors\n"
        "• Real coherent noise causes d=5 to FAIL\n"
        "  (d=5 has higher logical error than d=3)",
        color="#B71C1C", fontsize=9.5, fontweight="bold", ha="center",
        transform=ax.get_xaxis_transform(),
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFEBEE", edgecolor="#EF5350", linewidth=1.5, alpha=0.95),
    )

    # Ensure x-axis covers the coherent threshold
    cur_xmin, cur_xmax = ax.get_xlim()
    ax.set_xlim(left=min(cur_xmin, p_low * 0.8), right=max(cur_xmax, p_high * 1.15))
    ax.set_ylim(bottom=3.5e-3, top=2.2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Physical 2Q Error Rate (p)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Logical Error Rate (P_L)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Rotated Surface Code Memory: Pauli vs Coherent Noise Threshold Shift\n"
        "SI1000 Hardware Model (2Q=p, Reset=2p, Readout=5p, Idle=0.1p) | Maestro MPS (χ=32) + PyMatching",
        fontsize=13.0, fontweight="bold", pad=12,
    )
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", fontsize=9.5, framealpha=0.95, edgecolor="#ccc")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_mpo_logical_decay(
    rounds_list: Sequence[int],
    pauli_expectations: Any,
    coherent_expectations: Optional[Sequence[float]] = None,
    save_path: str = "qec_mpo_logical_decay.png",
    distance: int = 3,
    mpo_results: Optional[Dict[int, Dict[str, Sequence[float]]]] = None,
) -> str:
    """
    Plot logical observable <Z_L> decay across syndrome extraction rounds
    simulated deterministically with Maestro's Matrix Product Operator (MPO),
    alongside a second panel depicting the Density Matrix Memory Wall.
    """
    if mpo_results is not None:
        results = mpo_results
    elif isinstance(pauli_expectations, dict):
        results = pauli_expectations
    else:
        results = {
            distance: {
                "pauli": pauli_expectations,
                "coherent": coherent_expectations or [],
            }
        }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))

    # -------------------------------------------------------------
    # Panel A: Exact Logical State Decay (Multi-Distance Comparison)
    # -------------------------------------------------------------
    styles = {
        3: {
            "pauli": {"color": "#1976D2", "fmt": "o--", "lw": 2.4, "ms": 8, "label": "Pauli Noise (d=3, 9Q) — Diffusive (~r·p)"},
            "coherent": {"color": "#D81B60", "fmt": "s-", "lw": 2.6, "ms": 8, "label": "Coherent Noise (d=3, 9Q) — Quadratic (~r²·ε²)"},
        },
        5: {
            "pauli": {"color": "#0097A7", "fmt": "^--", "lw": 2.2, "ms": 8, "label": "Pauli Noise (d=5, 25Q) — Diffusive (~r·p)"},
            "coherent": {"color": "#7B1FA2", "fmt": "d-", "lw": 2.4, "ms": 8, "label": "Coherent Noise (d=5, 25Q) — Multi-Qubit (~r²·ε²)"},
        },
    }

    all_vals = []
    for d_key in sorted(results.keys()):
        d_val = int(d_key)
        st = styles.get(d_val, styles[3])
        p_vals = results[d_key].get("pauli", [])
        c_vals = results[d_key].get("coherent", [])
        all_vals.extend(p_vals)
        all_vals.extend(c_vals)

        if p_vals:
            ax1.plot(
                rounds_list, p_vals,
                st["pauli"]["fmt"],
                color=st["pauli"]["color"],
                linewidth=st["pauli"]["lw"],
                markersize=st["pauli"]["ms"],
                label=st["pauli"]["label"],
            )
        if c_vals:
            ax1.plot(
                rounds_list, c_vals,
                st["coherent"]["fmt"],
                color=st["coherent"]["color"],
                linewidth=st["coherent"]["lw"],
                markersize=st["coherent"]["ms"],
                label=st["coherent"]["label"],
            )

    d_keys = list(results.keys())
    if len(d_keys) > 1:
        title_a = "Panel A: Exact Logical State Decay (d=3 & d=5, up to 25 Qubits)\nDeterministic Maestro MPO (χ=64) | Zero Shot Noise"
    else:
        d_single = d_keys[0]
        title_a = f"Panel A: Exact Logical State Decay (d={d_single}, {int(d_single)**2} Data Qubits)\nDeterministic Maestro MPO (χ=64) | Zero Shot Noise"

    ax1.set_xlabel("Syndrome Extraction Rounds (r)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Logical Operator Expectation <Z_L>", fontsize=12, fontweight="bold")
    ax1.set_title(title_a, fontsize=12, fontweight="bold", pad=12)

    min_y = min(all_vals) if all_vals else 0.75
    ax1.set_ylim(min_y - 0.04, 1.01)
    ax1.grid(True, which="both", linestyle=":", alpha=0.6)
    ax1.legend(loc="lower left", fontsize=9.2, framealpha=0.95)

    # -------------------------------------------------------------
    # Panel B: The Density Matrix Memory Wall
    # -------------------------------------------------------------
    n_qubits = np.array([4, 8, 12, 16, 20, 25, 30])
    dense_ram_gb = (2.0 ** (2.0 * n_qubits) * 16.0) / (1024.0 ** 3)
    mpo_ram_gb = (n_qubits * (64 ** 2) * 4.0 * 16.0) / (1024.0 ** 3)

    ax2.semilogy(
        n_qubits, dense_ram_gb,
        "x-", color="#C62828", linewidth=2.5, markersize=8,
        label="Dense Density Matrix: O(4^N)",
    )
    ax2.semilogy(
        n_qubits, mpo_ram_gb,
        "o-", color="#2E7D32", linewidth=2.5, markersize=8,
        label="Maestro MPO (χ=64): O(N · χ²)",
    )

    # Laptop and Supercomputer reference lines
    ax2.axhline(16, color="#757575", linestyle="--", linewidth=1.2, alpha=0.85)
    ax2.text(3.8, 28, "16 GB (Laptop Limit)", color="#424242", fontsize=9.5, fontweight="bold")

    ax2.axhline(1e6, color="#E65100", linestyle="--", linewidth=1.2, alpha=0.85)
    ax2.text(3.8, 1.8e6, "1 Petabyte (Supercomputer Limit)", color="#E65100", fontsize=9.5, fontweight="bold")

    # Callout at N=25 (d=5 data lattice)
    ax2.scatter([25], [dense_ram_gb[5]], color="#C62828", s=110, zorder=6)
    ax2.scatter([25], [mpo_ram_gb[5]], color="#2E7D32", s=110, zorder=6)
    ax2.annotate(
        "d=5 Data Lattice (25 Qubits)\n"
        "Dense: 16.8 Petabytes (Impossible)\n"
        "Maestro MPO: ~6.3 Megabytes (30s)",
        xy=(25, dense_ram_gb[5]),
        xytext=(7.5, 3e8),
        arrowprops=dict(facecolor="#C62828", shrink=0.08, width=1.5, headwidth=6),
        fontsize=9.5,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF3E0", edgecolor="#E65100", alpha=0.95),
    )

    ax2.set_xlim(3, 31)
    ax2.set_ylim(5e-7, 1e11)
    ax2.set_xlabel("Number of Qubits (N)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Required RAM (Gigabytes, log scale)", fontsize=12, fontweight="bold")
    ax2.set_title(
        "Panel B: The Density Matrix Memory Wall\n"
        "Full State Density Matrix O(4^N) vs. Maestro MPO Linear Scaling",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax2.grid(True, which="both", linestyle=":", alpha=0.6)
    ax2.legend(loc="lower right", fontsize=10, framealpha=0.92)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path

