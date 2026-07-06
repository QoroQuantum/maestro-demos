"""
Shared Plotting Utilities for GPU MPS Crossover Benchmarks
===========================================================

Generates publication-quality figures from benchmark JSON results:
    - GPU vs CPU speedup heatmaps in (N, χ) space
    - Wall-clock scaling curves
    - Energy convergence with bond dimension
    - Entanglement-driven GPU advantage in quench dynamics
"""

import os
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ─────────────────────────────────────────────────────────────────────
# Color palettes and style
# ─────────────────────────────────────────────────────────────────────

COLORS = {
    'cpu': '#2196F3',
    'gpu': '#E91E63',
    'heisenberg': '#4CAF50',
    'anderson': '#FF9800',
    'crossover': '#9C27B0',
    'neutral': '#607D8B',
}

def setup_style():
    """Apply consistent plotting style."""
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'legend.fontsize': 9,
        'figure.dpi': 150,
        'savefig.bbox': 'tight',
        'savefig.dpi': 150,
    })


# ─────────────────────────────────────────────────────────────────────
# Crossover heatmap
# ─────────────────────────────────────────────────────────────────────

def plot_crossover_heatmap(results, title, save_path):
    """
    2D heatmap of GPU speedup over CPU in (N, χ) space.

    Args:
        results: List of dicts with keys:
            'n_qubits', 'chi', 'cpu_time', 'gpu_time'
        title: Plot title.
        save_path: Where to save the figure.
    """
    setup_style()

    if not results or not any(r.get('gpu_time') for r in results):
        print(f"    ⚠ No GPU data — skipping heatmap: {save_path}")
        return None

    # Extract unique axes
    n_vals = sorted(set(r['n_qubits'] for r in results if r.get('gpu_time')))
    chi_vals = sorted(set(r['chi'] for r in results if r.get('gpu_time')))

    if len(n_vals) < 2 or len(chi_vals) < 2:
        print(f"    ⚠ Insufficient data dimensions — skipping heatmap")
        return None

    # Build speedup matrix
    speedup = np.full((len(chi_vals), len(n_vals)), np.nan)
    for r in results:
        if r.get('gpu_time') and r['gpu_time'] > 0:
            i = chi_vals.index(r['chi'])
            j = n_vals.index(r['n_qubits'])
            speedup[i, j] = r['cpu_time'] / r['gpu_time']

    fig, ax = plt.subplots(figsize=(10, 6))

    mask = ~np.isnan(speedup)
    if mask.any():
        vmin = max(0.1, np.nanmin(speedup))
        vmax = max(2.0, np.nanmax(speedup))
    else:
        vmin, vmax = 0.1, 10.0

    im = ax.imshow(
        speedup, aspect='auto', origin='lower',
        cmap='RdYlGn', norm=LogNorm(vmin=vmin, vmax=vmax),
        interpolation='nearest',
    )

    ax.set_xticks(range(len(n_vals)))
    ax.set_xticklabels(n_vals)
    ax.set_yticks(range(len(chi_vals)))
    ax.set_yticklabels(chi_vals)
    ax.set_xlabel('System Size N (qubits)')
    ax.set_ylabel('Bond Dimension χ')
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax, label='GPU Speedup over CPU')

    # Annotate cells
    for i in range(len(chi_vals)):
        for j in range(len(n_vals)):
            val = speedup[i, j]
            if not np.isnan(val):
                color = 'white' if val > 2.0 or val < 0.5 else 'black'
                ax.text(j, i, f'{val:.1f}×', ha='center', va='center',
                        fontsize=9, fontweight='bold', color=color)

    # Draw crossover line (speedup = 1.0)
    ax.contour(speedup, levels=[1.0], colors=['black'],
               linewidths=[2], linestyles=['--'])

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  📊 Saved: {save_path}")
    return save_path


# ─────────────────────────────────────────────────────────────────────
# Scaling curves
# ─────────────────────────────────────────────────────────────────────

def plot_scaling_curves(results, x_key, group_key, title, save_path,
                        x_label=None, log_scale=True):
    """
    Wall-clock time vs a sweep variable, grouped by another variable.

    Args:
        results: List of result dicts.
        x_key: Key for x-axis ('n_qubits', 'chi', 'depth', 'n_steps').
        group_key: Key to group lines by ('chi', 'n_qubits', etc.).
        title: Plot title.
        save_path: Where to save the figure.
        x_label: Optional x-axis label.
        log_scale: Whether to use log scale on y-axis.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    colors_cycle = ['#E91E63', '#2196F3', '#4CAF50', '#FF9800',
                    '#9C27B0', '#00BCD4', '#795548']

    # Group data
    groups = {}
    for r in results:
        key = r.get(group_key, 'default')
        if key not in groups:
            groups[key] = {'x': [], 'cpu': [], 'gpu': []}
        groups[key]['x'].append(r[x_key])
        groups[key]['cpu'].append(r['cpu_time'])
        if r.get('gpu_time'):
            groups[key]['gpu'].append(r['gpu_time'])

    for idx, (group_val, data) in enumerate(sorted(groups.items())):
        c = colors_cycle[idx % len(colors_cycle)]
        sort_idx = np.argsort(data['x'])
        x = np.array(data['x'])[sort_idx]
        cpu = np.array(data['cpu'])[sort_idx]

        ax.plot(x, cpu, 'o-', color=c, linewidth=2, markersize=6,
                label=f'{group_key}={group_val} (CPU)')

        if data['gpu']:
            gpu = np.array(data['gpu'])[sort_idx[:len(data['gpu'])]]
            ax.plot(x[:len(gpu)], gpu, 's--', color=c, linewidth=2,
                    markersize=6, alpha=0.7,
                    label=f'{group_key}={group_val} (GPU)')

    ax.set_xlabel(x_label or x_key, fontsize=12)
    ax.set_ylabel('Wall-Clock Time (s)', fontsize=12)
    ax.set_title(title, fontsize=14)
    if log_scale:
        ax.set_yscale('log')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  📊 Saved: {save_path}")
    return save_path


# ─────────────────────────────────────────────────────────────────────
# Energy convergence
# ─────────────────────────────────────────────────────────────────────

def plot_energy_convergence(results, title, save_path):
    """
    Energy vs bond dimension — shows MPS accuracy convergence.

    Args:
        results: List of dicts with 'chi', 'energy', 'n_qubits'.
        title: Plot title.
        save_path: Where to save the figure.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    colors_cycle = ['#E91E63', '#2196F3', '#4CAF50', '#FF9800', '#9C27B0']

    # Group by system size
    groups = {}
    for r in results:
        n = r['n_qubits']
        if n not in groups:
            groups[n] = {'chi': [], 'energy': []}
        groups[n]['chi'].append(r['chi'])
        groups[n]['energy'].append(r['energy'])

    for idx, (n, data) in enumerate(sorted(groups.items())):
        c = colors_cycle[idx % len(colors_cycle)]
        sort_idx = np.argsort(data['chi'])
        chi = np.array(data['chi'])[sort_idx]
        energy = np.array(data['energy'])[sort_idx]

        ax.plot(chi, energy, 'o-', color=c, linewidth=2, markersize=7,
                label=f'N = {n}')

    ax.set_xlabel('Bond Dimension χ', fontsize=12)
    ax.set_ylabel('Energy E', fontsize=12)
    ax.set_xscale('log', base=2)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  📊 Saved: {save_path}")
    return save_path


# ─────────────────────────────────────────────────────────────────────
# Timing bar chart (CPU vs GPU)
# ─────────────────────────────────────────────────────────────────────

def plot_timing_bars(results, title, save_path):
    """
    Side-by-side bars comparing CPU vs GPU wall-clock time.

    Args:
        results: List of dicts with 'label', 'cpu_time', 'gpu_time' (optional).
        title: Plot title.
        save_path: Where to save the figure.
    """
    setup_style()

    labels = [r.get('label', f"N={r['n_qubits']}, χ={r['chi']}") for r in results]
    cpu_times = [r['cpu_time'] for r in results]
    gpu_times = [r.get('gpu_time', 0) for r in results]
    has_gpu = any(t > 0 for t in gpu_times)

    x = np.arange(len(labels))
    width = 0.35 if has_gpu else 0.6

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.5), 6))

    bars_cpu = ax.bar(x - (width / 2 if has_gpu else 0), cpu_times,
                      width, label='CPU', color=COLORS['cpu'], alpha=0.85,
                      edgecolor='black', linewidth=0.5)

    if has_gpu:
        bars_gpu = ax.bar(x + width / 2, gpu_times, width,
                          label='GPU', color=COLORS['gpu'], alpha=0.85,
                          edgecolor='black', linewidth=0.5)

        # Annotate speedup
        for i, (ct, gt) in enumerate(zip(cpu_times, gpu_times)):
            if gt > 0:
                speedup = ct / gt
                ax.annotate(
                    f'{speedup:.1f}×',
                    (x[i] + width / 2, gt),
                    textcoords='offset points', xytext=(0, 5),
                    ha='center', fontsize=9, fontweight='bold',
                    color=COLORS['gpu'],
                )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Wall-Clock Time (s)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis='y')

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  📊 Saved: {save_path}")
    return save_path


# ─────────────────────────────────────────────────────────────────────
# Quench dynamics: per-step timing
# ─────────────────────────────────────────────────────────────────────

def plot_per_step_timing(step_times_cpu, step_times_gpu, title, save_path):
    """
    Per-Trotter-step wall-clock time — shows how GPU advantage grows as
    entanglement accumulates.

    Args:
        step_times_cpu: List of per-step times (CPU).
        step_times_gpu: List of per-step times (GPU), or None.
        title: Plot title.
        save_path: Where to save the figure.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    steps = range(1, len(step_times_cpu) + 1)
    ax.plot(steps, step_times_cpu, 'o-', color=COLORS['cpu'], linewidth=2,
            markersize=5, label='CPU')

    if step_times_gpu:
        ax.plot(steps[:len(step_times_gpu)], step_times_gpu, 's-',
                color=COLORS['gpu'], linewidth=2, markersize=5, label='GPU')

    ax.set_xlabel('Trotter Step', fontsize=12)
    ax.set_ylabel('Time per Step (s)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  📊 Saved: {save_path}")
    return save_path


# ─────────────────────────────────────────────────────────────────────
# Summary crossover report
# ─────────────────────────────────────────────────────────────────────

def plot_crossover_summary(all_results, save_path):
    """
    Combined summary figure: scatter plot of all (N, χ) points colored by
    GPU speedup, with the crossover boundary highlighted.

    Args:
        all_results: List of dicts from both benchmarks and models.
        save_path: Where to save the figure.
    """
    setup_style()

    gpu_results = [r for r in all_results
                   if r.get('gpu_time') and r['gpu_time'] > 0]

    if not gpu_results:
        print("  ⚠ No GPU results — skipping crossover summary")
        return None

    fig, ax = plt.subplots(figsize=(10, 7))

    n_vals = [r['n_qubits'] for r in gpu_results]
    chi_vals = [r['chi'] for r in gpu_results]
    speedups = [r['cpu_time'] / r['gpu_time'] for r in gpu_results]

    sc = ax.scatter(
        n_vals, chi_vals, c=speedups, s=100,
        cmap='RdYlGn', norm=LogNorm(vmin=max(0.1, min(speedups)),
                                     vmax=max(2.0, max(speedups))),
        edgecolors='black', linewidths=0.5, zorder=5,
    )

    cbar = fig.colorbar(sc, ax=ax, label='GPU Speedup over CPU')

    # Highlight crossover region
    for n, chi, sp in zip(n_vals, chi_vals, speedups):
        if 0.8 < sp < 1.2:
            ax.scatter([n], [chi], s=200, facecolors='none',
                       edgecolors='black', linewidths=2, zorder=6)

    ax.set_xlabel('System Size N (qubits)', fontsize=13)
    ax.set_ylabel('Bond Dimension χ', fontsize=13)
    ax.set_yscale('log', base=2)
    ax.set_title('GPU MPS Crossover Threshold\n'
                 'Green = GPU wins, Red = CPU wins, '
                 'Circles = crossover region',
                 fontsize=13)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  📊 Saved: {save_path}")
    return save_path


# ─────────────────────────────────────────────────────────────────────
# JSON I/O helpers
# ─────────────────────────────────────────────────────────────────────

def save_results(results, filename):
    """Save benchmark results to JSON."""
    path = os.path.join(SCRIPT_DIR, filename)

    # Convert numpy types for JSON serialization
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serializable = []
    for r in results:
        serializable.append({k: _convert(v) for k, v in r.items()})

    with open(path, 'w') as f:
        json.dump(serializable, f, indent=2, default=_convert)

    print(f"  💾 Saved: {path}")
    return path


def load_results(filename):
    """Load benchmark results from JSON."""
    path = os.path.join(SCRIPT_DIR, filename)
    with open(path) as f:
        return json.load(f)
