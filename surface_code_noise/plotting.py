"""
Plotting functions for Surface Code QEC demo.

Separated from qec_demo.py for cleaner code structure.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_noise_comparison(results_list, noise_strengths, save_path,
                          d1_baseline=None):
    """
    Plot logical Z fidelity vs noise strength.

    Coherent noise degrades ⟨Z_L⟩ differently from Pauli noise.
    Optional d=1 baseline shows uncorrected single-qubit performance.
    """
    fig, ax = plt.subplots(figsize=(11, 7))

    palette_pauli = ['#4CAF50', '#2196F3', '#9C27B0']
    palette_coherent = ['#FF5722', '#E91E63', '#F44336']
    markers = ['o', 's', 'D']

    for i, results in enumerate(results_list):
        d = results.get('distance', '?')
        noise = results['noise_strengths']

        ax.plot(noise, results['pauli_logical_z'],
                f'{markers[i % 3]}--', color=palette_pauli[i % 3],
                linewidth=2, markersize=8, alpha=0.85,
                label=f'Pauli noise (d={d})')

        ax.plot(noise, results['coherent_logical_z'],
                f'{markers[i % 3]}-', color=palette_coherent[i % 3],
                linewidth=2.5, markersize=8,
                label=f'Coherent noise (d={d})')

    # d=1 uncorrected baseline
    if d1_baseline:
        noise = noise_strengths
        ax.plot(noise, d1_baseline['pauli_z'],
                'x:', color='#9E9E9E', linewidth=1.5, markersize=6,
                alpha=0.7, label='d=1 Pauli (uncorrected)')
        ax.plot(noise, d1_baseline['coherent_z'],
                '+:', color='#BDBDBD', linewidth=1.5, markersize=6,
                alpha=0.7, label='d=1 Coherent (uncorrected)')

    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.4,
               label='Ideal (no noise)')

    ax.set_xlabel('Noise Strength (ε = p)', fontsize=13)
    ax.set_ylabel('Logical Z Fidelity ⟨Z_L⟩', fontsize=13)
    ax.set_title(
        'Surface Code QEC: Coherent vs Pauli Noise\n'
        'Coherent errors accumulate constructively — '
        'invisible to Stim',
        fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='lower left')
    ax.grid(alpha=0.3)

    info = ("Pauli: random rotations (incoherent, partially cancel)\n"
            "Coherent: systematic over-rotations (accumulate)\n"
            "Gap = what Stim misses")
    ax.text(0.98, 0.02, info, transform=ax.transAxes,
            fontsize=9, va='bottom', ha='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF3E0',
                      edgecolor='#E65100', alpha=0.9))

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return save_path


def plot_spatial_errors(spatial_data, save_path):
    """
    Side-by-side heatmaps of per-data-qubit error deviation.

    Coherent noise shows structured spatial patterns;
    Pauli noise is more uniform.
    """
    model = spatial_data['model']
    d = model.d
    p = spatial_data['noise_strength']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Reshape data qubit deviations onto the d×d grid
    pauli_dev = np.abs(1.0 - spatial_data['pauli_z']).reshape(d, d)
    coherent_dev = np.abs(1.0 - spatial_data['coherent_z']).reshape(d, d)

    vmax = max(np.max(pauli_dev), np.max(coherent_dev)) * 1.1
    vmin = 0

    im1 = ax1.imshow(pauli_dev, cmap='YlOrRd', vmin=vmin, vmax=vmax,
                     aspect='equal')
    ax1.set_title(f'Pauli Noise (ε={p})\nRandom, uniform degradation',
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel('Column', fontsize=11)
    ax1.set_ylabel('Row', fontsize=11)
    plt.colorbar(im1, ax=ax1, label='Error deviation (1 − ⟨Z⟩)', shrink=0.8)

    # Annotate values
    for r in range(d):
        for c in range(d):
            ax1.text(c, r, f'{pauli_dev[r, c]:.3f}',
                     ha='center', va='center', fontsize=8,
                     color='white' if pauli_dev[r, c] > vmax * 0.6 else 'black')

    im2 = ax2.imshow(coherent_dev, cmap='YlOrRd', vmin=vmin, vmax=vmax,
                     aspect='equal')
    ax2.set_title(f'Coherent Noise (ε={p})\nStructured, correlated pattern',
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('Column', fontsize=11)
    ax2.set_ylabel('Row', fontsize=11)
    plt.colorbar(im2, ax=ax2, label='Error deviation (1 − ⟨Z⟩)', shrink=0.8)

    for r in range(d):
        for c in range(d):
            ax2.text(c, r, f'{coherent_dev[r, c]:.3f}',
                     ha='center', va='center', fontsize=8,
                     color='white' if coherent_dev[r, c] > vmax * 0.6 else 'black')

    fig.suptitle(
        f'Surface Code d={d}: Data Qubit Error Structure\n'
        f'Coherent noise creates spatial correlations invisible to '
        f'Pauli models',
        fontsize=14, fontweight='bold', y=1.04)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return save_path


def plot_backend_comparison(backend_results, distance, save_path):
    """
    Bar chart comparing all backends on the same noise circuit.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    names = list(backend_results.keys())
    logical_vals = [backend_results[n]['logical_z'] for n in names]
    times = [backend_results[n]['time_ms'] for n in names]

    colors = ['#2196F3', '#4CAF50', '#FF9800', '#FF5722']
    x = np.arange(len(names))

    # Left: Logical Z values
    ax1.bar(x, logical_vals, color=colors[:len(names)], alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=15, ha='right', fontsize=10)
    ax1.set_ylabel('Logical Z ⟨Z_L⟩', fontsize=12)
    ax1.set_title(f'Backend Comparison (d={distance}, p=0.06)\n'
                  f'Clifford backends agree; MPS adds coherent noise',
                  fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    for i, v in enumerate(logical_vals):
        ax1.text(i, v + 0.01, f'{v:.4f}', ha='center', fontsize=9,
                 fontweight='bold')

    # Right: Speed comparison (in milliseconds)
    ax2.bar(x, times, color=colors[:len(names)], alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=15, ha='right', fontsize=10)
    ax2.set_ylabel('Time (ms)', fontsize=12)
    ax2.set_title('Simulation Speed\n'
                  'Fast Clifford backends + full MPS — one tool',
                  fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    for i, t in enumerate(times):
        ax2.text(i, t + max(times) * 0.02, f'{t:.1f}ms', ha='center',
                 fontsize=9, fontweight='bold')

    fig.suptitle(
        'Maestro: All Simulation Backends in One Tool\n'
        'No need for Stim — Maestro does Clifford AND coherent noise',
        fontsize=15, fontweight='bold', y=1.04)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return save_path


def plot_distance_scaling(results_list, noise_strengths, save_path):
    """
    Plot the gap between Pauli and coherent fidelity across distances.

    Shows the coherent noise penalty with code distance.
    """
    if len(results_list) < 2:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Fidelity at fixed noise strength
    fixed_idx = min(3, len(noise_strengths) - 1)
    fixed_p = noise_strengths[fixed_idx]

    distances = []
    pauli_vals = []
    coherent_vals = []
    gaps = []

    for r in results_list:
        d = r['distance']
        distances.append(d)
        pauli_vals.append(r['pauli_logical_z'][fixed_idx])
        coherent_vals.append(r['coherent_logical_z'][fixed_idx])
        gaps.append(r['pauli_logical_z'][fixed_idx] -
                    r['coherent_logical_z'][fixed_idx])

    x = np.arange(len(distances))
    width = 0.35

    ax1.bar(x - width / 2, pauli_vals, width, color='#4CAF50',
            alpha=0.85, label='Pauli noise')
    ax1.bar(x + width / 2, coherent_vals, width, color='#FF5722',
            alpha=0.85, label='Coherent noise')

    ax1.set_xticks(x)
    ax1.set_xticklabels([f'd={d}\n({2*d**2-1}Q)' for d in distances])
    ax1.set_ylabel('Logical Z Fidelity ⟨Z_L⟩', fontsize=12)
    ax1.set_title(
        f'Logical Fidelity at ε=p={fixed_p}\n'
        f'Coherent noise is consistently worse',
        fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)

    # Right: gap vs distance
    ax2.bar(x, gaps, color='#E91E63', alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'd={d}' for d in distances])
    ax2.set_ylabel('Fidelity Gap (Pauli − Coherent)', fontsize=12)
    ax2.set_title('The Coherent Noise Penalty\n'
                  'Error grows with the number of CX gates',
                  fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    for i, gap in enumerate(gaps):
        ax2.text(i, gap + max(gaps) * 0.02, f'{gap:.4f}',
                 ha='center', fontsize=10, fontweight='bold',
                 color='#880E4F')

    fig.suptitle(
        'Surface Code: Why Coherent Noise Simulation Matters',
        fontsize=15, fontweight='bold', y=1.02)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return save_path
