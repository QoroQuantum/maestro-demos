#!/usr/bin/env python3
"""
Adaptive Bond Dimension — CPU↔GPU Handoff for Time Evolution
=============================================================

Demonstrates how Maestro makes it trivial to switch between CPU and
GPU backends during MPS time evolution. The key insight:

  • At low bond dimension (χ), CPU is faster — no GPU transfer overhead
  • At high χ, GPU wins — tensor contractions benefit from parallelism
  • Entanglement grows during time evolution → χ must increase
  • Maestro lets you switch with a single argument change

This example evolves a transverse-field Ising model (TFIM) on a 2D lattice:

  H = -J Σ Z_i Z_j  -  h Σ X_i

and monitors energy E(t). When entanglement growth causes the energy
estimate to change rapidly, we switch from low-χ CPU to high-χ GPU.

Usage:
    python adaptive_mps.py                 # CPU only (compare χ values)
    python adaptive_mps.py --gpu           # with CPU→GPU handoff
    python adaptive_mps.py --large         # 8×8 = 64 qubits
    python adaptive_mps.py --large --gpu   # 64 qubits with GPU
"""

import sys
import os
import time
import json

import numpy as np
import maestro
from maestro.circuits import QuantumCircuit

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ─────────────────────────────────────────────────────────────────────
# Lattice and circuit helpers
# ─────────────────────────────────────────────────────────────────────

def get_nn_bonds(lx, ly):
    """Nearest-neighbor bonds on a 2D square lattice."""
    bonds = []
    for x in range(lx):
        for y in range(ly):
            q = x * ly + y
            if x + 1 < lx:
                bonds.append((q, (x + 1) * ly + y))
            if y + 1 < ly:
                bonds.append((q, q + 1))
    return bonds


def build_pauli_observable(n_qubits, pauli_map):
    """Build a Pauli observable string like 'IIZZI...'."""
    labels = ['I'] * n_qubits
    for qubit, pauli in pauli_map.items():
        labels[qubit] = pauli
    return ''.join(labels)


def build_tfim_circuit(n, bonds, j, h, dt, n_steps):
    """Build a Trotterized TFIM circuit."""
    qc = QuantumCircuit()
    # Initial state: |+⟩^n
    for q in range(n):
        qc.h(q)
    # Trotter steps
    for _ in range(n_steps):
        for q1, q2 in bonds:
            qc.cx(q1, q2)
            qc.rz(q2, 2.0 * j * dt)
            qc.cx(q1, q2)
        for q in range(n):
            qc.h(q)
            qc.rz(q, 2.0 * h * dt)
            qc.h(q)
    return qc


def compute_energy(qc, n, bonds, j, h, chi, simulator_type, use_gpu):
    """Compute TFIM energy via MPS estimate()."""
    obs = []
    for q1, q2 in bonds:
        obs.append(build_pauli_observable(n, {q1: 'Z', q2: 'Z'}))
    for q in range(n):
        obs.append(build_pauli_observable(n, {q: 'X'}))

    sim_type = (maestro.SimulatorType.Gpu if use_gpu
                else maestro.SimulatorType.QCSim)

    result = qc.estimate(obs, maestro.SimulatorConfig(
        simulator_type=sim_type,
        simulation_type=maestro.SimulationType.MatrixProductState,
        max_bond_dimension=chi,
    ))

    exp_vals = result['expectation_values']
    n_bonds = len(bonds)
    e_zz = sum(-j * exp_vals[i] for i in range(n_bonds))
    e_x = sum(-h * exp_vals[n_bonds + i] for i in range(n))
    return e_zz + e_x


# ─────────────────────────────────────────────────────────────────────
# Experiment 1: Fixed χ comparison (CPU vs GPU)
# ─────────────────────────────────────────────────────────────────────

def run_fixed_chi(lx, ly, chi, n_steps, dt, j, h, use_gpu, label=""):
    """Run time evolution at a fixed χ, return energies and timing."""
    n = lx * ly
    bonds = get_nn_bonds(lx, ly)

    if label:
        print(f"\n  ── {label} ──")
    print(f"    χ={chi}, {'GPU' if use_gpu else 'CPU'}, "
          f"{n_steps} steps, dt={dt}")

    energies = []
    times = []
    step_durations = []

    for step in range(n_steps + 1):
        t0 = time.time()

        if step == 0:
            energy = -n  # Initial |+⟩ state energy
        else:
            qc = build_tfim_circuit(n, bonds, j, h, dt, step)
            energy = compute_energy(qc, n, bonds, j, h, chi,
                                    maestro.SimulatorType.QCSim, use_gpu)
        wall = time.time() - t0

        energies.append(energy)
        times.append(step * dt)
        step_durations.append(wall)

        if step % max(1, n_steps // 5) == 0:
            print(f"    step {step:3d}  t={step*dt:.2f}  "
                  f"E={energy:10.4f}  ({wall:.3f}s)")

    avg = np.mean(step_durations[1:]) if len(step_durations) > 1 else 0
    total = sum(step_durations)
    print(f"    ✓ Done: {total:.1f}s total, {avg:.3f}s/step avg")

    return {
        'energies': energies, 'times': times,
        'step_durations': step_durations,
        'total_time': total, 'avg_step_time': avg,
        'chi': chi, 'use_gpu': use_gpu, 'label': label,
    }


# ─────────────────────────────────────────────────────────────────────
# Experiment 2: Adaptive χ with handoff
# ─────────────────────────────────────────────────────────────────────

def run_adaptive(lx, ly, chi_low, chi_high, n_steps, dt, j, h,
                 use_gpu_high=False, threshold=0.5):
    """
    Time-evolve with adaptive bond dimension.
    Start at chi_low (CPU). When energy change exceeds threshold,
    switch to chi_high (GPU if available).
    """
    n = lx * ly
    bonds = get_nn_bonds(lx, ly)

    low_label = f"CPU χ={chi_low}"
    high_label = (f"GPU χ={chi_high}" if use_gpu_high
                  else f"CPU χ={chi_high}")

    print(f"\n  ── ADAPTIVE: {low_label} → {high_label} ──")
    print(f"    Handoff when |ΔE| > {threshold}")

    energies, times, step_durations, backends = [], [], [], []
    switched = False

    for step in range(n_steps + 1):
        chi = chi_low if not switched else chi_high
        use_gpu = False if not switched else use_gpu_high
        backend = 'low' if not switched else 'high'

        t0 = time.time()
        if step == 0:
            energy = -n
        else:
            qc = build_tfim_circuit(n, bonds, j, h, dt, step)
            energy = compute_energy(qc, n, bonds, j, h, chi,
                                    maestro.SimulatorType.QCSim, use_gpu)
        wall = time.time() - t0

        energies.append(energy)
        times.append(step * dt)
        step_durations.append(wall)
        backends.append(backend)

        if step % max(1, n_steps // 5) == 0:
            label = low_label if not switched else high_label
            print(f"    step {step:3d}  t={step*dt:.2f}  "
                  f"E={energy:10.4f}  {label}  ({wall:.3f}s)")

        if not switched and step >= 2:
            if abs(energies[-1] - energies[-2]) > threshold:
                switched = True
                print(f"\n    ⚡ HANDOFF at step {step} (t={step*dt:.2f}): "
                      f"{low_label} → {high_label}\n")

    total = sum(step_durations)
    low_steps = sum(1 for b in backends if b == 'low')
    high_steps = sum(1 for b in backends if b == 'high')
    low_time = sum(t for t, b in zip(step_durations, backends)
                   if b == 'low' and t > 0)
    high_time = sum(t for t, b in zip(step_durations, backends)
                    if b == 'high' and t > 0)

    print(f"    ✓ Done: {total:.1f}s total")
    print(f"      Low-χ:  {low_steps} steps, {low_time:.1f}s")
    print(f"      High-χ: {high_steps} steps, {high_time:.1f}s")

    return {
        'energies': energies, 'times': times,
        'step_durations': step_durations, 'backends': backends,
        'total_time': total, 'chi_low': chi_low, 'chi_high': chi_high,
        'use_gpu_high': use_gpu_high,
    }


# ─────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────

def plot_comparison(results_list, adaptive_result, config, path):
    """Compare fixed-χ runs and the adaptive run."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Left: Energy vs time
    ax1 = axes[0]
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63']

    for i, r in enumerate(results_list):
        c = colors[i % len(colors)]
        gpu_str = 'GPU' if r['use_gpu'] else 'CPU'
        ax1.plot(r['times'], r['energies'], '--',
                 color=c, linewidth=1.5, alpha=0.6,
                 label=f'{gpu_str} χ={r["chi"]}')

    # Adaptive run: color-code by backend
    if adaptive_result:
        ar = adaptive_result
        for i in range(1, len(ar['times'])):
            c = '#2196F3' if ar['backends'][i] == 'low' else '#E91E63'
            ax1.plot(ar['times'][i-1:i+1], ar['energies'][i-1:i+1], '-',
                     color=c, linewidth=3)
        # Legend entries
        ax1.plot([], [], '-', color='#2196F3', linewidth=3,
                 label=f'Adaptive low (CPU χ={ar["chi_low"]})')
        high_lbl = 'GPU' if ar['use_gpu_high'] else 'CPU'
        ax1.plot([], [], '-', color='#E91E63', linewidth=3,
                 label=f'Adaptive high ({high_lbl} χ={ar["chi_high"]})')

    ax1.set_xlabel('Simulation Time t', fontsize=12)
    ax1.set_ylabel('Energy E(t)', fontsize=12)
    ax1.set_title(f'Time Evolution — {config["lx"]}×{config["ly"]} TFIM',
                  fontsize=14)
    ax1.legend(fontsize=8, loc='lower left')
    ax1.grid(alpha=0.3)

    # Right: Timing per step
    ax2 = axes[1]
    bar_data = []
    bar_labels = []

    for r in results_list:
        gpu_str = 'GPU' if r['use_gpu'] else 'CPU'
        bar_data.append(r['avg_step_time'])
        bar_labels.append(f'{gpu_str}\nχ={r["chi"]}')

    if adaptive_result:
        ar = adaptive_result
        high_lbl = 'GPU' if ar['use_gpu_high'] else 'CPU'
        bar_data.append(np.mean(ar['step_durations'][1:]))
        bar_labels.append(f'Adaptive\n{ar["chi_low"]}→{ar["chi_high"]}')

    x = np.arange(len(bar_data))
    bar_colors = [colors[i % len(colors)] for i in range(len(results_list))]
    if adaptive_result:
        bar_colors.append('#9C27B0')

    bars = ax2.bar(x, bar_data, color=bar_colors, edgecolor='black',
                   alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(bar_labels, fontsize=10)
    ax2.set_ylabel('Avg Time per Step (s)', fontsize=12)
    ax2.set_title('Per-Step Cost Comparison', fontsize=14)
    ax2.grid(alpha=0.3, axis='y')

    # Annotate bars
    for bar, val in zip(bars, bar_data):
        ax2.annotate(f'{val:.3f}s', (bar.get_x() + bar.get_width()/2, val),
                     textcoords='offset points', xytext=(0, 5),
                     ha='center', fontsize=9, fontweight='bold')

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    use_gpu = '--gpu' in sys.argv
    large = '--large' in sys.argv

    if large:
        lx, ly = 8, 8
        chi_low, chi_high = 16, 128
        n_steps = 10
    else:
        lx, ly = 6, 6
        chi_low, chi_high = 16, 64
        n_steps = 10

    n = lx * ly
    j, h, dt = 1.0, 1.0, 0.2

    print(f"\n{'═'*60}")
    print(f"  ADAPTIVE BOND DIMENSION — MPS TIME EVOLUTION")
    print(f"{'═'*60}")
    print(f"  Lattice: {lx}×{ly} = {n} qubits")
    print(f"  Model:   TFIM  J={j}  h={h}")
    print(f"  Time:    T={n_steps*dt:.1f}, {n_steps} steps, dt={dt}")
    print(f"  χ:       low={chi_low}, high={chi_high}")
    print(f"  GPU:     {'Available' if use_gpu else 'CPU only'}")

    results_list = []

    # Run 1: Low χ on CPU (fast but approximate)
    r_low = run_fixed_chi(lx, ly, chi_low, n_steps, dt, j, h,
                          use_gpu=False, label="LOW χ (CPU)")
    results_list.append(r_low)

    # Run 2: High χ on CPU (accurate but slower)
    r_high_cpu = run_fixed_chi(lx, ly, chi_high, n_steps, dt, j, h,
                               use_gpu=False, label="HIGH χ (CPU)")
    results_list.append(r_high_cpu)

    # Run 3: High χ on GPU (if available — should be faster)
    r_high_gpu = None
    if use_gpu:
        r_high_gpu = run_fixed_chi(lx, ly, chi_high, n_steps, dt, j, h,
                                   use_gpu=True, label="HIGH χ (GPU)")
        results_list.append(r_high_gpu)

    # Run 4: Adaptive — start low, switch to high
    r_adaptive = run_adaptive(
        lx, ly, chi_low, chi_high, n_steps, dt, j, h,
        use_gpu_high=use_gpu, threshold=0.5,
    )

    # Plot
    config = {'lx': lx, 'ly': ly}
    plot_path = plot_comparison(
        results_list, r_adaptive, config,
        os.path.join(SCRIPT_DIR, 'adaptive_comparison.png'),
    )
    print(f"\n  📊 Saved: {plot_path}")

    # Summary
    print(f"\n{'═'*60}")
    print(f"  SUMMARY")
    print(f"{'═'*60}")
    print(f"\n  {lx}×{ly} = {n} qubits, {n_steps} Trotter steps\n")
    print(f"  {'Method':<25} {'Avg Step':>10} {'Total':>10}")
    print(f"  {'─'*47}")

    for r in results_list:
        gpu_str = 'GPU' if r['use_gpu'] else 'CPU'
        name = f"{gpu_str} χ={r['chi']}"
        print(f"  {name:<25} {r['avg_step_time']:>9.3f}s "
              f"{r['total_time']:>9.1f}s")

    high_lbl = 'GPU' if use_gpu else 'CPU'
    avg_adaptive = np.mean(r_adaptive['step_durations'][1:])
    print(f"  {'Adaptive ('+str(chi_low)+'→'+str(chi_high)+')':<25} "
          f"{avg_adaptive:>9.3f}s {r_adaptive['total_time']:>9.1f}s")

    # Show speedup
    if use_gpu and r_high_gpu:
        speedup = r_high_cpu['avg_step_time'] / max(r_high_gpu['avg_step_time'], 1e-9)
        print(f"\n  GPU speedup at χ={chi_high}: {speedup:.1f}×")

    adaptive_saving = 1.0 - r_adaptive['total_time'] / r_high_cpu['total_time']
    print(f"  Adaptive saves {adaptive_saving:.0%} vs running high-χ everywhere")

    print(f"\n  The key: Maestro lets you switch backends with one argument.")
    print(f"  No code rewrite needed — just change simulator_type and χ.\n")
