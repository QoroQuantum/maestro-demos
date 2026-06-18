#!/usr/bin/env python3
"""
Rydberg Atom Array — Spatial Correlations via Sampling
======================================================

Demonstrates Maestro's MPS sampling mode (execute) to measure the spatial
connected correlation function C(r) in a Z2-ordered Rydberg atom array.

Key Concept:
    Computing ⟨nᵢ nⱼ⟩ via expectation values would require O(N²) separate
    observables. Sampling does it in ONE run — extract all pairwise
    correlations from a single set of bitstrings.

The connected correlation function is:
    C(r) = ⟨nᵢ nᵢ₊ᵣ⟩ − ⟨nᵢ⟩ ⟨nᵢ₊ᵣ⟩

We compute the rectified version (-1)^r C(r) to reveal long-range Z2 order.

Output:
    rydberg_correlations.png  —  Spatial correlation decay plot

Usage:
    python rydberg_correlations.py
"""

import os
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import maestro
from rydberg_demo import create_rydberg_circuit

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    # ── Configuration ──
    N = 15                   # Manageable size for demo
    V_interaction = 10.0     # Strong blockade
    max_bond_dim = 32
    target_omega = 2.0       # Deep Z2 phase parameters
    target_delta = 5.0
    T_total = 15.0
    dt = 0.01
    steps = int(T_total / dt)
    num_shots = 2000

    print("=" * 65)
    print("  MAESTRO Demo: Quantum Correlations via Sampling")
    print("  Measuring spatial decay of crystalline order")
    print(f"  N={N} atoms, Ω={target_omega}, Δ={target_delta}")
    print("=" * 65)

    # ── Generate circuit ──
    qc = create_rydberg_circuit(
        N, target_omega, target_delta, V_interaction,
        steps=int(steps * 1.2), dt=dt
    )
    qc.measure_all()

    try:
        print(f"\n  Acquiring {num_shots} shots from MPS backend (χ={max_bond_dim})...")
        start_t = time.time()

        sample_res = qc.execute(maestro.SimulatorConfig(
            simulator_type=maestro.SimulatorType.QCSim,
            simulation_type=maestro.SimulationType.MatrixProductState,
            max_bond_dimension=max_bond_dim,
        ), shots=num_shots)

        print(f"  Sampling completed in {time.time() - start_t:.2f}s")

        if not (sample_res and 'counts' in sample_res):
            print("  Error: No counts returned.")
            return

        counts = sample_res['counts']

        # ── Parse bitstrings ──
        parsed_samples = []
        for state_str, count in counts.items():
            bits = [int(b) for b in reversed(state_str)]
            if len(bits) < N:
                bits = bits + [0] * (N - len(bits))
            for _ in range(count):
                parsed_samples.append(bits)

        samples_matrix = np.array(parsed_samples)  # (shots, N)
        densities = np.mean(samples_matrix, axis=0)

        # ── Compute connected correlation function C(r) ──
        print("  Computing spatial correlations C(r)...")
        max_r = N // 2
        correlations_r = np.zeros(max_r)

        for r in range(1, max_r + 1):
            c_r_sum = 0.0
            terms = 0

            for i in range(N - r):
                joint_expect = np.mean(
                    samples_matrix[:, i] * samples_matrix[:, i + r]
                )
                product_expect = densities[i] * densities[i + r]
                corr = (joint_expect - product_expect) * ((-1) ** r)
                c_r_sum += corr
                terms += 1

            correlations_r[r - 1] = c_r_sum / terms

        print("  Correlation decay computed.")

        # ── Visualization ──
        fig, ax = plt.subplots(figsize=(10, 6))
        r_vals = np.arange(1, max_r + 1)

        ax.plot(
            r_vals, correlations_r, 'o-',
            color='#00BCD4', label='Measured Correlation',
            linewidth=2, markersize=8
        )
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)

        ax.set_xlabel('Distance $r$ (sites)', fontsize=13)
        ax.set_ylabel(r'Rectified Correlation $(-1)^r C(r)$', fontsize=13)
        ax.set_title(
            f'Spatial Correlation Decay (N={N})\n'
            f'Long-Range Z2 Order via MPS Sampling ({num_shots} shots)',
            fontsize=14
        )
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=11)

        filename = os.path.join(SCRIPT_DIR, "rydberg_correlations.png")
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"\n  📊 Correlation plot saved: {filename}")
        print(
            "  Note: A flat positive line indicates the system forms a rigid "
            "crystal spanning the entire array."
        )

    except Exception as e:
        print(f"  Error: {e}")


if __name__ == "__main__":
    main()
