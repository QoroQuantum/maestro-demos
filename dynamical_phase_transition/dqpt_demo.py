#!/usr/bin/env python3
"""
Dynamical Quantum Phase Transition — Loschmidt Echo Cusps
==========================================================

Demonstrates dynamical quantum phase transitions (DQPTs) in the 1D
transverse-field Ising model (TFIM) after a sudden quench.

The Hamiltonian is:
    H = −J Σ ZᵢZᵢ₊₁  −  h Σ Xᵢ

Protocol:
  1. Prepare |ψ₀⟩ = |+⟩^⊗N  (ground state of H at h → ∞)
  2. Quench to H(h_f) with h_f < J (into the ferromagnetic phase)
  3. Track the Loschmidt rate function:
        λ(t) = −(1/N) ln|G(t)|²
     where G(t) = ⟨ψ₀|e^{-iHt}|ψ₀⟩ is the Loschmidt amplitude.

When the quench crosses the equilibrium critical point h_c = J,
the rate function λ(t) develops non-analytic cusps at critical times
— dynamical analogs of free-energy singularities.

The cusps occur at:
    t*_n = (n + 1/2) π / ε_{k*}

where k* is the critical momentum and ε_{k*} the quasiparticle energy.

Key observables:
  - Loschmidt rate function λ(t) — cusp singularities
  - Per-site magnetization ⟨Xᵢ(t)⟩ — order parameter oscillations
  - Comparison across quench strengths h_f

Output:
    dqpt_loschmidt.png — Rate function cusps + magnetization dynamics

Usage:
    python dqpt_demo.py               # 30-qubit chain, CPU
    python dqpt_demo.py --gpu         # GPU-accelerated MPS
    python dqpt_demo.py --large       # 80-qubit chain
    python dqpt_demo.py --large --gpu # 80 qubits + GPU
"""

import sys
import os
import time

import numpy as np
import maestro
from maestro.circuits import QuantumCircuit

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ─────────────────────────────────────────────────────────────────────
# Circuit construction
# ─────────────────────────────────────────────────────────────────────

def build_quench_circuit(n_qubits, j, h_final, dt, n_steps):
    """
    Build a Trotterized TFIM circuit for quench dynamics.

    Initial state: |+⟩^⊗N (all qubits in |+⟩)
    Evolution under: H = −J Σ ZᵢZᵢ₊₁ − h_final Σ Xᵢ

    First-order Trotter decomposition with nearest-neighbor ZZ and
    single-qubit X terms.

    Args:
        n_qubits: Number of qubits in the 1D chain.
        j: Ising coupling strength J.
        h_final: Post-quench transverse field h_f.
        dt: Trotter time step.
        n_steps: Number of Trotter steps.

    Returns:
        A QuantumCircuit instance.
    """
    qc = QuantumCircuit()

    # Initial state: |+⟩^⊗N
    for q in range(n_qubits):
        qc.h(q)

    # Trotter steps
    for _ in range(n_steps):
        # ZZ interactions: exp(i J dt ZᵢZᵢ₊₁) via CX-Rz-CX
        # Even bonds first, then odd bonds
        for q in range(0, n_qubits - 1, 2):
            qc.cx(q, q + 1)
            qc.rz(q + 1, 2.0 * j * dt)
            qc.cx(q, q + 1)

        for q in range(1, n_qubits - 1, 2):
            qc.cx(q, q + 1)
            qc.rz(q + 1, 2.0 * j * dt)
            qc.cx(q, q + 1)

        # Transverse field: exp(i h dt Xᵢ) = H·Rz(2h·dt)·H
        for q in range(n_qubits):
            qc.h(q)
            qc.rz(q, 2.0 * h_final * dt)
            qc.h(q)

    return qc


def build_x_observables(n_qubits):
    """Build per-qubit X observables: ['XIII..', 'IXII..', ...]"""
    obs = []
    for i in range(n_qubits):
        pauli = ['I'] * n_qubits
        pauli[i] = 'X'
        obs.append("".join(pauli))
    return obs


# ─────────────────────────────────────────────────────────────────────
# Measurements
# ─────────────────────────────────────────────────────────────────────

def compute_loschmidt_rate(x_expects, n_qubits):
    """
    Compute the Loschmidt rate function from per-site ⟨Xᵢ⟩ values.

    For the initial state |+⟩^⊗N, the return probability factorizes
    in the 1D TFIM as:
        |G(t)|² ≈ Πᵢ pᵢ(t)
    where pᵢ(t) = (1 + ⟨Xᵢ(t)⟩) / 2 is the probability that qubit i
    is still in |+⟩.

    The rate function is:
        λ(t) = −(1/N) ln|G(t)|² = −(1/N) Σᵢ ln(pᵢ)

    Args:
        x_expects: List of ⟨Xᵢ(t)⟩ expectation values.
        n_qubits: Number of qubits.

    Returns:
        (rate_function, avg_x)
    """
    log_return_prob = 0.0
    for x_val in x_expects:
        p_i = (1.0 + x_val) / 2.0
        p_i = max(p_i, 1e-15)  # numerical floor
        log_return_prob += np.log(p_i)

    rate_function = -log_return_prob / n_qubits
    avg_x = np.mean(x_expects)
    return rate_function, avg_x


# ─────────────────────────────────────────────────────────────────────
# Exact DQPT prediction (analytic, for reference)
# ─────────────────────────────────────────────────────────────────────

def exact_loschmidt_rate_tfim(j, h_f, times):
    """
    Compute the exact Loschmidt rate function for the 1D TFIM quench
    in the thermodynamic limit.

    For quench from h_i → ∞ (|+⟩ state) to h_f:
        λ(t) = −(1/π) ∫₀^π dk  ln[ cos²(εₖt) + sin²(εₖt) cos²(Δₖ) ]

    where εₖ = 2√((J cos k + h_f)² + (J sin k)²) is the post-quench
    dispersion, and Δₖ is the Bogoliubov angle difference.

    Args:
        j: Ising coupling J.
        h_f: Post-quench field.
        times: Array of time values.

    Returns:
        Array of λ(t) values.
    """
    n_k = 500
    k_vals = np.linspace(0.001, np.pi - 0.001, n_k)

    rates = np.zeros(len(times))

    for idx, t in enumerate(times):
        integrand = 0.0
        for k in k_vals:
            eps_k = 2.0 * np.sqrt((j * np.cos(k) + h_f)**2
                                   + (j * np.sin(k))**2)

            cos_theta_f = (j * np.cos(k) + h_f) / (eps_k / 2.0)

            # For h_i → ∞, cos(Δθ) = cos(θ_f)
            val = (np.cos(eps_k * t / 2.0))**2 + \
                  (np.sin(eps_k * t / 2.0) * cos_theta_f)**2
            val = max(val, 1e-30)
            integrand += -np.log(val)

        rates[idx] = integrand / n_k

    return rates


# ─────────────────────────────────────────────────────────────────────
# Main experiment
# ─────────────────────────────────────────────────────────────────────

def run_dqpt_sweep(n_qubits, j, h_values, n_steps, dt, chi, use_gpu):
    """
    Sweep over multiple quench field values, tracking λ(t) for each.

    Returns:
        List of result dicts, one per h_f value.
    """
    sim_type = (maestro.SimulatorType.Gpu if use_gpu
                else maestro.SimulatorType.QCSim)

    x_observables = build_x_observables(n_qubits)
    all_results = []

    for h_idx, h_f in enumerate(h_values):
        print(f"\n  ── Quench {h_idx+1}/{len(h_values)}: "
              f"h_f = {h_f:.2f} (h_f/J = {h_f/j:.2f}) ──")

        results = {
            'h_f': h_f, 'times': [], 'rate': [], 'avg_x': [],
            'step_durations': [],
        }

        # Step 0: initial |+⟩ state → λ=0, ⟨X⟩=1
        results['times'].append(0.0)
        results['rate'].append(0.0)
        results['avg_x'].append(1.0)
        results['step_durations'].append(0.0)

        for step in range(1, n_steps + 1):
            t0 = time.time()

            qc = build_quench_circuit(n_qubits, j, h_f, dt, step)

            res = qc.estimate(x_observables, maestro.SimulatorConfig(
                simulator_type=sim_type,
                simulation_type=maestro.SimulationType.MatrixProductState,
                max_bond_dimension=chi,
            ))

            wall = time.time() - t0
            t_val = step * dt

            x_expects = res['expectation_values']
            rate, avg_x = compute_loschmidt_rate(x_expects, n_qubits)

            results['times'].append(t_val)
            results['rate'].append(rate)
            results['avg_x'].append(avg_x)
            results['step_durations'].append(wall)

            if step % max(1, n_steps // 8) == 0 or step <= 2:
                print(f"    step {step:3d}  t={t_val:.2f}  "
                      f"λ={rate:.4f}  ⟨X⟩={avg_x:+.4f}  ({wall:.2f}s)")

        total = sum(results['step_durations'])
        print(f"    ✓ Done ({total:.1f}s)")
        all_results.append(results)

    return all_results


# ─────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────

def plot_dqpt(all_results, config, exact_dense, path):
    """
    Two-panel plot:
      Left:  Loschmidt rate function λ(t) with cusps
      Right: Average magnetization ⟨X(t)⟩
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    colors = ['#E91E63', '#1565C0', '#4CAF50', '#FF9800', '#9C27B0']

    # ── Left: Rate function ──
    ax1 = axes[0]

    for i, res in enumerate(all_results):
        c = colors[i % len(colors)]
        label = f'h_f/J = {res["h_f"]:.1f}'
        ax1.plot(res['times'], res['rate'], 'o', color=c,
                 markersize=5, alpha=0.8, label=f'MPS ({label})')

    # Dense exact reference curves (smooth lines)
    for i, (h_f, exact_t, exact_r) in enumerate(exact_dense):
        c = colors[i % len(colors)]
        ax1.plot(exact_t, exact_r, '-', color=c,
                 linewidth=2, alpha=0.5, label=f'Exact ({f"h_f/J = {h_f:.1f}"})')

    ax1.set_xlabel('Time t', fontsize=13)
    ax1.set_ylabel('Loschmidt Rate Function λ(t)', fontsize=13)
    ax1.set_title('Dynamical Quantum Phase Transition\n'
                  'Non-Analytic Cusps in λ(t)',
                  fontsize=14)
    ax1.legend(fontsize=8, loc='upper left', ncol=2)
    ax1.grid(alpha=0.3)

    # Info box
    info = (f"N = {config['n_qubits']} qubits, J = {config['j']}\n"
            f"χ = {config['chi']}, "
            f"{'GPU' if config['use_gpu'] else 'CPU'}\n"
            f"Quench: |+⟩ → H(h_f < J)")
    ax1.text(0.98, 0.02, info, transform=ax1.transAxes,
             fontsize=9, verticalalignment='bottom',
             horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                       edgecolor='gray', alpha=0.9))

    # ── Right: Magnetization ──
    ax2 = axes[1]

    for i, res in enumerate(all_results):
        c = colors[i % len(colors)]
        ax2.plot(res['times'], res['avg_x'], 'o-', color=c,
                 linewidth=2, markersize=4,
                 label=f'h_f/J = {res["h_f"]:.1f}')

    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.4)
    ax2.set_xlabel('Time t', fontsize=13)
    ax2.set_ylabel('Average ⟨X(t)⟩', fontsize=13)
    ax2.set_title('Transverse Magnetization\nOrder Parameter Dynamics',
                  fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)

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
        n_qubits = 80
        chi = 64
        n_steps = 40
        dt = 0.1
    else:
        n_qubits = 30
        chi = 32
        n_steps = 40
        dt = 0.1

    j = 1.0                              # Ising coupling (sets energy scale)
    T_total = n_steps * dt

    # Quench field values: subcritical (h_f < J) → DQPT occurs
    h_values = [0.2, 0.5, 0.8]

    print(f"\n{'═'*65}")
    print(f"  MAESTRO Demo: Dynamical Quantum Phase Transition")
    print(f"  Loschmidt Echo Cusps after TFIM Quench")
    print(f"{'═'*65}")
    print(f"\n  System:      {n_qubits}-qubit 1D chain")
    print(f"  Hamiltonian: H = −J Σ ZᵢZᵢ₊₁ − h Σ Xᵢ  (J = {j})")
    print(f"  Time:        T = {T_total:.1f}, {n_steps} steps, dt = {dt}")
    print(f"  MPS:         χ = {chi}")
    print(f"  Backend:     {'GPU' if use_gpu else 'CPU (QCSim)'}")
    print(f"\n  Protocol:")
    print(f"    1. Prepare |ψ₀⟩ = |+⟩^⊗{n_qubits}  "
          f"(paramagnetic ground state)")
    print(f"    2. Quench to h_f = {h_values}  "
          f"(into ferromagnetic regime)")
    print(f"    3. Track Loschmidt rate function λ(t)")
    print(f"\n  Critical point: h_c = J = {j}")
    print(f"  All quenches cross h_c → DQPTs expected!")

    start_time = time.time()

    # Run the sweep
    all_results = run_dqpt_sweep(
        n_qubits, j, h_values, n_steps, dt, chi, use_gpu
    )

    # Compute dense exact reference curves for smooth plotting
    print(f"\n  Computing exact analytic reference (dense)...")
    t_dense = np.linspace(0, T_total, 300)
    exact_dense = []
    for h_f in h_values:
        exact_r = exact_loschmidt_rate_tfim(j, h_f, t_dense)
        exact_dense.append((h_f, t_dense, exact_r))
    print(f"  ✓ Exact reference computed")

    config = {
        'n_qubits': n_qubits, 'j': j, 'chi': chi,
        'use_gpu': use_gpu,
    }

    plot_path = plot_dqpt(
        all_results, config, exact_dense,
        os.path.join(SCRIPT_DIR, 'dqpt_loschmidt.png'),
    )
    print(f"\n  📊 Saved: {plot_path}")

    # ── Summary ──
    total = time.time() - start_time
    print(f"\n{'═'*65}")
    print(f"  SUMMARY")
    print(f"{'═'*65}")
    print(f"  Total runtime: {total:.1f}s")
    print(f"\n  {'h_f':>6}  {'h_f/J':>6}  {'λ_max':>8}  "
          f"{'⟨X⟩_min':>8}  {'Time':>8}")
    print(f"  {'─'*42}")

    for res in all_results:
        rate_arr = np.array(res['rate'])
        x_arr = np.array(res['avg_x'])
        run_time = sum(res['step_durations'])
        print(f"  {res['h_f']:>6.2f}  {res['h_f']/j:>6.2f}  "
              f"{max(rate_arr):>8.4f}  {min(x_arr):>8.4f}  "
              f"{run_time:>7.1f}s")

    print(f"\n  Physics:")
    print(f"  The cusps in λ(t) are dynamical analogs of thermodynamic")
    print(f"  free-energy singularities. They occur when the Loschmidt")
    print(f"  amplitude G(t) = ⟨ψ₀|e^{{-iHt}}|ψ₀⟩ passes through zero")
    print(f"  in the complex plane — a \"Fisher zero\" crossing the real")
    print(f"  time axis.\n")
