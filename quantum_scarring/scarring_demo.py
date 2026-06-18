#!/usr/bin/env python3
"""
Quantum Many-Body Scarring — PXP Model on a Rydberg Chain
==========================================================

Simulates the anomalous thermalization dynamics of a 1D Rydberg atom chain
in the PXP limit (perfect nearest-neighbor blockade).

The PXP Hamiltonian is:
    H_PXP = Σᵢ  Pᵢ₋₁ Xᵢ Pᵢ₊₁

where Pᵢ = |0⟩⟨0|ᵢ = (I + Zᵢ)/2 projects onto the ground state of atom i.
This enforces the Rydberg blockade: no two adjacent atoms can be excited.

We approximate the PXP dynamics using the Rydberg Hamiltonian with strong
nearest-neighbor interaction V ≫ Ω:

    H = (Ω/2) Σ Xᵢ  +  V Σ nᵢ nᵢ₊₁

In the limit V/Ω → ∞, the dynamics within the constrained subspace reproduce
the PXP model.

Starting from the Néel state |Z₂⟩ = |01010…⟩, the system exhibits
**quantum many-body scarring**: persistent oscillations in the staggered
magnetization M(t) instead of rapid thermalization to M=0.

The key observable is the staggered magnetization:
    M(t) = (1/N) Σᵢ (-1)ⁱ ⟨Zᵢ(t)⟩

In a thermalizing system, M(t) decays monotonically to 0. With scarring,
M(t) oscillates persistently — a signature of non-thermal eigenstates.

For small systems (--small flag), exact diagonalization provides both
M(t) and the true quantum fidelity F(t) = |⟨Z₂|ψ(t)⟩|² for validation.

Output:
    quantum_scarring.png — Staggered magnetization revivals + ED fidelity

Usage:
    python scarring_demo.py               # 32 atoms, CPU
    python scarring_demo.py --gpu         # GPU-accelerated MPS
    python scarring_demo.py --large       # 64 atoms
    python scarring_demo.py --large --gpu # 64 atoms + GPU
    python scarring_demo.py --small       # 12 atoms + exact ED validation
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

def build_pxp_circuit(n_atoms, omega, interaction_v, dt, n_steps):
    """
    Build a Trotterized circuit for Rydberg dynamics in the PXP limit.

    The circuit implements first-order Trotter decomposition of:
        H = (Ω/2) Σ Xᵢ  +  V Σ nᵢ nᵢ₊₁

    with the initial state |Z₂⟩ = |01010…⟩ (Néel state).

    Args:
        n_atoms: Number of atoms in the 1D chain.
        omega: Rabi frequency Ω.
        interaction_v: Nearest-neighbor interaction strength V.
        dt: Trotter time step.
        n_steps: Number of Trotter steps.

    Returns:
        A QuantumCircuit instance.
    """
    qc = QuantumCircuit()

    # ── Initial state: Néel state |01010…⟩ ──
    for i in range(1, n_atoms, 2):
        qc.x(i)

    # ── Trotter steps ──
    for _ in range(n_steps):
        # ZZ interactions (nn blockade): exp(-i V dt nᵢ nᵢ₊₁)
        # nᵢ nᵢ₊₁ = (I - Zᵢ - Zᵢ₊₁ + ZᵢZᵢ₊₁) / 4
        # ZZ term: exp(-i θ ZᵢZᵢ₊₁) with θ = V·dt/4
        zz_theta = interaction_v * dt / 4.0

        # Even bonds
        for i in range(0, n_atoms - 1, 2):
            qc.cx(i, i + 1)
            qc.rz(i + 1, 2.0 * zz_theta)
            qc.cx(i, i + 1)

        # Odd bonds
        for i in range(1, n_atoms - 1, 2):
            qc.cx(i, i + 1)
            qc.rz(i + 1, 2.0 * zz_theta)
            qc.cx(i, i + 1)

        # Single-qubit rotations
        for i in range(n_atoms):
            # Drive: exp(-i (Ω/2) dt Xᵢ)
            qc.rx(i, omega * dt)

            # Detuning correction from nn interaction
            neighbors = 0
            if i > 0:
                neighbors += 1
            if i < n_atoms - 1:
                neighbors += 1
            z_correction = -neighbors * interaction_v * dt / 4.0
            if abs(z_correction) > 1e-12:
                qc.rz(i, 2.0 * z_correction)

    return qc


def build_z_observables(n_qubits):
    """Build per-qubit Z observables: ['ZIII..', 'IZII..', ...]"""
    obs = []
    for i in range(n_qubits):
        pauli = ['I'] * n_qubits
        pauli[i] = 'Z'
        obs.append("".join(pauli))
    return obs


def compute_staggered_magnetization(z_expects, n_atoms):
    """
    Staggered magnetization: M(t) = (1/N) Σᵢ (-1)^i ⟨Zᵢ(t)⟩

    For the Néel state |01010…⟩:
      - Even sites have ⟨Z⟩ = +1 (state |0⟩)
      - Odd sites have ⟨Z⟩ = -1 (state |1⟩)
      - So M(0) = +1

    Scarring manifests as M(t) oscillating instead of decaying to 0.
    """
    stag_mag = 0.0
    for i, z_val in enumerate(z_expects):
        stag_mag += ((-1) ** i) * z_val
    return stag_mag / n_atoms


# ─────────────────────────────────────────────────────────────────────
# Exact diagonalization reference (small systems only)
# ─────────────────────────────────────────────────────────────────────

def compute_exact_ed(n_atoms, omega, interaction_v, times):
    """
    Exact time evolution of the Rydberg Hamiltonian via ED.

    Only feasible for n_atoms ≤ ~16 (Hilbert space = 2^N).

    Returns dict with 'times', 'stag_mag', 'fidelity'.
    """
    dim = 2 ** n_atoms
    if dim > 2**18:
        return None

    print(f"  Computing exact ED reference (N={n_atoms}, dim={dim})...")

    # Build Hamiltonian matrix
    H = np.zeros((dim, dim))

    for state in range(dim):
        bits = [(state >> i) & 1 for i in range(n_atoms)]

        # Diagonal: V Σ nᵢ nᵢ₊₁ where nᵢ = bit i
        for i in range(n_atoms - 1):
            if bits[i] == 1 and bits[i+1] == 1:
                H[state, state] += interaction_v

        # Off-diagonal: (Ω/2) Σ Xᵢ
        for i in range(n_atoms):
            flipped = state ^ (1 << i)
            H[state, flipped] += omega / 2.0

    # Diagonalize
    eigenvalues, eigenvectors = np.linalg.eigh(H)

    # Initial state: Néel |01010…⟩
    neel_idx = 0
    for i in range(n_atoms):
        if i % 2 == 1:
            neel_idx |= (1 << i)

    psi0 = np.zeros(dim)
    psi0[neel_idx] = 1.0

    # Decompose in eigenbasis
    coeffs = eigenvectors.T @ psi0  # coeffs[k] = ⟨E_k|ψ₀⟩

    # Z operators for staggered magnetization
    Z_diag = np.zeros((n_atoms, dim))
    for state in range(dim):
        for i in range(n_atoms):
            Z_diag[i, state] = 1.0 - 2.0 * ((state >> i) & 1)

    results = {'times': [], 'stag_mag': [], 'fidelity': []}

    for t in times:
        # |ψ(t)⟩ = Σ_k c_k e^{-i E_k t} |E_k⟩
        phases = np.exp(-1j * eigenvalues * t)
        psi_t = eigenvectors @ (coeffs * phases)

        # Fidelity
        fidelity = abs(np.dot(psi0.conj(), psi_t)) ** 2

        # Staggered magnetization
        probs = np.abs(psi_t) ** 2
        stag_mag = 0.0
        for i in range(n_atoms):
            z_expect = np.sum(Z_diag[i] * probs)
            stag_mag += ((-1) ** i) * z_expect
        stag_mag /= n_atoms

        results['times'].append(t)
        results['stag_mag'].append(stag_mag)
        results['fidelity'].append(fidelity)

    print(f"  ✓ Exact ED computed for {len(times)} time points")
    return results


# ─────────────────────────────────────────────────────────────────────
# Main experiment
# ─────────────────────────────────────────────────────────────────────

def run_scarring_experiment(n_atoms, omega, interaction_v, n_steps, dt,
                            chi, use_gpu):
    """
    Time-evolve the Néel state under PXP-like dynamics.

    Tracks staggered magnetization via estimate() [noise-free].
    """
    sim_type = (maestro.SimulatorType.Gpu if use_gpu
                else maestro.SimulatorType.QCSim)

    observables = build_z_observables(n_atoms)

    results = {
        'times': [], 'stag_mag': [], 'step_durations': [],
    }

    print(f"\n  {'Step':>6}  {'t':>6}  {'M(t)':>10}  {'Time':>8}")
    print(f"  {'─'*38}")

    # Step 0: initial Néel state — M(0) = +1
    results['times'].append(0.0)
    results['stag_mag'].append(1.0)
    results['step_durations'].append(0.0)
    print(f"  {0:>6}  {0.0:>6.2f}  {1.0:>10.6f}  {'─':>8}")

    for step in range(1, n_steps + 1):
        t0 = time.time()
        t_val = step * dt

        qc = build_pxp_circuit(n_atoms, omega, interaction_v, dt, step)

        res = qc.estimate(observables, maestro.SimulatorConfig(
            simulator_type=sim_type,
            simulation_type=maestro.SimulationType.MatrixProductState,
            max_bond_dimension=chi,
        ))

        z_expects = res['expectation_values']
        stag_mag = compute_staggered_magnetization(z_expects, n_atoms)

        wall = time.time() - t0

        results['times'].append(t_val)
        results['stag_mag'].append(stag_mag)
        results['step_durations'].append(wall)

        if step % max(1, n_steps // 10) == 0 or step <= 3:
            print(f"  {step:>6}  {t_val:>6.2f}  {stag_mag:>10.6f}  "
                  f"{wall:>7.2f}s")

    total = sum(results['step_durations'])
    avg = np.mean(results['step_durations'][1:])
    print(f"\n  ✓ Done: {total:.1f}s total, {avg:.3f}s/step avg")

    return results


# ─────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────

def plot_scarring(results, config, exact, path):
    """
    Two-panel plot:
      Left:  Staggered magnetization M(t) — MPS vs ED
      Right: Exact fidelity F(t) (ED only, small systems)
    """
    has_exact = exact is not None

    if has_exact:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(9, 6))
        axes = [axes]

    times = results['times']
    stag_mag = results['stag_mag']

    # ── Left / Only: Staggered magnetization ──
    ax1 = axes[0]
    ax1.plot(times, stag_mag, 'o-', color='#1565C0', linewidth=2.5,
             markersize=4,
             label=f'MPS (χ={config["chi"]})')

    if has_exact:
        ax1.plot(exact['times'], exact['stag_mag'], '-', color='#4CAF50',
                 linewidth=2, alpha=0.8, label='Exact ED')

    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.4,
                label='Thermal equilibrium (M=0)')
    ax1.fill_between(times, 0, stag_mag, alpha=0.08, color='#1565C0')

    ax1.set_xlabel('Time t (units of 1/Ω)', fontsize=13)
    ax1.set_ylabel('Staggered Magnetization M(t)', fontsize=13)
    ax1.set_title('Quantum Many-Body Scarring\n'
                  'Néel Order Parameter Revivals',
                  fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    # Info box
    info = (f"N = {config['n_atoms']} atoms\n"
            f"V/Ω = {config['V']/config['omega']:.0f}\n"
            f"χ = {config['chi']}, "
            f"{'GPU' if config['use_gpu'] else 'CPU'}")
    ax1.text(0.98, 0.02, info, transform=ax1.transAxes,
             fontsize=9, verticalalignment='bottom',
             horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='aliceblue',
                       edgecolor='gray', alpha=0.9))

    # ── Right: Exact fidelity (ED only) ──
    if has_exact:
        ax2 = axes[1]
        ax2.plot(exact['times'], exact['fidelity'], '-', color='#E91E63',
                 linewidth=2.5, label='Exact F(t) = |⟨Z₂|ψ(t)⟩|²')
        ax2.fill_between(exact['times'], 0, exact['fidelity'],
                         alpha=0.1, color='#E91E63')

        # Mark revival peaks
        fid_arr = np.array(exact['fidelity'])
        for i in range(2, len(fid_arr) - 1):
            if fid_arr[i] > fid_arr[i-1] and fid_arr[i] > fid_arr[i+1]:
                if fid_arr[i] > 0.05:
                    ax2.plot(exact['times'][i], fid_arr[i], 'v',
                             color='#880E4F', markersize=10)

        ax2.axhline(y=1.0 / (2**config['n_atoms']),
                    color='gray', linestyle=':', alpha=0.5,
                    label=f'Thermal ≈ 2⁻ᴺ')

        ax2.set_xlabel('Time t (units of 1/Ω)', fontsize=13)
        ax2.set_ylabel('Return Probability F(t)', fontsize=13)
        ax2.set_title('Fidelity Revivals (Exact ED)\n'
                      'F(t) = |⟨Z₂|ψ(t)⟩|²',
                      fontsize=14)
        ax2.legend(fontsize=10)
        ax2.grid(alpha=0.3)
        ax2.set_ylim(bottom=-0.02)

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
    small = '--small' in sys.argv

    if small:
        n_atoms = 12
        chi = 64
        n_steps = 60
        dt = 0.2
    elif large:
        n_atoms = 64
        chi = 64
        n_steps = 60
        dt = 0.2
    else:
        n_atoms = 32
        chi = 32
        n_steps = 60
        dt = 0.2

    # PXP parameters
    omega = 1.0                 # Rabi frequency (sets the energy scale)
    interaction_v = 20.0        # Strong blockade V ≫ Ω → PXP limit
    T_total = n_steps * dt

    print(f"\n{'═'*65}")
    print(f"  MAESTRO Demo: Quantum Many-Body Scarring (PXP)")
    print(f"  Néel State Revivals in a Rydberg Chain")
    print(f"{'═'*65}")
    print(f"\n  System:      {n_atoms} atoms, 1D chain")
    print(f"  Parameters:  Ω = {omega}, V = {interaction_v} "
          f"(V/Ω = {interaction_v/omega:.0f})")
    print(f"  Time:        T = {T_total:.1f}, {n_steps} steps, dt = {dt}")
    print(f"  MPS:         χ = {chi}")
    print(f"  Backend:     {'GPU' if use_gpu else 'CPU (QCSim)'}")
    print(f"\n  Physics: In a generic quantum system, the Néel state would")
    print(f"  thermalize — M(t) decaying monotonically to 0. But the PXP")
    print(f"  model has quantum scars: special eigenstates that cause M(t)")
    print(f"  to oscillate persistently.")

    start_time = time.time()

    results = run_scarring_experiment(
        n_atoms, omega, interaction_v, n_steps, dt, chi, use_gpu,
    )

    # Exact ED reference (small systems only)
    exact = None
    if n_atoms <= 16:
        t_arr = np.array(results['times'])
        exact = compute_exact_ed(n_atoms, omega, interaction_v, t_arr)
    else:
        print(f"\n  ℹ Exact ED skipped (N={n_atoms} > 16)")
        print(f"  Run with --small for N=12 + exact ED validation")

    config = {
        'n_atoms': n_atoms, 'omega': omega, 'V': interaction_v,
        'chi': chi, 'use_gpu': use_gpu,
    }

    plot_path = plot_scarring(
        results, config, exact,
        os.path.join(SCRIPT_DIR, 'quantum_scarring.png'),
    )
    print(f"\n  📊 Saved: {plot_path}")

    # ── Summary ──
    total = time.time() - start_time
    stag_arr = np.array(results['stag_mag'])

    print(f"\n{'═'*65}")
    print(f"  SUMMARY")
    print(f"{'═'*65}")
    print(f"  Total runtime: {total:.1f}s")
    print(f"  M(t) range: {min(stag_arr):.4f} → {max(stag_arr):.4f}")

    if exact:
        mae_m = np.mean(np.abs(np.array(results['stag_mag'])
                                - np.array(exact['stag_mag'])))
        print(f"  MAE vs exact (M): {mae_m:.4f}")

        fid_arr = np.array(exact['fidelity'])
        peaks = []
        for i in range(2, len(fid_arr) - 1):
            if fid_arr[i] > fid_arr[i-1] and fid_arr[i] > fid_arr[i+1]:
                if fid_arr[i] > 0.05:
                    peaks.append((exact['times'][i], fid_arr[i]))

        if peaks:
            print(f"\n  Fidelity revival peaks (exact ED):")
            for t_peak, f_peak in peaks[:5]:
                print(f"    t = {t_peak:.2f}  →  F = {f_peak:.4f}")
            if len(peaks) >= 2:
                periods = [peaks[i+1][0] - peaks[i][0]
                           for i in range(len(peaks)-1)]
                avg_period = np.mean(periods)
                print(f"  Revival period: T_rev ≈ {avg_period:.2f}")

    # Detect M(t) oscillation extrema
    mag_extrema = []
    for i in range(2, len(stag_arr) - 1):
        if stag_arr[i] < stag_arr[i-1] and stag_arr[i] < stag_arr[i+1]:
            if stag_arr[i] < -0.1:
                mag_extrema.append((results['times'][i], stag_arr[i]))
        elif stag_arr[i] > stag_arr[i-1] and stag_arr[i] > stag_arr[i+1]:
            if stag_arr[i] > 0.1 and i > 5:
                mag_extrema.append((results['times'][i], stag_arr[i]))

    if mag_extrema:
        print(f"\n  M(t) oscillation extrema:")
        for t_peak, m_peak in mag_extrema[:6]:
            print(f"    t = {t_peak:.2f}  →  M = {m_peak:.4f}")

    print(f"\n  The persistent oscillations in M(t) are a signature of")
    print(f"  quantum many-body scars — non-thermal eigenstates that")
    print(f"  violate the eigenstate thermalization hypothesis (ETH).\n")
