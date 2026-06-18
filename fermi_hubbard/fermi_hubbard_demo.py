#!/usr/bin/env python3
"""
Adaptive Fermi-Hubbard Benchmark — Maestro Demo
=================================================

Demonstrates a 3-tier adaptive simulation pipeline for the 1D Fermi-Hubbard
model using Maestro.

Key Insight:
    For local quench dynamics, information propagates at a finite speed
    (Lieb-Robinson velocity), so most of the system remains frozen. We exploit
    this to solve a nominally huge system by simulating only the active
    light-cone region.

Pipeline:
    Tier 1 — "Scout" (Pauli Propagator, CPU):
        Runs a fast Clifford-only simulation on the FULL system to identify
        which sites have non-trivial dynamics. Cost: seconds.

    Tier 2 — "Sniper" (MPS, CPU, χ=64):
        Runs an MPS simulation ONLY on the active subregion detected by the
        scout. Quick physics preview at moderate accuracy.

    Tier 3 — "Precision" (MPS, GPU/CPU, χ=256):
        Re-runs the active subregion with higher bond dimension for converged,
        publication-quality results.

Physics:
    - 1D Fermi-Hubbard chain, N sites (2N qubits: spin-up + spin-down)
    - Jordan-Wigner mapping (nearest-neighbor, no JW strings)
    - Domain-wall initial state: left half filled, right half empty
    - Time evolution via Trotterization

Output:
    adaptive_hubbard_density.png  — Density profile comparison
    adaptive_hubbard_scaling.png  — Wall-clock time vs system size

Usage:
    python fermi_hubbard_demo.py             # CPU-only
    python fermi_hubbard_demo.py --gpu       # Enable GPU tier
    python fermi_hubbard_demo.py --scaling   # Include scaling sweep
"""

import sys
import os
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import maestro
from model import FermiHubbardModel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# CONFIGURATION
# =============================================================================

T_HOP = 1.0          # Hopping energy t
U_INT = 1.0          # On-site interaction U (U/t=1, metallic regime)
T_EVOLUTION = 5.0    # Total evolution time
N_STEPS = 50         # Trotter steps (dt = 0.1)

CHI_CPU = 64         # Tier 2: CPU moderate bond dimension
CHI_GPU = 256        # Tier 3: High bond dimension

SAFETY_MARGIN = 5    # Sites added around PP-detected active region

# System sizes for scaling sweep
SYSTEM_SIZES = [50, 100, 200]

GPU_ENABLED = '--gpu' in sys.argv
RUN_SCALING = '--scaling' in sys.argv


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def build_z_observables(n_qubits):
    """Build per-qubit Z observables: ['ZIII..', 'IZII..', ...]"""
    obs = []
    for i in range(n_qubits):
        pauli = ['I'] * n_qubits
        pauli[i] = 'Z'
        obs.append("".join(pauli))
    return obs


def z_to_density(exp_vals, n_active_sites):
    """
    Convert per-qubit ⟨Z⟩ values to particle density per site.

    n_σ = (1 − ⟨Z_σ⟩) / 2
    n_total = n_↑ + n_↓ (range [0, 2])
    """
    densities = []
    for i in range(n_active_sites):
        n_up = (1.0 - exp_vals[i]) / 2.0
        n_down = (1.0 - exp_vals[i + n_active_sites]) / 2.0
        densities.append(n_up + n_down)
    return densities


# =============================================================================
# TIER 1: SCOUT (Pauli Propagator)
# =============================================================================

def run_scout(n_sites, total_qubits, init_wall_idx):
    """
    Tier 1: Fast light-cone detection using Pauli Propagator.

    Uses a Clifford-only proxy circuit that preserves the connectivity
    structure of the Fermi-Hubbard Trotter circuit. Measures per-site ⟨Z_i⟩
    and identifies sites where dynamics have occurred.

    Returns (active_start, active_end, scout_time).
    """
    print(f"\n  Tier 1: Scout — Pauli Propagator on {total_qubits} qubits")

    scout_start_time = time.time()

    # Calibrate steps to the Lieb-Robinson light cone
    light_cone_radius = 2.0 * T_HOP * T_EVOLUTION
    scout_steps = max(2, int(light_cone_radius / 2) + 1)

    model = FermiHubbardModel(n_sites, t=T_HOP, u=U_INT)
    scout_circuit = model.build_clifford_scout_circuit(
        steps=scout_steps, init_wall_idx=init_wall_idx
    )

    obs_list = build_z_observables(total_qubits)

    config = maestro.SimulatorConfig(
        simulator_type=maestro.SimulatorType.QCSim,
        simulation_type=maestro.SimulationType.PauliPropagator,
    )
    result = scout_circuit.estimate(obs_list, config)
    z_vals = result['expectation_values']
    scout_elapsed = time.time() - scout_start_time

    # Detect active sites
    THRESHOLD = 0.001
    active_start = init_wall_idx
    active_end = init_wall_idx

    for i in range(n_sites):
        initial_z = -1.0 if i < init_wall_idx else 1.0
        if (abs(z_vals[i] - initial_z) > THRESHOLD or
                abs(z_vals[n_sites + i] - initial_z) > THRESHOLD):
            active_start = min(active_start, i)
            active_end = max(active_end, i + 1)

    active_start = max(0, active_start - SAFETY_MARGIN)
    active_end = min(n_sites, active_end + SAFETY_MARGIN)
    n_active = active_end - active_start

    print(f"    Active region: [{active_start}, {active_end}) "
          f"→ {n_active} sites ({2 * n_active} qubits)")
    print(f"    Completed in {scout_elapsed:.2f}s")

    return active_start, active_end, scout_elapsed


# =============================================================================
# TIER 2: SNIPER (MPS CPU)
# =============================================================================

def run_sniper_cpu(start, end, init_wall_idx, chi=CHI_CPU):
    """
    Tier 2: MPS simulation on CPU with moderate bond dimension.

    Returns (expectation_values, elapsed_time).
    """
    n_active = end - start
    n_qubits = 2 * n_active
    print(f"\n  Tier 2: Sniper — MPS CPU on {n_qubits} qubits (χ={chi})")

    model = FermiHubbardModel(n_active, t=T_HOP, u=U_INT)
    circuit = model.build_circuit(
        steps=N_STEPS, dt=T_EVOLUTION / N_STEPS,
        init_wall_idx=init_wall_idx, active_sites_range=(start, end)
    )
    obs_list = build_z_observables(n_qubits)

    config = maestro.SimulatorConfig(
        simulator_type=maestro.SimulatorType.QCSim,
        simulation_type=maestro.SimulationType.MatrixProductState,
        max_bond_dimension=chi,
    )

    start_time = time.time()
    result = circuit.estimate(obs_list, config)
    elapsed = time.time() - start_time

    print(f"    Completed in {elapsed:.2f}s")
    return result['expectation_values'], elapsed


# =============================================================================
# TIER 3: PRECISION (MPS GPU/CPU High χ)
# =============================================================================

def run_precision(start, end, init_wall_idx, chi=CHI_GPU, use_gpu=False):
    """
    Tier 3: MPS simulation with high bond dimension.

    Returns (expectation_values, elapsed_time).
    """
    n_active = end - start
    n_qubits = 2 * n_active
    device = "GPU" if use_gpu else "CPU"
    print(f"\n  Tier 3: Precision — MPS {device} on {n_qubits} qubits (χ={chi})")

    model = FermiHubbardModel(n_active, t=T_HOP, u=U_INT)
    circuit = model.build_circuit(
        steps=N_STEPS, dt=T_EVOLUTION / N_STEPS,
        init_wall_idx=init_wall_idx, active_sites_range=(start, end)
    )
    obs_list = build_z_observables(n_qubits)

    sim_type = maestro.SimulatorType.Gpu if use_gpu else maestro.SimulatorType.QCSim

    config = maestro.SimulatorConfig(
        simulator_type=sim_type,
        simulation_type=maestro.SimulationType.MatrixProductState,
        max_bond_dimension=chi,
    )

    start_time = time.time()
    result = circuit.estimate(obs_list, config)
    elapsed = time.time() - start_time

    print(f"    Completed in {elapsed:.2f}s")
    return result['expectation_values'], elapsed


# =============================================================================
# FULL PIPELINE
# =============================================================================

def run_pipeline(total_qubits, run_gpu=False):
    """
    Run the full 3-tier adaptive simulation pipeline.

    Returns a dict with all timing and physics data.
    """
    n_sites = total_qubits // 2
    init_wall_idx = n_sites // 2

    print(f"\n{'=' * 65}")
    print(f"  {total_qubits}-QUBIT FERMI-HUBBARD BENCHMARK")
    print(f"  {n_sites} sites, domain wall at site {init_wall_idx}")
    print(f"  t={T_HOP}, U={U_INT} (U/t={U_INT / T_HOP:.1f}), "
          f"T={T_EVOLUTION}, {N_STEPS} steps")
    print(f"{'=' * 65}")

    # Tier 1: Scout
    start, end, scout_time = run_scout(n_sites, total_qubits, init_wall_idx)
    n_active = end - start

    # Tier 2: MPS CPU
    cpu_vals, cpu_time = run_sniper_cpu(start, end, init_wall_idx)
    cpu_density = z_to_density(cpu_vals, n_active)

    # Tier 3: Precision (optional)
    gpu_density = None
    gpu_time = None
    try:
        gpu_vals, gpu_time = run_precision(
            start, end, init_wall_idx, use_gpu=run_gpu
        )
        gpu_density = z_to_density(gpu_vals, n_active)
    except Exception as e:
        print(f"    Precision tier failed: {e}")

    return {
        'total_qubits': total_qubits,
        'n_sites': n_sites,
        'init_wall_idx': init_wall_idx,
        'active_start': start,
        'active_end': end,
        'active_qubits': 2 * n_active,
        'scout_time': scout_time,
        'cpu_time': cpu_time,
        'cpu_density': cpu_density,
        'gpu_time': gpu_time,
        'gpu_density': gpu_density,
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_density_profile(data, use_gpu=False):
    """Plot particle density across the lattice."""
    fig, ax = plt.subplots(figsize=(14, 5))

    n_sites = data['n_sites']
    start = data['active_start']
    end = data['active_end']
    init_wall = data['init_wall_idx']
    global_x = list(range(start, end))

    # CPU result
    ax.bar(global_x, data['cpu_density'], color='#2F847C', width=1.0, alpha=0.7,
           label=f'MPS CPU (χ={CHI_CPU}, {data["cpu_time"]:.1f}s)')

    # Precision result (overlay if available)
    device = "GPU" if use_gpu else "CPU"
    if data['gpu_density'] is not None:
        ax.step(
            [x + 0.5 for x in global_x], data['gpu_density'],
            color='#E74C3C', linewidth=2, where='mid',
            label=f'MPS {device} (χ={CHI_GPU}, {data["gpu_time"]:.1f}s)'
        )

    # Frozen regions
    if start > 0:
        ax.bar(range(0, start), [2.0] * start, color='#B0B0B0', alpha=0.2,
               width=1.0, label='Frozen (filled)')
    if end < n_sites:
        ax.bar(range(end, n_sites), [0.0] * (n_sites - end), color='#D0D0D0',
               alpha=0.2, width=1.0, label='Frozen (empty)')

    # Domain wall
    ax.axvline(x=init_wall - 0.5, color='red', linestyle='--', linewidth=2,
               label='Initial domain wall')

    ax.set_xlabel('Lattice Site Index', fontsize=12)
    ax.set_ylabel('Total Density ⟨n↑⟩ + ⟨n↓⟩', fontsize=12)

    reduction = (data['total_qubits'] // data['active_qubits']
                 if data['active_qubits'] > 0 else 0)
    ax.set_title(
        f'{data["total_qubits"]}-Qubit Fermi-Hubbard: '
        f'Scout ({data["scout_time"]:.1f}s) → '
        f'CPU χ={CHI_CPU} ({data["cpu_time"]:.1f}s)\n'
        f'Active subspace: {data["active_qubits"]} qubits out of '
        f'{data["total_qubits"]} ({reduction}× reduction) | '
        f'U/t={U_INT / T_HOP:.1f}, T={T_EVOLUTION}',
        fontsize=10
    )
    ax.set_ylim(-0.1, 2.3)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()

    save_path = os.path.join(SCRIPT_DIR, 'adaptive_hubbard_density.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  📊 Saved: {save_path}")


def plot_scaling_sweep(results):
    """Plot wall-clock time vs total system size."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    sizes = [r['total_qubits'] for r in results]
    scout_times = [r['scout_time'] for r in results]
    cpu_times = [r['cpu_time'] for r in results]
    total_times = [s + c for s, c in zip(scout_times, cpu_times)]
    active_qubits = [r['active_qubits'] for r in results]

    # Left: Time vs System Size
    ax1.plot(sizes, scout_times, 'o-', color='#3498DB', linewidth=2,
             markersize=8, label='Tier 1: Scout (PP)')
    ax1.plot(sizes, cpu_times, 's-', color='#E74C3C', linewidth=2,
             markersize=8, label=f'Tier 2: MPS CPU (χ={CHI_CPU})')
    ax1.plot(sizes, total_times, 'D-', color='#2C3E50', linewidth=2,
             markersize=8, label='Total')

    ax1.set_xlabel('Total System Size (qubits)', fontsize=12)
    ax1.set_ylabel('Wall-Clock Time (s)', fontsize=12)
    ax1.set_title('Scaling: Time vs System Size', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    if len(cpu_times) >= 2:
        ax1.annotate(
            'MPS time flat →\nLight cone fixed!',
            xy=(sizes[-1], cpu_times[-1]),
            xytext=(sizes[-1] * 0.55, max(cpu_times) * 1.3),
            fontsize=10, color='#E74C3C',
            arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=1.5),
        )

    # Right: Active Qubits vs System Size
    ax2.bar(range(len(sizes)), active_qubits, color='#2F847C', alpha=0.8)
    ax2.set_xticks(range(len(sizes)))
    ax2.set_xticklabels([str(s) for s in sizes])
    ax2.set_xlabel('Total System Size (qubits)', fontsize=12)
    ax2.set_ylabel('Active Subspace (qubits)', fontsize=12)
    ax2.set_title('Active Qubits Stay Constant', fontsize=13)
    ax2.grid(axis='y', alpha=0.3)

    for i, (total, active) in enumerate(zip(sizes, active_qubits)):
        if active > 0:
            ratio = total / active
            ax2.text(i, active + 1, f'{ratio:.0f}× reduction',
                     ha='center', fontsize=9, fontweight='bold', color='#2C3E50')

    fig.tight_layout()
    save_path = os.path.join(SCRIPT_DIR, 'adaptive_hubbard_scaling.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  📊 Saved: {save_path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  MAESTRO Demo: Adaptive Fermi-Hubbard Simulation")
    print(f"  GPU: {'ENABLED' if GPU_ENABLED else 'DISABLED (use --gpu)'}")
    print("=" * 65)

    # ── 1. Density Profile: Full pipeline on main system ──
    main_size = max(SYSTEM_SIZES)
    density_data = run_pipeline(main_size, run_gpu=GPU_ENABLED)
    plot_density_profile(density_data, use_gpu=GPU_ENABLED)

    # ── 2. Scaling Sweep (optional) ──
    if RUN_SCALING:
        print(f"\n\n{'#' * 65}")
        print(f"  SCALING SWEEP: {SYSTEM_SIZES}")
        print(f"{'#' * 65}")

        sweep_results = []
        for total_qubits in SYSTEM_SIZES:
            data = run_pipeline(total_qubits, run_gpu=False)
            sweep_results.append(data)

        plot_scaling_sweep(sweep_results)

    # ── 3. Summary ──
    device = 'GPU' if GPU_ENABLED else 'CPU'
    print(f"\n\n{'=' * 75}")
    print(f"  BENCHMARK SUMMARY")
    print(f"  Physics: 1D Fermi-Hubbard, t={T_HOP}, U={U_INT} "
          f"(U/t={U_INT / T_HOP:.1f}), T={T_EVOLUTION}")
    print(f"  Method:  Trotter ({N_STEPS} steps, dt={T_EVOLUTION / N_STEPS:.3f})")
    print(f"  Tiers:   PP (scout) → MPS CPU (χ={CHI_CPU}) → "
          f"MPS {device} (χ={CHI_GPU})")
    print(f"{'=' * 75}")

    if density_data['gpu_time'] is not None:
        print(f"\n  Precision Tier ({main_size}Q, χ={CHI_GPU}): "
              f"{density_data['gpu_time']:.2f}s")
    print()
