#!/usr/bin/env python3
"""
Benchmark B: Real-Time Quench Dynamics
========================================

Prepares an initial product state, then suddenly quenches to a new Hamiltonian
and tracks Trotterized time evolution. Entanglement grows linearly with time,
so bond dimension requirements increase — this is where GPU advantage kicks in.

Models:
    1. Frustrated Heisenberg: Néel state |1010...⟩ → J₁-J₂ quench
    2. Anderson Impurity: Half-filled impurity → hybridization quench

Sweep parameters:
    Trotter steps: 5, 10, 20, 40, 60, 80, 100
    χ (bond dim):  16, 32, 64, 128, 256, 512, 1024
    N (qubits):    8, 16, 32, 48, 64, 80, 100

Default mode runs a targeted subset. Use --full for exhaustive.

Usage:
    python benchmark_quench_dynamics.py                      # CPU only
    python benchmark_quench_dynamics.py --gpu                 # Include GPU
    python benchmark_quench_dynamics.py --model heisenberg    # One model
    python benchmark_quench_dynamics.py --model anderson
    python benchmark_quench_dynamics.py --full --gpu          # Full sweep
"""

import sys
import os
import time

import numpy as np
import maestro

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from frustrated_heisenberg import FrustratedHeisenberg
from anderson_impurity import AndersonImpurity
from plotting import (
    plot_crossover_heatmap, plot_scaling_curves, plot_per_step_timing,
    plot_timing_bars, save_results,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Physics parameters
DT = 0.1  # Trotter time step

# Heisenberg model parameters
J1, J2 = 1.0, 0.5  # J₂/J₁ = 0.5 (critical frustration)

# Anderson model parameters
U_HUBBARD = 4.0     # On-site Coulomb repulsion
V_HYB = 1.0         # Hybridization
EPS_D = -2.0        # Impurity level (half-filling: ε_d = -U/2)

# Full sweep parameters
FULL_N_QUBITS = [8, 16, 32, 48, 64, 80, 100]
FULL_CHI_VALS = [16, 32, 64, 128, 256, 512, 1024]
FULL_STEP_VALS = [5, 10, 20, 40, 60, 80, 100]

# Default (targeted) sweep
DEFAULT_N_QUBITS = [8, 16, 32, 64]
DEFAULT_CHI_VALS = [16, 64, 128, 256]
DEFAULT_STEP_VALS = [5, 10, 20, 40]

# Parse CLI args
GPU_ENABLED = '--gpu' in sys.argv
FULL_SWEEP = '--full' in sys.argv

MODEL_FILTER = None
for i, arg in enumerate(sys.argv):
    if arg == '--model' and i + 1 < len(sys.argv):
        MODEL_FILTER = sys.argv[i + 1].lower()


# =============================================================================
# BENCHMARK CORE
# =============================================================================

def run_quench_benchmark(model_name, n_qubits, chi, n_steps, use_gpu):
    """
    Run a single quench dynamics benchmark point.

    Builds the full circuit for n_steps Trotter steps, simulates, and
    measures energy observables.

    Returns a result dict with timing and energy data.
    """
    device = "GPU" if use_gpu else "CPU"

    # Build circuit
    if model_name == 'heisenberg':
        m = FrustratedHeisenberg(n_qubits, j1=J1, j2=J2)
        qc = m.build_neel_quench_circuit(dt=DT, n_steps=n_steps)
        observables, coefficients = m.build_energy_observables()
    else:
        n_bath = max(1, (n_qubits - 2) // 2)
        m = AndersonImpurity(n_bath, u=U_HUBBARD, v=V_HYB, eps_d=EPS_D)
        n_qubits = m.n_qubits
        qc = m.build_quench_circuit(dt=DT, n_steps=n_steps)
        observables, coefficients = m.build_energy_observables()

    sim_type = (maestro.SimulatorType.Gpu if use_gpu
                else maestro.SimulatorType.QCSim)

    config = maestro.SimulatorConfig(
        simulator_type=sim_type,
        simulation_type=maestro.SimulationType.MatrixProductState,
        max_bond_dimension=chi,
    )

    # Run and time
    t0 = time.time()
    result = qc.estimate(observables, config)
    elapsed = time.time() - t0

    # Compute energy
    exp_vals = result['expectation_values']
    energy = sum(c * v for c, v in zip(coefficients, exp_vals))

    return {
        'model': model_name,
        'n_qubits': n_qubits,
        'chi': chi,
        'n_steps': n_steps,
        't_total': n_steps * DT,
        'device': device,
        'time': elapsed,
        'energy': float(energy),
    }


def run_per_step_timing(model_name, n_qubits, chi, max_steps, use_gpu):
    """
    Run incremental Trotter steps to measure per-step wall-clock time.

    This shows how GPU advantage grows as entanglement accumulates with
    each Trotter step.

    Returns list of per-step durations.
    """
    step_times = []

    for step in range(1, max_steps + 1):
        try:
            t0 = time.time()

            if model_name == 'heisenberg':
                m = FrustratedHeisenberg(n_qubits, j1=J1, j2=J2)
                qc = m.build_neel_quench_circuit(dt=DT, n_steps=step)
                observables, coefficients = m.build_energy_observables()
            else:
                n_bath = max(1, (n_qubits - 2) // 2)
                m = AndersonImpurity(n_bath, u=U_HUBBARD, v=V_HYB, eps_d=EPS_D)
                qc = m.build_quench_circuit(dt=DT, n_steps=step)
                observables, coefficients = m.build_energy_observables()

            sim_type = (maestro.SimulatorType.Gpu if use_gpu
                        else maestro.SimulatorType.QCSim)

            config = maestro.SimulatorConfig(
                simulator_type=sim_type,
                simulation_type=maestro.SimulationType.MatrixProductState,
                max_bond_dimension=chi,
            )

            result = qc.estimate(observables, config)
            elapsed = time.time() - t0
            step_times.append(elapsed)

            if step % max(1, max_steps // 8) == 0 or step <= 2:
                print(f"      step {step:3d}  ({elapsed:.3f}s)")

        except Exception as e:
            print(f"      step {step:3d}  FAILED: {e}")
            break

    return step_times


def run_sweep(model_name, n_list, chi_list, step_list, use_gpu):
    """
    Run the full parameter sweep for quench dynamics.

    Returns list of result dicts.
    """
    results = []
    total = len(n_list) * len(chi_list) * len(step_list)
    count = 0

    for n in n_list:
        for chi in chi_list:
            for n_steps in step_list:
                count += 1
                label = (f"  [{count}/{total}] {model_name}: "
                         f"N={n}, χ={chi}, steps={n_steps}")

                # CPU run
                try:
                    print(f"{label} — CPU ...", end='', flush=True)
                    cpu_res = run_quench_benchmark(
                        model_name, n, chi, n_steps, use_gpu=False
                    )
                    print(f" {cpu_res['time']:.2f}s  E={cpu_res['energy']:.4f}")
                except Exception as e:
                    print(f" FAILED: {e}")
                    continue

                # GPU run (optional)
                gpu_res = None
                if use_gpu:
                    try:
                        print(f"{' ' * len(label)} — GPU ...", end='', flush=True)
                        gpu_res = run_quench_benchmark(
                            model_name, n, chi, n_steps, use_gpu=True
                        )
                        speedup = cpu_res['time'] / max(gpu_res['time'], 1e-9)
                        print(f" {gpu_res['time']:.2f}s  "
                              f"({speedup:.1f}× speedup)")
                    except Exception as e:
                        print(f" FAILED: {e}")

                entry = {
                    'model': model_name,
                    'n_qubits': cpu_res['n_qubits'],
                    'chi': chi,
                    'n_steps': n_steps,
                    't_total': n_steps * DT,
                    'cpu_time': cpu_res['time'],
                    'energy': cpu_res['energy'],
                    'gpu_time': gpu_res['time'] if gpu_res else None,
                    'gpu_energy': gpu_res['energy'] if gpu_res else None,
                    'label': (f"N={cpu_res['n_qubits']}, χ={chi}, "
                              f"steps={n_steps}"),
                }
                results.append(entry)

    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    n_list = FULL_N_QUBITS if FULL_SWEEP else DEFAULT_N_QUBITS
    chi_list = FULL_CHI_VALS if FULL_SWEEP else DEFAULT_CHI_VALS
    step_list = FULL_STEP_VALS if FULL_SWEEP else DEFAULT_STEP_VALS

    gpu_available = maestro.is_gpu_available()
    use_gpu = GPU_ENABLED and gpu_available

    print(f"\n{'═' * 70}")
    print(f"  BENCHMARK B: REAL-TIME QUENCH DYNAMICS")
    print(f"{'═' * 70}")
    print(f"  Sweep:   N = {n_list}")
    print(f"           χ = {chi_list}")
    print(f"           steps = {step_list} (dt = {DT})")
    print(f"  GPU:     {'ENABLED' if use_gpu else 'DISABLED'}"
          f"{'  (not available)' if GPU_ENABLED and not gpu_available else ''}")
    print(f"  Mode:    {'FULL' if FULL_SWEEP else 'TARGETED'}")
    print(f"{'═' * 70}\n")

    all_results = []

    # ── Frustrated Heisenberg ──
    if MODEL_FILTER is None or MODEL_FILTER == 'heisenberg':
        print(f"\n{'─' * 70}")
        print(f"  MODEL 1: Frustrated J₁-J₂ Heisenberg (J₂/J₁ = {J2/J1:.1f})")
        print(f"  Néel state |1010...⟩ → quench dynamics")
        print(f"{'─' * 70}")

        heis_results = run_sweep('heisenberg', n_list, chi_list,
                                 step_list, use_gpu)
        all_results.extend(heis_results)

        # Per-step timing for a representative point
        rep_n = n_list[-1]
        rep_chi = chi_list[-1]
        rep_steps = min(20, step_list[-1])

        print(f"\n    Per-step timing: N={rep_n}, χ={rep_chi}")
        print(f"    CPU:")
        cpu_step_times = run_per_step_timing(
            'heisenberg', rep_n, rep_chi, rep_steps, use_gpu=False
        )
        gpu_step_times = None
        if use_gpu:
            print(f"    GPU:")
            gpu_step_times = run_per_step_timing(
                'heisenberg', rep_n, rep_chi, rep_steps, use_gpu=True
            )

        if cpu_step_times:
            plot_per_step_timing(
                cpu_step_times, gpu_step_times,
                f'Per-Step Timing — Heisenberg N={rep_n}, χ={rep_chi}',
                os.path.join(SCRIPT_DIR, 'quench_heisenberg_steps.png'),
            )

        # Heatmap for fixed step count
        if heis_results:
            mid_steps = step_list[len(step_list) // 2]
            heatmap_data = [r for r in heis_results if r['n_steps'] == mid_steps]
            plot_crossover_heatmap(
                heatmap_data,
                f'GPU Speedup — Heisenberg Quench (steps={mid_steps})',
                os.path.join(SCRIPT_DIR, 'quench_heisenberg_crossover.png'),
            )

    # ── Anderson Impurity ──
    if MODEL_FILTER is None or MODEL_FILTER == 'anderson':
        print(f"\n{'─' * 70}")
        print(f"  MODEL 2: Anderson Impurity (U={U_HUBBARD}, V={V_HYB})")
        print(f"  Half-filled impurity → hybridization quench")
        print(f"{'─' * 70}")

        anderson_results = run_sweep('anderson', n_list, chi_list,
                                     step_list, use_gpu)
        all_results.extend(anderson_results)

        # Per-step timing
        rep_n = n_list[-1]
        rep_chi = chi_list[-1]
        rep_steps = min(20, step_list[-1])

        print(f"\n    Per-step timing: N={rep_n}, χ={rep_chi}")
        print(f"    CPU:")
        cpu_step_times = run_per_step_timing(
            'anderson', rep_n, rep_chi, rep_steps, use_gpu=False
        )
        gpu_step_times = None
        if use_gpu:
            print(f"    GPU:")
            gpu_step_times = run_per_step_timing(
                'anderson', rep_n, rep_chi, rep_steps, use_gpu=True
            )

        if cpu_step_times:
            plot_per_step_timing(
                cpu_step_times, gpu_step_times,
                f'Per-Step Timing — Anderson N={rep_n}, χ={rep_chi}',
                os.path.join(SCRIPT_DIR, 'quench_anderson_steps.png'),
            )

        if anderson_results:
            mid_steps = step_list[len(step_list) // 2]
            heatmap_data = [r for r in anderson_results
                            if r['n_steps'] == mid_steps]
            plot_crossover_heatmap(
                heatmap_data,
                f'GPU Speedup — Anderson Quench (steps={mid_steps})',
                os.path.join(SCRIPT_DIR, 'quench_anderson_crossover.png'),
            )

    # ── Combined scaling curves ──
    if all_results:
        mid_steps = step_list[len(step_list) // 2]
        scaling_data = [r for r in all_results if r['n_steps'] == mid_steps]

        plot_scaling_curves(
            scaling_data, x_key='chi', group_key='n_qubits',
            title=f'Quench Dynamics: χ vs Wall-Clock (steps={mid_steps})',
            save_path=os.path.join(SCRIPT_DIR, 'quench_scaling_chi.png'),
            x_label='Bond Dimension χ',
        )
        plot_scaling_curves(
            scaling_data, x_key='n_qubits', group_key='chi',
            title=f'Quench Dynamics: N vs Wall-Clock (steps={mid_steps})',
            save_path=os.path.join(SCRIPT_DIR, 'quench_scaling_n.png'),
            x_label='System Size N (qubits)',
        )

        # Timing bars for large systems
        bar_data = [r for r in all_results
                    if r['n_steps'] == mid_steps
                    and r['n_qubits'] in (n_list[-1], n_list[-2])]
        if bar_data:
            plot_timing_bars(
                bar_data,
                'Quench Dynamics: CPU vs GPU Timing',
                os.path.join(SCRIPT_DIR, 'quench_timing_bars.png'),
            )

        save_results(all_results, 'quench_results.json')

    # ── Summary ──
    print(f"\n{'═' * 70}")
    print(f"  QUENCH DYNAMICS BENCHMARK SUMMARY")
    print(f"{'═' * 70}")
    print(f"\n  {'Model':<14} {'N':>4} {'χ':>6} {'steps':>6} {'T':>6} "
          f"{'CPU(s)':>8} {'GPU(s)':>8} {'Speedup':>8} {'Energy':>10}")
    print(f"  {'─' * 76}")

    for r in all_results:
        gpu_str = f"{r['gpu_time']:.2f}" if r.get('gpu_time') else '—'
        sp_str = (f"{r['cpu_time'] / r['gpu_time']:.1f}×"
                  if r.get('gpu_time') else '—')
        print(f"  {r['model']:<14} {r['n_qubits']:>4} {r['chi']:>6} "
              f"{r['n_steps']:>6} {r['t_total']:>6.1f} "
              f"{r['cpu_time']:>8.2f} {gpu_str:>8} "
              f"{sp_str:>8} {r['energy']:>10.4f}")

    # Expected crossover analysis
    if use_gpu:
        print(f"\n  GPU CROSSOVER ANALYSIS:")
        print(f"  {'─' * 50}")
        for r in all_results:
            if r.get('gpu_time') and r['gpu_time'] > 0:
                sp = r['cpu_time'] / r['gpu_time']
                marker = '✓ GPU wins' if sp > 1.0 else '✗ CPU wins'
                if 0.8 < sp < 1.2:
                    marker = '≈ CROSSOVER'
                print(f"    N={r['n_qubits']:>3}, χ={r['chi']:>4}, "
                      f"steps={r['n_steps']:>3}: "
                      f"{sp:.2f}× — {marker}")

    print()
