#!/usr/bin/env python3
"""
Benchmark A: Ground-State Preparation (VQE-style)
===================================================

Sweeps (system size N, bond dimension χ, circuit depth) on both the frustrated
J₁-J₂ Heisenberg chain and the Anderson impurity model. Measures energy
expectation values and wall-clock time for CPU and (optionally) GPU MPS.

The goal: find the (N, χ, depth) combinations where GPU MPS outperforms CPU.

Sweep parameters:
    N (qubits):     8, 16, 24, 32, 48, 64, 80, 100
    χ (bond dim):   16, 32, 64, 128, 256, 512, 1024
    depth (layers): 2, 4, 8, 12, 16

Default mode runs a targeted subset covering the crossover region.
Use --full for the exhaustive sweep.

Usage:
    python benchmark_ground_state.py                      # Both models, CPU
    python benchmark_ground_state.py --gpu                 # Include GPU
    python benchmark_ground_state.py --model heisenberg    # One model only
    python benchmark_ground_state.py --model anderson
    python benchmark_ground_state.py --full --gpu          # Full sweep + GPU
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
from anderson_plotting import (
    plot_crossover_heatmap, plot_scaling_curves,
    plot_energy_convergence, plot_timing_bars, save_results,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Full sweep parameters
FULL_N_QUBITS = [8, 16, 24, 32, 48, 64, 80, 100]
FULL_CHI_VALS = [16, 32, 64, 128, 256, 512, 1024]
FULL_DEPTH_VALS = [2, 4, 8, 12, 16]

# Default (targeted) sweep — covers the crossover region
DEFAULT_N_QUBITS = [8, 16, 32, 64]
DEFAULT_CHI_VALS = [16, 64, 128, 256]
DEFAULT_DEPTH_VALS = [2, 4, 8]

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

def run_single_benchmark(model, model_name, n_qubits, chi, depth, use_gpu):
    """
    Run a single (N, χ, depth) benchmark point.

    Returns a result dict with timing and energy data.
    """
    device = "GPU" if use_gpu else "CPU"

    # Build circuit
    if model_name == 'heisenberg':
        m = FrustratedHeisenberg(n_qubits)
        qc = m.build_vqe_ansatz(depth)
        observables, coefficients = m.build_energy_observables()
    else:
        n_bath = max(1, (n_qubits - 2) // 2)
        m = AndersonImpurity(n_bath)
        actual_n = m.n_qubits
        qc = m.build_vqe_ansatz(depth)
        observables, coefficients = m.build_energy_observables()
        n_qubits = actual_n  # Use actual qubit count

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
        'depth': depth,
        'device': device,
        'time': elapsed,
        'energy': float(energy),
    }


def run_sweep(model_name, n_qubits_list, chi_list, depth_list, use_gpu):
    """
    Run the full parameter sweep for a given model.

    Returns list of result dicts with 'cpu_time', 'gpu_time', etc.
    """
    results = []
    total = len(n_qubits_list) * len(chi_list) * len(depth_list)
    count = 0

    for n in n_qubits_list:
        for chi in chi_list:
            for depth in depth_list:
                count += 1
                label = (f"  [{count}/{total}] {model_name}: "
                         f"N={n}, χ={chi}, depth={depth}")

                # CPU run
                try:
                    print(f"{label} — CPU ...", end='', flush=True)
                    cpu_res = run_single_benchmark(
                        model_name, model_name, n, chi, depth, use_gpu=False
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
                        gpu_res = run_single_benchmark(
                            model_name, model_name, n, chi, depth, use_gpu=True
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
                    'depth': depth,
                    'cpu_time': cpu_res['time'],
                    'energy': cpu_res['energy'],
                    'gpu_time': gpu_res['time'] if gpu_res else None,
                    'gpu_energy': gpu_res['energy'] if gpu_res else None,
                    'label': f"N={cpu_res['n_qubits']}, χ={chi}, d={depth}",
                }
                results.append(entry)

    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    n_list = FULL_N_QUBITS if FULL_SWEEP else DEFAULT_N_QUBITS
    chi_list = FULL_CHI_VALS if FULL_SWEEP else DEFAULT_CHI_VALS
    depth_list = FULL_DEPTH_VALS if FULL_SWEEP else DEFAULT_DEPTH_VALS

    gpu_available = maestro.is_gpu_available()
    use_gpu = GPU_ENABLED and gpu_available

    print(f"\n{'═' * 70}")
    print(f"  BENCHMARK A: GROUND-STATE PREPARATION (VQE-STYLE)")
    print(f"{'═' * 70}")
    print(f"  Sweep:   N = {n_list}")
    print(f"           χ = {chi_list}")
    print(f"           depth = {depth_list}")
    print(f"  GPU:     {'ENABLED' if use_gpu else 'DISABLED'}"
          f"{'  (not available)' if GPU_ENABLED and not gpu_available else ''}")
    print(f"  Mode:    {'FULL' if FULL_SWEEP else 'TARGETED'}")
    print(f"{'═' * 70}\n")

    all_results = []

    # ── Frustrated Heisenberg ──
    if MODEL_FILTER is None or MODEL_FILTER == 'heisenberg':
        print(f"\n{'─' * 70}")
        print(f"  MODEL 1: Frustrated J₁-J₂ Heisenberg (J₂/J₁ = 0.5)")
        print(f"{'─' * 70}")

        heis_results = run_sweep('heisenberg', n_list, chi_list,
                                 depth_list, use_gpu)
        all_results.extend(heis_results)

        # Plots
        if heis_results:
            # Fix depth for heatmap — use the middle depth
            mid_depth = depth_list[len(depth_list) // 2]
            heatmap_data = [r for r in heis_results if r['depth'] == mid_depth]

            plot_crossover_heatmap(
                heatmap_data,
                f'GPU Speedup — Heisenberg (depth={mid_depth})',
                os.path.join(SCRIPT_DIR, 'gs_heisenberg_crossover.png'),
            )
            plot_energy_convergence(
                [r for r in heis_results if r['depth'] == mid_depth],
                f'Energy Convergence — Heisenberg (depth={mid_depth})',
                os.path.join(SCRIPT_DIR, 'gs_heisenberg_energy.png'),
            )

    # ── Anderson Impurity ──
    if MODEL_FILTER is None or MODEL_FILTER == 'anderson':
        print(f"\n{'─' * 70}")
        print(f"  MODEL 2: Anderson Impurity (SIAM)")
        print(f"{'─' * 70}")

        anderson_results = run_sweep('anderson', n_list, chi_list,
                                     depth_list, use_gpu)
        all_results.extend(anderson_results)

        if anderson_results:
            mid_depth = depth_list[len(depth_list) // 2]
            heatmap_data = [r for r in anderson_results
                            if r['depth'] == mid_depth]

            plot_crossover_heatmap(
                heatmap_data,
                f'GPU Speedup — Anderson (depth={mid_depth})',
                os.path.join(SCRIPT_DIR, 'gs_anderson_crossover.png'),
            )
            plot_energy_convergence(
                [r for r in anderson_results if r['depth'] == mid_depth],
                f'Energy Convergence — Anderson (depth={mid_depth})',
                os.path.join(SCRIPT_DIR, 'gs_anderson_energy.png'),
            )

    # ── Combined scaling ──
    if all_results:
        mid_depth = depth_list[len(depth_list) // 2]
        scaling_data = [r for r in all_results if r['depth'] == mid_depth]

        plot_scaling_curves(
            scaling_data, x_key='n_qubits', group_key='chi',
            title=f'Ground-State Timing: N vs Wall-Clock (depth={mid_depth})',
            save_path=os.path.join(SCRIPT_DIR, 'gs_scaling_n.png'),
            x_label='System Size N (qubits)',
        )
        plot_scaling_curves(
            scaling_data, x_key='chi', group_key='n_qubits',
            title=f'Ground-State Timing: χ vs Wall-Clock (depth={mid_depth})',
            save_path=os.path.join(SCRIPT_DIR, 'gs_scaling_chi.png'),
            x_label='Bond Dimension χ',
        )

        # Timing bars for a representative slice
        bar_data = [r for r in all_results
                    if r['depth'] == mid_depth
                    and r['n_qubits'] in (n_list[-1], n_list[-2])]
        if bar_data:
            plot_timing_bars(
                bar_data,
                'Ground-State: CPU vs GPU Timing',
                os.path.join(SCRIPT_DIR, 'gs_timing_bars.png'),
            )

        save_results(all_results, 'gs_results.json')

    # ── Summary ──
    print(f"\n{'═' * 70}")
    print(f"  GROUND-STATE BENCHMARK SUMMARY")
    print(f"{'═' * 70}")
    print(f"\n  {'Model':<14} {'N':>4} {'χ':>6} {'d':>4} {'CPU(s)':>8} "
          f"{'GPU(s)':>8} {'Speedup':>8} {'Energy':>10}")
    print(f"  {'─' * 66}")

    for r in all_results:
        gpu_str = f"{r['gpu_time']:.2f}" if r.get('gpu_time') else '—'
        sp_str = (f"{r['cpu_time'] / r['gpu_time']:.1f}×"
                  if r.get('gpu_time') else '—')
        print(f"  {r['model']:<14} {r['n_qubits']:>4} {r['chi']:>6} "
              f"{r['depth']:>4} {r['cpu_time']:>8.2f} {gpu_str:>8} "
              f"{sp_str:>8} {r['energy']:>10.4f}")

    print()
