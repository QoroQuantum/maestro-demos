#!/usr/bin/env python3
"""
GPU MPS Crossover — Master Benchmark Runner
=============================================

Runs both benchmarks (ground-state preparation + quench dynamics) on both
physics models (frustrated Heisenberg + Anderson impurity) and produces a
combined crossover analysis.

Usage:
    python run_all.py              # CPU-only, targeted sweep
    python run_all.py --gpu        # Include GPU comparisons
    python run_all.py --full       # Full parameter sweep
    python run_all.py --full --gpu # Full sweep with GPU
"""

import sys
import os
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import maestro
from plotting import plot_crossover_summary, load_results

GPU_ENABLED = '--gpu' in sys.argv
FULL_SWEEP = '--full' in sys.argv


def main():
    gpu_available = maestro.is_gpu_available()
    use_gpu = GPU_ENABLED and gpu_available

    print(f"\n{'═' * 70}")
    print(f"  GPU MPS CROSSOVER BENCHMARK SUITE")
    print(f"{'═' * 70}")
    print(f"""
  Two physics models that stress MPS simulation:

  Model 1: Frustrated Spin Chain (J₁-J₂ Heisenberg)
    H = J₁ Σᵢ S⃗ᵢ·S⃗ᵢ₊₁ + J₂ Σᵢ S⃗ᵢ·S⃗ᵢ₊₂
    J₂/J₁ = 0.5 (critical frustration point)
    NNN couplings → non-local gates → stresses swap mechanism

  Model 2: Anderson Impurity (SIAM)
    Single magnetic impurity + conduction electron bath
    Jordan-Wigner: [imp↑, imp↓, bath₁↑, bath₁↓, ...]
    Hubbard U → ZZ;  hybridization V → XX+YY hopping

  Two benchmarks:

  Benchmark A: Ground-State Preparation (VQE-style)
    Ry rotations + CX entangling layers
    Sweep: N (8–100), χ (16–1024), depth (2–16)

  Benchmark B: Real-Time Quench Dynamics
    Néel state → Trotterized evolution
    Entanglement grows linearly → χ requirements grow
    Sweep: steps (5–100), χ (16–1024), N (8–100)

  Goal: Find (N, χ, depth) crossover where GPU MPS > CPU MPS.
  Expected: χ ≳ 128–256 with deep circuits.
""")
    print(f"  GPU:  {'ENABLED' if use_gpu else 'DISABLED'}"
          f"{'  (not available)' if GPU_ENABLED and not gpu_available else ''}")
    print(f"  Mode: {'FULL' if FULL_SWEEP else 'TARGETED'}")
    print(f"{'═' * 70}\n")

    total_start = time.time()

    # ── Benchmark A: Ground-State Preparation ──
    print(f"\n{'#' * 70}")
    print(f"  BENCHMARK A: GROUND-STATE PREPARATION")
    print(f"{'#' * 70}\n")

    gs_args = [sys.executable, os.path.join(SCRIPT_DIR,
               'benchmark_ground_state.py')]
    if GPU_ENABLED:
        gs_args.append('--gpu')
    if FULL_SWEEP:
        gs_args.append('--full')

    os.system(' '.join(gs_args))

    # ── Benchmark B: Quench Dynamics ──
    print(f"\n{'#' * 70}")
    print(f"  BENCHMARK B: QUENCH DYNAMICS")
    print(f"{'#' * 70}\n")

    qd_args = [sys.executable, os.path.join(SCRIPT_DIR,
               'benchmark_quench_dynamics.py')]
    if GPU_ENABLED:
        qd_args.append('--gpu')
    if FULL_SWEEP:
        qd_args.append('--full')

    os.system(' '.join(qd_args))

    # ── Combined Crossover Analysis ──
    print(f"\n{'#' * 70}")
    print(f"  COMBINED CROSSOVER ANALYSIS")
    print(f"{'#' * 70}\n")

    all_results = []
    for fname in ['gs_results.json', 'quench_results.json']:
        fpath = os.path.join(SCRIPT_DIR, fname)
        if os.path.exists(fpath):
            try:
                all_results.extend(load_results(fname))
            except Exception as e:
                print(f"  ⚠ Could not load {fname}: {e}")

    if all_results:
        plot_crossover_summary(
            all_results,
            os.path.join(SCRIPT_DIR, 'gpu_crossover_summary.png'),
        )

    total_time = time.time() - total_start

    # ── Final Summary ──
    print(f"\n{'═' * 70}")
    print(f"  COMPLETE — Total runtime: {total_time:.1f}s")
    print(f"{'═' * 70}")
    print(f"\n  Results saved to: {SCRIPT_DIR}")
    print(f"  Data files:")
    for f in ['gs_results.json', 'quench_results.json']:
        fpath = os.path.join(SCRIPT_DIR, f)
        if os.path.exists(fpath):
            print(f"    ✓ {f}")
    print(f"\n  Plots:")
    for f in sorted(os.listdir(SCRIPT_DIR)):
        if f.endswith('.png'):
            print(f"    📊 {f}")

    if use_gpu:
        # Crossover analysis
        gpu_results = [r for r in all_results
                       if r.get('gpu_time') and r['gpu_time'] > 0]
        if gpu_results:
            wins = sum(1 for r in gpu_results
                       if r['cpu_time'] / r['gpu_time'] > 1.0)
            total_pts = len(gpu_results)
            print(f"\n  GPU wins: {wins}/{total_pts} benchmark points")

            # Find crossover boundary
            crossovers = [r for r in gpu_results
                          if 0.8 < r['cpu_time'] / r['gpu_time'] < 1.2]
            if crossovers:
                avg_chi = sum(r['chi'] for r in crossovers) / len(crossovers)
                avg_n = sum(r['n_qubits'] for r in crossovers) / len(crossovers)
                print(f"  Crossover region: N ≈ {avg_n:.0f}, χ ≈ {avg_chi:.0f}")
    else:
        print(f"\n  Run with --gpu to find the CPU→GPU crossover threshold.")
        print(f"  Expected crossover: χ ≳ 128–256 with deep circuits.")

    print()


if __name__ == '__main__':
    main()
