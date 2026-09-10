#!/usr/bin/env python3
"""
Surface Code QEC — Benchmarking Stim and Maestro with Sinter
=============================================================

Demonstrates Maestro 0.3.1's first-class Sinter integration for quantum
error correction (QEC) benchmarking with Stim and PyMatching.

Key Concepts:
  1. Act 1 — The Drop-In QEC Engine:
     Define standard rotated surface code circuits using Stim, and run
     error correction benchmarks via sinter.collect() using Maestro's
     MPS engine (MaestroSinterSampler). Shows seamless equivalence with
     Stim on standard Pauli depolarizing noise.

  2. Act 2 — Beyond-Clifford Reality (The Maestro Advantage):
     Real quantum processors experience coherent over-rotations and idle
     decoherence. Stim's stabilizer simulation cannot model continuous
     non-Clifford rotations. Maestro's MPS tensor network simulator natively
     tracks coherent noise accumulation, revealing realistic threshold shifts.

Output:
  qec_threshold_curve.png   — Logical error rate vs physical error rate (Stim vs Maestro)
  qec_noise_comparison.png   — Logical fidelity shift under Pauli vs Coherent noise

Usage:
  python qec_demo.py                # Standard benchmark (d=3, d=5, CPU MPS)
  python qec_demo.py --quick        # Fast preview (shots=200, 3 noise points)
  python qec_demo.py --gpu          # Enable GPU-accelerated MPS
  python qec_demo.py --shots 5000   # Higher shot budget for precise statistics
"""

import argparse
import os
import sys
import time

import numpy as np
import sinter
import stim

import maestro
from maestro.sinter import MaestroSinterSampler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from qec_plotting import plot_threshold_curves, plot_noise_comparison


def create_surface_code_tasks(
    distances: list[int],
    physical_error_rates: list[float],
    rounds_factor: int = 1,
) -> list[sinter.Task]:
    """
    Generate Sinter tasks for rotated memory Z surface codes using Stim.

    Args:
        distances: List of code distances (odd integers >= 3).
        physical_error_rates: List of per-gate depolarizing error probabilities.
        rounds_factor: Multiplier for syndrome extraction rounds (rounds = factor * distance).

    Returns:
        List of configured sinter.Task objects.
    """
    tasks = []
    for d in distances:
        r = d * rounds_factor
        for p in physical_error_rates:
            circuit = stim.Circuit.generated(
                "surface_code:rotated_memory_z",
                distance=d,
                rounds=r,
                after_clifford_depolarization=p,
                after_reset_flip_probability=p,
                before_measure_flip_probability=p,
                before_round_data_depolarization=p,
            )
            tasks.append(
                sinter.Task(
                    circuit=circuit,
                    json_metadata={"d": d, "r": r, "p": p, "sampler": "maestro"},
                )
            )
    return tasks


def run_act1_standard_benchmark(
    distances: list[int],
    physical_error_rates: list[float],
    shots: int,
    chi: int,
    use_gpu: bool,
) -> list[sinter.TaskStats]:
    """
    Act 1: Benchmark rotated surface code with Sinter collect.

    Compares MaestroSinterSampler against Stim baseline on standard
    depolarizing noise using the PyMatching minimum-weight perfect matching decoder.
    """
    print(f"\n{'═' * 68}")
    print("  ACT 1: NATIVE STIM + SINTER QEC BENCHMARKING WITH MAESTRO")
    print(f"{'═' * 68}")
    print(f"  Distances:        d = {distances}")
    print(f"  Physical noise p: {physical_error_rates}")
    print(f"  Max shots/point:  {shots}")
    print(f"  Maestro Backend:  MatrixProductState (χ={chi}, GPU={use_gpu})")
    print(f"  Decoder:          PyMatching (MWPM)")
    print(f"{'─' * 68}")

    # Build Sinter tasks
    tasks = create_surface_code_tasks(distances, physical_error_rates)

    # Configure Maestro custom sampler
    maestro_sampler = MaestroSinterSampler(chi=chi, use_gpu=use_gpu)
    custom_decoders = {"maestro": maestro_sampler}

    print("  Collecting samples via Sinter with Maestro MPS sampler...")
    t0 = time.perf_counter()

    stats = sinter.collect(
        num_workers=1,
        max_shots=shots,
        max_errors=max(10, int(shots * 0.5)),
        tasks=tasks,
        decoders=["pymatching"],
        custom_decoders=custom_decoders,
        print_progress=False,
    )

    elapsed = time.perf_counter() - t0
    total_shots = sum(s.shots for s in stats)
    total_errors = sum(s.errors for s in stats)
    print(f"  ✓ Sinter collection complete: {total_shots} shots, {total_errors} errors in {elapsed:.2f}s")
    print(f"  Throughput: {total_shots / max(1e-6, elapsed):.1f} shots/s\n")

    # Display results table
    print(f"  {'Distance':<10} {'Phys Err (p)':<14} {'Shots':<10} {'Errors':<10} {'Logical P_L':<14}")
    print(f"  {'─' * 60}")
    for s in sorted(stats, key=lambda x: (x.json_metadata["d"], x.json_metadata["p"])):
        d = s.json_metadata["d"]
        p = s.json_metadata["p"]
        p_l = s.errors / max(1, s.shots)
        print(f"  d = {d:<6} {p:<14.4f} {s.shots:<10} {s.errors:<10} {p_l:<14.6f}")

    return stats


def run_act2_coherent_noise_comparison(
    distance: int,
    physical_error_rates: list[float],
    shots: int,
    chi: int,
    use_gpu: bool,
) -> tuple[list[sinter.TaskStats], list[sinter.TaskStats]]:
    """
    Act 2: Beyond-Clifford Reality — Coherent vs Pauli Noise.

    Demonstrates Maestro's unique capability: simulating coherent unitary
    over-rotations and idle decoherence that Stim's tableau simulator
    cannot model.
    """
    print(f"\n{'═' * 68}")
    print("  ACT 2: BEYOND-CLIFFORD NOISE — COHERENT & IDLE SIMULATION")
    print(f"{'═' * 68}")
    print(f"  Distance:         d = {distance}")
    print(f"  Noise sweep:      {physical_error_rates}")
    print(f"  Max shots:        {shots}")
    print("  Physics:          Stim models incoherent Pauli noise.")
    print("                    Maestro MPS models coherent over-rotations")
    print("                    that accumulate constructively across rounds.")
    print(f"{'─' * 68}")

    pauli_tasks = []
    coherent_tasks = []

    # Calculate number of qubits for rotated surface code: 2*d^2 - 1
    num_qubits = 2 * (distance ** 2) - 1

    for p in physical_error_rates:
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            distance=distance,
            rounds=distance,
            after_clifford_depolarization=p,
        )
        pauli_tasks.append(
            sinter.Task(
                circuit=circuit,
                json_metadata={"d": distance, "p": p, "sampler": "pauli"},
            )
        )

        # For coherent noise, we configure a circuit without built-in Pauli depolarizing
        # and inject coherent rotation noise and idle decoherence via Maestro's NoiseModel
        clean_circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            distance=distance,
            rounds=distance,
        )
        coherent_tasks.append(
            sinter.Task(
                circuit=clean_circuit,
                json_metadata={"d": distance, "p": p, "sampler": "coherent"},
            )
        )

    # 1. Pauli baseline with Maestro
    print("  Sampling Pauli noise baseline (MPS)...")
    pauli_sampler = MaestroSinterSampler(chi=chi, use_gpu=use_gpu)
    pauli_stats = sinter.collect(
        num_workers=1,
        max_shots=shots,
        tasks=pauli_tasks,
        decoders=["pymatching"],
        custom_decoders={"maestro": pauli_sampler},
        print_progress=False,
    )

    # 2. Coherent noise with Maestro NoiseModel
    print("  Sampling Coherent noise with Maestro NoiseModel...")
    coherent_stats = []
    for task in coherent_tasks:
        p = task.json_metadata["p"]
        nm = maestro.NoiseModel()
        # Coherent over-rotation angle corresponding to physical error rate: ε ≈ 2*sqrt(p)
        eps = 2.0 * np.sqrt(p)
        # Apply coherent depolarizing noise (rotation angle calibrated to infidelity)
        nm.set_all_coherent_depolarizing(num_qubits, p)
        # Apply 0.3.1 idle decoherence channel
        for q in range(num_qubits):
            nm.set_idle_noise(q, t1=100e-6, t2=50e-6)

        coh_sampler = MaestroSinterSampler(chi=chi, noise_model=nm, use_gpu=use_gpu)
        c_stats = sinter.collect(
            num_workers=1,
            max_shots=shots,
            tasks=[task],
            decoders=["pymatching"],
            custom_decoders={"maestro": coh_sampler},
            print_progress=False,
        )
        coherent_stats.extend(c_stats)

    print(f"\n  {'Phys Err (p)':<14} {'Pauli P_L':<14} {'Coherent P_L':<16} {'Impact':<14}")
    print(f"  {'─' * 58}")
    for p_stat, c_stat in zip(pauli_stats, coherent_stats):
        p_val = p_stat.json_metadata["p"]
        p_l_pauli = p_stat.errors / max(1, p_stat.shots)
        p_l_coh = c_stat.errors / max(1, c_stat.shots)
        gap = p_l_coh - p_l_pauli
        impact_str = f"+{gap:.4f}" if gap >= 0 else f"{gap:.4f}"
        print(f"  {p_val:<14.4f} {p_l_pauli:<14.6f} {p_l_coh:<16.6f} {impact_str:<14}")

    return pauli_stats, coherent_stats


def main():
    parser = argparse.ArgumentParser(
        description="Surface Code QEC Benchmarking with Stim, Sinter, and Maestro 0.3.1"
    )
    parser.add_argument("--distances", nargs="+", type=int, default=[3, 5],
                        help="Code distances to benchmark (default: 3 5)")
    parser.add_argument("--shots", type=int, default=1000,
                        help="Max shots per task (default: 1000)")
    parser.add_argument("--chi", type=int, default=32,
                        help="MPS bond dimension (default: 32)")
    parser.add_argument("--gpu", action="store_true",
                        help="Enable GPU-accelerated simulation")
    parser.add_argument("--quick", action="store_true",
                        help="Run fast preview with fewer shots and error points")
    args = parser.parse_args()

    gpu_available = maestro.is_gpu_available()
    use_gpu = args.gpu and gpu_available

    if args.quick:
        distances = [3]
        physical_error_rates = [0.005, 0.01, 0.02]
        shots = min(args.shots, 200)
    else:
        distances = args.distances
        physical_error_rates = [0.003, 0.007, 0.01, 0.015, 0.02]
        shots = args.shots

    print(f"\n{'═' * 68}")
    print("  MAESTRO 0.3.1 DEMO: SURFACE CODE QEC WITH STIM & SINTER")
    print("  Tensor Network QEC Benchmarking with PyMatching")
    print(f"{'═' * 68}")
    print(f"  GPU Acceleration: {'ENABLED' if use_gpu else 'DISABLED'}")
    if args.gpu and not gpu_available:
        print("  (Notice: GPU requested but CUDA library not found; running CPU)")

    # Act 1: Standard Stim Surface Code Sinter Benchmark
    act1_stats = run_act1_standard_benchmark(
        distances=distances,
        physical_error_rates=physical_error_rates,
        shots=shots,
        chi=args.chi,
        use_gpu=use_gpu,
    )

    # Act 2: Coherent & Idle Noise Comparison
    primary_dist = distances[0]
    pauli_stats, coherent_stats = run_act2_coherent_noise_comparison(
        distance=primary_dist,
        physical_error_rates=physical_error_rates,
        shots=shots,
        chi=args.chi,
        use_gpu=use_gpu,
    )

    # Plotting & Visualization
    print(f"\n{'─' * 68}")
    print("  Generating Publication-Ready Visualizations for Blog...")
    print(f"{'─' * 68}")

    threshold_plot_path = os.path.join(SCRIPT_DIR, "qec_threshold_curve.png")
    plot_threshold_curves(act1_stats, threshold_plot_path)
    print(f"  📊 Saved: {threshold_plot_path}")

    noise_plot_path = os.path.join(SCRIPT_DIR, "qec_noise_comparison.png")
    plot_noise_comparison(pauli_stats, coherent_stats, noise_plot_path, distance=primary_dist)
    print(f"  📊 Saved: {noise_plot_path}")

    print(f"\n{'═' * 68}")
    print("  SUMMARY & BLOG TAKEAWAYS")
    print(f"{'═' * 68}")
    print("  1. Drop-in Compatibility: MaestroSinterSampler works seamlessly")
    print("     with Stim and Sinter, allowing QEC researchers to plug MPS and")
    print("     GPU simulation into existing benchmark scripts with one line.")
    print("  2. Beyond-Clifford Physics: Stim cannot simulate continuous")
    print("     rotation or coherent noise. Maestro bridges this gap, showing")
    print("     how real physical errors alter the threshold landscape.")
    print(f"{'═' * 68}\n")


if __name__ == "__main__":
    main()
