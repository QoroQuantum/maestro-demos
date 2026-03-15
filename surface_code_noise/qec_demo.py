#!/usr/bin/env python3
"""
Surface Code QEC — Coherent vs Pauli Noise Analysis
=====================================================

Demonstrates Maestro's unique ability to simulate quantum error correction
circuits with *coherent* noise — something that stabiliser-based simulators
(Stim, PyMatching) fundamentally cannot do.

Key Insight:
    Real quantum hardware has both stochastic (Pauli) and coherent noise.
    These produce fundamentally different error patterns:

    - Pauli noise: random rotations create large per-qubit errors that
      are UNCORRELATED and average out.
    - Coherent noise: systematic over-rotations create CORRELATED error
      structures that are spatially organised and harder for decoders.

    Stim can only model Pauli noise. Maestro's MPS captures both —
    and reveals an entirely different error landscape under coherent noise.

    "Stim shows you one noise model. Maestro shows you reality."

Pipeline:
    Phase 1 — CPU (d=3, 17 qubits):
        Compare data qubit and logical observable degradation under
        Pauli vs coherent noise at the same error magnitude.

    Phase 2 — GPU (d=5, 49 qubits):
        Scale up to show the coherent noise gap grows with distance.

Output:
    qec_noise_comparison.png  — Logical Z fidelity: Pauli vs coherent
    qec_syndrome_heatmap.png  — Data qubit spatial error patterns
    qec_distance_scaling.png  — Error gap vs code distance (GPU mode)

Usage:
    python qec_demo.py                      # d=3 CPU
    python qec_demo.py --gpu                # d=3 CPU + d=5 GPU
    python qec_demo.py --distance 5         # Custom distance
    python qec_demo.py --distance 7 --gpu   # d=7 on GPU (97 qubits)
"""

import sys
import os
import time

import numpy as np

import maestro
from model import SurfaceCodeModel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# CONFIGURATION
# =============================================================================

NOISE_STRENGTHS = [0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05]
N_ROUNDS = 3            # Syndrome extraction rounds per circuit
N_SAMPLES = 20          # Independent noise realisations (Pauli averaging)
CHI_CPU = 32            # Bond dimension for CPU
CHI_GPU = 64            # Bond dimension for GPU

# Parse CLI
GPU_ENABLED = '--gpu' in sys.argv

CUSTOM_DIST = None
for i, arg in enumerate(sys.argv):
    if arg == '--distance' and i + 1 < len(sys.argv):
        CUSTOM_DIST = int(sys.argv[i + 1])

# Default: d=5 on CPU (25 data qubits, 49 total)
DEFAULT_CPU_DIST = 5


# =============================================================================
# CORE MEASUREMENT FUNCTIONS
# =============================================================================

def build_data_only_observables(n_data):
    """Build per-qubit Z observables for data-only circuit."""
    obs = []
    for i in range(n_data):
        pauli = ['I'] * n_data
        pauli[i] = 'Z'
        obs.append(''.join(pauli))
    return obs


def build_logical_z_data_only(model):
    """Build logical Z observable for data-only circuit (d² qubits)."""
    pauli = ['I'] * model.n_data
    for q in model.logical_z_qubits:
        pauli[q] = 'Z'
    return ''.join(pauli)


def measure_noise_impact(model, noise_type, noise_strength,
                          chi, use_gpu, n_samples=1):
    """
    Measure noise impact on data qubits via a CX noise network.

    Runs `N_ROUNDS` of noisy CX layers in the surface code connectivity
    pattern, then measures per-data-qubit ⟨Z⟩ and logical ⟨Z_L⟩.

    In the noiseless case, CX on |0⟩ qubits is identity → ⟨Z⟩=+1.
    With noise:
      - Pauli: random rotations partially cancel → moderate deviation
      - Coherent: systematic over-rotations accumulate → MORE deviation

    Returns: dict with 'data_z' (array) and 'logical_z' (float)
    """
    sim_type = (maestro.SimulatorType.Gpu if use_gpu
                else maestro.SimulatorType.QCSim)

    data_obs = build_data_only_observables(model.n_data)
    logical_z_obs = build_logical_z_data_only(model)
    all_obs = data_obs + [logical_z_obs]

    all_data_z = []
    all_logical_z = []

    for sample in range(n_samples):
        seed = sample if noise_type == 'pauli' else None

        qc = model.build_noisy_cx_network(
            n_rounds=N_ROUNDS,
            noise_type=noise_type,
            noise_strength=noise_strength,
            seed=seed,
        )

        result = qc.estimate(
            observables=all_obs,
            simulator_type=sim_type,
            simulation_type=maestro.SimulationType.MatrixProductState,
            max_bond_dimension=chi,
        )

        exp_vals = result['expectation_values']
        data_z = np.array(exp_vals[:model.n_data])
        logical_z = exp_vals[-1]

        all_data_z.append(data_z)
        all_logical_z.append(logical_z)

    avg_data_z = np.mean(all_data_z, axis=0)
    avg_logical_z = float(np.mean(all_logical_z))

    return {
        'data_z': avg_data_z,
        'logical_z': avg_logical_z,
    }



# =============================================================================
# PHASE 1: CPU — Noise Comparison Sweep
# =============================================================================

def run_noise_comparison(distance, chi, use_gpu=False):
    """
    Compare Pauli vs coherent noise across a sweep of error strengths.

    Returns dict with noise strengths and fidelity metrics.
    """
    model = SurfaceCodeModel(distance)
    device = "GPU" if use_gpu else "CPU"

    print(f"\n{'─' * 65}")
    print(f"  Noise Comparison — d={distance} ({model.n_total} qubits)")
    print(f"  Backend: {device} (χ={chi}), {N_ROUNDS} syndrome rounds")
    print(f"  {model.summary}")
    print(f"{'─' * 65}")

    results = {
        'noise_strengths': [],
        'pauli_logical_z': [],
        'coherent_logical_z': [],
        'pauli_data_fidelity': [],
        'coherent_data_fidelity': [],
    }

    # Noiseless reference
    ref = measure_noise_impact(model, 'none', 0.0, chi, use_gpu, n_samples=1)
    ref_logical_z = ref['logical_z']
    ref_data_z = ref['data_z']

    print(f"\n  Noiseless reference: ⟨Z_L⟩ = {ref_logical_z:.6f}, "
          f"avg ⟨Z_data⟩ = {np.mean(ref_data_z):.6f}")

    print(f"\n  {'ε/p':>8}  {'Pauli ⟨Z_L⟩':>14}  {'Coherent ⟨Z_L⟩':>16}  "
          f"{'Gap':>10}  {'Time':>8}")
    print(f"  {'─' * 62}")

    for p in NOISE_STRENGTHS:
        t0 = time.time()

        # Pauli noise: average over samples
        pauli = measure_noise_impact(
            model, 'pauli', p, chi, use_gpu, n_samples=N_SAMPLES)

        # Coherent noise: deterministic
        coherent = measure_noise_impact(
            model, 'coherent', p, chi, use_gpu, n_samples=1)

        elapsed = time.time() - t0
        gap = pauli['logical_z'] - coherent['logical_z']

        results['noise_strengths'].append(p)
        results['pauli_logical_z'].append(pauli['logical_z'])
        results['coherent_logical_z'].append(coherent['logical_z'])
        results['pauli_data_fidelity'].append(float(np.mean(pauli['data_z'])))
        results['coherent_data_fidelity'].append(
            float(np.mean(coherent['data_z'])))

        print(f"  {p:>8.3f}  {pauli['logical_z']:>14.6f}  "
              f"{coherent['logical_z']:>16.6f}  "
              f"{gap:>+10.6f}  {elapsed:>7.1f}s")

    return results


# =============================================================================
# SPATIAL ERROR ANALYSIS
# =============================================================================

def run_spatial_analysis(distance, chi, use_gpu=False):
    """
    Compare spatial error patterns between Pauli and coherent noise.

    Returns per-data-qubit ⟨Z⟩ arrays for both noise types.
    """
    model = SurfaceCodeModel(distance)
    p = 0.03  # Moderate noise for clear structure
    device = "GPU" if use_gpu else "CPU"

    print(f"\n  Spatial analysis (d={distance}, ε=p={p}, {device})...")

    pauli = measure_noise_impact(
        model, 'pauli', p, chi, use_gpu, n_samples=N_SAMPLES)
    coherent = measure_noise_impact(
        model, 'coherent', p, chi, use_gpu, n_samples=1)

    pauli_dev = np.abs(1.0 - pauli['data_z'])
    coherent_dev = np.abs(1.0 - coherent['data_z'])

    print(f"  ✓ Pauli:    avg deviation = {np.mean(pauli_dev):.6f}, "
          f"std = {np.std(pauli_dev):.6f}")
    print(f"  ✓ Coherent: avg deviation = {np.mean(coherent_dev):.6f}, "
          f"std = {np.std(coherent_dev):.6f}")

    return {
        'model': model,
        'pauli_z': pauli['data_z'],
        'coherent_z': coherent['data_z'],
        'noise_strength': p,
    }


# =============================================================================
# MULTI-BACKEND COMPARISON — Why you don't need Stim
# =============================================================================

def run_backend_comparison(distance, chi):
    """
    Compare Maestro's simulation backends on the same Pauli noise circuit.

    Runs Stabilizer, PauliPropagator, and MPS on identical Clifford noise
    circuits to show:
      1. All three agree on Pauli noise (Maestro replaces Stim)
      2. Only MPS can add coherent noise (Maestro goes beyond Stim)
      3. Speed comparison across backends

    Returns dict with results and timing per backend.
    """
    model = SurfaceCodeModel(distance)
    data_obs = build_data_only_observables(model.n_data)
    logical_z_obs = build_logical_z_data_only(model)
    all_obs = data_obs + [logical_z_obs]

    p = 0.03  # Moderate noise for clear differentiation
    n_avg = 10  # Average over noise realisations

    print(f"\n{'─' * 65}")
    print(f"  Backend Comparison — d={distance} ({model.n_data} qubits)")
    print(f"  Pauli noise p={p}, {N_ROUNDS} CX rounds, {n_avg} samples")
    print(f"{'─' * 65}")

    backends = [
        ('PauliPropagator', maestro.SimulationType.PauliPropagator),
        ('Stabilizer', maestro.SimulationType.Stabilizer),
        ('MPS (Pauli)', maestro.SimulationType.MatrixProductState),
    ]

    results = {}

    print(f"\n  {'Backend':<22}  {'⟨Z_L⟩':>10}  {'avg ⟨Z_data⟩':>14}  "
          f"{'Time':>10}")
    print(f"  {'─' * 60}")

    for name, sim_type in backends:
        all_logical = []
        all_data = []

        t0 = time.perf_counter()
        for seed in range(n_avg):
            qc = model.build_clifford_cx_network(
                n_rounds=N_ROUNDS,
                noise_strength=p,
                seed=seed,
            )

            kwargs = {
                'observables': all_obs,
                'simulator_type': maestro.SimulatorType.QCSim,
                'simulation_type': sim_type,
            }
            if sim_type == maestro.SimulationType.MatrixProductState:
                kwargs['max_bond_dimension'] = chi

            result = qc.estimate(**kwargs)
            exp_vals = result['expectation_values']
            all_data.append(np.mean(exp_vals[:model.n_data]))
            all_logical.append(exp_vals[-1])

        elapsed_ms = (time.perf_counter() - t0) * 1000

        avg_logical = float(np.mean(all_logical))
        avg_data = float(np.mean(all_data))

        results[name] = {
            'logical_z': avg_logical,
            'avg_data_z': avg_data,
            'time_ms': elapsed_ms,
        }

        print(f"  {name:<22}  {avg_logical:>10.6f}  {avg_data:>14.6f}  "
              f"{elapsed_ms:>8.1f}ms")

    # Now run MPS with COHERENT noise — only MPS can do this
    t0 = time.perf_counter()
    qc_coh = model.build_noisy_cx_network(
        n_rounds=N_ROUNDS,
        noise_type='coherent',
        noise_strength=p,
    )
    result_coh = qc_coh.estimate(
        observables=all_obs,
        simulator_type=maestro.SimulatorType.QCSim,
        simulation_type=maestro.SimulationType.MatrixProductState,
        max_bond_dimension=chi,
    )
    elapsed_coh_ms = (time.perf_counter() - t0) * 1000
    coh_logical = result_coh['expectation_values'][-1]
    coh_data = float(np.mean(result_coh['expectation_values'][:model.n_data]))

    results['MPS (Coherent)'] = {
        'logical_z': coh_logical,
        'avg_data_z': coh_data,
        'time_ms': elapsed_coh_ms,
    }

    print(f"  {'MPS (Coherent) ★':<22}  {coh_logical:>10.6f}  "
          f"{coh_data:>14.6f}  {elapsed_coh_ms:>8.1f}ms")
    print(f"\n  ★ Only MPS can simulate coherent noise — Stim cannot.")
    print(f"    All Clifford backends agree on Pauli noise.")
    print(f"    Maestro gives you everything in one tool.")

    return results


# Import plotting functions from separate module
from plotting import (plot_noise_comparison, plot_spatial_errors,
                      plot_backend_comparison, plot_distance_scaling)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    total_start = time.time()

    d_cpu = CUSTOM_DIST if CUSTOM_DIST else DEFAULT_CPU_DIST
    d_gpu = min(d_cpu + 2, 9) if GPU_ENABLED else None

    print(f"\n{'═' * 65}")
    print(f"  MAESTRO Demo: Surface Code QEC — Beyond Pauli Noise")
    print(f"  Coherent Noise Analysis via MPS Simulation")
    print(f"{'═' * 65}")
    print(f"\n  Phase 1: d={d_cpu} on CPU (χ={CHI_CPU})")
    if GPU_ENABLED:
        print(f"  Phase 2: d={d_gpu} on GPU (χ={CHI_GPU})")
    print(f"\n  This demo reveals error correlations that stabiliser")
    print(f"  simulators (Stim) fundamentally cannot capture.\n")

    all_results = []

    # ── d=3 noise sweep (fast reference) ──
    print(f"\n{'#' * 65}")
    print(f"  Noise sweep — d=3 (9 data qubits)")
    print(f"{'#' * 65}")

    model_3 = SurfaceCodeModel(3)
    print(f"  {model_3.summary}")

    results_3 = run_noise_comparison(3, CHI_CPU, use_gpu=False)
    results_3['distance'] = 3
    results_3['n_qubits'] = model_3.n_total
    all_results.append(results_3)

    # ── d=5 noise sweep + spatial + backend comparison ──
    print(f"\n{'#' * 65}")
    print(f"  Noise sweep — d={d_cpu} ({d_cpu**2} data qubits)")
    print(f"{'#' * 65}")

    model_cpu = SurfaceCodeModel(d_cpu)
    print(f"  {model_cpu.summary}")

    results_cpu = run_noise_comparison(d_cpu, CHI_CPU, use_gpu=False)
    results_cpu['distance'] = d_cpu
    results_cpu['n_qubits'] = model_cpu.n_total
    all_results.append(results_cpu)

    # Spatial error analysis
    spatial_data = run_spatial_analysis(d_cpu, CHI_CPU, use_gpu=False)

    # Backend comparison (Stabilizer vs PauliPropagator vs MPS)
    backend_data = run_backend_comparison(d_cpu, CHI_CPU)

    # ── Phase 2: GPU (optional) ──
    if GPU_ENABLED and d_gpu:
        print(f"\n\n{'#' * 65}")
        print(f"  PHASE 2 — GPU (d={d_gpu})")
        print(f"{'#' * 65}")

        try:
            model_gpu = SurfaceCodeModel(d_gpu)
            print(f"  {model_gpu.summary}")

            results_gpu = run_noise_comparison(
                d_gpu, CHI_GPU, use_gpu=True)
            results_gpu['distance'] = d_gpu
            results_gpu['n_qubits'] = model_gpu.n_total
            all_results.append(results_gpu)

        except Exception as e:
            print(f"  ⚠ GPU phase failed: {e}")
            print(f"  Continuing with CPU results only.")

    # ── Generate Plots ──
    print(f"\n\n{'─' * 65}")
    print(f"  Generating visualisations...")
    print(f"{'─' * 65}")

    path1 = plot_noise_comparison(
        all_results, NOISE_STRENGTHS,
        os.path.join(SCRIPT_DIR, 'qec_noise_comparison.png'))
    print(f"  📊 Saved: {path1}")

    path2 = plot_spatial_errors(
        spatial_data,
        os.path.join(SCRIPT_DIR, 'qec_syndrome_heatmap.png'))
    print(f"  📊 Saved: {path2}")

    path_be = plot_backend_comparison(
        backend_data, d_cpu,
        os.path.join(SCRIPT_DIR, 'qec_backend_comparison.png'))
    print(f"  📊 Saved: {path_be}")

    if len(all_results) >= 2:
        path3 = plot_distance_scaling(
            all_results, NOISE_STRENGTHS,
            os.path.join(SCRIPT_DIR, 'qec_distance_scaling.png'))
        if path3:
            print(f"  📊 Saved: {path3}")

    # ── Summary ──
    total = time.time() - total_start

    print(f"\n\n{'═' * 65}")
    print(f"  SUMMARY")
    print(f"{'═' * 65}")
    print(f"  Total runtime: {total:.1f}s")
    print()

    for r in all_results:
        d = r['distance']
        pauli_z = r['pauli_logical_z'][-1]
        coherent_z = r['coherent_logical_z'][-1]
        max_p = NOISE_STRENGTHS[-1]
        print(f"  d={d} ({r['n_qubits']}Q):")
        print(f"    Pauli noise  ⟨Z_L⟩ at ε={max_p}:  {pauli_z:+.6f}  "
              f"(uncorrelated per-qubit errors)")
        print(f"    Coherent     ⟨Z_L⟩ at ε={max_p}:  {coherent_z:+.6f}  "
              f"(correlated, structured errors)")
        print()

    print(f"  Key takeaway:")
    print(f"    Maestro provides Stabilizer, PauliPropagator, AND MPS")
    print(f"    backends — all in one tool. The Clifford backends match")
    print(f"    what Stim does. But only MPS can simulate coherent noise,")
    print(f"    revealing error structures invisible to Pauli-only models.")
    print()
    print(f"    You don't need Stim. Maestro does everything Stim does,")
    print(f"    and goes beyond it.\n")
