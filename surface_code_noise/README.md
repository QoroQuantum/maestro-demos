# Surface Code QEC with Stim & Sinter in Maestro 0.3.1

> 📖 **Background Article:** Read [The Pauli Illusion: Why Standard QEC Decoders Fail Under Coherent Noise](https://qoroquantum.net/?news=the-pauli-illusion-why-standard-qec-decoders-fail-under-coherent-noise) for the physics and motivation behind this benchmark.
> 
> 🚀 **Benchmark QEC with Maestro and Stim.**
> Try Maestro GPU mode with a free trial at **[maestro.qoroquantum.net](https://maestro.qoroquantum.net)** — no credit card required.

## What It Does

This showcase demonstrates Maestro 0.3.1's native **Sinter and Stim integration** (`maestro.sinter`) alongside its **Matrix Product Operator (MPO)** density matrix engine for scientific quantum error correction (QEC) benchmarking:

1. **Drop-in Sinter Sampler**: Define standard rotated surface code circuits using Stim under the industry-standard SI1000 superconducting noise schedule (`2Q = p`, `Readout = 5p`, `Reset = 2p`, `Idle = 0.1p`), and run standard Sinter benchmarks with PyMatching using Maestro's Matrix Product State (MPS) engine (`MaestroSinterSampler`).
2. **Equivalent Pauli Baseline (Act 1)**: Demonstrates that Maestro seamlessly reproduces Stim's logical error rates on standard depolarizing noise with PyMatching.
3. **The Pauli Illusion & Stacked Threshold Shift (Act 2)**: Real quantum processors suffer from coherent over-rotations and idle dephasing during delays. Stim cannot simulate non-Clifford rotations. Maestro's tensor-network simulation natively tracks coherent noise accumulation and idle decoherence (`set_idle_noise`), demonstrating how realistic hardware noise shifts the fault-tolerance threshold downward. Stacking the coherent and Pauli curves reveals the **Pauli Illusion Zone**, where standard decoders predict error suppression from d=3 to d=5, but physical reality experiences higher error.
4. **Deterministic MPO Simulation & The Memory Wall (Act 3)**: Using `SimulationType.MatrixProductOperator`, Maestro simulates the exact mixed-state density matrix rho without Monte Carlo shot noise across distance-3 (9 data qubits) and distance-5 (25 data qubits). A standard dense density matrix for 25 qubits requires `2^50 * 16 bytes = 16.8 Petabytes` of RAM (impossible on any supercomputer). Maestro compresses the state into an MPO tensor train (bond dimension chi=32) requiring only **~1.6 MB of RAM** and running in seconds on a laptop.

---

## Quickstart

```bash
# Install requirements
pip install -r ../requirements.txt

# Run the standard benchmark (d=3, d=5 on CPU MPS + MPO exact benchmark)
python qec_demo.py

# Fast plot regeneration using cached benchmark stats (runs Act 3 MPO)
python qec_demo.py --plot-only

# Quick preview (fewer shots and noise points)
python qec_demo.py --quick

# Enable GPU-accelerated MPS simulation
python qec_demo.py --gpu

# Custom shot count
python qec_demo.py --shots 5000
```

Or open the interactive Jupyter notebook:
```bash
jupyter notebook surface_code_noise.ipynb
```

---

## Code Structure

| File | Purpose |
|------|---------|
| `qec_demo.py` | Full QEC benchmarking script: Stim circuit generation, Sinter collection, Act 1 baseline, Act 2 coherent noise, Act 3 MPO exact simulation, and plotting |
| `qec_plotting.py` | Publication-ready plotting helpers for stacked threshold shift, logical threshold curves, and dual-panel MPO observable decay / memory wall |
| `surface_code_model.py` | Rotated surface code lattice geometry and CX noise network model for exact MPO simulation |
| `surface_code_noise.ipynb` | Interactive step-by-step notebook tutorial covering all three acts |

---

## Generated Artifacts

| File | Description |
|------|-------------|
| `qec_threshold_shift.png` | Stacked threshold crossing comparison plotting d=3 and d=5 Pauli baseline alongside d=3 and d=5 Coherent + Idle noise, highlighting the shaded **Pauli Illusion Zone** |
| `qec_mpo_logical_decay.png` | Executive dual-panel figure: **Panel A** plots deterministic logical <Z_L> decay for d=3 and d=5 under Pauli vs Coherent noise (no shot noise); **Panel B** plots the **Density Matrix Memory Wall**, contrasting dense exponential RAM explosion (16.8 Petabytes at 25 qubits) against Maestro MPO's linear O(N * chi^2) efficiency (~1.6 MB) |
| `qec_threshold_curve.png` | Logical error rate (P_L) vs physical error rate (p) across code distances d=3 and d=5 |
| `qec_noise_comparison.png` | Shift in decoder performance between incoherent Pauli depolarizing noise and coherent/idle noise |

---

## Scientific Takeaways & Physical Insights

1. **Why PyMatching Fails Under Coherent Noise**:
   Minimum-Weight Perfect Matching (MWPM) decoders map syndrome detection events onto graph edges weighted by independent Poissonian error probabilities `ln((1 - p) / p)`. Continuous unitary over-rotations do not collapse into independent Pauli flips; they create correlated quantum superpositions across stabilizers. When matched by PyMatching, these correlated defect clusters are matched to incorrect boundaries, causing the decoder to actively introduce logical faults.
2. **The Pauli Illusion Zone**:
   In the physical error regime between the coherent threshold (~0.18%) and the Pauli threshold (~0.66%), standard Pauli-based simulations predict that scaling distance from d=3 to d=5 suppresses logical errors. On real hardware with coherent pulse over-rotations, however, d=5 exhibits a higher logical error rate than d=3 because larger codes accumulate more correlated unitary phases.
3. **The Density Matrix Memory Wall & Deterministic MPO**:
   Simulating mixed states for 25 qubits with standard density matrices requires 16.8 Petabytes of RAM, crashing all standard dense simulators. Maestro's Matrix Product Operator (MPO) simulator contracts the exact density matrix evolution in seconds with ~1.6 MB of RAM and zero Monte Carlo variance. It deterministically proves that while stochastic Pauli errors undergo diffusive random walks (slow decay), coherent systematic over-rotations interfere constructively, accelerating logical state collapse on larger code lattices.
