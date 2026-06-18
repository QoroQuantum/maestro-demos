# Maestro Examples

> 🚀 **Go beyond your laptop.** Try Maestro GPU mode with a **free trial.**
> Sign up at **[maestro.qoroquantum.net](https://maestro.qoroquantum.net)** to run these simulations at scale.

Example simulations demonstrating quantum simulation capabilities with the [Maestro](https://github.com/QoroQuantum/maestro) high-performance quantum circuit simulator — each structured as a **two-phase tutorial** showing the jump from local CPU simulation to GPU-accelerated execution.

## Why GPU Mode?

MPS simulation accuracy is controlled by **bond dimension** χ — but tensor contractions scale as **O(χ³)**. On a CPU, doubling χ means **8× the runtime**. For large systems with high entanglement, CPU-only simulation hits a wall fast.

Maestro's GPU backend (`SimulatorType.Gpu`) parallelizes those O(χ³) contractions on NVIDIA GPUs, delivering **10–100× speedups** on the expensive steps — with zero code changes:

```python
# CPU — works, but slow at high bond dimension
config = maestro.SimulatorConfig(
    simulator_type=maestro.SimulatorType.QCSim,
    simulation_type=maestro.SimulationType.MatrixProductState,
    max_bond_dimension=64,
)
result = qc.estimate(observables, config)

# GPU — same API, same code, just swap one argument
config = maestro.SimulatorConfig(
    simulator_type=maestro.SimulatorType.Gpu,          # ← GPU
    simulation_type=maestro.SimulationType.MatrixProductState,
    max_bond_dimension=256,                                   # ← go higher
)
result = qc.estimate(observables, config)
```

Every example below follows the same pattern: **Phase 1** runs locally on CPU with modest parameters to prove the physics works. **Phase 2** scales up with GPU mode — because your CPU shouldn't be the bottleneck.

---

## Getting Started

```bash
pip install qoro-maestro
```

👉 **[Start your free GPU trial →](https://maestro.qoroquantum.net)**

---

## Examples

### 1. [Rydberg Atom Simulation](./rydberg_atom_simulation) ⚡

Phase diagram and spatial correlations of a 64-atom Rydberg array. Sweeps (Δ, Ω) parameter space to map the Z2 ordered phase.

**The bottleneck:** 64 atoms × 144 parameter points × MPS simulation each. Scaling to higher bond dimension for accuracy multiplies cost by χ³.
**The fix:** GPU mode handles high-χ MPS at a fraction of the CPU time — sharper phase boundaries without the wait.

📓 **[Interactive notebook](./rydberg_atom_simulation/rydberg_atom_simulation.ipynb)** — step-by-step tutorial

---

### 2. [Classical Shadows](./classical_shadows) ⚡

Entanglement detection via MPS-based shadow tomography. Estimates 2nd Rényi entropy during Trotter evolution of the transverse-field Ising model.

**The bottleneck:** Hundreds of independent MPS snapshots at 36 qubits. Each snapshot is a full simulation — sequentially, it crawls.
**The fix:** GPU-accelerated MPS cuts per-snapshot time dramatically, making large-scale shadow tomography feasible.

📓 **[Interactive notebook](./classical_shadows/classical_shadows.ipynb)** — step-by-step tutorial

---

### 3. [Fermi-Hubbard Model](./fermi_hubbard) ⚡

Adaptive 3-tier simulation of a 200-qubit Fermi-Hubbard system. The Lieb-Robinson light cone reduces a 200-qubit problem to ~40 active qubits.

**The bottleneck:** The precision tier (χ=256) dominates runtime. O(χ³) tensor contractions on CPU take hours.
**The fix:** GPU acceleration on the precision tier → **~10× speedup** on the most expensive step. The whole pipeline finishes in minutes.

📓 **[Interactive notebook](./fermi_hubbard/fermi_hubbard.ipynb)** — step-by-step tutorial

---

### 4. [Adaptive Bond Dimension](./adaptive_bond_dimension) ⚡

CPU↔GPU backend switching during MPS time evolution. As entanglement grows, the simulation automatically upgrades from low-χ CPU to high-χ GPU.

**The bottleneck:** High bond dimension means O(χ³) per step. CPU grinds to a halt exactly when accuracy matters most.
**The fix:** GPU mode parallelizes the heavy tensor contractions — accurate AND fast. Same code, one argument change.

📓 **[Interactive notebook](./adaptive_bond_dimension/adaptive_bond_dimension.ipynb)** — step-by-step tutorial

---

### 5. [Quantum Many-Body Scarring](./quantum_scarring) ⚡

PXP fidelity revivals from the Néel state in a Rydberg chain. Tracks staggered magnetization oscillations — a dramatic ETH violation.

**The bottleneck:** 64 atoms × 60 Trotter steps at moderate bond dimension = hours on CPU.
**The fix:** GPU mode delivers the full revival structure of a 64-atom chain without overnight runs.

---

### 6. [Dynamical Quantum Phase Transition](./dynamical_phase_transition) ⚡

Loschmidt echo cusps after a sudden TFIM quench. Non-analytic singularities in the rate function — dynamical analogs of thermodynamic phase transitions.

**The bottleneck:** Multi-quench sweep × 40 Trotter steps each. Crisp cusps need χ=64+ on 80 qubits.
**The fix:** GPU mode makes the full 80-qubit, 3-quench sweep run in reasonable time.

---

### 7. [Surface Code Noise](./surface_code_noise) ⚡

Coherent vs Pauli noise analysis on surface code circuits. Reveals error correlations that stabiliser simulators (Stim) fundamentally cannot capture.

**The bottleneck:** Surface codes at d=5+ have 49+ qubits with deep CX networks. MPS at high χ is needed to faithfully track coherent noise correlations.
**The fix:** GPU mode enables high-fidelity noise characterisation at scale — understanding real hardware noise beyond Pauli approximations.

---

## Ready to Scale?

Every example in this repo works locally on CPU. But when you're ready to go beyond toy parameters:

1. **[Start your free GPU trial](https://maestro.qoroquantum.net)** — no credit card required
2. `pip install qoro-maestro`
3. Add `--gpu` to any example script

**That's it.** Same code, GPU scale.

```bash
# CPU (default)
python scarring_demo.py

# GPU — one flag, 10-100× faster
python scarring_demo.py --gpu
```

👉 **[maestro.qoroquantum.net](https://maestro.qoroquantum.net)**

## Maestro Features Demonstrated

| Feature | API | Examples |
|---------|-----|----------|
| Matrix Product State | `SimulationType.MatrixProductState` | All examples |
| Pauli Propagator | `SimulationType.PauliPropagator` | Fermi-Hubbard (Tier 1) |
| Backend configuration | `SimulatorConfig(...)` | All examples |
| Bond dimension control | `max_bond_dimension=χ` | All examples |
| Expectation values | `qc.estimate(obs, config)` | All examples |
| Bitstring sampling | `qc.execute(config, shots=N)` | Rydberg, Classical Shadows |
| CPU backend | `SimulatorType.QCSim` | All examples |
| GPU acceleration | `SimulatorType.Gpu` | All examples (Phase 2) |

## License

See [LICENSE](./LICENSE) for details.
