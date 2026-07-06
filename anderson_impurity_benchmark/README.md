# GPU MPS Crossover Benchmark

> 🚀 **Find where GPU MPS beats CPU MPS.** Sweeps system size, bond dimension, and circuit depth to identify the crossover threshold.

Two physically demanding models stress-test MPS simulation with competing interactions and non-local couplings. Two benchmarks probe the regimes where GPU parallelism pays off.

## Physics Models

### Model 1: Frustrated Spin Chain (J₁-J₂ Heisenberg)

```
H = J₁ Σᵢ S⃗ᵢ·S⃗ᵢ₊₁  +  J₂ Σᵢ S⃗ᵢ·S⃗ᵢ₊₂
```

- **Nearest-neighbour** (J₁) and **next-nearest-neighbour** (J₂) spin couplings compete → **frustration**
- **J₂/J₁ ≈ 0.5** is the critical point — the hardest regime for MPS due to strong entanglement
- NNN couplings map to **non-local 2-qubit gates** on the MPS chain → stresses swap mechanism

### Model 2: Anderson Impurity (Simplified Fermionic Prototype)

```
H = ε_d Σ_σ n_{d,σ}  +  U n_{d,↑} n_{d,↓}  +  Σ_{k,σ} ε_k n_{k,σ}  +  V Σ_{k,σ} (c†_{d,σ} c_{k,σ} + h.c.)
```

- Single magnetic impurity coupled to a bath of conduction electrons (**SIAM**)
- Mapped to qubits via **Jordan-Wigner transformation**
- Qubit layout: `[imp↑, imp↓, bath₁↑, bath₁↓, ...]`
- **Hubbard U** → ZZ interaction; **hybridization V** → XX+YY hopping terms

## Benchmarks

### Benchmark A: Ground-State Preparation

- **VQE-style ansatz** circuits with Ry rotations + CX entangling layers
- **Sweep:** system size (8–100 qubits), bond dimension (16–1024), circuit depth (2–16 layers)
- **Measure:** energy via Pauli expectation values ⟨XX⟩, ⟨YY⟩, ⟨ZZ⟩ on relevant pairs

### Benchmark B: Real-Time Quench Dynamics

- Prepare initial state (Néel state |1010…⟩) → suddenly change Hamiltonian → **Trotterized time evolution**
- Each Trotter step: decompose exp(-iHΔt) into 1- and 2-qubit gates (Rx, Rz, CX, CRz)
- **Entanglement grows linearly** with time → bond dimension requirements grow → GPU advantage kicks in
- **Sweep:** Trotter steps (5–100), bond dimension, system size

## GPU Crossover Threshold

The goal is to find the **(N, χ, circuit depth)** combinations where GPU MPS outperforms CPU MPS.

**Expected crossover:** large bond dimension (χ ≳ 128–256) and/or large system size with deep circuits (many Trotter steps).

## Usage

```bash
# CPU-only, targeted sweep (quick — ~minutes)
python run_all.py

# Include GPU comparisons
python run_all.py --gpu

# Full parameter sweep (exhaustive — can take hours)
python run_all.py --full

# Full sweep with GPU
python run_all.py --full --gpu
```

### Individual benchmarks

```bash
# Ground-state only, specific model
python benchmark_ground_state.py --model heisenberg
python benchmark_ground_state.py --model anderson --gpu

# Quench dynamics only
python benchmark_quench_dynamics.py --model heisenberg --gpu
python benchmark_quench_dynamics.py --full --gpu
```

## Output

### Data Files
- `gs_results.json` — Ground-state benchmark timing and energy data
- `quench_results.json` — Quench dynamics benchmark data

### Plots
- `gs_*_crossover.png` — GPU speedup heatmaps (ground-state)
- `gs_*_energy.png` — Energy convergence with bond dimension
- `gs_scaling_*.png` — Wall-clock scaling curves
- `quench_*_crossover.png` — GPU speedup heatmaps (quench dynamics)
- `quench_*_steps.png` — Per-Trotter-step timing (shows entanglement growth)
- `quench_scaling_*.png` — Scaling curves for quench dynamics
- `gpu_crossover_summary.png` — Combined crossover threshold scatter plot

## Key Insight

MPS tensor contractions scale as **O(χ³)**. CPU handles small χ efficiently, but at χ ≳ 128–256, the GPU's massively parallel tensor contractions dominate. The **quench dynamics** benchmark is especially revealing: as entanglement grows with each Trotter step, the effective bond dimension requirement increases, and the GPU advantage accelerates.

```python
# CPU — works, but slow at high bond dimension
config = maestro.SimulatorConfig(
    simulator_type=maestro.SimulatorType.QCSim,
    simulation_type=maestro.SimulationType.MatrixProductState,
    max_bond_dimension=64,
)

# GPU — same code, same API, just swap one argument
config = maestro.SimulatorConfig(
    simulator_type=maestro.SimulatorType.Gpu,          # ← GPU
    simulation_type=maestro.SimulationType.MatrixProductState,
    max_bond_dimension=512,                            # ← go higher
)
```

## Requirements

```bash
pip install qoro-maestro>=0.2.14 numpy matplotlib
```
