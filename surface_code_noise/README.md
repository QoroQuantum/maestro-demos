# Surface Code Noise: Coherent vs Pauli

> 🚀 **Try Maestro GPU mode with a free trial.**
> Sign up at **[maestro.qoroquantum.net](https://maestro.qoroquantum.net)** — no credit card required.

## What It Does

Simulates **noisy CX networks** in surface code topology using multiple Maestro backends to demonstrate:

1. **Coherent vs Pauli noise** — Reveals fundamentally different error landscapes that Stim cannot distinguish
2. **Multi-backend comparison** — PauliPropagator, Stabilizer, and MPS all agree on Clifford noise; only MPS can do coherent noise
3. **Spatial error structure** — Coherent noise creates structured, correlated patterns; Pauli noise is uniform

> *"Stim shows you one noise model. Maestro shows you reality."*

### Noise Models

| Noise Model | How it works | Backends |
|-------------|-------------|----------|
| **Pauli** (incoherent) | Random Rz/Rx rotations per qubit per round | PauliPropagator, Stabilizer, MPS |
| **Clifford Pauli** | Probabilistic X/Y/Z gate insertion | PauliPropagator, Stabilizer, MPS |
| **Coherent** (systematic) | Deterministic over-rotations that accumulate | **MPS only** |

## Usage

```bash
# CPU: d=3 and d=5 noise sweeps (~40s)
python qec_demo.py

# GPU: adds d=7 (97 qubits)
python qec_demo.py --gpu

# Custom distance
python qec_demo.py --distance 7 --gpu
```

Or use the Jupyter notebook: `surface_code_noise.ipynb`

## Code Structure

| File | Purpose |
|------|---------|
| `model.py` | `SurfaceCodeModel` — rotated surface code layout, CX networks, noise injection |
| `qec_demo.py` | Full pipeline: noise sweep, spatial analysis, backend comparison |
| `plotting.py` | Plot functions (noise comparison, heatmaps, backend bars, scaling) |
| `surface_code_noise.ipynb` | Interactive notebook version |

## Output

| File | Description |
|------|-------------|
| `qec_noise_comparison.png` | ⟨Z_L⟩ vs noise strength for d=3 and d=5 |
| `qec_syndrome_heatmap.png` | Spatial error patterns: uniform (Pauli) vs structured (coherent) |
| `qec_backend_comparison.png` | All backends agree on Pauli; only MPS captures coherent |
| `qec_distance_scaling.png` | Fidelity gap across distances |

## Configuration

| Parameter | Default |
|-----------|---------|
| Distances | d=3, d=5 (CPU) |
| CPU χ | 32 |
| GPU χ | 64 |
| Noise range | 0.005 – 0.05 |
| CX rounds | 3 |
| Pauli samples | 20 |

---

👉 **Ready for GPU-accelerated noise analysis?** [Start your free GPU trial](https://maestro.qoroquantum.net) and run with `--gpu`.
