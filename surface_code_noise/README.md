# Surface Code QEC with Stim & Sinter in Maestro 0.3.1

> 🚀 **Benchmark QEC with Maestro and Stim.**
> Try Maestro GPU mode with a free trial at **[maestro.qoroquantum.net](https://maestro.qoroquantum.net)** — no credit card required.

## What It Does

This showcase demonstrates Maestro 0.3.1's native **Sinter and Stim integration** (`maestro.sinter`) for quantum error correction (QEC) benchmarking:

1. **Drop-in Sinter Sampler**: Define standard rotated surface code circuits using Stim (`stim.Circuit.generated("surface_code:rotated_memory_z", ...)`), and run standard Sinter benchmarks with PyMatching using Maestro's Matrix Product State (MPS) engine (`MaestroSinterSampler`).
2. **Equivalent Pauli Baseline**: Demonstrates that Maestro seamlessly reproduces Stim's logical error rates on standard depolarizing noise.
3. **Beyond-Clifford Reality (The Maestro Advantage)**: Real quantum processors suffer from coherent over-rotations and idle dephasing during delays. Stim cannot simulate non-Clifford rotations. Maestro's tensor-network simulation natively tracks coherent noise accumulation and idle decoherence (`set_idle_noise`), demonstrating how realistic hardware noise affects the error threshold.

---

## Quickstart

```bash
# Install requirements
pip install -r ../requirements.txt

# Run the standard benchmark (d=3, d=5 on CPU MPS)
python qec_demo.py

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
| `qec_demo.py` | Full QEC benchmarking script: Stim circuit generation, Sinter collection, Act 1 baseline, Act 2 coherent noise, and plotting |
| `qec_plotting.py` | Publication-ready plotting helpers for logical threshold curves and noise comparisons |
| `surface_code_model.py` | Optional manual rotated surface code circuit builder for fine-grained gate inspection |
| `surface_code_noise.ipynb` | Interactive step-by-step notebook tutorial matching the blog post |

---

## Generated Artifacts

| File | Description |
|------|-------------|
| `qec_threshold_curve.png` | Logical error rate ($P_L$) vs physical error rate ($p$) across code distances $d=3$ and $d=5$ |
| `qec_noise_comparison.png` | Shift in decoder performance between incoherent Pauli depolarizing noise and coherent/idle noise |

---

## Blog Post Key Takeaways

* **Standard Interoperability**: `MaestroSinterSampler` conforms directly to `sinter.Sampler`, meaning you can drop Maestro into any existing Sinter benchmark script with a one-line `custom_decoders={"maestro": MaestroSinterSampler(chi=32)}`.
* **Beyond-Clifford Simulation**: While Stim is restricted to Clifford circuits, Maestro's MPS and GPU engines simulate general unitary rotations, coherent errors, and CPTP noise channels, unlocking realistic QEC threshold studies for hardware development.
