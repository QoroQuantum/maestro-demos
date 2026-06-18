"""
Classical Shadows — Helper Functions
=====================================

Reusable building blocks for the Maestro classical shadows showcase.
Each function does one thing, takes explicit arguments, returns results.
No globals, no side effects beyond circuit construction.
"""

import numpy as np
import maestro
from maestro.circuits import QuantumCircuit
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    """Single source of truth for all simulation parameters."""
    # Lattice
    lx: int = 6
    ly: int = 6

    # Hamiltonian
    j_coupling: float = 1.0       # ZZ coupling
    h_field: float = 1.0          # Transverse field (ordered phase)

    # Time evolution
    t_total: float = 2.0
    n_trotter_steps: int = 10

    # MPS bond dimensions
    chi_low: int = 16             # CPU low-bond stage
    chi_high: int = 64            # High-bond stage (GPU when available)
    entanglement_threshold: float = 0.5

    # Classical shadows
    n_shadows: int = 200          # Snapshots per depth point
    n_shots: int = 1000           # Bitstrings for Act 5 demo
    subsystem_size: int = 2       # Qubits in subsystem A

    # Sweep
    trotter_depths: List[int] = field(
        default_factory=lambda: [1, 2, 3, 4, 5, 6, 7, 8, 10]
    )

    # Hardware
    use_gpu: bool = False

    # Derived (computed in __post_init__)
    n_qubits: int = field(init=False)
    dt: float = field(init=False)

    def __post_init__(self):
        self.n_qubits = self.lx * self.ly
        self.dt = self.t_total / self.n_trotter_steps

    @property
    def simulator_type(self):
        """Backend selector: GPU or CPU."""
        return (maestro.SimulatorType.Gpu if self.use_gpu
                else maestro.SimulatorType.QCSim)

    @property
    def d_A(self):
        """Hilbert space dimension of subsystem A."""
        return 2 ** self.subsystem_size


# ─────────────────────────────────────────────────────────────────────
# Lattice geometry
# ─────────────────────────────────────────────────────────────────────

def site_index(x, y, ly):
    """Map 2D lattice coordinate (x, y) to linear qubit index."""
    return x * ly + y


def site_coords(idx, ly):
    """Map linear qubit index to 2D lattice coordinate (x, y)."""
    return idx // ly, idx % ly


def get_nn_bonds(lx: int, ly: int) -> List[Tuple[int, int]]:
    """Nearest-neighbor bonds on an LX×LY 2D rectangular lattice."""
    bonds = []
    for x in range(lx):
        for y in range(ly):
            idx = site_index(x, y, ly)
            if x + 1 < lx:
                bonds.append((idx, site_index(x + 1, y, ly)))
            if y + 1 < ly:
                bonds.append((idx, site_index(x, y + 1, ly)))
    return bonds


# ─────────────────────────────────────────────────────────────────────
# Circuit construction
# ─────────────────────────────────────────────────────────────────────

CLIFFORD_GATES = ['I', 'H', 'HS', 'SH', 'HSdg', 'SHSdg']


def apply_clifford_gate(qc: QuantumCircuit, qubit: int, label: str):
    """Apply a named single-qubit Clifford gate to a circuit."""
    if label == 'I':
        pass
    elif label == 'H':
        qc.h(qubit)
    elif label == 'HS':
        qc.s(qubit)
        qc.h(qubit)
    elif label == 'SH':
        qc.h(qubit)
        qc.s(qubit)
    elif label == 'HSdg':
        qc.sdg(qubit)
        qc.h(qubit)
    elif label == 'SHSdg':
        qc.sdg(qubit)
        qc.h(qubit)
        qc.s(qubit)


def build_tfim_trotter_circuit(
    n_qubits: int,
    bonds: List[Tuple[int, int]],
    j: float, h: float, dt: float,
    n_steps: int,
) -> QuantumCircuit:
    """
    Build a TFIM Trotterized time-evolution circuit.

    Prepares |+⟩^n, then applies `n_steps` first-order Trotter layers:
      exp(-i J dt ZZ) on each bond, exp(-i h dt X) on each qubit.
    """
    qc = QuantumCircuit()
    for q in range(n_qubits):
        qc.h(q)
    for _ in range(n_steps):
        for q1, q2 in bonds:
            qc.cx(q1, q2)
            qc.rz(q2, 2.0 * j * dt)
            qc.cx(q1, q2)
        for q in range(n_qubits):
            qc.h(q)
            qc.rz(q, 2.0 * h * dt)
            qc.h(q)
    return qc


def append_random_clifford_layer(
    qc: QuantumCircuit, n_qubits: int, rng: np.random.Generator
) -> List[str]:
    """
    Append a random single-qubit Clifford to each qubit.
    Returns the list of Clifford labels (needed for shadow reconstruction).
    """
    labels = []
    for q in range(n_qubits):
        label = rng.choice(CLIFFORD_GATES)
        labels.append(label)
        apply_clifford_gate(qc, q, label)
    return labels


# ─────────────────────────────────────────────────────────────────────
# PP Scout: identify the most entangled subsystem
# ─────────────────────────────────────────────────────────────────────

def scout_entanglement(config: Config, n_steps: int = 3) -> dict:
    """
    Scout phase: use Pauli Propagator on a TFIM Trotter circuit to
    identify the most and least entangled subsystems.

    Computes ⟨Z_i Z_j⟩ for all nearest-neighbor bonds and ranks by
    coordination-weighted entanglement score:
      score(i,j) = |⟨Z_i Z_j⟩| × (nn_i + nn_j) / 8

    Args:
        config: Config object
        n_steps: Number of Trotter steps for the scout circuit.
                 More steps = more time evolution = better differentiation
                 between bonds. Default=3 balances speed vs accuracy.

    Returns dict with 'hot_qubits', 'cold_qubits', and full correlation data.
    """
    n = config.n_qubits
    bonds = get_nn_bonds(config.lx, config.ly)

    qc = build_tfim_trotter_circuit(
        n, bonds, config.j_coupling, config.h_field, config.dt, n_steps=n_steps
    )

    observables = [
        build_pauli_observable(n, {q1: 'Z', q2: 'Z'})
        for q1, q2 in bonds
    ]
    result = qc.estimate(observables, maestro.SimulatorConfig(
        simulation_type=maestro.SimulationType.PauliPropagator,
    ))
    exp_vals = result['expectation_values']

    def coordination(q):
        x, y = site_coords(q, config.ly)
        nn = 0
        if x > 0: nn += 1
        if x < config.lx - 1: nn += 1
        if y > 0: nn += 1
        if y < config.ly - 1: nn += 1
        return nn

    scored = []
    for (q1, q2), zz in zip(bonds, exp_vals):
        nn_sum = coordination(q1) + coordination(q2)
        score = abs(zz) * nn_sum / 8.0
        scored.append(((q1, q2), zz, score, nn_sum))

    scored.sort(key=lambda x: x[2], reverse=True)

    hot_bond = scored[0]
    cold_bond = scored[-1]

    return {
        'hot_qubits': list(hot_bond[0]),
        'hot_corr': hot_bond[1],
        'hot_score': hot_bond[2],
        'hot_coord': hot_bond[3],
        'cold_qubits': list(cold_bond[0]),
        'cold_corr': cold_bond[1],
        'cold_score': cold_bond[2],
        'cold_coord': cold_bond[3],
        'scored_bonds': scored,
    }


# ─────────────────────────────────────────────────────────────────────
# Classical shadows: reconstruction and estimation
# ─────────────────────────────────────────────────────────────────────

def clifford_unitary_matrix(label: str) -> np.ndarray:
    """Return the 2×2 unitary matrix for a single-qubit Clifford gate."""
    I = np.eye(2, dtype=complex)
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    S = np.array([[1, 0], [0, 1j]], dtype=complex)
    Sdg = np.array([[1, 0], [0, -1j]], dtype=complex)
    gates = {
        'I': I, 'H': H, 'HS': H @ S, 'SH': S @ H,
        'HSdg': H @ Sdg, 'SHSdg': S @ H @ Sdg,
    }
    return gates[label]


def build_shadow_snapshot(
    bits: List[int],
    clifford_labels: List[str],
    subsystem_qubits: List[int],
) -> np.ndarray:
    """
    Build the reduced shadow density matrix ρ̂_A for an arbitrary subsystem.

    ρ̂_A = ⊗_{q∈A} (3 U_q† |b_q⟩⟨b_q| U_q − I)   [Huang et al. 2020]
    """
    single_qubit_shadows = []
    for q in subsystem_qubits:
        b = bits[q] if q < len(bits) else 0
        ket = np.array([[1 - b], [b]], dtype=complex)
        proj = ket @ ket.conj().T
        U = clifford_unitary_matrix(clifford_labels[q])
        shadow_q = 3.0 * (U.conj().T @ proj @ U) - np.eye(2, dtype=complex)
        single_qubit_shadows.append(shadow_q)

    rho = single_qubit_shadows[0]
    for i in range(1, len(single_qubit_shadows)):
        rho = np.kron(rho, single_qubit_shadows[i])
    return rho


def estimate_purity_from_shadows(shadows: List[np.ndarray]) -> float:
    """
    Unbiased U-statistics estimator for Tr(ρ_A²).
    Uses only cross-terms: Tr(ρ²) ≈ (2/M(M-1)) Σ_{i<j} Tr(ρ̂_i ρ̂_j)
    """
    d_A = shadows[0].shape[0]
    running_sum = np.zeros((d_A, d_A), dtype=complex)
    cross_total = 0.0
    for i, rho in enumerate(shadows):
        if i > 0:
            cross_total += np.real(np.trace(rho @ running_sum))
        running_sum += rho
    M = len(shadows)
    return float((2.0 * cross_total) / (M * (M - 1)))


def renyi_s2(purity: float, d_A: int) -> Tuple[float, float]:
    """Compute S₂ = -log₂(purity), with clamping. Returns (S₂, clamped_purity)."""
    clamped = float(np.clip(purity, 1.0 / d_A, 1.0))
    return -np.log2(clamped), clamped


# ─────────────────────────────────────────────────────────────────────
# Shadow snapshot collection
# ─────────────────────────────────────────────────────────────────────

def collect_shadow_snapshots(
    config: Config,
    n_trotter_steps: int,
    bonds: List[Tuple[int, int]],
    subsystem_qubits: List[int],
    verbose: bool = True,
) -> List[np.ndarray]:
    """
    Collect M shadow snapshots for a given Trotter depth and subsystem.

    Each snapshot: build circuit → random Clifford layer → measure → reconstruct ρ̂_A.
    """
    n = config.n_qubits
    shadows = []

    for s_idx in range(config.n_shadows):
        rng = np.random.default_rng(seed=s_idx)

        qc = build_tfim_trotter_circuit(
            n, bonds, config.j_coupling, config.h_field, config.dt, n_trotter_steps
        )
        labels = append_random_clifford_layer(qc, n, rng)
        qc.measure_all()

        result = qc.execute(maestro.SimulatorConfig(
            simulator_type=config.simulator_type,
            simulation_type=maestro.SimulationType.MatrixProductState,
            max_bond_dimension=config.chi_high if config.use_gpu else config.chi_low,
        ), shots=1)
        bitstring = list(result['counts'].keys())[0]
        bits = [int(b) for b in bitstring[:n]]

        rho = build_shadow_snapshot(bits, labels, subsystem_qubits)
        shadows.append(rho)

        if verbose and (s_idx + 1) % 100 == 0:
            print(f"    Collected {s_idx + 1}/{config.n_shadows} snapshots...")

    return shadows


# ─────────────────────────────────────────────────────────────────────
# Exact reference (statevector ED)
# ─────────────────────────────────────────────────────────────────────

def compute_exact_s2(
    config: Config,
    subsystem_qubits: Optional[List[int]] = None,
) -> Optional[dict]:
    """
    Compute exact S₂(t) for an arbitrary subsystem via full statevector simulation.

    Returns None if system is too large (n > 20).
    """
    n = config.n_qubits
    if n > 20:
        return None

    if subsystem_qubits is None:
        subsystem_qubits = list(range(config.subsystem_size))

    bonds = get_nn_bonds(config.lx, config.ly)
    A_size = len(subsystem_qubits)
    d_A = 2 ** A_size

    H_gate = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

    def rz_mat(theta):
        return np.array([[np.exp(-1j * theta / 2), 0],
                         [0, np.exp(1j * theta / 2)]], dtype=complex)

    def apply_single(state, gate, q):
        state = np.moveaxis(state, q, 0)
        shape = state.shape
        state = gate @ state.reshape(2, -1)
        state = state.reshape(shape)
        return np.moveaxis(state, 0, q)

    def apply_cx(state, ctrl, targ):
        n_q = len(state.shape)
        idx_1 = [slice(None)] * n_q
        idx_1[ctrl] = slice(1, 2)
        block1 = state[tuple(idx_1)].copy()
        state_t = np.moveaxis(block1, targ, 0)
        shape = state_t.shape
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        state_t = (X @ state_t.reshape(2, -1)).reshape(shape)
        block1_new = np.moveaxis(state_t, 0, targ)
        result = state.copy()
        result[tuple(idx_1)] = block1_new
        return result

    def trotter_step(state):
        for q1, q2 in bonds:
            state = apply_cx(state, q1, q2)
            state = apply_single(state, rz_mat(2.0 * config.j_coupling * config.dt), q2)
            state = apply_cx(state, q1, q2)
        for q in range(n):
            state = apply_single(state, H_gate, q)
            state = apply_single(state, rz_mat(2.0 * config.h_field * config.dt), q)
            state = apply_single(state, H_gate, q)
        return state

    def compute_s2_from_state(state):
        env_qubits = [q for q in range(n) if q not in subsystem_qubits]
        perm = list(subsystem_qubits) + env_qubits
        state_perm = np.transpose(state, perm)
        psi = state_perm.reshape(d_A, -1)
        rho_A = psi @ psi.conj().T
        tr_rho_sq = np.real(np.trace(rho_A @ rho_A))
        S2 = -np.log2(max(tr_rho_sq, 1.0 / d_A))
        return S2, tr_rho_sq

    state = np.ones((2,) * n, dtype=complex) / np.sqrt(2 ** n)

    max_depth = max(config.trotter_depths)
    s2_by_depth = {}
    for step in range(1, max_depth + 1):
        state = trotter_step(state)
        if step in config.trotter_depths:
            s2, purity = compute_s2_from_state(state)
            s2_by_depth[step] = (s2, purity)

    results = {'depths': [], 'times': [], 's2': [], 'purity': []}
    for d in config.trotter_depths:
        s2, purity = s2_by_depth[d]
        results['depths'].append(d)
        results['times'].append(d * config.dt)
        results['s2'].append(s2)
        results['purity'].append(purity)
    return results


# ─────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────

def plot_energy_evolution(times, energies, backends, save_path):
    """Plot E(t) during time evolution, annotating backend handoff."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#2196F3' if 'low' in b else '#E91E63' for b in backends]
    for i in range(len(times) - 1):
        ax.plot(times[i:i+2], energies[i:i+2],
                color=colors[i], linewidth=2, marker='o', markersize=5)
    switch_idx = next(
        (i for i in range(1, len(backends)) if backends[i] != backends[i-1]), None
    )
    if switch_idx is not None:
        ax.axvline(x=times[switch_idx], color='red', linestyle='--',
                   alpha=0.7, label=f'Backend handoff (t={times[switch_idx]:.2f})')
        ax.legend()
    ax.set_xlabel('Simulation Time t')
    ax.set_ylabel('Energy E(t)')
    ax.set_title(f'TFIM Energy Evolution — {len(energies)-1} Trotter Steps')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return save_path


def plot_scout_comparison(
    hot_results: dict,
    cold_results: dict,
    hot_exact: Optional[dict],
    cold_exact: Optional[dict],
    hot_qubits: List[int],
    cold_qubits: List[int],
    config: Config,
    save_path: str,
):
    """
    Plot S₂ vs time comparing the PP-scouted 'hot' subsystem against
    the 'cold' subsystem.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
    max_s2 = config.subsystem_size

    # ── Left panel: Hot subsystem (PP-selected) ──
    if hot_exact is not None:
        ax1.fill_between(hot_exact['times'], 0, hot_exact['s2'],
                         alpha=0.12, color='green')
        ax1.plot(hot_exact['times'], hot_exact['s2'],
                 'g-', linewidth=2.5, label='Exact (statevector ED)')

    ax1.plot(hot_results['times'], hot_results['s2'], 'o--',
             color='#7B1FA2', linewidth=2, markersize=7,
             label='Classical shadows (Maestro MPS)')
    ax1.axhline(y=max_s2, color='gray', linestyle=':', alpha=0.5)

    ax1.set_xlabel('Simulation Time t')
    ax1.set_ylabel('2nd-order Rényi Entropy S₂')
    hx0, hy0 = site_coords(hot_qubits[0], config.ly)
    hx1, hy1 = site_coords(hot_qubits[1], config.ly)
    ax1.set_title(f'PP-Scouted "Hot" Subsystem\n'
                  f'Qubits {hot_qubits} — site ({hx0},{hy0}),({hx1},{hy1})')
    ax1.legend(fontsize=9, loc='lower right')
    ax1.grid(alpha=0.3)
    ax1.set_ylim(bottom=-0.05, top=max_s2 + 0.15)

    full_tomo = 4 ** config.n_qubits
    ax1.text(0.02, 0.97,
             f'{config.n_shadows} snapshots/depth\n'
             f'Full tomo: 4^{config.n_qubits} ≈ {full_tomo:.1e}\n'
             f'Speedup: ×{full_tomo // max(config.n_shadows, 1):.1e}',
             transform=ax1.transAxes, fontsize=8, va='top',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#E8F5E9',
                       edgecolor='#2E7D32', alpha=0.85))

    # ── Right panel: Cold subsystem (contrast) ──
    if cold_exact is not None:
        ax2.fill_between(cold_exact['times'], 0, cold_exact['s2'],
                         alpha=0.12, color='green')
        ax2.plot(cold_exact['times'], cold_exact['s2'],
                 'g-', linewidth=2.5, label='Exact (statevector ED)')

    ax2.plot(cold_results['times'], cold_results['s2'], 's--',
             color='#E65100', linewidth=2, markersize=7,
             label='Classical shadows (Maestro MPS)')
    ax2.axhline(y=max_s2, color='gray', linestyle=':', alpha=0.5)

    ax2.set_xlabel('Simulation Time t')
    ax2.set_ylabel('2nd-order Rényi Entropy S₂')
    cx0, cy0 = site_coords(cold_qubits[0], config.ly)
    cx1, cy1 = site_coords(cold_qubits[1], config.ly)
    ax2.set_title(f'"Cold" Subsystem (Contrast)\n'
                  f'Qubits {cold_qubits} — site ({cx0},{cy0}),({cx1},{cy1})')
    ax2.legend(fontsize=9, loc='lower right')
    ax2.grid(alpha=0.3)
    ax2.set_ylim(bottom=-0.05, top=max_s2 + 0.15)

    fig.suptitle(f'Entanglement Growth — {config.lx}×{config.ly} TFIM  |  '
                 f'PP Scout → MPS Sniper',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return save_path


# ─────────────────────────────────────────────────────────────────────
# Pauli observable builder
# ─────────────────────────────────────────────────────────────────────

def build_pauli_observable(n_qubits: int, ops: dict) -> str:
    """Build a Pauli string: ops is a dict {qubit_idx: 'X'|'Y'|'Z'}."""
    pauli_chars = []
    for q in range(n_qubits):
        pauli_chars.append(ops.get(q, 'I'))
    return ''.join(pauli_chars)


# ─────────────────────────────────────────────────────────────────────
# Lattice heatmap
# ─────────────────────────────────────────────────────────────────────

def plot_lattice_heatmap(config: Config, scout: dict, save_path: str):
    """
    Plot the 2D TFIM lattice as a heatmap colored by PP scout score.
    Hot bonds are highlighted in gold, cold in blue.
    """
    lx, ly = config.lx, config.ly
    scored = scout['scored_bonds']
    hot_q = scout['hot_qubits']
    cold_q = scout['cold_qubits']

    site_score = np.zeros((lx, ly))
    for (q1, q2), zz, score, nn_sum in scored:
        x1, y1 = site_coords(q1, ly)
        x2, y2 = site_coords(q2, ly)
        site_score[x1, y1] = max(site_score[x1, y1], score)
        site_score[x2, y2] = max(site_score[x2, y2], score)

    fig, ax = plt.subplots(figsize=(max(5, ly * 1.2), max(5, lx * 1.2)))

    im = ax.imshow(site_score, cmap='YlOrRd', aspect='equal',
                   vmin=0, vmax=site_score.max() * 1.1)
    plt.colorbar(im, ax=ax, label='PP Scout Score (coord-weighted |⟨ZZ⟩|)')

    # Draw bonds
    for (q1, q2), zz, score, nn_sum in scored:
        x1, y1 = site_coords(q1, ly)
        x2, y2 = site_coords(q2, ly)
        alpha = 0.3 + 0.7 * (score / (site_score.max() + 1e-8))
        ax.plot([y1, y2], [x1, x2], 'k-', alpha=alpha, linewidth=1.5 * alpha)

    # Annotate sites
    for q in range(config.n_qubits):
        x, y = site_coords(q, ly)
        ax.text(y, x, str(q), ha='center', va='center', fontsize=8,
                fontweight='bold',
                color='white' if site_score[x, y] > site_score.max() * 0.5 else 'black')

    # Highlight hot and cold pairs
    for label, qubits, color, lw in [('HOT', hot_q, 'gold', 4), ('COLD', cold_q, 'deepskyblue', 3)]:
        x1, y1 = site_coords(qubits[0], ly)
        x2, y2 = site_coords(qubits[1], ly)
        ax.plot([y1, y2], [x1, x2], '-', color=color, linewidth=lw, zorder=5,
                label=f'{label}: qubits {qubits}')
        for x, y in [(x1, y1), (x2, y2)]:
            ax.add_patch(plt.Circle((y, x), 0.35, color=color, zorder=6, linewidth=2,
                                    fill=False))

    ax.set_xticks(range(ly))
    ax.set_yticks(range(lx))
    ax.set_xticklabels([f'y={i}' for i in range(ly)])
    ax.set_yticklabels([f'x={i}' for i in range(lx)])
    ax.legend(loc='upper right', fontsize=9)
    ax.set_title(f'PP Scout Heatmap — {lx}×{ly} TFIM Lattice\n'
                 f'Gold = HOT (bulk, high entanglement)  |  Blue = COLD (edge)')

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return save_path
