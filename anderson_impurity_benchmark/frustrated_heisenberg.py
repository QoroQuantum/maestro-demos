"""
Frustrated J₁-J₂ Heisenberg Spin Chain
========================================

Implements the frustrated Heisenberg model on a 1D chain:

    H = J₁ Σᵢ S⃗ᵢ·S⃗ᵢ₊₁  +  J₂ Σᵢ S⃗ᵢ·S⃗ᵢ₊₂

where S⃗ᵢ·S⃗ⱼ = (XᵢXⱼ + YᵢYⱼ + ZᵢZⱼ) / 4.

Nearest-neighbour (J₁) and next-nearest-neighbour (J₂) couplings compete,
producing magnetic frustration. The critical point J₂/J₁ ≈ 0.5 separates
the gapless spin-liquid regime from the dimerized phase and is the hardest
regime for MPS due to strong entanglement.

NNN (J₂) couplings map to non-local 2-qubit gates on the MPS chain, which
stresses the swap-based gate decomposition mechanism.

Provides:
    - VQE-style ansatz circuits for ground-state preparation (Benchmark A)
    - Néel-quench Trotterized circuits for time dynamics (Benchmark B)
    - Pauli observable builders for energy measurement
"""

import numpy as np
from maestro.circuits import QuantumCircuit


class FrustratedHeisenberg:
    """
    Frustrated J₁-J₂ Heisenberg spin chain.

    Args:
        n_qubits: Number of qubits (= number of spin-1/2 sites).
        j1: Nearest-neighbour coupling J₁ (default 1.0).
        j2: Next-nearest-neighbour coupling J₂ (default 0.5 → critical point).
    """

    def __init__(self, n_qubits, j1=1.0, j2=0.5):
        self.n_qubits = n_qubits
        self.j1 = j1
        self.j2 = j2

    # ─────────────────────────────────────────────────────────────────
    # Benchmark A: VQE-style ansatz for ground-state preparation
    # ─────────────────────────────────────────────────────────────────

    def build_vqe_ansatz(self, depth, params=None):
        """
        Build a hardware-efficient VQE ansatz.

        Architecture:
            Layer l = { Ry(θ) on each qubit } + { CX entangling ladder }

        The CX ladder alternates even/odd bond parity each layer to
        ensure full connectivity.

        Args:
            depth: Number of ansatz layers.
            params: 1D array of rotation angles, length = n_qubits * (depth + 1).
                    If None, random angles are generated.

        Returns:
            A QuantumCircuit instance.
        """
        n = self.n_qubits
        n_params = n * (depth + 1)

        if params is None:
            rng = np.random.default_rng(42)
            params = rng.uniform(-np.pi, np.pi, n_params)
        else:
            params = np.asarray(params)
            if params.size != n_params:
                raise ValueError(
                    f"Expected {n_params} parameters, got {params.size}"
                )

        qc = QuantumCircuit()

        # Initial Ry layer
        for q in range(n):
            qc.ry(q, float(params[q]))

        # Entangling layers
        for layer in range(depth):
            offset = n * (layer + 1)

            # CX ladder: even bonds on even layers, odd bonds on odd layers
            start = 0 if layer % 2 == 0 else 1
            for q in range(start, n - 1, 2):
                qc.cx(q, q + 1)

            # Ry rotations
            for q in range(n):
                qc.ry(q, float(params[offset + q]))

        return qc

    # ─────────────────────────────────────────────────────────────────
    # Benchmark B: Néel quench + Trotterized time evolution
    # ─────────────────────────────────────────────────────────────────

    def build_neel_quench_circuit(self, dt, n_steps):
        """
        Build a Trotterized time-evolution circuit after a Néel-state quench.

        Protocol:
            1. Prepare |ψ₀⟩ = |1010...⟩  (Néel state)
            2. Evolve under H = J₁ Σ S⃗ᵢ·S⃗ᵢ₊₁ + J₂ Σ S⃗ᵢ·S⃗ᵢ₊₂

        Each Trotter step decomposes exp(-iHΔt) into:
            - NN (J₁) Heisenberg interactions on (i, i+1) pairs
            - NNN (J₂) Heisenberg interactions on (i, i+2) pairs

        The NNN terms are non-local on the MPS chain and require swap
        operations, which is the key stress test.

        Args:
            dt: Trotter time step.
            n_steps: Number of Trotter steps.

        Returns:
            A QuantumCircuit instance.
        """
        n = self.n_qubits
        qc = QuantumCircuit()

        # ── State Preparation: Néel state |1010...⟩ ──
        for q in range(0, n, 2):
            qc.x(q)

        # ── Trotter Steps ──
        for _ in range(n_steps):
            # --- J₁: Nearest-neighbour Heisenberg (even bonds, then odd) ---
            for q in range(0, n - 1, 2):
                self._add_heisenberg_gate(qc, q, q + 1, self.j1 * dt)
            for q in range(1, n - 1, 2):
                self._add_heisenberg_gate(qc, q, q + 1, self.j1 * dt)

            # --- J₂: Next-nearest-neighbour Heisenberg (even, then odd) ---
            for q in range(0, n - 2, 2):
                self._add_heisenberg_gate(qc, q, q + 2, self.j2 * dt)
            for q in range(1, n - 2, 2):
                self._add_heisenberg_gate(qc, q, q + 2, self.j2 * dt)

        return qc

    # ─────────────────────────────────────────────────────────────────
    # Observable builders
    # ─────────────────────────────────────────────────────────────────

    def build_energy_observables(self):
        """
        Build Pauli observable strings for measuring the Hamiltonian energy.

        H = J₁ Σᵢ (XᵢXᵢ₊₁ + YᵢYᵢ₊₁ + ZᵢZᵢ₊₂) / 4
          + J₂ Σᵢ (XᵢXᵢ₊₂ + YᵢYᵢ₊₂ + ZᵢZᵢ₊₂) / 4

        Returns:
            (observables, coefficients) — lists of Pauli strings and their
            coefficients such that E = Σ cᵢ ⟨Oᵢ⟩.
        """
        n = self.n_qubits
        observables = []
        coefficients = []

        for pauli in ['X', 'Y', 'Z']:
            # NN terms (J₁)
            for i in range(n - 1):
                obs = ['I'] * n
                obs[i] = pauli
                obs[i + 1] = pauli
                observables.append(''.join(obs))
                coefficients.append(self.j1 / 4.0)

            # NNN terms (J₂)
            for i in range(n - 2):
                obs = ['I'] * n
                obs[i] = pauli
                obs[i + 2] = pauli
                observables.append(''.join(obs))
                coefficients.append(self.j2 / 4.0)

        return observables, coefficients

    def build_z_observables(self):
        """Build per-qubit Z observables for magnetization tracking."""
        n = self.n_qubits
        obs = []
        for i in range(n):
            pauli = ['I'] * n
            pauli[i] = 'Z'
            obs.append(''.join(pauli))
        return obs

    # ─────────────────────────────────────────────────────────────────
    # Gate decompositions
    # ─────────────────────────────────────────────────────────────────

    def _add_heisenberg_gate(self, qc, q1, q2, theta):
        """
        Implement exp(-i θ (XX + YY + ZZ) / 4) — isotropic Heisenberg gate.

        Decomposed into three commuting terms:
            exp(-iθ XX/4):  H-H → CX → Rz(θ/2) → CX → H-H
            exp(-iθ YY/4):  Sdg-Sdg → H-H → CX → Rz(θ/2) → CX → H-H → S-S
            exp(-iθ ZZ/4):  CX → Rz(θ/2) → CX

        Args:
            qc: QuantumCircuit to append to.
            q1, q2: Qubit indices.
            theta: Coupling angle (= J * dt).
        """
        angle = theta / 2.0

        # exp(-iθ XX/4)
        qc.h(q1); qc.h(q2)
        qc.cx(q1, q2)
        qc.rz(q2, angle)
        qc.cx(q1, q2)
        qc.h(q1); qc.h(q2)

        # exp(-iθ YY/4)
        qc.sdg(q1); qc.sdg(q2)
        qc.h(q1); qc.h(q2)
        qc.cx(q1, q2)
        qc.rz(q2, angle)
        qc.cx(q1, q2)
        qc.h(q1); qc.h(q2)
        qc.s(q1); qc.s(q2)

        # exp(-iθ ZZ/4)
        qc.cx(q1, q2)
        qc.rz(q2, angle)
        qc.cx(q1, q2)
