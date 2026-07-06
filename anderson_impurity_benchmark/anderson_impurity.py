"""
Single Impurity Anderson Model (SIAM)
======================================

Implements a simplified Anderson impurity model mapped to qubits via the
Jordan-Wigner transformation:

    H = ε_d Σ_σ n_{d,σ}  +  U n_{d,↑} n_{d,↓}
      + Σ_{k,σ} ε_k n_{k,σ}
      + V Σ_{k,σ} (c†_{d,σ} c_{k,σ} + h.c.)

Qubit layout (Jordan-Wigner):
    [imp↑, imp↓, bath₁↑, bath₁↓, bath₂↑, bath₂↓, ...]

Key mappings:
    - Hubbard U term → ZZ interaction on impurity qubits (0, 1)
    - Hybridization V → XX+YY hopping between impurity and bath
    - On-site energies ε → Rz single-qubit rotations

Provides:
    - VQE-style ansatz circuits for ground-state preparation (Benchmark A)
    - Quench circuits for time dynamics (Benchmark B)
    - Pauli observable builders for energy measurement
"""

import numpy as np
from maestro.circuits import QuantumCircuit


class AndersonImpurity:
    """
    Single Impurity Anderson Model (SIAM).

    Args:
        n_bath_sites: Number of bath sites (total qubits = 2 + 2 * n_bath_sites).
        u: Hubbard U (on-site Coulomb repulsion on impurity).
        v: Hybridization strength between impurity and bath.
        eps_d: Impurity energy level.
        eps_bath: Bath energy levels (array of length n_bath_sites, or scalar
                  for uniform spacing).
    """

    def __init__(self, n_bath_sites, u=4.0, v=1.0, eps_d=-2.0, eps_bath=None):
        self.n_bath_sites = n_bath_sites
        self.n_qubits = 2 + 2 * n_bath_sites  # imp↑, imp↓, bath...
        self.u = u
        self.v = v
        self.eps_d = eps_d

        if eps_bath is None:
            # Uniform bath: linearly spaced around zero
            if n_bath_sites == 1:
                self.eps_bath = np.array([0.0])
            else:
                self.eps_bath = np.linspace(-2.0, 2.0, n_bath_sites)
        else:
            self.eps_bath = np.atleast_1d(np.asarray(eps_bath, dtype=float))

    @property
    def imp_up(self):
        """Qubit index for impurity spin-up."""
        return 0

    @property
    def imp_down(self):
        """Qubit index for impurity spin-down."""
        return 1

    def bath_up(self, k):
        """Qubit index for bath site k, spin-up."""
        return 2 + 2 * k

    def bath_down(self, k):
        """Qubit index for bath site k, spin-down."""
        return 2 + 2 * k + 1

    # ─────────────────────────────────────────────────────────────────
    # Benchmark A: VQE-style ansatz for ground-state preparation
    # ─────────────────────────────────────────────────────────────────

    def build_vqe_ansatz(self, depth, params=None):
        """
        Build a hardware-efficient VQE ansatz tailored to the SIAM.

        Architecture per layer:
            1. Ry(θ) on each qubit
            2. CX between impurity and each bath site (spin-preserving)
            3. CX ladder within bath qubits

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

        for layer in range(depth):
            offset = n * (layer + 1)

            # Impurity-bath entangling: CX from imp to each bath site
            for k in range(self.n_bath_sites):
                qc.cx(self.imp_up, self.bath_up(k))
                qc.cx(self.imp_down, self.bath_down(k))

            # Intra-bath CX ladder (alternating parity)
            start = 0 if layer % 2 == 0 else 1
            for k in range(start, self.n_bath_sites - 1, 2):
                qc.cx(self.bath_up(k), self.bath_up(k + 1))
                qc.cx(self.bath_down(k), self.bath_down(k + 1))

            # Ry rotations
            for q in range(n):
                qc.ry(q, float(params[offset + q]))

        return qc

    # ─────────────────────────────────────────────────────────────────
    # Benchmark B: Quench + Trotterized time evolution
    # ─────────────────────────────────────────────────────────────────

    def build_quench_circuit(self, dt, n_steps, init_state='half_filled'):
        """
        Build a Trotterized time-evolution circuit.

        Protocol:
            1. Prepare initial state (half-filled impurity by default)
            2. Evolve under the full SIAM Hamiltonian

        Each Trotter step:
            a. On-site energies: Rz rotations
            b. Hubbard U: ZZ interaction on impurity
            c. Hybridization V: XX+YY hopping (impurity ↔ bath)

        Args:
            dt: Trotter time step.
            n_steps: Number of Trotter steps.
            init_state: 'half_filled' (impurity occupied, bath empty) or
                        'neel' (alternating occupation).

        Returns:
            A QuantumCircuit instance.
        """
        n = self.n_qubits
        qc = QuantumCircuit()

        # ── State Preparation ──
        if init_state == 'half_filled':
            # Impurity doubly occupied, bath empty
            qc.x(self.imp_up)
            qc.x(self.imp_down)
        elif init_state == 'neel':
            # Alternating: imp↑ occupied, imp↓ empty, bath1↑ occupied, etc.
            for q in range(0, n, 2):
                qc.x(q)
        else:
            raise ValueError(f"Unknown init_state: {init_state}")

        # ── Trotter Steps ──
        for _ in range(n_steps):
            # a. On-site energies: exp(-i ε_d dt n_{d,σ})
            #    n = (I - Z) / 2  →  exp(-i ε dt (I-Z)/2) = phase * Rz(ε dt)
            qc.rz(self.imp_up, self.eps_d * dt)
            qc.rz(self.imp_down, self.eps_d * dt)

            for k in range(self.n_bath_sites):
                qc.rz(self.bath_up(k), self.eps_bath[k] * dt)
                qc.rz(self.bath_down(k), self.eps_bath[k] * dt)

            # b. Hubbard U: exp(-i U dt n_{d,↑} n_{d,↓})
            #    n_↑ n_↓ = (I - Z_↑ - Z_↓ + Z_↑Z_↓) / 4
            self._add_hubbard_u(qc, dt)

            # c. Hybridization V: exp(-i V dt (XX + YY) / 2) for each bath site
            for k in range(self.n_bath_sites):
                self._add_hopping(qc, self.imp_up, self.bath_up(k), dt)
                self._add_hopping(qc, self.imp_down, self.bath_down(k), dt)

        return qc

    # ─────────────────────────────────────────────────────────────────
    # Observable builders
    # ─────────────────────────────────────────────────────────────────

    def build_energy_observables(self):
        """
        Build Pauli observable strings for measuring the SIAM energy.

        Returns:
            (observables, coefficients) — lists of Pauli strings and their
            coefficients such that E = Σ cᵢ ⟨Oᵢ⟩.
        """
        n = self.n_qubits
        observables = []
        coefficients = []

        def _pauli_string(pauli_map):
            labels = ['I'] * n
            for qubit, pauli in pauli_map.items():
                labels[qubit] = pauli
            return ''.join(labels)

        # On-site impurity energy: ε_d (I - Z) / 2 per spin
        for q in [self.imp_up, self.imp_down]:
            observables.append(_pauli_string({q: 'Z'}))
            coefficients.append(-self.eps_d / 2.0)

        # Bath energies: ε_k (I - Z) / 2 per spin
        for k in range(self.n_bath_sites):
            for q in [self.bath_up(k), self.bath_down(k)]:
                observables.append(_pauli_string({q: 'Z'}))
                coefficients.append(-self.eps_bath[k] / 2.0)

        # Hubbard U: U * n_↑ n_↓ = U/4 * (I - Z_↑ - Z_↓ + Z_↑Z_↓)
        # Z_↑ and Z_↓ terms
        observables.append(_pauli_string({self.imp_up: 'Z'}))
        coefficients.append(-self.u / 4.0)
        observables.append(_pauli_string({self.imp_down: 'Z'}))
        coefficients.append(-self.u / 4.0)
        # Z_↑ Z_↓ term
        observables.append(_pauli_string(
            {self.imp_up: 'Z', self.imp_down: 'Z'}
        ))
        coefficients.append(self.u / 4.0)

        # Hybridization V: V/2 * (XX + YY) for each imp-bath pair
        for k in range(self.n_bath_sites):
            for pauli in ['X', 'Y']:
                # Spin-up
                observables.append(_pauli_string(
                    {self.imp_up: pauli, self.bath_up(k): pauli}
                ))
                coefficients.append(self.v / 2.0)
                # Spin-down
                observables.append(_pauli_string(
                    {self.imp_down: pauli, self.bath_down(k): pauli}
                ))
                coefficients.append(self.v / 2.0)

        return observables, coefficients

    def build_impurity_observables(self):
        """Build observables for impurity occupation ⟨n_{d,σ}⟩ = (1−⟨Z⟩)/2."""
        n = self.n_qubits
        obs = []
        for q in [self.imp_up, self.imp_down]:
            pauli = ['I'] * n
            pauli[q] = 'Z'
            obs.append(''.join(pauli))
        return obs

    # ─────────────────────────────────────────────────────────────────
    # Gate decompositions
    # ─────────────────────────────────────────────────────────────────

    def _add_hubbard_u(self, qc, dt):
        """
        Implement exp(-i U dt n_↑ n_↓) on impurity qubits.

        n_↑ n_↓ = (I - Z_↑ - Z_↓ + Z_↑Z_↓) / 4
        → exp(-iα) · exp(+iα Z_↑) · exp(+iα Z_↓) · exp(-iα Z_↑Z_↓)
          with α = U·dt/4
        """
        alpha = self.u * dt / 4.0
        q_up = self.imp_up
        q_down = self.imp_down

        # Single-qubit Z rotations: exp(+iα Z) = Rz(-2α)
        qc.rz(q_up, -2.0 * alpha)
        qc.rz(q_down, -2.0 * alpha)

        # ZZ interaction: exp(-iα ZZ) via CX-Rz-CX
        qc.cx(q_up, q_down)
        qc.rz(q_down, 2.0 * alpha)
        qc.cx(q_up, q_down)

    def _add_hopping(self, qc, q1, q2, dt):
        """
        Implement exp(-i V dt (XX + YY) / 2) — hopping term.

        Decomposed as:
            exp(-iθ XX/2): H-H → CX → Rz(θ) → CX → H-H
            exp(-iθ YY/2): Sdg-Sdg → H-H → CX → Rz(θ) → CX → H-H → S-S

        where θ = V · dt.
        """
        theta = self.v * dt

        # exp(-iθ XX/2)
        qc.h(q1); qc.h(q2)
        qc.cx(q1, q2)
        qc.rz(q2, theta)
        qc.cx(q1, q2)
        qc.h(q1); qc.h(q2)

        # exp(-iθ YY/2)
        qc.sdg(q1); qc.sdg(q2)
        qc.h(q1); qc.h(q2)
        qc.cx(q1, q2)
        qc.rz(q2, theta)
        qc.cx(q1, q2)
        qc.h(q1); qc.h(q2)
        qc.s(q1); qc.s(q2)
