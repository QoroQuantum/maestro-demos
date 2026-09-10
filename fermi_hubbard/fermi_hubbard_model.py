"""
1D Fermi-Hubbard Model with Jordan-Wigner Mapping
===================================================

Provides the FermiHubbardModel class which builds Trotterized time-evolution
circuits for the 1D Fermi-Hubbard Hamiltonian using Maestro's QuantumCircuit.

Qubit layout:
    Qubits [0, N)     → spin-up electrons
    Qubits [N, 2N)    → spin-down electrons

Hamiltonian:
    H = -t Σ (c†_{i,σ} c_{i+1,σ} + h.c.) + U Σ n_{i,↑} n_{i,↓}

Under Jordan-Wigner (nearest-neighbor, no JW strings needed):
    Hopping:     exp(-i dt t (XX + YY) / 2)
    Interaction: exp(-i dt U/4 (I - Z_↑ - Z_↓ + Z_↑Z_↓))
"""

from maestro.circuits import QuantumCircuit


class FermiHubbardModel:
    """
    1D Fermi-Hubbard model with Jordan-Wigner mapping.

    Provides methods to build:
    - Full Trotterized time-evolution circuits (for MPS simulation)
    - Clifford-only proxy circuits (for Pauli Propagator scouting)
    """

    def __init__(self, n_sites, t=1.0, u=1.0):
        """
        Args:
            n_sites: Number of lattice sites.
            t: Hopping energy (kinetic term).
            u: On-site interaction strength.
        """
        self.n_sites = n_sites
        self.t = t
        self.u = u

    def build_circuit(self, steps, dt, init_wall_idx, active_sites_range=None):
        """
        Build a Trotterized time-evolution circuit.

        Args:
            steps: Number of Trotter steps.
            dt: Time per step.
            init_wall_idx: Global site index of the domain wall.
            active_sites_range: Optional (start, end) for subsystem simulation.
                When set, only sites in [start, end) are simulated.

        Returns:
            A QuantumCircuit instance.
        """
        circuit = QuantumCircuit()

        # ── State Preparation: Domain Wall ──
        if active_sites_range:
            start_site, end_site = active_sites_range
            n_active = end_site - start_site
            for local_i in range(n_active):
                if local_i + start_site < init_wall_idx:
                    circuit.x(local_i)               # Spin-up
                    circuit.x(local_i + n_active)     # Spin-down
        else:
            for i in range(init_wall_idx):
                circuit.x(i)                          # Spin-up
                circuit.x(i + self.n_sites)            # Spin-down

        # ── Trotter Steps ──
        n_active = (self.n_sites if not active_sites_range
                    else active_sites_range[1] - active_sites_range[0])
        up_offset = 0
        down_offset = n_active

        for _ in range(steps):
            # Even bonds
            for i in range(0, n_active - 1, 2):
                self._add_hopping(circuit, up_offset + i, up_offset + i + 1, dt)
                self._add_hopping(circuit, down_offset + i, down_offset + i + 1, dt)

            # Odd bonds
            for i in range(1, n_active - 1, 2):
                self._add_hopping(circuit, up_offset + i, up_offset + i + 1, dt)
                self._add_hopping(circuit, down_offset + i, down_offset + i + 1, dt)

            # On-site interaction
            for i in range(n_active):
                self._add_interaction(circuit, up_offset + i, down_offset + i, dt)

        return circuit

    def build_clifford_scout_circuit(self, steps, init_wall_idx):
        """
        Build a Clifford-only circuit for light-cone detection.

        Uses only Clifford gates (H, CX, CZ) which are compatible with the
        Pauli Propagator backend. Gates are applied within an expanding window
        around the domain wall, mimicking the Lieb-Robinson causal light cone.

        This avoids saturating the entire chain (Clifford gates are "full
        strength" unlike small-angle Rz gates).

        Args:
            steps: Number of Clifford proxy steps.
            init_wall_idx: Site index of the domain wall.

        Returns:
            A QuantumCircuit instance.
        """
        circuit = QuantumCircuit()

        # ── State Preparation: Domain Wall ──
        for i in range(init_wall_idx):
            circuit.x(i)
            circuit.x(i + self.n_sites)

        # ── Clifford Proxy Steps with Expanding Light Cone ──
        up_offset = 0
        down_offset = self.n_sites

        for step in range(steps):
            lc_start = max(0, init_wall_idx - step - 1)
            lc_end = min(self.n_sites, init_wall_idx + step + 2)

            # Even hopping bonds within window
            for i in range(max(0, lc_start), lc_end - 1, 2):
                circuit.h(up_offset + i)
                circuit.cx(up_offset + i, up_offset + i + 1)
                circuit.h(up_offset + i)
                circuit.h(down_offset + i)
                circuit.cx(down_offset + i, down_offset + i + 1)
                circuit.h(down_offset + i)

            # Odd hopping bonds within window
            for i in range(max(1, lc_start | 1), lc_end - 1, 2):
                circuit.cx(up_offset + i, up_offset + i + 1)
                circuit.cx(down_offset + i, down_offset + i + 1)

            # On-site interaction within window
            for i in range(lc_start, lc_end):
                circuit.cz(up_offset + i, down_offset + i)

        return circuit

    def _add_hopping(self, qc, q1, q2, dt):
        """
        Implement exp(-i θ (XX + YY) / 2) where θ = t * dt.

        Decomposed as:
            exp(-iθ XX/2): H-CX-Rz-CX-H
            exp(-iθ YY/2): S†-H-CX-Rz-CX-H-S
        """
        theta = self.t * dt

        # exp(-iθ XX/2)
        qc.h(q1); qc.h(q2)
        qc.cx(q1, q2); qc.rz(q2, theta); qc.cx(q1, q2)
        qc.h(q1); qc.h(q2)

        # exp(-iθ YY/2)
        qc.sdg(q1); qc.sdg(q2)
        qc.h(q1); qc.h(q2)
        qc.cx(q1, q2); qc.rz(q2, theta); qc.cx(q1, q2)
        qc.h(q1); qc.h(q2)
        qc.s(q1); qc.s(q2)

    def _add_interaction(self, qc, q_up, q_down, dt):
        """
        Implement exp(-i dt U n_↑ n_↓) where n = (I-Z)/2.

        Expanding: n_↑ n_↓ = (I - Z_↑ - Z_↓ + Z_↑Z_↓) / 4

        Decomposed as:
            exp(-iα) · exp(+iα Z_↑) · exp(+iα Z_↓) · exp(-iα Z_↑Z_↓)
            with α = U·dt/4
        """
        angle = self.u * dt / 4.0

        # Single-qubit Z rotations: exp(+iα Z) = Rz(-2α)
        qc.rz(q_up, -2.0 * angle)
        qc.rz(q_down, -2.0 * angle)

        # ZZ interaction: exp(-iα ZZ) via CX-Rz-CX
        qc.cx(q_up, q_down)
        qc.rz(q_down, 2.0 * angle)
        qc.cx(q_up, q_down)
