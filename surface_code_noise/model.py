"""
Rotated Surface Code — Circuit Construction & Noise Models
============================================================

Provides the SurfaceCodeModel class which builds noisy syndrome extraction
circuits for the rotated surface code, suitable for MPS simulation with
Maestro.

Qubit layout (rotated surface code, distance d):
    Data qubits:     d² qubits on the vertices of the rotated lattice
    Ancilla qubits:  (d²-1) qubits at plaquette centres (X and Z stabilisers)
    Total:           2d² - 1 qubits

Noise models:
    1. Pauli noise   — random Pauli rotations after each CX (incoherent)
    2. Coherent noise — systematic unitary over-rotations after each CX

The key insight: coherent noise builds up constructively across gates,
producing error correlations that stabiliser simulators (Stim) cannot
capture. MPS simulation faithfully tracks these correlations via the
entanglement structure.
"""

import math
import numpy as np
from maestro.circuits import QuantumCircuit


class SurfaceCodeModel:
    """
    Rotated surface code model with configurable noise.

    Builds syndrome extraction circuits and provides methods to inject
    Pauli (incoherent) or coherent noise for comparison.
    """

    def __init__(self, distance):
        """
        Args:
            distance: Code distance d (odd integer >= 3).
        """
        if distance < 3 or distance % 2 == 0:
            raise ValueError(f"Distance must be odd and >= 3, got {distance}")

        self.d = distance
        self.n_data = distance ** 2
        self.n_ancilla = distance ** 2 - 1

        # Qubit index allocation:
        #   [0, n_data)          → data qubits
        #   [n_data, n_total)    → ancilla qubits
        self.n_total = self.n_data + self.n_ancilla

        # Build the stabiliser layout
        self._build_layout()

    def _build_layout(self):
        """
        Build the qubit layout for the rotated surface code.

        In the rotated layout, data qubits sit on a d×d grid.
        Stabiliser plaquettes alternate between X-type and Z-type in
        a checkerboard pattern. The boundary conditions determine
        which plaquettes are partial (weight-2) vs full (weight-4).
        """
        d = self.d

        # Data qubits on a d×d grid: data qubit at (row, col) → index
        self.data_map = {}
        for r in range(d):
            for c in range(d):
                self.data_map[(r, c)] = r * d + c

        # Build stabiliser plaquettes
        # Each plaquette is centred between four data qubits
        # X-stabilisers and Z-stabilisers alternate in a checkerboard
        self.x_stabilisers = []  # list of (ancilla_idx, [data qubit indices])
        self.z_stabilisers = []

        ancilla_idx = self.n_data  # Start ancilla indices after data qubits

        # Plaquettes are centred at half-integer coordinates (r+0.5, c+0.5)
        for r in range(d - 1):
            for c in range(d - 1):
                # Four corners of this plaquette
                corners = [
                    (r, c), (r, c + 1),
                    (r + 1, c), (r + 1, c + 1),
                ]
                data_qubits = [self.data_map[pos] for pos in corners]

                # Checkerboard: (r+c) even → X-type, odd → Z-type
                if (r + c) % 2 == 0:
                    self.x_stabilisers.append((ancilla_idx, data_qubits))
                else:
                    self.z_stabilisers.append((ancilla_idx, data_qubits))
                ancilla_idx += 1

        # Boundary stabilisers (weight-2)
        # Top and bottom boundaries → Z-type
        for c in range(0, d - 1, 2):
            corners = [(0, c), (0, c + 1)]
            data_qubits = [self.data_map[pos] for pos in corners]
            self.z_stabilisers.append((ancilla_idx, data_qubits))
            ancilla_idx += 1

        for c in range(1 - (d % 2), d - 1, 2):
            corners = [(d - 1, c), (d - 1, c + 1)]
            data_qubits = [self.data_map[pos] for pos in corners]
            self.z_stabilisers.append((ancilla_idx, data_qubits))
            ancilla_idx += 1

        # Left and right boundaries → X-type
        for r in range(1, d - 1, 2):
            corners = [(r, 0), (r + 1, 0)]
            data_qubits = [self.data_map[pos] for pos in corners]
            self.x_stabilisers.append((ancilla_idx, data_qubits))
            ancilla_idx += 1

        for r in range(0 + (d % 2), d - 1, 2):
            corners = [(r, d - 1), (r + 1, d - 1)]
            data_qubits = [self.data_map[pos] for pos in corners]
            self.x_stabilisers.append((ancilla_idx, data_qubits))
            ancilla_idx += 1

        # Logical operators (for checking logical errors)
        # Logical X: row of X operators across the middle row
        self.logical_x_qubits = [self.data_map[(self.d // 2, c)]
                                  for c in range(self.d)]
        # Logical Z: column of Z operators down the middle column
        self.logical_z_qubits = [self.data_map[(r, self.d // 2)]
                                  for r in range(self.d)]

    def build_syndrome_circuit(self, n_rounds, noise_type='none',
                                noise_strength=0.0, seed=None):
        """
        Build a full syndrome extraction circuit.

        Args:
            n_rounds: Number of syndrome extraction rounds.
            noise_type: 'none', 'pauli', or 'coherent'.
            noise_strength: Error probability p (pauli) or rotation
                angle ε (coherent).
            seed: Random seed for Pauli noise sampling.

        Returns:
            A QuantumCircuit instance with measurements on ancilla qubits.
        """
        qc = QuantumCircuit()
        rng = np.random.default_rng(seed) if noise_type == 'pauli' else None

        for round_idx in range(n_rounds):
            # Reset ancilla qubits (prepare in |0⟩)
            # For round > 0, ancillae were measured; re-initialize them
            if round_idx > 0:
                for stab_list in [self.x_stabilisers, self.z_stabilisers]:
                    for anc, _ in stab_list:
                        # Reset: measure + conditional X (simulated as
                        # re-preparation). For circuit simulation, we just
                        # re-prepare fresh ancillae by identity (they start
                        # in |0⟩ after measurement).
                        pass

            # Prepare X-ancillae in |+⟩ state
            for anc, _ in self.x_stabilisers:
                qc.h(anc)

            # CX gates: standard ordering for rotated surface code
            # Step 1-4: entangle ancilla with data qubits
            for step in range(4):
                # X-stabilisers: CX with ancilla as control
                for anc, data_qubits in self.x_stabilisers:
                    if step < len(data_qubits):
                        dq = data_qubits[step]
                        qc.cx(anc, dq)
                        self._inject_noise(qc, anc, dq, noise_type,
                                           noise_strength, rng)

                # Z-stabilisers: CX with ancilla as target
                for anc, data_qubits in self.z_stabilisers:
                    if step < len(data_qubits):
                        dq = data_qubits[step]
                        qc.cx(dq, anc)
                        self._inject_noise(qc, dq, anc, noise_type,
                                           noise_strength, rng)

            # Rotate X-ancillae back to Z-basis for measurement
            for anc, _ in self.x_stabilisers:
                qc.h(anc)

        # Measure all ancilla qubits
        for stab_list in [self.x_stabilisers, self.z_stabilisers]:
            for anc, _ in stab_list:
                qc.measure(anc)

        return qc

    def build_noisy_idle_circuit(self, noise_type, noise_strength, seed=None):
        """
        Build a circuit that applies noise to all data qubits and then
        performs a single round of syndrome extraction.

        This isolates the effect of noise on the syndrome pattern.

        Args:
            noise_type: 'pauli' or 'coherent'.
            noise_strength: Error probability p or rotation ε.
            seed: Random seed for Pauli noise.

        Returns:
            A QuantumCircuit instance.
        """
        return self.build_syndrome_circuit(
            n_rounds=1, noise_type=noise_type,
            noise_strength=noise_strength, seed=seed,
        )

    def build_noisy_cx_network(self, n_rounds, noise_type='none',
                                noise_strength=0.0, seed=None):
        """
        Build a CX noise network using ONLY data qubits (d² qubits).

        Applies CX gates in the surface code's nearest-neighbour
        connectivity, then injects noise ONCE per qubit per round.
        The CX topology determines how errors correlate across qubits;
        the noise injection controls the per-qubit error magnitude.

        Physics:
            - All qubits start in |0⟩, so ⟨Z⟩ = +1 initially.
            - CX between |0⟩ qubits is identity — any ⟨Z⟩ deviation
              is purely from injected noise + CX error propagation.
            - Pauli noise: random rotations partially cancel across
              samples (incoherent, what Stim models).
            - Coherent noise: systematic rotations accumulate
              constructively round after round (what Stim misses).

        Args:
            n_rounds: Number of CX rounds.
            noise_type: 'none', 'pauli', or 'coherent'.
            noise_strength: Per-qubit rotation angle ε per round.
            seed: Random seed for Pauli noise.

        Returns:
            A QuantumCircuit on d² qubits (data only).
        """
        qc = QuantumCircuit()
        rng = np.random.default_rng(seed) if noise_type == 'pauli' else None

        # Build nearest-neighbour CX pairs from the d×d grid
        cx_pairs = self._get_nn_cx_pairs()

        for _ in range(n_rounds):
            # Apply CX layer (no noise per gate)
            for q1, q2 in cx_pairs[::2]:
                qc.cx(q1, q2)
            for q1, q2 in cx_pairs[1::2]:
                qc.cx(q1, q2)

            # Inject noise ONCE per qubit per round
            if noise_type != 'none' and noise_strength > 0:
                for q in range(self.n_data):
                    self._inject_qubit_noise(
                        qc, q, noise_type, noise_strength, rng)

        return qc

    def _get_nn_cx_pairs(self):
        """Get nearest-neighbour CX pairs from the d×d grid."""
        d = self.d
        pairs = []
        # Horizontal bonds
        for r in range(d):
            for c in range(d - 1):
                pairs.append((r * d + c, r * d + c + 1))
        # Vertical bonds
        for r in range(d - 1):
            for c in range(d):
                pairs.append((r * d + c, (r + 1) * d + c))
        return pairs

    def build_clifford_cx_network(self, n_rounds, noise_strength=0.0,
                                    seed=None):
        """
        Build a Clifford-compatible CX noise network (data qubits only).

        Uses probabilistic Pauli gate insertion (X, Y, Z with probability
        p/3 each) to approximate depolarising noise. Compatible with
        Stabilizer and PauliPropagator simulation backends.

        This is what Stim does — and Maestro can do it too, in the same
        tool that also does coherent noise MPS simulation.

        Args:
            n_rounds: Number of CX rounds.
            noise_strength: Depolarising probability p per qubit per round.
            seed: Random seed.

        Returns:
            A QuantumCircuit on d² qubits (Clifford gates only).
        """
        qc = QuantumCircuit()
        rng = np.random.default_rng(seed)

        cx_pairs = self._get_nn_cx_pairs()

        for _ in range(n_rounds):
            # CX layer
            for q1, q2 in cx_pairs[::2]:
                qc.cx(q1, q2)
            for q1, q2 in cx_pairs[1::2]:
                qc.cx(q1, q2)

            # Clifford noise: once per qubit per round
            if noise_strength > 0:
                for q in range(self.n_data):
                    self._inject_clifford_noise(qc, q, noise_strength, rng)

        return qc

    def _inject_clifford_noise(self, qc, qubit, p, rng):
        """Inject depolarising noise as probabilistic Pauli gates."""
        if p <= 0:
            return
        r = rng.random()
        if r < p / 3:
            qc.x(qubit)
        elif r < 2 * p / 3:
            qc.z(qubit)
        elif r < p:
            qc.x(qubit)
            qc.z(qubit)  # Y = iXZ (global phase irrelevant)

    def _inject_qubit_noise(self, qc, qubit, noise_type, strength, rng):
        """
        Inject per-qubit noise (one application per round).

        Pauli noise:
            Random Rz and Rx rotations with angles drawn from a Gaussian
            with standard deviation proportional to sqrt(p).
            This approximates a depolarising channel.

        Coherent noise:
            Systematic Rz(ε) + Rx(ε) over-rotation.
            These accumulate coherently round after round — the key
            effect that Stim's Pauli noise model misses entirely.
        """
        if noise_type == 'none' or strength <= 0:
            return

        if noise_type == 'pauli':
            sigma = math.sqrt(strength)
            qc.rz(qubit, rng.normal(0, sigma))
            qc.rx(qubit, rng.normal(0, sigma))

        elif noise_type == 'coherent':
            qc.rz(qubit, strength)
            qc.rx(qubit, strength)

    def build_ancilla_observables(self):
        """Build Z observables for all ancilla qubits (syndrome readout)."""
        obs = []
        for stab_list in [self.x_stabilisers, self.z_stabilisers]:
            for anc, _ in stab_list:
                pauli = ['I'] * self.n_total
                pauli[anc] = 'Z'
                obs.append(''.join(pauli))
        return obs

    def build_data_z_observables(self):
        """Build per-data-qubit Z observables."""
        obs = []
        for i in range(self.n_data):
            pauli = ['I'] * self.n_total
            pauli[i] = 'Z'
            obs.append(''.join(pauli))
        return obs

    def build_logical_z_observable(self):
        """Build the logical Z observable (column of Z operators)."""
        pauli = ['I'] * self.n_total
        for q in self.logical_z_qubits:
            pauli[q] = 'Z'
        return ''.join(pauli)

    def build_stabiliser_observables(self):
        """
        Build the full stabiliser observables (multi-qubit Pauli strings).

        Returns list of (type, observable_string) tuples.
        """
        obs = []
        for anc, data_qubits in self.x_stabilisers:
            pauli = ['I'] * self.n_total
            for dq in data_qubits:
                pauli[dq] = 'X'
            obs.append(('X', ''.join(pauli)))

        for anc, data_qubits in self.z_stabilisers:
            pauli = ['I'] * self.n_total
            for dq in data_qubits:
                pauli[dq] = 'Z'
            obs.append(('Z', ''.join(pauli)))

        return obs

    @property
    def summary(self):
        """Human-readable summary of the code layout."""
        return (
            f"Rotated surface code d={self.d}: "
            f"{self.n_data} data + {self.n_ancilla} ancilla = "
            f"{self.n_total} qubits, "
            f"{len(self.x_stabilisers)} X-stabs, "
            f"{len(self.z_stabilisers)} Z-stabs"
        )
