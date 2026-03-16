import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import PauliEvolutionGate


class QITEPropagation:

    def __init__(self, hamiltonian, approx, backend):

        self.H = hamiltonian
        self.approx = approx
        self.backend = backend

        self.num_qubits = hamiltonian.num_qubits

    def step(self, state, dt):

        S = self.approx.compute_S(state)
        b = self.approx.compute_b(state, self.H)

        # Solve S a = dt b
        a = self.approx.solve(S, b, dt)

        U_circuit = self.build_unitary(a)

        new_state = state.evolve(U_circuit)

        return new_state

    def build_unitary(self, coeffs):

        paulis = []
        weights = []

        for c, P in zip(coeffs, self.approx.pauli_basis):

            if abs(c) < 1e-12:
                continue

            # assumes P is a single Pauli operator
            paulis.append(P.paulis[0])
            weights.append(c)

        op = SparsePauliOp(paulis, weights)

        evo = PauliEvolutionGate(op, time=1.0)

        qc = QuantumCircuit(self.num_qubits)
        qc.append(evo, range(self.num_qubits))

        return qc