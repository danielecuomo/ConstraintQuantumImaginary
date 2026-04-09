import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import PauliEvolutionGate


class QITEPropagation:

    def __init__(self, hamiltonian, approx, backend):

        self.H = hamiltonian
        self.approx = approx
        self.backend = backend
        
        self.constraints = []
        self.targets = []

        self.num_qubits = hamiltonian.num_qubits

    def step(self, state, dt):
    
        G = self.build_constrained_generator(state)
    
        S = self.approx.compute_S(state)
        b = self.approx.compute_b(state, G)
    
        a = self.approx.solve(S, b, dt)
    
        U_circuit = self.build_unitary(a)
    
        return state.evolve(U_circuit)

    def build_unitary(self, coeffs):

        paulis = []
        weights = []

        for c, P in zip(coeffs, self.approx.pauli_basis):

            if abs(c) < 1e-12:
                continue
        
            for pauli, coeff in zip(P.paulis, P.coeffs):
                paulis.append(pauli)
                weights.append(c * coeff)

        op = SparsePauliOp(paulis, weights)

        evo = PauliEvolutionGate(op, time=1.0)

        qc = QuantumCircuit(self.num_qubits)
        qc.append(evo, range(self.num_qubits))

        return qc

    def build_constrained_generator(self, state):
    
        if not self.constraints:
            return self.H
    
        beta = self.compute_betas(state)
    
        G = self.H
    
        for b_i, B_i in zip(beta, self.constraints):
            G = G + B_i.scale(b_i)
    
        return G


    def compute_betas(self, state):

        m = len(self.constraints)
    
        if m == 0:
            return []
    
        Gmat = np.zeros((m, m))
        gamma = np.zeros(m)
    
        for i, Bi in enumerate(self.constraints):
            for j, Bj in enumerate(self.constraints):
                Gmat[i, j] = np.real(state.expectation_value((Bi @ Bj).operator))
    
            gamma[i] = -np.real(state.expectation_value((self.H @ Bi).operator))
    
        return np.linalg.lstsq(Gmat, gamma, rcond=None)[0]