import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import PauliEvolutionGate

class QITEPropagation:

    def __init__(self, hamiltonian, approx, backend):
        self.H = hamiltonian
        self.approx = approx
        self.backend = backend

    def step(self, state , dt):

        S = self.approx.compute_S(state)
        b = self.approx.compute_b(state, self.H.operator)
    
        a = self.approx.solve(S, b, dt)
    
        U_circuit = self.build_unitary(a, dt)
    
        new_state = state.evolve(U_circuit)
    
        return new_state


    def build_unitary(self, coeffs, dt):

        paulis = []
        weights = []
    
        for c, P in zip(coeffs, self.approx.pauli_basis):
    
            paulis.append(P.paulis[0])
            weights.append(c)
    
        op = SparsePauliOp(paulis, weights)
    
        evo = PauliEvolutionGate(op, time=dt)
    
        qc = QuantumCircuit(op.num_qubits)
        qc.append(evo, range(op.num_qubits))
    
        return qc