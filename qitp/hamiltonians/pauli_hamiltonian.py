from qiskit.quantum_info import SparsePauliOp

class PauliHamiltonian:

    def __init__(self, pauli_terms, coeffs):
        self.operator = SparsePauliOp(pauli_terms, coeffs)

    def num_qubits(self):
        return self.operator.num_qubits

    def expectation(self, statevector):
        return statevector.expectation_value(self.operator)
