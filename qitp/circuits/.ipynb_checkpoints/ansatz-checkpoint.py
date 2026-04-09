from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

class HardwareEfficientAnsatz:

    def __init__(self, n_qubits, depth):
        self.n = n_qubits
        self.depth = depth
        self.params = ParameterVector("theta", n_qubits * depth * 2)

    def build(self):

        qc = QuantumCircuit(self.n)
        p = 0

        for d in range(self.depth):

            for q in range(self.n):
                qc.ry(self.params[p], q)
                p += 1

            for q in range(self.n - 1):
                qc.cx(q, q + 1)

            for q in range(self.n):
                qc.rz(self.params[p], q)
                p += 1

        return qc
