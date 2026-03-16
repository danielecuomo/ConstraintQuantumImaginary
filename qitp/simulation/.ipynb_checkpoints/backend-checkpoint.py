from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector

class SimulatorBackend:

    def __init__(self):
        self.backend = AerSimulator(method="statevector")

    def run(self, circuit):

        state = Statevector.from_instruction(circuit)
        return state
