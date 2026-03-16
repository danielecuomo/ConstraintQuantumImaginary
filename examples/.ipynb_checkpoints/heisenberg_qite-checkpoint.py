from qiskit import QuantumCircuit
from qitp.hamiltonians.pauli_hamiltonian import PauliHamiltonian
from qitp.engine.propagation import QITEPropagation
from qitp.simulation.backend import SimulatorBackend

paulis = ["XX", "YY", "ZZ"]
coeffs = [1, 1, 1]

H = PauliHamiltonian(paulis, coeffs)

backend = SimulatorBackend()

engine = QITEPropagation(H, None, backend)

initial_circuit = QuantumCircuit(2)
state = backend.run(initial_circuit)

for _ in range(10):

    state = engine.step(state)

    energy = H.expectation(state)

    print("Energy:", energy)
