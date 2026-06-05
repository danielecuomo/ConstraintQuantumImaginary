import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import LieTrotter



class QITEPropagation:

    def __init__(self, hamiltonian, approx, backend):

        self.H = hamiltonian
        self.approx = approx
        self.backend = backend
        
        self.constraints = []
        self.targets = []

        self.num_qubits = hamiltonian.num_qubits

    def step(self, circuit, state, dt):

        G = self.build_constrained_generator(circuit)
    
        G_matrix = G.to_matrix()
    
        observables, g_products = (
            self.approx.build_measurement_cache(G)
        )
        
        measured = self.approx.measure_cache(
            circuit,
            self.backend,
            observables
        )
        
        S = self.approx.compute_S(
            measured
        )
        
        b = self.approx.compute_b(
            measured,
            g_products
        )
    
        a = self.approx.solve(S, b)
    
        U = self.build_unitary(a, dt)
    
        circuit = circuit.compose(U)

        if state is not None:
            state = state.evolve(U)
    
        return circuit, state

    def build_unitary(self, coeffs, dt):

        op = None
    
        for i, c in enumerate(coeffs):
    
            if abs(c) < 1e-12:
                continue
    
            term = c * self.approx.basis.operator(i)
    
            if op is None:
                op = term
            else:
                op += term
    
        if op is None:
    
            return QuantumCircuit(self.num_qubits)
    
        qc = QuantumCircuit(self.num_qubits)
    
        qc.append(
            PauliEvolutionGate(
                op,
                time=dt,
                synthesis=LieTrotter(reps=2)
            ),
            range(self.num_qubits)
        )
    
        return qc.decompose()

    def build_constrained_generator(self, circuit):
    
        if not self.constraints:
            return self.H
    
        beta = self.compute_betas(
            circuit,
            self.backend
        )
    
        G = self.H
    
        for b_i, B_i in zip(beta, self.constraints):
            G = G + b_i * B_i
    
        return G


    def compute_betas(
        self,
        circuit,
        backend
    ):
    
        m = len(self.constraints)
    
        if m == 0:
            return np.array([])
    
        labels_needed = set()
    
        gram_products = [
            [None] * m
            for _ in range(m)
        ]
    
        h_products = [None] * m
    
        #
        # Costruzione operatori
        #
    
        for i, Bi in enumerate(self.constraints):
    
            op = (
                self.H @ Bi
            ).simplify()
    
            h_products[i] = op
    
            for pauli in op.paulis:
    
                labels_needed.add(
                    pauli.to_label()
                )
    
            for j, Bj in enumerate(self.constraints):
    
                op = (
                    Bi @ Bj
                ).simplify()
    
                gram_products[i][j] = op
    
                for pauli in op.paulis:
    
                    labels_needed.add(
                        pauli.to_label()
                    )
    
        #
        # Misura unica
        #
    
        labels = list(labels_needed)
    
        observables = [
            SparsePauliOp(
                label,
                coeffs=[1.0]
            )
            for label in labels
        ]
    
        values = backend.expectations(
            circuit,
            observables
        )
    
        measured = dict(
            zip(labels, values)
        )
    
        #
        # Gamma
        #
    
        gamma = np.zeros(m)
    
        for i, op in enumerate(h_products):
    
            value = 0.0j
    
            for pauli, coeff in zip(
                op.paulis,
                op.coeffs
            ):
    
                value += (
                    coeff
                    * measured[
                        pauli.to_label()
                    ]
                )
    
            gamma[i] = -np.real(value)
    
        #
        # Gram
        #
    
        Gmat = np.zeros((m, m))
    
        for i in range(m):
    
            for j in range(m):
    
                op = gram_products[i][j]
    
                value = 0.0j
    
                for pauli, coeff in zip(
                    op.paulis,
                    op.coeffs
                ):
    
                    value += (
                        coeff
                        * measured[
                            pauli.to_label()
                        ]
                    )
    
                Gmat[i, j] = np.real(value)
    
        #
        # Solve
        #
    
        eigvals, eigvecs = np.linalg.eigh(Gmat)
    
        tol = 1e-12 * max(
            np.max(np.abs(eigvals)),
            1.0
        )
    
        inv_eigs = np.zeros_like(eigvals)
    
        mask = eigvals > tol
    
        inv_eigs[mask] = (
            1.0 / eigvals[mask]
        )
    
        Gplus = (
            eigvecs
            @ np.diag(inv_eigs)
            @ eigvecs.T
        )
    
        beta = Gplus @ gamma
    
        return beta

    def constrained_generator_matrix(self, circuit):

        G = self.build_constrained_generator(circuit)
    
        return G.to_matrix()