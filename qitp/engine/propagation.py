import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qitp.engine.unitary_approximation import UnitaryApproximation



class QITEPropagation:

    def __init__(self, hamiltonian, backend, approx):

        self.H = hamiltonian
        self.approx = approx
        self.backend = backend
        
        self.constraints = []
        self.targets = []
        #self.seeds = seeds
        #self.depth = depth
        #self.basis_class = basis_class

        self.num_qubits = hamiltonian.num_qubits

    def step(self, circuit, state, dt):

        G = self.build_constrained_generator(circuit)
    
        #basis = self.basis_class.from_seed(
        #    hamiltonian=G,
        #    seeds=self.seeds,
        #    depth=self.depth
        #)
    
        #self.approx = UnitaryApproximation(
        #    basis
        #)

 
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
    
        #a = self.approx.solve(S, b)
        direction, alpha = self.approx.solve_direction(
            S,
            b
        )
        
        effective_dt = (
            dt * alpha
        )
        
        
    
        U = self.approx.build_unitary(direction, effective_dt)

        if state is not None:
            circuit = QuantumCircuit(self.num_qubits)
    
            circuit.initialize(
            state.data,
            range(self.num_qubits)
            )
            state = state.evolve(U)
    
        circuit = circuit.compose(U)
    
        return circuit, state

    

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

    