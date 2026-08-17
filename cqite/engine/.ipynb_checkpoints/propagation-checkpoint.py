import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from cqite.engine.unitary_approximation import UnitaryApproximation

import time

class QITEPropagation:

    def __init__(self, hamiltonian, constraints, backend, approx):

        self.H = hamiltonian
        self.approx = approx
        self.backend = backend
        
        if isinstance(constraints, SparsePauliOp):
            self.constraints = [constraints]
        else:
            self.constraints = list(constraints)
        
        identity = SparsePauliOp(
            "I"*hamiltonian.num_qubits,
            coeffs=[1.0]
        )
        
        self.constraints.append(identity)
        self.targets = []

        self.num_qubits = hamiltonian.num_qubits

    def step(self, circuit, state, dt):
        G = self.build_constrained_generator(circuit, state)

        observables, g_products = (
            self.approx.build_measurement_cache(G)
        )

        measured = self.approx.measure_cache(
            circuit,
            self.backend,
            observables,
            state
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

        if state is None:

            U = self.approx.build_unitary(
                direction,
                effective_dt
            )
        
            circuit.compose(
                U,
                inplace=True
            )
        
        else:
        
            U = self.approx.build_operator(
                direction,
                effective_dt
            )

            state = state.evolve(U)
    
        return circuit, state

    

    def build_constrained_generator(
        self,
        circuit,
        state=None
    ):
    
        if not self.constraints:
            return self.H
    
        beta = self.compute_betas(
            circuit,
            self.backend,
            state
        )
    
        G = self.H
    
        for b_i, B_i in zip(beta, self.constraints):
            G += b_i * B_i
    
        return G


    def compute_betas(
        self,
        circuit,
        backend,
        state=None
    ):
    
        m = len(self.constraints)
    
        if m == 0:
            return np.array([])
    
        labels_needed = set()
    
        gram_products = [[None] * m for _ in range(m)]
        h_products = [None] * m
    
        add_label = labels_needed.add
        H = self.H
        constraints = self.constraints
    
        #
        # Costruzione operatori
        #
    
        for i, Bi in enumerate(constraints):
    
            op = (H @ Bi).simplify()
            h_products[i] = op
    
            for pauli in op.paulis:
                add_label(pauli.to_label())
    
            row = gram_products[i]
    
            for j, Bj in enumerate(constraints):
    
                op = (Bi @ Bj).simplify()
                row[j] = op
    
                for pauli in op.paulis:
                    add_label(pauli.to_label())
    
        #
        # Misura
        #
    
        cache = getattr(self.approx, "observable_cache", {})
    
        labels = list(labels_needed)
    
        observables = []
    
        for label in labels:
    
            obs = cache.get(label)
    
            if obs is None:
                obs = SparsePauliOp(label, coeffs=[1.0])
                cache[label] = obs
    
            observables.append(obs)
    
        values = backend.expectations(
            circuit,
            observables,
            state
        )
    
        measured = dict(zip(labels, values))
        get = measured.__getitem__
    
        #
        # Gamma
        #
    
        gamma = np.empty(m)
    
        for i in range(m):
    
            op = h_products[i]
    
            paulis = op.paulis
            coeffs = op.coeffs
    
            value = 0.0j
    
            for k in range(len(coeffs)):
                value += coeffs[k] * get(paulis[k].to_label())
    
            gamma[i] = -value.real
    
        #
        # Gram
        #
    
        Gmat = np.empty((m, m))
    
        for i in range(m):
    
            row = gram_products[i]
    
            for j in range(m):
    
                op = row[j]
    
                paulis = op.paulis
                coeffs = op.coeffs
    
                value = 0.0j
    
                for k in range(len(coeffs)):
                    value += coeffs[k] * get(paulis[k].to_label())
    
                Gmat[i, j] = value.real
    
        #
        # Solve
        #
    
        eigvals, eigvecs = np.linalg.eigh(Gmat)
    
        tol = 1e-12 * max(np.max(np.abs(eigvals)), 1.0)
    
        inv_eigs = np.zeros_like(eigvals)
    
        mask = eigvals > tol
        inv_eigs[mask] = 1.0 / eigvals[mask]
    
        return (
            eigvecs
            @ (inv_eigs[:, None] * eigvecs.T)
            @ gamma
        )

    def constrained_generator_matrix(
        self,
        circuit,
        state=None
    ):
    
        G = self.build_constrained_generator(
            circuit,
            state
        )
    
        return G.to_matrix()

    