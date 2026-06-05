import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import PauliEvolutionGate

from scipy.linalg import expm
from scipy.linalg import pinv

class QITEPropagation:

    def __init__(self, hamiltonian, approx, backend):

        self.H = hamiltonian
        self.approx = approx
        self.backend = backend
        
        self.constraints = []
        self.targets = []

        self.num_qubits = hamiltonian.num_qubits

    def step(self, circuit, state, dt):

        G = self.build_constrained_generator(state)
    
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
        #circuit = QuantumCircuit(self.num_qubits)
    
        #circuit.initialize(
        #    state.data,
        #    range(self.num_qubits)
        #)
    
        #circuit.compose(
        #    U,
        #    inplace=True
        #)
    
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
                time=dt
            ),
            range(self.num_qubits)
        )
    
        return qc

    def build_constrained_generator(self, state):
    
        if not self.constraints:
            return self.H
    
        beta = self.compute_betas(state)
    
        G = self.H
    
        for b_i, B_i in zip(beta, self.constraints):
            G = G + b_i * B_i
    
        return G


    def compute_betas(self, state):

        m = len(self.constraints)
    
        if m == 0:
            return np.array([])
    
        psi = state.data
    
        Gmat = np.zeros((m, m))
        gamma = np.zeros(m)
    
        HPsi = self.H.to_matrix() @ psi
    
        for i, Bi in enumerate(self.constraints):

            HPsi = self.H.to_matrix() @ psi
        
            BiPsi = Bi.to_matrix() @ psi
        
            gamma[i] = -np.real(
                np.vdot(HPsi, BiPsi)
            )
        
            for j, Bj in enumerate(self.constraints):
        
                BjPsi = Bj.to_matrix() @ psi
        
                Gmat[i, j] = np.real(
                    np.vdot(BiPsi, BjPsi)
                )
    
        # Diagonalizzazione della Gram matrix
        eigvals, eigvecs = np.linalg.eigh(Gmat)
    
        # Tolleranza relativa
        tol = 1e-12 * max(np.max(np.abs(eigvals)), 1.0)
    
        # Pseudoinversa spettrale
        inv_eigs = np.zeros_like(eigvals)
    
        mask = eigvals > tol
    
        inv_eigs[mask] = 1.0 / eigvals[mask]
    
        Gplus = (
            eigvecs
            @ np.diag(inv_eigs)
            @ eigvecs.T
        )
    
        beta = Gplus @ gamma
    
        return beta

    def constrained_generator_matrix(self, state):

        G = self.build_constrained_generator(state)
    
        return G.to_matrix()