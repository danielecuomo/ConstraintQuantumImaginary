import numpy as np


class UnitaryApproximation:

    def __init__(self, pauli_basis):
        """
        pauli_basis : list[SparsePauliOp]
        """
        self.pauli_basis = pauli_basis
        self.m = len(pauli_basis)

        # Precompute pairwise products A_i A_j
        self.products = [[None]*self.m for _ in range(self.m)]

        for i in range(self.m):
            for j in range(self.m):
                self.products[i][j] = pauli_basis[i] @ pauli_basis[j]

    def compute_S(self, state):
        """
        S_ij = Re(<psi | A_i A_j | psi>)
        """
        S = np.zeros((self.m, self.m))

        for i in range(self.m):
            for j in range(i, self.m):

                op = self.products[i][j]
                val = state.expectation_value(op)
                S_ij = 0.5 * (val + np.conjugate(val))  # enforce Hermitian symmetry
                S[i, j] = S_ij.real
                S[j, i] = S_ij.real

        return S

    def compute_b(self, state, H):
    
        b = np.zeros(self.m)
    
        for i in range(self.m):
    
            total = 0.0
    
            for p in H.terms:
                Hk = p.to_qiskit()
                op = self.pauli_basis[i] @ Hk
                val = state.expectation_value(op)
                total += p.coeff * val
    
            b[i] = -np.real(total)
    
        return b

    def solve(self, S, b, dt):
        """
        Solve S a = dt b
        """
        reg = 1e-4
        S_reg = S + reg * np.eye(self.m)

        return np.linalg.solve(S_reg, dt * b)