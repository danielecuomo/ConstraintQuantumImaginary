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
                val = state.expectation_value(op).real

                S[i, j] = val
                S[j, i] = val

        return S

    def compute_b(self, state, H):
        """
        b_i = -Re(<psi | A_i H | psi>)
        """
        b = np.zeros(self.m)

        for i in range(self.m):

            val = 0.0

            for Pk in H.terms:

                # PauliString -> SparsePauliOp
                Hk = Pk.to_qiskit()

                op = self.pauli_basis[i] @ Hk

                val += Pk.coeff * state.expectation_value(op)

            b[i] = -val.real

        return b

    def solve(self, S, b, dt):
        """
        Solve S a = dt b
        """
        reg = 1e-4
        S_reg = S + reg * np.eye(self.m)

        return np.linalg.solve(S_reg, dt * b)