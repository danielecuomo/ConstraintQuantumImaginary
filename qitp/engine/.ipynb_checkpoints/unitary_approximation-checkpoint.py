import numpy as np

class UnitaryApproximation:

    def __init__(self, pauli_basis):
        self.pauli_basis = pauli_basis

    def compute_S(self, state):

        m = len(self.pauli_basis)
        S = np.zeros((m, m))

        for i in range(m):
            for j in range(m):

                op = self.pauli_basis[i] @ self.pauli_basis[j]
                S[i, j] = state.expectation_value(op).real

        return S

    def compute_b(self, state, H):

        m = len(self.pauli_basis)
        b = np.zeros(m)

        for i in range(m):

            op = self.pauli_basis[i] @ H
            b[i] = -state.expectation_value(op).real

        return b

    def solve(self, S, b, dt):

        reg=1e-4
        
        S_reg = S + reg * np.eye(S.shape[0])
        return np.linalg.solve(S_reg, dt*b)
