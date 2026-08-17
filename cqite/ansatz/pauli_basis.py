from .base import OperatorBasis


class PauliBasis(OperatorBasis):

    def __init__(self, paulis):

        self.paulis = paulis

    @property
    def size(self):

        return len(self.paulis)

   
    def operator(self, i):

        return self.paulis[i]

    def matrix(self, i):

        return self.paulis[i].to_matrix()

    def product(self, i, j):

        return self.paulis[i] @ self.paulis[j]