from .base import OperatorBasis

from qiskit.quantum_info import SparsePauliOp

import numpy as np


class SymplecticGeneratorBasis(OperatorBasis):

    def __init__(self, generators):

        self.generators = generators

    @property
    def size(self):

        return len(self.generators)

    def matrix(self, i):

        return self.generators[i].to_matrix()

    def product(self, i, j):

        return self.generators[i] @ self.generators[j]

    def operator(self, i):

        return self.generators[i]

    #
    # --------------------------------------------------
    # symplectic conversion
    # --------------------------------------------------
    #

    @staticmethod
    def pauli_to_symplectic(label):

        n = len(label)

        x = np.zeros(
            n,
            dtype=np.uint8
        )

        z = np.zeros(
            n,
            dtype=np.uint8
        )

        for i, p in enumerate(label):

            if p == "X":

                x[i] = 1

            elif p == "Z":

                z[i] = 1

            elif p == "Y":

                x[i] = 1
                z[i] = 1

        return np.concatenate(
            [x, z]
        )

    @staticmethod
    def symplectic_to_pauli(v):

        n = len(v) // 2

        x = v[:n]
        z = v[n:]

        chars = []

        for xi, zi in zip(x, z):

            if xi == 0 and zi == 0:

                chars.append("I")

            elif xi == 1 and zi == 0:

                chars.append("X")

            elif xi == 0 and zi == 1:

                chars.append("Z")

            else:

                chars.append("Y")

        return "".join(chars)

    #
    # --------------------------------------------------
    # GF(2) basis extraction
    # --------------------------------------------------
    #

    @classmethod
    def gf2_basis(
        cls,
        vectors
    ):

        vectors = [
            np.array(v, dtype=np.uint8)
            for v in vectors
        ]

        pivots = []

        basis = []

        for v in vectors:

            w = v.copy()

            for b, p in zip(
                basis,
                pivots
            ):

                if w[p]:

                    w ^= b

            nz = np.flatnonzero(w)

            if len(nz) == 0:

                continue

            p = nz[0]

            pivots.append(p)

            basis.append(w)

        return basis

    #
    # --------------------------------------------------
    # build from closure
    # --------------------------------------------------
    #

    @classmethod
    def from_operators(
        cls,
        operators
    ):

        vectors = []

        for op in operators:

            op = op.simplify()

            for p in op.paulis:

                vectors.append(

                    cls.pauli_to_symplectic(
                        p.to_label()
                    )
                )

        basis_vectors = cls.gf2_basis(
            vectors
        )

        generators = []

        for v in basis_vectors:

            label = cls.symplectic_to_pauli(
                v
            )

            generators.append(

                SparsePauliOp(
                    label,
                    coeffs=[1.0]
                )
            )

        return cls(
            generators
        )