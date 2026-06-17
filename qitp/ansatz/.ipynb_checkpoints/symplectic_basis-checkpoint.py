from .base import OperatorBasis

from qiskit.quantum_info import SparsePauliOp

import numpy as np


class SymplecticBasis(OperatorBasis):

    def __init__(self, paulis):

        self.paulis = paulis

    @property
    def size(self):

        return len(self.paulis)

    def matrix(self, i):

        return self.paulis[i].to_matrix()

    def product(self, i, j):

        return self.paulis[i] @ self.paulis[j]

    def operator(self, i):

        return self.paulis[i]

    #
    # ------------------------------------------
    # commutator
    # ------------------------------------------
    #

    @staticmethod
    def commutator(A, B):

        C = (A @ B) - (B @ A)

        C = C.simplify()

        if len(C.paulis) == 0:

            return None

        return SymplecticBasis.clean(
            -1j * C
        )

    #
    # ------------------------------------------
    # symplectic representation
    # ------------------------------------------
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

    #
    # ------------------------------------------
    # GF(2) insertion
    # ------------------------------------------
    #

    @staticmethod
    def insert_vector(
        vector,
        basis
    ):

        v = vector.copy()

        for b in basis:

            pivot = np.flatnonzero(b)[0]

            if v[pivot]:

                v ^= b

        if not np.any(v):

            return False

        basis.append(v)

        basis.sort(
            key=lambda x:
            np.flatnonzero(x)[0]
        )

        return True

    #
    # ------------------------------------------
    # closure
    # ------------------------------------------
    #

    @classmethod
    def from_seed(
        cls,
        hamiltonian,
        seeds,
        depth=1
    ):

        #
        # GF(2) basis
        #

        symplectic_basis = []

        #
        # Pauli generators selected
        #

        selected = []

        #
        # explore closure
        #

        frontier = [
            cls.clean(op)
            for op in seeds
        ]

        visited = []

        for op in frontier:

            visited.append(op)

        for _ in range(depth):

            new_frontier = []

            for op in frontier:

                #
                # inspect every Pauli
                #

                for p in op.paulis:

                    label = p.to_label()

                    v = cls.pauli_to_symplectic(
                        label
                    )

                    independent = (
                        cls.insert_vector(
                            v,
                            symplectic_basis
                        )
                    )

                    if independent:

                        selected.append(

                            SparsePauliOp(
                                label,
                                coeffs=[1.0]
                            )

                        )

                comm = cls.commutator(
                    hamiltonian,
                    op
                )

                if comm is None:

                    continue

                coeff_norm = np.linalg.norm(
                    comm.coeffs
                )

                if coeff_norm > 1e-12:

                    comm = SparsePauliOp(
                        comm.paulis,
                        coeffs=(
                            comm.coeffs
                            / coeff_norm
                        )
                    )

                comm = cls.clean(comm)

                new_frontier.append(
                    comm
                )

                visited.append(
                    comm
                )

            frontier = new_frontier

            if len(symplectic_basis) >= (
                2 * hamiltonian.num_qubits
            ):
                break

        return cls(
            selected
        )

    #
    # ------------------------------------------
    # rotations
    # ------------------------------------------
    #

    @classmethod
    def from_rotations(
        cls,
        hamiltonian,
        rotations,
        depth=1
    ):

        seeds = cls.dressed_generators(
            hamiltonian.num_qubits,
            rotations
        )

        return cls.from_seed(
            hamiltonian,
            seeds,
            depth
        )

    @classmethod
    def dress(
        cls,
        operator,
        generator,
        theta
    ):

        AP = (
            generator
            @ operator
        ).simplify()

        PA = (
            operator
            @ generator
        ).simplify()

        if AP == PA:

            return operator

        return cls.clean(

            np.cos(
                2 * theta
            ) * operator

            - np.sin(
                2 * theta
            ) * (1j * AP)

        )

    @classmethod
    def dressed_generators(
        cls,
        nqubits,
        rotations
    ):

        generators = [

            SparsePauliOp(
                "I" * i
                + "Z"
                + "I" * (
                    nqubits - i - 1
                ),
                coeffs=[1.0]
            )

            for i in range(
                nqubits
            )
        ]

        for rot in rotations:

            generators = [

                cls.dress(
                    op,
                    rot["generator"],
                    rot["theta"]
                )

                for op in generators
            ]

        return generators

    @classmethod
    def clean(
        cls,
        op
    ):

        op = op.simplify()

        return SparsePauliOp(
            op.paulis,
            coeffs=np.real(
                op.coeffs
            )
        ).simplify()