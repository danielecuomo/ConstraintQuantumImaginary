from .base import OperatorBasis

from qiskit.quantum_info import SparsePauliOp

import numpy as np


class CommutatorBasis(OperatorBasis):

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

    @staticmethod
    def commutator(A, B):

        C = (A @ B) - (B @ A)

        C = C.simplify()

        if len(C.paulis) == 0:
            return None

        return (-1j * C).simplify()

    @staticmethod
    def canonical(op):

        op = op.simplify()

        labels = op.paulis.to_labels()
        coeffs = op.coeffs

        return tuple(
            sorted(
                zip(
                    labels,
                    np.round(coeffs, 12)
                )
            )
        )

    @classmethod
    def from_seed(
        cls,
        hamiltonian,
        seeds,
        depth=1
    ):

        operators = []
        frontier = []
        seen = set()

        for op in seeds:

            op = op.simplify()

            key = cls.canonical(op)

            if key in seen:
                continue

            seen.add(key)

            operators.append(op)
            frontier.append(op)

        for _ in range(depth):

            new_frontier = []

            for op in frontier:

                comm = cls.commutator(
                    hamiltonian,
                    op
                )

                if comm is None:
                    continue

                key = cls.canonical(comm)

                if key in seen:
                    continue

                seen.add(key)

                operators.append(comm)
                new_frontier.append(comm)

            frontier = new_frontier

            if not frontier:
                break

        return cls(operators)


    @classmethod
    def from_rotations(
        cls,
        hamiltonian,
        rotations,
        depth
    ):
    
        nqubits = hamiltonian.num_qubits
    
        frontier = cls.dressed_generators(
            nqubits,
            rotations
        )
    
        basis = list(frontier)
    
        for _ in range(depth):
    
            new_frontier = []
    
            for op in frontier:
    
                comm = cls.commutator(
                    hamiltonian,
                    op
                )
    
                if len(comm.paulis):
    
                    new_frontier.append(comm)
    
            basis.extend(new_frontier)
    
            frontier = new_frontier
    
        return cls(basis)
    

    import numpy as np
    from qiskit.quantum_info import SparsePauliOp
    
    @classmethod
    def dress(
        cls,
        operator,
        generator,
        theta
    ):
        AP = (generator @ operator).simplify()
        PA = (operator @ generator).simplify()
    
        #
        # commutano
        #
    
        if AP == PA:
            return operator
    
        #
        # anticommutano
        #
    
        return (
            np.cos(2 * theta) * operator
            - np.sin(2 * theta) * (1j * AP)
        ).simplify()
    
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
                + "I" * (nqubits - i - 1),
                coeffs=[1.0]
            )
            for i in range(nqubits)
        ]
    
        for rot in rotations:
    
            A = rot["generator"]
            theta = rot["theta"]
    
            generators = [
                cls.dress(
                    op,
                    A,
                    theta
                )
                for op in generators
            ]
    
        return generators