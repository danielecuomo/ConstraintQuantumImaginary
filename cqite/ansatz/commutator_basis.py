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

        return CommutatorBasis.clean(-1j * C)

    @staticmethod
    def anticommutator(A, B):


        C = (A @ B) + (B @ A)
        C = C.simplify()


        if len(C.paulis) == 0:
            return None


        return CommutatorBasis.clean(C)

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
    
        #
        # seed
        #
    
        for op in seeds:
    
            op = cls.clean(op)
    
            if not cls.linearly_independent(
                op,
                operators
            ):
                continue
    
            operators.append(op)
            frontier.append(op)
    
        #
        # anticommutator hierarchy
        #
    
        for _ in range(depth):
    
            new_frontier = []
    
            for op in frontier:
    
                comm = cls.anticommutator(
                    hamiltonian,
                    op
                )
    
                if comm is None:
                    continue
    
                coeff_norm = np.linalg.norm(comm.coeffs)


                if coeff_norm <= 1e-12:
                    continue


                comm = SparsePauliOp(
                    comm.paulis,
                    coeffs=comm.coeffs / coeff_norm
                )


                comm = cls.clean(comm)
    
                if not cls.linearly_independent(
                    comm,
                    operators
                ):
                    continue
    
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
        depth=1
    ):
    
        seeds = cls.dressed_generators(
            hamiltonian.num_qubits,
            rotations
        )
    
        return cls.from_seed(
            hamiltonian,
            seeds,
            depth=depth
        )
    

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
        # commute
        #
    
        if AP == PA:
            return operator
    
        #
        # anticommute
        #
    
        return CommutatorBasis.clean(
            np.cos(2 * theta) * operator
            - np.sin(2 * theta) * (1j * AP)
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

    @staticmethod
    def pauli_vector(op):
    
        return {
            p.to_label(): c
            for p, c in zip(
                op.paulis,
                op.coeffs
            )
        }

    @classmethod
    def linearly_independent(
        cls,
        op,
        basis,
        tol=1e-12
    ):
    
        if not basis:
            return True
    
        operators = basis + [op]
    
        labels = sorted({
            p.to_label()
            for O in operators
            for p in O.paulis
        })
    
        M = np.zeros(
            (
                len(operators),
                len(labels)
            ),
            dtype=complex
        )
    
        label_index = {
            label: i
            for i, label in enumerate(labels)
        }
    
        for row, O in enumerate(operators):

            for pauli, coeff in zip(
                O.paulis,
                O.coeffs
            ):
        
                col = label_index[
                    pauli.to_label()
                ]
        
                M[row, col] = 1.0#coeff
        
            norm = np.linalg.norm(M[row])
        
            if norm > tol:
                M[row] /= norm
    
        rank_before = np.linalg.matrix_rank(
            M[:-1],
            tol
        )
    
        rank_after = np.linalg.matrix_rank(
            M,
            tol
        )
    
        return rank_after > rank_before

    @classmethod
    def clean(cls, op):
    
        op = op.simplify()
    
        imag_coeffs = np.imag(op.coeffs)
        idx = np.argmax(np.abs(imag_coeffs))
        max_imag = np.abs(imag_coeffs[idx])
    
        if max_imag > 1e-6:
    
            print(
                f"[clean] max imaginary coefficient = {max_imag:.3e}"
            )
    
            print(
                "offending Pauli:",
                op.paulis[idx].to_label()
            )
    
            print(
                "coefficient:",
                op.coeffs[idx]
            )
    
            raise ValueError(
                "Non-hermitian operator generated"
            )
    
        return SparsePauliOp(
            op.paulis,
            coeffs=np.real(op.coeffs)
        ).simplify()
