from collections import defaultdict
from qiskit.quantum_info import SparsePauliOp
from dataclasses import dataclass


class PauliHamiltonian:

    def __init__(self, pauli_strings):

        if not pauli_strings:
            raise ValueError("Hamiltonian cannot be empty")

        n = len(pauli_strings[0])

        for p in pauli_strings:
            if len(p) != n:
                raise ValueError("All Pauli strings must have same length")

        self.terms = list(pauli_strings)

        paulis = [p.string for p in self.terms]
        coeffs = [p.coeff for p in self.terms]

        self.operator = SparsePauliOp(paulis, coeffs)

    @property
    def num_qubits(self):
        return len(self.terms[0])

    def expectation(self, statevector):
        return statevector.expectation_value(self.operator)

    def __iter__(self):
        """Allows: for pauli, coeff in H"""
        for p in self.terms:
            yield p.string, p.coeff

    def __matmul__(self, other):

        if self.num_qubits != other.num_qubits:
            raise ValueError("Hamiltonians must act on same number of qubits")

        result = defaultdict(complex)

        for p1 in self.terms:
            for p2 in other.terms:

                prod = p1 @ p2
                result[prod.string] += prod.coeff

        new_terms = [
            PauliString(p, c)
            for p, c in result.items()
            if abs(c) > 1e-12
        ]

        return PauliHamiltonian(new_terms)

    def __add__(self, other):

        if self.num_qubits != other.num_qubits:
            raise ValueError("Hamiltonians must act on same number of qubits")
    
        result = defaultdict(complex)
    
        for p in self.terms:
            result[p.string] += p.coeff
    
        for p in other.terms:
            result[p.string] += p.coeff
    
        new_terms = [
            PauliString(s, c)
            for s, c in result.items()
            if abs(c) > 1e-12
        ]
        
        if len(new_terms) == 0:
            return PauliHamiltonian.zero(self.num_qubits)
        
        return PauliHamiltonian(new_terms)

    def __sub__(self, other):

        if self.num_qubits != other.num_qubits:
            raise ValueError("Hamiltonians must act on same number of qubits")
    
        return self + other.scale(-1)
    
    def scale(self, scalar):

        new_terms = [
            PauliString(p.string, p.coeff * scalar)
            for p in self.terms
        ]
    
        return PauliHamiltonian(new_terms)

    def to_matrix(self):
        return self.operator.to_matrix()

    @classmethod
    def zero(cls, num_qubits):

        return cls([
            PauliString("I" * num_qubits, 0.0)
        ])

    @property
    def is_zero(self):
    
        return all(abs(p.coeff) < 1e-12 for p in self.terms)



@dataclass(frozen=True)
class PauliString:
    string: str
    coeff: complex = 1.0

    _mult_table = {
        ('I','I'):(1,'I'), ('I','X'):(1,'X'), ('I','Y'):(1,'Y'), ('I','Z'):(1,'Z'),
        ('X','I'):(1,'X'), ('Y','I'):(1,'Y'), ('Z','I'):(1,'Z'),

        ('X','X'):(1,'I'), ('Y','Y'):(1,'I'), ('Z','Z'):(1,'I'),

        ('X','Y'):(1j,'Z'), ('Y','X'):(-1j,'Z'),
        ('Y','Z'):(1j,'X'), ('Z','Y'):(-1j,'X'),
        ('Z','X'):(1j,'Y'), ('X','Z'):(-1j,'Y')
    }

    def __len__(self):
        return len(self.string)

    def __matmul__(self, other: "PauliString") -> "PauliString":

        if len(self) != len(other):
            raise ValueError("Pauli strings must act on same number of qubits")

        phase = self.coeff * other.coeff
        result = []

        for a, b in zip(self.string, other.string):
            p, s = self._mult_table[(a, b)]
            phase *= p
            result.append(s)

        return PauliString("".join(result), phase)

    def to_qiskit(self):
        return SparsePauliOp(self.string, coeffs=[self.coeff])

    