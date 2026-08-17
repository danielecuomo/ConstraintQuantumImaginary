from __future__ import annotations

from typing import List, Tuple
from numpy import pi
import numpy as np
from qiskit.quantum_info import Pauli


class Hamiltonian:

    def __init__(self, size: list[int]):
        if len(size) != 2:
            raise ValueError("size must be [rows, cols]")

        self.size = size
        self.num_qubits = size[0] * size[1]

        self.terms: list[tuple[str, float]] = []

        self._build_terms()

    # ----------------------------------------------------------

    def _nearest_neighbors(self):
        rows, cols = self.size
        n = self.num_qubits

        if rows == 1 or cols == 1:
            return [(i, i + 1) for i in range(n - 1)]

        edges = []
        for i in range(n):
            if (i + 1) % cols != 0:
                edges.append((i, i + 1))
            if i + cols < n:
                edges.append((i, i + cols))
        return edges

    # ----------------------------------------------------------

    def _pauli_term(self, ops):
        label = ["I"] * self.num_qubits
        for i, op in ops.items():
            label[i] = op
        return "".join(label)

    def _build_terms(self):
        raise NotImplementedError

    # default trivial bound
    def norm_upperbound(self):
        return sum(abs(c) for _, c in self.terms)


# ==========================================================
# Helper for anticommutation grouping
# ==========================================================

def _anticommutes(a: str, b: str) -> bool:
    return Pauli(a).anticommutes(Pauli(b))


def _group_bound(terms: List[Tuple[str, float]]):

    groups: List[List[Tuple[str, float]]] = []

    for label, coeff in terms:

        placed = False

        for g in groups:
            if all(_anticommutes(label, l) for l, _ in g):
                g.append((label, coeff))
                placed = True
                break

        if not placed:
            groups.append([(label, coeff)])

    bound = 0.0

    for g in groups:
        bound += np.sqrt(sum(c * c for _, c in g))

    return bound
