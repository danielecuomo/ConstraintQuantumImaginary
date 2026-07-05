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


# ==========================================================
# TFIM MODEL
# ==========================================================

class TFIHamiltonian(Hamiltonian):

    def __init__(self, size, gamma=pi / 4, eta=pi / 4):
        self.gamma = gamma
        self.eta = eta
        super().__init__(size)

    def _build_terms(self):

        self.terms = []

        for i, j in self._nearest_neighbors():
            self.terms.append(
                (self._pauli_term({i: "Z", j: "Z"}), self.gamma)
            )

        for i in range(self.num_qubits):
            self.terms.append(
                (self._pauli_term({i: "X"}), self.eta)
            )

    def norm_upperbound(self):
        return _group_bound(self.terms)


# ==========================================================
# XY MODEL
# ==========================================================

class XYHamiltonian(Hamiltonian):

    def __init__(self, size, gamma=pi / 4):
        self.gamma = gamma
        super().__init__(size)

    def _build_terms(self):

        self.terms = []

        for i, j in self._nearest_neighbors():

            self.terms.append(
                (self._pauli_term({i: "X", j: "X"}), self.gamma)
            )

            self.terms.append(
                (self._pauli_term({i: "Y", j: "Y"}), self.gamma)
            )

    def norm_upperbound(self):
        return _group_bound(self.terms)

class ScalableHamiltonian(Hamiltonian):

    PLAQUETTES = {
        1: [(0, 0)],
    
        2: [
            (0, 0), (1, 0)
        ],
    
        3: [
            (0, 0), (1, 0), (0, 1)
        ],
        
        4: [
            (0, 0), (1, 0), (0, 1), (1, 1)
        ],
    
        5: [
            (0, 0), (1, 0), (0, 1), (1, 1), (-1, 0)
        ],
    
        6: [
            (0, 0), (1, 0), (0, 1), (1, 1), (-1, 0), (-1, 1)
        ]
    }

    def __init__(self, n: int):
        if n not in self.PLAQUETTES:
            raise ValueError("Supported n: 1 to 5")

        self.n = n
        self.squares = self.PLAQUETTES[n]

        self.coords = self._build_vertices()
        self.num_qubits = len(self.coords)

        self.terms: list[tuple[str, float]] = []

        self._build_terms()

    # ----------------------------------------------------------

    def _build_vertices(self):
        vertices = set()

        for x, y in self.squares:
            vertices.update([
                (x, y),
                (x + 1, y),
                (x, y + 1),
                (x + 1, y + 1),
            ])

        return sorted(vertices)

    # ----------------------------------------------------------

    def _coord_index_map(self):
        return {coord: i for i, coord in enumerate(self.coords)}

    # ----------------------------------------------------------

    def _nearest_neighbors(self):
        coord_to_idx = self._coord_index_map()
        edges = []

        for (x, y), i in coord_to_idx.items():
            for nbr in [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]:
                if nbr in coord_to_idx:
                    j = coord_to_idx[nbr]
                    if i < j:
                        edges.append((i, j))

        return edges

    # ----------------------------------------------------------

    def _pauli_term(self, ops):
        label = ["I"] * self.num_qubits
        for i, op in ops.items():
            label[i] = op
        return "".join(label)

    # ----------------------------------------------------------

    def _build_terms(self):
        raise NotImplementedError

    # ----------------------------------------------------------

    def norm_upperbound(self):
        return sum(abs(c) for _, c in self.terms)
    
    def draw(self, with_labels=True, show_edges=True):
        import matplotlib.pyplot as plt
    
        fig, ax = plt.subplots()
    
        # --- draw plaquettes (squares) ---
        for x, y in self.squares:
            square = plt.Rectangle((x, y), 1, 1, fill=False, linewidth=2)
            ax.add_patch(square)
    
        # --- draw edges ---
        if show_edges:
            coord_to_idx = self._coord_index_map()
    
            for i, j in self._nearest_neighbors():
                x1, y1 = self.coords[i]
                x2, y2 = self.coords[j]
                ax.plot([x1, x2], [y1, y2])
    
        # --- draw vertices ---
        xs = [x for x, y in self.coords]
        ys = [y for x, y in self.coords]
        ax.scatter(xs, ys, s=80, zorder=3)
    
        # --- labels ---
        if with_labels:
            for i, (x, y) in enumerate(self.coords):
                ax.text(x, y, str(i), ha='center', va='center', fontsize=8)
    
        ax.set_aspect('equal')
        ax.set_title(f"Scalable lattice (n={self.n})")
    
        # clean axes
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(min(xs) - 1, max(xs) + 1)
        ax.set_ylim(min(ys) - 1, max(ys) + 1)
    
        plt.show()
        
    def get_network(self):
        """
        Build, cache, and return the interaction graph.
    
        Nodes:
            integer indices (matching qubits)
    
        Node attributes:
            - "pos": (x, y) lattice coordinates
    
        Edges:
            nearest-neighbor couplings
        """
        if hasattr(self, "_graph"):
            return self._graph
    
        import networkx as nx
    
        G = nx.Graph()
    
        # --- nodes with positions ---
        for i, coord in enumerate(self.coords):
            G.add_node(i, pos=coord)
    
        # --- edges ---
        G.add_edges_from(self._nearest_neighbors())
    
        # --- cache both graph and position dict ---
        self._graph = G
        self._pos = {i: coord for i, coord in enumerate(self.coords)}
    
        return G

class ScalableDiagHamiltonian(ScalableHamiltonian):

    def __init__(self, n, gamma=0.7, target = 0.4):
        self.gamma = gamma
        self.target = target
        super().__init__(n)

    def _build_terms(self):
        for i, j in self._nearest_neighbors():
            self.terms.append(
                (self._pauli_term({i: "Z", j: "Z"}), self.gamma)
            )
        
        self.cterms = []

        for i in range(self.num_qubits):
            self.cterms.append(
                (self._pauli_term({i: "Y"}), 1.0)
            )

    def norm_upperbound(self):
        return _group_bound(self.terms)