import numpy as np
from scipy.linalg import expm
from qiskit.quantum_info import SparsePauliOp

from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import LieTrotter

from qiskit import QuantumCircuit

class UnitaryApproximation:

    def __init__(self, basis, trot):

        self.basis = basis
        self.trot = trot

        self.num_qubits = basis.paulis[0].num_qubits
    
        self.m = basis.size
    
        self.products = [
            [None] * self.m
            for _ in range(self.m)
        ]
    
        # insieme di tutte le stringhe di Pauli
        self.observable_cache = {}
        self.observable_labels = set()
    
        for i in range(self.m):

            for j in range(self.m):
        
                op = basis.product(i, j).simplify()
        
                self.products[i][j] = op
        
                for pauli in op.paulis:
        
                    label = pauli.to_label()
        
                    self.observable_labels.add(label)
        
                    if label not in self.observable_cache:
        
                        self.observable_cache[label] = (
                            SparsePauliOp(
                                label,
                                coeffs=[1.0]
                            )
                        )

    def build_measurement_cache(self, G):

        labels = set(self.observable_labels)
    
        g_products = []
    
        for i in range(self.m):
    
            P = self.basis.operator(i)
    
            op = (G @ P).simplify()
    
            g_products.append(op)
    
            for pauli in op.paulis:
    
                labels.add(
                    pauli.to_label()
                )
    
        observables = {
            label: SparsePauliOp(
                label,
                coeffs=[1.0]
            )
            for label in labels
        }
    
        return observables, g_products

    def measure_cache(
        self,
        circuit,
        backend,
        observables
    ):
    
        labels = list(
            observables.keys()
        )
    
        ops = [
            observables[label]
            for label in labels
        ]
    
        values = backend.expectations(
            circuit,
            ops
        )
    
        return dict(
            zip(labels, values)
        )

    def compute_S(
        self,
        measured
    ):
    
        S = np.zeros(
            (self.m, self.m),
            dtype=complex
        )
    
        for i in range(self.m):
    
            for j in range(self.m):
    
                op = self.products[i][j]
    
                value = 0.0j
    
                for pauli, coeff in zip(
                    op.paulis,
                    op.coeffs
                ):
    
                    value += (
                        coeff
                        * measured[
                            pauli.to_label()
                        ]
                    )
    
                S[i, j] = value
    
        return S

    def compute_b(
        self,
        measured,
        g_products
    ):
    
        b = np.zeros(self.m)
    
        for i, op in enumerate(g_products):
    
            value = 0.0j
    
            for pauli, coeff in zip(
                op.paulis,
                op.coeffs
            ):
    
                value += (
                    coeff
                    * measured[
                        pauli.to_label()
                    ]
                )
    
            b[i] = -2.0 * np.imag(value)
    
        return b

    def solve(
        self,
        S,
        b
    ):
    
        return np.linalg.lstsq(
            S,
            b,
            rcond=1e-10
        )[0]

    
    def solve_direction(
        self,
        S,
        b
    ):
    
        M = np.real(
            S + S.conj().T
        )
    
        #
        # Rayleigh solution
        #
        a = np.linalg.lstsq(
            M,
            b,
            rcond=1e-10
        )[0]
    
        #
        # normalize in M metric
        #
        norm = np.sqrt(
            np.real(
                a @ M @ a
            )
        )
    
        if norm < 1e-12:
    
            return (
                np.zeros_like(a),
                0.0
            )
    
        direction = (
            a / norm
        )
    
        #
        # optimal scale gamma
        #
        gamma = np.real(
            direction @ b
        )
    
        return (
            direction,
            gamma
        )

    #Test
    def build_linear_system(
        self,
        measured,
        g_products
    ):
    
        S = self.compute_S(
            measured
        )
    
        b = self.compute_b(
            measured,
            g_products
        )
    
        M = np.real(
            S + S.conj().T
        )
    
        return M, b

    def build_unitary(self, coeffs, dt):
    
        op = None
    
        for i, c in enumerate(coeffs):
    
            if abs(c) < 1e-12:
                continue
    
            term = c * self.basis.operator(i)
    
            if op is None:
                op = term
            else:
                op += term
    
        if op is None:
    
            return QuantumCircuit(self.num_qubits)
    
        qc = QuantumCircuit(self.num_qubits)
    
        qc.append(
            PauliEvolutionGate(
                op,
                time=dt,
                synthesis=LieTrotter(reps=self.trot)
            ),
            range(self.num_qubits)
        )
    
        return qc.decompose()