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
        
        self.compiled_products = [
            [None] * self.m
            for _ in range(self.m)
        ]
    
        self.observable_cache = {}
        self.observable_labels = set()
    
        for i in range(self.m):

            for j in range(self.m):
        
                op = basis.product(i, j).simplify()

                self.products[i][j] = op
                
                self.compiled_products[i][j] = (
                    tuple(p.to_label() for p in op.paulis),
                    op.coeffs
                )
        
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
    
        basis = self.basis
        cache = self.observable_cache
    
        g_products = []
    
        for i in range(self.m):
    
            op = (G @ basis.operator(i)).simplify()
    
            compiled = (
                tuple(p.to_label() for p in op.paulis),
                op.coeffs
            )
    
            g_products.append(compiled)
    
            labels.update(compiled[0])
    
        observables = {}
    
        for label in labels:
    
            obs = cache.get(label)
    
            if obs is None:
    
                obs = SparsePauliOp(
                    label,
                    coeffs=[1.0]
                )
    
                cache[label] = obs
    
            observables[label] = obs
    
        return observables, g_products

    def measure_cache(
        self,
        circuit,
        backend,
        observables,
        state=None
    ):
    
        labels = list(observables.keys())
    
        ops = [
            observables[label]
            for label in labels
        ]
    
        values = backend.expectations(
            circuit,
            ops,
            state
        )
    
        return dict(zip(labels, values))

    def compute_S(self, measured):
    
        m = self.m
        compiled = self.compiled_products
        get = measured.__getitem__
    
        S = np.empty((m, m), dtype=complex)
    
        for i in range(m):
    
            row = compiled[i]
    
            for j in range(m):
    
                labels, coeffs = row[j]
    
                value = 0.0j
    
                for label, coeff in zip(labels, coeffs):
                    value += coeff * get(label)
    
                S[i, j] = value
    
        return S

    def compute_b(
        self,
        measured,
        g_products
    ):
    
        m = self.m
        get = measured.__getitem__
    
        b = np.empty(m)
    
        for i in range(m):
    
            labels, coeffs = g_products[i]
    
            value = 0.0j
    
            for label, coeff in zip(labels, coeffs):
                value += coeff * get(label)
    
            b[i] = -2.0 * value.imag
    
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
    
        op = self._build_operator(coeffs)
    
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

    def _build_operator(self, coeffs):

        op = None
    
        for i, c in enumerate(coeffs):
    
            if abs(c) < 1e-12:
                continue
    
            term = c * self.basis.operator(i)
    
            if op is None:
                op = term
            else:
                op += term
    
        return op

    
    
    def build_operator(self, coeffs, dt):
        from scipy.linalg import expm
        from qiskit.quantum_info import Operator
    
        op = self._build_operator(coeffs)
    
        if op is None:
            return Operator(
                np.eye(
                    2**self.num_qubits,
                    dtype=complex
                )
            )
    
        return Operator(
            expm(
                -1j * dt * op.to_matrix()
            )
        )