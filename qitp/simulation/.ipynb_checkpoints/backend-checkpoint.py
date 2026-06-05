from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
from qiskit.primitives import StatevectorEstimator

import numpy as np


class SimulatorBackend:

    def __init__(self):

        self.backend = AerSimulator(method="statevector")
        self.estimator = StatevectorEstimator()

    def run(self, circuit):

        return Statevector.from_instruction(circuit)

    def expectation(
        self,
        circuit,
        operator
    ):

        result = (
            self.estimator
            .run([(circuit, operator)])
            .result()
        )

        return result[0].data.evs

    def expectations(
        self,
        circuit,
        observables
    ):
    
        result = (
            self.estimator
            .run(
                [(circuit, observables)]
            )
            .result()
        )
    
        values = np.asarray(
            result[0].data.evs
        ).reshape(-1)
    
        return values
    