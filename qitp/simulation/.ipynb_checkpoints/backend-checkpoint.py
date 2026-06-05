from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
from qiskit.primitives import StatevectorEstimator
from qiskit_aer.primitives import EstimatorV2

import numpy as np


class SimulatorBackend:

    def __init__(self):

        self.backend = AerSimulator(method="statevector")
        self.estimator = EstimatorV2()

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
    