import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate


from qiskit.quantum_info import SparsePauliOp


def anticommuting_generator(
    constraint
):

    label = (
        constraint
        .paulis[0]
        .to_label()
    )

    chars = list(label)

    for i, p in enumerate(chars):

        if p == "I":
            continue

        if p == "X":

            chars[i] = "Y"

        elif p == "Y":

            chars[i] = "X"

        elif p == "Z":

            chars[i] = "X"

        break

    return SparsePauliOp(
        "".join(chars),
        coeffs=[1.0]
    )

import numpy as np


def expectation_zero_state(
    operator
):

    value = 0.0

    for pauli, coeff in zip(
        operator.paulis,
        operator.coeffs
    ):

        label = pauli.to_label()

        if all(
            c in ("I", "Z")
            for c in label
        ):

            value += np.real(coeff)

    return value

def calibration(
    constraint,
    generator
):

    a = expectation_zero_state(
        constraint
    )

    iAP = (
        1j
        * (generator @ constraint)
    ).simplify()

    b = expectation_zero_state(
        iAP
    )

    return a, b


def target_to_angle(
    target,
    a,
    b
):

    R = np.sqrt(
        a*a + b*b
    )

    if R < 1e-12:

        raise ValueError(
            "Constraint not controllable"
        )

    phi = np.arctan2(
        b,
        a
    )

    theta = 0.5 * (
        phi
        + np.arccos(
            target / R
        )
    )

    return theta

def constraint_rotation(
    constraint,
    target
):

    generator = (
        anticommuting_generator(
            constraint
        )
    )

    a, b = calibration(
        constraint,
        generator
    )

    theta = target_to_angle(
        target,
        a,
        b
    )

    return {
        "constraint":
            constraint,
        "generator":
            generator,
        "theta":
            theta,
        "a":
            a,
        "b":
            b
    }
    
from qiskit import QuantumCircuit
from qiskit.circuit.library import (
    PauliEvolutionGate
)


def prepare_state(
    nqubits,
    rotations
):

    qc = QuantumCircuit(
        nqubits
    )

    for rot in rotations:

        qc.append(
            PauliEvolutionGate(
                rot["generator"],
                time=rot["theta"]
            ),
            range(nqubits)
        )

    return qc