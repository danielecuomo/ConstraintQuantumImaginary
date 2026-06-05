def run_qite(
    prop,
    circuit,
    H,
    Mz,
    dt,
    nsteps
):
    import numpy as np

    energies = []
    magnetizations = []

    state = None

    for _ in range(nsteps):

        circuit, state = prop.step(
            circuit,
            state,
            dt
        )

        E = prop.backend.expectation(
            circuit,
            H
        )

        energies.append(
            np.real(E)
        )

        M = prop.backend.expectation(
            circuit,
            Mz
        )

        magnetizations.append(
            np.real(M)
        )

    return {
        "energies": energies,
        "magnetizations": magnetizations,
        "circuit": circuit
    }

from qiskit import QuantumCircuit


def prepare_state(state_string):

    n = len(state_string)

    qc = QuantumCircuit(n)

    #
    # Qiskit usa little endian:
    # il carattere più a destra è il qubit 0
    #

    for q, symbol in enumerate(
        reversed(state_string)
    ):

        if symbol == "0":

            pass

        elif symbol == "1":

            qc.x(q)

        elif symbol == "+":

            qc.h(q)

        elif symbol == "-":

            qc.x(q)
            qc.h(q)

        else:

            raise ValueError(
                f"Unknown symbol '{symbol}'"
            )

    return qc