def run_qite(
    prop,
    circuit,
    H,
    Mz,
    dt,
    nsteps,
    state = None
):
    import numpy as np

    energies = []
    magnetizations = []

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