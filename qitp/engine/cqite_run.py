def run_qite(
    prop,
    circuit,
    H,
    Mz,
    dt,
    nsteps,
    state=None
):

    import numpy as np
    from tqdm.notebook import tqdm

    energies = []
    magnetizations = []

    pbar = tqdm(total=nsteps, desc="QITE")

    for step in range(nsteps):

        circuit, state = prop.step(
            circuit,
            state,
            dt
        )

        E = np.real(
            prop.backend.expectation(
                circuit,
                H,
                state
            )
        )

        if isinstance(Mz, (list, tuple)):
            
            M = np.real(
                prop.backend.expectations(
                    circuit,
                    Mz,
                    state
                )
            )
        
        else:
                
            M = np.real(
                prop.backend.expectation(
                    circuit,
                    Mz,
                    state
                )
            )

        energies.append(E)
        magnetizations.append(M)

        pbar.update(1)
        pbar.set_postfix_str(
            f"step {step + 1}/{nsteps}"
        )

    pbar.close()

    return {
        "energies": energies,
        "magnetizations": magnetizations,
        "circuit": circuit,
    }