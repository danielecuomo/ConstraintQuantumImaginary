def cqite_cycle(
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
        "magnetizations": magnetizations[:-1],
        "circuit": circuit,
        "state": state
    }



from cqite.engine.prepare_state import constraint_rotation, prepare_state, density_from_rotations, support_only
from cqite.ansatz.commutator_basis import CommutatorBasis
from cqite.engine.unitary_approximation import UnitaryApproximation
from qiskit.quantum_info import Statevector, SparsePauliOp
from cqite.engine.propagation import QITEPropagation

def cqite(
    H,
    C,
    target,
    degree,
    depth,
    trot,
    dt,
    nsteps,
    backend
):
    #
    # Step 1
    #

    rotations = [
        constraint_rotation(
            c,
            target,
        )
        for c in C
    ]

    #
    # Step 2
    #

    rho = density_from_rotations(
        H.num_qubits,
        rotations,
        degree=degree,
    )

    seeds = support_only(rho)

    basis = CommutatorBasis.from_seed(
        hamiltonian=H,
        seeds=seeds,
        depth=depth,
    )

    #
    # Step 3
    #

    approx = UnitaryApproximation(
        basis,
        trot=trot,
    )

    #
    # Step 4
    #

    prop = QITEPropagation(
        H,
        C,
        backend,
        approx,
    )

    #
    # Step 5
    #

    circuit = prepare_state(
        H.num_qubits,
        rotations,
    )
    
    circuit = circuit.decompose()

    #
    # Step 6
    #

    psi = Statevector.from_instruction(
        circuit
    )

    #
    # Step 7
    #

    results = cqite_cycle(
        prop,
        None,
        H,
        C,
        dt,
        nsteps,
        state=psi,
    )

    return results