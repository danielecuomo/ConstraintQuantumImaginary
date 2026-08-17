import numpy as np
from qiskit.quantum_info import SparsePauliOp, Statevector, Pauli
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate

from cqite.ansatz.commutator_basis import CommutatorBasis

from itertools import combinations


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





def prepare_real_state(
    nqubits,
    target
):

    theta = np.arcsin(
        target
    )

    qc = QuantumCircuit(
        nqubits
    )

    for q in range(nqubits):

        qc.ry(
            theta,
            q
        )

    state = Statevector.from_instruction(
        qc
    )

    return qc, state





def zero_density_pauli(nqubits):

    rho = SparsePauliOp([""], coeffs=[1.0])

    for _ in range(nqubits):

        local = SparsePauliOp(
            ["I", "Z"],
            coeffs=[0.5, 0.5]
        )

        rho = local if rho.num_qubits == 0 else rho.tensor(local)

    return rho.simplify()


def conjugate_pauli_rotation(op, generator, theta):
    """
    Applica

        op -> exp(-i theta P) op exp(+i theta P)

    dove generator = P è un singolo termine Pauli.
    """

    P = generator.paulis[0]

    new_labels = []
    new_coeffs = []

    for coeff, Q in zip(op.coeffs, op.paulis):

        if P.commutes(Q):

            new_labels.append(Q.to_label())
            new_coeffs.append(coeff)

        else:

            product = SparsePauliOp(Q).compose(
                SparsePauliOp(P)
            ).simplify()
            
            label = product.paulis[0].to_label()
            phase = product.coeffs[0]
            
            new_labels.append(Q.to_label())
            new_coeffs.append(
                coeff * np.cos(2 * theta)
            )
            
            new_labels.append(label)
            new_coeffs.append(
                coeff * (-1j) * phase * np.sin(2 * theta)
            )

    return SparsePauliOp(
        new_labels,
        coeffs=new_coeffs
    ).simplify()




def density_from_rotations(
    nqubits,
    rotations,
    degree=None
):
    """
    Costruisce

        rho ~= Π_i (I + A_i)

    troncata al grado 'degree'.

    degree = 0:
        I

    degree = 1:
        I + Σ_i A_i

    degree = 2:
        I + Σ_i A_i + Σ_{i<j} A_i A_j

    ...

    degree >= m:
        espansione completa.
    """

    #
    # Costruisci i constraint vestiti
    #

    dressed = []

    for q in range(nqubits):
    
        label = ["I"] * nqubits
        label[q] = "Z"
    
        op = SparsePauliOp(
            ["".join(label)],
            coeffs=[1.0]
        )
    
        for rot in rotations:
    
            op = conjugate_pauli_rotation(
                op,
                rot["generator"],
                rot["theta"]
            ).simplify()
    
        dressed.append(op)

    m = len(dressed)

    if degree is None:
        degree = m

    degree = min(degree, m)

    #
    # Termine identità
    #

    rho = SparsePauliOp(
        "I" * nqubits,
        coeffs=[1.0]
    )

    #
    # Ordini 1..degree
    #

    for order in range(1, degree + 1):

        for subset in combinations(range(m), order):

            term = dressed[subset[0]]

            for idx in subset[1:]:

                term = (
                    term @ dressed[idx]
                ).simplify()

            rho = (rho + term).simplify()

    #
    # Normalizzazione come density operator
    #

    rho = rho / (2 ** nqubits)

    return rho.simplify()



def support_only(rho):
    """
    Sostituisce tutti i coefficienti con 1
    e rimuove eventuali duplicati.
    """

    labels = sorted(
        {
            p.to_label()
            for p in rho.paulis
        }
    )

    return SparsePauliOp(
        labels,
        coeffs=np.ones(len(labels))
    )