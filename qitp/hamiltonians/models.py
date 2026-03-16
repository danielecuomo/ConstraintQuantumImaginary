def heisenberg_chain(n, J=1.0):

    paulis = []
    coeffs = []

    for i in range(n - 1):
        paulis += ["XX", "YY", "ZZ"]
        coeffs += [J, J, J]

    return paulis, coeffs
