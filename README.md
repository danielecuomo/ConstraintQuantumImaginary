# Constrained Quantum Imaginary Time Evolution

This repository contains a research implementation of constrained quantum imaginary time evolution (CQITE). It advances quantum states toward low-energy states while enforcing expectation-value constraints through unitary updates.

The code uses Qiskit and Aer for Pauli-operator manipulation and statevector simulation. The notebooks provide validation cases and comparisons with constrained imaginary-time evolution results for small spin systems and graph-based density-functional examples.

## Features

- Construct constraint rotations and prepare corresponding initial states.
- Build Pauli-operator bases from seed operators and a Hamiltonian.
- Estimate constrained imaginary-time updates using a unitary approximation.
- Simulate expectation values with a Qiskit Aer statevector backend.
- Explore validation and reference comparisons in Jupyter notebooks.

## Repository layout

```text
cqite/      CQITE implementation
examples/   Validation and comparison notebooks, with selected figures
```

## Installation

Use Python 3 and install the project dependencies from the repository root:

```bash
git clone https://github.com/danielecuomo/ConstraintQuantumImaginary.git
cd ConstraintQuantumImaginary
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Examples

Launch JupyterLab from the repository root:

```bash
jupyter lab examples
```

The notebooks include:

- `cqite_validation_xxzzy.ipynb` and `cqite_validation_zzx.ipynb`: validation examples and evolution plots.
- `cqite_penz_leeuwen_corrected_full_comparison.ipynb`: comparison of CQITE with constrained imaginary-time evolution for a triangular graph.
- `penz_van_leeuwen_CQITE.ipynb`: evaluates CQITE against the analytic constrained-search functional.
- `penz_van_leeuwen_figure_6.ipynb`: plots the two-particle triangular-domain functional and its regions.

Some notebooks run multiple simulations and may take time to complete. Their parameter cells control the grid density and number of evolution steps.

## Citation

The graph-based density-functional examples use results from:

> Markus Penz and Robert van Leeuwen, “Density-functional theory on graphs,” *Journal of Chemical Physics* 155, 244111 (2021). [https://doi.org/10.1063/5.0074249](https://doi.org/10.1063/5.0074249)

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for the full license text.
