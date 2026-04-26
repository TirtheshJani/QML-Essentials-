"""H2 helpers shared across weeks 9-11.

PennyLane qchem coordinates are in bohr by default; 1 bohr = 0.529177 A.
"""

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

BOHR_PER_ANGSTROM = 1.0 / 0.529177210903


def h2_hamiltonian(r_bohr):
    """Return (H, n_qubits) for H2 with the two nuclei placed on the z-axis
    at +/- r_bohr / 2."""
    coords = pnp.array(
        [0.0, 0.0, -r_bohr / 2, 0.0, 0.0, r_bohr / 2], requires_grad=False
    )
    H, n = qml.qchem.molecular_hamiltonian(["H", "H"], coords)
    return H, n


def fci_energy(H, n_qubits):
    """Exact ground-state energy by direct diagonalization of H. Cheap at n=4."""
    mat = qml.matrix(H, wire_order=range(n_qubits))
    return float(np.linalg.eigvalsh(mat)[0])


def hf_energy(H, n_qubits, n_electrons=2):
    """Energy of the Hartree-Fock reference |1...1 0...0> on the H qubits."""
    hf = qml.qchem.hf_state(n_electrons, n_qubits)
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit():
        qml.BasisState(hf, wires=range(n_qubits))
        return qml.expval(H)

    return float(circuit())
