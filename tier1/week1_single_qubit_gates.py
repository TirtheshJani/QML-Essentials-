"""Week 1 — Single-qubit gates and the Bloch sphere.

Codebook I.1–I.4. Working through:
  - state preparation: |0>, |1>, |+>, |->, |i>
  - Pauli gates (X, Y, Z) and Hadamard
  - parametrized rotations (RX, RY, RZ)
  - reading the Bloch vector via <X>, <Y>, <Z>
"""

import numpy as np
import pennylane as qml

dev = qml.device("default.qubit", wires=1)


@qml.qnode(dev)
def bloch_vector(state_prep):
    state_prep()
    return qml.expval(qml.PauliX(0)), qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))


@qml.qnode(dev)
def measure_z_after(state_prep, gate):
    state_prep()
    gate()
    return qml.expval(qml.PauliZ(0))


def fmt_bloch(name, vec):
    x, y, z = vec
    print(f"  {name:>6s}:  <X>={x:+.3f}  <Y>={y:+.3f}  <Z>={z:+.3f}   |r|={np.linalg.norm(vec):.3f}")


def section(title):
    print("\n" + title)
    print("-" * len(title))


def main():
    section("1. Bloch coordinates of the cardinal states")
    states = {
        "|0>":  lambda: None,
        "|1>":  lambda: qml.PauliX(0),
        "|+>":  lambda: qml.Hadamard(0),
        "|->":  lambda: (qml.PauliX(0), qml.Hadamard(0)),
        "|+i>": lambda: (qml.Hadamard(0), qml.S(0)),
    }
    for name, prep in states.items():
        fmt_bloch(name, bloch_vector(prep))

    section("2. Pauli gates flip the expected axes")
    # Start from |+> (on +x axis); Z should flip it to |-> (-x axis).
    fmt_bloch("|+>",     bloch_vector(lambda: qml.Hadamard(0)))
    fmt_bloch("Z|+>",    bloch_vector(lambda: (qml.Hadamard(0), qml.PauliZ(0))))
    # X on |0> -> |1>: <Z> goes +1 -> -1.
    print(f"  <Z> on |0>:  {measure_z_after(lambda: None, lambda: None):+.3f}")
    print(f"  <Z> on X|0>: {measure_z_after(lambda: None, lambda: qml.PauliX(0)):+.3f}")

    section("3. RY(theta) sweeps |0> toward |1> through the xz-plane")
    print("  theta(deg)    <Z>      P(0)     P(1)")
    for theta_deg in (0, 30, 45, 60, 90, 120, 180):
        theta = np.deg2rad(theta_deg)

        @qml.qnode(dev)
        def circuit(t=theta):
            qml.RY(t, wires=0)
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=0)

        z_exp, probs = circuit()
        # P(0) = (1 + <Z>) / 2 — sanity check the textbook identity.
        print(f"  {theta_deg:>6d}     {z_exp:+.3f}    {probs[0]:.3f}    {probs[1]:.3f}")

    section("4. Hadamard is its own inverse: H H |0> == |0>")
    fmt_bloch("HH|0>", bloch_vector(lambda: (qml.Hadamard(0), qml.Hadamard(0))))

    section("5. Inspect the circuit drawer on a small example")

    @qml.qnode(dev)
    def demo():
        qml.Hadamard(0)
        qml.RZ(np.pi / 4, wires=0)
        qml.RX(np.pi / 3, wires=0)
        return qml.expval(qml.PauliZ(0))

    demo()
    print(qml.draw(demo)())


if __name__ == "__main__":
    main()
