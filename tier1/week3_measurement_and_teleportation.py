"""Week 3 — Measurement and quantum teleportation.

Codebook I.8-I.10. Working through:
  - mid-circuit measurement and classical feedforward (qml.measure / qml.cond)
  - the standard teleportation circuit:
      qubit 0 (Alice's payload), qubits 1+2 = shared Bell pair
      Alice does CNOT(0,1) then H(0), measures both, sends m0,m1 to Bob
      Bob applies X^m1 then Z^m0 to qubit 2
  - verification: Bob's Bloch vector matches Alice's input for
    several test states (cardinal + a generic angle)
"""

import numpy as np
import pennylane as qml

dev = qml.device("default.qubit")


def prepare_payload(theta, phi):
    """Put an arbitrary single-qubit state on wire 0: cos(t/2)|0> + e^{i*phi} sin(t/2)|1>."""
    qml.RY(theta, wires=0)
    qml.RZ(phi, wires=0)


def teleport(theta, phi):
    """Run the full protocol and return Bob's Bloch vector (<X>, <Y>, <Z>) on wire 2."""

    @qml.qnode(dev)
    def circuit():
        # 1. Alice's payload on wire 0.
        prepare_payload(theta, phi)
        # 2. Shared Bell pair across wires 1 (Alice) and 2 (Bob).
        qml.Hadamard(wires=1)
        qml.CNOT(wires=[1, 2])
        # 3. Alice's Bell-basis measurement, written as CNOT + H then computational.
        qml.CNOT(wires=[0, 1])
        qml.Hadamard(wires=0)
        m0 = qml.measure(0)
        m1 = qml.measure(1)
        # 4. Bob's classical-controlled corrections.
        qml.cond(m1, qml.PauliX)(wires=2)
        qml.cond(m0, qml.PauliZ)(wires=2)
        return (
            qml.expval(qml.PauliX(2)),
            qml.expval(qml.PauliY(2)),
            qml.expval(qml.PauliZ(2)),
        )

    return circuit()


def expected_bloch(theta, phi):
    """Analytic Bloch vector for cos(t/2)|0> + e^{i*phi} sin(t/2)|1>."""
    return (
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    )


def section(title):
    print("\n" + title)
    print("-" * len(title))


def main():
    section("1. Mid-circuit measurement collapses the wire it acts on")
    # Prepare |+>, measure in Z, then ask for <Z> on the same wire. The
    # post-measurement state is |0> or |1>, so <Z> conditional on that
    # outcome is +/- 1 -- never the pre-measurement value of 0.
    dev1 = qml.device("default.qubit")

    @qml.qnode(dev1)
    def measure_then_z():
        qml.Hadamard(0)
        qml.measure(0)
        return qml.expval(qml.PauliZ(0))

    @qml.set_shots(shots=4000)
    @qml.qnode(qml.device("default.qubit", seed=0))
    def measure_then_sample():
        qml.Hadamard(0)
        m = qml.measure(0)
        return qml.sample(m)

    print(f"  <Z> after measuring |+> in Z (averaged):  {measure_then_z():+.3f}   (expect ~0)")
    samples = measure_then_sample()
    p1 = float(np.mean(samples))
    print(f"  P(measured 1) over 4000 shots:           {p1:.3f}        (expect ~0.5)")

    section("2. Teleport several test states; compare Bob's Bloch vector to Alice's")
    print(f"  {'state':>14s}    {'<X> got':>8s}  {'exp':>6s}     {'<Y> got':>8s}  {'exp':>6s}     {'<Z> got':>8s}  {'exp':>6s}")
    test_states = [
        ("|0>",                 0.0,             0.0),
        ("|1>",                 np.pi,           0.0),
        ("|+>",                 np.pi / 2,       0.0),
        ("|->",                 np.pi / 2,       np.pi),
        ("|+i>",                np.pi / 2,       np.pi / 2),
        ("RY(1.0)RZ(0.7)|0>",   1.0,             0.7),
    ]
    max_err = 0.0
    for name, theta, phi in test_states:
        got = teleport(theta, phi)
        exp = expected_bloch(theta, phi)
        err = max(abs(g - e) for g, e in zip(got, exp))
        max_err = max(max_err, err)
        print(f"  {name:>14s}    {got[0]:+.3f}   {exp[0]:+.3f}     {got[1]:+.3f}   {exp[1]:+.3f}     {got[2]:+.3f}   {exp[2]:+.3f}")
    print(f"\n  max component-wise error across all states: {max_err:.2e}")

    section("3. Drop the corrections: Bob's qubit is then maximally mixed")
    # Without the conditional X/Z, averaging over Alice's random outcomes
    # leaves Bob with the identity -- no information transmitted.
    @qml.qnode(dev)
    def teleport_no_correction(theta, phi):
        prepare_payload(theta, phi)
        qml.Hadamard(wires=1)
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[0, 1])
        qml.Hadamard(wires=0)
        qml.measure(0)
        qml.measure(1)
        return (
            qml.expval(qml.PauliX(2)),
            qml.expval(qml.PauliY(2)),
            qml.expval(qml.PauliZ(2)),
        )

    bob = teleport_no_correction(1.0, 0.7)
    print(f"  no-correction Bob's <X>,<Y>,<Z>: {bob[0]:+.3f}, {bob[1]:+.3f}, {bob[2]:+.3f}   (expect ~0)")

    section("4. Inspect the teleportation circuit")

    @qml.qnode(dev)
    def teleport_drawn():
        prepare_payload(1.0, 0.7)
        qml.Hadamard(wires=1)
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[0, 1])
        qml.Hadamard(wires=0)
        m0 = qml.measure(0)
        m1 = qml.measure(1)
        qml.cond(m1, qml.PauliX)(wires=2)
        qml.cond(m0, qml.PauliZ)(wires=2)
        return qml.expval(qml.PauliZ(2))

    teleport_drawn()
    print(qml.draw(teleport_drawn)())


if __name__ == "__main__":
    main()
