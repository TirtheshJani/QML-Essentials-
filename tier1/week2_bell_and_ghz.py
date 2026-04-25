"""Week 2 — Multi-qubit, Bell and GHZ states.

Codebook I.5-I.7. Working through:
  - the 2-qubit computational basis and tensor products
  - the four Bell states from H + CNOT (+ X/Z dressing)
  - <Z@Z> and <X@X> as entanglement witnesses
  - the single-qubit reduced state of a Bell pair is maximally mixed
  - GHZ_n: n-way perfectly correlated measurement outcomes
"""

import numpy as np
import pennylane as qml

dev2 = qml.device("default.qubit", wires=2)
dev3 = qml.device("default.qubit", wires=3)


def bell_prep(kind):
    """Prepare one of the four Bell states. kind in {'phi+','phi-','psi+','psi-'}."""
    if kind in ("psi+", "psi-"):
        qml.PauliX(1)
    qml.Hadamard(0)
    qml.CNOT(wires=[0, 1])
    if kind in ("phi-", "psi-"):
        qml.PauliZ(0)


@qml.qnode(dev2)
def bell_state(kind):
    bell_prep(kind)
    return qml.state()


@qml.qnode(dev2)
def bell_correlations(kind):
    bell_prep(kind)
    return (
        qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)),
        qml.expval(qml.PauliX(0) @ qml.PauliX(1)),
        qml.expval(qml.PauliY(0) @ qml.PauliY(1)),
    )


@qml.qnode(dev2)
def bell_marginal_probs(kind, wire):
    bell_prep(kind)
    return qml.probs(wires=wire)


@qml.qnode(dev3)
def ghz_state():
    qml.Hadamard(0)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    return qml.state()


@qml.qnode(dev3)
def ghz_probs():
    qml.Hadamard(0)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    return qml.probs(wires=[0, 1, 2])


def section(title):
    print("\n" + title)
    print("-" * len(title))


def fmt_state(vec):
    labels = ["|00>", "|01>", "|10>", "|11>"]
    parts = []
    for amp, label in zip(vec, labels):
        if abs(amp) < 1e-9:
            continue
        parts.append(f"({amp.real:+.3f}{amp.imag:+.3f}j) {label}")
    return "  +  ".join(parts) if parts else "0"


def main():
    section("1. The four Bell states")
    print("  expected: phi+ = (|00>+|11>)/sqrt(2),  psi+ = (|01>+|10>)/sqrt(2), etc.")
    for kind in ("phi+", "phi-", "psi+", "psi-"):
        print(f"  {kind:>5s}:  {fmt_state(bell_state(kind))}")

    section("2. Two-qubit Pauli correlations identify each Bell state")
    print(f"  {'state':>5s}   <Z@Z>    <X@X>    <Y@Y>")
    for kind in ("phi+", "phi-", "psi+", "psi-"):
        zz, xx, yy = bell_correlations(kind)
        print(f"  {kind:>5s}   {zz:+.3f}   {xx:+.3f}   {yy:+.3f}")

    section("3. Reduced state of one qubit in a Bell pair is maximally mixed")
    # If we ignore qubit 1, the marginal on qubit 0 should be (0.5, 0.5) -- no
    # local information survives. That is the operational signature of
    # maximal entanglement.
    for kind in ("phi+", "psi-"):
        p0 = bell_marginal_probs(kind, 0)
        p1 = bell_marginal_probs(kind, 1)
        print(f"  {kind}:  P(q0)={p0[0]:.3f},{p0[1]:.3f}   P(q1)={p1[0]:.3f},{p1[1]:.3f}")

    section("4. GHZ_3 amplitude vector and joint probabilities")
    psi = ghz_state()
    nonzero = [(format(i, "03b"), amp) for i, amp in enumerate(psi) if abs(amp) > 1e-9]
    for bits, amp in nonzero:
        print(f"  amp(|{bits}>) = {amp.real:+.3f}{amp.imag:+.3f}j")
    probs = ghz_probs()
    print("  joint P:", {format(i, "03b"): round(float(p), 3) for i, p in enumerate(probs) if p > 1e-9})

    section("5. GHZ_3 sampled outcomes are always all-0 or all-1")

    @qml.set_shots(shots=2000)
    @qml.qnode(qml.device("default.qubit", wires=3, seed=0))
    def ghz_shots():
        qml.Hadamard(0)
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        return qml.sample(wires=[0, 1, 2])

    samples = ghz_shots()
    bitstrings = ["".join(str(b) for b in row) for row in samples]
    counts = {}
    for s in bitstrings:
        counts[s] = counts.get(s, 0) + 1
    total = sum(counts.values())
    print(f"  2000 shots ->  {dict(sorted(counts.items()))}")
    forbidden = sum(c for s, c in counts.items() if s not in ("000", "111"))
    print(f"  shots outside {{000, 111}}: {forbidden} / {total}  (must be 0)")

    section("6. Inspect the GHZ circuit")

    @qml.qnode(dev3)
    def ghz_drawn():
        qml.Hadamard(0)
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        return [qml.expval(qml.PauliZ(i)) for i in range(3)]

    ghz_drawn()
    print(qml.draw(ghz_drawn)())


if __name__ == "__main__":
    main()
