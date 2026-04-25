"""Week 8 — Tier 1 checkpoint: Bell states + Grover, from scratch.

No reference to weeks 2 or 5. Constructions rebuilt from memory and
verified by independent properties (Bell-state Pauli fingerprints;
Grover amplitude curve hitting the predicted peak).

Pass criteria, all enforced by assertions at the bottom:
  - all four Bell states have the textbook (<X@X>, <Y@Y>, <Z@Z>) signature
  - Grover on n=4 with one marked item peaks at k=3 with P > 0.96
  - Grover on n=4 with M=4 matched marked items hits P = 1.0 at k=1
"""

import numpy as np
import pennylane as qml


# ----- Bell states ---------------------------------------------------------

BELL_KINDS = ("phi+", "phi-", "psi+", "psi-")

# Textbook signatures (XX, YY, ZZ).
BELL_SIGNATURES = {
    "phi+": (+1, -1, +1),
    "phi-": (-1, +1, +1),
    "psi+": (+1, +1, -1),
    "psi-": (-1, -1, -1),
}


def make_bell(kind):
    """Phi+ from H + CNOT; the other three from X/Z dressing on the input qubits."""
    if kind in ("psi+", "psi-"):
        qml.PauliX(1)         # flips Phi -> Psi (swaps which qubits agree)
    qml.Hadamard(0)
    qml.CNOT(wires=[0, 1])
    if kind in ("phi-", "psi-"):
        qml.PauliZ(0)         # flips +-> - (sign on the |11> branch)


def bell_signature(kind):
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def circuit():
        make_bell(kind)
        return (
            qml.expval(qml.PauliX(0) @ qml.PauliX(1)),
            qml.expval(qml.PauliY(0) @ qml.PauliY(1)),
            qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)),
        )

    xx, yy, zz = circuit()
    return float(xx), float(yy), float(zz)


# ----- Grover --------------------------------------------------------------


def phase_oracle(marked_bitstrings, wires):
    """Flip the phase of each marked bitstring. MSB first in `wires`."""
    for bits in marked_bitstrings:
        zeros = [w for w, b in zip(wires, bits) if b == 0]
        for w in zeros:
            qml.PauliX(w)
        qml.Hadamard(wires[-1])
        qml.MultiControlledX(wires=wires)
        qml.Hadamard(wires[-1])
        for w in zeros:
            qml.PauliX(w)


def diffusion(wires):
    """2|s><s| - I = H^n X^n MCZ X^n H^n."""
    for w in wires:
        qml.Hadamard(w)
    for w in wires:
        qml.PauliX(w)
    qml.Hadamard(wires[-1])
    qml.MultiControlledX(wires=wires)
    qml.Hadamard(wires[-1])
    for w in wires:
        qml.PauliX(w)
    for w in wires:
        qml.Hadamard(w)


def grover(n, marked_bitstrings, k):
    dev = qml.device("default.qubit", wires=n)
    wires = list(range(n))

    @qml.qnode(dev)
    def circuit():
        for w in wires:
            qml.Hadamard(w)
        for _ in range(k):
            phase_oracle(marked_bitstrings, wires)
            diffusion(wires)
        return qml.probs(wires=wires)

    return circuit()


def bits(x, n):
    return tuple(int(c) for c in format(x, f"0{n}b"))


# ----- Run and verify ------------------------------------------------------


def section(title):
    print("\n" + title)
    print("-" * len(title))


def main():
    section("Bell states: Pauli correlation fingerprints")
    print(f"  {'state':>5s}    {'<X@X>':>6s}  {'<Y@Y>':>6s}  {'<Z@Z>':>6s}    {'expected':>14s}    pass")
    bell_pass = True
    for kind in BELL_KINDS:
        sig = bell_signature(kind)
        expected = BELL_SIGNATURES[kind]
        ok = all(abs(g - e) < 1e-10 for g, e in zip(sig, expected))
        bell_pass &= ok
        print(f"  {kind:>5s}    {sig[0]:+.3f}  {sig[1]:+.3f}  {sig[2]:+.3f}    {expected!s:>14s}    {ok}")

    section("Grover n=4: amplitude curve for target = 1011")
    n = 4
    target = 0b1011
    marked = [bits(target, n)]
    print(f"  {'k':>2s}    P(target)")
    p_at_k = []
    for k in range(0, 7):
        probs = grover(n, marked, k)
        p = float(probs[target])
        p_at_k.append(p)
        print(f"  {k:>2d}    {p:.4f}")
    k_pred = int(np.floor(np.pi / 4 * np.sqrt(2 ** n)))
    peak_p = p_at_k[k_pred]
    print(f"\n  predicted optimum: k* = {k_pred}    P at k* = {peak_p:.4f}")

    section("Grover n=4: M=4 matched marked items must hit P = 1 at k=1")
    n = 4
    # Choose any 4 distinct strings; M=4 in N=16 -> sin(theta) = 1/2,
    # so one iteration rotates exactly into the marked subspace.
    marked4 = [bits(x, n) for x in (1, 5, 9, 13)]
    probs = grover(n, marked4, 1)
    p_total = sum(float(probs[x]) for x in (1, 5, 9, 13))
    print(f"  P(any of {{1,5,9,13}}) at k=1: {p_total:.6f}")

    section("Checkpoint assertions")
    assert bell_pass, "Bell state signatures wrong"
    assert peak_p > 0.96, f"Grover peak too low at k={k_pred}: {peak_p}"
    assert abs(p_total - 1.0) < 1e-10, f"Grover M=4 must give P=1 at k=1, got {p_total}"
    print("  PASS: Bell signatures + Grover single + Grover M=4 all green.")


if __name__ == "__main__":
    main()
