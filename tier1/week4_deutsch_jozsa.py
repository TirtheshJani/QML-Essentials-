"""Week 4 — Deutsch-Jozsa and oracles.

Codebook I.11-I.12. Working through:
  - phase oracles: U_f |x> = (-1)^{f(x)} |x>
  - the Deutsch-Jozsa circuit:  H^n -- U_f -- H^n -- measure
  - the verdict: P(all zeros) = 1 if f is constant, 0 if f is balanced
  - works in a single query for any n; classical worst case is 2^{n-1} + 1
"""

import numpy as np
import pennylane as qml


def oracle_constant_zero(_wires):
    """f(x) = 0 for all x. Phase oracle: identity (do nothing)."""
    pass


def oracle_constant_one(wires):
    """f(x) = 1 for all x. Phase oracle: global -1, realised by Z;X on any wire."""
    # Apply -I via X-Z-X-Z on a single wire (equivalent to global phase -1).
    w = wires[0]
    qml.PauliX(w)
    qml.PauliZ(w)
    qml.PauliX(w)
    qml.PauliZ(w)


def oracle_balanced_first_bit(wires):
    """f(x) = x_0. Phase oracle: Z on wire 0 -> (-1)^{x_0}."""
    qml.PauliZ(wires[0])


def oracle_balanced_parity(wires):
    """f(x) = x_0 XOR x_1 XOR ... Phase oracle: Z on every wire -> (-1)^{sum x_i}."""
    for w in wires:
        qml.PauliZ(w)


ORACLES = {
    "constant 0":               (oracle_constant_zero,        "constant"),
    "constant 1":               (oracle_constant_one,         "constant"),
    "balanced f(x)=x_0":        (oracle_balanced_first_bit,   "balanced"),
    "balanced f(x)=parity":     (oracle_balanced_parity,      "balanced"),
}


def deutsch_jozsa(oracle, n):
    dev = qml.device("default.qubit", wires=n)
    wires = list(range(n))

    @qml.qnode(dev)
    def circuit():
        for w in wires:
            qml.Hadamard(w)
        oracle(wires)
        for w in wires:
            qml.Hadamard(w)
        return qml.probs(wires=wires)

    return circuit()


def verdict(probs, tol=1e-9):
    p_all_zero = float(probs[0])
    if p_all_zero > 1 - tol:
        return "constant", p_all_zero
    if p_all_zero < tol:
        return "balanced", p_all_zero
    return "ambiguous", p_all_zero


def section(title):
    print("\n" + title)
    print("-" * len(title))


def main():
    section("1. Phase oracle sanity check on n=1: state after H -- U_f -- H")
    # For n=1 with f(x)=x_0 the algorithm distinguishes the two functions
    # (Deutsch's original algorithm). We just look at the probs.
    for name, (oracle, _) in ORACLES.items():
        if "parity" in name:
            continue
        probs = deutsch_jozsa(oracle, n=1)
        print(f"  {name:>22s}:  P(0)={probs[0]:.3f}   P(1)={probs[1]:.3f}")

    section("2. Deutsch-Jozsa for n = 2, 3, 4, 5: P(measure all zeros) per oracle")
    print(f"  {'oracle':>22s}     n=2     n=3     n=4     n=5    verdict (n=5)   truth")
    for name, (oracle, truth) in ORACLES.items():
        line = f"  {name:>22s}  "
        for n in (2, 3, 4, 5):
            probs = deutsch_jozsa(oracle, n=n)
            line += f"  {float(probs[0]):.3f}"
        probs5 = deutsch_jozsa(oracle, n=5)
        v, _ = verdict(probs5)
        line += f"     {v:>9s}     {truth}"
        print(line)

    section("3. Sampled outcomes (n=4, 1000 shots) for each oracle")
    n = 4

    def run_shots(oracle):
        dev = qml.device("default.qubit", wires=n, seed=0)

        @qml.set_shots(shots=1000)
        @qml.qnode(dev)
        def c():
            for w in range(n):
                qml.Hadamard(w)
            oracle(list(range(n)))
            for w in range(n):
                qml.Hadamard(w)
            return qml.sample(wires=range(n))

        return c()

    for name, (oracle, truth) in ORACLES.items():
        samples = run_shots(oracle)
        bitstrings = ["".join(str(int(b)) for b in row) for row in samples]
        all_zero = sum(1 for s in bitstrings if s == "0" * n)
        total = len(bitstrings)
        print(f"  {name:>22s}:  all-zero shots = {all_zero}/{total}   (constant ⇒ {total}, balanced ⇒ 0)")

    section("4. Query-count comparison (in theory)")
    print(f"  {'n':>3s}   {'classical worst case':>22s}   {'DJ quantum':>12s}")
    for n in (2, 5, 10, 20):
        classical_worst = 2 ** (n - 1) + 1
        print(f"  {n:>3d}   {classical_worst:>22d}   {1:>12d}")

    section("5. Inspect the n=3 DJ circuit for the parity oracle")
    n = 3
    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev)
    def dj_drawn():
        for w in range(n):
            qml.Hadamard(w)
        oracle_balanced_parity(list(range(n)))
        for w in range(n):
            qml.Hadamard(w)
        return qml.probs(wires=range(n))

    dj_drawn()
    print(qml.draw(dj_drawn)())


if __name__ == "__main__":
    main()
