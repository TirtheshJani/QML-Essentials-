"""Week 5 — Grover's search.

Codebook I.13. Working through:
  - phase oracle for an arbitrary marked bitstring (X-sandwich + multi-Z)
  - diffusion operator 2|s><s| - I as H^n X^n MCZ X^n H^n
  - the iteration recipe: amplitude grows like sin((2k+1) theta) with
    sin(theta) = sqrt(M/N), so optimal k ~ floor(pi/4 * sqrt(N/M))
  - plotting (textual) P(marked) vs k to see the oscillation
"""

import numpy as np
import pennylane as qml


def phase_oracle(marked, wires):
    """Flip the phase of every bitstring in `marked`.

    Each marked string is encoded MSB-first matching `wires` order.
    Implemented as: X on 0-positions, MultiControlledZ (via Hadamard
    sandwich + MultiControlledX), X back. Repeated per marked string.
    """
    for bits in marked:
        zeros = [w for w, b in zip(wires, bits) if b == 0]
        for w in zeros:
            qml.PauliX(w)
        qml.Hadamard(wires[-1])
        qml.MultiControlledX(wires=wires)
        qml.Hadamard(wires[-1])
        for w in zeros:
            qml.PauliX(w)


def diffusion(wires):
    """The Grover diffusion 2|s><s| - I = H^n (2|0><0| - I) H^n."""
    n = len(wires)
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


def grover(n, marked, k):
    """Run k Grover iterations on n qubits with the given marked bitstrings.
    Returns the full probability vector (length 2^n).
    """
    dev = qml.device("default.qubit", wires=n)
    wires = list(range(n))

    @qml.qnode(dev)
    def circuit():
        for w in wires:
            qml.Hadamard(w)
        for _ in range(k):
            phase_oracle(marked, wires)
            diffusion(wires)
        return qml.probs(wires=wires)

    return circuit()


def bits_of(x, n):
    return tuple(int(b) for b in format(x, f"0{n}b"))


def section(title):
    print("\n" + title)
    print("-" * len(title))


def main():
    n = 4
    N = 2 ** n
    target = 11  # 1011
    marked_one = [bits_of(target, n)]

    section(f"1. Single marked item (n={n}, target={target} = {format(target, f'0{n}b')})")
    print(f"  classical baseline: random guess succeeds with probability M/N = 1/{N} = {1/N:.4f}")
    print("\n  k    P(marked)    P(top-3 non-marked sum)    spectrum top-3")
    k_max = int(np.ceil(np.pi / 4 * np.sqrt(N))) + 4
    best_k, best_p = 0, 0.0
    for k in range(0, k_max + 1):
        probs = grover(n, marked_one, k)
        p_marked = float(probs[target])
        if p_marked > best_p:
            best_p, best_k = p_marked, k
        order = np.argsort(probs)[::-1]
        top3 = ", ".join(f"|{format(int(i), f'0{n}b')}>={probs[i]:.3f}" for i in order[:3])
        non_marked_top = sum(probs[i] for i in order[:3] if i != target)
        print(f"  {k:>2d}    {p_marked:.4f}        {non_marked_top:.4f}            {top3}")
    optimal = int(np.floor(np.pi / 4 * np.sqrt(N)))
    print(f"\n  formula optimum:       k* = floor(pi/4 * sqrt(N))            = {optimal}")
    print(f"  empirical best:        k  = {best_k}    P(marked) = {best_p:.4f}")

    section("2. Multiple marked items: optimal k* ~ floor(pi/4 * sqrt(N/M))")
    print(f"  {'M':>2s}    {'predicted k*':>12s}    {'best k empirical':>17s}    {'P(any marked) at best k':>24s}")
    rng = np.random.default_rng(0)
    all_strings = [bits_of(i, n) for i in range(N)]
    for M in (1, 2, 4, 6):
        marked_idx = rng.choice(N, size=M, replace=False)
        marked = [all_strings[i] for i in marked_idx]
        predicted = int(np.floor(np.pi / 4 * np.sqrt(N / M)))
        # Search only within the first amplification cycle so we compare
        # the formula's prediction to the *first* probability peak, not
        # subsequent revivals that the sin^2 oscillation also reaches.
        best_k, best_total = 0, 0.0
        for k in range(0, max(predicted + 2, 2)):
            probs = grover(n, marked, k)
            total = float(sum(probs[i] for i in marked_idx))
            if total > best_total:
                best_total, best_k = total, k
        print(f"  {M:>2d}    {predicted:>12d}    {best_k:>17d}    {best_total:>24.4f}")

    section("3. Quantum vs classical query count to find one of M items in N")
    print(f"  {'n':>3s}   {'N':>6s}   {'M':>3s}   {'classical avg':>14s}   {'Grover ~':>10s}")
    for n2 in (4, 6, 8, 10):
        N2 = 2 ** n2
        for M in (1, 4):
            classical = (N2 + 1) / (M + 1)  # expected unstructured search
            grover_iters = int(np.floor(np.pi / 4 * np.sqrt(N2 / M)))
            print(f"  {n2:>3d}   {N2:>6d}   {M:>3d}   {classical:>14.1f}   {grover_iters:>10d}")

    section("4. Inspect a single Grover iteration (n=3, target=101)")
    n3 = 3
    marked3 = [bits_of(0b101, n3)]
    dev3 = qml.device("default.qubit", wires=n3)

    @qml.qnode(dev3)
    def grover3_drawn():
        for w in range(n3):
            qml.Hadamard(w)
        phase_oracle(marked3, list(range(n3)))
        diffusion(list(range(n3)))
        return qml.probs(wires=range(n3))

    grover3_drawn()
    print(qml.draw(grover3_drawn)())


if __name__ == "__main__":
    main()
