"""Week 12 — QAOA MaxCut: graph, cost Hamiltonian, brute-force optimum.

Tier 2 / 2B.1. Working through:
  - generating a small 3-regular random graph (6 nodes, 9 edges) with networkx
  - encoding MaxCut as the Ising Hamiltonian
       H_C = sum_{(i,j) in E} 1/2 (1 - Z_i Z_j)
    so that <x|H_C|x> = (#cut edges in partition x) for any basis state |x>
  - brute-force enumerating all 2^n bitstrings to fix the ground truth
  - sanity checks: H_C eigenvalues match brute-force counts; the equal
    superposition prepared by H^{otimes n} averages to |E|/2 cut edges
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pennylane as qml

from tier2.utils.maxcut import (
    random_regular_graph,
    brute_force_maxcut,
    cost_value,
    cost_hamiltonian,
)

N_NODES = 6
DEGREE = 3
SEED = 0


def section(title):
    print("\n" + title)
    print("-" * len(title))


def main():
    section("1. Graph: 3-regular random on 6 nodes")
    g = random_regular_graph(N_NODES, DEGREE, seed=SEED)
    n = g.number_of_nodes()
    print(f"  nodes              : {n}")
    print(f"  edges ({g.number_of_edges()}) : {sorted(g.edges)}")
    print(f"  degrees            : {[d for _, d in g.degree()]}")

    section("2. Brute-force MaxCut on all 2^n = 64 bitstrings")
    best, winners = brute_force_maxcut(g)
    print(f"  optimum cut value  : {best} of {g.number_of_edges()} edges")
    print(f"  # optimal partitions: {len(winners)}")
    for x in winners[:4]:
        print(f"    bitstring {x}  cut={cost_value(g, x)}")

    section("3. Cost Hamiltonian H_C = sum 0.5 (I - Z_i Z_j)")
    H = cost_hamiltonian(g)
    coeffs, ops = H.terms()
    print(f"  # Pauli terms      : {len(coeffs)}  (= 2 * |E| = {2 * g.number_of_edges()})")
    print(f"  identity coeff sum : {sum(float(c) for c, o in zip(coeffs, ops) if isinstance(o, qml.Identity)):+.4f}")
    print(f"  ZZ pairs:")
    for c, o in list(zip(coeffs, ops))[:6]:
        print(f"    {float(c):+.2f}   {o}")

    section("4. Eigenvalue check: <x|H_C|x> equals the cut count")
    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev)
    def basis_energy(bits):
        qml.BasisState(np.array(bits), wires=range(n))
        return qml.expval(H)

    print(f"  {'bitstring':>10s}  {'<H_C>':>8s}  {'brute-force':>12s}  match")
    mismatch = 0
    for x_int in (0, 5, 13, 21, 42, 63):
        bits = tuple((x_int >> (n - 1 - k)) & 1 for k in range(n))
        e = float(basis_energy(bits))
        bf = cost_value(g, bits)
        ok = abs(e - bf) < 1e-9
        mismatch += 0 if ok else 1
        print(f"  {''.join(map(str, bits)):>10s}  {e:>8.3f}  {bf:>12d}  {ok}")

    section("5. Equal superposition baseline (random-guess MaxCut)")

    @qml.qnode(dev)
    def uniform_energy():
        for w in range(n):
            qml.Hadamard(w)
        return qml.expval(H)

    e_uniform = float(uniform_energy())
    expected = g.number_of_edges() / 2
    print(f"  <H_C> on |+>^n     : {e_uniform:.4f}")
    print(f"  expected |E|/2     : {expected:.4f}")
    print( "  -> a uniform random cut leaves ~half the edges cut on a random graph")

    section("Checkpoint assertions")
    assert mismatch == 0, "H_C eigenvalues did not match brute-force cut counts"
    assert abs(e_uniform - expected) < 1e-9, (
        f"uniform baseline off: {e_uniform} vs {expected}"
    )
    # the cost Hamiltonian's largest eigenvalue equals best cut
    mat = qml.matrix(H, wire_order=range(n))
    assert abs(float(np.linalg.eigvalsh(mat)[-1]) - best) < 1e-9, (
        "max eigenvalue of H_C should equal MaxCut optimum"
    )
    print(f"  PASS: H_C diagonal in computational basis equals cut counts; "
          f"|+>^n averages to |E|/2; max eigenvalue = MaxCut = {best}.")


if __name__ == "__main__":
    main()
