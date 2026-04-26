"""MaxCut helpers shared across weeks 12-14.

MaxCut on graph G = (V, E): partition V into two sets to maximize the number
of edges with one endpoint in each set. Encoded on |V| qubits as

    H_C = 1/2 * sum_{(i,j) in E} (1 - Z_i Z_j),

with the partition read off from a computational-basis bitstring.
"""

import networkx as nx
import numpy as np
import pennylane as qml


def random_regular_graph(n, d=3, seed=0):
    """A connected d-regular graph on n nodes — small, non-trivial, repeatable."""
    return nx.random_regular_graph(d=d, n=n, seed=seed)


def cost_value(graph, bitstring):
    """Number of cut edges for an assignment given as a tuple/list of 0/1, MSB-first."""
    return sum(1 for i, j in graph.edges if bitstring[i] != bitstring[j])


def brute_force_maxcut(graph):
    """Return (best_cut, [bitstrings_achieving_it])."""
    n = graph.number_of_nodes()
    best = -1
    winners = []
    for x in range(2 ** n):
        bits = tuple((x >> (n - 1 - k)) & 1 for k in range(n))
        c = cost_value(graph, bits)
        if c > best:
            best = c
            winners = [bits]
        elif c == best:
            winners.append(bits)
    return best, winners


def cost_hamiltonian(graph):
    """H_C = sum_{(i,j) in E} 0.5 * (I - Z_i Z_j). Eigenvalue on a basis state
    equals the number of cut edges."""
    coeffs, ops = [], []
    for i, j in graph.edges:
        coeffs.append(0.5)
        ops.append(qml.Identity(i))
        coeffs.append(-0.5)
        ops.append(qml.PauliZ(i) @ qml.PauliZ(j))
    return qml.Hamiltonian(coeffs, ops)


def approximation_ratio(achieved, optimum):
    return achieved / optimum if optimum > 0 else 1.0
