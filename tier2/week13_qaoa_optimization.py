"""Week 13 — p-layer QAOA on MaxCut.

Tier 2 / 2B.2. Working through:
  - the QAOA ansatz: alternating cost (e^{-i gamma H_C}) and X-mixer
    (e^{-i beta H_M}, H_M = sum X_i) layers, p of each
  - cost-layer decomposition: each Z_i Z_j term -> CNOT - RZ(-gamma) - CNOT
  - mixer layer: independent RX(2 beta) on every qubit
  - outer-loop optimization with Adam + multiple random initialisations
    (the QAOA landscape has plenty of local minima)
  - approximation ratio rho = <H_C>(opt) / C* growing with p
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from tier2.utils.maxcut import (
    random_regular_graph,
    brute_force_maxcut,
    cost_hamiltonian,
)

N_NODES = 6
DEGREE = 3
SEED = 0

P_VALUES = (1, 2, 3)
N_RESTARTS = 10
ADAM_STEPS = 200
ADAM_LR = 0.05
TOL = 1e-6
MIN_ITERS_BEFORE_BREAK = 10


def section(title):
    print("\n" + title)
    print("-" * len(title))


def make_qaoa(graph, p):
    """Returns a QNode taking params of shape (p, 2) = [(gamma_l, beta_l)]_{l=1..p}.

    The cost layer e^{-i gamma H_C} is implemented edge-by-edge; the mixer
    layer e^{-i beta sum X_i} = prod_i RX(2 beta).
    """
    n = graph.number_of_nodes()
    H = cost_hamiltonian(graph)
    edges = list(graph.edges)
    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev, interface="autograd")
    def expectation(params):
        for w in range(n):
            qml.Hadamard(w)
        for layer in range(p):
            gamma, beta = params[layer]
            for (i, j) in edges:
                qml.CNOT(wires=[i, j])
                qml.RZ(-gamma, wires=j)
                qml.CNOT(wires=[i, j])
            for w in range(n):
                qml.RX(2 * beta, wires=w)
        return qml.expval(H)

    return expectation, n


def negative_expectation(expectation):
    """Adam minimises; we want to maximise <H_C>, so flip the sign."""
    def neg(params):
        return -expectation(params)
    return neg


def optimise_one_restart(neg_cost, p, seed):
    rng = np.random.default_rng(seed)
    # gamma in (0, pi), beta in (0, pi/2): the "fundamental" QAOA box.
    init = rng.uniform(low=[0.0, 0.0], high=[np.pi, np.pi / 2], size=(p, 2))
    params = pnp.array(init, requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=ADAM_LR)
    e_prev = None
    for k in range(ADAM_STEPS):
        params, e_now = opt.step_and_cost(neg_cost, params)
        e_now = float(e_now)
        if e_prev is not None and k > MIN_ITERS_BEFORE_BREAK and abs(e_now - e_prev) < TOL:
            break
        e_prev = e_now
    e_final = float(neg_cost(params))
    return -e_final, np.asarray(params)


def main():
    section("1. Graph and ground truth")
    g = random_regular_graph(N_NODES, DEGREE, seed=SEED)
    n = g.number_of_nodes()
    best, _ = brute_force_maxcut(g)
    print(f"  graph              : 3-regular, {n} nodes, {g.number_of_edges()} edges")
    print(f"  optimum cut C*     : {best}")
    print(f"  random-cut average : {g.number_of_edges() / 2}")

    section("2. Run QAOA at p in {1, 2, 3} with multiple restarts")
    print(f"  {'p':>2s}  {'best <H_C>':>11s}  {'rho':>6s}  {'restarts':>9s}  best (gamma_l, beta_l) per layer")
    results = {}
    for p in P_VALUES:
        cost_qnode, _ = make_qaoa(g, p)
        neg_cost = negative_expectation(cost_qnode)
        best_e, best_params = -np.inf, None
        for r in range(N_RESTARTS):
            e, params = optimise_one_restart(neg_cost, p, seed=100 * p + r)
            if e > best_e:
                best_e, best_params = e, params
        rho = best_e / best
        results[p] = (best_e, rho, best_params)
        param_str = ", ".join(
            f"({best_params[l, 0]:+.3f},{best_params[l, 1]:+.3f})"
            for l in range(p)
        )
        print(f"  {p:>2d}  {best_e:>11.5f}  {rho:>6.4f}  {N_RESTARTS:>9d}  {param_str}")

    section("3. Sample bitstrings from the best p=3 circuit")
    _, _, params3 = results[3]
    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev)
    def probs(params):
        for w in range(n):
            qml.Hadamard(w)
        for layer in range(3):
            gamma, beta = params[layer]
            for (i, j) in g.edges:
                qml.CNOT(wires=[i, j])
                qml.RZ(-gamma, wires=j)
                qml.CNOT(wires=[i, j])
            for w in range(n):
                qml.RX(2 * beta, wires=w)
        return qml.probs(wires=range(n))

    pvec = np.asarray(probs(pnp.array(params3, requires_grad=False)))
    top_idx = np.argsort(-pvec)[:6]
    print(f"  {'bitstring':>10s}  {'prob':>8s}  {'cut':>4s}  optimum?")
    for idx in top_idx:
        bits = tuple((idx >> (n - 1 - k)) & 1 for k in range(n))
        cut = sum(1 for (i, j) in g.edges if bits[i] != bits[j])
        is_opt = "yes" if cut == best else ""
        print(f"  {''.join(map(str, bits)):>10s}  {float(pvec[idx]):>8.4f}  {cut:>4d}  {is_opt}")
    p_opt = float(sum(
        pvec[idx] for idx in range(2 ** n)
        if sum(1 for (i, j) in g.edges
               if ((idx >> (n - 1 - i)) & 1) != ((idx >> (n - 1 - j)) & 1)) == best
    ))
    print(f"\n  total prob mass on optimal cuts: {p_opt:.4f}")

    section("Checkpoint assertions")
    rho_p1 = results[1][1]
    rho_p3 = results[3][1]
    # p=1 on 3-regular graphs has the FGG'14 lower bound rho >= ~0.692; we
    # comfortably clear it with restarts on this 6-node instance
    assert rho_p1 >= 0.69, f"p=1 approximation ratio too low: {rho_p1:.4f}"
    assert rho_p3 >= 0.85, f"p=3 approximation ratio below target: {rho_p3:.4f}"
    assert results[3][1] >= results[1][1] - 1e-6, "p=3 should not be worse than p=1"
    assert p_opt > 0.5, (
        f"best p=3 circuit puts only {p_opt:.3f} mass on optimal cuts"
    )
    print(f"  PASS: rho(p=1)={rho_p1:.4f}, rho(p=3)={rho_p3:.4f}; "
          f"P(optimal cut at p=3)={p_opt:.3f}.")


if __name__ == "__main__":
    main()
