"""Week 6 — Quantum Fourier Transform.

Codebook I.14-I.15. Working through:
  - hand-rolling the QFT: H on each wire + controlled phase gates +
    final swap to reverse wire order
  - sanity-checks: QFT|0...0> = uniform superposition; QFT^-1 . QFT = I
  - matching the analytic formula |x> -> (1/sqrt(N)) sum_y e^{2 pi i x y / N} |y>
  - the Shor-relevant property: QFT of a state periodic with period r
    concentrates amplitude on multiples of N/r
"""

import numpy as np
import pennylane as qml


def qft_manual(wires):
    """Standard textbook QFT: H + controlled phases per qubit, then reverse-order swaps."""
    n = len(wires)
    for j in range(n):
        qml.Hadamard(wires[j])
        for k in range(j + 1, n):
            angle = np.pi / (2 ** (k - j))
            qml.ctrl(qml.PhaseShift, control=wires[k])(angle, wires=wires[j])
    for j in range(n // 2):
        qml.SWAP(wires=[wires[j], wires[n - 1 - j]])


def basis_prep(x, wires):
    """Prepare the computational basis state |x> with x encoded MSB-first on `wires`."""
    n = len(wires)
    bits = format(x, f"0{n}b")
    for w, b in zip(wires, bits):
        if b == "1":
            qml.PauliX(w)


def analytic_qft_row(x, n):
    """The analytic QFT amplitude vector for input |x>: y-th entry = e^{2 pi i x y / N} / sqrt(N)."""
    N = 2 ** n
    return np.array([np.exp(2j * np.pi * x * y / N) for y in range(N)]) / np.sqrt(N)


def section(title):
    print("\n" + title)
    print("-" * len(title))


def main():
    n = 3
    N = 2 ** n

    section(f"1. Hand-rolled QFT vs qml.QFT vs analytic formula (n={n})")
    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev)
    def manual(x):
        basis_prep(x, list(range(n)))
        qft_manual(list(range(n)))
        return qml.state()

    @qml.qnode(dev)
    def builtin(x):
        basis_prep(x, list(range(n)))
        qml.QFT(wires=range(n))
        return qml.state()

    max_err_manual = 0.0
    max_err_builtin = 0.0
    print(f"  {'x':>3s}    {'||manual - analytic||_inf':>26s}    {'||builtin - analytic||_inf':>27s}")
    for x in range(N):
        psi_manual = manual(x)
        psi_builtin = builtin(x)
        psi_exact = analytic_qft_row(x, n)
        e_m = float(np.max(np.abs(psi_manual - psi_exact)))
        e_b = float(np.max(np.abs(psi_builtin - psi_exact)))
        max_err_manual = max(max_err_manual, e_m)
        max_err_builtin = max(max_err_builtin, e_b)
        print(f"  {x:>3d}    {e_m:>26.2e}    {e_b:>27.2e}")
    print(f"\n  worst-case errors:  manual={max_err_manual:.2e}   builtin={max_err_builtin:.2e}")

    section("2. QFT|0...0> is the uniform superposition")

    @qml.qnode(dev)
    def qft_zero():
        qft_manual(list(range(n)))
        return qml.probs(wires=range(n))

    probs = qft_zero()
    print(f"  P(y) for y=0..{N-1}:  {[round(float(p), 4) for p in probs]}")
    print(f"  expect 1/N = {1/N:.4f} on every entry")

    section("3. Inverse QFT undoes QFT for every basis state")

    @qml.qnode(dev)
    def round_trip(x):
        basis_prep(x, list(range(n)))
        qml.QFT(wires=range(n))
        qml.adjoint(qml.QFT)(wires=range(n))
        return qml.probs(wires=range(n))

    max_err = 0.0
    for x in range(N):
        p = round_trip(x)
        # Should be a delta at index x.
        err = max(abs(float(p[x]) - 1.0), max(float(p[i]) for i in range(N) if i != x))
        max_err = max(max_err, err)
    print(f"  worst-case |QFT^-1 QFT |x> - |x>| over all x: {max_err:.2e}")

    section("4. Period -> frequency: QFT of a periodic state concentrates on N/r")
    # Construct |psi> = (1/sqrt(K)) sum_{j=0}^{K-1} |x0 + j*r> with K = N/r.
    # QFT|psi> peaks at multiples of N/r = K.
    n_big = 4
    N_big = 2 ** n_big
    dev_big = qml.device("default.qubit", wires=n_big)

    @qml.qnode(dev_big)
    def qft_periodic(period, offset):
        K = N_big // period
        amps = np.zeros(N_big, dtype=complex)
        for j in range(K):
            amps[(offset + j * period) % N_big] = 1.0 / np.sqrt(K)
        qml.StatePrep(amps, wires=range(n_big), normalize=False)
        qml.QFT(wires=range(n_big))
        return qml.probs(wires=range(n_big))

    for period in (2, 4, 8):
        probs = qft_periodic(period=period, offset=0)
        peaks = [(y, float(probs[y])) for y in range(N_big) if float(probs[y]) > 1e-6]
        peak_str = ", ".join(f"y={y}: P={p:.3f}" for y, p in peaks)
        print(f"  period r={period}  ->  N/r={N_big // period}.  QFT peaks at: {peak_str}")

    section("5. Inspect the n=3 QFT circuit (manual)")

    @qml.qnode(dev)
    def qft_drawn():
        basis_prep(0b101, list(range(n)))
        qft_manual(list(range(n)))
        return qml.probs(wires=range(n))

    qft_drawn()
    print(qml.draw(qft_drawn)())


if __name__ == "__main__":
    main()
