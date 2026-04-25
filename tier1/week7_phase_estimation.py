"""Week 7 — Quantum Phase Estimation (skim).

Codebook I.16. Working through:
  - the QPE circuit:
      H^t on counting register, controlled-U^{2^k} from each counting
      wire onto the eigenstate, inverse QFT on counting wires
  - reading the counting register as integer y -> phase estimate y / 2^t
  - exact recovery when phi is dyadic and fits in t bits (T,S,Z gates)
  - approximate recovery for non-dyadic phi (RZ at 2 pi / 3): show the
    error scales as 2^{-t}
"""

import numpy as np
import pennylane as qml


def controlled_phase_power(power, phi, control, target):
    """Apply C-U^{power} for U = PhaseShift(2 pi * phi) on |1>: just controlled phase."""
    qml.ctrl(qml.PhaseShift, control=control)(2 * np.pi * phi * power, wires=target)


def qpe_manual(t, phi, eigenstate=1):
    """Run QPE for U = diag(1, e^{2 pi i phi}) on the eigenstate |eigenstate>.

    Counting wires are 0..t-1, with wire 0 chosen as the MSB of the output
    integer: it controls U^{2^{t-1}}, wire t-1 controls U^1. After the
    inverse QFT, measuring wires 0..t-1 yields integer y and phi ~ y / 2^t.
    """
    target = t  # single-qubit eigenstate on wire t
    dev = qml.device("default.qubit", wires=t + 1)
    counting = list(range(t))

    @qml.qnode(dev)
    def circuit():
        if eigenstate == 1:
            qml.PauliX(target)
        for w in counting:
            qml.Hadamard(w)
        for j in range(t):
            power = 2 ** (t - 1 - j)
            controlled_phase_power(power, phi, control=counting[j], target=target)
        qml.adjoint(qml.QFT)(wires=counting)
        return qml.probs(wires=counting)

    return circuit()


def qpe_builtin(t, phi, eigenstate=1):
    """Same QPE using qml.QuantumPhaseEstimation for cross-check."""
    target = t
    dev = qml.device("default.qubit", wires=t + 1)
    counting = list(range(t))
    U = np.diag([1.0, np.exp(2j * np.pi * phi)])

    @qml.qnode(dev)
    def circuit():
        if eigenstate == 1:
            qml.PauliX(target)
        qml.QuantumPhaseEstimation(U, target_wires=[target], estimation_wires=counting)
        return qml.probs(wires=counting)

    return circuit()


def best_estimate(probs):
    """Return (y_argmax, phi_estimate, peak_prob) reading wire 0 as MSB."""
    y = int(np.argmax(probs))
    t = int(np.log2(len(probs)))
    return y, y / (2 ** t), float(probs[y])


def section(title):
    print("\n" + title)
    print("-" * len(title))


def main():
    section("1. Dyadic phases recovered exactly when phi fits in t bits")
    print(f"  {'gate':>6s}    {'phi':>10s}    {'t':>2s}    {'y_max':>5s}    {'phi_est':>10s}    {'P(peak)':>10s}    {'method':>8s}")
    cases = [
        ("Z",  1/2, 1),
        ("S",  1/4, 2),
        ("T",  1/8, 3),
        ("T",  1/8, 5),  # extra precision still gives exact answer
    ]
    for name, phi, t in cases:
        for label, fn in (("manual", qpe_manual), ("builtin", qpe_builtin)):
            probs = fn(t, phi)
            y, est, peak = best_estimate(probs)
            print(f"  {name:>6s}    {phi:>10.6f}    {t:>2d}    {y:>5d}    {est:>10.6f}    {peak:>10.4f}    {label:>8s}")

    section("2. Non-dyadic phase: phi = 1/3, watch the estimate sharpen with t")
    phi = 1 / 3
    print(f"  true phi = {phi:.6f}")
    print(f"  {'t':>2s}    {'y_max':>5s}    {'phi_est':>10s}    {'|err|':>10s}    {'P(peak)':>10s}    {'2^{-t}':>10s}")
    for t in (3, 4, 5, 6, 7, 8):
        probs = qpe_manual(t, phi)
        y, est, peak = best_estimate(probs)
        err = abs(est - phi)
        print(f"  {t:>2d}    {y:>5d}    {est:>10.6f}    {err:>10.4e}    {peak:>10.4f}    {2**-t:>10.4e}")

    section("3. Manual vs builtin agreement on the non-dyadic case (t=6)")
    probs_m = qpe_manual(6, 1/3)
    probs_b = qpe_builtin(6, 1/3)
    diff = float(np.max(np.abs(probs_m - probs_b)))
    print(f"  ||probs_manual - probs_builtin||_inf = {diff:.2e}")

    section("4. Distribution shape for non-dyadic phi (t=4)")
    # The two bins straddling 2^t * phi share most of the mass.
    t = 4
    probs = qpe_manual(t, 1/3)
    expected_y = 2 ** t / 3  # = 16/3 ~ 5.33
    print(f"  expected fractional y = 2^t * phi = {expected_y:.4f}  (between bins 5 and 6)")
    top = sorted(enumerate(probs), key=lambda kv: -kv[1])[:4]
    for y, p in top:
        print(f"    y={y:>3d}   phi_est={y/2**t:.4f}   P={float(p):.4f}")

    section("5. Inspect a small QPE circuit (t=3, U=T)")
    t = 3
    target = t
    dev = qml.device("default.qubit", wires=t + 1)

    @qml.qnode(dev)
    def qpe_drawn():
        qml.PauliX(target)
        for w in range(t):
            qml.Hadamard(w)
        for j in range(t):
            power = 2 ** (t - 1 - j)
            controlled_phase_power(power, 1/8, control=j, target=target)
        qml.adjoint(qml.QFT)(wires=range(t))
        return qml.probs(wires=range(t))

    qpe_drawn()
    print(qml.draw(qpe_drawn)())


if __name__ == "__main__":
    main()
