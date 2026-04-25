# Week 3 — Measurement and quantum teleportation

**Codebook I.8–I.10.** Companion to `week3_measurement_and_teleportation.py`.

## 1. The Born rule

For a state $|\psi\rangle$ and an observable $O = \sum_m m\,\Pi_m$ (spectral decomposition), measurement returns outcome $m$ with probability

$$P(m) = \langle\psi|\Pi_m|\psi\rangle = \|\Pi_m|\psi\rangle\|^2,$$

and the post-measurement state is

$$|\psi'\rangle = \frac{\Pi_m|\psi\rangle}{\sqrt{P(m)}}.$$

For a single-qubit $Z$ measurement of $|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$:

$$P(0) = |\alpha|^2,\quad P(1) = |\beta|^2,\quad \text{post-state} = |0\rangle\ \text{or}\ |1\rangle.$$

Mid-circuit measurement is identical, but the outcome can be **fed forward** as a classical control bit for subsequent gates.

## 2. The teleportation problem

Alice holds an unknown qubit $|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$ and wants to transfer it to Bob, with whom she shares one qubit of an entangled pair. Resources: **1 ebit** (one Bell pair) plus **2 classical bits** of communication. No-cloning forbids copying, so the protocol must destroy Alice's copy.

## 3. The protocol

Wires: 0 = Alice's payload $|\psi\rangle$; 1 = Alice's half of the Bell pair; 2 = Bob's half. Initial joint state:

$$|\psi\rangle_0\otimes|\Phi^+\rangle_{12} = \tfrac{1}{\sqrt2}\big(\alpha|0\rangle_0 + \beta|1\rangle_0\big)\otimes\big(|00\rangle_{12} + |11\rangle_{12}\big).$$

Alice applies $\mathrm{CNOT}_{0\to 1}$ then $H_0$, which transforms her two qubits from the computational basis to the Bell basis (this is the standard Bell-basis measurement decomposition). Expanding:

$$|\psi\rangle_0|\Phi^+\rangle_{12} = \tfrac12 \sum_{m_0, m_1\in\{0,1\}} |m_0 m_1\rangle_{01}\otimes Z^{m_0} X^{m_1}|\psi\rangle_2.$$

So if Alice measures $(m_0, m_1)$, Bob's qubit collapses to $Z^{m_0} X^{m_1}|\psi\rangle$. Alice transmits $(m_0, m_1)$ classically; Bob applies the inverse correction:

$$|\psi\rangle_2 = X^{m_1} Z^{m_0}\,(Z^{m_0} X^{m_1}|\psi\rangle_2).$$

(Pauli operators are involutions, so $\sigma^2 = I$.)

## 4. Why corrections are essential

If Bob skips the corrections, his marginal — averaged over Alice's four equally likely outcomes — is

$$\rho_2 = \tfrac{1}{4}\sum_{m_0, m_1} (Z^{m_0} X^{m_1})|\psi\rangle\langle\psi|(X^{m_1} Z^{m_0}) = \tfrac{I}{2}.$$

The **Pauli twirl** maps any pure state to the maximally mixed state. Without the two classical bits, Bob receives no information — **no faster-than-light signaling**.

## 5. What the script verifies

- A mid-circuit $Z$ measurement on $|+\rangle$ collapses the qubit; $\langle Z\rangle$ averages to $0$ and the sampled outcome is $50/50$.
- Bob's Bloch vector matches Alice's input across six test states (four cardinal plus a generic $R_Y(1.0)R_Z(0.7)|0\rangle$) to $\sim 5\times 10^{-16}$.
- Removing the conditional $X^{m_1}Z^{m_0}$ leaves Bob with $\vec r = 0$ — the maximally mixed state.

## 6. Cost summary

| resource | classical sending | quantum teleportation |
|----------|------------------|-----------------------|
| transferring 1 qubit of unknown $\|\psi\rangle$ | impossible (no measurement reads $\alpha,\beta$) | 1 ebit + 2 classical bits |

The corollary is **superdense coding**: 1 ebit + 1 qubit transmits 2 classical bits. Together they show that ebits, qubits, and classical bits are interconvertible at fixed exchange rates.
