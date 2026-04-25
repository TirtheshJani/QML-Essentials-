# Week 1 — Single-qubit gates and the Bloch sphere

**Codebook I.1–I.4.** Companion to `week1_single_qubit_gates.py`.

## 1. The qubit

A pure single-qubit state is a unit vector in $\mathbb{C}^2$:

$$|\psi\rangle = \alpha|0\rangle + \beta|1\rangle, \qquad |\alpha|^2 + |\beta|^2 = 1.$$

Global phase is unphysical, so the *physical* state space is two real parameters. The standard parametrization places it on the surface of a sphere:

$$|\psi(\theta,\varphi)\rangle = \cos\!\tfrac{\theta}{2}\,|0\rangle + e^{i\varphi}\sin\!\tfrac{\theta}{2}\,|1\rangle, \qquad \theta\in[0,\pi],\ \varphi\in[0,2\pi).$$

## 2. Pauli matrices and the Bloch vector

$$X = \begin{pmatrix}0 & 1\\ 1 & 0\end{pmatrix},\quad Y = \begin{pmatrix}0 & -i\\ i & 0\end{pmatrix},\quad Z = \begin{pmatrix}1 & 0\\ 0 & -1\end{pmatrix}.$$

Each is Hermitian and unitary, with eigenvalues $\pm 1$. They satisfy $\sigma_a^2 = I$, $\sigma_a\sigma_b = i\varepsilon_{abc}\sigma_c + \delta_{ab}I$.

The **Bloch vector** of a state $|\psi\rangle$ is

$$\vec{r} = (\langle X\rangle, \langle Y\rangle, \langle Z\rangle), \qquad \langle\sigma_a\rangle := \langle\psi|\sigma_a|\psi\rangle.$$

For a pure state, $|\vec{r}|=1$. In the angular parametrization above,

$$\vec{r} = (\sin\theta\cos\varphi,\ \sin\theta\sin\varphi,\ \cos\theta).$$

Equivalently the density matrix is $\rho = \tfrac12(I + \vec{r}\cdot\vec\sigma)$.

## 3. Cardinal states

| state | Bloch axis | $(\langle X\rangle,\langle Y\rangle,\langle Z\rangle)$ |
|------|-----------|-----------|
| $\|0\rangle$ | $+z$ | $(0,0,+1)$ |
| $\|1\rangle$ | $-z$ | $(0,0,-1)$ |
| $\|+\rangle = (\|0\rangle+\|1\rangle)/\sqrt2$ | $+x$ | $(+1,0,0)$ |
| $\|-\rangle = (\|0\rangle-\|1\rangle)/\sqrt2$ | $-x$ | $(-1,0,0)$ |
| $\|+i\rangle = (\|0\rangle+i\|1\rangle)/\sqrt2$ | $+y$ | $(0,+1,0)$ |

## 4. Hadamard

$$H = \tfrac{1}{\sqrt2}\begin{pmatrix}1 & 1\\ 1 & -1\end{pmatrix} = \tfrac{1}{\sqrt2}(X+Z).$$

Properties: $H^2 = I$, $H|0\rangle = |+\rangle$, $H|1\rangle = |-\rangle$, $HZH = X$, $HXH = Z$.

## 5. Rotation gates

For each axis $a\in\{x,y,z\}$,

$$R_a(\theta) = e^{-i\theta\sigma_a/2} = \cos\!\tfrac{\theta}{2}\,I - i\sin\!\tfrac{\theta}{2}\,\sigma_a.$$

Acting on $|0\rangle$:

$$R_Y(\theta)|0\rangle = \cos\!\tfrac{\theta}{2}|0\rangle + \sin\!\tfrac{\theta}{2}|1\rangle.$$

So the $Z$-basis statistics after $R_Y(\theta)$ on $|0\rangle$ are

$$P(0) = \cos^2\!\tfrac{\theta}{2}, \qquad P(1) = \sin^2\!\tfrac{\theta}{2}, \qquad \langle Z\rangle = \cos\theta.$$

This is the identity verified in the script: at $\theta=60^\circ$, $\cos^2 30^\circ = 0.75$.

## 6. Useful gate identities

- $X = R_X(\pi)$ up to global phase, similarly for $Y, Z$.
- $S = \mathrm{diag}(1, i) = R_Z(\pi/2)$ up to global phase; $T = \mathrm{diag}(1, e^{i\pi/4})$.
- The set $\{H, T, \mathrm{CNOT}\}$ is universal for quantum computation.

## 7. What the script verifies

- All five cardinal states give the predicted Bloch coordinates to machine precision.
- Pauli gates flip the expected axes (e.g. $Z|+\rangle = |-\rangle$).
- The angle sweep matches $P(0) = \cos^2(\theta/2)$ at $\theta\in\{0,30,45,60,90,120,180\}^\circ$.
- $H^2 = I$: Bloch vector returns to $(0,0,+1)$ after two Hadamards on $|0\rangle$.
