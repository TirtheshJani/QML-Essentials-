# Week 16 — Hybrid quantum-classical classifier with TorchLayer

**Tier 2 / 2C.2.** Companion to `week16_variational_classifier.py`.

## 1. Why TorchLayer

A QNode is a function. `qml.qnn.TorchLayer` wraps it as a `torch.nn.Module` so its trainable weights register with `torch.optim`, gradients flow through `loss.backward()`, and the quantum block stacks like any other layer. The inputs argument of the QNode becomes the layer's input tensor; the named weight arguments become `torch.nn.Parameter`s. After wrapping, the rest of the model is plain PyTorch.

```python
qnode(inputs, weights) -> [<Z_0>, <Z_1>, <Z_2>, <Z_3>]
TorchLayer(qnode, weight_shapes={"weights": (n_layers, n_qubits)})
```

The layer maps a `(batch, n_features)` tensor to a `(batch, n_qubits)` tensor; from there `nn.Linear(n_qubits, 1)` gives a logit and `nn.BCEWithLogitsLoss` does the right thing.

## 2. Architecture

| block | params | role |
|------|------:|-----|
| `AngleEmbedding(rotation="Y")` | 0 | $R_Y(x_i)$ on qubit $i$ — the encoding from week 15 |
| `BasicEntanglerLayers(n_layers=2)` | 8 | $L=2$ alternating $R_Y(\theta) +$ ring CNOT |
| Quantum readout | 0 | $\langle Z_w\rangle$ on each of the 4 qubits |
| `nn.Linear(4, 1)` | 5 | classical head: 4 weights + 1 bias |
| `BCEWithLogitsLoss` | 0 | binary cross-entropy with sigmoid built in |

**Total: 13 trainable parameters.** Eight live on the quantum device, five in the classical head.

## 3. Why a 4-D quantum readout, not just $\langle Z_0\rangle$

Returning all four expectations gives the classical head a richer, non-linear feature vector to combine. Geometrically: the entangler layers spread the 4-D Iris point onto the 4 single-qubit Bloch vectors of the resulting state; the linear head finds the best linear combination of those projections. Returning only $\langle Z_0\rangle$ would force the quantum block to do the entire decision; this split is the literal definition of *hybrid*.

## 4. Training

50 epochs of Adam at lr 0.05, batch size 16. Per-epoch loss / train acc / test acc:

| epoch | loss | train | test |
|------:|-----:|------:|-----:|
| 0 | 0.70 | 0.61 | 0.60 |
| 5 | 0.48 | 0.86 | 0.90 |
| 10 | 0.30 | 0.93 | 1.00 |
| 20 | 0.15 | 0.96 | 1.00 |
| 49 | 0.11 | 0.95 | 1.00 |

Test accuracy locks at 1.00 by epoch 10 and stays there; train accuracy plateaus at 0.95 — the model is mildly overfitting toward the end (loss still decreases but test accuracy saturates), which on 80 training examples and 13 parameters is unavoidable. The TorchLayer wrapping costs nothing measurable: this matches week 15's raw-PennyLane angle result.

## 5. The quantum gain over a 1-Linear baseline

A single `nn.Linear(4, 1)` (5 parameters, no quantum block) trained under the same protocol reaches **95 %** test accuracy. The hybrid model with 8 added quantum parameters reaches **100 %** — a **+5 pp** lift. That five-point gain is small in absolute terms, but it is real: the quantum block contributes a non-linear map that the linear classifier cannot replicate. Whether it justifies the 8 added parameters is a question for week 17, which compares against a *parameter-matched* classical MLP.

## 6. The TorchLayer footgun (worth flagging)

`qml.qnn.TorchLayer` defaults to `init_method=torch.nn.init.uniform_(-0.1, 0.1)` for weights. With the angle encoding, that small init keeps gradients well-conditioned. With `StronglyEntanglingLayers` or larger circuits, the same init can land in a barren-plateau region (week 14) and the loss never moves. If a hybrid model trains noisily, init scale is the first thing to check.

## 7. What the script verifies

- Final test accuracy ≥ 0.97 (the plan's threshold) — actual 1.00.
- Training loss falls by more than $2\times$ over the run (actual: $0.70 \to 0.11$).
- Hybrid is at least as good as a parameter-cheaper linear baseline ($1.00 \ge 0.95$).
