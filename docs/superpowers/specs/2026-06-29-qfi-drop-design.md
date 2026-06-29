# QFI-Drop: Information-Geometric Quantum-Weight Pruning for VQ-GNNs

**Date:** 2026-06-29
**Status:** Design — approved in brainstorming, pending implementation plan
**Objective (locked):** Trainability → accuracy. Improve quantum-gradient flow / escape the under-fitting collapse; report test accuracy and fold-variance as downstream metrics.
**Datasets:** MUTAG, PROTEINS, IMDB-binary, IMDB-multi (TUDataset graph classification via QGCN).

---

## 1. Motivation: the current "pruning" is a no-op by construction

The existing `--algorithm pruning` path in [`src/qdrop/`](../../../src/qdrop/) does not measurably change results on any dataset. Tracing [`core.py`](../../../src/qdrop/core.py) + [`session.py`](../../../src/qdrop/session.py) end-to-end and verifying numerically, the mechanism is a noisy near-identity perturbation on the quantum-weight gradients. Five defects (all verified by a standalone reproduction):

| # | Defect | Evidence | Consequence |
|---|--------|----------|-------------|
| 1 | Masks **gradients**, never prunes **weights**. Parameters/gates are never zeroed or removed. | `build_pruned_gradient` masks `accumulated_grad` only; params untouched. | No sparsity, no compression, no structural regularization. Cannot prune by construction. |
| 2 | Importance signal is **sign-broken**. Min-max normalization gives the most-negative accumulated gradient keep-probability ≈ 0. | reproduction: `p[argmin] = 0.0000`. | Large-magnitude negative gradients (highly important directions) are dropped first. Importance should use `|grad|` or curvature, not signed grad. |
| 3 | The softmax/log machinery is **mathematically a no-op**: `softmax(log(x)) ≡ x/Σx`. | reproduction: MATCH to 1e-6. | ~30 lines of dead complexity obscuring defect #2. |
| 4 | Sampling **with replacement** collapses the keep-set. ratio 0.8 → draw 12/15 but only ~7.7/15 *unique* kept. | reproduction: mean unique 7.7/15. | ~half the params get a *random* zero-gradient each prune step, re-rolled constantly → averages to noise. |
| 5 | The schedule runs **backwards**: keep-ratio rises 0.80 → 1.00 over training (`×e^0.1` every 5 steps). | reproduction: 0.80 → 1.00. | Prunes *less* over time, converging to plain SGD — opposite of "anneal toward sparse." |

**Net effect:** ≈ plain SGD on the quantum weights plus re-rolled noise. This is why every dataset shows "not much impact."

### 1.1 Corroborating evidence from the bug log

This is consistent with the documented training history in [`.wolf/buglog.json`](../../../../.wolf/buglog.json):

- **bug-001** — QGCN under-fits (MUTAG `recall=1.0` in 9/10 folds, mean acc 0.734 vs ~0.665 majority baseline). Root cause: unbounded `feature_reduction` output fed to `AngleEmbedding` wraps past π → discontinuous, near-flat loss. Fixed with `tanh(x)·π` input bound.
- **bug-005** — "train-loss spikes synced with `prune_window`." Direct evidence the old pruning was **actively destabilizing** training (the 10× accumulated-gradient spike from defect #4/#1). Removing it is a net positive on its own.
- **bug-006 / bug-007** — Barren plateau named explicitly (McClean 2018). A *global, blind* small-angle re-init (Grant 2019) was tried in Phase 2B and **regressed both datasets**, then reverted. The adapter today uses default uniform `[0, 2π]` init — i.e. the circuit sits in the barren-prone regime.

The repeated, partially-regressing heuristic fixes (input bound, LayerNorm, init tricks, width tuning) share a root limitation: **they act blind to where the loss landscape is actually flat.** QFI-Drop's contribution is to *measure* that geometry and act selectively.

---

## 2. Core idea

Replace gradient-magnitude importance with **information-geometric importance**: the Quantum Fisher Information / Fubini–Study metric tensor **F** of the variational layer. In curved parameter space a small Euclidean gradient does not imply an unimportant parameter; `F_ii` measures how strongly a parameter reshapes the quantum state.

Two coupled actions, which are **one principle** — the numerical-conditioning boundary of Quantum Natural Gradient (QNG):

- **Informative directions** (well-conditioned, `F_ii` large): apply QNG preconditioning `g' = (F + εI)⁻¹ g` — geometry-correct steps that move uniformly in state space. **This is the primary mechanism**, and it does not depend on the circuit being barren: it corrects an *ill-conditioned* optimization landscape, which is exactly the "representation-bound" failure Phase 2C found on PROTEINS (see cerebrum decision log).
- **Barren / degenerate directions** (`F_ii ≈ 0`): **freeze** (zero gradient). Here `(F + εI)⁻¹` is reg-dominated → `≈ (1/ε)·g`, an enormous step driven by noise. Freezing is a **numerical-stability safeguard**, not the headline feature.

So the freeze threshold is not an arbitrary hyperparameter — it is the boundary below which QNG preconditioning is numerically degenerate.

> **Calibration (honors `.wolf/cerebrum.md` KEY LESSON).** These QGCN circuits are *shallow* (`n_layers ≤ 2`), so this is **not** a textbook barren plateau (which needs deep/wide random circuits). The team already established this and a global barren-plateau fix (Phase 2B small-angle init) *regressed*. QFI-Drop is therefore framed as **geometry-correct optimization on a measured, input-averaged metric**, not as barren-plateau escape. If the freeze set is usually small (few near-zero `F_ii`), that is the *expected* outcome on shallow circuits — the win comes from QNG conditioning of the informative subspace, not from freezing.

**Novel methodological piece:** because `AngleEmbedding` makes F **input-dependent**, importance must be an *expectation of F over a probe batch of node features*, not F at a single point. bug-005 shows this input distribution is non-stationary across epochs, which motivates re-probing each epoch.

---

## 3. Algorithm: QFI-Drop (per quantum layer, weights θ ∈ ℝᵖ)

Per quantum layer (each `QuantumCircuitAdapter` exposing `quantum_layer.qnode` and `weights` of shape `(n_layers, n_qubits)`):

1. **Probe (once per epoch, in `start_epoch`):**
   `F = E_{x∼probe_batch}[ metric_tensor(qnode)(x, θ) ]`, a p×p matrix. Cache it for the epoch.
   Importance `s_i = F_ii`.
2. **Partition (adaptive quantile — no magic constants):**
   `Barren  B = { i : s_i ≤ quantile(s, ρ_low) }` (default ρ_low = 1/3, bottom tercile).
   `Informative I = complement(B)`.
3. **Per-step gradient surgery (in `after_backward`):**
   - `i ∈ B`: `g_i ← 0` (freeze).
   - `i ∈ I`: `g_I ← (F_II + εI)⁻¹ g_I` via `apply_fubini_study_precondition` (reuses [`qng.py`](../../../src/qml_techniques/qng.py)). `ε` ties to the freeze threshold.
   - The existing optimizer (Adam) then consumes the modified `.grad` — same QNG-then-Adam composition already used by `QNGAdam`.
4. **Re-probe each epoch:** a frozen param rejoins `I` if its curvature grows (and vice-versa). Log frozen-fraction + F spectrum via the existing `qdrop_curve` snapshot.

### 3.1 Why this is the lazy-but-correct architecture

QFI-Drop lives **entirely inside the existing `after_backward` grad hook** plus a per-epoch probe in `start_epoch` ([`torch_runtime.py`](../../../src/qdrop/backends/torch_runtime.py)). It reuses `apply_fubini_study_precondition` as a **pure function** — no optimizer swap, no change to the training loop in [`graph_training.py`](../../../src/training/graph_training.py). The `QNGAdam` *class* is not required (kept as an alternative wiring). This mirrors how `pruning`/`dropout` already work: grad surgery + epoch state.

---

## 4. Integration surface

| File | Change |
|------|--------|
| `src/qdrop/types.py` | Add `qfi` to `SUPPORTED_QDROP_ALGORITHMS`; add `QDropConfig` fields: `qfi_reg` (ε, default 1e-4), `barren_quantile` (default 0.333), `probe_batch_size` (default 32), `reprobe_every` (default 1 epoch). |
| `src/qdrop/qfi.py` *(new)* | `metric_provider` factory: closes over a `QuantumCircuitAdapter` + a probe-batch sampler; returns data-averaged F. Pure-function `build_freeze_mask(F, quantile)` and `precondition_informative(grad, F, mask, reg)`. |
| `src/qdrop/session.py` | New `algorithm == "qfi"` path: `start_epoch` triggers probe+cache+partition; `process_tensor_grad` applies freeze + precondition. Keeps existing pruning/dropout paths intact. |
| `src/qdrop/backends/torch_runtime.py` | Wire the probe at `start_epoch` (needs access to the layer's `qnode` and a probe batch — supplied via the spec/adapter). |
| `src/training/graph_training.py` | `--algorithm qfi` choice; pass a probe batch (a fixed held-out mini-batch of node features) into the runtime at epoch start; extend `snapshot_qdrop_state` to log F spectrum + frozen-fraction. |
| `src/qml_techniques/qng.py` | Reuse `apply_fubini_study_precondition` (function only). **Branch dependency** — see §7. |

---

## 5. Success criteria

**Primary (objective = trainability):**
- Mean test accuracy ↑ vs `--algorithm baseline` on all four datasets.
- **`recall` no longer pinned at 1.0** — the model stops collapsing to the majority class (the defining symptom of the under-fitting in bug-001).

**Trainability evidence (mechanism works as claimed):**
- Conditioning of the *informative* subspace improves: condition number `κ(F_II)` falls / the QNG-preconditioned update direction stabilizes across epochs (vs the raw-gradient direction thrashing).
- Epochs-to-best-val decreases vs baseline; per-epoch val-acc variance falls.
- Frozen-fraction is logged for transparency. **Do not** treat a small frozen-fraction as failure — on these shallow circuits it is expected (see §2 calibration).

**Secondary:**
- Fold-variance (`std_accuracy`) ↓.

A run that raises accuracy but leaves `recall≈1.0` has *not* validated the mechanism (it got lucky on threshold) and must be investigated.

---

## 6. Testing

- **Unit:** `build_freeze_mask` quantile correctness; `precondition_informative` equals `(F+εI)⁻¹g` on the informative block and 0 on barren; F-probe diagonal is positive and matches `qml.metric_tensor` on a 2-qubit toy qnode.
- **Property:** preconditioned step never amplifies a frozen direction; partition is a true bipartition.
- **Pipeline smoke:** QFI-Drop on synthetic graphs (follow the existing `tests/qdb/test_pipeline.py` pattern) — one epoch, asserts no NaN, mask shapes match weight shapes, probe runs on `default.qubit`.
- **Benchmark:** the four datasets, baseline vs qfi, seeds fixed (existing `set_seed`).

---

## 7. Risks & open implementation decisions

1. **`metric_tensor` on `lightning.qubit` + `adjoint`.** For `n_qubits ≥ 12` (PROTEINS uses 16 per bug-007) the adapter switches to `lightning.qubit`/adjoint, where `qml.metric_tensor` support is uncertain. **Decision needed:** compute the probe on a `default.qubit` shadow of the qnode, or use the block-diagonal metric approximation. Verify in plan step 1.
2. **Branch strategy.** `qng.py` + `small_angle.py` live on `feat/qml_techniques`, not `main`. **Decision needed:** build QFI-Drop on a branch off `feat/qml_techniques`, or cherry-pick `apply_fubini_study_precondition` into `src/qdrop/`. Recommend branching off `feat/qml_techniques` so QNG/QFI code co-locates.
3. **Probe-batch policy.** Fixed held-out batch (stable, comparable across epochs) vs running estimate (tracks non-stationarity). Recommend a fixed probe batch sampled once per fold for comparability; revisit if F drifts too much.
4. **Probe cost.** p×p metric × probe-batch circuit evals, once/epoch. Tiny at this circuit size; confirm wall-clock on PROTEINS (largest p).
5. **QNG-then-Adam composition** preconditions twice (F⁻¹ then Adam's diagonal). This matches the established `QNGAdam` pattern; flagged as a known, accepted interaction.

---

## 8. Out of scope (YAGNI)

- **Approach B (Quantum Lottery Ticket / structural gate removal)** — higher compute/risk; revisit only if QFI-Drop validates and a compression story is wanted.
- **Compression-for-NISQ objective** — circuits are tiny; nothing meaningful to compress.
- **Global blind re-initialization** — already tried (Phase 2B) and regressed; QFI-Drop acts selectively instead.
- **Fixing the legacy `pruning`/`dropout` paths** — left intact for backward compatibility; QFI-Drop is a new `--algorithm`, not a rewrite of the old ones.
