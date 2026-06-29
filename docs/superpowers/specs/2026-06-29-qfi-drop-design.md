# QFI-Drop: Spectral-Leverage Quantum-Fisher Pruning with Unified Natural-Gradient Updates for Quantum GNNs

**Date:** 2026-06-29
**Status:** Design — approved in brainstorming; proceeding to implementation plan
**Working name / config flag:** `QFI-Drop` / `--algorithm qfi`
**Objective (locked):** Trainability → accuracy. Improve quantum-gradient flow / escape the under-fitting collapse; report test accuracy and fold-variance as downstream metrics.
**Datasets:** MUTAG, PROTEINS, IMDB-binary, IMDB-multi (TUDataset graph classification via QGCN).

---

## 0. Contributions (the paper's claims)

An **incremental, reviewer-defensible** method synthesized from existing work — a new criterion + a unification + a new domain. Not a fundamental new mechanism; claims are **conditional on beating the QAdaPrune baseline** (§5).

1. **A pruning criterion from the QFIM spectrum.** Score each quantum parameter (gate) by its **spectral kept-energy** `e_j = Σ_{i∈Kept} U_ji²` — the fraction of that gate's energy lying in the high-curvature eigensubspace of the data-averaged Quantum Fisher Information Matrix. Freeze low-`e_j` gates. This uses the full eigenstructure, vs QAdaPrune's gradient-difference proxy or a naive QFIM diagonal.
2. **A unified prune-and-optimize rule from one geometric object.** The same QFIM yields the prune scores (spectral leverage) *and* the optimizer step (natural-gradient `(F_SS+εI)⁻¹g_S` on the surviving sub-manifold). Prior work uses these objects for one job each.
3. **First QFI-based circuit-parameter pruning of a Quantum GNN**, with the QFIM averaged over the node-feature manifold inside a message-passing layer — distinct from graph-input compression and from one-shot structural QNN pruning (§2).

---

## 1. Motivation: the current "pruning" is a no-op by construction

`--algorithm pruning` ([`core.py`](../../../src/qdrop/core.py), [`session.py`](../../../src/qdrop/session.py)) is a noisy near-identity perturbation, verified by reproduction:

| # | Defect | Evidence | Consequence |
|---|--------|----------|-------------|
| 1 | Masks **gradients**, never prunes **weights**. | `build_pruned_gradient` masks `accumulated_grad`; params untouched. | No capacity change. |
| 2 | **Sign-broken** importance (min-max → most-negative grad keep-prob ≈ 0). | `p[argmin]=0.0000`. | Important directions dropped first. |
| 3 | `softmax(log(x)) ≡ x/Σx` — **a no-op**. | MATCH to 1e-6. | Dead complexity. |
| 4 | Sampling **with replacement** collapses keep-set (0.8 → ~7.7/15 unique). | mean 7.7/15. | Re-rolled noise each step. |
| 5 | Schedule runs **backwards** (keep-ratio 0.80 → 1.00). | 0.80→1.00. | Prunes *less* over time. |

**Net:** ≈ plain SGD + noise. Hence no impact on any dataset.

### 1.1 Bug-log corroboration ([`.wolf/buglog.json`](../../../../.wolf/buglog.json))
- **bug-001** — under-fitting (MUTAG `recall=1.0` in 9/10 folds, acc 0.734 vs ~0.665 majority); cause was unbounded `AngleEmbedding` input, fixed with `tanh(x)·π`.
- **bug-005** — "train-loss spikes synced with `prune_window`": the old pruning actively destabilized training.
- **bug-006/007** — barren plateau named; a *global* small-angle re-init regressed and was reverted. Circuits are shallow (`n_layers ≤ 2`).

The prior fixes acted **blind to where the landscape is flat**. QFI-Drop measures that geometry and acts selectively.

---

## 2. Prior art and honest positioning

QFI/Fisher-based VQC pruning is an existing, active line. QFI-Drop's components are published; the **combination + domain** is the contribution.

| Method | Prune by | Step by | When | Domain |
|--------|----------|---------|------|--------|
| [QAdaPrune (2024)](https://arxiv.org/abs/2408.13352) — closest | gradient-difference proxy, freeze | plain GD | training | 4×4 MNIST/FashionMNIST, VQE |
| [Sculpting Quantum Landscapes (2025)](https://arxiv.org/abs/2506.21940) | — | — | **init only** (FS-metric conditioning) | generic PQC |
| [q-Group one-shot pruning (2025)](https://arxiv.org/html/2512.24019) / [LiePrune (2025)](https://arxiv.org/pdf/2512.09469) | quantum geometric metric, **structural (gates), one-shot** | n/a | post hoc | generic QNN |
| [Quantum Natural Gradient (2020)](https://quantum-journal.org/papers/q-2020-05-25-269/) | — | FS metric / QFIM | training | generic PQC |
| [Guided Graph Compression for QGNN (2025)](https://arxiv.org/abs/2506.09862) | **input graph** (nodes/features), not circuit | GD | preprocessing | Quantum GNN |
| [Quantum dropout (2023)](https://arxiv.org/pdf/2310.04120) | random unitary removal | GD | training | generic PQC |
| **QFI-Drop (this work)** | **QFIM spectral-leverage, per-gate freeze** | **QNG on surviving submatrix (same QFIM)** | **training** | **Quantum GNN** |

**The wedge.** QAdaPrune prunes by a *proxy* and steps by *plain GD* — unrelated procedures. Sculpting conditions the metric only at *init*. QNG steps by the metric but never prunes. Guided Graph Compression compresses the *input graph*, not the circuit. The structural-pruning papers remove gates *one-shot, post hoc*, on generic QNNs. **No surveyed work uses the QFIM for per-gate pruning *and* the natural-gradient step simultaneously during training on a QGNN.** That is QFI-Drop.

> **Confidence:** based on targeted searches + one full read (QAdaPrune). A full systematic related-work review remains good practice before submission, but contribution #3 is defensible given the distinctions above.

---

## 3. Method

Per quantum layer with weights θ ∈ ℝᵖ (each `QuantumCircuitAdapter` exposes `quantum_layer.qnode` and `weights` of shape `(n_layers, n_qubits)`; p ≤ ~32, so dense eigendecomposition is trivial).

### 3.1 Probe (once per epoch, `start_epoch`)
`F = E_{x∼probe_batch}[ metric_tensor(qnode)(x, θ) ]` — a p×p symmetric PSD matrix. F is input-dependent (AngleEmbedding) and the input distribution is non-stationary (bug-005) → average over a fixed probe batch, re-probe each epoch. Computed under `no_grad` and treated as a constant for the epoch.

### 3.2 Spectral leverage → per-gate prune mask
Eigendecompose `F = UΛUᵀ`, `λ₁≥…≥λ_p≥0`. Informative eigenset `Kept = {i : λ_i ≥ ρ·λ_max}` (default ρ = 1e-3). Per-gate **kept-energy**:
```
e_j = Σ_{i ∈ Kept} U_ji²      ∈ [0, 1]      (rows of U are orthonormal ⇒ Σ_all i U_ji² = 1)
```
`e_j` = fraction of gate j's energy in the high-curvature subspace. **Freeze** gates with `e_j < τ` (τ = a low quantile of `{e_j}`, default 33rd percentile — adaptive, no magic absolute). Survivors `S = {j : e_j ≥ τ}`.

### 3.3 Unified update (`after_backward`) — the core
One QFIM, both jobs:
```
g_j ← 0                              for j ∉ S        (prune / freeze)
g_S ← (F_SS + εI)^{-1} g_S                            (QNG on the surviving sub-manifold)
```
`F_SS` = F restricted to surviving indices; reuses [`qng.py`](../../../src/qml_techniques/qng.py)'s `(F+εI)⁻¹g` solver on the submatrix. Adam then consumes `g'` (same QNG-then-Adam composition as `QNGAdam`).

### 3.4 Re-probe each epoch
Spectrum + mask recomputed per epoch; a gate rejoins `S` as its leverage grows. Logged: spectrum, `log10 κ(F_SS)` (the Sculpting trainability quantity), frozen-gate count, mean `e_j`.

### 3.5 Why this is the lazy-but-correct architecture
Lives **entirely in the existing `after_backward` hook** + a per-epoch probe in `start_epoch` ([`torch_runtime.py`](../../../src/qdrop/backends/torch_runtime.py)). No optimizer swap, no training-loop change in [`graph_training.py`](../../../src/training/graph_training.py). Reuses the metric-tensor provider and `(F+εI)⁻¹` solver from `qng.py`; adds `torch.linalg.eigh` for the leverage scores. The `QNGAdam` *class* is not required.

---

## 4. Integration surface

| File | Change |
|------|--------|
| `src/qdrop/types.py` | Add `qfi`, `qadaprune` to `SUPPORTED_QDROP_ALGORITHMS`; `QDropConfig` fields `qfi_reg` (ε,1e-4), `spectral_ratio` (ρ,1e-3), `keep_quantile` (τ,0.33), `probe_batch_size` (32), `reprobe_every` (1). |
| `src/qdrop/qfi.py` *(new)* | `metric_provider` factory (adapter + probe-batch → data-averaged F); pure functions `spectral_leverage(F, ρ)→e`, `prune_and_precondition(g, F, e, τ, ε)` (§3.2–3.3 via `eigh`+`solve`). |
| `src/qdrop/qadaprune.py` *(new, baseline)* | QAdaPrune gradient-difference freezing — the **mandatory comparison baseline**, same hook framework. |
| `src/qdrop/session.py` | `algorithm ∈ {"qfi","qadaprune"}` paths; `start_epoch` probes+caches; `process_tensor_grad` applies §3.3. Legacy paths untouched. |
| `src/qdrop/backends/torch_runtime.py` | Wire the probe at `start_epoch` (needs `qnode` + probe batch via the spec/adapter). |
| `src/training/graph_training.py` | `--algorithm {qfi,qadaprune}` choices; supply a fixed probe batch at epoch start; extend `snapshot_qdrop_state` to log spectrum + `κ(F_SS)` + frozen-count. |
| `src/qml_techniques/qng.py` | Reuse metric-provider wiring + `(F+εI)⁻¹` solver. **Branch dependency** — §7.2. |

---

## 5. Success criteria & experiment matrix

The method is justified **only if it beats QAdaPrune**. This comparison *is* the paper.

**Baselines (required):** `baseline` (no prune) · **QAdaPrune** (grad-diff freeze) · random freeze.
**Ablations (each isolates one contribution):**
- diagonal-`F_jj` importance **vs** spectral kept-energy `e_j` → isolates contribution 1.
- freeze + plain GD on survivors **vs** freeze + QNG on `F_SS` → isolates contribution 2.

**Primary outcomes:**
- Mean test accuracy: QFI-Drop > QAdaPrune > baseline on the four datasets (or clearly characterize where/why not).
- **`recall` un-pins from 1.0** (the under-fitting signature lifts).

**Trainability evidence:** `log10 κ(F_SS)` falls over training; epochs-to-best-val ↓; per-epoch val-acc variance ↓.
**Secondary:** fold-variance (`std_accuracy`) ↓.

A run that lifts accuracy but leaves `recall≈1.0`, or that does not beat QAdaPrune, has **not** validated the method.

---

## 6. Testing

- **Unit:** `spectral_leverage` — `e_j∈[0,1]`, sums correctly, recovers all-ones when ρ→0; `prune_and_precondition` zeros frozen gates and equals `(F_SS+εI)⁻¹g_S` on survivors; F-probe matches `qml.metric_tensor` on a 2-qubit toy qnode.
- **Property:** frozen gates get exactly zero update; survivor step never exceeds `‖g_S‖/ε`; mask is a true bipartition.
- **Baseline parity:** QAdaPrune reproduces its freeze behavior on a toy circuit.
- **Pipeline smoke:** QFI-Drop one epoch on synthetic graphs (follow `tests/qdb/test_pipeline.py`): no NaN, shapes intact, probe runs on `default.qubit`.
- **Benchmark:** four datasets × {baseline, QAdaPrune, random, qfi, ablations}, seeds fixed via `set_seed`.

---

## 7. Risks & open implementation decisions

1. **`metric_tensor` on `lightning.qubit` + `adjoint`** (PROTEINS `n_qubits=16`): support uncertain. **Decision:** probe on a `default.qubit` shadow of the qnode, or block-diagonal metric. **Plan step 1 — verify before building anything else; this gates the method.**
2. **Branch strategy:** `qng.py` is on `feat/qml_techniques`, not `main`. **Decision:** branch QFI-Drop off `feat/qml_techniques` so QNG/QFI code co-locates.
3. **No backprop through `eigh`/`solve`:** F is computed under `no_grad` once per epoch and used as a constant preconditioner (as QNG does). Avoids eigendecomposition gradient instability.
4. **QGNN prior-art:** contribution #3 defensible against graph-compression (2506.09862) and structural pruning (2512.24019/2512.09469); a full systematic review before submission is still recommended.
5. **Probe-batch policy:** fixed held-out batch per fold (comparable across epochs).
6. **τ/ρ sensitivity** and **QNG-then-Adam double preconditioning** (matches `QNGAdam`) — to be ablated.

---

## 8. Out of scope (YAGNI)

- **Structural gate removal / Quantum Lottery Ticket** — revisit only if QFI-Drop validates and a compression story is wanted.
- **Compression-for-NISQ objective** — circuits are tiny.
- **Global blind re-initialization** — already tried (Phase 2B) and regressed.
- **Rewriting legacy `pruning`/`dropout`** — left intact for back-compat.
- **Graph-topology-aware importance** (degree/substructure weighting) — a candidate *next* contribution, deferred to keep this paper focused.
