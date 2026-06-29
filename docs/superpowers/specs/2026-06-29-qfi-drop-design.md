# QFI-Drop: Spectral Quantum-Fisher Pruning with Unified Natural-Gradient Updates for Quantum GNNs

**Date:** 2026-06-29
**Status:** Design — approved in brainstorming, pending implementation plan
**Working name / config flag:** `QFI-Drop` / `--algorithm qfi`
**Objective (locked):** Trainability → accuracy. Improve quantum-gradient flow / escape the under-fitting collapse; report test accuracy and fold-variance as downstream metrics.
**Datasets:** MUTAG, PROTEINS, IMDB-binary, IMDB-multi (TUDataset graph classification via QGCN).

---

## 0. Contributions (the paper's claims)

This is an **incremental, reviewer-defensible** method synthesized from existing work — a new criterion + a unification + a new domain. It is **not** a fundamental new mechanism, and its claims are **conditional on beating the QAdaPrune baseline** (see §5).

1. **A pruning criterion from the true QFIM spectrum** — eigendirections of the data-averaged Quantum Fisher Information Matrix — rather than the gradient-difference proxy of QAdaPrune. Geometry-exact and sign-correct.
2. **A unified prune-and-optimize rule from one geometric object**: the same QFIM eigendecomposition supplies both the prune set (near-null eigendirections, *truncated*) and the optimizer step (Tikhonov-regularized natural gradient on the kept eigensubspace). Prior work uses these objects for one job each, never both at once.
3. **First QFI-based pruning of a Quantum GNN / graph-classification model**, with the QFIM averaged over the node-feature manifold inside a message-passing layer.

---

## 1. Motivation: the current "pruning" is a no-op by construction

The existing `--algorithm pruning` path in [`src/qdrop/`](../../../src/qdrop/) does not measurably change results on any dataset. Tracing [`core.py`](../../../src/qdrop/core.py) + [`session.py`](../../../src/qdrop/session.py) end-to-end and verifying numerically, the mechanism is a noisy near-identity perturbation on the quantum-weight gradients. Five defects (all verified by a standalone reproduction):

| # | Defect | Evidence | Consequence |
|---|--------|----------|-------------|
| 1 | Masks **gradients**, never prunes **weights**. | `build_pruned_gradient` masks `accumulated_grad` only; params untouched. | No sparsity, no capacity change. |
| 2 | Importance signal is **sign-broken** (min-max normalize → most-negative grad keep-prob ≈ 0). | `p[argmin] = 0.0000`. | Important negative-gradient directions dropped first. |
| 3 | The softmax/log machinery is **a no-op**: `softmax(log(x)) ≡ x/Σx`. | MATCH to 1e-6. | Dead complexity hiding defect #2. |
| 4 | Sampling **with replacement** collapses the keep-set (ratio 0.8 → ~7.7/15 unique). | mean unique 7.7/15. | ~half the params get a *random* zero-grad each step → noise. |
| 5 | Schedule runs **backwards** (keep-ratio 0.80 → 1.00). | 0.80 → 1.00. | Prunes *less* over time. |

**Net effect:** ≈ plain SGD on the quantum weights plus re-rolled noise. This is why every dataset shows "not much impact."

### 1.1 Corroborating evidence from the bug log

Consistent with [`.wolf/buglog.json`](../../../../.wolf/buglog.json):
- **bug-001** — QGCN under-fits (MUTAG `recall=1.0` in 9/10 folds, acc 0.734 vs ~0.665 majority). Cause: unbounded `AngleEmbedding` input wraps past π → discontinuous, near-flat loss. Fixed with `tanh(x)·π`.
- **bug-005** — "train-loss spikes synced with `prune_window`": the old pruning was **actively destabilizing** training. Removing it is a net positive.
- **bug-006/007** — barren plateau named (McClean 2018); a *global, blind* small-angle re-init regressed both datasets and was reverted. The architecture is shallow (`n_layers ≤ 2`).

The repeated, partially-regressing heuristic fixes share a limitation: **they act blind to where the loss landscape is flat.** QFI-Drop measures that geometry and acts selectively.

---

## 2. Prior art and honest positioning

QFI/Fisher-based pruning of variational quantum circuits is an existing, active line. The components of QFI-Drop are all published; the **specific combination + domain** is the contribution.

| Method | Prune by | Step by | When | Domain |
|--------|----------|---------|------|--------|
| [QAdaPrune (2024)](https://arxiv.org/abs/2408.13352) — closest prior art | gradient-difference proxy ("approx Hessian"), freeze | plain GD | training | 4×4 MNIST/FashionMNIST, VQE |
| [Sculpting Quantum Landscapes (2025)](https://arxiv.org/abs/2506.21940) | — (no pruning) | — | **init only** (meta-learned FS-metric conditioning) | generic PQC |
| [One-Shot Structured Pruning via Quantum Geometric Metrics (2025)](https://arxiv.org/html/2512.24019) | geometric metric, **structural** (gates), one-shot | n/a | post hoc | generic QNN |
| [Quantum Natural Gradient (2020)](https://quantum-journal.org/papers/q-2020-05-25-269/) | — | FS metric / QFIM | training | generic PQC |
| [A General Approach to Dropout in QNNs (2023)](https://arxiv.org/pdf/2310.04120) | random unitary removal | GD | training | generic PQC |
| **QFI-Drop (this work)** | **QFIM spectrum (eigendirection truncation)** | **QNG on same QFIM** | **training** | **Quantum GNN** |

**The wedge.** QAdaPrune prunes by a *proxy* and steps by *plain GD* — two unrelated procedures. Sculpting conditions the FS metric only at *initialization*. QNG steps by the FS metric but never prunes. **No surveyed work uses the QFIM for prune *and* step simultaneously during training**, where the truncation threshold and the QNG conditioning floor are the same geometric boundary. That unification, on QGNNs, is QFI-Drop.

> **Confidence caveat.** Positioning is based on a search-snippet + one full read (QAdaPrune). A complete related-work sweep — especially **quantum-GNN pruning / graph-VQC compression** — is required before submission to confirm contribution #3 (see §7).

---

## 3. Method

Per quantum layer with weights θ ∈ ℝᵖ (each `QuantumCircuitAdapter` exposes `quantum_layer.qnode` and `weights` of shape `(n_layers, n_qubits)`; p ≤ ~32 here, so dense eigendecomposition is trivial).

### 3.1 Probe (once per epoch, in `start_epoch`)
Compute the **data-averaged QFIM**:
`F = E_{x∼probe_batch}[ metric_tensor(qnode)(x, θ) ]`  — a p×p symmetric PSD matrix.
F is **input-dependent** (AngleEmbedding), and bug-005 shows the input distribution is non-stationary across epochs → average over a probe batch and re-probe each epoch.

### 3.2 Eigendecompose and partition
`F = U Λ Uᵀ`, eigenvalues `λ₁ ≥ … ≥ λ_p ≥ 0`. Small λ_i ⇒ eigendirection `u_i` barely changes the quantum state (flat / redundant / barren). Partition by **adaptive ratio** (no magic constant):
`Frozen = { i : λ_i / λ_max < ρ }` (default ρ = 1e-3); `Kept = complement`.

### 3.3 Unified update (in `after_backward`) — the core
A spectrally-truncated, Tikhonov-regularized natural gradient:

```
g' = Σ_{i ∈ Kept} [ 1 / (λ_i + ε) ] · (u_iᵀ g) · u_i
```

- **Truncation over `Frozen`** *is* the pruning: those eigendirections are dropped entirely (set to 0), instead of receiving the `1/ε` blow-up that plain regularized QNG would give a near-null direction.
- **`1/(λ_i+ε)` over `Kept`** is the QNG step on the well-conditioned subspace.
- One eigendecomposition; both jobs. The existing optimizer (Adam) then consumes `g'` — same QNG-then-Adam composition as `QNGAdam`.

### 3.4 Re-probe each epoch
Eigenstructure recomputed per epoch; a direction rejoins `Kept` as its curvature grows. Logged each epoch: spectrum, `log10(λ_max/λ_min_kept)` (condition number — the Sculpting trainability quantity), frozen-subspace dimension.

### 3.5 Interpretability bridge (parameter-space view)
The frozen set is a *subspace* (span of `{u_i : i∈Frozen}`), not individual gates. For a gate-level reading, report each parameter's **kept-energy** `e_j = Σ_{i∈Kept} (U_{ji})²`; low `e_j` ⇒ parameter j is effectively inactive. This recovers a QAdaPrune-style per-parameter pruning report from the spectral method.

### 3.6 Why this is the lazy-but-correct architecture
QFI-Drop lives **entirely inside the existing `after_backward` grad hook** + a per-epoch probe in `start_epoch` ([`torch_runtime.py`](../../../src/qdrop/backends/torch_runtime.py)). No optimizer swap, no training-loop change in [`graph_training.py`](../../../src/training/graph_training.py). Reuses [`qng.py`](../../../src/qml_techniques/qng.py): the metric-tensor provider, and the matrix machinery (`torch.linalg.eigh` replaces / complements `solve`). The `QNGAdam` *class* is not required.

---

## 4. Integration surface

| File | Change |
|------|--------|
| `src/qdrop/types.py` | Add `qfi` to `SUPPORTED_QDROP_ALGORITHMS`; `QDropConfig` fields: `qfi_reg` (ε, 1e-4), `spectral_ratio` (ρ, 1e-3), `probe_batch_size` (32), `reprobe_every` (1). |
| `src/qdrop/qfi.py` *(new)* | `metric_provider` factory (adapter + probe-batch sampler → data-averaged F); pure functions `spectral_partition(F, ρ)` and `truncated_natural_gradient(g, F, ρ, ε)` (the §3.3 formula via `eigh`). |
| `src/qdrop/qadaprune.py` *(new, baseline)* | QAdaPrune gradient-difference freezing — the **mandatory comparison baseline**, implemented in the same hook framework. |
| `src/qdrop/session.py` | `algorithm == "qfi"` path: `start_epoch` probes+caches+partitions; `process_tensor_grad` applies the truncated NG. Legacy `pruning`/`dropout` untouched. |
| `src/qdrop/backends/torch_runtime.py` | Wire the probe at `start_epoch` (needs `qnode` + probe batch via the spec/adapter). |
| `src/training/graph_training.py` | `--algorithm {qfi,qadaprune}` choices; supply a fixed probe batch at epoch start; extend `snapshot_qdrop_state` to log spectrum + condition number + frozen-dim. |
| `src/qml_techniques/qng.py` | Reuse metric-tensor provider wiring. **Branch dependency** — see §7. |

---

## 5. Success criteria & experiment matrix

The method's existence is justified **only if it beats QAdaPrune**. This comparison *is* the paper.

**Baselines (required):** `baseline` (no prune) · **QAdaPrune** (grad-diff freeze) · random/dropout freeze.
**Ablations (each isolates one contribution):**
- diagonal-`F_ii` freeze **vs** full-spectral truncation → isolates contribution 1 (spectrum vs diagonal).
- truncate-only (freeze, plain GD on kept) **vs** truncate + QNG `1/(λ+ε)` → isolates contribution 2 (the unification).

**Primary outcomes:**
- Mean test accuracy: QFI-Drop > QAdaPrune > baseline on the four datasets (or clearly characterize where/why not).
- **`recall` un-pins from 1.0** (the under-fitting signature lifts).

**Trainability evidence:** kept-subspace condition number `log10 κ(F_Kept)` falls over training; epochs-to-best-val ↓; per-epoch val-acc variance ↓.
**Secondary:** fold-variance (`std_accuracy`) ↓.

A run that lifts accuracy but leaves `recall≈1.0`, or that does not beat QAdaPrune, has **not** validated the method.

---

## 6. Testing

- **Unit:** `spectral_partition` ratio correctness; `truncated_natural_gradient` equals the §3.3 sum and zeros the frozen subspace; on a symmetric PSD toy F it matches `U(Λ+εI)⁻¹Uᵀg` with truncation; F-probe matches `qml.metric_tensor` on a 2-qubit toy qnode.
- **Property:** frozen eigendirections receive exactly zero update; kept directions never amplified beyond `1/ε`; partition is a true bipartition.
- **Baseline parity:** QAdaPrune implementation reproduces its freeze behavior on a toy circuit.
- **Pipeline smoke:** QFI-Drop one epoch on synthetic graphs (follow `tests/qdb/test_pipeline.py`): no NaN, shapes intact, probe runs on `default.qubit`.
- **Benchmark:** four datasets × {baseline, QAdaPrune, qfi, ablations}, seeds fixed.

---

## 7. Risks & open implementation decisions

1. **`metric_tensor` on `lightning.qubit` + `adjoint`** (PROTEINS `n_qubits=16`): support uncertain. **Decision:** compute the probe on a `default.qubit` shadow of the qnode, or use a block-diagonal metric. Verify in plan step 1 — this gates everything.
2. **Branch strategy:** `qng.py` lives on `feat/qml_techniques`, not `main`. **Recommend** branching QFI-Drop off `feat/qml_techniques` so QNG/QFI code co-locates.
3. **Eigendecomposition is not differentiated through.** F is computed under `no_grad` once per epoch and treated as a constant preconditioner (as QNG does); we do *not* backprop through `eigh`. Avoids `eigh` gradient instabilities.
4. **QGNN prior-art sweep** (contribution #3) must be completed before submission.
5. **Probe-batch policy:** fixed held-out batch per fold (comparable across epochs) — recommended over a running estimate.
6. **Spectral threshold ρ sensitivity** + **QNG-then-Adam double preconditioning** (matches established `QNGAdam` pattern) — flagged, to be ablated.

---

## 8. Out of scope (YAGNI)

- **Structural gate removal / Quantum Lottery Ticket** — higher compute/risk; revisit only if spectral QFI-Drop validates and a compression story is wanted.
- **Compression-for-NISQ objective** — circuits are tiny.
- **Global blind re-initialization** — already tried (Phase 2B) and regressed.
- **Rewriting the legacy `pruning`/`dropout` paths** — left intact for back-compat; QFI-Drop is a new `--algorithm`.
- **Graph-topology-aware importance** (degree/substructure weighting) — a candidate *next* contribution, deliberately deferred to keep this paper's claims focused.
