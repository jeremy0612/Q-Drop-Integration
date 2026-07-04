# Quantum Graph Training Report — MUTAG & PROTEINS

**Branch:** `feat/diffusion_block` | **Commit:** `ffdd534f` | **Run:** [26709901452](https://github.com/jeremy0612/Q-Drop-Integration/actions/runs/26709901452)
**Algorithm:** `dropout` | **Model:** QGCN

## Run Overview

| Property | Value |
|----------|-------|
| Workflow | `QDB Graph Training — MUTAG & PROTEINS (Q-Drop algos only)` |
| Branch | `feat/diffusion_block` |
| Commit | `ffdd534f` |
| Run | [26709901452](https://github.com/jeremy0612/Q-Drop-Integration/actions/runs/26709901452) |
| Algorithm | `dropout` |
| Model | `QGCN` |
| Datasets | MUTAG, PROTEINS |

## Dataset Overview

| Dataset | Source | Graphs | Classes | Node Feature Dim | Task |
|---------|--------|-------:|--------:|-----------------:|------|
| MUTAG | [—](#) | — | — | — | — |
| PROTEINS | [—](#) | — | — | — | — |

## Shared Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 20 |
| Learning rate | 0.0050 |
| Weight decay | 0.0010 |
| Batch size | 16 |
| Q-depths | — |
| Quantum width | 8 |
| Folds | 3 |
| Early stop patience | 15 |
| Validation frequency | — |
| Gradient clip | — |
| LR scheduler | — |
| Class weights | — |
| Q-Drop schedule | — |
| Dropout probability | — |
| Dropped wires / layer | — |
| Forward output masking | — |
| Quantum lr scale | — |
| Seed | 42 |

## Aggregate Results

| Dataset | Accuracy | F1 | ROC AUC | PR AUC | Precision | Recall |
|---------|----------|----|---------|--------|-----------|--------|
| MUTAG | — | — | — | — | — | — |
| PROTEINS | — | — | — | — | — | — |

## MUTAG

### Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 20 |
| Learning rate | 0.0050 |
| Weight decay | 0.0010 |
| Batch size | 16 |
| Q-depths | — |
| Quantum width | 8 |
| Folds | 3 |
| Early stop patience | 15 |
| Validation frequency | — |
| Gradient clip | — |
| LR scheduler | — |
| Class weights | — |
| Quantum lr scale | — |
| Seed | 42 |

### Aggregate Results

| Metric | Mean | Std |
|--------|------|-----|
| ACCURACY | — | — |
| PRECISION | — | — |
| RECALL | — | — |
| F1 | — | — |
| ROC_AUC | — | — |
| PR_AUC | — | — |

### Per-Fold Results

| Fold | Test Loss | ACCURACY | PRECISION | RECALL | F1 | ROC_AUC | PR_AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |

## PROTEINS

### Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 20 |
| Learning rate | 0.0050 |
| Weight decay | 0.0010 |
| Batch size | 16 |
| Q-depths | — |
| Quantum width | 8 |
| Folds | 3 |
| Early stop patience | 15 |
| Validation frequency | — |
| Gradient clip | — |
| LR scheduler | — |
| Class weights | — |
| Quantum lr scale | — |
| Seed | 42 |

### Aggregate Results

| Metric | Mean | Std |
|--------|------|-----|
| ACCURACY | — | — |
| PRECISION | — | — |
| RECALL | — | — |
| F1 | — | — |
| ROC_AUC | — | — |
| PR_AUC | — | — |

### Per-Fold Results

| Fold | Test Loss | ACCURACY | PRECISION | RECALL | F1 | ROC_AUC | PR_AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |

> No baseline found; this run will become the baseline after merge to `main`.

## Visualizations

### Performance Overview

![Performance Overview](https://asset.cml.dev/79b94ba8fb06fb7af04ea8635993edec4d17366a?cml=png&cache-bypass=54bc3494-1a83-4a48-a1f3-8a97c168f204)

### Best-Fold Learning Curves

![Best-Fold Learning Curves](https://asset.cml.dev/76e9a0bd4da175a7eec7246cda41591ea9a0e30c?cml=png&cache-bypass=301a969c-f6d1-4576-a1d6-7c56bbef154b)

![](https://cml.dev/watermark.png#ffdd534fea7a663cbf01f3884f71134559e8ce2c "CML watermark QDB training — dropout")
