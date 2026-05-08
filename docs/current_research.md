## Project Architectures and Algorithms — CISlab Research

### Overview

This workspace contains three interrelated projects in quantum machine learning:

| Project | Domain | Framework | Task |
|---------|--------|-----------|------|
| **Q-Drop-Implementation** | Quantum Orthogonal Neural Networks | TensorFlow + PennyLane | Image classification (MNIST, Fashion-MNIST, MedMNIST) |
| **HQGC** | Hybrid Quantum-Classical Graph Neural Networks | PyTorch + PennyLane + PyG | Smart contract vulnerability detection |
| **Q_Drop_Intergration** | (Placeholder) | — | Future integration of Q-Drop into HQGC |

---

## 1. Q-Drop-Implementation

### 1.1 Core Quantum Primitive: RBS Gate

Defined in [rbs_gate.py](Q-Drop-Implementation/src/utils/rbs_gate.py), the **Reconfigurable Beam Splitter (RBS)** gate is a 2-qubit gate with unitary matrix:

```
       | 1     0       0     0 |
RBS(θ) = | 0   cos(θ)  sin(θ)  0 |
       | 0  -sin(θ)  cos(θ)  0 |
       | 0     0       0     1 |
```

It acts on the `|01⟩` and `|10⟩` subspace, preserving Hamming weight — the key property that makes the network orthogonal.

### 1.2 Data Encoding: Vector Loader

`vector_loader()` ([rbs_gate.py:52](Q-Drop-Implementation/src/utils/rbs_gate.py#L52)) encodes a classical vector into a quantum state using a cascade of RBS gates. The process:

1. Applies `PauliX` on wire 0 to initialize `|100...0⟩`
2. Converts the input vector into angular parameters via `convert_array()` — which computes successive `arccos` values from the L2-normalized input
3. Applies a chain of RBS gates: `RBS(α_i)` on wires `(i, i+1)` sequentially

This produces an amplitude-encoded quantum state where the amplitudes correspond to the normalized input vector.

### 1.3 Variational Ansatz: Pyramid Circuit

`pyramid_circuit()` ([rbs_gate.py:66](Q-Drop-Implementation/src/utils/rbs_gate.py#L66)) is the trainable quantum layer. It arranges RBS gates in a **diamond/pyramid pattern** across `2n - 2` layers for `n` qubits:

- Odd layers: RBS gates on even-indexed pairs `(0,1), (2,3), ...`
- Even layers: RBS gates on odd-indexed pairs `(1,2), (3,4), ...`
- The number of gates per layer grows then shrinks (pyramid shape)
- For 6 qubits: **15 trainable parameters** total

This ensures the unitary remains within the orthogonal group `O(n)`, meaning the quantum circuit implements an orthogonal transformation.

### 1.4 HybridModel Architecture

Defined in [orthogonal_nn.py](Q-Drop-Implementation/src/models/orthogonal_nn.py), the full model is:

```
Input (28x28 image)
  → Flatten
  → Dense(6, linear)          [classical pre-processing]
  → Quantum Circuit (6 qubits):
      → vector_loader(input)   [amplitude encoding]
      → pyramid_circuit(θ)     [15 trainable params]
      → Measure ⟨Z⟩ on all 6 wires
  → Dense(1, sigmoid)          [classical post-processing → binary class]
```

Images are preprocessed to 4x4 (16 pixels), then projected to 6 features via the dense layer, matching the 6-qubit circuit. The model uses `tf.map_fn` to process each sample through the quantum circuit individually.

### 1.5 Algorithm: Scheduled Gradient Pruning

Defined in [pruning.py](Q-Drop-Implementation/src/utils/pruning.py), this is a two-phase training algorithm for quantum parameters:

**Phase 1 — Accumulation** (`accumulate_window` steps, default 10):

- Gradients for quantum weights are accumulated (summed) across steps
- All parameters (classical + quantum) are updated normally via the optimizer

**Phase 2 — Pruning** (`prune_window` steps, default 8):

- The accumulated gradients are **min-max normalized** then passed through `log()` to create sampling logits
- `num_samples = prune_ratio × num_params` indices are drawn via **categorical sampling** (weighted by gradient magnitude)
- Only the sampled parameters receive gradient updates; the rest are zeroed out
- Classical parameters are updated normally
- Accumulated gradients are reset after each pruning step

**Prune Ratio Scheduling** ([pruning.py:141](Q-Drop-Implementation/src/utils/pruning.py#L141)):

- `PruneScheduler` exponentially increases `prune_ratio` by factor `e^0.1` every 5 pruning steps
- Capped at 1.0 (all parameters updated)
- This means early training prunes aggressively (fewer parameters updated), gradually relaxing to full updates

The intuition: low-magnitude accumulated gradients indicate less important parameters, so they are pruned probabilistically. Over time, as the model converges, pruning relaxes.

### 1.6 Algorithm: Dynamic Quantum Dropout

Defined in [dropout.py](Q-Drop-Implementation/src/utils/dropout.py) and implemented inline in the notebooks.

**Wire-level dropout masks** are predefined binary vectors mapping each of the 15 RBS parameters to the qubit wire it belongs to:

```python
theta_wire_0 = [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1]  # params on wire 0
theta_wire_1 = [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0]  # params on wire 1
```

**Dropout mechanism** (from the notebook training loop):

1. At each epoch, a coin flip (`p=0.5`) decides whether to apply dropout (never on epoch 1)
2. If applied: gradients corresponding to dropped wire parameters are **zeroed out** (not the weights, but their gradients)
3. Additionally, the **output measurement** on the dropped wire is zeroed out in the forward pass
4. Supports dropping 1 or 2 wires simultaneously

This is analogous to classical dropout but operates at the **quantum wire level** — entire qubit pathways are disabled during training, forcing the circuit to build redundant representations.

### 1.7 Training Protocol

From the notebooks, both algorithms use:

- **Cosine decay** learning rate: 0.3 → 0.03 (or 0.0003)
- **Adam** optimizer
- **Monte Carlo resampling**: 10 simulations with confidence intervals
- **Binary cross-entropy** loss
- Datasets: 2-class subsets (e.g., MNIST digits 3 vs 6), images downsampled to **4x4**
- Small training sets (~500 samples) — typical for quantum ML experiments

---

## 2. HQGC — Hybrid Quantum-Classical Graph Convolutional Network

### 2.1 Classical Baseline: GCN / GNN

Two classical GCN implementations serve as baselines:

**PyTorch Geometric GNN** ([PyTorch_GCN.py](HQGC-Hybrid-Quantum-Classical-Graph-Convolutional-Network/code/models/PyTorch_GCN.py)):

```
Node features x
  → [GCNConv → LeakyReLU(0.2)] × L layers
  → global_mean_pool (graph-level readout)
  → Linear classifier → 1 output (binary)
```

Uses the standard PyG `GCNConv` layers.

**Custom GCN** ([Custom_GCN_Model.py](HQGC-Hybrid-Quantum-Classical-Graph-Convolutional-Network/code/models/Custom_GCN_Model.py)):
Same architecture but accepts a **pluggable `GCNConvLayer`** — either classical `GCNConv` or quantum `QGCNConv`.

### 2.2 Custom GCNConv Layer

Defined in [custom_gcn_conv.py](HQGC-Hybrid-Quantum-Classical-Graph-Convolutional-Network/code/models/gcn_conv_layers/custom_gcn_conv.py), this implements the standard GCN message-passing:

1. Add self-loops to adjacency
2. Optionally transform features via `Linear(in, out)` (controlled by `no_node_NN` flag)
3. Compute symmetric normalization: `D^{-1/2} A D^{-1/2}`
4. Message passing: `x_j * norm` aggregated via sum
5. Add learned bias

### 2.3 Quantum Node Embedding (QNN)

Defined in [qnn_node_embedding.py](HQGC-Hybrid-Quantum-Classical-Graph-Convolutional-Network/code/models/qnn_node_embedding.py):

```python
def quantum_net(n_qubits, n_layers):
    @qml.qnode(dev, interface="torch")
    def qnode(inputs, weights):
        AngleEmbedding(inputs, wires)        # Encode features as rotation angles
        BasicEntanglerLayers(weights, wires)  # Variational layers with CNOT entanglement
        return [⟨Z_i⟩ for i in range(n_qubits)]
```

- **AngleEmbedding**: Each input feature → `RX(feature_i)` rotation on qubit `i`
- **BasicEntanglerLayers**: `n_layers` repetitions of single-qubit `RX` rotations + ring of CNOT gates
- **Output**: Pauli-Z expectation values on all qubits → `[N, n_qubits]` tensor
- Wrapped as `qml.qnn.TorchLayer` for seamless PyTorch integration
- Number of qubits constrained to powers of 2: {2, 4, 8, 16}, max 16

### 2.4 Quantum GCN Convolution (QGCNConv)

Defined in [qgcn_conv.py](HQGC-Hybrid-Quantum-Classical-Graph-Convolutional-Network/code/models/gcn_conv_layers/qgcn_conv.py), this replaces the classical linear transform in GCN with a quantum circuit:

```
Node features x [N, in_channels]
  → Linear(in_channels, n_qubits)   [dimension reduction, if needed]
  → quantum_net(n_qubits, n_layers) [VQC node embedding]
  → Add self-loops
  → Symmetric normalization
  → Message passing (sum aggregation)
  → Add bias
  → Output [N, n_qubits]
```

The quantum circuit replaces the `Wx` linear transformation with a variational quantum circuit, enabling the model to learn non-linear node embeddings in Hilbert space.

### 2.5 QGCN Full Model

Defined in [quantum_gcn.py](HQGC-Hybrid-Quantum-Classical-Graph-Convolutional-Network/code/models/quantum_gcn.py):

```
Graph input (x, edge_index, batch)
  → [QGCNConv(depth_i) → LeakyReLU(0.2)] × L layers
  → global_mean_pool (graph-level embedding)
  → Linear(n_qubits, output_dims) [classifier]
  → (optional) Linear(1, 1) [readout]
```

Default configuration: `q_depths = [1, 1]` (2 quantum GCN layers, each with depth-1 variational circuits).

### 2.6 Data Pipeline

[load_vulnerability_data.py](HQGC-Hybrid-Quantum-Classical-Graph-Convolutional-Network/code/data/load_vulnerability_data.py) converts JSON-formatted smart contract graphs into PyG `Data` objects:

- **Node features** → `x` tensor
- **Edge list** `[source, edge_type, target]` → `edge_index` (edge types discarded)
- **Binary label** (vulnerable/non-vulnerable) → `y`

### 2.7 Training Configuration

From [train_fast.py](HQGC-Hybrid-Quantum-Classical-Graph-Convolutional-Network/code/train_fast.py):

- **Optimizer**: Adam, lr=0.001
- **Loss**: BCEWithLogitsLoss (binary classification)
- **Early stopping**: patience=10 validation checks, monitoring validation accuracy
- **Mixed precision**: automatic when CUDA available
- **Datasets**: reentrancy, timestamp, integer overflow vulnerabilities
- **Metrics**: Accuracy, Precision, Recall, F1
- **TensorBoard** logging for visualization

---

## 3. Architectural Comparison

| Aspect | Q-Drop (Orthogonal QNN) | HQGC (Quantum GCN) |
|--------|------------------------|---------------------|
| **Quantum gate** | RBS (orthogonal) | RX + CNOT (general unitary) |
| **Encoding** | Amplitude encoding via RBS cascade | Angle embedding (RX rotations) |
| **Ansatz** | Pyramid circuit (preserves orthogonality) | BasicEntanglerLayers (ring topology) |
| **Qubits** | 6 fixed | 2–16 (power of 2, adaptive) |
| **Parameters** | 15 (fixed for 6 qubits) | `n_layers × n_qubits` per layer |
| **Framework** | TensorFlow/Keras + PennyLane | PyTorch + PyG + PennyLane |
| **Optimization** | Scheduled pruning / Dynamic dropout | Standard backpropagation |
| **Task** | Image classification | Graph classification |

---

## 4. Key Algorithmic Innovations

1. **Scheduled Gradient Pruning**: A novel training strategy that alternates between accumulating quantum gradients and probabilistically pruning low-importance parameters, with an exponential schedule that gradually relaxes pruning intensity.

2. **Dynamic Quantum Dropout**: Wire-level dropout for quantum circuits — entire qubit pathways are stochastically disabled during training by masking both forward outputs and backward gradients, analogous to classical dropout but respecting quantum circuit topology.

3. **Quantum GCN Convolution**: Replacing the linear node-feature transformation in GCN with a variational quantum circuit (AngleEmbedding + BasicEntanglerLayers), allowing graph-structured data to be processed through parameterized quantum operations within the message-passing framework.
