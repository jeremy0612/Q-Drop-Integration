"""Integrated TensorFlow model for Q-Drop and HQGC MNIST experiments."""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import random
import warnings
from typing import Dict, List, Optional, Sequence

import numpy as np
import pennylane as qml
import tensorflow as tf
from pennylane.templates import AngleEmbedding, BasicEntanglerLayers

from qdrop import QDropConfig, QDropRuntimeFactory
from qdrop.specs.pennylane_tf import PennyLaneTensorFlowSpecFactory

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

DEFAULT_ALGORITHM_PARAMS = {
    "accumulate_window": 10,
    "prune_window": 8,
    "prune_ratio": 0.8,
    "schedule": True,
}
SUPPORTED_ALGORITHMS = {"baseline", "pruning", "dropout", "both"}
def _set_random_seed(random_seed: int) -> None:
    """Seed every RNG used by the integrated model."""
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)
    qml.numpy.random.seed(random_seed)


def _resolve_algorithm_params(
    algorithm_params: Optional[Dict[str, object]],
) -> Dict[str, object]:
    """Merge caller overrides with the default Q-Drop settings."""
    resolved_params = DEFAULT_ALGORITHM_PARAMS.copy()
    if algorithm_params is not None:
        resolved_params.update(algorithm_params)
    return resolved_params


class IntegratedQDropHQGCModel(tf.keras.Model):
    """
    Integrated model combining:
    - Q-Drop scheduled gradient pruning and dynamic dropout
    - HQGC quantum components (AngleEmbedding + BasicEntanglerLayers)
    - MNIST image classification
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        algorithm: str = "pruning",
        algorithm_params: Optional[Dict[str, object]] = None,
        apply_dropout: bool = False,
        random_seed: int = 42,
    ):
        super().__init__()

        if algorithm not in SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"Unsupported algorithm '{algorithm}'. "
                f"Expected one of: {sorted(SUPPORTED_ALGORITHMS)}"
            )

        _set_random_seed(random_seed)

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.algorithm = algorithm
        self.random_seed = random_seed
        self.algorithm_params = _resolve_algorithm_params(algorithm_params)
        self.apply_dropout_flag = apply_dropout

        self.flatten_layer = tf.keras.layers.Flatten()
        self.preprocess_dense = tf.keras.layers.Dense(
            n_qubits,
            activation="relu",
            dtype=tf.float32,
        )
        self.quantum_weights = self.add_weight(
            shape=(n_layers, n_qubits),
            initializer="zeros",
            trainable=True,
            dtype=tf.float32,
            name="quantum_weights",
        )
        self.quantum_device = qml.device("default.qubit", wires=n_qubits)
        self.quantum_circuit = self._build_quantum_circuit()
        self.qdrop_forward_mask = tf.Variable(
            tf.ones((n_qubits,), dtype=tf.float32),
            trainable=False,
            name="qdrop_forward_mask",
        )
        self.postprocess_dense = tf.keras.layers.Dense(32, activation="relu", dtype=tf.float32)
        self.output_dense = tf.keras.layers.Dense(2, activation="softmax", dtype=tf.float32)
        self.qdrop_quantum_layer = PennyLaneTensorFlowSpecFactory.create_adapter(
            layer_id="integrated_quantum_layer",
            parameter=self.quantum_weights,
            num_wires=self.n_qubits,
            mask_builder=self._build_qdrop_mask,
            set_forward_mask=self._set_qdrop_forward_mask,
            supports_forward_mask=True,
        )
        self.qdrop_runtime = self._build_qdrop_runtime()

        # Backward-compatible aliases for existing scripts and docs.
        self.flatten = self.flatten_layer
        self.dense_preprocess = self.preprocess_dense
        self.dense_postprocess = self.postprocess_dense
        self.output_layer = self.output_dense
        self.dev = self.quantum_device
        self.pruning_algorithm = self.qdrop_runtime if self.algorithm in {"pruning", "both"} else None
        self.dropout_algorithm = self.qdrop_runtime if self.algorithm in {"dropout", "both"} else None
        self.pruning_algo = self.pruning_algorithm
        self.dropout_algo = self.dropout_algorithm

    def _build_quantum_circuit(self):
        """Create the PennyLane circuit used during forward passes."""

        @qml.qnode(self.quantum_device, interface="numpy", diff_method="parameter-shift")
        def quantum_circuit(inputs, weights):
            AngleEmbedding(inputs, wires=range(self.n_qubits), rotation="X")
            BasicEntanglerLayers(weights, wires=range(self.n_qubits))
            return [qml.expval(qml.PauliZ(wire_index)) for wire_index in range(self.n_qubits)]

        return quantum_circuit

    def _build_qdrop_runtime(self):
        if self.algorithm == "baseline":
            return None

        return QDropRuntimeFactory.create_tensorflow(
            quantum_layers=self.qdrop_layers(),
            config=QDropConfig(
                algorithm=self.algorithm,
                accumulate_window=self.algorithm_params.get("accumulate_window", 10),
                prune_window=self.algorithm_params.get("prune_window", 8),
                prune_ratio=self.algorithm_params.get("prune_ratio", 0.8),
                schedule=self.algorithm_params.get("schedule", True),
                dropout_prob=1.0 if self.apply_dropout_flag else 0.5,
                n_drop_wires=1,
                enable_forward_mask=True,
            ),
        )

    def _build_qdrop_mask(self, wire_ids: Sequence[int]) -> tf.Tensor:
        mask = np.zeros((self.n_layers, self.n_qubits), dtype=bool)
        for wire_index in wire_ids:
            if 0 <= wire_index < self.n_qubits:
                mask[:, wire_index] = True
        return tf.constant(mask, dtype=tf.bool)

    def _set_qdrop_forward_mask(self, dropout_state) -> None:
        mask = np.ones((self.n_qubits,), dtype=np.float32)
        if dropout_state is not None and dropout_state.enabled:
            for wire_index in dropout_state.dropped_wires:
                if 0 <= wire_index < self.n_qubits:
                    mask[wire_index] = 0.0
        self.qdrop_forward_mask.assign(mask)

    def qdrop_layers(self):
        return [self.qdrop_quantum_layer]

    def _build_pruning_algorithm(self) -> Optional[object]:
        return self.pruning_algorithm

    def _build_dropout_algorithm(self) -> Optional[object]:
        return self.dropout_algorithm

    @staticmethod
    def _replace_nan_values(tensor: tf.Tensor) -> tf.Tensor:
        """Replace NaN values with zeros to keep training numerically stable."""
        return tf.where(tf.math.is_nan(tensor), tf.zeros_like(tensor), tensor)

    @staticmethod
    def _to_numpy(value):
        """Convert TensorFlow values to NumPy in eager-safe fashion."""
        return value.numpy() if tf.executing_eagerly() else np.asarray(value)

    def _run_quantum_batch(self, embedded_inputs: tf.Tensor) -> tf.Tensor:
        """Execute the quantum circuit for each sample in the batch."""
        batch_size = tf.shape(embedded_inputs)[0]
        batch_quantum_outputs: List[tf.Tensor] = []
        quantum_weights_np = self._to_numpy(self.quantum_weights)

        for batch_index in tf.range(batch_size):
            sample_inputs = embedded_inputs[batch_index]
            circuit_output = np.asarray(
                self.quantum_circuit(self._to_numpy(sample_inputs), quantum_weights_np),
                dtype=np.float32,
            )
            circuit_output = circuit_output * self._to_numpy(self.qdrop_forward_mask)
            batch_quantum_outputs.append(tf.constant(circuit_output, dtype=tf.float32))

        return tf.stack(batch_quantum_outputs)

    def _sanitize_gradients(self, gradients: List[Optional[tf.Tensor]]) -> List[tf.Tensor]:
        """Replace missing or NaN gradients with zeros."""
        sanitized_gradients: List[tf.Tensor] = []
        for gradient, variable in zip(gradients, self.trainable_variables):
            if gradient is None:
                sanitized_gradients.append(tf.zeros_like(variable))
            else:
                sanitized_gradients.append(self._replace_nan_values(gradient))
        return sanitized_gradients

    def _sanitize_trainable_variables(self) -> None:
        """Remove NaNs from model weights after each optimization step."""
        for variable in self.trainable_variables:
            variable.assign(self._replace_nan_values(variable))

    def call(self, inputs, training=None):
        del training

        inputs = tf.cast(inputs, tf.float32)
        flattened_inputs = self.flatten_layer(inputs)
        embedded_inputs = self.preprocess_dense(flattened_inputs)
        quantum_outputs = self._run_quantum_batch(embedded_inputs)
        sanitized_outputs = self._replace_nan_values(quantum_outputs)
        postprocessed_outputs = self.postprocess_dense(sanitized_outputs)
        return self.output_dense(postprocessed_outputs)

    def train_step(self, data):
        inputs, labels = data

        if self.qdrop_runtime is not None and self.optimizer is not None:
            self.qdrop_runtime.start_epoch(int(self.optimizer.iterations.numpy()) + 1)

        with tf.GradientTape() as tape:
            predictions = self(inputs, training=True)
            loss = self.compiled_loss(
                labels,
                predictions,
                regularization_losses=self.losses,
            )

        raw_gradients = tape.gradient(loss, self.trainable_variables)
        processed_gradients = raw_gradients
        if self.qdrop_runtime is not None:
            processed_gradients = self.qdrop_runtime.process_gradients(
                raw_gradients,
                self.trainable_variables,
            )

        sanitized_gradients = self._sanitize_gradients(processed_gradients)
        self.optimizer.apply_gradients(zip(sanitized_gradients, self.trainable_variables))
        if self.qdrop_runtime is not None:
            self.qdrop_runtime.after_step()
        self._sanitize_trainable_variables()

        self.compiled_metrics.update_state(labels, predictions)
        metric_results = {metric.name: metric.result() for metric in self.metrics}
        metric_results["loss"] = loss
        return metric_results
