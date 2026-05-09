"""QGAT variational quantum circuit for node embedding (RY/RZ + CZ ring entanglement)."""
import pennylane as qml

_DEVICE_CACHE = {}


def _get_device(n_qubits: int):
    if n_qubits in _DEVICE_CACHE:
        return _DEVICE_CACHE[n_qubits]
    try:
        dev = qml.device("lightning.gpu", wires=n_qubits, shots=None)
    except Exception:
        dev = qml.device("lightning.qubit", wires=n_qubits, shots=None)
    _DEVICE_CACHE[n_qubits] = dev
    return dev


def quantum_net(n_qubits: int, n_layers: int, device_name=None, max_qubits: int = 8):
    actual_qubits = min(n_qubits, max_qubits)
    dev = _get_device(actual_qubits)

    @qml.qnode(dev, interface="torch", diff_method="adjoint")
    def circuit(inputs, q_weights):
        qml.AngleEmbedding(inputs, wires=range(actual_qubits), rotation="Y")
        for layer in range(n_layers):
            for i in range(actual_qubits):
                qml.RY(q_weights[layer, i, 0], wires=i)
                qml.RZ(q_weights[layer, i, 1], wires=i)
            for i in range(actual_qubits):
                qml.CZ(wires=[i, (i + 1) % actual_qubits])
        return [qml.expval(qml.PauliZ(i)) for i in range(actual_qubits)]

    return qml.qnn.TorchLayer(circuit, {"q_weights": (n_layers, actual_qubits, 2)})
