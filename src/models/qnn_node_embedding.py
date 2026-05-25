"""Quantum node embedding utilities for graph models."""

try:
    from .quantum_circuit_adapter import QuantumCircuitAdapter
except ImportError:
    from models.quantum_circuit_adapter import QuantumCircuitAdapter


def quantum_net(n_qubits, n_layers, use_strongly_entangling=False):
    """Create a Q-Drop-compatible quantum node embedding layer."""
    return QuantumCircuitAdapter(n_qubits=n_qubits, n_layers=n_layers, use_strongly_entangling=use_strongly_entangling)
