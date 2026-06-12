"""
Correctness tests: dqsim statevector and pblock vs Qiskit Aer statevector.

For each QASMBench circuit, distributes using DisqcoDistributor (lowered=True),
then simulates with SHOTS shots via three independent paths:

  1. dqsim statevector   — simulate_monolithic_shots on dist_circuit_as_mono
  2. dqsim pblock        — simulate_distributed_shots on the DistributedCircuit
  3. Qiskit Aer SV       — AerSimulator(statevector) on dist_circuit_as_mono

All three paths operate on the same distributed circuit representation so qubit
ordering is consistent. Data-qubit marginals (Q-prefixed registers) are compared
within statistical tolerance.
"""

from __future__ import annotations

import pytest
import qasmpi
from bosonic_converters import CircuitConverters
from bosonic_model.qasm import QasmError, Translator
from bosonic_converters.remote_link import (
    RemoteLinkGatePhiPlus,
    RemoteLinkGatePsiMinus,
    RemoteLinkGatePsiPlus,
)
from bosonic_sdk.distributor.distributors.disqco_distributor import DisqcoDistributor
from qiskit.circuit import CircuitInstruction
from qiskit.circuit.library import UnitaryGate
from qiskit.result import marginal_counts
from qiskit_aer import AerSimulator

from dqsim import simulate_distributed_shots, simulate_monolithic_shots

SEED = 42
SHOTS = 1000
# σ_max = sqrt(0.25 / SHOTS) ≈ 0.0158 for 1000 shots; TOL = 5σ ≈ 0.079
TOL = 0.08

_BENCH_CIRCUITS = [
    ("deutsch_n2",      2, 1),
    ("toffoli_n3",      2, 3),
    ("adder_n4",        2, 2),
    ("qft_n4",          2, 2),
    ("bell_n4",         2, 2),
    # ("qaoa_n6",         3, 3),
    # ("qpe_n9",          3, 4),
    # ("ising_n10",       2, 5),
    # ("qft_n18",         2, 9),
    # ("square_root_n18", 2, 9),
    # ("dnn_n16",         2, 8),
    # ("cc_n12",          2, 6),
    # ("bv_n14",          2, 7),
]

_distributor = DisqcoDistributor()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _data_qubit_indices(monolithic) -> list[int]:
    """Physical qubit indices belonging to data (Q-prefixed) registers.

    DisqcoDistributor names data registers Q<node>_<orig> and comm registers
    C<node>_<name>. We only marginalise over the former.
    """
    indices = []
    for reg in monolithic.qregs.values():
        if reg.name.startswith("Q"):
            indices.extend(range(reg.base, reg.base + reg.size))
    return sorted(indices)


def _data_cbit_indices(monolithic, data_phys: set[int]) -> list[int]:
    """Classical bit indices that record measurements of data (Q-prefixed) qubits."""
    cbits: set[int] = set()
    for inst in monolithic.instructions:
        if getattr(inst, "kind", None) == "measure" and inst.qubit in data_phys:
            cbits.add(inst.cbit)
    return sorted(cbits)


def _marginalise(probs: dict[int, float], data_indices: list[int]) -> dict[int, float]:
    """Reduce a full-cbit probability dict to data-qubit marginals.

    dqsim (format_cbits) and Qiskit both encode states with bit j = cbit j,
    so data_indices (absolute cbit positions) are used directly as bit positions.
    """
    out: dict[int, float] = {}
    for state, p in probs.items():
        data_state = 0
        for bit, q in enumerate(data_indices):
            if (state >> q) & 1:
                data_state |= 1 << bit
        out[data_state] = out.get(data_state, 0.0) + p
    return out


def _sv_marginals(dist_circuit_as_mono, data_cbit_indices: list[int]) -> dict[int, float]:
    counts = simulate_monolithic_shots(dist_circuit_as_mono, shots=SHOTS, seed=SEED)
    full = {int(bs, 2): n / SHOTS for bs, n in counts.items()}
    return _marginalise(full, data_cbit_indices)


def _pblock_marginals(dist, data_cbit_indices: list[int]) -> dict[int, float]:
    counts = simulate_distributed_shots(dist, shots=SHOTS, seed=SEED)
    full = {int(bs, 2): n / SHOTS for bs, n in counts.items()}
    return _marginalise(full, data_cbit_indices)


def _substitute_remote_gates(qc) -> None:
    """Replace remote-link opaque gates with their unitary matrices in-place."""
    phi_plus = UnitaryGate(RemoteLinkGatePhiPlus().to_matrix(), label="remote_link_phi_plus")
    psi_minus = UnitaryGate(RemoteLinkGatePsiMinus().to_matrix(), label="remote_link_psi_minus")
    psi_plus = UnitaryGate(RemoteLinkGatePsiPlus().to_matrix(), label="remote_link_psi_plus")
    for i, ci in enumerate(qc.data):
        op = ci.operation
        if isinstance(op, RemoteLinkGatePhiPlus) or op.name == "remote_epr":
            qc.data[i] = CircuitInstruction(phi_plus, ci.qubits, ci.clbits)
        elif isinstance(op, RemoteLinkGatePsiMinus):
            qc.data[i] = CircuitInstruction(psi_minus, ci.qubits, ci.clbits)
        elif isinstance(op, RemoteLinkGatePsiPlus):
            qc.data[i] = CircuitInstruction(psi_plus, ci.qubits, ci.clbits)


def _aer_marginals(dist_circuit_as_mono, data_cbit_indices: list[int]) -> dict[int, float]:
    qc = CircuitConverters.to_qiskit(dist_circuit_as_mono)
    for _ in range(3):
        if any(inst.operation.definition for inst in qc.data):
            qc = qc.decompose()
        else:
            break
    _substitute_remote_gates(qc)
    backend = AerSimulator(method="statevector")
    result = backend.run(qc, shots=SHOTS, seed_simulator=SEED).result()
    raw_counts = marginal_counts(result.get_counts(), data_cbit_indices)
    return {int(bs.replace(" ", ""), 2): count / SHOTS for bs, count in raw_counts.items()}


def _assert_close(
    a: dict[int, float],
    b: dict[int, float],
    label_a: str,
    label_b: str,
    tol: float = TOL,
) -> None:
    for state in set(a) | set(b):
        pa, pb = a.get(state, 0.0), b.get(state, 0.0)
        assert abs(pa - pb) < tol, (
            f"|{state:b}⟩: {label_a}={pa:.4f}, {label_b}={pb:.4f}, "
            f"diff={abs(pa - pb):.4f} > tol={tol}"
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCorrectness:
    """Each circuit is distributed once; sv, pblock, and Aer are compared in one shot."""

    @pytest.mark.parametrize(
        "name,nodes,qpn", _BENCH_CIRCUITS, ids=[t[0] for t in _BENCH_CIRCUITS]
    )
    def test_sv_and_pblock_match_aer(
        self, name: str, nodes: int, qpn: int
    ) -> None:
        circuit = Translator().from_qasm(qasmpi.get_circuit(name))

        try:
            dist = _distributor.distribute(
                circuit, nodes=nodes, qubits_per_node=qpn, lowered=True
            )
        except (ValueError, NotImplementedError) as exc:
            pytest.skip(f"distribution failed: {exc}")

        mono = dist.as_monolithic_circuit()
        data_indices = _data_qubit_indices(mono)

        assert data_indices, (
            f"no Q-prefixed registers found in distributed circuit for {name}; "
            "DisqcoDistributor may have changed its register naming convention"
        )

        data_cbit_indices = _data_cbit_indices(mono, set(data_indices))

        sv = _sv_marginals(mono, data_cbit_indices)
        pblock = _pblock_marginals(dist, data_cbit_indices)
        aer = _aer_marginals(mono, data_cbit_indices)

        _assert_close(sv, aer, "dqsim-sv", "aer")
        _assert_close(pblock, aer, "dqsim-pblock", "aer")
