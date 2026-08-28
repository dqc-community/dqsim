from __future__ import annotations

import json
import time

from bosonic_model import Circuit, Register
from bosonic_model.instructions import (
    CcxInstruction,
    CxInstruction,
    HInstruction,
    MeasureInstruction,
)

from dqsim import extract_circuit_features, select_simulator, simulate_auto_shots
import dqsim.selection as selection_module


def _circuit(n: int, instructions: list, creg_size: int = 0) -> Circuit:
    cregs = {"c": Register(name="c", size=creg_size, base=0)} if creg_size else {}
    return Circuit(
        qregs={"q": Register(name="q", size=n, base=0)},
        cregs=cregs,
        instructions=instructions,
    )


def test_dense_small_circuit_prefers_statevector() -> None:
    instructions = [HInstruction(qubit=q, qubits=[q]) for q in range(6)]
    for _ in range(5):
        instructions.extend(
            CxInstruction(control=q, target=q + 1, qubits=[q, q + 1], params=[])
            for q in range(5)
        )

    result = select_simulator(_circuit(6, instructions), shots=10, allow_aer=False)

    assert result.selected == "dqsim-sv"
    assert result.features.estimated_mps_entanglement_risk > 0.4


def test_high_qubit_low_entanglement_circuit_prefers_mps() -> None:
    instructions = [HInstruction(qubit=q, qubits=[q]) for q in range(14)]
    instructions.extend(
        CxInstruction(control=q, target=q + 1, qubits=[q, q + 1], params=[])
        for q in range(0, 13, 2)
    )

    result = select_simulator(_circuit(14, instructions), shots=3, allow_aer=False)

    assert result.selected == "dqsim-mps"
    assert result.features.estimated_mps_entanglement_risk < 0.25


def test_distributed_circuit_makes_pblock_eligible() -> None:
    circuit = _circuit(
        4,
        [
            HInstruction(qubit=0, qubits=[0]),
            CxInstruction(control=0, target=1, qubits=[0, 1], params=[]),
            CxInstruction(control=2, target=3, qubits=[2, 3], params=[]),
        ],
    )

    class FakeDistributed:
        qubits_per_node = {0: [0, 1], 1: [2, 3]}

        def as_monolithic_circuit(self) -> Circuit:
            return circuit

    result = select_simulator(distributed=FakeDistributed(), shots=1000, allow_aer=False)

    assert result.selected == "pblock"
    assert result.features.is_distributed
    assert result.features.num_nodes == 2


def test_mps_rejected_for_unsupported_multi_qubit_gate() -> None:
    circuit = _circuit(
        3,
        [
            HInstruction(qubit=0, qubits=[0]),
            CcxInstruction(control1=0, control2=1, target=2, qubits=[0, 1, 2], params=[]),
        ],
    )

    result = select_simulator(circuit, shots=10, allow_aer=False)
    rejected = {candidate.simulator: candidate for candidate in result.rejected}

    assert result.selected == "dqsim-sv"
    assert "dqsim-mps" in rejected
    assert "ccx" in (rejected["dqsim-mps"].rejection_reason or "")


def test_feature_extraction_and_auto_shots_smoke() -> None:
    circuit = _circuit(
        2,
        [
            HInstruction(qubit=0, qubits=[0]),
            CxInstruction(control=0, target=1, qubits=[0, 1], params=[]),
            MeasureInstruction(qubit=0, cbit=0, qubits=[0]),
            MeasureInstruction(qubit=1, cbit=1, qubits=[1]),
        ],
        creg_size=2,
    )

    features = extract_circuit_features(circuit)
    counts, selection = simulate_auto_shots(
        circuit, shots=10, seed=42, allow_aer=False, return_selection=True
    )

    assert features.num_qubits == 2
    assert features.num_two_qubit_gates == 1
    assert selection.selected in {"dqsim-sv", "dqsim-mps"}
    assert sum(counts.values()) == 10

def _write_profile(
    directory,
    name: str,
    simulator: str,
    num_qubits: int,
    num_instructions: int,
    num_shots: int,
    parallel_execution_ms: float,
) -> None:
    (directory / name).write_text(
        json.dumps(
            {
                "simulator": simulator,
                "circuit_variant": "original",
                "num_qubits": num_qubits,
                "num_instructions": num_instructions,
                "num_shots": num_shots,
                "parallel_execution_ms": parallel_execution_ms,
            }
        )
    )


def test_profile_history_adjusts_scores_from_nearby_json(tmp_path) -> None:
    instructions = [HInstruction(qubit=q, qubits=[q]) for q in range(4)]
    instructions.extend(
        CxInstruction(control=q, target=q + 1, qubits=[q, q + 1], params=[])
        for q in range(3)
    )
    circuit = _circuit(4, instructions)

    base = select_simulator(circuit, shots=100, allow_aer=False)
    _write_profile(
        tmp_path,
        "sample_original_sv.json",
        "dqsim_statevector",
        4,
        len(instructions),
        100,
        0.1,
    )
    _write_profile(
        tmp_path,
        "sample_original_mps.json",
        "dqsim_mps",
        4,
        len(instructions),
        100,
        100.0,
    )

    with_history = select_simulator(
        circuit,
        shots=100,
        allow_aer=False,
        use_profile_history=True,
        profile_history_dir=tmp_path,
    )

    base_scores = {candidate.simulator: candidate.score for candidate in base.candidates}
    history_scores = {candidate.simulator: candidate.score for candidate in with_history.candidates}

    assert with_history.profile_history_used
    assert with_history.profile_history_matches == 2
    assert (
        with_history.profile_history_estimates["dqsim-sv"]
        < with_history.profile_history_estimates["dqsim-mps"]
    )
    assert history_scores["dqsim-mps"] > base_scores["dqsim-mps"]
    assert "Profile history adjusted" in with_history.reason


def test_pilot_mode_uses_measured_winner_when_confidence_is_low(monkeypatch) -> None:
    circuit = _circuit(
        2,
        [
            HInstruction(qubit=0, qubits=[0]),
            CxInstruction(control=0, target=1, qubits=[0, 1], params=[]),
        ],
    )
    calls: list[tuple[str, int]] = []

    def fake_run_backend_shots(
        simulator,
        circuit_arg,
        distributed,
        shots,
        seed,
        collect_profile,
        max_bond_dimension,
        truncation_threshold,
    ):
        calls.append((simulator, shots))
        if simulator == "dqsim-sv":
            time.sleep(0.002)
        elif simulator == "dqsim-mps":
            time.sleep(0.0001)
        return {"0": shots}

    monkeypatch.setattr(selection_module, "_run_backend_shots", fake_run_backend_shots)

    counts, selection = simulate_auto_shots(
        circuit,
        shots=10,
        seed=7,
        allow_aer=False,
        return_selection=True,
        selection_mode="pilot",
        confidence_threshold=1.0,
        pilot_shots=2,
    )

    assert selection.selection_mode == "pilot"
    assert selection.pilot_used
    assert selection.pilot_shots == 2
    assert selection.selected == "dqsim-mps"
    assert selection.pilot_timings_ms["dqsim-mps"] < selection.pilot_timings_ms["dqsim-sv"]
    assert counts == {"0": 10}
    assert ("dqsim-sv", 2) in calls
    assert ("dqsim-mps", 2) in calls
    assert ("dqsim-mps", 10) in calls

