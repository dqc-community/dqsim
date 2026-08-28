from __future__ import annotations

import math

import numpy as np
import pytest
from bosonic_model import Circuit, Register
from bosonic_model.instructions import (
    CcxInstruction,
    ConditionalInstruction,
    Condition,
    CxInstruction,
    CyInstruction,
    CzInstruction,
    HInstruction,
    MeasureInstruction,
    ResetInstruction,
    RyInstruction,
    SwapInstruction,
    XInstruction,
)

from dqsim import MpsSimulator, simulate_monolithic


SEED = 42
TOL = 1e-8


def _circuit(n: int, instructions: list, cregs: dict | None = None) -> Circuit:
    return Circuit(
        qregs={"q": Register(name="q", size=n, base=0)},
        cregs=cregs or {},
        instructions=instructions,
    )


def _assert_probs_close(a: dict[int, float], b: dict[int, float], tol: float = TOL) -> None:
    for state in set(a) | set(b):
        assert abs(a.get(state, 0.0) - b.get(state, 0.0)) < tol


def _assert_mps_matches_statevector(circuit: Circuit) -> None:
    sv_result = simulate_monolithic(circuit, mode="state_vector", seed=SEED)
    mps_result = simulate_monolithic(circuit, mode="mps", seed=SEED)
    _assert_probs_close(sv_result.probabilities(), mps_result.probabilities())
    assert np.max(np.abs(sv_result.statevector - mps_result.statevector)) < TOL
    assert abs(sum(mps_result.probabilities().values()) - 1.0) < TOL


class TestMpsSimulation:
    def test_product_state(self) -> None:
        instructions = [
                    HInstruction(qubit=0, qubits=[0]),
                    RyInstruction(qubit=1, qubits=[1], theta=math.pi / 3, params=[math.pi / 3]),
                    RyInstruction(qubit=2, qubits=[2], theta=math.pi / 5, params=[math.pi / 5]),
                ]

        _assert_mps_matches_statevector(
            _circuit(3,instructions)
        )

    def test_non_adjacent_and_reversed_two_qubit_gates(self) -> None:
        instructions = [
                           HInstruction(qubit=0, qubits=[0]),
                           HInstruction(qubit=3, qubits=[3]),
                           CxInstruction(control=3, target=1, qubits=[3, 1], params=[]),
                           CzInstruction(control=2, target=0, qubits=[2, 0], params=[]),
                           SwapInstruction(a=3, b=0, qubits=[3, 0], params=[]),
                           CxInstruction(control=0, target=2, qubits=[0, 2], params=[]),
                       ]

        _assert_mps_matches_statevector(
            _circuit(4, instructions)
        )

    def test_cy_gate_hint_matches_statevector(self) -> None:
        instructions = [
            HInstruction(qubit=2, qubits=[2]),
            RyInstruction(qubit=0, qubits=[0], theta=0.91, params=[0.91]),
            CyInstruction(control=2, target=0, qubits=[2, 0], params=[]),
            CyInstruction(control=0, target=2, qubits=[0, 2], params=[]),
        ]

        _assert_mps_matches_statevector(_circuit(3, instructions))

    def test_entangled_gates(self) -> None:
        instructions = [
                           HInstruction(qubit=0, qubits=[0]),
                           HInstruction(qubit=2, qubits=[2]),
                           RyInstruction(qubit=0, qubits=[0], theta=0.123, params=[0.123]),
                           CxInstruction(control=2, target=1, qubits=[2, 1], params=[]),
                           CxInstruction(control=1, target=0, qubits=[1, 0], params=[]),
                           SwapInstruction(a=1, b=0, qubits=[1, 0], params=[]),
                           RyInstruction(qubit=2, qubits=[2], theta=0.456, params=[0.456]),
                           CxInstruction(control=2, target=1, qubits=[2, 1], params=[]),
                           SwapInstruction(a=2, b=1, qubits=[2, 1], params=[]),
                           RyInstruction(qubit=1, qubits=[1], theta=0.789, params=[0.789]),
                           CzInstruction(control=1, target=0, qubits=[1, 0], params=[]),
                           CxInstruction(control=0, target=1, qubits=[0, 1], params=[]),
                       ]

        _assert_mps_matches_statevector(
            _circuit(3, instructions)
        )

    def test_mid_circuit_measurement_and_conditional(self) -> None:
        instructions = [
                           XInstruction(qubit=0, qubits=[0]),
                           MeasureInstruction(qubit=0, cbit=0, qubits=[0]),
                           ConditionalInstruction(
                               condition=Condition(creg_base=0, creg_size=1, creg_value=1),
                               op=XInstruction(qubit=1, qubits=[1]),
                               qubits=[1],
                           ),
                       ]

        circuit = _circuit(
            2,
            instructions,
            cregs={"c": Register(name="c", size=1, base=0)},
        )

        result = simulate_monolithic(circuit, mode="mps", seed=SEED)
        assert result.classical_bits == {0: 1}
        assert result.probabilities() == {3: 1.0}

    def test_reset(self) -> None:
        intructions = [
                          XInstruction(qubit=0, qubits=[0]),
                          ResetInstruction(qubit=0, qubits=[0]),
                      ]

        circuit = _circuit(
            1, intructions
        )
        assert simulate_monolithic(circuit, mode="mps", seed=SEED).probabilities() == {0: 1.0}

    def test_truncated_mps_remains_normalized(self) -> None:
        instructions = [
                           HInstruction(qubit=0, qubits=[0]),
                           HInstruction(qubit=1, qubits=[1]),
                           HInstruction(qubit=2, qubits=[2]),
                           CxInstruction(control=0, target=3, qubits=[0, 3], params=[]),
                           CxInstruction(control=1, target=2, qubits=[1, 2], params=[]),
                           RyInstruction(qubit=3, qubits=[3], theta=0.77, params=[0.77]),
                           CxInstruction(control=3, target=1, qubits=[3, 1], params=[]),
                       ]
        circuit = _circuit(
            4, instructions
        )

        probs = MpsSimulator(seed=SEED, max_bond_dimension=1).simulate(circuit).probabilities()
        assert all(p >= -1e-12 for p in probs.values())
        assert abs(sum(probs.values()) - 1.0) < 1e-8

    def test_convenience_function_accepts_mps_bond_dimension(self) -> None:
        circuit = _circuit(
            3,
            [
                HInstruction(qubit=0, qubits=[0]),
                HInstruction(qubit=1, qubits=[1]),
                CxInstruction(control=0, target=2, qubits=[0, 2], params=[]),
                CxInstruction(control=1, target=2, qubits=[1, 2], params=[]),
            ],
        )
        probs = simulate_monolithic(
            circuit,
            mode="mps",
            seed=SEED,
            max_bond_dimension=1,
            truncation_threshold=1e-12,
        ).probabilities()
        assert all(p >= -1e-12 for p in probs.values())
        assert abs(sum(probs.values()) - 1.0) < 1e-8

    def test_fast_two_qubit_kernels_are_profiled(self) -> None:
        circuit = _circuit(
            3,
            [
                HInstruction(qubit=0, qubits=[0]),
                CxInstruction(control=0, target=1, qubits=[0, 1], params=[]),
                CzInstruction(control=1, target=2, qubits=[1, 2], params=[]),
                SwapInstruction(a=0, b=2, qubits=[0, 2], params=[]),
            ],
        )

        _assert_mps_matches_statevector(circuit)
        result = MpsSimulator(seed=SEED).simulate_shots(circuit, shots=2, collect_profile=True)
        profile = result["profile"]

        assert profile["mps_2q_fast_kernels_enabled"] is True
        assert profile["mps_2q_fast_kernel_mode"] == "auto"
        assert profile["mps_2q_fast_kernels_used"] is True
        assert profile["mps_2q_fast_kernel_applications"] > 0
        assert profile["mps_2q_permutation_kernel_applications"] > 0
        assert profile["mps_svd_count"] + profile["mps_svd_skipped_count"] == profile[
            "mps_adjacent_2q_applications"
        ]

    def test_auto_fast_kernels_skip_high_shot_product_permutations(self) -> None:
        circuit = _circuit(
            2,
            [
                XInstruction(qubit=0, qubits=[0]),
                CxInstruction(control=0, target=1, qubits=[0, 1], params=[]),
            ],
        )

        result = MpsSimulator(seed=SEED).simulate_shots(circuit, shots=200, collect_profile=True)
        profile = result["profile"]

        assert profile["mps_2q_fast_kernel_mode"] == "auto"
        assert profile["mps_2q_fast_kernel_applications"] == 0
        assert profile["mps_2q_fast_kernel_auto_skipped_applications"] == 200
        assert profile["mps_svd_count"] == 200
        assert profile["mps_svd_skipped_count"] == 0

    def test_unsupported_multi_qubit_gate_raises(self) -> None:
        circuit = _circuit(
            3,
            [
                HInstruction(qubit=0, qubits=[0]),
                CcxInstruction(control1=0, control2=1, target=2, qubits=[0, 1, 2], params=[]),
            ],
        )
        with pytest.raises(NotImplementedError, match="one- and two-qubit"):
            simulate_monolithic(circuit, mode="mps", seed=SEED)
