from __future__ import annotations

import pytest
import qasmpi
from bosonic_model.qasm import Translator

from dqsim import simulate_monolithic_shots


SEED = 42
SHOTS = 1000
# Comparing two independent 1000-shot empirical distributions. For a binary
# event with p=0.5, the standard deviation of the difference is about 0.022.
TOL = 0.12

_CLIFFORD_CIRCUITS = [
    ("deutsch_n2", 2, 1),
    ("grover_n2", 2, 1),
    ("iswap_n2", 2, 1),
    ("qrng_n4", 2, 2),
    ("cat_state_n4", 2, 2),
    ("hs4_n4", 2, 2),
    ("lpn_n5", 3, 2),
    ("bb84_n8", 2, 4),
]


def _from_qasmpi(name: str):
    return Translator().from_qasm(qasmpi.get_circuit(name))


def _count_probs(counts: dict[str, int]) -> dict[int, float]:
    return {int(bits, 2): count / SHOTS for bits, count in counts.items()}


def _assert_distributions_match(
    statevector: dict[int, float],
    stabilizer: dict[int, float],
    *,
    circuit_name: str,
) -> None:
    for state in set(statevector) | set(stabilizer):
        p_sv = statevector.get(state, 0.0)
        p_stab = stabilizer.get(state, 0.0)
        assert abs(p_sv - p_stab) < TOL, (
            f"{circuit_name} state {state:b}: "
            f"statevector={p_sv:.4f}, stabilizer={p_stab:.4f}, "
            f"diff={abs(p_sv - p_stab):.4f} > tol={TOL}"
        )


class TestStabilizerSimulation:
    @pytest.mark.parametrize(
        "name,nodes,qpn", _CLIFFORD_CIRCUITS, ids=[t[0] for t in _CLIFFORD_CIRCUITS]
    )
    def test_clifford_qasmbench_shots_match_statevector(
        self, name: str, nodes: int, qpn: int
    ) -> None:
        del nodes, qpn
        circuit = _from_qasmpi(name)

        statevector_counts = simulate_monolithic_shots(
            circuit, mode="state_vector", shots=SHOTS, seed=SEED
        )
        stabilizer_counts = simulate_monolithic_shots(
            circuit, mode="stabilizer", shots=SHOTS, seed=SEED
        )

        _assert_distributions_match(
            _count_probs(statevector_counts),
            _count_probs(stabilizer_counts),
            circuit_name=name,
        )
