from __future__ import annotations

import json
import math

from bosonic_model import (
    Circuit,
    Condition,
    ConditionalInstruction,
    DistributedCircuit,
    GateInstruction,
    Register,
)
from bosonic_model.instructions import CxInstruction, HInstruction, MeasureInstruction, RzzInstruction

from dqsim import PBlockSimulator, StatevectorSimulator

SEED = 42
SHOTS = 1000


def _original_circuit() -> Circuit:
    return Circuit(
        qregs={"q": Register(name="q", size=2, base=0)},
        cregs={"c": Register(name="c", size=2, base=0)},
        instructions=[
            HInstruction(qubit=0, qubits=[0]),
            MeasureInstruction(qubit=0, cbit=0, qubits=[0]),
            ConditionalInstruction(
                condition=Condition(creg_base=0, creg_size=1, creg_value=1),
                op=CxInstruction(control=0, target=1, qubits=[0, 1], params=[]),
                qubits=[0, 1],
            ),
            MeasureInstruction(qubit=1, cbit=1, qubits=[1]),
        ],
    )


def _symbolic_distributed_circuit() -> DistributedCircuit:
    h = HInstruction(qubit=0, qubits=[0])
    measure_control = MeasureInstruction(qubit=0, cbit=0, qubits=[0])
    conditional = ConditionalInstruction(
        condition=Condition(creg_base=0, creg_size=1, creg_value=1),
        op=GateInstruction(name="remote_cx", qubits=[0, 1], params=[], opaque=True),
        qubits=[0, 1],
    )
    measure_target = MeasureInstruction(qubit=1, cbit=1, qubits=[1])

    qregs = {"q": Register(name="q", size=2, base=0)}
    cregs = {"c": Register(name="c", size=2, base=0)}
    distributed = DistributedCircuit(
        qubits_per_node={0: [0], 1: [1]},
        circuits={
            0: Circuit(qregs=qregs, cregs=cregs, instructions=[h, measure_control, conditional]),
            1: Circuit(qregs=qregs, cregs=cregs, instructions=[conditional, measure_target]),
        },
    )
    distributed._instruction_index = {
        id(h): 0,
        id(measure_control): 1,
        id(conditional): 2,
        id(measure_target): 3,
    }
    return distributed


def _original_parametric_circuit() -> Circuit:
    return Circuit(
        qregs={"q": Register(name="q", size=2, base=0)},
        cregs={"c": Register(name="c", size=2, base=0)},
        instructions=[
            HInstruction(qubit=0, qubits=[0]),
            MeasureInstruction(qubit=0, cbit=0, qubits=[0]),
            HInstruction(qubit=1, qubits=[1]),
            ConditionalInstruction(
                condition=Condition(creg_base=0, creg_size=1, creg_value=1),
                op=RzzInstruction(
                    a=0,
                    b=1,
                    theta=math.pi,
                    params=[math.pi],
                    qubits=[0, 1],
                ),
                qubits=[0, 1],
            ),
            HInstruction(qubit=1, qubits=[1]),
            MeasureInstruction(qubit=1, cbit=1, qubits=[1]),
        ],
    )


def _symbolic_parametric_distributed_circuit() -> DistributedCircuit:
    h_control = HInstruction(qubit=0, qubits=[0])
    measure_control = MeasureInstruction(qubit=0, cbit=0, qubits=[0])
    h_before = HInstruction(qubit=1, qubits=[1])
    conditional = ConditionalInstruction(
        condition=Condition(creg_base=0, creg_size=1, creg_value=1),
        op=GateInstruction(
            name="remote_rzz",
            qubits=[0, 1],
            params=[math.pi],
            opaque=True,
        ),
        qubits=[0, 1],
    )
    h_after = HInstruction(qubit=1, qubits=[1])
    measure_target = MeasureInstruction(qubit=1, cbit=1, qubits=[1])

    qregs = {"q": Register(name="q", size=2, base=0)}
    cregs = {"c": Register(name="c", size=2, base=0)}
    distributed = DistributedCircuit(
        qubits_per_node={0: [0], 1: [1]},
        circuits={
            0: Circuit(
                qregs=qregs,
                cregs=cregs,
                instructions=[h_control, measure_control, conditional],
            ),
            1: Circuit(
                qregs=qregs,
                cregs=cregs,
                instructions=[h_before, conditional, h_after, measure_target],
            ),
        },
    )
    distributed._instruction_index = {
        id(h_control): 0,
        id(measure_control): 1,
        id(h_before): 2,
        id(conditional): 3,
        id(h_after): 4,
        id(measure_target): 5,
    }
    return distributed


def test_pblock_simulates_symbolic_remote_conditional_cx() -> None:
    original = StatevectorSimulator(seed=SEED).simulate_shots(
        _original_circuit(),
        shots=SHOTS,
    )
    symbolic = PBlockSimulator(seed=SEED).simulate_shots(
        _symbolic_distributed_circuit(),
        shots=SHOTS,
    )

    assert symbolic == original


def test_pblock_profiles_symbolic_remote_conditional_cx(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    counts = PBlockSimulator(seed=SEED).simulate_shots(
        _symbolic_distributed_circuit(),
        shots=64,
        profile=True,
    )

    assert sum(counts.values()) == 64
    files = list((tmp_path / "dqsim_profiles").glob("pblock_shots_profile_*.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text())
    assert data["num_shots"] == 64
    assert isinstance(data["merge_calls"], int)
    assert data["merge_calls"] > 0


def test_pblock_simulates_symbolic_remote_parametric_conditional() -> None:
    original = StatevectorSimulator(seed=SEED).simulate_shots(
        _original_parametric_circuit(),
        shots=SHOTS,
    )
    symbolic = PBlockSimulator(seed=SEED).simulate_shots(
        _symbolic_parametric_distributed_circuit(),
        shots=SHOTS,
    )

    assert symbolic == original
