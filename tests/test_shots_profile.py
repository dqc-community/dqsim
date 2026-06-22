"""Tests for simulate_shots(profile=True) on each native Rust simulator."""

from __future__ import annotations

import json

import pytest
from bosonic_model import Circuit, Register
from bosonic_model.instructions import CxInstruction, HInstruction, MeasureInstruction
from bosonic_sdk.distributor.distributors.bosonic_distributor import BosonicDistributor

from dqsim import MpsSimulator, PBlockSimulator, StabilizerSimulator, StatevectorSimulator

SEED = 42


def _bell_circuit() -> Circuit:
    return Circuit(
        qregs={"q": Register(name="q", size=2, base=0)},
        cregs={"c": Register(name="c", size=2, base=0)},
        instructions=[
            HInstruction(qubit=0, qubits=[0]),
            CxInstruction(control=0, target=1, qubits=[0, 1], params=[]),
            MeasureInstruction(qubit=0, cbit=0, qubits=[0]),
            MeasureInstruction(qubit=1, cbit=1, qubits=[1]),
        ],
    )


def _distributed_bell_circuit():
    distributed = BosonicDistributor().distribute(
        _bell_circuit(), nodes=2, qubits_per_node=1
    )
    return distributed


def _assert_common_fields(data: dict, shots: int) -> None:
    assert data["num_shots"] == shots
    assert len(data["shot_times"]) == shots
    assert all(isinstance(t, float) and t >= 0.0 for t in data["shot_times"])
    for key in ("preprocessing_time", "shots_total_time", "total_time"):
        assert isinstance(data[key], float)
        assert data[key] >= 0.0
    assert data["shots_total_time"] <= data["total_time"] + 1e-6


class TestStatevectorShotsProfile:
    def test_profile_false_does_not_write_file(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        sim = StatevectorSimulator(seed=SEED)
        counts = sim.simulate_shots(_bell_circuit(), shots=50)
        assert sum(counts.values()) == 50
        assert not (tmp_path / "dqsim_profiles").exists()

    def test_profile_true_writes_json_file(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        sim = StatevectorSimulator(seed=SEED)
        shots = 64

        counts = sim.simulate_shots(_bell_circuit(), shots=shots, profile=True)
        assert sum(counts.values()) == shots

        files = list((tmp_path / "dqsim_profiles").glob("statevector_shots_profile_*.json"))
        assert len(files) == 1
        data = json.loads(files[0].read_text())
        _assert_common_fields(data, shots)
        assert isinstance(data["fusion_time"], float) and data["fusion_time"] >= 0.0


class TestMpsShotsProfile:
    def test_profile_false_does_not_write_file(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        sim = MpsSimulator(seed=SEED)
        counts = sim.simulate_shots(_bell_circuit(), shots=50)
        assert sum(counts.values()) == 50
        assert not (tmp_path / "dqsim_profiles").exists()

    def test_profile_true_writes_json_file(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        sim = MpsSimulator(seed=SEED)
        shots = 64

        counts = sim.simulate_shots(_bell_circuit(), shots=shots, profile=True)
        assert sum(counts.values()) == shots

        files = list((tmp_path / "dqsim_profiles").glob("mps_shots_profile_*.json"))
        assert len(files) == 1
        data = json.loads(files[0].read_text())
        _assert_common_fields(data, shots)
        assert isinstance(data["svd_calls"], int) and data["svd_calls"] > 0
        assert isinstance(data["svd_time"], float) and data["svd_time"] >= 0.0


class TestStabilizerShotsProfile:
    def test_profile_false_does_not_write_file(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        sim = StabilizerSimulator(seed=SEED)
        counts = sim.simulate_shots(_bell_circuit(), shots=50)
        assert sum(counts.values()) == 50
        assert not (tmp_path / "dqsim_profiles").exists()

    def test_profile_true_writes_json_file(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        sim = StabilizerSimulator(seed=SEED)
        shots = 64

        counts = sim.simulate_shots(_bell_circuit(), shots=shots, profile=True)
        assert sum(counts.values()) == shots

        files = list((tmp_path / "dqsim_profiles").glob("stabilizer_shots_profile_*.json"))
        assert len(files) == 1
        data = json.loads(files[0].read_text())
        _assert_common_fields(data, shots)
        assert isinstance(data["measure_calls"], int) and data["measure_calls"] > 0
        assert isinstance(data["measure_time"], float) and data["measure_time"] >= 0.0


class TestPBlockShotsProfile:
    def test_profile_false_does_not_write_file(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        sim = PBlockSimulator(seed=SEED)
        counts = sim.simulate_shots(_distributed_bell_circuit(), shots=50)
        assert sum(counts.values()) == 50
        assert not (tmp_path / "dqsim_profiles").exists()

    def test_profile_true_writes_json_file(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        sim = PBlockSimulator(seed=SEED)
        shots = 64

        counts = sim.simulate_shots(_distributed_bell_circuit(), shots=shots, profile=True)
        assert sum(counts.values()) == shots

        files = list((tmp_path / "dqsim_profiles").glob("pblock_shots_profile_*.json"))
        assert len(files) == 1
        data = json.loads(files[0].read_text())
        _assert_common_fields(data, shots)
        assert isinstance(data["fusion_time"], float) and data["fusion_time"] >= 0.0
        assert isinstance(data["merge_calls"], int) and data["merge_calls"] > 0
