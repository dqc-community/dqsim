"""Tests for StatevectorSimulator.simulate_shots profiling (profile=True)."""

from __future__ import annotations

import json

from bosonic_model import Circuit, CxInstruction, HInstruction, Register

from dqsim import StatevectorSimulator


def _bell_circuit() -> Circuit:
    return Circuit(
        qregs={"q": Register(name="q", size=2, base=0)},
        instructions=[
            HInstruction(qubit=0),
            CxInstruction(control=0, target=1),
        ],
    )


class TestShotsProfile:
    def test_profile_false_does_not_write_file(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        sim = StatevectorSimulator(seed=42)

        counts = sim.simulate_shots(_bell_circuit(), shots=50)

        assert isinstance(counts, dict)
        assert sum(counts.values()) == 50
        assert not (tmp_path / "dqsim_profiles").exists()

    def test_profile_true_writes_json_file(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        sim = StatevectorSimulator(seed=42)
        shots = 64

        counts = sim.simulate_shots(_bell_circuit(), shots=shots, profile=True)

        assert isinstance(counts, dict)
        assert sum(counts.values()) == shots

        profile_dir = tmp_path / "dqsim_profiles"
        assert profile_dir.is_dir()
        files = list(profile_dir.glob("shots_profile_*.json"))
        assert len(files) == 1

        data = json.loads(files[0].read_text())
        assert data["num_shots"] == shots
        assert len(data["shot_times"]) == shots
        assert all(isinstance(t, float) and t >= 0.0 for t in data["shot_times"])
        for key in ("preprocessing_time", "fusion_time", "shots_total_time", "total_time"):
            assert isinstance(data[key], float)
            assert data[key] >= 0.0
        assert data["shots_total_time"] <= data["total_time"] + 1e-6
