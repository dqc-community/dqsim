"""
Performance benchmark: dqsim vs Qiskit Aer.

Run with:  pytest benchmarking/benchmarking_suite.py -v -s

Reports per-shot latency (µs) averaged over SHOTS:
  dqsim-sv     : StatevectorSimulator.simulate_shots(SHOTS) / SHOTS
  dqsim-pblock : PBlockSimulator.simulate_shots(SHOTS) / SHOTS
  dqsim-stab   : StabilizerSimulator.simulate_shots(SHOTS) / SHOTS
  aer          : AerSimulator().run(qc, shots=SHOTS) / SHOTS
"""

from __future__ import annotations

import json
import os
import pathlib
import time
from dataclasses import dataclass

import yaml

import qasmpi
from bosonic_model.qasm import Translator
from bosonic_converters import CircuitConverters
from bosonic_sdk.distributor.distributors.disqco_distributor import DisqcoDistributor
from bosonic_sdk.simulation.simulator import Simulator as BosonicSimulator

from dqsim import PBlockSimulator, StabilizerSimulator, StatevectorSimulator

_PROFILE_DIR = pathlib.Path("dqsim_profiles")


def _read_new_profile(prefix: str, before: set[str]) -> dict | None:
    """Read the profile JSON written by a simulate_shots(profile=True) call.

    Rust writes shots_profile_<unix_ns>.json with no return value pointing at the
    path, so we snapshot the directory listing before the call and diff after —
    the new file (there's exactly one per call) is the profile we just produced.
    """
    if not _PROFILE_DIR.is_dir():
        return None
    after = {p.name for p in _PROFILE_DIR.glob(f"{prefix}_shots_profile_*.json")}
    new = after - before
    if not new:
        return None
    return json.loads((_PROFILE_DIR / sorted(new)[-1]).read_text())


def _snapshot(prefix: str) -> set[str]:
    if not _PROFILE_DIR.is_dir():
        return set()
    return {p.name for p in _PROFILE_DIR.glob(f"{prefix}_shots_profile_*.json")}


_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}


def _resolve_profile_flag(yaml_default: bool) -> bool:
    """DQSIM_PROFILE env var (set via `make perf PROFILE=true`) overrides the
    YAML config's `profile` key when present; otherwise the YAML value wins."""
    raw = os.environ.get("DQSIM_PROFILE")
    if raw is None:
        return yaml_default
    normalized = raw.strip().lower()
    if normalized in _TRUTHY:
        return True
    if normalized in _FALSY:
        return False
    raise ValueError(
        f"Invalid DQSIM_PROFILE value {raw!r}; use one of {sorted(_TRUTHY | _FALSY)}"
    )



@dataclass
class CircuitSpec:
    name: str
    nodes: int
    qubits_per_node: int


class BenchmarkConfig:
    def __init__(self, path: pathlib.Path) -> None:
        raw = yaml.safe_load(path.read_text())
        self.seed: int = raw["config"]["seed"]
        self.shots: int = raw["config"]["shots"]
        self.profile: bool = _resolve_profile_flag(raw["config"].get("profile", False))
        self.circuits: list[CircuitSpec] = [
            CircuitSpec(c["name"], c["nodes"], c["qubits_per_node"])
            for c in self._flatten_circuits(raw["circuits"])
        ]

    @staticmethod
    def _flatten_circuits(entries: list[dict]) -> list[dict]:
        circuits = []
        for entry in entries:
            if "name" in entry:
                circuits.append(entry)
                continue
            for group in entry.values():
                circuits.extend(group)
        return circuits



@dataclass
class DistributedTimings:
    sv_shots_ms: float
    pblock_lowered_shots_ms: float
    pblock_symbolic_shots_ms: float | None
    aer_shots_ms: float
    sv_profile: dict | None = None
    pblock_lowered_profile: dict | None = None


@dataclass
class MonolithicTimings:
    sv_shots_ms: float
    stabilizer_shots_ms: float | None
    sv_profile: dict | None = None
    stabilizer_profile: dict | None = None


@dataclass
class BenchmarkResult:
    name: str
    num_qubits: int
    distributed_timings: DistributedTimings
    monolithic_timings: MonolithicTimings



class BenchmarkRunner:
    def __init__(self, config: BenchmarkConfig) -> None:
        self._config = config
        self._dist = DisqcoDistributor()

    def run_all(self) -> list[BenchmarkResult]:
        results = []
        total = len(self._config.circuits)
        for idx, spec in enumerate(self._config.circuits, 1):
            print(f"  [{idx}/{total}] {spec.name} ...", flush=True)
            results.append(self._run_one(spec))
        return results

    def _run_one(self, spec: CircuitSpec) -> BenchmarkResult:
        circuit = Translator().from_qasm(qasmpi.get_circuit(spec.name))
        n = max(r.base + r.size for r in circuit.qregs.values())

        distributed_lowered = self._dist.distribute(
            circuit, nodes=spec.nodes, qubits_per_node=spec.qubits_per_node, lowered=True
        )
        try:
            distributed_symbolic = self._dist.distribute(
                circuit, nodes=spec.nodes, qubits_per_node=spec.qubits_per_node, lowered=False
            )
        except ValueError:
            distributed_symbolic = None

        distributed_as_monolithic = distributed_lowered.as_monolithic_circuit()

        dist_sv_ms, dist_sv_profile = self._time_sv_shots(distributed_as_monolithic)
        pblock_lowered_ms, pblock_lowered_profile = self._time_pblock_shots(distributed_lowered)
        pblock_symbolic_ms, _ = (
            self._time_pblock_shots(distributed_symbolic) if distributed_symbolic else (None, None)
        )
        mono_sv_ms, mono_sv_profile = self._time_sv_shots(circuit)
        stabilizer_ms, stabilizer_profile = self._time_stabilizer_shots(circuit)

        return BenchmarkResult(
            name=spec.name,
            num_qubits=n,
            distributed_timings=DistributedTimings(
                sv_shots_ms=dist_sv_ms,
                pblock_lowered_shots_ms=pblock_lowered_ms,
                pblock_symbolic_shots_ms=pblock_symbolic_ms,
                aer_shots_ms=self._time_aer_shots(distributed_as_monolithic),
                sv_profile=dist_sv_profile,
                pblock_lowered_profile=pblock_lowered_profile,
            ),
            monolithic_timings=MonolithicTimings(
                sv_shots_ms=mono_sv_ms,
                stabilizer_shots_ms=stabilizer_ms,
                sv_profile=mono_sv_profile,
                stabilizer_profile=stabilizer_profile,
            ),
        )

    def _elapsed_ms(self, fn) -> float:
        t0 = time.perf_counter()
        fn()
        return (time.perf_counter() - t0) * 1_000

    def _time_sv_shots(self, circuit) -> tuple[float, dict | None]:
        sim = StatevectorSimulator(seed=self._config.seed)
        profile = self._config.profile
        before = _snapshot("statevector") if profile else set()
        ms = self._elapsed_ms(
            lambda: sim.simulate_shots(circuit, shots=self._config.shots, profile=profile)
        )
        return ms, (_read_new_profile("statevector", before) if profile else None)

    def _time_pblock_shots(self, distributed) -> tuple[float, dict | None]:
        sim = PBlockSimulator(seed=self._config.seed)
        profile = self._config.profile
        before = _snapshot("pblock") if profile else set()
        ms = self._elapsed_ms(
            lambda: sim.simulate_shots(distributed, shots=self._config.shots, profile=profile)
        )
        return ms, (_read_new_profile("pblock", before) if profile else None)

    def _time_stabilizer_shots(self, circuit) -> tuple[float | None, dict | None]:
        sim = StabilizerSimulator(seed=self._config.seed)
        profile = self._config.profile
        before = _snapshot("stabilizer") if profile else set()
        try:
            ms = self._elapsed_ms(
                lambda: sim.simulate_shots(circuit, shots=self._config.shots, profile=profile)
            )
        except RuntimeError as exc:
            if "Unsupported instruction" in str(exc):
                return None, None
            raise
        return ms, (_read_new_profile("stabilizer", before) if profile else None)

    def _time_aer_shots(self, circuit) -> float:
        qc, sim, backend = self._prepare_aer(circuit)
        return self._elapsed_ms(lambda: sim.simulate(qc, backend, shots=self._config.shots))

    def _prepare_aer(self, circuit):
        qc = CircuitConverters.to_qiskit(circuit)
        if qc.num_clbits == 0:
            qc.measure_all()
        sim = BosonicSimulator()
        qc = sim.prepare(qc)
        backend = sim.build_backend("statevector")
        return qc, sim, backend



class BenchmarkReporter:
    _HEADER = (
        f"{'Circuit':<16}  {'Qb':>4}  "
        f"{'sv shot(µs)':>12}  {'pblock(lowered=T)':>18}  {'pblock(lowered=F)':>18}  {'aer shot(µs)':>12}"
    )
    _SEP = "-" * len(_HEADER)
    _STABILIZER_HEADER = (
        f"{'Circuit':<16}  {'Qb':>4}  {'sv shot(µs)':>12}  {'stabilizer shot(µs)':>20}"
    )
    _STABILIZER_SEP = "-" * len(_STABILIZER_HEADER)

    def __init__(self, config: BenchmarkConfig) -> None:
        self._shots = config.shots
        self._profile = config.profile

    def print(self, results: list[BenchmarkResult]) -> None:
        print(f"\n\nPerformance: dqsim vs Qiskit Aer  (SHOTS={self._shots}, single-call timing)\n")
        print(self._HEADER)
        print(self._SEP)
        for r in results:
            print(self._format_row(r))
        print(self._SEP)
        print(f"\n\nPerformance: dqsim StabilizerSimulator on original monolithic circuits  (SHOTS={self._shots})\n")
        print(self._STABILIZER_HEADER)
        print(self._STABILIZER_SEP)
        for r in results:
            print(self._format_stabilizer_row(r))
        print(self._STABILIZER_SEP)
        if self._profile:
            self._print_profile_breakdown(results)

    def _print_profile_breakdown(self, results: list[BenchmarkResult]) -> None:
        print("\n\nsimulate_shots(profile=True) breakdown (ms, totals across all shots)\n")
        for r in results:
            print(f"  {r.name} ({r.num_qubits} qubits):")
            self._print_one_profile("    sv (distributed-as-monolithic)", r.distributed_timings.sv_profile)
            self._print_one_profile("    pblock (lowered)", r.distributed_timings.pblock_lowered_profile)
            self._print_one_profile("    sv (monolithic)", r.monolithic_timings.sv_profile)
            self._print_one_profile("    stabilizer", r.monolithic_timings.stabilizer_profile)

    @staticmethod
    def _print_one_profile(label: str, profile: dict | None) -> None:
        if profile is None:
            return
        extras = ", ".join(
            f"{k}={v}"
            for k, v in profile.items()
            if k not in ("shot_times", "num_shots", "preprocessing_time", "fusion_time", "shots_total_time", "total_time")
        )
        fusion = f", fusion={profile['fusion_time'] * 1000:.3f}" if "fusion_time" in profile else ""
        print(
            f"{label}: preprocessing={profile['preprocessing_time'] * 1000:.3f}{fusion}, "
            f"shots_total={profile['shots_total_time'] * 1000:.3f}, total={profile['total_time'] * 1000:.3f}"
            + (f", {extras}" if extras else "")
        )

    def _format_row(self, r: BenchmarkResult) -> str:
        t = r.distributed_timings
        sv_us = t.sv_shots_ms / self._shots * 1_000
        pblock_lowered_us = t.pblock_lowered_shots_ms / self._shots * 1_000
        pblock_symbolic_us = t.pblock_symbolic_shots_ms / self._shots * 1_000 if t.pblock_symbolic_shots_ms is not None else None
        aer_us = t.aer_shots_ms / self._shots * 1_000

        return (
            f"{r.name:<16}  {r.num_qubits:>4}  "
            f"{sv_us:>12.2f}  {pblock_lowered_us:>18.2f}  {self._fmt_ms(pblock_symbolic_us, 18)}  {aer_us:>12.2f}"
        )

    def _format_stabilizer_row(self, r: BenchmarkResult) -> str:
        sv_us = r.monolithic_timings.sv_shots_ms / self._shots * 1_000
        stabilizer_us = (
            r.monolithic_timings.stabilizer_shots_ms / self._shots * 1_000
            if r.monolithic_timings.stabilizer_shots_ms is not None
            else None
        )
        return (
            f"{r.name:<16}  {r.num_qubits:>4}  "
            f"{sv_us:>12.2f}  {self._fmt_ms(stabilizer_us, 20)}"
        )

    @staticmethod
    def _fmt_ms(v: float | None, w: int) -> str:
        return f"{v:>{w}.2f}" if v is not None else f"{'N/A':>{w}}"




_SUITE_FILE = pathlib.Path(__file__).parent / "benchmarking_suite.yaml"


class TestPerformance:
    def test_benchmark_table(self) -> None:
        config = BenchmarkConfig(_SUITE_FILE)
        results = BenchmarkRunner(config).run_all()
        BenchmarkReporter(config).print(results)
