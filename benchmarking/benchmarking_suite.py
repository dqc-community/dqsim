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
class CircuitTimings:
    sv_shots_ms: float
    pblock_lowered_shots_ms: float
    pblock_symbolic_shots_ms: float | None
    stabilizer_shots_ms: float | None
    aer_shots_ms: float


@dataclass
class BenchmarkResult:
    name: str
    num_qubits: int
    timings: CircuitTimings



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

        monolithic = distributed_lowered.as_monolithic_circuit()

        return BenchmarkResult(
            name=spec.name,
            num_qubits=n,
            timings=CircuitTimings(
                sv_shots_ms=self._time_sv_shots(monolithic),
                pblock_lowered_shots_ms=self._time_pblock_shots(distributed_lowered),
                pblock_symbolic_shots_ms=self._time_pblock_shots(distributed_symbolic) if distributed_symbolic else None,
                stabilizer_shots_ms=self._time_stabilizer_shots(circuit),
                aer_shots_ms=self._time_aer_shots(monolithic),
            ),
        )

    def _elapsed_ms(self, fn) -> float:
        t0 = time.perf_counter()
        fn()
        return (time.perf_counter() - t0) * 1_000

    def _time_sv_shots(self, circuit) -> float:
        sim = StatevectorSimulator(seed=self._config.seed)
        return self._elapsed_ms(lambda: sim.simulate_shots(circuit, shots=self._config.shots))

    def _time_pblock_shots(self, distributed) -> float:
        sim = PBlockSimulator(seed=self._config.seed)
        return self._elapsed_ms(lambda: sim.simulate_shots(distributed, shots=self._config.shots))

    def _time_stabilizer_shots(self, circuit) -> float | None:
        sim = StabilizerSimulator(seed=self._config.seed)
        try:
            return self._elapsed_ms(lambda: sim.simulate_shots(circuit, shots=self._config.shots))
        except RuntimeError as exc:
            if "Unsupported instruction" in str(exc):
                return None
            raise

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
        f"{'sv shot(µs)':>12}  {'pblock(lowered=T)':>18}  {'pblock(lowered=F)':>18}  "
        f"{'stabilizer shot(µs)':>14}  {'aer shot(µs)':>12}"
    )
    _SEP = "-" * len(_HEADER)

    def __init__(self, config: BenchmarkConfig) -> None:
        self._shots = config.shots

    def print(self, results: list[BenchmarkResult]) -> None:
        print(f"\n\nPerformance: dqsim vs Qiskit Aer  (SHOTS={self._shots}, single-call timing)\n")
        print(self._HEADER)
        print(self._SEP)
        for r in results:
            print(self._format_row(r))
        print(self._SEP)

    def _format_row(self, r: BenchmarkResult) -> str:
        t = r.timings
        sv_us = t.sv_shots_ms / self._shots * 1_000
        pblock_lowered_us = t.pblock_lowered_shots_ms / self._shots * 1_000
        pblock_symbolic_us = t.pblock_symbolic_shots_ms / self._shots * 1_000 if t.pblock_symbolic_shots_ms is not None else None
        stabilizer_us = t.stabilizer_shots_ms / self._shots * 1_000 if t.stabilizer_shots_ms is not None else None
        aer_us = t.aer_shots_ms / self._shots * 1_000

        return (
            f"{r.name:<16}  {r.num_qubits:>4}  "
            f"{sv_us:>12.2f}  {pblock_lowered_us:>18.2f}  {self._fmt_ms(pblock_symbolic_us, 18)}  "
            f"{self._fmt_ms(stabilizer_us, 14)}  {aer_us:>12.2f}"
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
