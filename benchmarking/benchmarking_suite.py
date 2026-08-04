"""
Performance benchmark: dqsim vs Qiskit Aer.

Run with: pytest benchmarking/benchmarking_suite.py -v -s

The report is split by circuit variant so each row compares simulators running the
same circuit shape:
  original  dqsim SV/MPS and Aer run the original QASMBench circuit
  lowered   dqsim SV/MPS, PBlock, and Aer run the same lowered distributed circuit

Aer table values use Aer's per-experiment simulator ``time_taken`` metadata,
which excludes Python-side job setup/formatting overhead; wall times are kept in
JSON diagnostics. Profiling data is saved to stable JSON files under
benchmarking/profiles/. Each run clears stale JSON files, replaces the current
files, and embeds a comparison section against the previous contents when one
exists.
"""

from __future__ import annotations

import json
import pathlib
import time
from dataclasses import dataclass
from typing import Callable

import qasmpi
import yaml
from bosonic_converters import CircuitConverters
from bosonic_model.qasm import Translator
from bosonic_sdk.distributor.distributors.disqco_distributor import DisqcoDistributor

from dqsim import MpsSimulator, PBlockSimulator, StatevectorSimulator

try:
    from bosonic_converters.remote_link import (
        RemoteLinkGatePhiPlus,
        RemoteLinkGatePsiMinus,
        RemoteLinkGatePsiPlus,
    )
    from qiskit.circuit import CircuitInstruction
    from qiskit.circuit.library import UnitaryGate
    from qiskit_aer import AerSimulator
except ImportError as exc:  # pragma: no cover - depends on optional benchmark extra
    AerSimulator = None
    AER_IMPORT_ERROR = exc
else:
    AER_IMPORT_ERROR = None


AER_SOURCE_URLS = {
    "python_job": "https://github.com/Qiskit/qiskit-aer/blob/0.17.2/qiskit_aer/backends/aerbackend.py#L443-L496",
    "controller_execute": "https://github.com/Qiskit/qiskit-aer/blob/0.17.2/src/controllers/aer_controller.hpp#L604-L607",
    "experiment_time": "https://github.com/Qiskit/qiskit-aer/blob/0.17.2/src/simulators/circuit_executor.hpp#L714-L722",
    "sample_measure": "https://github.com/Qiskit/qiskit-aer/blob/0.17.2/src/simulators/circuit_executor.hpp#L1064-L1070",
    "fusion": "https://github.com/Qiskit/qiskit-aer/blob/0.17.2/src/transpile/fusion.hpp#L942-L945",
    "statevector_apply": "https://github.com/Qiskit/qiskit-aer/tree/0.17.2/src/simulators/statevector",
    "mps_apply": "https://github.com/Qiskit/qiskit-aer/tree/0.17.2/src/simulators/matrix_product_state",
}

DQSIM_SOURCE_LABELS = {
    "dqsim_statevector": "src/monolithic/statevector/simulator.rs::simulate_shots",
    "dqsim_mps": "src/monolithic/mps/simulator.rs::simulate_shots",
    "dqsim_pblock_lowered": "src/distributed/pblock/simulator.rs::simulate_shots",
    "dqsim_pblock_symbolic": "src/distributed/pblock/simulator.rs::simulate_shots",
}

FUNCTION_COMPARISON_NOTE = (
    "Stock Qiskit Aer does not expose per-gate C++ function timings through the "
    "Python Result API. These entries compare the nearest measured phases and "
    "annotate the Aer source function that emits or contains each timing. Exact "
    "function-level timings require external profiling or an instrumented Aer build."
)


@dataclass
class CircuitSpec:
    name: str
    nodes: int
    qubits_per_node: int
    shots: int | None = None


class BenchmarkConfig:
    def __init__(self, path: pathlib.Path) -> None:
        raw = yaml.safe_load(path.read_text())
        self.seed: int = raw["config"]["seed"]
        self.shots: int = int(raw["config"]["shots"])
        if self.shots <= 0:
            raise ValueError("config.shots must be positive")

        circuits: list[CircuitSpec] = []
        for c in raw["circuits"]:
            shots = c.get("shots")
            if shots is not None:
                shots = int(shots)
                if shots <= 0:
                    name = c["name"]
                    raise ValueError(f"{name}.shots must be positive")
            circuits.append(
                CircuitSpec(c["name"], c["nodes"], c["qubits_per_node"], shots)
            )
        self.circuits = circuits


@dataclass
class VariantTimings:
    num_qubits: int
    sv_shots_ms: float | None = None
    mps_shots_ms: float | None = None
    pblock_shots_ms: float | None = None
    aer_statevector_shots_ms: float | None = None
    aer_mps_shots_ms: float | None = None


@dataclass
class BenchmarkResult:
    name: str
    shots: int
    original: VariantTimings
    lowered: VariantTimings
    symbolic_pblock_shots_ms: float | None
    symbolic_num_qubits: int | None
    profiles: dict[str, dict]


class BenchmarkRunner:
    def __init__(self, config: BenchmarkConfig) -> None:
        self._config = config
        self._dist = DisqcoDistributor()
        self._profile_dir = pathlib.Path(__file__).parent / "profiles"
        self._profile_dir.mkdir(exist_ok=True)
        self._previous_profiles = self._load_previous_profiles()

    def run_all(self) -> list[BenchmarkResult]:
        self._clear_profile_dir()
        results = []
        total = len(self._config.circuits)
        for idx, spec in enumerate(self._config.circuits, 1):
            print(f"  [{idx}/{total}] {spec.name} ...", flush=True)
            results.append(self._run_one(spec))
        return results

    def _run_one(self, spec: CircuitSpec) -> BenchmarkResult:
        shots = spec.shots if spec.shots is not None else self._config.shots
        circuit = Translator().from_qasm(qasmpi.get_circuit(spec.name))
        original_qubits = self._num_qubits(circuit)

        distributed_lowered = self._dist.distribute(
            circuit, nodes=spec.nodes, qubits_per_node=spec.qubits_per_node, lowered=True
        )
        lowered_monolithic = distributed_lowered.as_monolithic_circuit()
        lowered_qubits = self._num_qubits(lowered_monolithic)

        profiles: dict[str, dict] = {}

        original_sv_timing, original_sv_profile = self._run_optional(
            "dqsim-sv-original",
            lambda: self._time_sv_shots(circuit, "original", shots),
        )
        self._store_profile(profiles, "original_sv", original_sv_profile)

        original_mps_timing, original_mps_profile = self._run_optional(
            "dqsim-mps-original",
            lambda: self._time_mps_shots(circuit, "original", shots),
        )
        self._store_profile(profiles, "original_mps", original_mps_profile)

        original_aer_sv_timing, original_aer_sv_profile = self._run_optional(
            "aer-statevector-original",
            lambda: self._time_aer_shots(circuit, "statevector", "original", shots),
        )
        self._store_profile(profiles, "original_aer_statevector", original_aer_sv_profile)

        original_aer_mps_timing, original_aer_mps_profile = self._run_optional(
            "aer-mps-original",
            lambda: self._time_aer_shots(circuit, "matrix_product_state", "original", shots),
        )
        self._store_profile(profiles, "original_aer_mps", original_aer_mps_profile)

        lowered_sv_timing, lowered_sv_profile = self._run_optional(
            "dqsim-sv-lowered",
            lambda: self._time_sv_shots(lowered_monolithic, "lowered", shots),
        )
        self._store_profile(profiles, "lowered_sv", lowered_sv_profile)

        lowered_mps_timing, lowered_mps_profile = self._run_optional(
            "dqsim-mps-lowered",
            lambda: self._time_mps_shots(lowered_monolithic, "lowered", shots),
        )
        self._store_profile(profiles, "lowered_mps", lowered_mps_profile)

        lowered_pblock_timing, lowered_pblock_profile = self._run_optional(
            "dqsim-pblock-lowered",
            lambda: self._time_pblock_shots(
                distributed_lowered, "dqsim_pblock_lowered", "lowered", shots
            ),
        )
        self._store_profile(profiles, "lowered_pblock", lowered_pblock_profile)

        lowered_aer_sv_timing, lowered_aer_sv_profile = self._run_optional(
            "aer-statevector-lowered",
            lambda: self._time_aer_shots(lowered_monolithic, "statevector", "lowered", shots),
        )
        self._store_profile(profiles, "lowered_aer_statevector", lowered_aer_sv_profile)

        lowered_aer_mps_timing, lowered_aer_mps_profile = self._run_optional(
            "aer-mps-lowered",
            lambda: self._time_aer_shots(
                lowered_monolithic, "matrix_product_state", "lowered", shots
            ),
        )
        self._store_profile(profiles, "lowered_aer_mps", lowered_aer_mps_profile)

        symbolic_pblock_timing = None
        symbolic_num_qubits = None
        try:
            distributed_symbolic = self._dist.distribute(
                circuit,
                nodes=spec.nodes,
                qubits_per_node=spec.qubits_per_node,
                lowered=False,
            )
        except ValueError as exc:
            print(f"    Skipped symbolic distribution: {exc}", flush=True)
        else:
            symbolic_num_qubits = max(
                (q for qubits in distributed_symbolic.qubits_per_node.values() for q in qubits),
                default=-1,
            ) + 1
            symbolic_pblock_timing, symbolic_pblock_profile = self._run_optional(
                "dqsim-pblock-symbolic",
                lambda: self._time_pblock_shots(
                    distributed_symbolic, "dqsim_pblock_symbolic", "symbolic", shots
                ),
            )
            self._store_profile(profiles, "symbolic_pblock", symbolic_pblock_profile)

        result = BenchmarkResult(
            name=spec.name,
            shots=shots,
            original=VariantTimings(
                num_qubits=original_qubits,
                sv_shots_ms=original_sv_timing,
                mps_shots_ms=original_mps_timing,
                aer_statevector_shots_ms=original_aer_sv_timing,
                aer_mps_shots_ms=original_aer_mps_timing,
            ),
            lowered=VariantTimings(
                num_qubits=lowered_qubits,
                sv_shots_ms=lowered_sv_timing,
                mps_shots_ms=lowered_mps_timing,
                pblock_shots_ms=lowered_pblock_timing,
                aer_statevector_shots_ms=lowered_aer_sv_timing,
                aer_mps_shots_ms=lowered_aer_mps_timing,
            ),
            symbolic_pblock_shots_ms=symbolic_pblock_timing,
            symbolic_num_qubits=symbolic_num_qubits,
            profiles=profiles,
        )
        self._save_profiles(result)
        return result

    @staticmethod
    def _num_qubits(circuit) -> int:
        return max((r.base + r.size for r in circuit.qregs.values()), default=0)

    @staticmethod
    def _run_optional(
        label: str, fn: Callable[[], tuple[float, dict | None]]
    ) -> tuple[float | None, dict | None]:
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - benchmark should continue per backend
            print(f"    Skipped {label}: {type(exc).__name__}: {exc}", flush=True)
            return None, None

    @staticmethod
    def _store_profile(profiles: dict[str, dict], key: str, profile: dict | None) -> None:
        if profile is not None:
            profiles[key] = profile

    @staticmethod
    def _profile_elapsed_ms(profile: dict, fallback_ms: float) -> float:
        total = profile.get("total_time_ms")
        if total is not None:
            return float(total)
        return float(
            profile.get("preprocessing_ms", 0.0)
            + profile.get("gate_fusion_ms", 0.0)
            + profile.get("parallel_execution_ms", fallback_ms)
        )

    def _time_profiled_call(
        self, simulator_name: str, circuit_variant: str, fn: Callable[[], object]
    ) -> tuple[float, dict | None]:
        t0 = time.perf_counter()
        result = fn()
        elapsed_ms = (time.perf_counter() - t0) * 1_000

        profile_obj = result.get("profile") if isinstance(result, dict) else None
        if profile_obj is None:
            return elapsed_ms, None

        profile = dict(profile_obj)
        profile.setdefault("profile_schema", "shots_v1")
        profile.setdefault("simulator", simulator_name)
        profile["circuit_variant"] = circuit_variant
        profile["source_function_breakdown"] = self._build_dqsim_source_function_breakdown(profile)
        return self._profile_elapsed_ms(profile, elapsed_ms), profile

    def _time_sv_shots(
        self, circuit, circuit_variant: str, shots: int
    ) -> tuple[float, dict | None]:
        sim = StatevectorSimulator(seed=self._config.seed)
        return self._time_profiled_call(
            "dqsim_statevector",
            circuit_variant,
            lambda: sim.simulate_shots(
                circuit, shots=shots, collect_profile=True
            ),
        )

    def _time_mps_shots(
        self, circuit, circuit_variant: str, shots: int
    ) -> tuple[float, dict | None]:
        sim = MpsSimulator(seed=self._config.seed)
        return self._time_profiled_call(
            "dqsim_mps",
            circuit_variant,
            lambda: sim.simulate_shots(
                circuit, shots=shots, collect_profile=True
            ),
        )

    def _time_pblock_shots(
        self, distributed, simulator_name: str, circuit_variant: str, shots: int
    ) -> tuple[float, dict | None]:
        sim = PBlockSimulator(seed=self._config.seed)
        return self._time_profiled_call(
            simulator_name,
            circuit_variant,
            lambda: sim.simulate_shots(
                distributed, shots=shots, collect_profile=True
            ),
        )

    def _time_aer_shots(
        self, circuit, method: str, circuit_variant: str, shots: int
    ) -> tuple[float, dict | None]:
        if AerSimulator is None:
            raise RuntimeError(f"qiskit-aer is not available: {AER_IMPORT_ERROR}")

        total_t0 = time.perf_counter()
        prep_t0 = time.perf_counter()
        qc = self._prepare_qiskit_circuit(circuit)
        preprocessing_ms = (time.perf_counter() - prep_t0) * 1_000

        backend_t0 = time.perf_counter()
        backend = AerSimulator(method=method, seed_simulator=self._config.seed)
        backend_setup_ms = (time.perf_counter() - backend_t0) * 1_000

        exec_t0 = time.perf_counter()
        result = backend.run(
            qc, shots=shots, seed_simulator=self._config.seed
        ).result()
        counts = result.get_counts()
        aer_job_wall_ms = (time.perf_counter() - exec_t0) * 1_000
        wall_total_time_ms = (time.perf_counter() - total_t0) * 1_000

        experiment = result.results[0]
        experiment_metadata = dict(getattr(experiment, "metadata", {}) or {})
        result_dict = result.to_dict()
        result_metadata = dict(result_dict.get("metadata", {}) or {})
        experiment_dicts = result_dict.get("results") or []
        if experiment_dicts:
            dict_metadata = dict(experiment_dicts[0].get("metadata", {}) or {})
            if dict_metadata:
                experiment_metadata = dict_metadata
        experiment_time_s = getattr(experiment, "time_taken", None)
        if experiment_time_s is None:
            experiment_time_s = experiment_metadata.get("time_taken")

        if experiment_time_s is None:
            aer_experiment_time_ms = aer_job_wall_ms
            timing_basis = "aer_job_wall_fallback"
        else:
            aer_experiment_time_ms = float(experiment_time_s) * 1_000
            timing_basis = "aer_experiment_time_taken"

        result_time_taken = getattr(result, "time_taken", None)
        aer_result_time_taken_ms = (
            None if result_time_taken is None else float(result_time_taken) * 1_000
        )
        sample_measure_time = experiment_metadata.get("sample_measure_time")
        sample_measure_time_ms = self._seconds_to_ms(sample_measure_time)
        aer_time_taken_execute_ms = self._seconds_to_ms(
            result_metadata.get("time_taken_execute")
            or experiment_metadata.get("time_taken_execute")
        )
        fusion_metadata = experiment_metadata.get("fusion", {})
        if not isinstance(fusion_metadata, dict):
            fusion_metadata = {}
        aer_fusion_time_taken_ms = self._seconds_to_ms(fusion_metadata.get("time_taken"))
        aer_wall_overhead_ms = wall_total_time_ms - aer_experiment_time_ms
        aer_job_overhead_ms = aer_job_wall_ms - aer_experiment_time_ms
        aer_result_overhead_ms = (
            None
            if aer_result_time_taken_ms is None
            else aer_result_time_taken_ms - aer_experiment_time_ms
        )

        simulator_name = (
            "qiskit_aer_mps"
            if method == "matrix_product_state"
            else "qiskit_aer_statevector"
        )
        profile = {
            "profile_schema": "shots_v1",
            "simulator": simulator_name,
            "backend_method": method,
            "circuit_variant": circuit_variant,
            "timing_basis": timing_basis,
            "num_shots": shots,
            "num_qubits": qc.num_qubits,
            "num_instructions": len(qc.data),
            "preprocessing_ms": preprocessing_ms,
            "backend_setup_ms": backend_setup_ms,
            "gate_fusion_ms": 0.0,
            "parallel_execution_ms": aer_experiment_time_ms,
            "aer_experiment_time_taken_ms": aer_experiment_time_ms,
            "aer_result_time_taken_ms": aer_result_time_taken_ms,
            "aer_job_wall_ms": aer_job_wall_ms,
            "aer_time_taken_execute_ms": aer_time_taken_execute_ms,
            "aer_fusion_time_taken_ms": aer_fusion_time_taken_ms,
            "aer_fusion_applied": fusion_metadata.get("applied"),
            "sample_measure_time_ms": sample_measure_time_ms,
            "wall_total_time_ms": wall_total_time_ms,
            "aer_wall_overhead_ms": aer_wall_overhead_ms,
            "aer_job_overhead_ms": aer_job_overhead_ms,
            "aer_result_overhead_ms": aer_result_overhead_ms,
            "aer_metadata_summary": {
                "experiment_metadata_keys": sorted(experiment_metadata.keys()),
                "result_metadata_keys": sorted(result_metadata.keys()),
                "method": experiment_metadata.get("method"),
                "device": experiment_metadata.get("device"),
                "parallel_shots": experiment_metadata.get("parallel_shots"),
                "parallel_state_update": experiment_metadata.get("parallel_state_update"),
                "fusion": {
                    key: fusion_metadata.get(key)
                    for key in (
                        "enabled",
                        "applied",
                        "method",
                        "threshold",
                        "max_fused_qubits",
                        "parallelization",
                        "time_taken",
                    )
                    if key in fusion_metadata
                },
            },
            "per_shot_stats": None,
            "total_time_ms": aer_experiment_time_ms,
            "num_count_states": len(counts),
        }
        profile["source_function_breakdown"] = self._build_aer_source_function_breakdown(profile)
        return self._profile_elapsed_ms(profile, profile["total_time_ms"]), profile

    @staticmethod
    def _seconds_to_ms(value) -> float | None:
        if value is None:
            return None
        return float(value) * 1_000

    @staticmethod
    def _profile_metric(profile: dict, key: str | None) -> float | None:
        if key is None:
            return None
        value = profile.get(key)
        return None if value is None else float(value)

    @classmethod
    def _phase_compare(
        cls,
        phase: str,
        dqsim_profile: dict,
        dqsim_metric: str | None,
        aer_profile: dict,
        aer_metric: str | None,
        aer_function: str,
        aer_source: str,
        notes: str,
    ) -> dict:
        dqsim_ms = cls._profile_metric(dqsim_profile, dqsim_metric)
        aer_ms = cls._profile_metric(aer_profile, aer_metric)
        delta_ms = None
        ratio = None
        if dqsim_ms is not None and aer_ms is not None:
            delta_ms = dqsim_ms - aer_ms
            ratio = None if aer_ms == 0 else dqsim_ms / aer_ms
        return {
            "phase": phase,
            "dqsim_metric": dqsim_metric,
            "dqsim_ms": dqsim_ms,
            "aer_metric": aer_metric,
            "aer_ms": aer_ms,
            "delta_vs_aer_ms": delta_ms,
            "dqsim_to_aer_ratio": ratio,
            "aer_function": aer_function,
            "aer_source": aer_source,
            "notes": notes,
        }

    @staticmethod
    def _build_dqsim_source_function_breakdown(profile: dict) -> list[dict]:
        simulator = profile.get("simulator", "")
        source = DQSIM_SOURCE_LABELS.get(simulator, "src/*/simulator.rs::simulate_shots")
        return [
            {
                "phase": "preprocessing",
                "metric": "preprocessing_ms",
                "time_ms": profile.get("preprocessing_ms"),
                "dqsim_function": f"{source} setup",
                "notes": "Circuit deserialization/setup and conversion into simulator-ready data.",
            },
            {
                "phase": "gate_fusion",
                "metric": "gate_fusion_ms",
                "time_ms": profile.get("gate_fusion_ms"),
                "dqsim_function": "src/types.rs::fuse_circuit or fuse_pblock_entries",
                "notes": "Local gate fusion before shot execution.",
            },
            {
                "phase": "shot_execution",
                "metric": "parallel_execution_ms",
                "time_ms": profile.get("parallel_execution_ms"),
                "dqsim_function": source,
                "notes": "Shot loop execution, including gate application and measurement work.",
            },
            {
                "phase": "total_profiled_time",
                "metric": "total_time_ms",
                "time_ms": profile.get("total_time_ms"),
                "dqsim_function": source,
                "notes": "Profile total emitted by DQSim for simulate_shots.",
            },
        ]

    @staticmethod
    def _build_aer_source_function_breakdown(profile: dict) -> list[dict]:
        method = profile.get("backend_method")
        apply_source = (
            AER_SOURCE_URLS["mps_apply"]
            if method == "matrix_product_state"
            else AER_SOURCE_URLS["statevector_apply"]
        )
        return [
            {
                "phase": "python_job_wrapper",
                "metric": "aer_result_time_taken_ms",
                "time_ms": profile.get("aer_result_time_taken_ms"),
                "aer_function": "AerBackend._execute_circuits_job",
                "aer_source": AER_SOURCE_URLS["python_job"],
                "notes": "Compile, assemble, config generation, C++ execution, and result formatting timer.",
            },
            {
                "phase": "controller_execute",
                "metric": "aer_time_taken_execute_ms",
                "time_ms": profile.get("aer_time_taken_execute_ms"),
                "aer_function": "Controller::execute",
                "aer_source": AER_SOURCE_URLS["controller_execute"],
                "notes": "C++ controller-level execution metadata when emitted by Aer.",
            },
            {
                "phase": "fusion",
                "metric": "aer_fusion_time_taken_ms",
                "time_ms": profile.get("aer_fusion_time_taken_ms"),
                "aer_function": "Fusion::optimize_circuit",
                "aer_source": AER_SOURCE_URLS["fusion"],
                "notes": "Aer transpile fusion time when fusion metadata is present.",
            },
            {
                "phase": "experiment_execution",
                "metric": "aer_experiment_time_taken_ms",
                "time_ms": profile.get("aer_experiment_time_taken_ms"),
                "aer_function": "CircuitExecutor experiment execution / State::apply_ops",
                "aer_source": AER_SOURCE_URLS["experiment_time"],
                "notes": "Per-experiment simulator timer used as the reported Aer benchmark value.",
            },
            {
                "phase": "state_apply_ops",
                "metric": "aer_experiment_time_taken_ms",
                "time_ms": profile.get("aer_experiment_time_taken_ms"),
                "aer_function": "State::apply_op / apply_gate / apply_measure",
                "aer_source": apply_source,
                "notes": "Contained inside Aer experiment time; stock Aer does not split this into per-function timings.",
            },
            {
                "phase": "sample_measure",
                "metric": "sample_measure_time_ms",
                "time_ms": profile.get("sample_measure_time_ms"),
                "aer_function": "CircuitExecutor::sample_measure",
                "aer_source": AER_SOURCE_URLS["sample_measure"],
                "notes": "Only populated when Aer takes the sample-measure optimization path.",
            },
            {
                "phase": "wall_overhead",
                "metric": "aer_wall_overhead_ms",
                "time_ms": profile.get("aer_wall_overhead_ms"),
                "aer_function": "Python benchmark wrapper around backend.run(...).result().get_counts()",
                "aer_source": "benchmarking/benchmarking_suite.py::_time_aer_shots",
                "notes": "Wall time outside the reported Aer experiment timer.",
            },
        ]

    @staticmethod
    def _aer_counterpart_keys(profile_key: str) -> list[str]:
        mapping = {
            "original_sv": ["original_aer_statevector"],
            "original_mps": ["original_aer_mps"],
            "lowered_sv": ["lowered_aer_statevector"],
            "lowered_mps": ["lowered_aer_mps"],
            "lowered_pblock": ["lowered_aer_statevector", "lowered_aer_mps"],
        }
        return mapping.get(profile_key, [])

    @classmethod
    def _build_aer_function_comparison(cls, dqsim_profile: dict, aer_profile: dict) -> dict:
        return {
            "schema": "source_mapped_phase_comparison_v1",
            "note": FUNCTION_COMPARISON_NOTE,
            "aer_simulator": aer_profile.get("simulator"),
            "aer_backend_method": aer_profile.get("backend_method"),
            "aer_timing_basis": aer_profile.get("timing_basis"),
            "phases": [
                cls._phase_compare(
                    "preprocessing_or_conversion",
                    dqsim_profile,
                    "preprocessing_ms",
                    aer_profile,
                    "preprocessing_ms",
                    "BenchmarkRunner._prepare_qiskit_circuit",
                    "benchmarking/benchmarking_suite.py::_prepare_qiskit_circuit",
                    "Both are local benchmark-side setup costs, not Aer C++ kernel time.",
                ),
                cls._phase_compare(
                    "fusion_or_compilation",
                    dqsim_profile,
                    "gate_fusion_ms",
                    aer_profile,
                    "aer_fusion_time_taken_ms",
                    "Fusion::optimize_circuit",
                    AER_SOURCE_URLS["fusion"],
                    "DQSim gate fusion is explicit; Aer fusion is available only if Aer emits fusion metadata.",
                ),
                cls._phase_compare(
                    "shot_or_experiment_execution",
                    dqsim_profile,
                    "parallel_execution_ms",
                    aer_profile,
                    "aer_experiment_time_taken_ms",
                    "CircuitExecutor experiment execution / State::apply_ops",
                    AER_SOURCE_URLS["experiment_time"],
                    "Closest fair timing comparison: DQSim shot loop versus Aer experiment timer.",
                ),
                cls._phase_compare(
                    "sample_measure",
                    dqsim_profile,
                    None,
                    aer_profile,
                    "sample_measure_time_ms",
                    "CircuitExecutor::sample_measure",
                    AER_SOURCE_URLS["sample_measure"],
                    "Aer-specific sampling fast path; DQSim measurement work is included in its shot loop.",
                ),
                cls._phase_compare(
                    "python_job_overhead",
                    dqsim_profile,
                    None,
                    aer_profile,
                    "aer_wall_overhead_ms",
                    "AerBackend._execute_circuits_job and result formatting wrapper",
                    AER_SOURCE_URLS["python_job"],
                    "Shown for diagnosing overhead; it is not part of the reported Aer benchmark total.",
                ),
            ],
        }

    @classmethod
    def _profiles_with_aer_function_comparisons(cls, profiles: dict[str, dict]) -> dict[str, dict]:
        enriched = {key: dict(profile) for key, profile in profiles.items()}
        for key, profile in enriched.items():
            if str(profile.get("simulator", "")).startswith("qiskit_aer"):
                profile.setdefault("aer_function_comparison", {
                    "schema": "aer_source_breakdown_v1",
                    "note": FUNCTION_COMPARISON_NOTE,
                    "phases": profile.get("source_function_breakdown", []),
                })
                continue

            counterparts = []
            for aer_key in cls._aer_counterpart_keys(key):
                aer_profile = enriched.get(aer_key)
                if aer_profile is None:
                    continue
                comparison = cls._build_aer_function_comparison(profile, aer_profile)
                comparison["aer_profile_key"] = aer_key
                counterparts.append(comparison)

            if counterparts:
                profile["aer_function_comparison"] = {
                    "schema": "dqsim_to_aer_source_mapped_phase_comparison_v1",
                    "note": FUNCTION_COMPARISON_NOTE,
                    "counterparts": counterparts,
                }
            else:
                profile.setdefault("aer_function_comparison", {
                    "schema": "no_direct_aer_counterpart_v1",
                    "note": "No direct Aer comparison is attached for this profile, usually because it is symbolic-only or unsupported by Aer.",
                    "counterparts": [],
                })
        return enriched

    @staticmethod
    def _prepare_qiskit_circuit(circuit):
        qc = CircuitConverters.to_qiskit(circuit)
        BenchmarkRunner._substitute_remote_gates(qc)
        if qc.num_clbits == 0:
            qc.measure_all()
        return qc

    @staticmethod
    def _substitute_remote_gates(qc) -> None:
        phi_plus = UnitaryGate(RemoteLinkGatePhiPlus().to_matrix(), label="remote_link_phi_plus")
        psi_minus = UnitaryGate(RemoteLinkGatePsiMinus().to_matrix(), label="remote_link_psi_minus")
        psi_plus = UnitaryGate(RemoteLinkGatePsiPlus().to_matrix(), label="remote_link_psi_plus")
        for i, ci in enumerate(qc.data):
            op = ci.operation
            name = getattr(op, "name", "")
            if isinstance(op, RemoteLinkGatePhiPlus) or name in {
                "remote_link_phi_plus",
                "remote_epr",
            }:
                qc.data[i] = CircuitInstruction(phi_plus, ci.qubits, ci.clbits)
            elif isinstance(op, RemoteLinkGatePsiMinus) or name == "remote_link_psi_minus":
                qc.data[i] = CircuitInstruction(psi_minus, ci.qubits, ci.clbits)
            elif isinstance(op, RemoteLinkGatePsiPlus) or name == "remote_link_psi_plus":
                qc.data[i] = CircuitInstruction(psi_plus, ci.qubits, ci.clbits)

    @staticmethod
    def _profile_snapshot(profile: dict) -> dict:
        snapshot = dict(profile)
        snapshot.pop("comparison", None)
        return snapshot

    @staticmethod
    def _metric_delta(current: float | None, previous: float | None) -> dict | None:
        if current is None or previous is None:
            return None

        delta = current - previous
        delta_percent = None if previous == 0 else 100.0 * delta / previous
        return {
            "current": current,
            "previous": previous,
            "delta": delta,
            "delta_percent": delta_percent,
        }

    @staticmethod
    def _per_shot_ms(profile: dict) -> float | None:
        shots = profile.get("num_shots")
        total_ms = profile.get("total_time_ms")
        if not shots or total_ms is None:
            return None
        return total_ms / shots

    @classmethod
    def _build_comparison(cls, current: dict, previous: dict | None) -> dict:
        if previous is None:
            return {
                "previous_profile": None,
                "metrics": {},
                "note": "No previous profile was available; this run is the baseline for the next comparison.",
            }

        metrics = {}
        for key in (
            "preprocessing_ms",
            "backend_setup_ms",
            "gate_fusion_ms",
            "parallel_execution_ms",
            "aer_experiment_time_taken_ms",
            "aer_result_time_taken_ms",
            "aer_job_wall_ms",
            "aer_time_taken_execute_ms",
            "aer_fusion_time_taken_ms",
            "sample_measure_time_ms",
            "wall_total_time_ms",
            "aer_wall_overhead_ms",
            "aer_job_overhead_ms",
            "aer_result_overhead_ms",
            "total_time_ms",
        ):
            metric = cls._metric_delta(current.get(key), previous.get(key))
            if metric is not None:
                metrics[key] = metric

        per_shot_metric = cls._metric_delta(
            cls._per_shot_ms(current), cls._per_shot_ms(previous)
        )
        if per_shot_metric is not None:
            metrics["per_shot_ms"] = per_shot_metric

        return {
            "previous_profile": cls._profile_snapshot(previous),
            "metrics": metrics,
        }

    def _load_previous_profiles(self) -> dict[str, dict]:
        profiles = {}
        for filename in self._profile_dir.glob("*.json"):
            try:
                with open(filename, encoding="utf-8") as f:
                    profiles[filename.name] = json.load(f)
            except json.JSONDecodeError:
                print(f"    Ignoring unreadable previous profile: {filename}", flush=True)
        return profiles

    def _clear_profile_dir(self) -> None:
        for pattern in ("*.json", "*.json.tmp"):
            for filename in self._profile_dir.glob(pattern):
                filename.unlink()

    def _save_profiles(self, result: BenchmarkResult) -> None:
        run_id = time.strftime("%Y%m%d_%H%M%S")
        profiles = self._profiles_with_aer_function_comparisons(result.profiles)
        for suffix, profile in profiles.items():
            filename = self._profile_dir / f"{result.name}_{suffix}.json"
            previous = self._previous_profiles.get(filename.name)

            profile = dict(profile)
            profile.setdefault("circuit", result.name)
            profile["profile_file"] = filename.name
            profile["run_id"] = run_id
            profile["comparison"] = self._build_comparison(profile, previous)

            tmp_filename = filename.with_suffix(".json.tmp")
            with open(tmp_filename, "w", encoding="utf-8") as f:
                json.dump(profile, f, indent=2)
                f.write("\n")
            tmp_filename.replace(filename)
            print(f"    Saved {suffix} profile: {filename}", flush=True)


class BenchmarkReporter:
    _ORIGINAL_HEADER = (
        f"{'Circuit':<16}  {'Shots':>5}  {'Qb':>4}  "
        f"{'dqsim-sv(us)':>14}  {'dqsim-mps(us)':>14}  "
        f"{'aer-sv(us)':>12}  {'aer-mps(us)':>12}"
    )
    _LOWERED_HEADER = (
        f"{'Circuit':<16}  {'Shots':>5}  {'Qb':>4}  "
        f"{'dqsim-sv(us)':>14}  {'dqsim-mps(us)':>14}  "
        f"{'pblock(us)':>12}  {'aer-sv(us)':>12}  {'aer-mps(us)':>12}"
    )
    _SYMBOLIC_HEADER = (
        f"{'Circuit':<16}  {'Shots':>5}  {'Qb':>4}  {'pblock-symbolic(us)':>20}"
    )

    def __init__(self, config: BenchmarkConfig) -> None:
        self._default_shots = config.shots

    def print(self, results: list[BenchmarkResult]) -> None:
        print(
            f"\n\nOriginal circuit performance "
            f"(default SHOTS={self._default_shots}, per-row timing)\n"
        )
        self._print_table(self._ORIGINAL_HEADER, self._format_original_row, results)

        print(
            f"\n\nLowered distributed-circuit performance "
            f"(default SHOTS={self._default_shots}, per-row timing)\n"
        )
        self._print_table(self._LOWERED_HEADER, self._format_lowered_row, results)

        if any(r.symbolic_num_qubits is not None for r in results):
            print(
                "\n\nSymbolic PBlock smoke timing "
                "(not cross-simulator; unsupported symbolic gates are skipped)\n"
            )
            self._print_table(self._SYMBOLIC_HEADER, self._format_symbolic_row, results)

    @staticmethod
    def _print_table(header: str, row_fn: Callable[[BenchmarkResult], str], results) -> None:
        sep = "-" * len(header)
        print(header)
        print(sep)
        for result in results:
            print(row_fn(result))
        print(sep)

    def _format_original_row(self, r: BenchmarkResult) -> str:
        t = r.original
        return (
            f"{r.name:<16}  {r.shots:>5}  {t.num_qubits:>4}  "
            f"{self._fmt_us(t.sv_shots_ms, r.shots, 14)}  "
            f"{self._fmt_us(t.mps_shots_ms, r.shots, 14)}  "
            f"{self._fmt_us(t.aer_statevector_shots_ms, r.shots, 12)}  "
            f"{self._fmt_us(t.aer_mps_shots_ms, r.shots, 12)}"
        )

    def _format_lowered_row(self, r: BenchmarkResult) -> str:
        t = r.lowered
        return (
            f"{r.name:<16}  {r.shots:>5}  {t.num_qubits:>4}  "
            f"{self._fmt_us(t.sv_shots_ms, r.shots, 14)}  "
            f"{self._fmt_us(t.mps_shots_ms, r.shots, 14)}  "
            f"{self._fmt_us(t.pblock_shots_ms, r.shots, 12)}  "
            f"{self._fmt_us(t.aer_statevector_shots_ms, r.shots, 12)}  "
            f"{self._fmt_us(t.aer_mps_shots_ms, r.shots, 12)}"
        )

    def _format_symbolic_row(self, r: BenchmarkResult) -> str:
        q = "N/A" if r.symbolic_num_qubits is None else str(r.symbolic_num_qubits)
        return (
            f"{r.name:<16}  {r.shots:>5}  {q:>4}  "
            f"{self._fmt_us(r.symbolic_pblock_shots_ms, r.shots, 20)}"
        )

    def _fmt_us(self, value_ms: float | None, shots: int, width: int) -> str:
        if value_ms is None:
            return f"{'N/A':>{width}}"
        return f"{value_ms / shots * 1_000:>{width}.2f}"


_SUITE_FILE = pathlib.Path(__file__).parent / "benchmarking_suite.yaml"


class TestPerformance:
    def test_benchmark_table(self) -> None:
        config = BenchmarkConfig(_SUITE_FILE)
        results = BenchmarkRunner(config).run_all()
        BenchmarkReporter(config).print(results)