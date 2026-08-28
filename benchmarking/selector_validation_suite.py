"""Validate simulator-selector choices against measured backend timings.

Run the default configured QASMBench subset:
    pytest benchmarking/selector_validation_suite.py -v -s

Useful knobs:
    DQSIM_SELECTOR_VALIDATION_SCOPE=suite|small|all
    DQSIM_SELECTOR_VALIDATION_CIRCUITS=bell_n4,qft_n4
    DQSIM_SELECTOR_VALIDATION_MAX_CIRCUITS=10
    DQSIM_SELECTOR_VALIDATION_REPEATS=3
    DQSIM_SELECTOR_VALIDATION_OUTPUT=benchmarking/selector_validation.json
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import statistics
import time
from collections.abc import Callable
from typing import Any

import pytest
import qasmpi
from bosonic_model.qasm import Translator
from bosonic_sdk.distributor.distributors.disqco_distributor import DisqcoDistributor

from dqsim import select_simulator

from benchmarking.benchmarking_suite import BenchmarkConfig, BenchmarkRunner, CircuitSpec, _SUITE_FILE


QASMBENCH_SOURCE_URL = "https://github.com/pnnl/QASMBench"
DEFAULT_OUTPUT_FILE = pathlib.Path(__file__).parent / "selector_validation.json"
ORIGINAL_SIMULATORS = ("dqsim-sv", "dqsim-mps", "aer-sv", "aer-mps")
LOWERED_SIMULATORS = ("dqsim-sv", "dqsim-mps", "pblock", "aer-sv", "aer-mps")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _output_file() -> pathlib.Path:
    raw = os.getenv("DQSIM_SELECTOR_VALIDATION_OUTPUT")
    if raw:
        return pathlib.Path(raw)
    return DEFAULT_OUTPUT_FILE


def _configured_specs(config: BenchmarkConfig) -> list[CircuitSpec]:
    override = os.getenv("DQSIM_SELECTOR_VALIDATION_CIRCUITS")
    if override:
        names = [name.strip() for name in override.split(",") if name.strip()]
        return [_derived_spec(name, config.shots) for name in names]

    scope = os.getenv("DQSIM_SELECTOR_VALIDATION_SCOPE", "suite").strip().lower()
    if scope == "suite":
        specs = list(config.circuits)
    elif scope in {"small", "all-small", "qasmbench-small"}:
        specs = [_derived_spec(name, config.shots) for name in qasmpi.list_circuits(size="small")]
    elif scope in {"all", "qasmbench-all"}:
        specs = [_derived_spec(name, config.shots) for name in qasmpi.list_circuits()]
    else:
        raise ValueError(
            "DQSIM_SELECTOR_VALIDATION_SCOPE must be suite, small, or all"
        )

    max_circuits = os.getenv("DQSIM_SELECTOR_VALIDATION_MAX_CIRCUITS")
    if max_circuits:
        specs = specs[: int(max_circuits)]
    return specs


def _derived_spec(name: str, default_shots: int) -> CircuitSpec:
    circuit = Translator().from_qasm(qasmpi.get_circuit(name))
    num_qubits = BenchmarkRunner._num_qubits(circuit)
    nodes = _env_int("DQSIM_SELECTOR_VALIDATION_NODES", 2)
    qpn = max(1, math.ceil(num_qubits / nodes))
    shots = os.getenv("DQSIM_SELECTOR_VALIDATION_SHOTS")
    if shots:
        shot_count = int(shots)
    elif num_qubits <= 8:
        shot_count = default_shots
    elif num_qubits <= 14:
        shot_count = min(default_shots, 10)
    else:
        shot_count = min(default_shots, 1)
    return CircuitSpec(name=name, nodes=nodes, qubits_per_node=qpn, shots=shot_count)


def _timing_stats(samples: list[float]) -> dict[str, Any]:
    mean_ms = statistics.fmean(samples)
    stdev_ms = statistics.stdev(samples) if len(samples) > 1 else 0.0
    ci95_ms = None
    if len(samples) > 1:
        ci95_ms = 1.96 * stdev_ms / math.sqrt(len(samples))
    return {
        "samples_ms": samples,
        "mean_ms": mean_ms,
        "min_ms": min(samples),
        "max_ms": max(samples),
        "stdev_ms": stdev_ms,
        "ci95_ms": ci95_ms,
    }


def _measure_backend(
    simulator: str,
    repeats: int,
    run_once: Callable[[], tuple[float, dict | None]],
) -> dict[str, Any]:
    samples: list[float] = []
    errors: list[str] = []
    last_profile: dict | None = None

    for _ in range(repeats):
        try:
            elapsed_ms, profile = run_once()
        except Exception as exc:  # noqa: BLE001 - validation records backend failures
            errors.append(f"{type(exc).__name__}: {exc}")
            break
        samples.append(float(elapsed_ms))
        if profile is not None:
            last_profile = profile

    if not samples:
        return {
            "simulator": simulator,
            "status": "failed",
            "errors": errors,
        }

    payload = {
        "simulator": simulator,
        "status": "ok",
        "runs": len(samples),
        "errors": errors,
        **_timing_stats(samples),
    }
    if last_profile is not None:
        payload["last_profile_summary"] = {
            "simulator": last_profile.get("simulator"),
            "timing_basis": last_profile.get("timing_basis", "profile_total"),
            "num_qubits": last_profile.get("num_qubits"),
            "num_instructions": last_profile.get("num_instructions"),
            "parallel_execution_ms": last_profile.get("parallel_execution_ms"),
            "total_time_ms": last_profile.get("total_time_ms"),
        }
    return payload


def _backend_runner(
    runner: BenchmarkRunner,
    simulator: str,
    variant: str,
    circuit,
    lowered_monolithic,
    distributed_lowered,
    shots: int,
) -> Callable[[], tuple[float, dict | None]]:
    if variant == "original":
        if simulator == "dqsim-sv":
            return lambda: runner._time_sv_shots(circuit, "selector_original", shots)
        if simulator == "dqsim-mps":
            return lambda: runner._time_mps_shots(circuit, "selector_original", shots)
        if simulator == "aer-sv":
            return lambda: runner._time_aer_shots(circuit, "statevector", "selector_original", shots)
        if simulator == "aer-mps":
            return lambda: runner._time_aer_shots(
                circuit, "matrix_product_state", "selector_original", shots
            )

    if variant == "lowered":
        if simulator == "dqsim-sv":
            return lambda: runner._time_sv_shots(lowered_monolithic, "selector_lowered", shots)
        if simulator == "dqsim-mps":
            return lambda: runner._time_mps_shots(lowered_monolithic, "selector_lowered", shots)
        if simulator == "pblock":
            return lambda: runner._time_pblock_shots(
                distributed_lowered, "dqsim_pblock_lowered", "selector_lowered", shots
            )
        if simulator == "aer-sv":
            return lambda: runner._time_aer_shots(
                lowered_monolithic, "statevector", "selector_lowered", shots
            )
        if simulator == "aer-mps":
            return lambda: runner._time_aer_shots(
                lowered_monolithic, "matrix_product_state", "selector_lowered", shots
            )

    raise ValueError(f"{simulator} is not supported for {variant}")


def _best_backend(timings: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    successes = [
        (simulator, timing)
        for simulator, timing in timings.items()
        if timing.get("status") == "ok" and timing.get("mean_ms") is not None
    ]
    if not successes:
        return None
    successes.sort(key=lambda item: (float(item[1]["mean_ms"]), item[0]))
    winner, winning_timing = successes[0]
    runner_up = successes[1] if len(successes) > 1 else None
    payload: dict[str, Any] = {
        "simulator": winner,
        "mean_ms": winning_timing["mean_ms"],
        "ci95_ms": winning_timing.get("ci95_ms"),
    }
    if runner_up is not None:
        runner_name, runner_timing = runner_up
        margin_ms = float(runner_timing["mean_ms"]) - float(winning_timing["mean_ms"])
        payload.update(
            {
                "runner_up": runner_name,
                "runner_up_mean_ms": runner_timing["mean_ms"],
                "margin_ms": margin_ms,
                "speedup_vs_runner_up": float(runner_timing["mean_ms"])
                / float(winning_timing["mean_ms"]),
            }
        )
    return payload


def _validate_variant(
    runner: BenchmarkRunner,
    spec: CircuitSpec,
    variant: str,
    circuit,
    lowered_monolithic,
    distributed_lowered,
    repeats: int,
    use_profile_history: bool,
) -> dict[str, Any]:
    shots = spec.shots if spec.shots is not None else runner._config.shots
    if variant == "original":
        selection = select_simulator(
            circuit,
            shots=shots,
            allow_aer=True,
            use_profile_history=use_profile_history,
        )
        simulators = ORIGINAL_SIMULATORS
    elif variant == "lowered":
        selection = select_simulator(
            distributed=distributed_lowered,
            shots=shots,
            allow_aer=True,
            use_profile_history=use_profile_history,
        )
        simulators = LOWERED_SIMULATORS
    else:  # pragma: no cover - guarded by caller
        raise ValueError(f"unsupported variant: {variant}")

    print(
        f"    {spec.name}/{variant}: selector={selection.selected} "
        f"confidence={selection.confidence:.3f}",
        flush=True,
    )

    candidate_by_sim = {candidate.simulator: candidate.to_dict() for candidate in selection.candidates}
    timings = {}
    for simulator in simulators:
        print(f"      timing {simulator} ({repeats} run(s)) ...", flush=True)
        if variant == "original" and simulator == "pblock":
            continue
        timings[simulator] = _measure_backend(
            simulator,
            repeats,
            _backend_runner(
                runner,
                simulator,
                variant,
                circuit,
                lowered_monolithic,
                distributed_lowered,
                shots,
            ),
        )
        timings[simulator]["selector_candidate"] = candidate_by_sim.get(simulator)

    best = _best_backend(timings)
    selected_timing = timings.get(selection.selected)
    selector_correct = bool(best and selection.selected == best["simulator"])
    return {
        "variant": variant,
        "shots": shots,
        "selection": {
            "selected": selection.selected,
            "confidence": selection.confidence,
            "reason": selection.reason,
            "result": selection.to_dict(),
        },
        "timings_ms": timings,
        "actual_best": best,
        "selected_timing_ms": selected_timing,
        "selector_correct": selector_correct,
    }


def _validate_circuit(
    runner: BenchmarkRunner,
    distributor: DisqcoDistributor,
    spec: CircuitSpec,
    repeats: int,
    use_profile_history: bool,
) -> dict[str, Any]:
    circuit = Translator().from_qasm(qasmpi.get_circuit(spec.name))
    shots = spec.shots if spec.shots is not None else runner._config.shots
    payload: dict[str, Any] = {
        "name": spec.name,
        "shots": shots,
        "nodes": spec.nodes,
        "qubits_per_node": spec.qubits_per_node,
        "variants": {},
    }

    try:
        distributed_lowered = distributor.distribute(
            circuit,
            nodes=spec.nodes,
            qubits_per_node=spec.qubits_per_node,
            lowered=True,
        )
        lowered_monolithic = distributed_lowered.as_monolithic_circuit()
    except Exception as exc:  # noqa: BLE001 - validation records distribution failures
        payload["distribution_error"] = f"{type(exc).__name__}: {exc}"
        lowered_monolithic = None
        distributed_lowered = None

    payload["original_num_qubits"] = BenchmarkRunner._num_qubits(circuit)
    payload["variants"]["original"] = _validate_variant(
        runner,
        spec,
        "original",
        circuit,
        lowered_monolithic,
        distributed_lowered,
        repeats,
        use_profile_history,
    )

    if lowered_monolithic is not None and distributed_lowered is not None:
        payload["lowered_num_qubits"] = BenchmarkRunner._num_qubits(lowered_monolithic)
        payload["variants"]["lowered"] = _validate_variant(
            runner,
            spec,
            "lowered",
            circuit,
            lowered_monolithic,
            distributed_lowered,
            repeats,
            use_profile_history,
        )
    return payload


def _summary(circuits: list[dict[str, Any]]) -> dict[str, Any]:
    variants = []
    for circuit in circuits:
        for variant in circuit.get("variants", {}).values():
            variants.append(variant)
    evaluated = [v for v in variants if v.get("actual_best") is not None]
    correct = [v for v in evaluated if v.get("selector_correct")]
    by_variant: dict[str, dict[str, Any]] = {}
    for name in sorted({v["variant"] for v in evaluated}):
        subset = [v for v in evaluated if v["variant"] == name]
        subset_correct = [v for v in subset if v.get("selector_correct")]
        by_variant[name] = {
            "evaluated": len(subset),
            "selector_correct": len(subset_correct),
            "accuracy": None if not subset else len(subset_correct) / len(subset),
        }
    return {
        "circuits": len(circuits),
        "variants_evaluated": len(evaluated),
        "selector_correct": len(correct),
        "accuracy": None if not evaluated else len(correct) / len(evaluated),
        "by_variant": by_variant,
    }


class TestSelectorValidation:
    def test_selector_matches_measured_best_simulator(self) -> None:
        config = BenchmarkConfig(_SUITE_FILE)
        specs = _configured_specs(config)
        if not specs:
            pytest.skip("no selector-validation circuits configured")

        repeats = _env_int("DQSIM_SELECTOR_VALIDATION_REPEATS", 3)
        use_profile_history = _env_bool("DQSIM_SELECTOR_VALIDATION_USE_PROFILE_HISTORY", True)
        output_file = _output_file()
        output_file.parent.mkdir(parents=True, exist_ok=True)

        runner = BenchmarkRunner(config)
        distributor = DisqcoDistributor()
        circuits = []
        started = time.perf_counter()
        for index, spec in enumerate(specs, 1):
            print(f"  [{index}/{len(specs)}] selector validation {spec.name} ...", flush=True)
            circuits.append(
                _validate_circuit(
                    runner,
                    distributor,
                    spec,
                    repeats,
                    use_profile_history,
                )
            )

        report = {
            "schema": "dqsim_selector_validation_v1",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "qasmbench_source_url": QASMBENCH_SOURCE_URL,
            "qasmpi_circuit_count": len(qasmpi.list_circuits()),
            "config": {
                "scope": os.getenv("DQSIM_SELECTOR_VALIDATION_SCOPE", "suite"),
                "circuits": [spec.name for spec in specs],
                "repeats": repeats,
                "use_profile_history": use_profile_history,
                "timing_confidence_interval": "95% CI over repeated profiled timings; null when repeats=1",
                "selector_confidence": "heuristic selector score in [0, 1], not a statistical CI",
            },
            "summary": _summary(circuits),
            "elapsed_wall_ms": (time.perf_counter() - started) * 1_000,
            "circuits": circuits,
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
            f.write("\n")
        print(f"\nSelector validation JSON: {output_file}", flush=True)
        print(json.dumps(report["summary"], indent=2), flush=True)

        assert output_file.exists()
        assert report["summary"]["variants_evaluated"] > 0
