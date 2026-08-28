from __future__ import annotations

import importlib.util
import json
import math
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from dqsim._core import MpsSimulator, PBlockSimulator, StatevectorSimulator


_SIM_ORDER = ("dqsim-sv", "dqsim-mps", "pblock", "aer-sv", "aer-mps")
_MPS_SUPPORTED_REMOTE_GATES = {
    "remote_link_phi_plus",
    "remote_epr",
    "epr",
    "remote_link_psi_minus",
    "remote_link_psi_plus",
    "nonlocal_cz",
    "remote_cz",
    "remote_cx",
    "remote_barrier",
}
_IGNORED_NAMES = {"id", "u0", "barrier", "classical"}
_PROFILE_SUFFIXES = (
    ("_original_aer_statevector.json", "original", "aer-sv"),
    ("_original_aer_mps.json", "original", "aer-mps"),
    ("_original_sv.json", "original", "dqsim-sv"),
    ("_original_mps.json", "original", "dqsim-mps"),
    ("_lowered_aer_statevector.json", "lowered", "aer-sv"),
    ("_lowered_aer_mps.json", "lowered", "aer-mps"),
    ("_lowered_pblock.json", "lowered", "pblock"),
    ("_lowered_sv.json", "lowered", "dqsim-sv"),
    ("_lowered_mps.json", "lowered", "dqsim-mps"),
)
_PROFILE_SIMULATOR_ALIASES = {
    "dqsim_statevector": "dqsim-sv",
    "dqsim_sv": "dqsim-sv",
    "dqsim-mps": "dqsim-mps",
    "dqsim_mps": "dqsim-mps",
    "dqsim_pblock": "pblock",
    "dqsim_pblock_lowered": "pblock",
    "dqsim_pblock_symbolic": "pblock",
    "pblock": "pblock",
    "qiskit_aer_statevector": "aer-sv",
    "aer_statevector": "aer-sv",
    "aer_sv": "aer-sv",
    "qiskit_aer_mps": "aer-mps",
    "qiskit_aer_matrix_product_state": "aer-mps",
    "aer_mps": "aer-mps",
}


@dataclass(frozen=True)
class CircuitFeatures:
    num_qubits: int
    num_instructions: int
    num_one_qubit_gates: int
    num_two_qubit_gates: int
    num_multi_qubit_gates: int
    num_measurements: int
    num_resets: int
    num_conditionals: int
    num_barriers: int
    num_remote_gates: int
    num_nodes: int
    is_distributed: bool
    two_qubit_density: float
    measurement_density: float
    average_two_qubit_distance: float
    max_two_qubit_distance: int
    estimated_mps_routing_swaps: int
    estimated_mps_adjacent_2q_applications: int
    estimated_mps_entanglement_risk: float
    unsupported_mps_gates: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SimulatorCandidate:
    simulator: str
    eligible: bool
    score: float | None
    reasons: tuple[str, ...] = ()
    rejection_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _ProfilePoint:
    simulator: str
    variant: str
    num_qubits: int
    num_instructions: int
    num_shots: int
    parallel_execution_ms: float
    path: Path

    @property
    def per_shot_us(self) -> float:
        return 1000.0 * self.parallel_execution_ms / max(1, self.num_shots)


@dataclass(frozen=True)
class SelectionResult:
    selected: str
    confidence: float
    reason: str
    features: CircuitFeatures
    candidates: tuple[SimulatorCandidate, ...] = field(default_factory=tuple)
    selection_mode: str = "heuristic"
    profile_history_used: bool = False
    profile_history_matches: int = 0
    profile_history_estimates: dict[str, float] = field(default_factory=dict)
    pilot_used: bool = False
    pilot_shots: int | None = None
    pilot_timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def eligible(self) -> tuple[SimulatorCandidate, ...]:
        return tuple(candidate for candidate in self.candidates if candidate.eligible)

    @property
    def rejected(self) -> tuple[SimulatorCandidate, ...]:
        return tuple(candidate for candidate in self.candidates if not candidate.eligible)

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected": self.selected,
            "confidence": self.confidence,
            "reason": self.reason,
            "features": self.features.to_dict(),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "selection_mode": self.selection_mode,
            "profile_history_used": self.profile_history_used,
            "profile_history_matches": self.profile_history_matches,
            "profile_history_estimates": dict(self.profile_history_estimates),
            "pilot_used": self.pilot_used,
            "pilot_shots": self.pilot_shots,
            "pilot_timings_ms": dict(self.pilot_timings_ms),
        }


def select_simulator(
    circuit: Any | None = None,
    *,
    distributed: Any | None = None,
    shots: int = 1000,
    allow_aer: bool = True,
    mps_routing: str = "restore",
    use_profile_history: bool = False,
    profile_history_dir: str | Path | None = None,
) -> SelectionResult:
    """Choose the most likely fastest simulator for a circuit.

    The selector is intentionally heuristic. It filters out unsupported
    simulators, scores the remaining backends from cheap circuit features, and
    returns an explainable result. Set use_profile_history=True to nudge
    scores with nearby benchmark JSONs from benchmarking/profiles.
    """

    circuit = _resolve_circuit(circuit, distributed)
    if circuit is None:
        raise ValueError("select_simulator requires circuit= or distributed=")
    if shots <= 0:
        raise ValueError("shots must be positive")

    features = extract_circuit_features(circuit, distributed=distributed)
    candidates = _score_candidates(features, shots, allow_aer, mps_routing)
    history_used = False
    history_matches = 0
    history_estimates: dict[str, float] = {}
    if use_profile_history:
        candidates, history_used, history_matches, history_estimates = _apply_profile_history(
            candidates, features, shots, profile_history_dir
        )

    eligible = [candidate for candidate in candidates if candidate.eligible]
    if not eligible:
        raise RuntimeError("No eligible simulator found for this circuit")

    eligible.sort(key=lambda candidate: (candidate.score or math.inf, _sim_sort(candidate.simulator)))
    selected = eligible[0]
    confidence = _confidence(eligible)
    reason = _selection_reason(selected, eligible, features)
    if history_used:
        reason += (
            f" Profile history adjusted {len(history_estimates)} simulator estimates "
            f"from {history_matches} nearby profile entries."
        )
    return SelectionResult(
        selected=selected.simulator,
        confidence=confidence,
        reason=reason,
        features=features,
        candidates=tuple(candidates),
        profile_history_used=history_used,
        profile_history_matches=history_matches,
        profile_history_estimates=history_estimates,
    )


def extract_circuit_features(
    circuit: Any,
    *,
    distributed: Any | None = None,
) -> CircuitFeatures:
    instructions = list(_effective_instructions(_instructions(circuit)))
    num_qubits = _num_qubits(circuit)
    if distributed is not None:
        num_qubits = max(num_qubits, _num_distributed_qubits(distributed))
    num_nodes = _num_nodes(distributed)

    one_q = two_q = multi_q = measurements = resets = conditionals = barriers = remote = 0
    distances: list[int] = []
    unsupported_mps: list[str] = []

    for inst in instructions:
        name = _instruction_name(inst)
        qubits = _instruction_qubits(inst)
        arity = len(qubits)

        if name == "measure":
            measurements += 1
            continue
        if name == "reset":
            resets += 1
            continue
        if name == "conditional":
            conditionals += 1
            continue
        if name == "barrier":
            barriers += 1
            continue
        if name.startswith("remote") or name in _MPS_SUPPORTED_REMOTE_GATES:
            remote += 1

        if arity == 1:
            one_q += 1
        elif arity == 2:
            two_q += 1
            distances.append(abs(qubits[0] - qubits[1]))
        elif arity > 2:
            multi_q += 1

        rejection = _mps_instruction_rejection(name, arity)
        if rejection is not None:
            unsupported_mps.append(rejection)

    avg_distance = sum(distances) / len(distances) if distances else 0.0
    max_distance = max(distances, default=0)
    routing_swaps = sum(max(0, 2 * (distance - 1)) for distance in distances)
    adjacent_2q = two_q + routing_swaps
    two_q_density = two_q / max(1, len(instructions))
    measurement_density = measurements / max(1, len(instructions))
    two_q_per_qubit = two_q / max(1, num_qubits)
    distance_pressure = avg_distance / max(1, num_qubits - 1)
    entanglement_risk = min(1.0, (two_q_per_qubit / 8.0) + (0.35 * distance_pressure))

    return CircuitFeatures(
        num_qubits=num_qubits,
        num_instructions=len(instructions),
        num_one_qubit_gates=one_q,
        num_two_qubit_gates=two_q,
        num_multi_qubit_gates=multi_q,
        num_measurements=measurements,
        num_resets=resets,
        num_conditionals=conditionals,
        num_barriers=barriers,
        num_remote_gates=remote,
        num_nodes=num_nodes,
        is_distributed=distributed is not None,
        two_qubit_density=two_q_density,
        measurement_density=measurement_density,
        average_two_qubit_distance=avg_distance,
        max_two_qubit_distance=max_distance,
        estimated_mps_routing_swaps=routing_swaps,
        estimated_mps_adjacent_2q_applications=adjacent_2q,
        estimated_mps_entanglement_risk=entanglement_risk,
        unsupported_mps_gates=tuple(sorted(set(unsupported_mps))),
    )


def simulate_auto_shots(
    circuit: Any | None = None,
    *,
    distributed: Any | None = None,
    shots: int = 1000,
    seed: int | None = None,
    allow_aer: bool = False,
    return_selection: bool = False,
    mps_routing: str = "restore",
    collect_profile: bool = False,
    max_bond_dimension: int | None = None,
    truncation_threshold: float = 1e-12,
    selection_mode: str = "heuristic",
    confidence_threshold: float = 0.65,
    pilot_shots: int = 3,
    use_profile_history: bool = False,
    profile_history_dir: str | Path | None = None,
) -> Any:
    """Run shots using the selected simulator.

    By default this only selects native dqsim simulators. Set allow_aer=True if
    the automatic runner may choose and execute Qiskit Aer. Use
    selection_mode="pilot" to run a tiny low-confidence pilot benchmark before
    the full run.
    """

    circuit = _resolve_circuit(circuit, distributed)
    selection = select_simulator(
        circuit,
        distributed=distributed,
        shots=shots,
        allow_aer=allow_aer,
        mps_routing=mps_routing,
        use_profile_history=use_profile_history,
        profile_history_dir=profile_history_dir,
    )
    selection = _apply_selection_mode(
        selection,
        selection_mode,
        confidence_threshold,
        pilot_shots,
        circuit,
        distributed,
        shots,
        seed,
        max_bond_dimension,
        truncation_threshold,
    )

    result = _run_backend_shots(
        selection.selected,
        circuit,
        distributed,
        shots,
        seed,
        collect_profile,
        max_bond_dimension,
        truncation_threshold,
    )

    if return_selection:
        return result, selection
    return result


def _apply_selection_mode(
    selection: SelectionResult,
    selection_mode: str,
    confidence_threshold: float,
    pilot_shots: int,
    circuit: Any,
    distributed: Any | None,
    shots: int,
    seed: int | None,
    max_bond_dimension: int | None,
    truncation_threshold: float,
) -> SelectionResult:
    if selection_mode not in {"heuristic", "pilot", "always_pilot"}:
        raise ValueError('selection_mode must be "heuristic", "pilot", or "always_pilot"')
    if confidence_threshold <= 0.0 or confidence_threshold > 1.0:
        raise ValueError("confidence_threshold must be in (0.0, 1.0]")
    if pilot_shots <= 0:
        raise ValueError("pilot_shots must be positive")

    selection = replace(selection, selection_mode=selection_mode)
    if selection_mode == "heuristic":
        return selection
    if selection_mode == "pilot" and selection.confidence >= confidence_threshold:
        return selection

    pilot_n = min(shots, pilot_shots)
    timings_ms: dict[str, float] = {}
    failures: dict[str, str] = {}
    for candidate in selection.eligible:
        started = time.perf_counter()
        try:
            _run_backend_shots(
                candidate.simulator,
                circuit,
                distributed,
                pilot_n,
                seed,
                False,
                max_bond_dimension,
                truncation_threshold,
            )
        except Exception as exc:  # pragma: no cover - defensive for optional backends
            failures[candidate.simulator] = str(exc)
        else:
            timings_ms[candidate.simulator] = (time.perf_counter() - started) * 1000.0

    if not timings_ms:
        reason = selection.reason + " Pilot benchmark was attempted but all eligible simulators failed."
        if failures:
            reason += " Failures: " + _format_failures(failures) + "."
        return replace(
            selection,
            reason=reason,
            pilot_used=True,
            pilot_shots=pilot_n,
            pilot_timings_ms={},
        )

    winner = min(timings_ms, key=lambda sim: (timings_ms[sim], _sim_sort(sim)))
    ordered_timings = sorted(timings_ms, key=lambda sim: (timings_ms[sim], _sim_sort(sim)))
    timing_summary = ", ".join(f"{sim}={timings_ms[sim]:.3f} ms" for sim in ordered_timings)
    reason = (
        f"{winner} selected by {pilot_n}-shot pilot benchmark after "
        f"heuristic confidence {selection.confidence:.2f}. "
        f"Pilot timings: {timing_summary}."
    )
    if failures:
        reason += " Skipped failed pilots: " + _format_failures(failures) + "."

    return replace(
        selection,
        selected=winner,
        confidence=max(selection.confidence, _pilot_confidence(timings_ms)),
        reason=reason,
        pilot_used=True,
        pilot_shots=pilot_n,
        pilot_timings_ms=dict(timings_ms),
    )


def _run_backend_shots(
    simulator: str,
    circuit: Any,
    distributed: Any | None,
    shots: int,
    seed: int | None,
    collect_profile: bool,
    max_bond_dimension: int | None,
    truncation_threshold: float,
) -> Any:
    if simulator == "dqsim-sv":
        return StatevectorSimulator(seed=seed).simulate_shots(
            circuit, shots=shots, collect_profile=collect_profile
        )
    if simulator == "dqsim-mps":
        return MpsSimulator(
            seed=seed,
            max_bond_dimension=max_bond_dimension,
            truncation_threshold=truncation_threshold,
        ).simulate_shots(circuit, shots=shots, collect_profile=collect_profile)
    if simulator == "pblock":
        if distributed is None:
            raise ValueError("PBlock selection requires distributed=")
        return PBlockSimulator(seed=seed).simulate_shots(
            distributed, shots=shots, collect_profile=collect_profile
        )
    if simulator == "aer-sv":
        return _simulate_aer_shots(circuit, "statevector", shots, seed)
    if simulator == "aer-mps":
        return _simulate_aer_shots(circuit, "matrix_product_state", shots, seed)
    raise RuntimeError(f"Unknown selected simulator: {simulator}")


def _pilot_confidence(timings_ms: dict[str, float]) -> float:
    if len(timings_ms) < 2:
        return 0.95
    ordered = sorted(timings_ms.values())
    winner = max(ordered[0], 1e-9)
    runner_up = ordered[1]
    ratio = runner_up / winner
    return min(0.95, max(0.65, 0.60 + (ratio - 1.0) * 0.35))


def _format_failures(failures: dict[str, str]) -> str:
    return ", ".join(f"{sim} ({message})" for sim, message in sorted(failures.items()))


def _score_candidates(
    features: CircuitFeatures,
    shots: int,
    allow_aer: bool,
    mps_routing: str,
) -> list[SimulatorCandidate]:
    candidates = [
        _score_dqsim_sv(features, shots),
        _score_dqsim_mps(features, shots, mps_routing),
        _score_pblock(features, shots),
    ]
    if allow_aer:
        if _aer_available():
            candidates.extend([_score_aer_sv(features, shots), _score_aer_mps(features, shots)])
        else:
            candidates.extend(
                [
                    SimulatorCandidate(
                        "aer-sv", False, None, rejection_reason="qiskit-aer is not installed"
                    ),
                    SimulatorCandidate(
                        "aer-mps", False, None, rejection_reason="qiskit-aer is not installed"
                    ),
                ]
            )
    return candidates


def _apply_profile_history(
    candidates: list[SimulatorCandidate],
    features: CircuitFeatures,
    shots: int,
    profile_history_dir: str | Path | None,
) -> tuple[list[SimulatorCandidate], bool, int, dict[str, float]]:
    points = _load_profile_history(profile_history_dir)
    if not points:
        return candidates, False, 0, {}

    wanted_variant = "lowered" if features.is_distributed else "original"
    variant_points = [point for point in points if point.variant in {wanted_variant, "unknown"}]
    if not variant_points:
        variant_points = points

    estimates: dict[str, float] = {}
    matches = 0
    for candidate in candidates:
        if not candidate.eligible:
            continue
        nearby = sorted(
            (
                (_profile_distance(point, features, shots), point)
                for point in variant_points
                if point.simulator == candidate.simulator
            ),
            key=lambda item: item[0],
        )[:3]
        if not nearby:
            continue
        weight_total = 0.0
        weighted_us = 0.0
        for distance, point in nearby:
            weight = 1.0 / (0.15 + distance)
            weight_total += weight
            weighted_us += point.per_shot_us * weight
        estimates[candidate.simulator] = weighted_us / weight_total
        matches += len(nearby)

    if len(estimates) < 2:
        return candidates, False, matches, estimates

    best_us = max(1e-9, min(estimates.values()))
    adjusted: list[SimulatorCandidate] = []
    for candidate in candidates:
        if not candidate.eligible or candidate.score is None or candidate.simulator not in estimates:
            adjusted.append(candidate)
            continue
        ratio = max(1.0, estimates[candidate.simulator] / best_us)
        history_penalty = math.log2(ratio) * 2.5
        reason = f"profile history estimate {estimates[candidate.simulator]:.2f} us/shot"
        adjusted.append(
            SimulatorCandidate(
                candidate.simulator,
                candidate.eligible,
                max(0.0, candidate.score + history_penalty),
                candidate.reasons + (reason,),
                candidate.rejection_reason,
            )
        )
    return adjusted, True, matches, estimates


def _load_profile_history(profile_history_dir: str | Path | None) -> list[_ProfilePoint]:
    history_dir = _default_profile_history_dir() if profile_history_dir is None else Path(profile_history_dir)
    if not history_dir.exists() or not history_dir.is_dir():
        return []

    points: list[_ProfilePoint] = []
    for path in sorted(history_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        simulator = _profile_simulator_name(data.get("simulator"), path.name)
        if simulator is None:
            continue
        try:
            num_qubits = int(data.get("num_qubits", 0))
            num_instructions = int(data.get("num_instructions", 0))
            num_shots = int(data.get("num_shots", 0))
            parallel_ms = float(data.get("parallel_execution_ms", data.get("total_time_ms", 0.0)))
        except (TypeError, ValueError):
            continue
        if num_qubits <= 0 or num_instructions <= 0 or num_shots <= 0 or parallel_ms <= 0.0:
            continue

        points.append(
            _ProfilePoint(
                simulator=simulator,
                variant=_profile_variant(data.get("circuit_variant"), path.name),
                num_qubits=num_qubits,
                num_instructions=num_instructions,
                num_shots=num_shots,
                parallel_execution_ms=parallel_ms,
                path=path,
            )
        )
    return points


def _default_profile_history_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "benchmarking" / "profiles"


def _profile_simulator_name(raw: Any, filename: str) -> str | None:
    if isinstance(raw, str):
        key = raw.lower().replace("-", "_")
        simulator = _PROFILE_SIMULATOR_ALIASES.get(key)
        if simulator is not None:
            return simulator
    for suffix, _variant, simulator in _PROFILE_SUFFIXES:
        if filename.endswith(suffix):
            return simulator
    return None


def _profile_variant(raw: Any, filename: str) -> str:
    if isinstance(raw, str):
        lowered = raw.lower()
        if lowered in {"original", "lowered"}:
            return lowered
    for suffix, variant, _simulator in _PROFILE_SUFFIXES:
        if filename.endswith(suffix):
            return variant
    if "_lowered_" in filename:
        return "lowered"
    if "_original_" in filename:
        return "original"
    return "unknown"


def _profile_distance(point: _ProfilePoint, features: CircuitFeatures, shots: int) -> float:
    qubit_gap = abs(point.num_qubits - features.num_qubits) / max(1, features.num_qubits)
    instruction_gap = abs(point.num_instructions - features.num_instructions) / max(
        1, features.num_instructions
    )
    shot_gap = abs(math.log2(max(1, point.num_shots)) - math.log2(max(1, shots)))
    return qubit_gap + 0.35 * instruction_gap + 0.10 * shot_gap


def _score_dqsim_sv(features: CircuitFeatures, shots: int) -> SimulatorCandidate:
    state_penalty = max(0, features.num_qubits - 12) * 0.85
    large_state_penalty = max(0, features.num_qubits - 16) * 2.5
    score = (
        0.45 * features.num_qubits
        + 0.010 * features.num_instructions
        + 0.020 * features.num_two_qubit_gates
        + state_penalty
        + large_state_penalty
        + _shot_pressure(shots) * 0.20
    )
    reasons = ["flat statevector kernels are cheap at this qubit count"]
    if features.estimated_mps_entanglement_risk >= 0.45:
        reasons.append("estimated MPS bond growth is high")
    if features.num_qubits >= 14:
        reasons.append("2^n state size is becoming the limiting cost")
    return SimulatorCandidate("dqsim-sv", True, score, tuple(reasons))


def _score_dqsim_mps(
    features: CircuitFeatures,
    shots: int,
    mps_routing: str,
) -> SimulatorCandidate:
    if features.unsupported_mps_gates:
        return SimulatorCandidate(
            "dqsim-mps",
            False,
            None,
            rejection_reason="unsupported by MPS: " + ", ".join(features.unsupported_mps_gates),
        )

    routing_factor = 0.0018 if mps_routing in {"lazy", "lookahead"} else 0.003
    if mps_routing == "lookahead":
        routing_factor = 0.0012
    score = (
        0.15 * features.num_qubits
        + 0.018 * features.num_instructions
        + 0.030 * features.num_two_qubit_gates
        + routing_factor * features.estimated_mps_routing_swaps
        + 4.8 * features.estimated_mps_entanglement_risk
        + _shot_pressure(shots) * 0.10
    )
    if features.num_qubits >= 12 and features.estimated_mps_entanglement_risk < 0.35:
        score -= 1.25

    reasons = ["avoids full 2^n state when entanglement stays limited"]
    if features.estimated_mps_routing_swaps:
        reasons.append(
            f"estimated routing swaps: {features.estimated_mps_routing_swaps}"
        )
    if features.estimated_mps_entanglement_risk >= 0.45:
        reasons.append("bond-growth risk is high")
    return SimulatorCandidate("dqsim-mps", True, max(0.0, score), tuple(reasons))


def _score_pblock(features: CircuitFeatures, shots: int) -> SimulatorCandidate:
    if not features.is_distributed:
        return SimulatorCandidate(
            "pblock",
            False,
            None,
            rejection_reason="requires a distributed circuit",
        )

    distributed_bonus = 1.0 + 0.25 * max(0, features.num_nodes - 1)
    structure_bonus = (
        0.16 * features.num_remote_gates
        + 0.035 * features.num_measurements
        + 0.050 * features.num_resets
        + 0.070 * features.num_conditionals
    )
    structure_bonus = min(structure_bonus, 2.0)
    score = (
        2.2
        + 0.020 * features.num_qubits
        + 0.009 * features.num_instructions
        + 0.018 * features.num_two_qubit_gates
        - distributed_bonus
        - structure_bonus
        + _shot_pressure(shots) * 0.08
    )
    if features.num_qubits >= 14 and shots <= 10:
        score += 4.0
    if features.num_qubits >= 14 and features.num_two_qubit_gates < features.num_qubits * 3:
        score += 1.5
    if features.num_instructions >= 200 or features.num_two_qubit_gates >= 60:
        score += 1.5

    reasons = ["distributed circuit supplied"]
    if structure_bonus > 0:
        reasons.append("remote/reset/measure structure matches PBlock")
    return SimulatorCandidate("pblock", True, max(0.0, score), tuple(reasons))


def _score_aer_sv(features: CircuitFeatures, shots: int) -> SimulatorCandidate:
    overhead = 2.0 if features.num_qubits <= 8 else 0.8
    score = (
        overhead
        + 0.36 * features.num_qubits
        + 0.007 * features.num_instructions
        + 0.014 * features.num_two_qubit_gates
        + max(0, features.num_qubits - 16) * 1.4
        + _shot_pressure(shots) * 0.04
    )
    if features.num_instructions >= 150 and features.num_qubits >= 10:
        score -= 1.0
    return SimulatorCandidate(
        "aer-sv",
        True,
        max(0.0, score),
        ("optimized Aer C++ statevector kernels",),
    )


def _score_aer_mps(features: CircuitFeatures, shots: int) -> SimulatorCandidate:
    if features.unsupported_mps_gates:
        return SimulatorCandidate(
            "aer-mps",
            False,
            None,
            rejection_reason="likely unsupported by MPS-style execution: "
            + ", ".join(features.unsupported_mps_gates),
        )

    small_high_shot_penalty = 1.8 if features.num_qubits <= 8 and shots >= 100 else 0.0
    score = (
        1.1
        + 0.13 * features.num_qubits
        + 0.012 * features.num_instructions
        + 0.022 * features.num_two_qubit_gates
        + 0.010 * features.estimated_mps_routing_swaps
        + 3.8 * features.estimated_mps_entanglement_risk
        + small_high_shot_penalty
        + _shot_pressure(shots) * 0.03
    )
    if features.num_qubits >= 12 and features.estimated_mps_entanglement_risk < 0.35:
        score -= 1.1
    return SimulatorCandidate(
        "aer-mps",
        True,
        max(0.0, score),
        ("Aer MPS can be strong on high-qubit low-entanglement circuits",),
    )


def _selection_reason(
    selected: SimulatorCandidate,
    eligible: list[SimulatorCandidate],
    features: CircuitFeatures,
) -> str:
    runner_up = eligible[1] if len(eligible) > 1 else None
    summary = (
        f"{features.num_qubits} qubits, {features.num_instructions} instructions, "
        f"{features.num_two_qubit_gates} two-qubit gates"
    )
    if runner_up is None or selected.score is None or runner_up.score is None:
        margin = "no runner-up"
    else:
        margin = f"next best: {runner_up.simulator}"
    reason = "; ".join(selected.reasons) if selected.reasons else "lowest heuristic score"
    return f"{selected.simulator} selected for {summary}; {reason}; {margin}."


def _confidence(eligible: list[SimulatorCandidate]) -> float:
    if len(eligible) < 2:
        return 0.95
    first = eligible[0].score or 0.0
    second = eligible[1].score or first
    gap = max(0.0, second - first)
    return min(0.95, max(0.52, 0.55 + gap / max(4.0, second)))


def _shot_pressure(shots: int) -> float:
    return math.log2(max(1, shots))


def _resolve_circuit(circuit: Any | None, distributed: Any | None) -> Any | None:
    if circuit is not None:
        return circuit
    if distributed is not None and hasattr(distributed, "as_monolithic_circuit"):
        return distributed.as_monolithic_circuit()
    return None


def _instructions(circuit: Any) -> list[Any]:
    instructions = getattr(circuit, "instructions", None)
    if instructions is None:
        return []
    return list(instructions)


def _effective_instructions(instructions: list[Any]) -> list[Any]:
    out = []
    for inst in instructions:
        out.append(inst)
        if _instruction_name(inst) == "conditional":
            op = getattr(inst, "op", None)
            if op is not None:
                out.append(op)
    return out


def _num_qubits(circuit: Any) -> int:
    if hasattr(circuit, "num_qubits"):
        num_qubits = getattr(circuit, "num_qubits")
        if callable(num_qubits):
            try:
                return int(num_qubits())
            except TypeError:
                pass
        else:
            return int(num_qubits)

    qregs = getattr(circuit, "qregs", {}) or {}
    max_qubit = 0
    for reg in qregs.values():
        base = int(getattr(reg, "base", 0))
        size = int(getattr(reg, "size", 0))
        max_qubit = max(max_qubit, base + size)
    return max_qubit


def _num_distributed_qubits(distributed: Any) -> int:
    qpn = getattr(distributed, "qubits_per_node", None) or {}
    qubits = [int(q) for values in qpn.values() for q in values]
    return max(qubits, default=-1) + 1


def _num_nodes(distributed: Any | None) -> int:
    if distributed is None:
        return 0
    qpn = getattr(distributed, "qubits_per_node", None)
    if qpn is not None:
        return len(qpn)
    circuits = getattr(distributed, "circuits", None)
    if circuits is not None:
        return len(circuits)
    return 1


def _instruction_name(inst: Any) -> str:
    for attr in ("kind", "name"):
        value = getattr(inst, attr, None)
        if isinstance(value, str) and value:
            return value.lower()
    cls = type(inst).__name__.lower()
    if cls.endswith("instruction"):
        cls = cls[: -len("instruction")]
    return cls


def _instruction_qubits(inst: Any) -> list[int]:
    qubits = []
    direct = getattr(inst, "qubits", None)
    if direct is not None and not callable(direct):
        try:
            qubits.extend(int(q) for q in direct)
        except TypeError:
            pass
    for attr in (
        "qubit",
        "control",
        "target",
        "control1",
        "control2",
        "control3",
        "control4",
        "a",
        "b",
    ):
        value = getattr(inst, attr, None)
        if value is not None:
            qubits.append(int(value))
    return sorted(set(qubits))


def _mps_instruction_rejection(name: str, arity: int) -> str | None:
    if name in _IGNORED_NAMES:
        return None
    if name == "remote_cu1":
        return "remote_cu1"
    if name.startswith("circuit-"):
        return name
    if name.startswith("remote") and name not in _MPS_SUPPORTED_REMOTE_GATES:
        return name
    if arity > 2:
        return f"{name or 'multi-qubit'}({arity}q)"
    return None


def _aer_available() -> bool:
    return importlib.util.find_spec("qiskit_aer") is not None


def _sim_sort(simulator: str) -> int:
    try:
        return _SIM_ORDER.index(simulator)
    except ValueError:
        return len(_SIM_ORDER)


def _simulate_aer_shots(
    circuit: Any,
    method: str,
    shots: int,
    seed: int | None,
) -> dict[str, int]:
    try:
        from bosonic_converters import CircuitConverters
        from qiskit.circuit import CircuitInstruction
        from qiskit.circuit.library import UnitaryGate
        from qiskit_aer import AerSimulator
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError("Aer execution requires qiskit-aer and bosonic-converters") from exc

    qc = CircuitConverters.to_qiskit(circuit)
    for _ in range(3):
        if any(getattr(inst.operation, "definition", None) for inst in qc.data):
            qc = qc.decompose()
        else:
            break

    try:
        from bosonic_converters.remote_link import (
            RemoteLinkGatePhiPlus,
            RemoteLinkGatePsiMinus,
            RemoteLinkGatePsiPlus,
        )
    except ImportError:
        pass
    else:
        phi_plus = UnitaryGate(RemoteLinkGatePhiPlus().to_matrix(), label="remote_link_phi_plus")
        psi_minus = UnitaryGate(
            RemoteLinkGatePsiMinus().to_matrix(), label="remote_link_psi_minus"
        )
        psi_plus = UnitaryGate(RemoteLinkGatePsiPlus().to_matrix(), label="remote_link_psi_plus")
        for i, ci in enumerate(qc.data):
            op = ci.operation
            if isinstance(op, RemoteLinkGatePhiPlus) or op.name == "remote_epr":
                qc.data[i] = CircuitInstruction(phi_plus, ci.qubits, ci.clbits)
            elif isinstance(op, RemoteLinkGatePsiMinus):
                qc.data[i] = CircuitInstruction(psi_minus, ci.qubits, ci.clbits)
            elif isinstance(op, RemoteLinkGatePsiPlus):
                qc.data[i] = CircuitInstruction(psi_plus, ci.qubits, ci.clbits)

    backend = AerSimulator(method=method, seed_simulator=seed)
    return dict(backend.run(qc, shots=shots, seed_simulator=seed).result().get_counts())
