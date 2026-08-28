#!/usr/bin/env python3
"""
Summarize which simulator fits each benchmark circuit family best.

Run after `make perf` so benchmarking/profiles contains fresh JSON profiles.
The report only ranks simulators that ran the same circuit variant with the
same shot count. PBlock is therefore compared on lowered distributed circuits,
not against original circuits it cannot run directly.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROFILE_SUFFIXES = (
    ("_original_aer_statevector.json", "original", "aer-sv"),
    ("_original_aer_mps.json", "original", "aer-mps"),
    ("_original_sv.json", "original", "dqsim-sv"),
    ("_original_mps.json", "original", "dqsim-mps"),
    ("_lowered_aer_statevector.json", "lowered", "aer-sv"),
    ("_lowered_aer_mps.json", "lowered", "aer-mps"),
    ("_lowered_pblock.json", "lowered", "pblock"),
    ("_lowered_sv.json", "lowered", "dqsim-sv"),
    ("_lowered_mps.json", "lowered", "dqsim-mps"),
    ("_symbolic_pblock.json", "symbolic", "pblock-symbolic"),
)

SIM_ORDER = ("dqsim-sv", "dqsim-mps", "pblock", "aer-sv", "aer-mps")
VARIANT_ORDER = {"original": 0, "lowered": 1, "symbolic": 2}


@dataclass(frozen=True)
class ProfileEntry:
    circuit: str
    variant: str
    simulator: str
    family: str
    metric_ms: float
    total_ms: float
    parallel_ms: float
    shots: int
    qubits: int
    instructions: int
    raw: dict

    @property
    def per_shot_us(self) -> float:
        return 1_000.0 * self.metric_ms / self.shots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profiles",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "benchmarking" / "profiles",
        help="Directory containing benchmark profile JSON files.",
    )
    parser.add_argument(
        "--metric",
        choices=("parallel", "total"),
        default="parallel",
        help=(
            "Timing metric for ranking. 'parallel' compares simulator execution "
            "phases; 'total' uses each profile total."
        ),
    )
    parser.add_argument(
        "--variant",
        choices=("all", "original", "lowered"),
        default="all",
        help="Restrict the report to one comparable circuit variant.",
    )
    parser.add_argument(
        "--include-symbolic",
        action="store_true",
        help="Also show symbolic PBlock smoke profiles. They are not cross-simulator comparisons.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="Write machine-readable analysis to this path.",
    )
    return parser.parse_args()


def load_entries(profile_dir: Path, metric: str, include_symbolic: bool) -> list[ProfileEntry]:
    entries = []
    for path in sorted(profile_dir.glob("*.json")):
        parsed = parse_profile_filename(path.name)
        if parsed is None:
            continue

        circuit, variant, simulator = parsed
        if variant == "symbolic" and not include_symbolic:
            continue

        data = json.loads(path.read_text(encoding="utf-8"))
        metric_key = "parallel_execution_ms" if metric == "parallel" else "total_time_ms"
        metric_ms = float(data[metric_key])
        entries.append(
            ProfileEntry(
                circuit=circuit,
                variant=variant,
                simulator=simulator,
                family=circuit_family(circuit),
                metric_ms=metric_ms,
                total_ms=float(data["total_time_ms"]),
                parallel_ms=float(data["parallel_execution_ms"]),
                shots=int(data["num_shots"]),
                qubits=int(data["num_qubits"]),
                instructions=int(data["num_instructions"]),
                raw=data,
            )
        )
    return entries


def parse_profile_filename(filename: str) -> tuple[str, str, str] | None:
    for suffix, variant, simulator in PROFILE_SUFFIXES:
        if filename.endswith(suffix):
            return filename[: -len(suffix)], variant, simulator
    return None


def circuit_family(circuit: str) -> str:
    if circuit.startswith(("deutsch", "bv")):
        return "oracle"
    if circuit.startswith(("toffoli", "cc")):
        return "controlled"
    if circuit.startswith("adder"):
        return "arithmetic"
    if circuit.startswith(("qft", "qpe")):
        return "phase/fourier"
    if circuit.startswith("bell"):
        return "entanglement"
    if circuit.startswith(("qaoa", "ising")):
        return "local/variational"
    return "other"


def comparable_cases(
    entries: Iterable[ProfileEntry], variant_filter: str
) -> dict[tuple[str, str], list[ProfileEntry]]:
    grouped: dict[tuple[str, str], list[ProfileEntry]] = defaultdict(list)
    for entry in entries:
        if entry.variant == "symbolic":
            continue
        if variant_filter != "all" and entry.variant != variant_filter:
            continue
        grouped[(entry.variant, entry.circuit)].append(entry)

    return {
        key: sorted(value, key=lambda item: (item.metric_ms, sim_sort_key(item.simulator)))
        for key, value in grouped.items()
        if len(value) >= 2
    }


def sim_sort_key(simulator: str) -> int:
    try:
        return SIM_ORDER.index(simulator)
    except ValueError:
        return len(SIM_ORDER)


def case_sort_key(item: tuple[tuple[str, str], list[ProfileEntry]]) -> tuple[int, str]:
    (variant, circuit), _entries = item
    return (VARIANT_ORDER.get(variant, 99), circuit)


def fmt_ms(value: float) -> str:
    return f"{value:.4f}"


def fmt_ratio(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.2f}x"


def rank_line(entries: list[ProfileEntry]) -> str:
    return " > ".join(f"{entry.simulator}:{entry.per_shot_us:.2f}us" for entry in entries)


def print_case_table(cases: dict[tuple[str, str], list[ProfileEntry]], metric: str) -> None:
    print(f"\nFair Simulator Fit ({metric} timing)")
    print("=" * 128)
    print(
        f"{'Variant':<9} {'Circuit':<14} {'Family':<18} {'Qb':>4} {'Inst':>6} "
        f"{'Shots':>7} {'Winner':<10} {'Win us/shot':>12} {'Margin':>8}  Ranking"
    )
    print("-" * 128)

    for (_variant, _circuit), entries in sorted(cases.items(), key=case_sort_key):
        winner = entries[0]
        runner_up = entries[1] if len(entries) > 1 else None
        margin = None if runner_up is None else runner_up.metric_ms / winner.metric_ms
        print(
            f"{winner.variant:<9} {winner.circuit:<14} {winner.family:<18} "
            f"{winner.qubits:>4} {winner.instructions:>6} {winner.shots:>7} "
            f"{winner.simulator:<10} {winner.per_shot_us:>12.2f} "
            f"{fmt_ratio(margin):>8}  {rank_line(entries)}"
        )


def print_win_summary(cases: dict[tuple[str, str], list[ProfileEntry]]) -> None:
    wins: dict[tuple[str, str, str], int] = defaultdict(int)
    ranks: dict[tuple[str, str, str], list[int]] = defaultdict(list)

    for entries in cases.values():
        for idx, entry in enumerate(entries, 1):
            key = (entry.variant, entry.family, entry.simulator)
            ranks[key].append(idx)
        winner = entries[0]
        wins[(winner.variant, winner.family, winner.simulator)] += 1

    groups = sorted({(variant, family) for variant, family, _sim in ranks})
    print("\nWins By Circuit Family")
    print("=" * 86)
    print(f"{'Variant':<9} {'Family':<18} {'Cases':>5} {'Wins':<34} {'Best avg rank':<18}")
    print("-" * 86)
    for variant, family in groups:
        sims = sorted(
            {sim for v, f, sim in ranks if v == variant and f == family},
            key=sim_sort_key,
        )
        case_count = max(len(ranks[(variant, family, sim)]) for sim in sims)
        win_parts = [
            f"{sim}:{wins.get((variant, family, sim), 0)}"
            for sim in sims
            if wins.get((variant, family, sim), 0)
        ]
        avg_rank = sorted(
            (
                statistics.mean(ranks[(variant, family, sim)]),
                sim,
            )
            for sim in sims
        )[0]
        print(
            f"{variant:<9} {family:<18} {case_count:>5} "
            f"{', '.join(win_parts) or 'none':<34} "
            f"{avg_rank[1]} ({avg_rank[0]:.2f})"
        )


def print_dominance(cases: dict[tuple[str, str], list[ProfileEntry]]) -> None:
    by_case = {
        case: {entry.simulator: entry for entry in entries}
        for case, entries in cases.items()
    }
    sims = sorted({sim for entries in by_case.values() for sim in entries}, key=sim_sort_key)

    print("\nDominance Checks")
    print("=" * 104)
    print(
        f"{'Simulator':<10} {'Comparable':>10} {'Wins':>6} {'Beaten by every shared rival?':<34} "
        "Pairwise faster-than"
    )
    print("-" * 104)
    for sim in sims:
        comparable = [entries for entries in by_case.values() if sim in entries]
        win_count = sum(
            1 for entries in comparable if entries[sim].metric_ms == min(e.metric_ms for e in entries.values())
        )
        pair_parts = []
        globally_dominated = True
        for other in sims:
            if other == sim:
                continue
            shared = [entries for entries in by_case.values() if sim in entries and other in entries]
            if not shared:
                continue
            faster = sum(1 for entries in shared if entries[sim].metric_ms < entries[other].metric_ms)
            if faster > 0:
                globally_dominated = False
            pair_parts.append(f"vs {other}:{faster}/{len(shared)}")

        if win_count > 0:
            globally_dominated = False
        verdict = "yes" if globally_dominated and comparable else "no"
        print(
            f"{sim:<10} {len(comparable):>10} {win_count:>6} {verdict:<34} "
            f"{'; '.join(pair_parts)}"
        )


def print_mps_diagnostics(cases: dict[tuple[str, str], list[ProfileEntry]]) -> None:
    mps_entries = [
        entry
        for entries in cases.values()
        for entry in entries
        if entry.simulator == "dqsim-mps" and entry.raw.get("mps_logical_2q_gates") is not None
    ]
    if not mps_entries:
        return

    print("\nDQSim MPS Routing Diagnostics")
    print("=" * 118)
    print(
        f"{'Variant':<9} {'Circuit':<14} {'Mode':<10} {'2Q':>6} {'Swaps':>7} "
        f"{'Swaps/2Q':>9} {'SVDs':>7} {'SVD ms':>10} {'Max bond':>8} {'Avg dist':>8}"
    )
    print("-" * 118)
    for entry in sorted(mps_entries, key=lambda item: (VARIANT_ORDER[item.variant], item.circuit)):
        two_q = int(entry.raw.get("mps_logical_2q_gates") or 0)
        swaps = int(entry.raw.get("mps_routing_swaps") or 0)
        swaps_per_gate = 0.0 if two_q == 0 else swaps / two_q
        print(
            f"{entry.variant:<9} {entry.circuit:<14} "
            f"{entry.raw.get('mps_routing_mode') or 'unknown':<10} "
            f"{two_q:>6} {swaps:>7} {swaps_per_gate:>9.2f} "
            f"{int(entry.raw.get('mps_svd_count') or 0):>7} "
            f"{float(entry.raw.get('mps_svd_time_ms') or 0.0):>10.4f} "
            f"{int(entry.raw.get('mps_max_observed_bond_dimension') or 0):>8} "
            f"{float(entry.raw.get('mps_average_routed_distance') or 0.0):>8.2f}"
        )


def analysis_json(cases: dict[tuple[str, str], list[ProfileEntry]], metric: str) -> dict:
    rows = []
    for (_variant, _circuit), entries in sorted(cases.items(), key=case_sort_key):
        rows.append(
            {
                "variant": entries[0].variant,
                "circuit": entries[0].circuit,
                "family": entries[0].family,
                "metric": metric,
                "winner": entries[0].simulator,
                "ranking": [
                    {
                        "simulator": entry.simulator,
                        "metric_ms": entry.metric_ms,
                        "per_shot_us": entry.per_shot_us,
                        "qubits": entry.qubits,
                        "instructions": entry.instructions,
                        "shots": entry.shots,
                    }
                    for entry in entries
                ],
            }
        )
    return {"metric": metric, "cases": rows}


def main() -> int:
    args = parse_args()
    entries = load_entries(args.profiles, args.metric, args.include_symbolic)
    cases = comparable_cases(entries, args.variant)
    if not cases:
        print(f"No comparable profiles found in {args.profiles}")
        return 1

    print(
        "Timing note: comparisons are only within the same circuit variant. "
        "The default metric is parallel_execution_ms, which is the closest "
        "profiled simulator-execution phase across DQSim and Aer profiles."
    )
    print_case_table(cases, args.metric)
    print_win_summary(cases)
    print_dominance(cases)
    print_mps_diagnostics(cases)

    if args.json:
        args.json.write_text(
            json.dumps(analysis_json(cases, args.metric), indent=2),
            encoding="utf-8",
        )
        print(f"\nWrote analysis JSON: {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
