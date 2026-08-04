#!/usr/bin/env python3
"""
Utility script to summarize and analyze dqsim profiling data.

Usage:
    python3 scripts/summarize_profiles.py                    # Show all profiles
    python3 scripts/summarize_profiles.py <path_to_profile>  # Show specific profile
    python3 scripts/summarize_profiles.py --compare <dir>    # Compare profiles
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path


def _fmt_optional(value: float | None, width: int = 10) -> str:
    if value is None:
        return f"{'N/A':>{width}}"
    return f"{value:>{width}.4f}"


def _simd_label(data: dict) -> str:
    if data.get("statevector_simd_enabled") is None:
        return ""
    if data.get("statevector_simd_used"):
        return data.get("statevector_simd_backend") or "on"
    if data.get("statevector_simd_enabled"):
        return "ready"
    return "off"


def _two_qubit_label(data: dict) -> str:
    if data.get("statevector_2q_kernels_enabled") is None:
        return ""
    if data.get("statevector_2q_kernels_used"):
        return data.get("statevector_2q_kernel_gates") or "on"
    if data.get("statevector_2q_kernels_enabled"):
        return "none"
    return "off"


def _branch_label(data: dict) -> str:
    if data.get("statevector_shot_branching_enabled") is None:
        return ""
    if data.get("statevector_shot_branching_used"):
        return "term"
    if data.get("statevector_shot_branching_enabled"):
        return "fallback"
    return "off"


def _trunc_label(data: dict) -> str:
    if data.get("statevector_qubit_truncation_enabled") is None:
        return ""
    if data.get("statevector_qubit_truncation_used"):
        original = data.get("statevector_original_num_qubits")
        effective = data.get("statevector_effective_num_qubits")
        if original is not None and effective is not None:
            return f"{original}->{effective}"
        return "on"
    if data.get("statevector_qubit_truncation_enabled"):
        return "none"
    return "off"


@dataclass
class ProfileStats:
    """Statistics from a profile."""

    filename: str
    simulator: str
    timing_basis: str | None
    num_shots: int
    num_qubits: int
    num_instructions: int
    preprocessing_ms: float
    gate_fusion_ms: float
    parallel_execution_ms: float
    total_time_ms: float
    aer_experiment_time_taken_ms: float | None
    aer_result_time_taken_ms: float | None
    aer_job_wall_ms: float | None
    aer_time_taken_execute_ms: float | None
    aer_fusion_time_taken_ms: float | None
    sample_measure_time_ms: float | None
    wall_total_time_ms: float | None
    aer_wall_overhead_ms: float | None
    raw: dict

    @classmethod
    def from_file(cls, filepath: Path) -> "ProfileStats":
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            filename=filepath.name,
            simulator=data.get("simulator", ""),
            timing_basis=data.get("timing_basis"),
            num_shots=data["num_shots"],
            num_qubits=data["num_qubits"],
            num_instructions=data["num_instructions"],
            preprocessing_ms=data["preprocessing_ms"],
            gate_fusion_ms=data["gate_fusion_ms"],
            parallel_execution_ms=data["parallel_execution_ms"],
            total_time_ms=data["total_time_ms"],
            aer_experiment_time_taken_ms=data.get("aer_experiment_time_taken_ms"),
            aer_result_time_taken_ms=data.get("aer_result_time_taken_ms"),
            aer_job_wall_ms=data.get("aer_job_wall_ms"),
            aer_time_taken_execute_ms=data.get("aer_time_taken_execute_ms"),
            aer_fusion_time_taken_ms=data.get("aer_fusion_time_taken_ms"),
            sample_measure_time_ms=data.get("sample_measure_time_ms"),
            wall_total_time_ms=data.get("wall_total_time_ms"),
            aer_wall_overhead_ms=data.get("aer_wall_overhead_ms"),
            raw=data,
        )

    def per_shot_ms(self) -> float:
        """Total time per shot in milliseconds."""
        return self.total_time_ms / self.num_shots

    def preprocessing_pct(self) -> float:
        """Preprocessing as percentage of total."""
        return 100.0 * self.preprocessing_ms / self.total_time_ms

    def gate_fusion_pct(self) -> float:
        """Gate fusion as percentage of total."""
        return 100.0 * self.gate_fusion_ms / self.total_time_ms

    def execution_pct(self) -> float:
        """Parallel execution as percentage of total."""
        return 100.0 * self.parallel_execution_ms / self.total_time_ms

    def per_instruction_per_shot_us(self) -> float:
        """Parallel execution time per instruction per shot in microseconds."""
        if self.num_instructions == 0:
            return 0.0
        return 1000.0 * self.parallel_execution_ms / (
            self.num_shots * self.num_instructions
        )

    def uses_aer_experiment_time(self) -> bool:
        return self.timing_basis == "aer_experiment_time_taken"


def print_profile(stats: ProfileStats, detailed: bool = True) -> None:
    """Print a formatted profile summary."""
    print(f"\nProfile: {stats.filename}")
    print("=" * 70)

    print("\nCircuit Info:")
    if stats.simulator:
        print(f"  Simulator:        {stats.simulator}")
    print(f"  Qubits:           {stats.num_qubits}")
    print(f"  Instructions:     {stats.num_instructions} (fused)")
    print(f"  Shots:            {stats.num_shots}")
    if stats.timing_basis:
        print(f"  Timing basis:     {stats.timing_basis}")
    if stats.raw.get("statevector_simd_enabled") is not None:
        print(f"  SV SIMD enabled:  {stats.raw.get('statevector_simd_enabled')}")
        print(f"  SV SIMD used:     {stats.raw.get('statevector_simd_used')}")
        print(f"  SV SIMD backend:  {stats.raw.get('statevector_simd_backend') or 'N/A'}")
        print(f"  SV SIMD min q:    {stats.raw.get('statevector_simd_min_qubits')}")
    if stats.raw.get("statevector_2q_kernels_enabled") is not None:
        print(f"  SV 2q kernels:   {_two_qubit_label(stats.raw)}")
    if stats.raw.get("statevector_shot_branching_enabled") is not None:
        print(f"  SV branching on:  {stats.raw.get('statevector_shot_branching_enabled')}")
        print(f"  SV branching use: {stats.raw.get('statevector_shot_branching_used')}")
        print(f"  SV branch strat:  {stats.raw.get('statevector_shot_branching_strategy') or 'N/A'}")
    if stats.raw.get("statevector_qubit_truncation_enabled") is not None:
        print(f"  SV trunc on:      {stats.raw.get('statevector_qubit_truncation_enabled')}")
        print(f"  SV trunc used:    {stats.raw.get('statevector_qubit_truncation_used')}")
        print(f"  SV trunc q:       {_trunc_label(stats.raw)}")
        removed = stats.raw.get("statevector_removed_qubits") or []
        print(f"  SV trunc removed: {removed if removed else 'N/A'}")

    print("\nTiming (ms):")
    if stats.uses_aer_experiment_time():
        print(f"  Preprocessing:    {stats.preprocessing_ms:10.4f}  (diagnostic)")
        print(f"  Backend/Job Exec: {stats.parallel_execution_ms:10.4f}  (reported)")
    else:
        print(
            f"  Preprocessing:    {stats.preprocessing_ms:10.4f}  "
            f"({stats.preprocessing_pct():5.1f}%)"
        )
        print(
            f"  Gate Fusion:      {stats.gate_fusion_ms:10.4f}  "
            f"({stats.gate_fusion_pct():5.1f}%)"
        )
        print(
            f"  Parallel Exec:    {stats.parallel_execution_ms:10.4f}  "
            f"({stats.execution_pct():5.1f}%)"
        )
    print(f"  {'-' * 40}")
    print(f"  Total:            {stats.total_time_ms:10.4f}  (100.0%)")

    if stats.uses_aer_experiment_time():
        print("\nAer Diagnostics (ms):")
        print(f"  Experiment time:  {_fmt_optional(stats.aer_experiment_time_taken_ms)}")
        print(f"  Controller exec:  {_fmt_optional(stats.aer_time_taken_execute_ms)}")
        print(f"  Fusion time:      {_fmt_optional(stats.aer_fusion_time_taken_ms)}")
        print(f"  Sample measure:   {_fmt_optional(stats.sample_measure_time_ms)}")
        print(f"  Result time:      {_fmt_optional(stats.aer_result_time_taken_ms)}")
        print(f"  Job wall time:    {_fmt_optional(stats.aer_job_wall_ms)}")
        print(f"  Full wall time:   {_fmt_optional(stats.wall_total_time_ms)}")
        print(f"  Wall overhead:    {_fmt_optional(stats.aer_wall_overhead_ms)}")

    print("\nPer-Shot Analysis:")
    print(f"  Time per shot:    {stats.per_shot_ms():10.4f} ms")
    per_inst_per_shot_us = stats.per_instruction_per_shot_us()
    print(f"  Per instr/shot:   {per_inst_per_shot_us:10.4f} us")

    if detailed:
        print_function_breakdown(stats)
    print()


def print_function_breakdown(stats: ProfileStats) -> None:
    source_breakdown = stats.raw.get("source_function_breakdown") or []
    if source_breakdown:
        print("\nSource Function Breakdown:")
        print(f"  {'Phase':<28} {'Metric':<28} {'Time (ms)':>10}  Function")
        for item in source_breakdown:
            function = item.get("dqsim_function") or item.get("aer_function") or ""
            print(
                f"  {item.get('phase', ''):<28} {item.get('metric', ''):<28} "
                f"{_fmt_optional(item.get('time_ms'))}  {function}"
            )

    comparison = stats.raw.get("aer_function_comparison") or {}
    counterparts = comparison.get("counterparts") or []
    if counterparts:
        print("\nAer Function Comparison:")
        print(f"  Note: {comparison.get('note', '')}")
        for counterpart in counterparts:
            aer_label = counterpart.get("aer_profile_key", counterpart.get("aer_simulator", "Aer"))
            print(f"\n  Compared to: {aer_label} ({counterpart.get('aer_timing_basis', 'unknown basis')})")
            print(f"  {'Phase':<30} {'DQSim ms':>10} {'Aer ms':>10} {'Ratio':>10}  Aer function")
            for phase in counterpart.get("phases", []):
                ratio = phase.get("dqsim_to_aer_ratio")
                ratio_s = "N/A" if ratio is None else f"{ratio:.4f}"
                print(
                    f"  {phase.get('phase', ''):<30} "
                    f"{_fmt_optional(phase.get('dqsim_ms'))} "
                    f"{_fmt_optional(phase.get('aer_ms'))} "
                    f"{ratio_s:>10}  {phase.get('aer_function', '')}"
                )


def print_comparison(profiles: list[ProfileStats]) -> None:
    """Print a comparison table of multiple profiles."""
    if not profiles:
        print("No profiles to compare.")
        return

    print("\nProfile Comparison")
    print("=" * 120)
    print(
        f"{'Filename':<40} {'Basis':<24} {'Qubits':>6} {'Instr':>6} "
        f"{'Shots':>8} {'SIMD':>7} {'2Q':>12} {'Branch':>8} {'Trunc':>8} {'Total (ms)':>12} {'Per-Shot (ms)':>14}"
    )
    print("-" * 120)

    for stats in sorted(profiles, key=lambda s: s.total_time_ms):
        basis = stats.timing_basis or "profile_total"
        print(
            f"{stats.filename:<40} {basis:<24} {stats.num_qubits:>6} "
            f"{stats.num_instructions:>6} {stats.num_shots:>8} "
            f"{_simd_label(stats.raw):>7} {_two_qubit_label(stats.raw):>12} {_branch_label(stats.raw):>8} {_trunc_label(stats.raw):>8} "
            f"{stats.total_time_ms:>12.4f} "
            f"{stats.per_shot_ms():>14.4f}"
        )

    print()


def find_profiles(directory: Path) -> list[Path]:
    """Find all JSON profile files in a directory."""
    if not directory.exists():
        print(f"Directory not found: {directory}")
        return []

    return sorted(directory.glob("*.json"))


def main() -> None:
    if len(sys.argv) < 2:
        profile_dir = Path(__file__).parent.parent / "benchmarking" / "profiles"
        profiles = find_profiles(profile_dir)

        if not profiles:
            print(f"No profiles found in {profile_dir}")
            print("Run 'make perf' or 'pytest benchmarking/benchmarking_suite.py -v' to generate profiles.")
            return

        stats = [ProfileStats.from_file(p) for p in profiles]
        print_comparison(stats)

        for stat in stats:
            print_profile(stat, detailed=False)

    elif sys.argv[1] == "--compare":
        if len(sys.argv) < 3:
            print("Usage: summarize_profiles.py --compare <directory>")
            sys.exit(1)

        directory = Path(sys.argv[2])
        profiles = find_profiles(directory)

        if not profiles:
            print(f"No profiles found in {directory}")
            return

        stats = [ProfileStats.from_file(p) for p in profiles]
        print_comparison(stats)

    else:
        filepath = Path(sys.argv[1])
        if not filepath.exists():
            print(f"File not found: {filepath}")
            sys.exit(1)

        stats = ProfileStats.from_file(filepath)
        print_profile(stats, detailed=True)


if __name__ == "__main__":
    main()