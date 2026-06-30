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
    wall_total_time_ms: float | None

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
            wall_total_time_ms=data.get("wall_total_time_ms"),
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


def print_profile(stats: ProfileStats) -> None:
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
        if stats.aer_experiment_time_taken_ms is not None:
            print(f"  Experiment time:  {stats.aer_experiment_time_taken_ms:10.4f}")
        if stats.aer_result_time_taken_ms is not None:
            print(f"  Result time:      {stats.aer_result_time_taken_ms:10.4f}")
        if stats.aer_job_wall_ms is not None:
            print(f"  Job wall time:    {stats.aer_job_wall_ms:10.4f}")
        if stats.wall_total_time_ms is not None:
            print(f"  Full wall time:   {stats.wall_total_time_ms:10.4f}")
            overhead_ms = stats.wall_total_time_ms - stats.total_time_ms
            print(f"  Wall overhead:    {overhead_ms:10.4f}")

    print("\nPer-Shot Analysis:")
    print(f"  Time per shot:    {stats.per_shot_ms():10.4f} ms")
    per_inst_per_shot_us = stats.per_instruction_per_shot_us()
    print(f"  Per instr/shot:   {per_inst_per_shot_us:10.4f} us")
    print()


def print_comparison(profiles: list[ProfileStats]) -> None:
    """Print a comparison table of multiple profiles."""
    if not profiles:
        print("No profiles to compare.")
        return

    print("\nProfile Comparison")
    print("=" * 120)
    print(
        f"{'Filename':<40} {'Basis':<24} {'Qubits':>6} {'Instr':>6} "
        f"{'Shots':>8} {'Total (ms)':>12} {'Per-Shot (ms)':>14}"
    )
    print("-" * 120)

    for stats in sorted(profiles, key=lambda s: s.total_time_ms):
        basis = stats.timing_basis or "profile_total"
        print(
            f"{stats.filename:<40} {basis:<24} {stats.num_qubits:>6} "
            f"{stats.num_instructions:>6} {stats.num_shots:>8} "
            f"{stats.total_time_ms:>12.4f} {stats.per_shot_ms():>14.4f}"
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
            print_profile(stat)

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
        print_profile(stats)


if __name__ == "__main__":
    main()