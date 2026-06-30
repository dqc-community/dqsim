#!/usr/bin/env python3
"""
Export profiling data to different formats.

Usage:
    python3 scripts/export_profiles.py --csv                 # Export all to CSV
    python3 scripts/export_profiles.py --csv <output.csv>    # Export to specific file
    python3 scripts/export_profiles.py --html                # Generate HTML report
"""

import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def _fmt_optional_ms(value) -> str:
    if value is None:
        return ""
    return f"{float(value):.4f}"


def _load_profiles() -> list[tuple[Path, dict]]:
    profile_dir = Path(__file__).parent.parent / "benchmarking" / "profiles"
    profiles = sorted(profile_dir.glob("*.json"))

    if not profiles:
        print(f"No profiles found in {profile_dir}")
        return []

    loaded = []
    for profile_path in profiles:
        with open(profile_path, encoding="utf-8") as pf:
            loaded.append((profile_path, json.load(pf)))
    return loaded


def export_to_csv(output_path: Optional[Path] = None) -> None:
    """Export all profiles to CSV format."""
    loaded_profiles = _load_profiles()
    if not loaded_profiles:
        return

    profile_dir = Path(__file__).parent.parent / "benchmarking" / "profiles"
    if output_path is None:
        output_path = profile_dir / f"profiles_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Filename",
                "Simulator",
                "Timing Basis",
                "Qubits",
                "Instructions",
                "Shots",
                "Preprocessing (ms)",
                "Gate Fusion (ms)",
                "Parallel Execution (ms)",
                "Total (ms)",
                "Aer Experiment Time (ms)",
                "Aer Result Time (ms)",
                "Aer Job Wall (ms)",
                "Wall Total (ms)",
                "Per-Shot (ms)",
                "Per-Instruction-Per-Shot (us)",
            ]
        )

        for profile_path, data in loaded_profiles:
            num_shots = data["num_shots"]
            total_ms = data["total_time_ms"]
            num_inst = data["num_instructions"]

            per_shot_ms = total_ms / num_shots
            if num_inst > 0:
                per_inst_per_shot_us = 1000.0 * data["parallel_execution_ms"] / (
                    num_shots * num_inst
                )
            else:
                per_inst_per_shot_us = 0.0

            writer.writerow(
                [
                    profile_path.name,
                    data.get("simulator", ""),
                    data.get("timing_basis", "profile_total"),
                    data["num_qubits"],
                    num_inst,
                    num_shots,
                    f"{data['preprocessing_ms']:.4f}",
                    f"{data['gate_fusion_ms']:.4f}",
                    f"{data['parallel_execution_ms']:.4f}",
                    f"{total_ms:.4f}",
                    _fmt_optional_ms(data.get("aer_experiment_time_taken_ms")),
                    _fmt_optional_ms(data.get("aer_result_time_taken_ms")),
                    _fmt_optional_ms(data.get("aer_job_wall_ms")),
                    _fmt_optional_ms(data.get("wall_total_time_ms")),
                    f"{per_shot_ms:.4f}",
                    f"{per_inst_per_shot_us:.4f}",
                ]
            )

    print(f"Exported profiles to: {output_path}")


def export_to_html(output_path: Optional[Path] = None) -> None:
    """Export profiles to HTML report."""
    loaded_profiles = _load_profiles()
    if not loaded_profiles:
        return

    profile_dir = Path(__file__).parent.parent / "benchmarking" / "profiles"
    if output_path is None:
        output_path = profile_dir / f"profiles_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"

    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>DQSim Profiling Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: right; }
        th { background-color: #4CAF50; color: white; }
        tr:nth-child(even) { background-color: #f2f2f2; }
        .filename, .text { text-align: left; }
        h1 { color: #333; }
        .timestamp { color: #666; font-size: 0.9em; }
    </style>
</head>
<body>
    <h1>DQSim Profiling Report</h1>
    <p class="timestamp">Generated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>

    <h2>Profile Summary</h2>
    <table>
        <tr>
            <th class="filename">Profile</th>
            <th class="text">Simulator</th>
            <th class="text">Timing Basis</th>
            <th>Qubits</th>
            <th>Instructions</th>
            <th>Shots</th>
            <th>Preprocessing (ms)</th>
            <th>Gate Fusion (ms)</th>
            <th>Parallel Exec (ms)</th>
            <th>Total (ms)</th>
            <th>Aer Job Wall (ms)</th>
            <th>Wall Total (ms)</th>
            <th>Per-Shot (ms)</th>
        </tr>
"""

    for profile_path, data in loaded_profiles:
        html_content += f"""        <tr>
            <td class="filename">{profile_path.name}</td>
            <td class="text">{data.get('simulator', '')}</td>
            <td class="text">{data.get('timing_basis', 'profile_total')}</td>
            <td>{data['num_qubits']}</td>
            <td>{data['num_instructions']}</td>
            <td>{data['num_shots']}</td>
            <td>{data['preprocessing_ms']:.4f}</td>
            <td>{data['gate_fusion_ms']:.4f}</td>
            <td>{data['parallel_execution_ms']:.4f}</td>
            <td>{data['total_time_ms']:.4f}</td>
            <td>{_fmt_optional_ms(data.get('aer_job_wall_ms'))}</td>
            <td>{_fmt_optional_ms(data.get('wall_total_time_ms'))}</td>
            <td>{data['total_time_ms'] / data['num_shots']:.4f}</td>
        </tr>
"""

    html_content += """    </table>
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Exported HTML report to: {output_path}")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 scripts/export_profiles.py --csv                 # Export to CSV")
        print("  python3 scripts/export_profiles.py --csv <output.csv>    # Export to specific file")
        print("  python3 scripts/export_profiles.py --html                # Export to HTML")
        print("  python3 scripts/export_profiles.py --html <output.html>  # Export to specific file")
        sys.exit(1)

    if sys.argv[1] == "--csv":
        output = None
        if len(sys.argv) > 2:
            output = Path(sys.argv[2])
        export_to_csv(output)

    elif sys.argv[1] == "--html":
        output = None
        if len(sys.argv) > 2:
            output = Path(sys.argv[2])
        export_to_html(output)

    else:
        print(f"Unknown option: {sys.argv[1]}")
        sys.exit(1)


if __name__ == "__main__":
    main()