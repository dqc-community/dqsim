# DQSim Profiler Guide

## Overview

The DQSim profiler provides detailed timing information for different stages of quantum circuit simulation. This guide explains how to use the profiler and analyze the profiling data.

## Profiling Architecture

The profiler tracks execution time across the following stages:

### 1. **Preprocessing** (`preprocessing_ms`)
- Circuit deserialization from JSON to Rust representation
- Initial state vector creation
- Classical bits initialization

### 2. **Gate Fusion** (`gate_fusion_ms`)
- Consecutive single-qubit gates are identified and combined into composite gates
- This optimization happens once per simulation, before the parallel shot loop
- Time includes both fusion identification and composition

### 3. **Parallel Execution** (`parallel_execution_ms`)
- Execution of the actual shot loop
- 1000 independent shots executed in parallel using Rayon
- Each shot applies the fused gates and handles measurements
- Time measured for the entire parallel block

### 4. **Total Time** (`total_time_ms`)
- Sum of preprocessing, gate fusion, and parallel execution
- Does NOT include JSON serialization to Python (already accounted for in preprocessing)

## Using the Profiler

### Basic Usage

```python
from dqsim import StatevectorSimulator
from bosonic_model.qasm import Translator
import qasmpi

# Load a circuit
circuit = Translator().from_qasm(qasmpi.get_circuit("deutsch_n2"))

# Create simulator
sim = StatevectorSimulator(seed=42)

# Run with profiling enabled
result = sim.simulate_shots(
    circuit, 
    shots=1000,
    collect_profile=True  # Enable profiling
)

# Extract results
counts = result["counts"]
profile = result["profile"]

print(f"Total time: {profile['total_time_ms']:.2f}ms")
print(f"  Preprocessing: {profile['preprocessing_ms']:.4f}ms")
print(f"  Gate fusion: {profile['gate_fusion_ms']:.4f}ms")
print(f"  Parallel execution: {profile['parallel_execution_ms']:.4f}ms")
```

### Using the Benchmarking Suite

The benchmarking suite automatically collects profiling data for all circuits:

```bash
# Run benchmarks with profiling (default)
make perf

# Profiling JSON files are saved to: benchmarking/profiles/
# Format: {circuit_name}_{simulator}_{timestamp}.json
```

## Profiling Data Format

Each profile is a JSON object with the following structure:

```json
{
  "num_shots": 1000,
  "num_qubits": 4,
  "num_instructions": 14,
  "preprocessing_ms": 0.217864,
  "gate_fusion_ms": 0.020338,
  "parallel_execution_ms": 6.266558,
  "per_shot_stats": null,
  "total_time_ms": 6.522431
}
```

### Field Descriptions

- **num_shots**: Number of independent circuit executions
- **num_qubits**: Number of qubits in the circuit
- **num_instructions**: Number of fused instructions (after optimization)
- **preprocessing_ms**: Time for circuit deserialization and state initialization (milliseconds)
- **gate_fusion_ms**: Time for gate fusion optimization (milliseconds)
- **parallel_execution_ms**: Time for the parallel shot loop (milliseconds)
- **per_shot_stats**: Per-shot timing data (currently null, reserved for future use)
- **total_time_ms**: Total elapsed time (milliseconds)

## Analyzing Profiling Data

### Time Breakdown

The total time is distributed among three stages:

```
Total = Preprocessing + Gate Fusion + Parallel Execution
```

For a 4-qubit circuit with 1000 shots:

| Stage | Time (ms) | Percentage |
|-------|-----------|------------|
| Preprocessing | 0.22 | 3.3% |
| Gate Fusion | 0.02 | 0.3% |
| Parallel Execution | 6.27 | 96.1% |
| **Total** | **6.52** | **100%** |

### Key Insights

1. **Parallel Execution Dominance**: The parallel execution phase typically dominates (>90%), especially for larger shot counts.

2. **Preprocessing Overhead**: Decreases as a percentage with more shots, but is a fixed cost per simulation.

3. **Gate Fusion Benefit**: The gate fusion step is typically very fast but provides significant speedup in the parallel execution phase.

### Comparing Simulators

When comparing different simulators (SV, MPS, PBlock), look at:

1. **Total time per shot**: `total_time_ms / num_shots`
2. **Parallel execution efficiency**: `parallel_execution_ms / (num_shots * num_instructions)`
3. **Overhead ratio**: `(preprocessing_ms + gate_fusion_ms) / total_time_ms`

## Utility Scripts

### View Profile Summary

```bash
# Display a summary of all profiles in the profiles directory
python3 scripts/summarize_profiles.py

# View a specific profile
python3 scripts/summarize_profiles.py benchmarking/profiles/deutsch_n2_sv_20260622_085831.json
```

### Compare Profiles

```bash
# Compare profiles between different simulators or circuit sizes
python3 scripts/compare_profiles.py benchmarking/profiles/deutsch_n2_sv_*.json
```

## Implementation Details

### Rust Side (src/profiling.rs)

The profiling data structure is defined in Rust:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShotsProfile {
    pub num_shots: usize,
    pub num_qubits: usize,
    pub num_instructions: usize,
    pub preprocessing_ms: f64,
    pub gate_fusion_ms: f64,
    pub parallel_execution_ms: f64,
    pub per_shot_stats: Option<Vec<ShotStats>>,
    pub total_time_ms: f64,
}
```

### Python Side (benchmarking/benchmarking_suite.py)

The benchmarking suite collects and saves profiling data:

1. Calls `simulate_shots(..., collect_profile=True)`
2. Extracts the profile from the result dictionary
3. Saves to JSON file with timestamp in `benchmarking/profiles/`

## Future Enhancements

Potential improvements to the profiling system:

1. **Per-shot statistics**: Track individual shot execution times
2. **Per-instruction timing**: Break down gate execution by gate type
3. **Memory profiling**: Track peak memory usage
4. **Custom profiling regions**: Allow user-defined profiling checkpoints
5. **Comparative analysis**: Built-in benchmarking comparison tools
6. **Export formats**: Support for CSV, HDF5, and other formats

## Performance Tips

Based on profiling results, here are some optimization strategies:

### For Preprocessing-bound Circuits
- Reuse the same simulator instance for multiple runs
- Batch circuit analysis before simulation

### For Parallel-Execution-bound Circuits
- Increase shot count to amortize preprocessing overhead
- Consider MPS for circuits where gate fusion doesn't help
- Use PBlock for distributed simulation

### General Optimization
- Enable gate fusion (automatic, always on)
- Use appropriate seed values for reproducibility
- Profile different circuit structures to identify bottlenecks

## Troubleshooting

### Profile data not being collected

Ensure you're calling `simulate_shots` with `collect_profile=True`:

```python
# ✓ Correct
result = sim.simulate_shots(circuit, shots=1000, collect_profile=True)

# ✗ Wrong
result = sim.simulate_shots(circuit, shots=1000)  # collect_profile defaults to False
```

### Profiling overhead

The profiling itself adds minimal overhead (< 1% in typical cases). If you notice significant slowdowns, verify that you're not collecting per-shot statistics unnecessarily.

### Missing profile directory

The `benchmarking/profiles/` directory is automatically created when you run the benchmarking suite. If it doesn't exist, create it manually:

```bash
mkdir -p benchmarking/profiles/
```

## References

- [Rust Profiling Module](../src/profiling.rs)
- [Statevector Simulator](../src/monolithic/statevector/simulator.rs)
- [Benchmarking Suite](./benchmarking_suite.py)
- [Test Profiling Script](../test_profiling.py)
