# DQSim Profiler Quick Start Guide

## 5-Minute Setup

### 1. Enable Profiling in Your Code

```python
from dqsim import StatevectorSimulator

# Create simulator
sim = StatevectorSimulator(seed=42)

# Run with profiling enabled
result = sim.simulate_shots(
    circuit, 
    shots=1000,
    collect_profile=True  # ← Enable profiling
)

# Extract timing data
profile = result["profile"]
print(f"Total time: {profile['total_time_ms']:.2f} ms")
print(f"Per shot: {profile['total_time_ms'] / profile['num_shots']:.4f} ms")
```

### 2. View Profiling Data from Benchmarks

```bash
# Run benchmarks (profiles saved automatically)
make perf

# View summary of all profiles
python3 scripts/summarize_profiles.py

# View specific profile details
python3 scripts/summarize_profiles.py benchmarking/profiles/deutsch_n2_sv_*.json
```

### 3. Export Data for Analysis

```bash
# Export to CSV (for Excel, Python, etc.)
python3 scripts/export_profiles.py --csv

# Export to HTML (for browser viewing)
python3 scripts/export_profiles.py --html

# Files saved to: benchmarking/profiles/
```

## Understanding Profile Data

Each profile contains:

| Field | Meaning | Example |
|-------|---------|---------|
| `num_shots` | Number of independent circuit runs | 1000 |
| `num_qubits` | Qubits in the circuit | 4 |
| `num_instructions` | Fused gate instructions | 14 |
| `preprocessing_ms` | Circuit parsing time | 0.22 ms |
| `gate_fusion_ms` | Gate optimization time | 0.02 ms |
| `parallel_execution_ms` | Simulation time | 6.27 ms |
| `total_time_ms` | Total elapsed time | 6.52 ms |

### Time Breakdown

```
Total = Preprocessing + Gate Fusion + Parallel Execution
 6.52 =      0.22    +     0.02    +       6.27
```

Typically:
- **Preprocessing**: 1-3% (fixed cost)
- **Gate Fusion**: 0.1-0.3% (optimization)
- **Parallel Execution**: 96-99% (actual simulation)

## Common Tasks

### Q: How do I profile just my circuit?

```python
sim = StatevectorSimulator()
result = sim.simulate_shots(my_circuit, shots=100, collect_profile=True)
print(result["profile"])
```

### Q: How do I compare two circuits?

```bash
# Run benchmarks for both
python3 scripts/summarize_profiles.py --compare benchmarking/profiles/
```

### Q: How do I track performance over time?

The profiling system automatically timestamps each profile:
```
deutsch_n2_sv_20260622_085831.json
              ↑ timestamp
```

Use the CSV export to track changes:
```bash
python3 scripts/export_profiles.py --csv profiles_$(date +%Y%m%d).csv
```

### Q: Can I disable profiling overhead?

Yes! Just don't pass `collect_profile=True`:
```python
# No profiling overhead (default)
result = sim.simulate_shots(circuit, shots=1000)
```

## Sample Output

### Console View
```
Profile: deutsch_n2_sv_20260622_085831.json
==========================================

Circuit Info:
  Qubits:           4
  Instructions:     14 (fused)
  Shots:            1000

Timing (ms):
  Preprocessing:        0.2179  (  3.3%)
  Gate Fusion:          0.0203  (  0.3%)
  Parallel Exec:        6.2666  ( 96.1%)
  ────────────────────────────────────────
  Total:                6.5224  (100.0%)

Per-Shot Analysis:
  Time per shot:        0.0065 ms
  Per instr/shot:       0.4476 µs
```

### CSV Export
```
Filename,Qubits,Instructions,Shots,Preprocessing (ms),...
deutsch_n2_sv_20260622_085831.json,4,14,1000,0.2179,...
adder_n4_sv_20260622_085832.json,6,51,1000,0.3064,...
```

## Troubleshooting

### No profiles being generated?

1. Ensure you're using `collect_profile=True`
2. Check that `benchmarking/profiles/` directory exists
3. Verify the simulation completes without errors

### Profiles seem slow?

The profiling overhead is minimal (<1%). If times seem high:
- Check if other processes are running
- Try running with fewer shots first
- Use larger shot counts to amortize preprocessing

### Can't import scripts?

Make sure you're in the dqsim_bench directory:
```bash
cd /path/to/dqsim_bench
python3 scripts/summarize_profiles.py
```

## Next Steps

- Read [PROFILING.md](PROFILING.md) for comprehensive documentation
- See [PROFILER_IMPLEMENTATION.md](PROFILER_IMPLEMENTATION.md) for technical details
- Run `test_profiling.py` to verify the system
- Check out the analysis tools: `scripts/summarize_profiles.py` and `scripts/export_profiles.py`

## Tips for Best Results

1. **Use consistent shot counts** for fair comparison
2. **Multiple runs** average out system variability
3. **Export to CSV** for statistical analysis
4. **Track over time** to catch performance regressions
5. **Profile different circuit types** to understand trade-offs

Happy profiling! 🎯
