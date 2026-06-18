# `simulate_shots` profiler

## Context

`StatevectorSimulator.simulate_shots` (`src/monolithic/statevector/simulator.rs`) runs a
circuit `shots` times in parallel (via Rayon) and returns a bitstring-count dict. The
existing `SimulationProfile` only instruments the single-shot `simulate()` path
(`apply_one_qubit`/`apply_n_qubit`/`measure_qubit` call counts and time). `simulate_shots`
currently has no profiling at all.

`simulate_shots` has three phases:
1. **Preprocessing** — deserialize the circuit JSON, build the initial statevector.
2. **Gate fusion** — `fuse_circuit(...)`, run once before the parallel loop.
3. **Parallel shot execution** — `shots` independent runs, each looping over fused
   instructions and calling `apply_one_qubit_seq` / `apply_n_qubit_seq` /
   `measure_qubit_seq`.

This is run via `make perf` → `benchmarking/benchmarking_suite.py`, which times whole
`simulate_shots` calls from Python but has no visibility into the breakdown above.

## Goal

Add opt-in profiling to `StatevectorSimulator.simulate_shots` that captures elapsed time
for each of the three phases above, plus per-shot wall-clock latency, and writes it to a
local JSON file for later inspection. No change to behavior or return value when
profiling is off.

## Scope

`StatevectorSimulator.simulate_shots` only. `MpsSimulator` and `PBlockSimulator` have
similarly-named `simulate_shots` methods but are out of scope for this change.

## API

```rust
#[pyo3(signature = (circuit, shots=1000, profile=false))]
pub fn simulate_shots(
    &self,
    py: Python,
    circuit: &Bound<PyAny>,
    shots: usize,
    profile: bool,
) -> PyResult<PyObject>
```

- `profile=False` (default): unchanged behavior — returns the counts dict, no file
  written, negligible added overhead (a few extra `Instant::now()` calls).
- `profile=True`: same return value (counts dict). Additionally, on success, writes a
  JSON profile file to `./dqsim_profiles/shots_profile_<unix_timestamp_ns>.json`
  (directory created if missing) before returning.

No new Python-facing return type, no `profile_path` argument — the directory and
filename pattern are fixed.

## Timing instrumentation

All durations measured with `std::time::Instant`, recorded in seconds (consistent with
the existing `SimulationProfile` convention):

| Field | Measures |
|---|---|
| `preprocessing_time` | JSON deserialize (`model_dump_json` call + `serde_json::from_str`) + initial statevector construction |
| `fusion_time` | `fuse_circuit(&rust_circuit.instructions)` |
| `shots_total_time` | Entire `py.allow_threads` parallel block (wall clock, includes Rayon scheduling) |
| `total_time` | Whole `simulate_shots` function body |
| `shot_times` | `Vec<f64>`, one entry per shot, wall-clock elapsed for that shot's full instruction loop (measured inside the parallel closure, indexed by shot number) |
| `num_shots` | `shots` (for convenience when reading the JSON standalone) |

### Per-shot timing and ordering

`(0..shots).into_par_iter()` is an `IndexedParallelIterator`, so collecting its `.map()`
output into a `Vec` preserves shot order regardless of which worker thread executed which
shot. The per-shot closure changes from returning just `Result<String, String>` to
`Result<(String, f64), String>` (bitstring, elapsed seconds), measured with
`Instant::now()`/`.elapsed()` around the existing per-shot fused-instruction loop.

This replaces the current `try_fold`/`try_reduce` streaming reduction with a
`.collect::<Result<Vec<_>, _>>()` followed by a plain-Rust pass that builds the counts
`HashMap` and the `shot_times: Vec<f64>` from the collected `(String, f64)` pairs. For
typical shot counts (~1000) the extra `Vec` is a few tens of KB — not a meaningful
regression — and this only happens when `profile=true`; the existing fold/reduce path
is kept for `profile=false` so the non-profiling case is untouched.

## JSON shape

```json
{
  "preprocessing_time": 0.000123,
  "fusion_time": 0.000045,
  "shots_total_time": 0.041,
  "total_time": 0.0412,
  "num_shots": 1000,
  "shot_times": [0.000041, 0.000039, ...]
}
```

## Implementation notes

- Add an internal `ShotsProfile` struct (`#[derive(Serialize)]`, not a `pyclass` — it's
  only ever serialized to disk, never handed back into Python) in
  `src/monolithic/statevector/simulator.rs`, alongside the existing `ProfileAcc` /
  `SimulationProfile`.
- Directory creation: `std::fs::create_dir_all("dqsim_profiles")`.
- Filename timestamp: `SystemTime::now().duration_since(UNIX_EPOCH)` → nanoseconds.
- Serialization: `serde_json::to_string_pretty(&profile)` + `std::fs::write(path, ..)`.
- File I/O happens after `py.allow_threads` returns (back on the GIL-holding thread), so
  no need to worry about GIL/thread-safety for the write itself.
- Errors creating the directory or writing the file map to `PyRuntimeError`, consistent
  with existing error handling in this function (e.g. the `allow_threads` error mapping
  a few lines above).

## Testing

- New Rust/pytest test: call `simulate_shots(circuit, shots=N, profile=True)` against a
  small fixed circuit, assert a file matching `dqsim_profiles/shots_profile_*.json`
  exists, parse it, and assert: `num_shots == N`, `len(shot_times) == N`, all timing
  fields are present and `>= 0`, and `shots_total_time <= total_time` (sanity check on
  ordering of the phases).
- Confirm `profile=False` (default) leaves existing tests/benchmarking suite behavior
  unchanged (no new file written, same return value).

## Out of scope

- `MpsSimulator.simulate_shots` / `PBlockSimulator.simulate_shots` instrumentation.
- Wiring this into `benchmarking/benchmarking_suite.py` (the suite continues to time
  whole calls from Python as it does today; turning on `profile=True` there is a
  follow-up, not part of this change).
- Configurable output path/directory.
- Per-shot gate-call-count breakdown (only elapsed time is captured per shot).
