# `simulate_shots` Profiler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add opt-in stage + per-shot timing instrumentation to `StatevectorSimulator.simulate_shots`, dumped as a local JSON file.

**Architecture:** A new `profile: bool` parameter on the existing PyO3 method `simulate_shots`. When `true`, the method times preprocessing, gate fusion, and the parallel shot loop (plus per-shot wall-clock latency captured inside each parallel closure), serializes the result with `serde_json`, and writes it to a timestamped file under `./dqsim_profiles/`. When `false` (default), behavior and return value are unchanged.

**Tech Stack:** Rust (PyO3 0.22, Rayon, serde/serde_json — all already dependencies), pytest (existing test suite), `bosonic_model` (already a core dependency, used to build a small test circuit).

## Global Constraints

- Scope is `StatevectorSimulator.simulate_shots` only (`src/monolithic/statevector/simulator.rs`) — do not touch `MpsSimulator` or `PBlockSimulator`.
- `profile=False` (default) must leave the existing return value (a `dict[str, int]` of bitstring counts) and behavior completely unchanged — no file written, no new error paths exercised.
- `profile=True` still returns the same counts dict — no change to the Python-facing return type.
- Output file path is fixed: `./dqsim_profiles/shots_profile_<unix_timestamp_ns>.json`, directory created if missing. No `profile_path` argument.
- All durations are in seconds (`f64`), matching the existing `SimulationProfile` convention in the same file.
- Per-shot timing must be ordered by shot index (`shot_times[i]` corresponds to shot `i`), relying on `(0..shots).into_par_iter()` being an `IndexedParallelIterator`.
- JSON fields, exactly: `preprocessing_time`, `fusion_time`, `shots_total_time`, `total_time`, `num_shots`, `shot_times` (array).

---

### Task 1: Add profiling instrumentation + JSON dump to `simulate_shots`

**Files:**
- Modify: `src/monolithic/statevector/simulator.rs` (imports near line 1-16, new struct/helper near the existing `ProfileAcc`/`SimulationProfile` definitions around line 30-72, and the `simulate_shots` method body at lines 199-246)
- Test: `tests/test_shots_profile.py` (new file)

**Interfaces:**
- Consumes: existing `fuse_circuit`, `run_fused_par`, `format_cbits` (all already in scope in this file); `Circuit` (from `crate::types`).
- Produces: `StatevectorSimulator.simulate_shots(circuit, shots=1000, profile=False)` — Python-callable signature with the new `profile` kwarg. Internal `ShotsProfile` struct and `write_shots_profile` function are private to this file; no other task depends on their internals.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_shots_profile.py`:

```python
"""Tests for StatevectorSimulator.simulate_shots profiling (profile=True)."""

from __future__ import annotations

import json

from bosonic_model import Circuit, CxInstruction, HInstruction, Register

from dqsim import StatevectorSimulator


def _bell_circuit() -> Circuit:
    return Circuit(
        qregs={"q": Register(name="q", size=2, base=0)},
        instructions=[
            HInstruction(qubit=0),
            CxInstruction(control=0, target=1),
        ],
    )


class TestShotsProfile:
    def test_profile_false_does_not_write_file(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        sim = StatevectorSimulator(seed=42)

        counts = sim.simulate_shots(_bell_circuit(), shots=50)

        assert isinstance(counts, dict)
        assert sum(counts.values()) == 50
        assert not (tmp_path / "dqsim_profiles").exists()

    def test_profile_true_writes_json_file(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        sim = StatevectorSimulator(seed=42)
        shots = 64

        counts = sim.simulate_shots(_bell_circuit(), shots=shots, profile=True)

        assert isinstance(counts, dict)
        assert sum(counts.values()) == shots

        profile_dir = tmp_path / "dqsim_profiles"
        assert profile_dir.is_dir()
        files = list(profile_dir.glob("shots_profile_*.json"))
        assert len(files) == 1

        data = json.loads(files[0].read_text())
        assert data["num_shots"] == shots
        assert len(data["shot_times"]) == shots
        assert all(isinstance(t, float) and t >= 0.0 for t in data["shot_times"])
        for key in ("preprocessing_time", "fusion_time", "shots_total_time", "total_time"):
            assert isinstance(data[key], float)
            assert data[key] >= 0.0
        assert data["shots_total_time"] <= data["total_time"] + 1e-6
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --python 3.11 --extra test pytest tests/test_shots_profile.py -v`

Expected: `test_profile_false_does_not_write_file` PASSES (no behavior change yet), `test_profile_true_writes_json_file` FAILS with `TypeError: simulate_shots() got an unexpected keyword argument 'profile'` (the `profile` kwarg does not exist on the currently-built extension).

- [ ] **Step 3: Update imports in `src/monolithic/statevector/simulator.rs`**

Replace the top of the file (lines 1-14):

```rust
use std::collections::HashMap;
use std::time::Instant;

use num_complex::Complex64;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

use crate::engine::{apply_n_qubit, apply_one_qubit, apply_n_qubit_seq, apply_one_qubit_seq, measure_qubit, measure_qubit_seq, marginal_probs, sample_counts};
use crate::gates;
use crate::types::{Circuit, FusedInstruction, Instruction, format_cbits, fuse_circuit};
```

with:

```rust
use std::collections::HashMap;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use num_complex::Complex64;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::Serialize;

use crate::engine::{apply_n_qubit, apply_one_qubit, apply_n_qubit_seq, apply_one_qubit_seq, measure_qubit, measure_qubit_seq, marginal_probs, sample_counts};
use crate::gates;
use crate::types::{Circuit, FusedInstruction, Instruction, format_cbits, fuse_circuit};
```

- [ ] **Step 4: Add the `ShotsProfile` struct and `write_shots_profile` helper**

Insert this new section directly after the existing `ProfileAcc` struct definition (after line 30, i.e. right before the `// SimulationProfile` section comment that starts at line 32):

```rust
// ---------------------------------------------------------------------------
// ShotsProfile (simulate_shots instrumentation, dumped to JSON on disk)
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct ShotsProfile {
    preprocessing_time: f64,
    fusion_time: f64,
    shots_total_time: f64,
    total_time: f64,
    num_shots: usize,
    shot_times: Vec<f64>,
}

fn write_shots_profile(profile: &ShotsProfile) -> std::io::Result<()> {
    let dir = std::path::Path::new("dqsim_profiles");
    std::fs::create_dir_all(dir)?;
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let path = dir.join(format!("shots_profile_{ts}.json"));
    let json = serde_json::to_string_pretty(profile)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    std::fs::write(path, json)
}
```

- [ ] **Step 5: Replace the `simulate_shots` method body**

Find this method on `StatevectorSimulator` (currently lines 199-246):

```rust
    #[pyo3(signature = (circuit, shots=1000))]
    pub fn simulate_shots(&self, py: Python, circuit: &Bound<PyAny>, shots: usize) -> PyResult<PyObject> {
        let json_str: String = circuit.call_method0("model_dump_json")?.extract()?;
        let rust_circuit: Circuit = serde_json::from_str(&json_str).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Circuit JSON parse error: {e}"))
        })?;

        let n = rust_circuit.num_qubits();
        let num_cbits = rust_circuit.num_cbits();
        let base_seed = self.seed.unwrap_or_else(|| rand::thread_rng().gen());

        let initial_state = {
            let mut s = vec![C::new(0.0, 0.0); 1 << n];
            s[0] = C::new(1.0, 0.0);
            s
        };

        // Fuse consecutive single-qubit gates once, before the parallel shot loop.
        let fused = fuse_circuit(&rust_circuit.instructions);

        let counts = py.allow_threads(|| -> Result<HashMap<String, usize>, String> {
            (0..shots)
                .into_par_iter()
                .map(|i| -> Result<String, String> {
                    let mut state = initial_state.clone();
                    let mut cbits: HashMap<usize, i32> = HashMap::new();
                    let mut rng = ChaCha8Rng::seed_from_u64(base_seed.wrapping_add(i as u64));
                    for fi in &fused {
                        run_fused_par(&mut state, fi, n, &mut cbits, &mut rng)?;
                    }
                    Ok(format_cbits(&cbits, num_cbits))
                })
                .try_fold(
                    HashMap::new,
                    |mut m, r| { let k = r?; *m.entry(k).or_insert(0) += 1; Ok(m) },
                )
                .try_reduce(
                    HashMap::new,
                    |mut a, b| { for (k, v) in b { *a.entry(k).or_insert(0) += v; } Ok(a) },
                )
        }).map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

        let d = PyDict::new_bound(py);
        for (k, v) in &counts {
            d.set_item(k, v)?;
        }
        Ok(d.into())
    }
```

Replace it with:

```rust
    #[pyo3(signature = (circuit, shots=1000, profile=false))]
    pub fn simulate_shots(
        &self,
        py: Python,
        circuit: &Bound<PyAny>,
        shots: usize,
        profile: bool,
    ) -> PyResult<PyObject> {
        let total_t0 = Instant::now();

        let preprocessing_t0 = Instant::now();
        let json_str: String = circuit.call_method0("model_dump_json")?.extract()?;
        let rust_circuit: Circuit = serde_json::from_str(&json_str).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Circuit JSON parse error: {e}"))
        })?;

        let n = rust_circuit.num_qubits();
        let num_cbits = rust_circuit.num_cbits();
        let base_seed = self.seed.unwrap_or_else(|| rand::thread_rng().gen());

        let initial_state = {
            let mut s = vec![C::new(0.0, 0.0); 1 << n];
            s[0] = C::new(1.0, 0.0);
            s
        };
        let preprocessing_time = preprocessing_t0.elapsed().as_secs_f64();

        // Fuse consecutive single-qubit gates once, before the parallel shot loop.
        let fusion_t0 = Instant::now();
        let fused = fuse_circuit(&rust_circuit.instructions);
        let fusion_time = fusion_t0.elapsed().as_secs_f64();

        let shots_t0 = Instant::now();
        let (counts, shot_times): (HashMap<String, usize>, Vec<f64>) = if profile {
            let results = py.allow_threads(|| -> Result<Vec<(String, f64)>, String> {
                (0..shots)
                    .into_par_iter()
                    .map(|i| -> Result<(String, f64), String> {
                        let shot_t0 = Instant::now();
                        let mut state = initial_state.clone();
                        let mut cbits: HashMap<usize, i32> = HashMap::new();
                        let mut rng = ChaCha8Rng::seed_from_u64(base_seed.wrapping_add(i as u64));
                        for fi in &fused {
                            run_fused_par(&mut state, fi, n, &mut cbits, &mut rng)?;
                        }
                        let bits = format_cbits(&cbits, num_cbits);
                        Ok((bits, shot_t0.elapsed().as_secs_f64()))
                    })
                    .collect()
            })
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

            let mut counts: HashMap<String, usize> = HashMap::new();
            let mut shot_times: Vec<f64> = Vec::with_capacity(results.len());
            for (bits, t) in results {
                *counts.entry(bits).or_insert(0) += 1;
                shot_times.push(t);
            }
            (counts, shot_times)
        } else {
            let counts = py.allow_threads(|| -> Result<HashMap<String, usize>, String> {
                (0..shots)
                    .into_par_iter()
                    .map(|i| -> Result<String, String> {
                        let mut state = initial_state.clone();
                        let mut cbits: HashMap<usize, i32> = HashMap::new();
                        let mut rng = ChaCha8Rng::seed_from_u64(base_seed.wrapping_add(i as u64));
                        for fi in &fused {
                            run_fused_par(&mut state, fi, n, &mut cbits, &mut rng)?;
                        }
                        Ok(format_cbits(&cbits, num_cbits))
                    })
                    .try_fold(
                        HashMap::new,
                        |mut m, r| { let k = r?; *m.entry(k).or_insert(0) += 1; Ok(m) },
                    )
                    .try_reduce(
                        HashMap::new,
                        |mut a, b| { for (k, v) in b { *a.entry(k).or_insert(0) += v; } Ok(a) },
                    )
            })
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
            (counts, Vec::new())
        };
        let shots_total_time = shots_t0.elapsed().as_secs_f64();

        let d = PyDict::new_bound(py);
        for (k, v) in &counts {
            d.set_item(k, v)?;
        }

        if profile {
            let total_time = total_t0.elapsed().as_secs_f64();
            let shots_profile = ShotsProfile {
                preprocessing_time,
                fusion_time,
                shots_total_time,
                total_time,
                num_shots: shots,
                shot_times,
            };
            write_shots_profile(&shots_profile).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to write shots profile: {e}"
                ))
            })?;
        }

        Ok(d.into())
    }
```

- [ ] **Step 6: Build the extension**

Run: `uvx --python 3.11 maturin develop --skip-install`

Expected: builds cleanly, no `cargo` errors. If there are borrow-checker errors about `initial_state`/`fused`/`base_seed`/`num_cbits` being moved into both the `if` and `else` branch closures, confirm both closures only *borrow* these (via `&fused`, `initial_state.clone()`, etc.) — they should, since the new code mirrors the original borrowing pattern in each branch.

- [ ] **Step 7: Run the tests to verify they pass**

Run: `uv run --python 3.11 --extra test pytest tests/test_shots_profile.py -v`

Expected: both tests PASS.

- [ ] **Step 8: Run the full existing test suite to confirm no regressions**

Run: `uv run --python 3.11 --extra test pytest tests/ -v`

Expected: all tests pass (same as before this change — this change is additive and the `profile=False` path is byte-for-byte the original logic).

- [ ] **Step 9: Commit**

```bash
git add src/monolithic/statevector/simulator.rs tests/test_shots_profile.py
git commit -m "$(cat <<'EOF'
Add opt-in profiling to simulate_shots, dumped as JSON

simulate_shots(profile=True) now times preprocessing, gate fusion, and
the parallel shot loop, plus per-shot wall-clock latency, and writes
the result to ./dqsim_profiles/shots_profile_<ts>.json. Default
behavior (profile=False) is unchanged.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review Notes

- **Spec coverage:** preprocessing/fusion/shots_total/total timings ✓ (Step 5), per-shot array ordered by index ✓ (indexed `0..shots` + `.collect()` into `Vec`), JSON dump to `./dqsim_profiles/shots_profile_<ts>.json` ✓ (Step 4), `profile=False` default unchanged ✓ (else-branch is the original code verbatim, no file write), scope limited to `StatevectorSimulator` ✓, no `profile_path` arg ✓, seconds units ✓, exact JSON field names ✓.
- **Placeholder scan:** none — every step has complete code or an exact command with expected output.
- **Type consistency:** `ShotsProfile` fields match the JSON shape in the spec exactly; `simulate_shots` signature (`circuit, shots=1000, profile=false`) matches the spec's API section; test assertions read the same field names the Rust struct serializes (serde's default field-name behavior, no `#[serde(rename)]` needed since Rust field names already match the desired JSON keys).
