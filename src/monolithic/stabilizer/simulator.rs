use std::collections::HashMap;
use std::time::Instant;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::{SeedableRng, Rng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::Serialize;

use crate::profiling::write_shots_profile;
use crate::types::{Circuit, Instruction, format_cbits};

// ---------------------------------------------------------------------------
// ShotsProfile (simulate_shots instrumentation, dumped to JSON on disk)
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct ShotsProfile {
    preprocessing_time: f64,
    shots_total_time: f64,
    total_time: f64,
    num_shots: usize,
    shot_times: Vec<f64>,
    /// Total count/time of measure_qubit calls across all shots — the only
    /// non-constant-time operation in the Clifford tableau update rules.
    measure_calls: u64,
    measure_time: f64,
}

// ---------------------------------------------------------------------------
// StabilizerState
// Represents a single active instance of a Clifford binary check matrix.
// ---------------------------------------------------------------------------
struct StabilizerState {
    num_qubits: usize,
    tableau: Vec<Vec<bool>>,
}

impl StabilizerState {
    /// Initialises a clean tableau corresponding to the |00...0> ground state.
    pub fn new(num_qubits: usize) -> Self {
        let mut tableau = vec![vec![false; 2 * num_qubits + 1]; 2 * num_qubits];
        for i in 0..num_qubits {
            tableau[i][i] = true;                           // Destabilizer X bits
            tableau[i + num_qubits][i + num_qubits] = true; // Stabilizer Z bits
        }
        Self { num_qubits, tableau }
    }

    /// Applies a Hadamard gate to the specified qubit.
    pub fn apply_h(&mut self, qubit: usize) {
        let n = self.num_qubits;
        for r in 0..(2 * n) {
            self.tableau[r].swap(qubit, qubit + n);
            self.tableau[r][2 * n] ^= self.tableau[r][qubit] && self.tableau[r][qubit + n];
        }
    }

    /// Applies a Phase gate (S) to the specified qubit.
    pub fn apply_s(&mut self, qubit: usize) {
        let n = self.num_qubits;
        for r in 0..(2 * n) {
            self.tableau[r][2 * n] ^= self.tableau[r][qubit] && self.tableau[r][qubit + n];
            self.tableau[r][qubit + n] ^= self.tableau[r][qubit];
        }
    }

    /// Applies a Controlled-NOT (CNOT) gate between control and target qubits.
    pub fn apply_cnot(&mut self, control: usize, target: usize) {
        let n = self.num_qubits;
        for r in 0..(2 * n) {
            self.tableau[r][2 * n] ^= self.tableau[r][control]
                && self.tableau[r][target + n]
                && (self.tableau[r][target] ^ self.tableau[r][control + n] ^ true);
            
            self.tableau[r][target] ^= self.tableau[r][control];
            self.tableau[r][control + n] ^= self.tableau[r][target + n];
        }
    }

    /// Measures the specified qubit and updates the stabilizer state.
    pub fn measure_qubit<R: Rng>(&mut self, qubit: usize, rng: &mut R) -> u8 {
        let n = self.num_qubits;
        let phase_col = 2 * n;
        
        // Find if any stabilizer generator anti-commutes with the measurement operator
        let mut anti_commute_row = None;
        for r in n..(2 * n) {
            if self.tableau[r][qubit] {
                anti_commute_row = Some(r);
                break;
            }
        }

        match anti_commute_row {
            // Case 1: Outcome is random (50/50 probability)
            Some(p_prime) => {
                let outcome = if rng.gen() { 1u8 } else { 0u8 };

                for r in 0..(2 * n) {
                    if r != p_prime && self.tableau[r][qubit] {
                        self.row_add(p_prime, r);
                    }
                }

                self.tableau[p_prime - n] = self.tableau[p_prime].clone();

                for c in 0..phase_col {
                    self.tableau[p_prime][c] = false;
                }
                self.tableau[p_prime][qubit + n] = true;
                self.tableau[p_prime][phase_col] = outcome == 1;

                outcome
            }
            // Case 2: Outcome is deterministic and can be derived from current stabilizers
            None => {
                let mut scratch = vec![false; 2 * n + 1];
                for r in 0..n {
                    if self.tableau[r][qubit] {
                        let src = r + n;
                        let mut phase_exponent = 0i32;
                        for i in 0..n {
                            let x_s = self.tableau[src][i];
                            let z_s = self.tableau[src][i + n];
                            let x_t = scratch[i];
                            let z_t = scratch[i + n];
                            phase_exponent += self.g_phase(x_s, z_s, x_t, z_t);
                        }
                        if self.tableau[src][phase_col] { phase_exponent += 2; }
                        if scratch[phase_col] { phase_exponent += 2; }
                        phase_exponent = phase_exponent.rem_euclid(4);
                        scratch[phase_col] = phase_exponent == 2;
                        for i in 0..(2 * n) {
                            scratch[i] ^= self.tableau[src][i];
                        }
                    }
                }
                if scratch[phase_col] { 1u8 } else { 0u8 }
            }
        }
    }

    /// Adds a row to another row in the check matrix according to Clifford accumulation rules.
    fn row_add(&mut self, source: usize, target: usize) {
        let n = self.num_qubits;
        let phase_col = 2 * n;
        let mut phase_exponent = 0i32;

        for i in 0..n {
            let x_s = self.tableau[source][i];
            let z_s = self.tableau[source][i + n];
            let x_t = self.tableau[target][i];
            let z_t = self.tableau[target][i + n];
            phase_exponent += self.g_phase(x_s, z_s, x_t, z_t);
        }

        if self.tableau[source][phase_col] { phase_exponent += 2; }
        if self.tableau[target][phase_col] { phase_exponent += 2; }

        phase_exponent = phase_exponent.rem_euclid(4);
        self.tableau[target][phase_col] = phase_exponent == 2;

        for i in 0..(2 * n) {
            self.tableau[target][i] ^= self.tableau[source][i];
        }
    }

    /// Computes the phase exponent (0, 1, 2, or 3 representing 1, i, -1, -i) when multiplying 
    /// two Pauli operators. Mapping: (false, false) = I, (true, false) = X, (true, true) = Y, (false, true) = Z.
    fn g_phase(&self, x1: bool, z1: bool, x2: bool, z2: bool) -> i32 {
        // If either operator is the identity (I), multiplying does not alter the phase (returns 0)
        if !x1 && !z1 { return 0; }
        if !x2 && !z2 { return 0; }
        if x1 && !z1 {
            // First operator is X
            // X * Y = iZ -> introduces a factor of i, phase exponent increases by 1
            if x2 && z2 { return 1; }
            // X * Z = -iY -> equivalent to i^1 * (-Y) in this convention, phase exponent increases by 1
            if !x2 && z2 { return 1; }
        } else if x1 && z1 {
            // First operator is Y
            // Y * Z = iX -> introduces a factor of i, phase exponent increases by 1
            if !x2 && z2 { return 1; }
            // Y * X = -iZ -> introduces a factor of -i, phase exponent decreases by 1
            if x2 && !z2 { return -1; }
        } else if !x1 && z1 {
            // First operator is Z
            // Z * X = iY -> introduces a factor of i, phase exponent increases by 1
            if x2 && !z2 { return 1; }
            // Z * Y = -iX -> introduces a factor of -i, phase exponent decreases by 1
            if x2 && z2 { return -1; }
        }
        0
    }
}

// ---------------------------------------------------------------------------
// StabilizerSimulator
// ---------------------------------------------------------------------------
#[pyclass]
pub struct StabilizerSimulator {
    seed: Option<u64>,
}

#[pymethods]
impl StabilizerSimulator {
    #[new]
    #[pyo3(signature = (seed=None))]
    pub fn new(seed: Option<u64>) -> Self {
        Self { seed }
    }

    /// Run the circuit `shots` times independently, utilizing the Clifford stabilizer formalism.
    /// Returns a dict[str, int] containing classical register bitstring count metrics.
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
        // 1. Serialize entire circuit to JSON (one boundary crossing)
        let json_str: String = circuit.call_method0("model_dump_json")?.extract()?;
        // 2. Deserialize in Rust — no more Python calls until we return
        let rust_circuit: Circuit = serde_json::from_str(&json_str).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Circuit JSON parse error: {e}"))
        })?;

        let n = rust_circuit.num_qubits();
        let num_cbits = rust_circuit.num_cbits();
        let base_seed = self.seed.unwrap_or_else(|| rand::thread_rng().gen());
        let preprocessing_time = preprocessing_t0.elapsed().as_secs_f64();

        let shots_t0 = Instant::now();
        let (counts, shot_times, measure_calls, measure_time) = if profile {
            let results = py.allow_threads(|| -> Result<Vec<(String, f64, u64, f64)>, String> {
                (0..shots)
                    .into_par_iter()
                    .map(|i| -> Result<(String, f64, u64, f64), String> {
                        let shot_t0 = Instant::now();
                        let mut sim = StabilizerState::new(n);
                        let mut cbits: HashMap<usize, i32> = HashMap::new();
                        let mut rng = ChaCha8Rng::seed_from_u64(base_seed.wrapping_add(i as u64));
                        let (mc, mt) = run_shot_instructions(
                            &mut sim, &rust_circuit.instructions, &mut cbits, &mut rng, true,
                        )?;
                        let bits = format_cbits(&cbits, num_cbits);
                        Ok((bits, shot_t0.elapsed().as_secs_f64(), mc, mt))
                    })
                    .collect()
            }).map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

            let mut counts: HashMap<String, usize> = HashMap::new();
            let mut shot_times: Vec<f64> = Vec::with_capacity(results.len());
            let mut measure_calls = 0u64;
            let mut measure_time = 0.0f64;
            for (bits, t, mc, mt) in results {
                *counts.entry(bits).or_insert(0) += 1;
                shot_times.push(t);
                measure_calls += mc;
                measure_time += mt;
            }
            (counts, shot_times, measure_calls, measure_time)
        } else {
            // Execute all independent shot iterations in parallel using Rayon threads
            let counts = py.allow_threads(|| -> Result<HashMap<String, usize>, String> {
                (0..shots)
                    .into_par_iter()
                    .map(|i| -> Result<String, String> {
                        let mut sim = StabilizerState::new(n);
                        let mut cbits: HashMap<usize, i32> = HashMap::new();
                        let mut rng = ChaCha8Rng::seed_from_u64(base_seed.wrapping_add(i as u64));
                        run_shot_instructions(
                            &mut sim, &rust_circuit.instructions, &mut cbits, &mut rng, false,
                        )?;
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
            (counts, Vec::new(), 0u64, 0.0f64)
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
                shots_total_time,
                total_time,
                num_shots: shots,
                shot_times,
                measure_calls,
                measure_time,
            };
            write_shots_profile("stabilizer", &shots_profile).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to write shots profile: {e}"
                ))
            })?;
        }

        Ok(d.into())
    }
}

// ---------------------------------------------------------------------------
// Per-shot instruction dispatcher (used inside py.allow_threads for parallel shots)
// ---------------------------------------------------------------------------

fn run_shot_instructions(
    sim: &mut StabilizerState,
    instructions: &[Instruction],
    cbits: &mut HashMap<usize, i32>,
    rng: &mut impl Rng,
    track_measure: bool,
) -> Result<(u64, f64), String> {
    let mut measure_calls = 0u64;
    let mut measure_time = 0.0f64;
    for inst in instructions {
        match inst {
            // -- Single-qubit fixed ------------------------------------------
            Instruction::H { qubit } => sim.apply_h(*qubit),
            Instruction::S { qubit } => sim.apply_s(*qubit),
            Instruction::Sdg { qubit } => {
                sim.apply_s(*qubit);
                sim.apply_s(*qubit);
                sim.apply_s(*qubit);
            }
            Instruction::X { qubit } => {
                sim.apply_h(*qubit);
                sim.apply_s(*qubit);
                sim.apply_s(*qubit);
                sim.apply_h(*qubit);
            }
            Instruction::Y { qubit } => {
                sim.apply_h(*qubit);
                sim.apply_s(*qubit);
                sim.apply_s(*qubit);
                sim.apply_h(*qubit);
                sim.apply_s(*qubit);
                sim.apply_s(*qubit);
            }
            Instruction::Z { qubit } => {
                sim.apply_s(*qubit);
                sim.apply_s(*qubit);
            }
            // -- Two-qubit fixed ---------------------------------------------
            Instruction::Cx { control, target } => sim.apply_cnot(*control, *target),
            // -- Measurement -------------------------------------------------
            Instruction::Measure { qubit, cbit } => {
                let outcome = if track_measure {
                    let t0 = Instant::now();
                    let o = sim.measure_qubit(*qubit, rng);
                    measure_time += t0.elapsed().as_secs_f64();
                    measure_calls += 1;
                    o
                } else {
                    sim.measure_qubit(*qubit, rng)
                };
                cbits.insert(*cbit, outcome as i32);
            }
            // -- Reset -------------------------------------------------------
            Instruction::Reset { qubit } => {
                let outcome = if track_measure {
                    let t0 = Instant::now();
                    let o = sim.measure_qubit(*qubit, rng);
                    measure_time += t0.elapsed().as_secs_f64();
                    measure_calls += 1;
                    o
                } else {
                    sim.measure_qubit(*qubit, rng)
                };
                if outcome == 1 {
                    sim.apply_h(*qubit);
                    sim.apply_s(*qubit);
                    sim.apply_s(*qubit);
                    sim.apply_h(*qubit);
                }
            }
            // -- No-ops ------------------------------------------------------
            Instruction::Barrier | Instruction::Id { .. } => {}
            other => return Err(format!(
                "Unsupported instruction '{:?}' for Clifford stabilizer simulation mode.",
                other
            )),
        }
    }
    Ok((measure_calls, measure_time))
}
