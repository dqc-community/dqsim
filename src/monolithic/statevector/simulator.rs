use std::collections::HashMap;
use std::sync::OnceLock;
use std::time::Instant;

use num_complex::Complex64;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

use crate::engine::{
    apply_cswap_kernel, apply_cswap_kernel_seq, apply_cx_kernel, apply_cx_kernel_seq,
    apply_cz_kernel, apply_cz_kernel_seq, apply_multi_controlled_x_kernel,
    apply_multi_controlled_x_kernel_seq, apply_n_qubit, apply_n_qubit_seq, apply_one_qubit,
    apply_one_qubit_seq, apply_swap_kernel, apply_swap_kernel_seq, marginal_probs, measure_qubit,
    measure_qubit_seq, sample_counts, statevector_multi_control_kernels_enabled_for,
    statevector_simd_backend, statevector_simd_enabled, statevector_simd_min_qubits,
    statevector_two_qubit_kernel_gates, statevector_two_qubit_kernels_enabled,
};
use crate::gates;
use crate::profiling::{ShotLoopProfiler, ShotsProfile};
use crate::types::{
    format_cbits, fuse_circuit, Circuit, Condition, FusedInstruction, Instruction, Register,
};

type C = Complex64;

// ---------------------------------------------------------------------------
// Profiling accumulator (internal, not a pyclass)
// ---------------------------------------------------------------------------

#[derive(Default)]
struct ProfileAcc {
    oq_calls: u64,
    oq_time: f64,
    nq_calls: u64,
    nq_time: f64,
    mq_calls: u64,
    mq_time: f64,
}

// ---------------------------------------------------------------------------
// SimulationProfile
// ---------------------------------------------------------------------------

#[pyclass]
pub struct SimulationProfile {
    #[pyo3(get)]
    pub apply_one_qubit_calls: u64,
    #[pyo3(get)]
    pub apply_one_qubit_time: f64,
    #[pyo3(get)]
    pub apply_n_qubit_calls: u64,
    #[pyo3(get)]
    pub apply_n_qubit_time: f64,
    #[pyo3(get)]
    pub measure_qubit_calls: u64,
    #[pyo3(get)]
    pub measure_qubit_time: f64,
    #[pyo3(get)]
    pub total_time: f64,
}

#[pymethods]
impl SimulationProfile {
    fn __repr__(&self) -> String {
        let total = self.total_time.max(1e-9);
        format!(
            "SimulationProfile (total: {:.2} ms)\n  apply_one_qubit : {:4} calls  {:8.2} ms  ({:.1}%)\n  apply_n_qubit   : {:4} calls  {:8.2} ms  ({:.1}%)\n  measure_qubit   : {:4} calls  {:8.2} ms  ({:.1}%)",
            self.total_time * 1000.0,
            self.apply_one_qubit_calls,
            self.apply_one_qubit_time * 1000.0,
            100.0 * self.apply_one_qubit_time / total,
            self.apply_n_qubit_calls,
            self.apply_n_qubit_time * 1000.0,
            100.0 * self.apply_n_qubit_time / total,
            self.measure_qubit_calls,
            self.measure_qubit_time * 1000.0,
            100.0 * self.measure_qubit_time / total,
        )
    }
}

// ---------------------------------------------------------------------------
// SimulationResult
// ---------------------------------------------------------------------------

#[pyclass]
pub struct SimulationResult {
    sv: Vec<C>,
    #[pyo3(get)]
    pub num_qubits: usize,
    cbits: HashMap<usize, i32>,
    prof: Option<Py<SimulationProfile>>,
}

impl SimulationResult {
    pub(crate) fn new(
        sv: Vec<C>,
        num_qubits: usize,
        cbits: HashMap<usize, i32>,
        prof: Option<Py<SimulationProfile>>,
    ) -> Self {
        Self {
            sv,
            num_qubits,
            cbits,
            prof,
        }
    }
}

#[pymethods]
impl SimulationResult {
    /// Raw complex amplitudes as a NumPy array of shape (2^n,).
    #[getter]
    fn statevector<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<C>> {
        self.sv.clone().into_pyarray_bound(py)
    }

    /// Final classical register state. Keys are absolute cbit indices.
    #[getter]
    fn classical_bits(&self, py: Python) -> PyObject {
        let d = PyDict::new_bound(py);
        for (&k, &v) in &self.cbits {
            d.set_item(k, v).unwrap();
        }
        d.into()
    }

    /// Profiling data, or None if the simulator was not run with profile=True.
    #[getter]
    fn profile(&self, py: Python) -> PyObject {
        match &self.prof {
            None => py.None(),
            Some(p) => p.clone_ref(py).into_py(py),
        }
    }

    /// Full or marginal probability distribution. Keys are integer basis states.
    /// qubits[0] is MSB of the output index. If None, all qubits in descending order.
    #[pyo3(signature = (qubits=None))]
    fn probabilities(&self, py: Python, qubits: Option<Vec<usize>>) -> PyObject {
        let qs: Vec<usize> = qubits.unwrap_or_else(|| (0..self.num_qubits).rev().collect());
        let probs = marginal_probs(&self.sv, self.num_qubits, &qs);
        let d = PyDict::new_bound(py);
        for (j, p) in probs.iter().enumerate() {
            if *p > 0.0 {
                d.set_item(j, p).unwrap();
            }
        }
        d.into()
    }

    /// Sample the distribution. Bitstrings have qubits[0] as the leftmost (MSB) character.
    #[pyo3(signature = (shots=1000, qubits=None, seed=None))]
    fn counts(
        &self,
        py: Python,
        shots: usize,
        qubits: Option<Vec<usize>>,
        seed: Option<u64>,
    ) -> PyObject {
        let qs: Vec<usize> = qubits.unwrap_or_else(|| (0..self.num_qubits).rev().collect());
        let mut rng = match seed {
            Some(s) => ChaCha8Rng::seed_from_u64(s),
            None => ChaCha8Rng::from_entropy(),
        };
        let c = sample_counts(&self.sv, self.num_qubits, shots, &mut rng, Some(&qs));
        let d = PyDict::new_bound(py);
        for (k, v) in c {
            d.set_item(k, v).unwrap();
        }
        d.into()
    }

    /// Compute |<self|other>|^2.
    fn fidelity(&self, other: PyReadonlyArray1<C>) -> f64 {
        let arr = other.as_array();
        if arr.len() != self.sv.len() {
            return 0.0;
        }
        let dot: C = self
            .sv
            .iter()
            .zip(arr.iter())
            .map(|(a, b)| a.conj() * b)
            .sum();
        dot.norm_sqr()
    }
}

// ---------------------------------------------------------------------------
// StatevectorSimulator
// ---------------------------------------------------------------------------

#[pyclass]
pub struct StatevectorSimulator {
    seed: Option<u64>,
    profile: bool,
}

#[pymethods]
impl StatevectorSimulator {
    #[new]
    #[pyo3(signature = (seed=None, profile=false))]
    pub fn new(seed: Option<u64>, profile: bool) -> Self {
        Self { seed, profile }
    }

    /// Run the circuit `shots` times independently, collapsing state on every mid-circuit
    /// measurement per shot. Returns a dict[str, int] of bitstring counts Ã¢â‚¬â€ the true
    /// shot distribution including classically-conditioned corrections.
    ///
    /// Returns: (counts_dict, profile_dict)
    /// The profile_dict contains detailed timing information if available.
    #[pyo3(signature = (circuit, shots=1000, collect_profile=false))]
    pub fn simulate_shots(
        &self,
        py: Python,
        circuit: &Bound<PyAny>,
        shots: usize,
        collect_profile: bool,
    ) -> PyResult<PyObject> {
        let total_t0 = Instant::now();

        // PREPROCESSING: Circuit deserialization
        let preproc_t0 = Instant::now();
        let json_str: String = circuit.call_method0("model_dump_json")?.extract()?;

        let rust_circuit: Circuit = serde_json::from_str(&json_str).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Circuit JSON parse error: {e}"))
        })?;
        let preprocessing_ms = preproc_t0.elapsed().as_secs_f64() * 1000.0;

        let original_num_qubits = rust_circuit.num_qubits();
        let truncation_enabled = statevector_qubit_truncation_enabled();
        let truncation = if truncation_enabled {
            build_count_qubit_truncation(&rust_circuit)
                .map_err(pyo3::exceptions::PyRuntimeError::new_err)?
        } else {
            None
        };
        let truncation_used = truncation.is_some();
        let removed_qubits = truncation
            .as_ref()
            .map(|plan| plan.removed_qubits.clone())
            .unwrap_or_default();
        let simulation_circuit = truncation
            .as_ref()
            .map(|plan| &plan.circuit)
            .unwrap_or(&rust_circuit);

        let n = simulation_circuit.num_qubits();
        let num_cbits = simulation_circuit.num_cbits();
        let base_seed = self.seed.unwrap_or_else(|| rand::thread_rng().gen());

        let initial_state = {
            let mut s = vec![C::new(0.0, 0.0); 1 << n];
            s[0] = C::new(1.0, 0.0);
            s
        };

        // GATE FUSION: Fuse consecutive single-qubit gates
        let fusion_t0 = Instant::now();
        let fused = fuse_circuit(&simulation_circuit.instructions);
        let gate_fusion_ms = fusion_t0.elapsed().as_secs_f64() * 1000.0;
        let num_instructions = fused.len();
        let two_qubit_kernels_enabled = statevector_two_qubit_kernels_enabled();
        let two_qubit_kernels_used = two_qubit_kernels_enabled
            && fused
                .iter()
                .any(fused_instruction_uses_statevector_two_qubit_kernel);

        let shot_loop_profiler = ShotLoopProfiler::start("statevector", n, shots, num_instructions);

        let shot_branching_plan = if statevector_shot_branching_enabled() {
            terminal_measurement_plan(&fused)
        } else {
            None
        };
        let shot_branching_used = shot_branching_plan.is_some();
        let shot_branching_strategy = shot_branching_plan
            .as_ref()
            .map(|plan| terminal_measurement_strategy(plan, n).to_string());

        // PARALLEL EXECUTION: Run shots
        let exec_t0 = Instant::now();
        let counts = py
            .allow_threads(|| -> Result<HashMap<String, usize>, String> {
                if let Some(plan) = &shot_branching_plan {
                    run_terminal_measurement_branching(
                        &initial_state,
                        &fused,
                        plan,
                        n,
                        num_cbits,
                        shots,
                        base_seed,
                    )
                } else {
                    run_independent_shots(&initial_state, &fused, n, num_cbits, shots, base_seed)
                }
            })
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

        let parallel_execution_ms = exec_t0.elapsed().as_secs_f64() * 1000.0;

        if let Some(profiler) = shot_loop_profiler {
            profiler.finish();
        }

        // Build result dict
        let d = PyDict::new_bound(py);
        for (k, v) in &counts {
            d.set_item(k, v)?;
        }

        // Build profile if requested
        if collect_profile {
            let total_time_ms = total_t0.elapsed().as_secs_f64() * 1000.0;

            let profile = ShotsProfile {
                num_shots: shots,
                num_qubits: original_num_qubits,
                num_instructions,
                preprocessing_ms,
                gate_fusion_ms,
                parallel_execution_ms,
                per_shot_stats: None,
                statevector_simd_enabled: Some(statevector_simd_enabled()),
                statevector_simd_backend: statevector_simd_backend().map(str::to_owned),
                statevector_simd_used: Some(
                    statevector_simd_enabled() && n >= statevector_simd_min_qubits(),
                ),
                statevector_simd_min_qubits: Some(statevector_simd_min_qubits()),
                statevector_two_qubit_kernels_enabled: Some(two_qubit_kernels_enabled),
                statevector_two_qubit_kernels_used: Some(two_qubit_kernels_used),
                statevector_two_qubit_kernel_gates: two_qubit_kernels_enabled
                    .then_some(statevector_two_qubit_kernel_gates().to_string()),
                statevector_shot_branching_enabled: Some(statevector_shot_branching_enabled()),
                statevector_shot_branching_used: Some(shot_branching_used),
                statevector_shot_branching_strategy: shot_branching_strategy,
                mps_shot_branching_enabled: None,
                mps_shot_branching_used: None,
                mps_shot_branching_strategy: None,
                statevector_qubit_truncation_enabled: Some(truncation_enabled),
                statevector_qubit_truncation_used: Some(truncation_used),
                statevector_qubit_truncation_strategy: truncation_used
                    .then_some("count_relevance_remap".to_string()),
                statevector_original_num_qubits: Some(original_num_qubits),
                statevector_effective_num_qubits: Some(n),
                statevector_removed_qubits: truncation_used.then_some(removed_qubits.clone()),
                total_time_ms,
            };

            let profile_json_str = profile.to_json_string().map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Profile serialization error: {e}"
                ))
            })?;

            // Parse JSON string to Python dict
            let json_module = py.import_bound("json")?;
            let profile_dict = json_module.call_method1("loads", (&profile_json_str,))?;

            // Return dict with both counts and profile
            let result_dict = PyDict::new_bound(py);
            result_dict.set_item("counts", &d)?;
            result_dict.set_item("profile", &profile_dict)?;
            return Ok(result_dict.into());
        }

        Ok(d.into())
    }

    /// Run the circuit and return a SimulationResult.
    pub fn simulate(&self, py: Python, circuit: &Bound<PyAny>) -> PyResult<SimulationResult> {
        // 1. Serialize entire circuit to JSON (one boundary crossing)
        let json_str: String = circuit.call_method0("model_dump_json")?.extract()?;

        // 2. Deserialize in Rust Ã¢â‚¬â€ no more Python calls until we return
        let rust_circuit: Circuit = serde_json::from_str(&json_str).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Circuit JSON parse error: {e}"))
        })?;

        // 3. Initialise statevector |0...0Ã¢Å¸Â©
        let n = rust_circuit.num_qubits();
        let mut state = vec![C::new(0.0, 0.0); 1 << n];
        state[0] = C::new(1.0, 0.0);
        let mut cbits: HashMap<usize, i32> = HashMap::new();

        let seed = self.seed.unwrap_or_else(|| rand::thread_rng().gen());
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        let mut acc: Option<ProfileAcc> = if self.profile {
            Some(ProfileAcc::default())
        } else {
            None
        };

        let t0 = Instant::now();

        // 4. Execute all instructions natively in Rust
        for inst in &rust_circuit.instructions {
            run_instruction(&mut state, inst, n, &mut cbits, &mut rng, &mut acc)?;
        }

        let total_time = t0.elapsed().as_secs_f64();

        // 5. Build SimulationProfile if requested
        let prof = acc
            .map(|a| {
                Py::new(
                    py,
                    SimulationProfile {
                        apply_one_qubit_calls: a.oq_calls,
                        apply_one_qubit_time: a.oq_time,
                        apply_n_qubit_calls: a.nq_calls,
                        apply_n_qubit_time: a.nq_time,
                        measure_qubit_calls: a.mq_calls,
                        measure_qubit_time: a.mq_time,
                        total_time,
                    },
                )
            })
            .transpose()?;

        Ok(SimulationResult::new(state, n, cbits, prof))
    }
}

// ---------------------------------------------------------------------------
// Instruction dispatcher
// ---------------------------------------------------------------------------

fn cu_inner_matrix(theta: f64, phi: f64, lam: f64, gamma: f64) -> [[C; 2]; 2] {
    let phase = C::new(gamma.cos(), gamma.sin());
    let inner = gates::u(theta, phi, lam);
    [
        [phase * inner[0][0], phase * inner[0][1]],
        [phase * inner[1][0], phase * inner[1][1]],
    ]
}
fn run_instruction(
    state: &mut [C],
    inst: &Instruction,
    n: usize,
    cbits: &mut HashMap<usize, i32>,
    rng: &mut impl Rng,
    acc: &mut Option<ProfileAcc>,
) -> PyResult<()> {
    match inst {
        // -- Single-qubit fixed ------------------------------------------
        Instruction::Id { .. } | Instruction::U0 { .. } => {}

        Instruction::X { qubit } => do_oq(state, &gates::X, *qubit, n, acc),
        Instruction::Y { qubit } => do_oq(state, &gates::Y, *qubit, n, acc),
        Instruction::Z { qubit } => do_oq(state, &gates::Z, *qubit, n, acc),
        Instruction::H { qubit } => {
            let m = gates::h();
            do_oq(state, &m, *qubit, n, acc);
        }
        Instruction::S { qubit } => {
            let m = gates::s_gate();
            do_oq(state, &m, *qubit, n, acc);
        }
        Instruction::Sdg { qubit } => {
            let m = gates::sdg();
            do_oq(state, &m, *qubit, n, acc);
        }
        Instruction::T { qubit } => {
            let m = gates::t_gate();
            do_oq(state, &m, *qubit, n, acc);
        }
        Instruction::Tdg { qubit } => {
            let m = gates::tdg();
            do_oq(state, &m, *qubit, n, acc);
        }
        Instruction::Sx { qubit } => {
            let m = gates::sx();
            do_oq(state, &m, *qubit, n, acc);
        }
        Instruction::Sxdg { qubit } => {
            let m = gates::sxdg();
            do_oq(state, &m, *qubit, n, acc);
        }

        // -- Single-qubit parametric -------------------------------------
        Instruction::U3 {
            qubit,
            theta,
            phi,
            lam,
        } => {
            let m = gates::u3(*theta, *phi, *lam);
            do_oq(state, &m, *qubit, n, acc);
        }
        Instruction::U2 { qubit, phi, lam } => {
            let m = gates::u2(*phi, *lam);
            do_oq(state, &m, *qubit, n, acc);
        }
        Instruction::U1 { qubit, lam } => {
            let m = gates::u1(*lam);
            do_oq(state, &m, *qubit, n, acc);
        }
        Instruction::U {
            qubit,
            theta,
            phi,
            lam,
        } => {
            let m = gates::u(*theta, *phi, *lam);
            do_oq(state, &m, *qubit, n, acc);
        }
        Instruction::P { qubit, lam } => {
            let m = gates::p(*lam);
            do_oq(state, &m, *qubit, n, acc);
        }
        Instruction::Rx { qubit, theta } => {
            let m = gates::rx(*theta);
            do_oq(state, &m, *qubit, n, acc);
        }
        Instruction::Ry { qubit, theta } => {
            let m = gates::ry(*theta);
            do_oq(state, &m, *qubit, n, acc);
        }
        Instruction::Rz { qubit, phi } => {
            let m = gates::rz(*phi);
            do_oq(state, &m, *qubit, n, acc);
        }

        // -- Two-qubit fixed ---------------------------------------------
        Instruction::Cx { control, target } => {
            if statevector_two_qubit_kernels_enabled() {
                do_cx_kernel(state, *control, *target, n, acc);
            } else {
                do_nq(state, &m4(gates::cnot()), &[*control, *target], n, acc);
            }
        }
        Instruction::Cz { control, target } => {
            if statevector_two_qubit_kernels_enabled() {
                do_cz_kernel(state, *control, *target, n, acc);
            } else {
                do_nq(state, &m4(gates::cz()), &[*control, *target], n, acc);
            }
        }
        Instruction::Cy { control, target } => {
            if statevector_two_qubit_kernels_enabled() {
                do_controlled_oq_kernel(state, &gates::Y, *control, *target, n, acc);
            } else {
                do_nq(state, &m4(gates::cy()), &[*control, *target], n, acc);
            }
        }
        Instruction::Ch { control, target } => {
            if statevector_two_qubit_kernels_enabled() {
                let m = gates::h();
                do_controlled_oq_kernel(state, &m, *control, *target, n, acc);
            } else {
                do_nq(state, &m4(gates::ch()), &[*control, *target], n, acc);
            }
        }
        Instruction::Swap { a, b } => {
            if statevector_two_qubit_kernels_enabled() {
                do_swap_kernel(state, *a, *b, n, acc);
            } else {
                do_nq(state, &m4(gates::swap()), &[*a, *b], n, acc);
            }
        }
        Instruction::Csx { control, target } => {
            if statevector_two_qubit_kernels_enabled() {
                let m = gates::sx();
                do_controlled_oq_kernel(state, &m, *control, *target, n, acc);
            } else {
                do_nq(state, &m4(gates::csx()), &[*control, *target], n, acc);
            }
        }

        // -- Two-qubit parametric ----------------------------------------
        Instruction::Crx {
            control,
            target,
            theta,
        } => {
            if statevector_two_qubit_kernels_enabled() {
                let m = gates::rx(*theta);
                do_controlled_oq_kernel(state, &m, *control, *target, n, acc);
            } else {
                do_nq(state, &m4(gates::crx(*theta)), &[*control, *target], n, acc);
            }
        }
        Instruction::Cry {
            control,
            target,
            theta,
        } => {
            if statevector_two_qubit_kernels_enabled() {
                let m = gates::ry(*theta);
                do_controlled_oq_kernel(state, &m, *control, *target, n, acc);
            } else {
                do_nq(state, &m4(gates::cry(*theta)), &[*control, *target], n, acc);
            }
        }
        Instruction::Crz {
            control,
            target,
            lam,
        } => {
            if statevector_two_qubit_kernels_enabled() {
                let m = gates::rz(*lam);
                do_controlled_oq_kernel(state, &m, *control, *target, n, acc);
            } else {
                do_nq(state, &m4(gates::crz(*lam)), &[*control, *target], n, acc);
            }
        }
        Instruction::Cu1 {
            control,
            target,
            lam,
        } => {
            if statevector_two_qubit_kernels_enabled() {
                let m = gates::u1(*lam);
                do_controlled_oq_kernel(state, &m, *control, *target, n, acc);
            } else {
                do_nq(state, &m4(gates::cu1(*lam)), &[*control, *target], n, acc);
            }
        }
        Instruction::Cp {
            control,
            target,
            lam,
        } => {
            if statevector_two_qubit_kernels_enabled() {
                let m = gates::p(*lam);
                do_controlled_oq_kernel(state, &m, *control, *target, n, acc);
            } else {
                do_nq(state, &m4(gates::cp(*lam)), &[*control, *target], n, acc);
            }
        }
        Instruction::Cu3 {
            control,
            target,
            theta,
            phi,
            lam,
        } => {
            if statevector_two_qubit_kernels_enabled() {
                let m = gates::u3(*theta, *phi, *lam);
                do_controlled_oq_kernel(state, &m, *control, *target, n, acc);
            } else {
                do_nq(
                    state,
                    &m4(gates::cu3(*theta, *phi, *lam)),
                    &[*control, *target],
                    n,
                    acc,
                );
            }
        }
        Instruction::Cu {
            control,
            target,
            theta,
            phi,
            lam,
            gamma,
        } => {
            if statevector_two_qubit_kernels_enabled() {
                let m = cu_inner_matrix(*theta, *phi, *lam, *gamma);
                do_controlled_oq_kernel(state, &m, *control, *target, n, acc);
            } else {
                do_nq(
                    state,
                    &m4(gates::cu(*theta, *phi, *lam, *gamma)),
                    &[*control, *target],
                    n,
                    acc,
                );
            }
        }
        Instruction::Rxx { a, b, theta } => {
            do_nq(state, &m4(gates::rxx(*theta)), &[*a, *b], n, acc);
        }
        Instruction::Rzz { a, b, theta } => {
            do_nq(state, &m4(gates::rzz(*theta)), &[*a, *b], n, acc);
        }

        // -- Three-qubit -------------------------------------------------
        Instruction::Ccx {
            control1,
            control2,
            target,
        } => {
            if statevector_multi_control_kernels_enabled_for(n) {
                let controls = [*control1, *control2];
                do_multi_controlled_x_kernel(state, *target, n, &controls, acc);
            } else {
                do_nq(
                    state,
                    &m8(gates::ccx()),
                    &[*control1, *control2, *target],
                    n,
                    acc,
                );
            }
        }
        Instruction::Cswap {
            control,
            target1,
            target2,
        } => {
            if statevector_multi_control_kernels_enabled_for(n) {
                do_cswap_kernel(state, *control, *target1, *target2, n, acc);
            } else {
                do_nq(
                    state,
                    &m8(gates::cswap()),
                    &[*control, *target1, *target2],
                    n,
                    acc,
                );
            }
        }
        Instruction::Rccx {
            control1,
            control2,
            target,
        } => {
            do_nq(
                state,
                &m8(gates::rccx()),
                &[*control1, *control2, *target],
                n,
                acc,
            );
        }

        // -- Four-qubit --------------------------------------------------
        Instruction::Rc3x {
            control1,
            control2,
            control3,
            target,
        } => {
            do_nq(
                state,
                &m16(gates::rc3x()),
                &[*control1, *control2, *control3, *target],
                n,
                acc,
            );
        }
        Instruction::C3x {
            control1,
            control2,
            control3,
            target,
        } => {
            if statevector_multi_control_kernels_enabled_for(n) {
                let controls = [*control1, *control2, *control3];
                do_multi_controlled_x_kernel(state, *target, n, &controls, acc);
            } else {
                do_nq(
                    state,
                    &m16(gates::c3x()),
                    &[*control1, *control2, *control3, *target],
                    n,
                    acc,
                );
            }
        }
        Instruction::C3sqrtx {
            control1,
            control2,
            control3,
            target,
        } => {
            if statevector_multi_control_kernels_enabled_for(n) {
                let sx = gates::sx();
                let controls = [(*control1, true), (*control2, true), (*control3, true)];
                do_multi_controlled_oq_kernel(state, &sx, *target, n, &controls, acc);
            } else {
                do_nq(
                    state,
                    &m16(gates::c3sqrtx()),
                    &[*control1, *control2, *control3, *target],
                    n,
                    acc,
                );
            }
        }

        // -- Five-qubit --------------------------------------------------
        Instruction::C4x {
            control1,
            control2,
            control3,
            control4,
            target,
        } => {
            if statevector_multi_control_kernels_enabled_for(n) {
                let controls = [*control1, *control2, *control3, *control4];
                do_multi_controlled_x_kernel(state, *target, n, &controls, acc);
            } else {
                do_nq(
                    state,
                    &m32(gates::c4x()),
                    &[*control1, *control2, *control3, *control4, *target],
                    n,
                    acc,
                );
            }
        }

        // -- Cross-node / generic ----------------------------------------
        Instruction::Gate { name, qubits, .. } => {
            match name.to_lowercase().as_str() {
                "remote_link_phi_plus" => {
                    do_nq(state, &m4(gates::phi_plus()), qubits, n, acc);
                }
                "remote_link_psi_minus" => {
                    do_nq(state, &m4(gates::psi_minus()), qubits, n, acc);
                }
                "remote_link_psi_plus" => {
                    do_nq(state, &m4(gates::psi_plus()), qubits, n, acc);
                }
                "nonlocal_cz" | "remote_cz" => {
                    do_nq(state, &m4(gates::cz()), qubits, n, acc);
                }
                "remote_cx" => {
                    do_nq(state, &m4(gates::cnot()), qubits, n, acc);
                }
                "remote_epr" => {
                    do_nq(state, &m4(gates::phi_plus()), qubits, n, acc);
                }
                "remote_barrier" => {
                    // no-op
                }
                "remote_cu1" => {
                    return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                        "Symbolic 'remote_cu1' cannot be simulated natively. Distribute with lowered=True.",
                    ));
                }
                other => {
                    return Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
                        "Unsupported generic gate: {other:?}. Decompose it before simulating."
                    )));
                }
            }
        }

        // -- Measurement -------------------------------------------------
        Instruction::Measure { qubit, cbit } => {
            let outcome = do_mq(state, *qubit, n, rng, acc);
            cbits.insert(*cbit, outcome as i32);
        }

        // -- Classical control -------------------------------------------
        Instruction::Conditional { condition, op } => {
            let mut actual: u64 = 0;
            for bit in 0..condition.creg_size {
                let val = *cbits.get(&(condition.creg_base + bit)).unwrap_or(&0) as u64;
                actual |= val << bit;
            }
            if actual == condition.creg_value {
                run_instruction(state, op, n, cbits, rng, acc)?;
            }
        }

        // -- Reset -------------------------------------------------------
        Instruction::Reset { qubit } => {
            let outcome = do_mq(state, *qubit, n, rng, acc);
            if outcome == 1 {
                do_oq(state, &gates::X, *qubit, n, acc);
            }
        }

        // -- No-ops ------------------------------------------------------
        Instruction::Barrier | Instruction::Classical { .. } => {}
    }
    Ok(())
}

struct QubitTruncation {
    circuit: Circuit,
    removed_qubits: Vec<usize>,
}

fn statevector_qubit_truncation_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        let Ok(value) = std::env::var("DQSIM_STATEVECTOR_QUBIT_TRUNCATION") else {
            return false;
        };
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on" | "count" | "counts" | "inactive"
        )
    })
}

fn build_count_qubit_truncation(circuit: &Circuit) -> Result<Option<QubitTruncation>, String> {
    let original_n = circuit.num_qubits();
    if original_n == 0 {
        return Ok(None);
    }

    let keep = count_relevant_qubits(circuit, original_n);
    if keep.iter().all(|&value| value) {
        return Ok(None);
    }

    let retained_qubits: Vec<usize> = keep
        .iter()
        .enumerate()
        .filter_map(|(qubit, &retain)| retain.then_some(qubit))
        .collect();
    let removed_qubits: Vec<usize> = keep
        .iter()
        .enumerate()
        .filter_map(|(qubit, &retain)| (!retain).then_some(qubit))
        .collect();

    let mut remap = vec![None; original_n];
    for (new_qubit, &old_qubit) in retained_qubits.iter().enumerate() {
        remap[old_qubit] = Some(new_qubit);
    }

    let mut instructions = Vec::with_capacity(circuit.instructions.len());
    for inst in &circuit.instructions {
        if let Some(remapped) = remap_instruction_for_truncation(inst, &remap)? {
            instructions.push(remapped);
        }
    }

    let mut qregs = HashMap::new();
    if !retained_qubits.is_empty() {
        qregs.insert(
            "__dqsim_truncated".to_string(),
            Register {
                name: "__dqsim_truncated".to_string(),
                size: retained_qubits.len(),
                base: 0,
            },
        );
    }

    Ok(Some(QubitTruncation {
        circuit: Circuit {
            qregs,
            cregs: circuit.cregs.clone(),
            instructions,
        },
        removed_qubits,
    }))
}

fn count_relevant_qubits(circuit: &Circuit, num_qubits: usize) -> Vec<bool> {
    let mut required_qubits = vec![false; num_qubits];
    let mut required_cbits = vec![true; circuit.num_cbits()];

    for inst in circuit.instructions.iter().rev() {
        match inst {
            Instruction::Measure { qubit, cbit } => {
                if *cbit < required_cbits.len() && required_cbits[*cbit] {
                    mark_qubit(&mut required_qubits, *qubit);
                }
            }
            Instruction::Conditional { condition, op } => {
                let touched = op.qubits();
                if touches_required_qubit(&touched, &required_qubits) {
                    mark_qubits(&mut required_qubits, &touched);
                    mark_condition_cbits(&mut required_cbits, condition);
                }
            }
            _ => {
                let touched = inst.qubits();
                if touches_required_qubit(&touched, &required_qubits) {
                    mark_qubits(&mut required_qubits, &touched);
                }
            }
        }
    }

    required_qubits
}

fn touches_required_qubit(qubits: &[usize], required_qubits: &[bool]) -> bool {
    qubits
        .iter()
        .any(|&qubit| required_qubits.get(qubit).copied().unwrap_or(false))
}

fn mark_qubit(required_qubits: &mut [bool], qubit: usize) {
    if let Some(required) = required_qubits.get_mut(qubit) {
        *required = true;
    }
}

fn mark_qubits(required_qubits: &mut [bool], qubits: &[usize]) {
    for &qubit in qubits {
        mark_qubit(required_qubits, qubit);
    }
}

fn mark_condition_cbits(required_cbits: &mut Vec<bool>, condition: &Condition) {
    let end = condition.creg_base.saturating_add(condition.creg_size);
    if required_cbits.len() < end {
        required_cbits.resize(end, false);
    }
    for cbit in condition.creg_base..end {
        required_cbits[cbit] = true;
    }
}

fn remap_instruction_for_truncation(
    inst: &Instruction,
    remap: &[Option<usize>],
) -> Result<Option<Instruction>, String> {
    let touched = inst.qubits();
    if touched.is_empty() {
        return Ok(Some(inst.clone()));
    }

    let retained = instruction_touches_retained_qubits(&touched, remap)?;
    if !retained {
        return Ok(None);
    }

    let mut remapped = inst.clone();
    remap_instruction_qubits_in_place(&mut remapped, remap)?;
    Ok(Some(remapped))
}

fn instruction_touches_retained_qubits(
    touched: &[usize],
    remap: &[Option<usize>],
) -> Result<bool, String> {
    let mut any_retained = false;
    let mut any_removed = false;
    for &qubit in touched {
        match remap.get(qubit).copied().flatten() {
            Some(_) => any_retained = true,
            None if qubit < remap.len() => any_removed = true,
            None => return Err(format!("qubit index {qubit} is outside the circuit width")),
        }
    }

    if any_retained && any_removed {
        return Err(
            "qubit truncation found a partially retained instruction; relevance analysis is too aggressive"
                .to_string(),
        );
    }

    Ok(any_retained)
}

fn remap_instruction_qubits_in_place(
    inst: &mut Instruction,
    remap: &[Option<usize>],
) -> Result<(), String> {
    match inst {
        Instruction::Id { qubit }
        | Instruction::X { qubit }
        | Instruction::Y { qubit }
        | Instruction::Z { qubit }
        | Instruction::H { qubit }
        | Instruction::S { qubit }
        | Instruction::Sdg { qubit }
        | Instruction::T { qubit }
        | Instruction::Tdg { qubit }
        | Instruction::Sx { qubit }
        | Instruction::Sxdg { qubit }
        | Instruction::U3 { qubit, .. }
        | Instruction::U2 { qubit, .. }
        | Instruction::U1 { qubit, .. }
        | Instruction::U { qubit, .. }
        | Instruction::P { qubit, .. }
        | Instruction::Rx { qubit, .. }
        | Instruction::Ry { qubit, .. }
        | Instruction::Rz { qubit, .. }
        | Instruction::U0 { qubit }
        | Instruction::Measure { qubit, .. }
        | Instruction::Reset { qubit } => remap_required_qubit(qubit, remap),
        Instruction::Cx { control, target }
        | Instruction::Cz { control, target }
        | Instruction::Cy { control, target }
        | Instruction::Ch { control, target }
        | Instruction::Csx { control, target }
        | Instruction::Crx {
            control, target, ..
        }
        | Instruction::Cry {
            control, target, ..
        }
        | Instruction::Crz {
            control, target, ..
        }
        | Instruction::Cu1 {
            control, target, ..
        }
        | Instruction::Cp {
            control, target, ..
        }
        | Instruction::Cu3 {
            control, target, ..
        }
        | Instruction::Cu {
            control, target, ..
        } => {
            remap_required_qubit(control, remap)?;
            remap_required_qubit(target, remap)
        }
        Instruction::Swap { a, b }
        | Instruction::Rxx { a, b, .. }
        | Instruction::Rzz { a, b, .. } => {
            remap_required_qubit(a, remap)?;
            remap_required_qubit(b, remap)
        }
        Instruction::Ccx {
            control1,
            control2,
            target,
        }
        | Instruction::Rccx {
            control1,
            control2,
            target,
        } => {
            remap_required_qubit(control1, remap)?;
            remap_required_qubit(control2, remap)?;
            remap_required_qubit(target, remap)
        }
        Instruction::Cswap {
            control,
            target1,
            target2,
        } => {
            remap_required_qubit(control, remap)?;
            remap_required_qubit(target1, remap)?;
            remap_required_qubit(target2, remap)
        }
        Instruction::Rc3x {
            control1,
            control2,
            control3,
            target,
        }
        | Instruction::C3x {
            control1,
            control2,
            control3,
            target,
        }
        | Instruction::C3sqrtx {
            control1,
            control2,
            control3,
            target,
        } => {
            remap_required_qubit(control1, remap)?;
            remap_required_qubit(control2, remap)?;
            remap_required_qubit(control3, remap)?;
            remap_required_qubit(target, remap)
        }
        Instruction::C4x {
            control1,
            control2,
            control3,
            control4,
            target,
        } => {
            remap_required_qubit(control1, remap)?;
            remap_required_qubit(control2, remap)?;
            remap_required_qubit(control3, remap)?;
            remap_required_qubit(control4, remap)?;
            remap_required_qubit(target, remap)
        }
        Instruction::Gate { qubits, .. } => {
            for qubit in qubits {
                remap_required_qubit(qubit, remap)?;
            }
            Ok(())
        }
        Instruction::Conditional { op, .. } => {
            remap_instruction_qubits_in_place(op.as_mut(), remap)
        }
        Instruction::Barrier | Instruction::Classical { .. } => Ok(()),
    }
}

fn remap_required_qubit(qubit: &mut usize, remap: &[Option<usize>]) -> Result<(), String> {
    let old = *qubit;
    let Some(new_qubit) = remap.get(old).copied().flatten() else {
        return Err(format!("qubit index {old} was not retained by truncation"));
    };
    *qubit = new_qubit;
    Ok(())
}
#[derive(Clone, Copy)]
struct TerminalMeasurement {
    qubit: usize,
    cbit: usize,
}

struct TerminalMeasurementPlan {
    measurements: Vec<TerminalMeasurement>,
}

fn statevector_shot_branching_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        let Ok(value) = std::env::var("DQSIM_STATEVECTOR_SHOT_BRANCHING") else {
            return false;
        };
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on" | "terminal" | "terminal_measurement"
        )
    })
}

#[derive(Clone, Copy)]
enum TerminalSamplingMode {
    Auto,
    Marginal,
    FullState,
}

fn statevector_shot_branching_sampling_mode() -> TerminalSamplingMode {
    static MODE: OnceLock<TerminalSamplingMode> = OnceLock::new();
    *MODE.get_or_init(|| {
        let Ok(value) = std::env::var("DQSIM_STATEVECTOR_SHOT_BRANCHING_SAMPLE_MODE") else {
            return TerminalSamplingMode::Auto;
        };

        match value.trim().to_ascii_lowercase().as_str() {
            "marginal" | "legacy" => TerminalSamplingMode::Marginal,
            "full" | "full_state" | "basis" => TerminalSamplingMode::FullState,
            _ => TerminalSamplingMode::Auto,
        }
    })
}

fn terminal_measurement_plan(fused: &[FusedInstruction<'_>]) -> Option<TerminalMeasurementPlan> {
    let mut saw_measurement = false;
    let mut measurements = Vec::new();
    let mut seen_cbits = Vec::new();
    let mut measured_qubits = Vec::new();

    for fi in fused {
        match fi {
            FusedInstruction::Fused1Q { qubit, .. } => {
                if measured_qubits.contains(qubit) {
                    return None;
                }
            }
            FusedInstruction::Original(inst) => match inst {
                Instruction::Measure { qubit, cbit } => {
                    saw_measurement = true;
                    if seen_cbits.contains(cbit) {
                        return None;
                    }
                    seen_cbits.push(*cbit);
                    measured_qubits.push(*qubit);
                    measurements.push(TerminalMeasurement {
                        qubit: *qubit,
                        cbit: *cbit,
                    });
                }
                Instruction::Barrier | Instruction::Classical { .. } => {}
                Instruction::Reset { .. } | Instruction::Conditional { .. } => return None,
                _ => {
                    if saw_measurement {
                        let mut touches_measured = false;
                        inst.for_each_qubit(|q| {
                            if measured_qubits.contains(&q) {
                                touches_measured = true;
                            }
                        });
                        if touches_measured {
                            return None;
                        }
                    }
                }
            },
        }
    }

    if measurements.is_empty() {
        return None;
    }

    Some(TerminalMeasurementPlan { measurements })
}

fn fused_instruction_uses_statevector_two_qubit_kernel(fi: &FusedInstruction<'_>) -> bool {
    match fi {
        FusedInstruction::Original(inst) => instruction_uses_statevector_two_qubit_kernel(inst),
        FusedInstruction::Fused1Q { .. } => false,
    }
}

fn instruction_uses_statevector_two_qubit_kernel(inst: &Instruction) -> bool {
    matches!(
        inst,
        Instruction::Cx { .. }
            | Instruction::Cz { .. }
            | Instruction::Cy { .. }
            | Instruction::Ch { .. }
            | Instruction::Swap { .. }
            | Instruction::Csx { .. }
            | Instruction::Crx { .. }
            | Instruction::Cry { .. }
            | Instruction::Crz { .. }
            | Instruction::Cu1 { .. }
            | Instruction::Cp { .. }
            | Instruction::Cu3 { .. }
            | Instruction::Cu { .. }
            | Instruction::Ccx { .. }
            | Instruction::Cswap { .. }
            | Instruction::C3x { .. }
            | Instruction::C3sqrtx { .. }
            | Instruction::C4x { .. }
    )
}
fn run_independent_shots(
    initial_state: &[C],
    fused: &[FusedInstruction<'_>],
    n: usize,
    num_cbits: usize,
    shots: usize,
    base_seed: u64,
) -> Result<HashMap<String, usize>, String> {
    (0..shots)
        .into_par_iter()
        .map(|i| -> Result<String, String> {
            let mut state = initial_state.to_vec();
            let mut cbits: HashMap<usize, i32> = HashMap::new();
            let mut rng = ChaCha8Rng::seed_from_u64(base_seed.wrapping_add(i as u64));
            for fi in fused {
                run_fused_par(&mut state, fi, n, &mut cbits, &mut rng)?;
            }
            Ok(format_cbits(&cbits, num_cbits))
        })
        .try_fold(HashMap::new, |mut m, r| {
            let k = r?;
            *m.entry(k).or_insert(0) += 1;
            Ok(m)
        })
        .try_reduce(HashMap::new, |mut a, b| {
            for (k, v) in b {
                *a.entry(k).or_insert(0) += v;
            }
            Ok(a)
        })
}

fn run_terminal_measurement_branching(
    initial_state: &[C],
    fused: &[FusedInstruction<'_>],
    plan: &TerminalMeasurementPlan,
    n: usize,
    num_cbits: usize,
    shots: usize,
    base_seed: u64,
) -> Result<HashMap<String, usize>, String> {
    let mut state = initial_state.to_vec();
    let mut cbits: HashMap<usize, i32> = HashMap::new();
    let mut prefix_rng = ChaCha8Rng::seed_from_u64(base_seed);

    for fi in fused {
        if matches!(fi, FusedInstruction::Original(Instruction::Measure { .. })) {
            continue;
        }
        run_fused_par(&mut state, fi, n, &mut cbits, &mut prefix_rng)?;
    }

    let layout = terminal_sampling_layout(plan);
    if terminal_measurement_uses_full_state(&layout, n) {
        Ok(sample_terminal_full_state(
            &state, num_cbits, plan, shots, base_seed,
        ))
    } else {
        Ok(sample_terminal_marginal(
            &state, num_cbits, plan, &layout, n, shots, base_seed,
        ))
    }
}

struct TerminalSamplingLayout {
    sampled_qubits: Vec<usize>,
    measurement_indices: Vec<usize>,
}

fn terminal_measurement_strategy(plan: &TerminalMeasurementPlan, n: usize) -> &'static str {
    let layout = terminal_sampling_layout(plan);
    if terminal_measurement_uses_full_state(&layout, n) {
        "terminal_measurement_full_state_batch"
    } else {
        "terminal_measurement_marginal_batch"
    }
}

fn terminal_measurement_uses_full_state(layout: &TerminalSamplingLayout, n: usize) -> bool {
    match statevector_shot_branching_sampling_mode() {
        TerminalSamplingMode::Auto => layout.sampled_qubits.len() == n,
        TerminalSamplingMode::Marginal => false,
        TerminalSamplingMode::FullState => true,
    }
}

fn terminal_sampling_layout(plan: &TerminalMeasurementPlan) -> TerminalSamplingLayout {
    let mut sampled_qubits = Vec::new();
    let mut measurement_indices = Vec::with_capacity(plan.measurements.len());
    for measurement in &plan.measurements {
        let idx = sampled_qubits
            .iter()
            .position(|&q| q == measurement.qubit)
            .unwrap_or_else(|| {
                sampled_qubits.push(measurement.qubit);
                sampled_qubits.len() - 1
            });
        measurement_indices.push(idx);
    }

    TerminalSamplingLayout {
        sampled_qubits,
        measurement_indices,
    }
}

fn sample_terminal_marginal(
    state: &[C],
    num_cbits: usize,
    plan: &TerminalMeasurementPlan,
    layout: &TerminalSamplingLayout,
    n: usize,
    shots: usize,
    base_seed: u64,
) -> HashMap<String, usize> {
    let probs = marginal_probs(state, n, &layout.sampled_qubits);
    let cdf = cumulative_distribution(probs.into_iter());
    let mut rng = ChaCha8Rng::seed_from_u64(base_seed);
    let width = layout.sampled_qubits.len();

    let mut counts = HashMap::new();
    for (sample_idx, count) in sample_cdf_counts(&cdf, shots, &mut rng) {
        let key = terminal_sample_key(
            sample_idx,
            width,
            plan,
            &layout.measurement_indices,
            num_cbits,
        );
        *counts.entry(key).or_insert(0) += count;
    }

    counts
}

fn sample_terminal_full_state(
    state: &[C],
    num_cbits: usize,
    plan: &TerminalMeasurementPlan,
    shots: usize,
    base_seed: u64,
) -> HashMap<String, usize> {
    let cdf = cumulative_distribution(state.iter().map(|amp| amp.norm_sqr()));
    let mut rng = ChaCha8Rng::seed_from_u64(base_seed);

    let mut counts = HashMap::new();
    for (basis, count) in sample_cdf_counts(&cdf, shots, &mut rng) {
        let key = terminal_basis_key(basis, plan, num_cbits);
        *counts.entry(key).or_insert(0) += count;
    }

    counts
}

fn cumulative_distribution(probs: impl Iterator<Item = f64>) -> Vec<f64> {
    let mut cdf = Vec::with_capacity(probs.size_hint().0);
    let mut acc = 0.0;
    for p in probs {
        acc += p;
        cdf.push(acc);
    }
    if let Some(last) = cdf.last_mut() {
        *last = 1.0;
    }
    cdf
}

fn sample_cdf_counts(cdf: &[f64], shots: usize, rng: &mut impl Rng) -> Vec<(usize, usize)> {
    if cdf.is_empty() || shots == 0 {
        return Vec::new();
    }

    if shots.saturating_mul(4) < cdf.len() {
        let mut sampled = HashMap::with_capacity(shots);
        for _ in 0..shots {
            let idx = sample_cdf(cdf, rng);
            *sampled.entry(idx).or_insert(0) += 1;
        }
        sampled.into_iter().collect()
    } else {
        let mut sampled = vec![0usize; cdf.len()];
        for _ in 0..shots {
            let idx = sample_cdf(cdf, rng);
            sampled[idx] += 1;
        }
        sampled
            .into_iter()
            .enumerate()
            .filter(|&(_, count)| count != 0)
            .collect()
    }
}

fn sample_cdf(cdf: &[f64], rng: &mut impl Rng) -> usize {
    let r: f64 = rng.gen();
    cdf.partition_point(|&c| c < r)
        .min(cdf.len().saturating_sub(1))
}

fn terminal_basis_key(basis: usize, plan: &TerminalMeasurementPlan, num_cbits: usize) -> String {
    if num_cbits == 0 {
        return String::new();
    }

    let mut bits = vec![b'0'; num_cbits];
    for measurement in &plan.measurements {
        if measurement.cbit >= num_cbits {
            continue;
        }
        let measured = ((basis >> measurement.qubit) & 1) != 0;
        bits[num_cbits - 1 - measurement.cbit] = if measured { b'1' } else { b'0' };
    }
    String::from_utf8(bits).expect("terminal measurement keys are ASCII bits")
}

fn terminal_sample_key(
    sample_idx: usize,
    width: usize,
    plan: &TerminalMeasurementPlan,
    measurement_indices: &[usize],
    num_cbits: usize,
) -> String {
    if num_cbits == 0 {
        return String::new();
    }

    let mut bits = vec![b'0'; num_cbits];
    for (measurement, &qubit_idx) in plan.measurements.iter().zip(measurement_indices) {
        if measurement.cbit >= num_cbits {
            continue;
        }
        let measured = ((sample_idx >> (width - 1 - qubit_idx)) & 1) != 0;
        bits[num_cbits - 1 - measurement.cbit] = if measured { b'1' } else { b'0' };
    }
    String::from_utf8(bits).expect("terminal measurement keys are ASCII bits")
}

#[cfg(test)]
mod shot_branching_tests {
    use super::*;

    fn plan(measurements: &[(usize, usize)]) -> TerminalMeasurementPlan {
        TerminalMeasurementPlan {
            measurements: measurements
                .iter()
                .map(|&(qubit, cbit)| TerminalMeasurement { qubit, cbit })
                .collect(),
        }
    }

    #[test]
    fn terminal_basis_key_maps_qubits_to_classical_bits() {
        let plan = plan(&[(0, 2), (2, 0)]);

        assert_eq!(terminal_basis_key(0b101, &plan, 3), "101");
        assert_eq!(terminal_basis_key(0b001, &plan, 3), "100");
    }

    #[test]
    fn full_state_terminal_sampler_preserves_certain_basis_state() {
        let plan = plan(&[(0, 0), (1, 1), (2, 2)]);
        let mut state = vec![C::new(0.0, 0.0); 8];
        state[0b101] = C::new(1.0, 0.0);

        let counts = sample_terminal_full_state(&state, 3, &plan, 32, 7);

        assert_eq!(counts.len(), 1);
        assert_eq!(counts.get("101"), Some(&32));
    }
}

// ---------------------------------------------------------------------------
// GIL-free instruction dispatcher (used inside py.allow_threads for parallel shots)
// ---------------------------------------------------------------------------

fn run_instruction_par(
    state: &mut [C],
    inst: &Instruction,
    n: usize,
    cbits: &mut HashMap<usize, i32>,
    rng: &mut impl Rng,
) -> Result<(), String> {
    match inst {
        // -- Single-qubit fixed ------------------------------------------
        Instruction::Id { .. } | Instruction::U0 { .. } => {}

        Instruction::X { qubit } => apply_one_qubit_seq(state, &gates::X, *qubit, n, &[]),
        Instruction::Y { qubit } => apply_one_qubit_seq(state, &gates::Y, *qubit, n, &[]),
        Instruction::Z { qubit } => apply_one_qubit_seq(state, &gates::Z, *qubit, n, &[]),
        Instruction::H { qubit } => {
            apply_one_qubit_seq(state, &gates::h(), *qubit, n, &[]);
        }
        Instruction::S { qubit } => {
            apply_one_qubit_seq(state, &gates::s_gate(), *qubit, n, &[]);
        }
        Instruction::Sdg { qubit } => {
            apply_one_qubit_seq(state, &gates::sdg(), *qubit, n, &[]);
        }
        Instruction::T { qubit } => {
            apply_one_qubit_seq(state, &gates::t_gate(), *qubit, n, &[]);
        }
        Instruction::Tdg { qubit } => {
            apply_one_qubit_seq(state, &gates::tdg(), *qubit, n, &[]);
        }
        Instruction::Sx { qubit } => {
            apply_one_qubit_seq(state, &gates::sx(), *qubit, n, &[]);
        }
        Instruction::Sxdg { qubit } => {
            apply_one_qubit_seq(state, &gates::sxdg(), *qubit, n, &[]);
        }

        // -- Single-qubit parametric -------------------------------------
        Instruction::U3 {
            qubit,
            theta,
            phi,
            lam,
        } => {
            apply_one_qubit_seq(state, &gates::u3(*theta, *phi, *lam), *qubit, n, &[]);
        }
        Instruction::U2 { qubit, phi, lam } => {
            apply_one_qubit_seq(state, &gates::u2(*phi, *lam), *qubit, n, &[]);
        }
        Instruction::U1 { qubit, lam } => {
            apply_one_qubit_seq(state, &gates::u1(*lam), *qubit, n, &[]);
        }
        Instruction::U {
            qubit,
            theta,
            phi,
            lam,
        } => {
            apply_one_qubit_seq(state, &gates::u(*theta, *phi, *lam), *qubit, n, &[]);
        }
        Instruction::P { qubit, lam } => {
            apply_one_qubit_seq(state, &gates::p(*lam), *qubit, n, &[]);
        }
        Instruction::Rx { qubit, theta } => {
            apply_one_qubit_seq(state, &gates::rx(*theta), *qubit, n, &[]);
        }
        Instruction::Ry { qubit, theta } => {
            apply_one_qubit_seq(state, &gates::ry(*theta), *qubit, n, &[]);
        }
        Instruction::Rz { qubit, phi } => {
            apply_one_qubit_seq(state, &gates::rz(*phi), *qubit, n, &[]);
        }

        // -- Two-qubit fixed ---------------------------------------------
        Instruction::Cx { control, target } => {
            if statevector_two_qubit_kernels_enabled() {
                apply_cx_kernel_seq(state, *control, *target, n);
            } else {
                apply_n_qubit_seq(state, &m4(gates::cnot()), &[*control, *target], n);
            }
        }
        Instruction::Cz { control, target } => {
            if statevector_two_qubit_kernels_enabled() {
                apply_cz_kernel_seq(state, *control, *target, n);
            } else {
                apply_n_qubit_seq(state, &m4(gates::cz()), &[*control, *target], n);
            }
        }
        Instruction::Cy { control, target } => {
            if statevector_two_qubit_kernels_enabled() {
                apply_one_qubit_seq(state, &gates::Y, *target, n, &[(*control, true)]);
            } else {
                apply_n_qubit_seq(state, &m4(gates::cy()), &[*control, *target], n);
            }
        }
        Instruction::Ch { control, target } => {
            if statevector_two_qubit_kernels_enabled() {
                apply_one_qubit_seq(state, &gates::h(), *target, n, &[(*control, true)]);
            } else {
                apply_n_qubit_seq(state, &m4(gates::ch()), &[*control, *target], n);
            }
        }
        Instruction::Swap { a, b } => {
            if statevector_two_qubit_kernels_enabled() {
                apply_swap_kernel_seq(state, *a, *b, n);
            } else {
                apply_n_qubit_seq(state, &m4(gates::swap()), &[*a, *b], n);
            }
        }
        Instruction::Csx { control, target } => {
            if statevector_two_qubit_kernels_enabled() {
                apply_one_qubit_seq(state, &gates::sx(), *target, n, &[(*control, true)]);
            } else {
                apply_n_qubit_seq(state, &m4(gates::csx()), &[*control, *target], n);
            }
        }

        // -- Two-qubit parametric ----------------------------------------
        Instruction::Crx {
            control,
            target,
            theta,
        } => {
            if statevector_two_qubit_kernels_enabled() {
                apply_one_qubit_seq(state, &gates::rx(*theta), *target, n, &[(*control, true)]);
            } else {
                apply_n_qubit_seq(state, &m4(gates::crx(*theta)), &[*control, *target], n);
            }
        }
        Instruction::Cry {
            control,
            target,
            theta,
        } => {
            if statevector_two_qubit_kernels_enabled() {
                apply_one_qubit_seq(state, &gates::ry(*theta), *target, n, &[(*control, true)]);
            } else {
                apply_n_qubit_seq(state, &m4(gates::cry(*theta)), &[*control, *target], n);
            }
        }
        Instruction::Crz {
            control,
            target,
            lam,
        } => {
            if statevector_two_qubit_kernels_enabled() {
                apply_one_qubit_seq(state, &gates::rz(*lam), *target, n, &[(*control, true)]);
            } else {
                apply_n_qubit_seq(state, &m4(gates::crz(*lam)), &[*control, *target], n);
            }
        }
        Instruction::Cu1 {
            control,
            target,
            lam,
        } => {
            if statevector_two_qubit_kernels_enabled() {
                apply_one_qubit_seq(state, &gates::u1(*lam), *target, n, &[(*control, true)]);
            } else {
                apply_n_qubit_seq(state, &m4(gates::cu1(*lam)), &[*control, *target], n);
            }
        }
        Instruction::Cp {
            control,
            target,
            lam,
        } => {
            if statevector_two_qubit_kernels_enabled() {
                apply_one_qubit_seq(state, &gates::p(*lam), *target, n, &[(*control, true)]);
            } else {
                apply_n_qubit_seq(state, &m4(gates::cp(*lam)), &[*control, *target], n);
            }
        }
        Instruction::Cu3 {
            control,
            target,
            theta,
            phi,
            lam,
        } => {
            if statevector_two_qubit_kernels_enabled() {
                apply_one_qubit_seq(
                    state,
                    &gates::u3(*theta, *phi, *lam),
                    *target,
                    n,
                    &[(*control, true)],
                );
            } else {
                apply_n_qubit_seq(
                    state,
                    &m4(gates::cu3(*theta, *phi, *lam)),
                    &[*control, *target],
                    n,
                );
            }
        }
        Instruction::Cu {
            control,
            target,
            theta,
            phi,
            lam,
            gamma,
        } => {
            if statevector_two_qubit_kernels_enabled() {
                apply_one_qubit_seq(
                    state,
                    &cu_inner_matrix(*theta, *phi, *lam, *gamma),
                    *target,
                    n,
                    &[(*control, true)],
                );
            } else {
                apply_n_qubit_seq(
                    state,
                    &m4(gates::cu(*theta, *phi, *lam, *gamma)),
                    &[*control, *target],
                    n,
                );
            }
        }
        Instruction::Rxx { a, b, theta } => {
            apply_n_qubit_seq(state, &m4(gates::rxx(*theta)), &[*a, *b], n);
        }
        Instruction::Rzz { a, b, theta } => {
            apply_n_qubit_seq(state, &m4(gates::rzz(*theta)), &[*a, *b], n);
        }

        // -- Three-qubit -------------------------------------------------
        Instruction::Ccx {
            control1,
            control2,
            target,
        } => {
            if statevector_multi_control_kernels_enabled_for(n) {
                let controls = [*control1, *control2];
                apply_multi_controlled_x_kernel_seq(state, &controls, *target, n);
            } else {
                apply_n_qubit_seq(
                    state,
                    &m8(gates::ccx()),
                    &[*control1, *control2, *target],
                    n,
                );
            }
        }
        Instruction::Cswap {
            control,
            target1,
            target2,
        } => {
            if statevector_multi_control_kernels_enabled_for(n) {
                apply_cswap_kernel_seq(state, *control, *target1, *target2, n);
            } else {
                apply_n_qubit_seq(
                    state,
                    &m8(gates::cswap()),
                    &[*control, *target1, *target2],
                    n,
                );
            }
        }
        Instruction::Rccx {
            control1,
            control2,
            target,
        } => {
            apply_n_qubit_seq(
                state,
                &m8(gates::rccx()),
                &[*control1, *control2, *target],
                n,
            );
        }

        // -- Four-qubit --------------------------------------------------
        Instruction::Rc3x {
            control1,
            control2,
            control3,
            target,
        } => {
            apply_n_qubit_seq(
                state,
                &m16(gates::rc3x()),
                &[*control1, *control2, *control3, *target],
                n,
            );
        }
        Instruction::C3x {
            control1,
            control2,
            control3,
            target,
        } => {
            if statevector_multi_control_kernels_enabled_for(n) {
                let controls = [*control1, *control2, *control3];
                apply_multi_controlled_x_kernel_seq(state, &controls, *target, n);
            } else {
                apply_n_qubit_seq(
                    state,
                    &m16(gates::c3x()),
                    &[*control1, *control2, *control3, *target],
                    n,
                );
            }
        }
        Instruction::C3sqrtx {
            control1,
            control2,
            control3,
            target,
        } => {
            if statevector_multi_control_kernels_enabled_for(n) {
                let sx = gates::sx();
                let controls = [(*control1, true), (*control2, true), (*control3, true)];
                apply_one_qubit_seq(state, &sx, *target, n, &controls);
            } else {
                apply_n_qubit_seq(
                    state,
                    &m16(gates::c3sqrtx()),
                    &[*control1, *control2, *control3, *target],
                    n,
                );
            }
        }

        // -- Five-qubit --------------------------------------------------
        Instruction::C4x {
            control1,
            control2,
            control3,
            control4,
            target,
        } => {
            if statevector_multi_control_kernels_enabled_for(n) {
                let controls = [*control1, *control2, *control3, *control4];
                apply_multi_controlled_x_kernel_seq(state, &controls, *target, n);
            } else {
                apply_n_qubit_seq(
                    state,
                    &m32(gates::c4x()),
                    &[*control1, *control2, *control3, *control4, *target],
                    n,
                );
            }
        }

        // -- Cross-node / generic ----------------------------------------
        Instruction::Gate { name, qubits, .. } => {
            match name.to_lowercase().as_str() {
                "remote_link_phi_plus" => {
                    apply_n_qubit_seq(state, &m4(gates::phi_plus()), qubits, n);
                }
                "remote_link_psi_minus" => {
                    apply_n_qubit_seq(state, &m4(gates::psi_minus()), qubits, n);
                }
                "remote_link_psi_plus" => {
                    apply_n_qubit_seq(state, &m4(gates::psi_plus()), qubits, n);
                }
                "nonlocal_cz" | "remote_cz" => {
                    apply_n_qubit_seq(state, &m4(gates::cz()), qubits, n);
                }
                "remote_cx" => {
                    apply_n_qubit_seq(state, &m4(gates::cnot()), qubits, n);
                }
                "remote_epr" => {
                    apply_n_qubit_seq(state, &m4(gates::phi_plus()), qubits, n);
                }
                "remote_barrier" => {
                    // no-op
                }
                "remote_cu1" => {
                    return Err(
                        "Symbolic 'remote_cu1' cannot be simulated natively. Distribute with lowered=True."
                            .to_string(),
                    );
                }
                other if other.starts_with("circuit-") => {
                    return Err(format!(
                        "Opaque symbolic subcircuit {other:?} cannot be simulated natively. Distribute with lowered=True."
                    ));
                }
                other => {
                    return Err(format!(
                        "Unsupported generic gate: {other:?}. Decompose it before simulating."
                    ));
                }
            }
        }

        // -- Measurement -------------------------------------------------
        Instruction::Measure { qubit, cbit } => {
            let outcome = measure_qubit_seq(state, *qubit, n, rng);
            cbits.insert(*cbit, outcome as i32);
        }

        // -- Classical control -------------------------------------------
        Instruction::Conditional { condition, op } => {
            let mut actual: u64 = 0;
            for bit in 0..condition.creg_size {
                let val = *cbits.get(&(condition.creg_base + bit)).unwrap_or(&0) as u64;
                actual |= val << bit;
            }
            if actual == condition.creg_value {
                run_instruction_par(state, op, n, cbits, rng)?;
            }
        }

        // -- Reset -------------------------------------------------------
        Instruction::Reset { qubit } => {
            let outcome = measure_qubit_seq(state, *qubit, n, rng);
            if outcome == 1 {
                apply_one_qubit_seq(state, &gates::X, *qubit, n, &[]);
            }
        }

        // -- No-ops ------------------------------------------------------
        Instruction::Barrier | Instruction::Classical { .. } => {}
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Fused instruction dispatcher (used in parallel shot loops)
// ---------------------------------------------------------------------------

fn run_fused_par(
    state: &mut [C],
    fi: &FusedInstruction<'_>,
    n: usize,
    cbits: &mut HashMap<usize, i32>,
    rng: &mut impl Rng,
) -> Result<(), String> {
    match fi {
        FusedInstruction::Fused1Q { qubit, matrix } => {
            apply_one_qubit_seq(state, matrix, *qubit, n, &[]);
            Ok(())
        }
        FusedInstruction::Original(inst) => run_instruction_par(state, inst, n, cbits, rng),
    }
}

// ---------------------------------------------------------------------------
// Timed engine wrappers
// ---------------------------------------------------------------------------

#[inline]
fn do_oq(state: &mut [C], u: &[[C; 2]; 2], target: usize, n: usize, acc: &mut Option<ProfileAcc>) {
    match acc {
        None => apply_one_qubit(state, u, target, n, &[]),
        Some(a) => {
            let t = Instant::now();
            apply_one_qubit(state, u, target, n, &[]);
            a.oq_time += t.elapsed().as_secs_f64();
            a.oq_calls += 1;
        }
    }
}

#[inline]
fn do_nq(state: &mut [C], u: &[Vec<C>], qubits: &[usize], n: usize, acc: &mut Option<ProfileAcc>) {
    match acc {
        None => apply_n_qubit(state, u, qubits, n),
        Some(a) => {
            let t = Instant::now();
            apply_n_qubit(state, u, qubits, n);
            a.nq_time += t.elapsed().as_secs_f64();
            a.nq_calls += 1;
        }
    }
}

#[inline]
fn do_controlled_oq_kernel(
    state: &mut [C],
    u: &[[C; 2]; 2],
    control: usize,
    target: usize,
    n: usize,
    acc: &mut Option<ProfileAcc>,
) {
    let controls = [(control, true)];
    do_multi_controlled_oq_kernel(state, u, target, n, &controls, acc);
}

#[inline]
fn do_multi_controlled_oq_kernel(
    state: &mut [C],
    u: &[[C; 2]; 2],
    target: usize,
    n: usize,
    controls: &[(usize, bool)],
    acc: &mut Option<ProfileAcc>,
) {
    do_statevector_two_qubit_kernel(state, acc, |state| {
        apply_one_qubit(state, u, target, n, controls)
    });
}

#[inline]
fn do_multi_controlled_x_kernel(
    state: &mut [C],
    target: usize,
    n: usize,
    controls: &[usize],
    acc: &mut Option<ProfileAcc>,
) {
    do_statevector_two_qubit_kernel(state, acc, |state| {
        apply_multi_controlled_x_kernel(state, controls, target, n)
    });
}

#[inline]
fn do_cx_kernel(
    state: &mut [C],
    control: usize,
    target: usize,
    n: usize,
    acc: &mut Option<ProfileAcc>,
) {
    do_statevector_two_qubit_kernel(state, acc, |state| {
        apply_cx_kernel(state, control, target, n)
    });
}

#[inline]
fn do_cz_kernel(
    state: &mut [C],
    control: usize,
    target: usize,
    n: usize,
    acc: &mut Option<ProfileAcc>,
) {
    do_statevector_two_qubit_kernel(state, acc, |state| {
        apply_cz_kernel(state, control, target, n)
    });
}

#[inline]
fn do_swap_kernel(state: &mut [C], a: usize, b: usize, n: usize, acc: &mut Option<ProfileAcc>) {
    do_statevector_two_qubit_kernel(state, acc, |state| apply_swap_kernel(state, a, b, n));
}

#[inline]
fn do_cswap_kernel(
    state: &mut [C],
    control: usize,
    a: usize,
    b: usize,
    n: usize,
    acc: &mut Option<ProfileAcc>,
) {
    do_statevector_two_qubit_kernel(state, acc, |state| {
        apply_cswap_kernel(state, control, a, b, n)
    });
}

#[inline]
fn do_statevector_two_qubit_kernel<F>(state: &mut [C], acc: &mut Option<ProfileAcc>, op: F)
where
    F: FnOnce(&mut [C]),
{
    match acc {
        None => op(state),
        Some(a) => {
            let t = Instant::now();
            op(state);
            a.nq_time += t.elapsed().as_secs_f64();
            a.nq_calls += 1;
        }
    }
}
#[inline]
fn do_mq(
    state: &mut [C],
    qubit: usize,
    n: usize,
    rng: &mut impl Rng,
    acc: &mut Option<ProfileAcc>,
) -> u8 {
    match acc {
        None => measure_qubit(state, qubit, n, rng),
        Some(a) => {
            let t = Instant::now();
            let outcome = measure_qubit(state, qubit, n, rng);
            a.mq_time += t.elapsed().as_secs_f64();
            a.mq_calls += 1;
            outcome
        }
    }
}

// ---------------------------------------------------------------------------
// Matrix-to-Vec converters (fixed-size arrays Ã¢â€ â€™ Vec<Vec<C>> for apply_n_qubit)
// ---------------------------------------------------------------------------

fn m4(m: [[C; 4]; 4]) -> Vec<Vec<C>> {
    m.iter().map(|row| row.to_vec()).collect()
}
fn m8(m: [[C; 8]; 8]) -> Vec<Vec<C>> {
    m.iter().map(|row| row.to_vec()).collect()
}
fn m16(m: [[C; 16]; 16]) -> Vec<Vec<C>> {
    m.iter().map(|row| row.to_vec()).collect()
}
fn m32(m: [[C; 32]; 32]) -> Vec<Vec<C>> {
    m.iter().map(|row| row.to_vec()).collect()
}
