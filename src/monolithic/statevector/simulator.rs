use std::collections::HashMap;
use std::time::Instant;

use num_complex::Complex64;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

use crate::engine::{
    apply_n_qubit, apply_n_qubit_seq, apply_one_qubit, apply_one_qubit_seq, marginal_probs,
    measure_qubit, measure_qubit_seq, sample_counts,
};
use crate::gates;
use crate::profiling::ShotLoopProfiler;
use crate::types::{format_cbits, fuse_circuit, Circuit, FusedInstruction, Instruction};

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
    /// measurement per shot. Returns a dict[str, int] of bitstring counts — the true
    /// shot distribution including classically-conditioned corrections.
    #[pyo3(signature = (circuit, shots=1000))]
    pub fn simulate_shots(
        &self,
        py: Python,
        circuit: &Bound<PyAny>,
        shots: usize,
    ) -> PyResult<PyObject> {
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

        let shot_loop_profiler = ShotLoopProfiler::start("statevector", n, shots, fused.len());
        let counts = py
            .allow_threads(|| -> Result<HashMap<String, usize>, String> {
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
            })
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
        if let Some(profiler) = shot_loop_profiler {
            profiler.finish();
        }

        let d = PyDict::new_bound(py);
        for (k, v) in &counts {
            d.set_item(k, v)?;
        }
        Ok(d.into())
    }

    /// Run the circuit and return a SimulationResult.
    pub fn simulate(&self, py: Python, circuit: &Bound<PyAny>) -> PyResult<SimulationResult> {
        // 1. Serialize entire circuit to JSON (one boundary crossing)
        let json_str: String = circuit.call_method0("model_dump_json")?.extract()?;

        // 2. Deserialize in Rust — no more Python calls until we return
        let rust_circuit: Circuit = serde_json::from_str(&json_str).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Circuit JSON parse error: {e}"))
        })?;

        // 3. Initialise statevector |0...0⟩
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
            do_nq(state, &m4(gates::cnot()), &[*control, *target], n, acc);
        }
        Instruction::Cz { control, target } => {
            do_nq(state, &m4(gates::cz()), &[*control, *target], n, acc);
        }
        Instruction::Cy { control, target } => {
            do_nq(state, &m4(gates::cy()), &[*control, *target], n, acc);
        }
        Instruction::Ch { control, target } => {
            do_nq(state, &m4(gates::ch()), &[*control, *target], n, acc);
        }
        Instruction::Swap { a, b } => {
            do_nq(state, &m4(gates::swap()), &[*a, *b], n, acc);
        }
        Instruction::Csx { control, target } => {
            do_nq(state, &m4(gates::csx()), &[*control, *target], n, acc);
        }

        // -- Two-qubit parametric ----------------------------------------
        Instruction::Crx {
            control,
            target,
            theta,
        } => {
            do_nq(state, &m4(gates::crx(*theta)), &[*control, *target], n, acc);
        }
        Instruction::Cry {
            control,
            target,
            theta,
        } => {
            do_nq(state, &m4(gates::cry(*theta)), &[*control, *target], n, acc);
        }
        Instruction::Crz {
            control,
            target,
            lam,
        } => {
            do_nq(state, &m4(gates::crz(*lam)), &[*control, *target], n, acc);
        }
        Instruction::Cu1 {
            control,
            target,
            lam,
        } => {
            do_nq(state, &m4(gates::cu1(*lam)), &[*control, *target], n, acc);
        }
        Instruction::Cp {
            control,
            target,
            lam,
        } => {
            do_nq(state, &m4(gates::cp(*lam)), &[*control, *target], n, acc);
        }
        Instruction::Cu3 {
            control,
            target,
            theta,
            phi,
            lam,
        } => {
            do_nq(
                state,
                &m4(gates::cu3(*theta, *phi, *lam)),
                &[*control, *target],
                n,
                acc,
            );
        }
        Instruction::Cu {
            control,
            target,
            theta,
            phi,
            lam,
            gamma,
        } => {
            do_nq(
                state,
                &m4(gates::cu(*theta, *phi, *lam, *gamma)),
                &[*control, *target],
                n,
                acc,
            );
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
            do_nq(
                state,
                &m8(gates::ccx()),
                &[*control1, *control2, *target],
                n,
                acc,
            );
        }
        Instruction::Cswap {
            control,
            target1,
            target2,
        } => {
            do_nq(
                state,
                &m8(gates::cswap()),
                &[*control, *target1, *target2],
                n,
                acc,
            );
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
            do_nq(
                state,
                &m16(gates::c3x()),
                &[*control1, *control2, *control3, *target],
                n,
                acc,
            );
        }
        Instruction::C3sqrtx {
            control1,
            control2,
            control3,
            target,
        } => {
            do_nq(
                state,
                &m16(gates::c3sqrtx()),
                &[*control1, *control2, *control3, *target],
                n,
                acc,
            );
        }

        // -- Five-qubit --------------------------------------------------
        Instruction::C4x {
            control1,
            control2,
            control3,
            control4,
            target,
        } => {
            do_nq(
                state,
                &m32(gates::c4x()),
                &[*control1, *control2, *control3, *control4, *target],
                n,
                acc,
            );
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
                "remote_barrier" | "remote_cu1" => {
                    // no-op
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
            apply_n_qubit_seq(state, &m4(gates::cnot()), &[*control, *target], n);
        }
        Instruction::Cz { control, target } => {
            apply_n_qubit_seq(state, &m4(gates::cz()), &[*control, *target], n);
        }
        Instruction::Cy { control, target } => {
            apply_n_qubit_seq(state, &m4(gates::cy()), &[*control, *target], n);
        }
        Instruction::Ch { control, target } => {
            apply_n_qubit_seq(state, &m4(gates::ch()), &[*control, *target], n);
        }
        Instruction::Swap { a, b } => {
            apply_n_qubit_seq(state, &m4(gates::swap()), &[*a, *b], n);
        }
        Instruction::Csx { control, target } => {
            apply_n_qubit_seq(state, &m4(gates::csx()), &[*control, *target], n);
        }

        // -- Two-qubit parametric ----------------------------------------
        Instruction::Crx {
            control,
            target,
            theta,
        } => {
            apply_n_qubit_seq(state, &m4(gates::crx(*theta)), &[*control, *target], n);
        }
        Instruction::Cry {
            control,
            target,
            theta,
        } => {
            apply_n_qubit_seq(state, &m4(gates::cry(*theta)), &[*control, *target], n);
        }
        Instruction::Crz {
            control,
            target,
            lam,
        } => {
            apply_n_qubit_seq(state, &m4(gates::crz(*lam)), &[*control, *target], n);
        }
        Instruction::Cu1 {
            control,
            target,
            lam,
        } => {
            apply_n_qubit_seq(state, &m4(gates::cu1(*lam)), &[*control, *target], n);
        }
        Instruction::Cp {
            control,
            target,
            lam,
        } => {
            apply_n_qubit_seq(state, &m4(gates::cp(*lam)), &[*control, *target], n);
        }
        Instruction::Cu3 {
            control,
            target,
            theta,
            phi,
            lam,
        } => {
            apply_n_qubit_seq(
                state,
                &m4(gates::cu3(*theta, *phi, *lam)),
                &[*control, *target],
                n,
            );
        }
        Instruction::Cu {
            control,
            target,
            theta,
            phi,
            lam,
            gamma,
        } => {
            apply_n_qubit_seq(
                state,
                &m4(gates::cu(*theta, *phi, *lam, *gamma)),
                &[*control, *target],
                n,
            );
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
            apply_n_qubit_seq(
                state,
                &m8(gates::ccx()),
                &[*control1, *control2, *target],
                n,
            );
        }
        Instruction::Cswap {
            control,
            target1,
            target2,
        } => {
            apply_n_qubit_seq(
                state,
                &m8(gates::cswap()),
                &[*control, *target1, *target2],
                n,
            );
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
            apply_n_qubit_seq(
                state,
                &m16(gates::c3x()),
                &[*control1, *control2, *control3, *target],
                n,
            );
        }
        Instruction::C3sqrtx {
            control1,
            control2,
            control3,
            target,
        } => {
            apply_n_qubit_seq(
                state,
                &m16(gates::c3sqrtx()),
                &[*control1, *control2, *control3, *target],
                n,
            );
        }

        // -- Five-qubit --------------------------------------------------
        Instruction::C4x {
            control1,
            control2,
            control3,
            control4,
            target,
        } => {
            apply_n_qubit_seq(
                state,
                &m32(gates::c4x()),
                &[*control1, *control2, *control3, *control4, *target],
                n,
            );
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
                "remote_barrier" | "remote_cu1" => {
                    // no-op
                }
                other if other.starts_with("circuit-") => {
                    // opaque Qiskit subcircuit — no-op
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
// Matrix-to-Vec converters (fixed-size arrays → Vec<Vec<C>> for apply_n_qubit)
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
