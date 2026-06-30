use std::collections::HashMap;
use std::time::Instant;

use nalgebra::DMatrix;
use num_complex::Complex64;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::gates;
use crate::monolithic::statevector::SimulationResult;
use crate::profiling::{ShotLoopProfiler, ShotsProfile};
use crate::types::{format_cbits, Circuit, Instruction};

type C = Complex64;

#[derive(Clone)]
struct Tensor {
    left: usize,
    right: usize,
    data: Vec<C>,
}

impl Tensor {
    fn zero(left: usize, right: usize) -> Self {
        Self {
            left,
            right,
            data: vec![C::new(0.0, 0.0); left * 2 * right],
        }
    }

    #[inline]
    fn idx(&self, left: usize, state: usize, right: usize) -> usize {
        (left * 2 + state) * self.right + right
    }

    #[inline]
    fn get(&self, left: usize, state: usize, right: usize) -> C {
        self.data[self.idx(left, state, right)]
    }

    #[inline]
    fn set(&mut self, left: usize, state: usize, right: usize, value: C) {
        let idx = self.idx(left, state, right);
        self.data[idx] = value;
    }
}

struct Mps {
    tensors: Vec<Tensor>,
    max_bond_dimension: Option<usize>,
    truncation_threshold: f64,
}

impl Mps {
    fn new(
        num_qubits: usize,
        max_bond_dimension: Option<usize>,
        truncation_threshold: f64,
    ) -> Self {
        let mut tensors = Vec::with_capacity(num_qubits);
        for _ in 0..num_qubits {
            let mut tensor = Tensor::zero(1, 1);
            tensor.set(0, 0, 0, C::new(1.0, 0.0));
            tensors.push(tensor);
        }
        Self {
            tensors,
            max_bond_dimension,
            truncation_threshold,
        }
    }

    fn apply_1q(&mut self, qubit: usize, mat: &[[C; 2]; 2]) {
        let old = self.tensors[qubit].clone();
        let mut new = Tensor::zero(old.left, old.right);
        for left in 0..old.left {
            for right in 0..old.right {
                for (out, row) in mat.iter().enumerate() {
                    let mut acc = C::new(0.0, 0.0);
                    for (input, gate_element) in row.iter().enumerate() {
                        acc += *gate_element * old.get(left, input, right);
                    }
                    new.set(left, out, right, acc);
                }
            }
        }
        self.tensors[qubit] = new;
    }

    fn apply_2q(&mut self, a: usize, b: usize, mat: &[[C; 4]; 4]) -> PyResult<()> {
        if a == b {
            return Ok(());
        }
        let (lo, hi, reversed) = if a < b { (a, b, false) } else { (b, a, true) };

        for pos in ((lo + 1)..hi).rev() {
            self.apply_adjacent_2q(pos, &gates::swap())?;
        }

        let gate = if reversed {
            reverse_2q_order(mat)
        } else {
            *mat
        };
        self.apply_adjacent_2q(lo, &gate)?;

        for pos in (lo + 1)..hi {
            self.apply_adjacent_2q(pos, &gates::swap())?;
        }
        Ok(())
    }

    fn apply_adjacent_2q(&mut self, q: usize, mat: &[[C; 4]; 4]) -> PyResult<()> {
        let left_tensor = self.tensors[q].clone();
        let right_tensor = self.tensors[q + 1].clone();
        if left_tensor.right != right_tensor.left {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Invalid MPS bond dimensions",
            ));
        }

        let left_dim = left_tensor.left;
        let bond_dim = left_tensor.right;
        let right_dim = right_tensor.right;
        let mut theta = DMatrix::<C>::zeros(left_dim * 2, 2 * right_dim);

        for left in 0..left_dim {
            for right in 0..right_dim {
                for out0 in 0..2 {
                    for out1 in 0..2 {
                        let mut acc = C::new(0.0, 0.0);
                        let out_idx = out0 * 2 + out1;
                        for in0 in 0..2 {
                            for in1 in 0..2 {
                                let in_idx = in0 * 2 + in1;
                                let mut input_amp = C::new(0.0, 0.0);
                                for bond in 0..bond_dim {
                                    input_amp += left_tensor.get(left, in0, bond)
                                        * right_tensor.get(bond, in1, right);
                                }
                                acc += mat[out_idx][in_idx] * input_amp;
                            }
                        }
                        theta[(left * 2 + out0, out1 * right_dim + right)] = acc;
                    }
                }
            }
        }

        let svd = theta.svd(true, true);
        let u = svd
            .u
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("MPS SVD did not return U"))?;
        let vt = svd.v_t.ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("MPS SVD did not return Vt")
        })?;

        let mut keep = svd
            .singular_values
            .iter()
            .filter(|&&s| s > self.truncation_threshold)
            .count()
            .max(1);
        if let Some(max_bond) = self.max_bond_dimension {
            keep = keep.min(max_bond.max(1));
        }
        let kept_norm = svd
            .singular_values
            .iter()
            .take(keep)
            .map(|s| s * s)
            .sum::<f64>()
            .sqrt()
            .max(1e-15);
        let total_norm = svd
            .singular_values
            .iter()
            .map(|s| s * s)
            .sum::<f64>()
            .sqrt();
        let discarded_weight = svd
            .singular_values
            .iter()
            .skip(keep)
            .map(|s| s * s)
            .sum::<f64>();
        let truncation_scale = if discarded_weight > 0.0 {
            total_norm / kept_norm
        } else {
            1.0
        };

        let mut new_left = Tensor::zero(left_dim, keep);
        let mut new_right = Tensor::zero(keep, right_dim);
        for left in 0..left_dim {
            for state in 0..2 {
                let row = left * 2 + state;
                for bond in 0..keep {
                    new_left.set(left, state, bond, u[(row, bond)]);
                }
            }
        }
        for bond in 0..keep {
            let sigma = C::new(svd.singular_values[bond] * truncation_scale, 0.0);
            for state in 0..2 {
                for right in 0..right_dim {
                    let col = state * right_dim + right;
                    new_right.set(bond, state, right, sigma * vt[(bond, col)]);
                }
            }
        }

        self.tensors[q] = new_left;
        self.tensors[q + 1] = new_right;
        Ok(())
    }

    fn to_statevector(&self) -> Vec<C> {
        let num_qubits = self.tensors.len();
        let mut state = vec![C::new(0.0, 0.0); 1 << num_qubits];
        for (basis, amp) in state.iter_mut().enumerate() {
            let mut work = vec![C::new(1.0, 0.0)];
            for (qubit, tensor) in self.tensors.iter().enumerate() {
                let bit = (basis >> qubit) & 1;
                let mut next = vec![C::new(0.0, 0.0); tensor.right];
                for (left, left_amp) in work.iter().enumerate().take(tensor.left) {
                    for (right, next_amp) in next.iter_mut().enumerate().take(tensor.right) {
                        *next_amp += *left_amp * tensor.get(left, bit, right);
                    }
                }
                work = next;
            }
            *amp = work[0];
        }
        state
    }

    fn measure(&mut self, qubit: usize, rng: &mut impl Rng) -> usize {
        let state = self.to_statevector();
        let p1: f64 = state
            .iter()
            .enumerate()
            .filter(|(idx, _)| ((*idx >> qubit) & 1) == 1)
            .map(|(_, amp)| amp.norm_sqr())
            .sum();
        let p1 = p1.clamp(0.0, 1.0);
        let outcome = if rng.gen::<f64>() < p1 { 1 } else { 0 };
        let prob = if outcome == 1 { p1 } else { 1.0 - p1 };
        self.project_qubit(qubit, outcome, prob);
        outcome
    }

    fn project_qubit(&mut self, qubit: usize, outcome: usize, prob: f64) {
        let scale = if prob > 0.0 { 1.0 / prob.sqrt() } else { 0.0 };
        let tensor = &mut self.tensors[qubit];
        for left in 0..tensor.left {
            for state in 0..2 {
                for right in 0..tensor.right {
                    let value = if state == outcome {
                        tensor.get(left, state, right) * C::new(scale, 0.0)
                    } else {
                        C::new(0.0, 0.0)
                    };
                    tensor.set(left, state, right, value);
                }
            }
        }
    }
}

#[pyclass]
pub struct MpsSimulator {
    seed: Option<u64>,
    max_bond_dimension: Option<usize>,
    truncation_threshold: f64,
}

#[pymethods]
impl MpsSimulator {
    #[new]
    #[pyo3(signature = (seed=None, max_bond_dimension=None, truncation_threshold=1e-12))]
    pub fn new(
        seed: Option<u64>,
        max_bond_dimension: Option<usize>,
        truncation_threshold: f64,
    ) -> Self {
        Self {
            seed,
            max_bond_dimension,
            truncation_threshold,
        }
    }

    pub fn simulate(&self, _py: Python, circuit: &Bound<PyAny>) -> PyResult<SimulationResult> {
        let json_str: String = circuit.call_method0("model_dump_json")?.extract()?;
        let rust_circuit: Circuit = serde_json::from_str(&json_str).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Circuit JSON parse error: {e}"))
        })?;

        let mut mps = Mps::new(
            rust_circuit.num_qubits(),
            self.max_bond_dimension,
            self.truncation_threshold,
        );
        let mut cbits: HashMap<usize, i32> = HashMap::new();
        let seed = self.seed.unwrap_or_else(|| rand::thread_rng().gen());
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        for inst in &rust_circuit.instructions {
            run_instruction(&mut mps, inst, &mut cbits, &mut rng)?;
        }

        Ok(SimulationResult::new(
            mps.to_statevector(),
            rust_circuit.num_qubits(),
            cbits,
            None,
        ))
    }

    #[pyo3(signature = (circuit, shots=1000, collect_profile=false))]
    pub fn simulate_shots(
        &self,
        py: Python,
        circuit: &Bound<PyAny>,
        shots: usize,
        collect_profile: bool,
    ) -> PyResult<PyObject> {
        let total_t0 = Instant::now();
        let preprocessing_t0 = Instant::now();
        let json_str: String = circuit.call_method0("model_dump_json")?.extract()?;
        let rust_circuit: Circuit = serde_json::from_str(&json_str).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Circuit JSON parse error: {e}"))
        })?;

        let num_qubits = rust_circuit.num_qubits();
        let num_instructions = rust_circuit.instructions.len();
        let num_cbits = rust_circuit.num_cbits();
        let base_seed = self.seed.unwrap_or_else(|| rand::thread_rng().gen());
        let preprocessing_ms = preprocessing_t0.elapsed().as_secs_f64() * 1000.0;

        let shot_loop_profiler =
            ShotLoopProfiler::start("mps", num_qubits, shots, num_instructions);
        let exec_t0 = Instant::now();
        let mut counts: HashMap<String, usize> = HashMap::new();
        for shot in 0..shots {
            let mut mps = Mps::new(
                num_qubits,
                self.max_bond_dimension,
                self.truncation_threshold,
            );
            let mut cbits: HashMap<usize, i32> = HashMap::new();
            let mut rng = ChaCha8Rng::seed_from_u64(base_seed.wrapping_add(shot as u64));
            for inst in &rust_circuit.instructions {
                run_instruction(&mut mps, inst, &mut cbits, &mut rng)?;
            }
            let key = format_cbits(&cbits, num_cbits);
            *counts.entry(key).or_insert(0) += 1;
        }
        let parallel_execution_ms = exec_t0.elapsed().as_secs_f64() * 1000.0;
        if let Some(profiler) = shot_loop_profiler {
            profiler.finish();
        }

        let d = PyDict::new_bound(py);
        for (key, value) in &counts {
            d.set_item(key, value)?;
        }

        if collect_profile {
            let profile = ShotsProfile {
                num_shots: shots,
                num_qubits,
                num_instructions,
                preprocessing_ms,
                gate_fusion_ms: 0.0,
                parallel_execution_ms,
                per_shot_stats: None,
                total_time_ms: total_t0.elapsed().as_secs_f64() * 1000.0,
            };

            let profile_json_str = profile.to_json_string().map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Profile serialization error: {e}"
                ))
            })?;
            let json_module = py.import_bound("json")?;
            let profile_dict = json_module.call_method1("loads", (&profile_json_str,))?;

            let result_dict = PyDict::new_bound(py);
            result_dict.set_item("counts", &d)?;
            result_dict.set_item("profile", &profile_dict)?;
            return Ok(result_dict.into());
        }

        Ok(d.into())
    }
}

fn run_instruction(
    mps: &mut Mps,
    inst: &Instruction,
    cbits: &mut HashMap<usize, i32>,
    rng: &mut impl Rng,
) -> PyResult<()> {
    match inst {
        Instruction::Id { .. }
        | Instruction::U0 { .. }
        | Instruction::Barrier
        | Instruction::Classical { .. } => {}
        Instruction::X { qubit } => mps.apply_1q(*qubit, &gates::X),
        Instruction::Y { qubit } => mps.apply_1q(*qubit, &gates::Y),
        Instruction::Z { qubit } => mps.apply_1q(*qubit, &gates::Z),
        Instruction::H { qubit } => mps.apply_1q(*qubit, &gates::h()),
        Instruction::S { qubit } => mps.apply_1q(*qubit, &gates::s_gate()),
        Instruction::Sdg { qubit } => mps.apply_1q(*qubit, &gates::sdg()),
        Instruction::T { qubit } => mps.apply_1q(*qubit, &gates::t_gate()),
        Instruction::Tdg { qubit } => mps.apply_1q(*qubit, &gates::tdg()),
        Instruction::Sx { qubit } => mps.apply_1q(*qubit, &gates::sx()),
        Instruction::Sxdg { qubit } => mps.apply_1q(*qubit, &gates::sxdg()),
        Instruction::U3 {
            qubit,
            theta,
            phi,
            lam,
        }
        | Instruction::U {
            qubit,
            theta,
            phi,
            lam,
        } => mps.apply_1q(*qubit, &gates::u3(*theta, *phi, *lam)),
        Instruction::U2 { qubit, phi, lam } => mps.apply_1q(*qubit, &gates::u2(*phi, *lam)),
        Instruction::U1 { qubit, lam } | Instruction::P { qubit, lam } => {
            mps.apply_1q(*qubit, &gates::u1(*lam));
        }
        Instruction::Rx { qubit, theta } => mps.apply_1q(*qubit, &gates::rx(*theta)),
        Instruction::Ry { qubit, theta } => mps.apply_1q(*qubit, &gates::ry(*theta)),
        Instruction::Rz { qubit, phi } => mps.apply_1q(*qubit, &gates::rz(*phi)),
        Instruction::Cx { control, target } => mps.apply_2q(*control, *target, &gates::cnot())?,
        Instruction::Cz { control, target } => mps.apply_2q(*control, *target, &gates::cz())?,
        Instruction::Cy { control, target } => mps.apply_2q(*control, *target, &gates::cy())?,
        Instruction::Ch { control, target } => mps.apply_2q(*control, *target, &gates::ch())?,
        Instruction::Swap { a, b } => mps.apply_2q(*a, *b, &gates::swap())?,
        Instruction::Csx { control, target } => mps.apply_2q(*control, *target, &gates::csx())?,
        Instruction::Crx {
            control,
            target,
            theta,
        } => mps.apply_2q(*control, *target, &gates::crx(*theta))?,
        Instruction::Cry {
            control,
            target,
            theta,
        } => mps.apply_2q(*control, *target, &gates::cry(*theta))?,
        Instruction::Crz {
            control,
            target,
            lam,
        } => mps.apply_2q(*control, *target, &gates::crz(*lam))?,
        Instruction::Cu1 {
            control,
            target,
            lam,
        }
        | Instruction::Cp {
            control,
            target,
            lam,
        } => mps.apply_2q(*control, *target, &gates::cu1(*lam))?,
        Instruction::Cu3 {
            control,
            target,
            theta,
            phi,
            lam,
        } => mps.apply_2q(*control, *target, &gates::cu3(*theta, *phi, *lam))?,
        Instruction::Cu {
            control,
            target,
            theta,
            phi,
            lam,
            gamma,
        } => mps.apply_2q(*control, *target, &gates::cu(*theta, *phi, *lam, *gamma))?,
        Instruction::Rxx { a, b, theta } => mps.apply_2q(*a, *b, &gates::rxx(*theta))?,
        Instruction::Rzz { a, b, theta } => mps.apply_2q(*a, *b, &gates::rzz(*theta))?,
        Instruction::Gate { name, qubits, .. } => match name.to_lowercase().as_str() {
            "remote_link_phi_plus" | "remote_epr" | "epr" => {
                mps.apply_2q(qubits[0], qubits[1], &gates::phi_plus())?;
            }
            "remote_link_psi_minus" => {
                mps.apply_2q(qubits[0], qubits[1], &gates::psi_minus())?;
            }
            "remote_link_psi_plus" => {
                mps.apply_2q(qubits[0], qubits[1], &gates::psi_plus())?;
            }
            "nonlocal_cz" | "remote_cz" => {
                mps.apply_2q(qubits[0], qubits[1], &gates::cz())?;
            }
            "remote_cx" => {
                mps.apply_2q(qubits[0], qubits[1], &gates::cnot())?;
            }
            "remote_barrier" => {}
            "remote_cu1" => {
                return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                    "Symbolic 'remote_cu1' cannot be simulated natively. Distribute with lowered=True.",
                ));
            }
            other if other.starts_with("circuit-") => {
                return Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
                    "Opaque symbolic subcircuit {other:?} cannot be simulated natively. Distribute with lowered=True."
                )));
            }
            other => {
                return Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
                    "MPS simulator does not support generic gate {other:?}"
                )));
            }
        },
        Instruction::Measure { qubit, cbit } => {
            let outcome = mps.measure(*qubit, rng);
            cbits.insert(*cbit, outcome as i32);
        }
        Instruction::Reset { qubit } => {
            let outcome = mps.measure(*qubit, rng);
            if outcome == 1 {
                mps.apply_1q(*qubit, &gates::X);
            }
        }
        Instruction::Conditional { condition, op } => {
            let mut actual: u64 = 0;
            for bit in 0..condition.creg_size {
                let val = *cbits.get(&(condition.creg_base + bit)).unwrap_or(&0) as u64;
                actual |= val << bit;
            }
            if actual == condition.creg_value {
                run_instruction(mps, op, cbits, rng)?;
            }
        }
        _ => {
            return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "MPS simulator only supports one- and two-qubit unitary gates plus measurement, reset, and conditionals",
            ));
        }
    }
    Ok(())
}

fn reverse_2q_order(mat: &[[C; 4]; 4]) -> [[C; 4]; 4] {
    let mut out = [[C::new(0.0, 0.0); 4]; 4];
    for a_out in 0..2 {
        for b_out in 0..2 {
            for a_in in 0..2 {
                for b_in in 0..2 {
                    let row = a_out * 2 + b_out;
                    let col = a_in * 2 + b_in;
                    let swapped_row = b_out * 2 + a_out;
                    let swapped_col = b_in * 2 + a_in;
                    out[row][col] = mat[swapped_row][swapped_col];
                }
            }
        }
    }
    out
}
