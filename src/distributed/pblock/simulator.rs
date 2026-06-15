use std::collections::{HashMap, HashSet};

use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

use crate::engine::{
    apply_n_qubit, apply_n_qubit_seq, apply_one_qubit, apply_one_qubit_seq, marginal_probs,
    measure_qubit, measure_qubit_seq,
};
use crate::gates;
use crate::profiling::ShotLoopProfiler;
use crate::types::{format_cbits, fuse_pblock_entries, Circuit, FusedPBlockEntry, Instruction};

use super::model::{m16, m32, m4, m8, Block, BlockPool, C};

// ---------------------------------------------------------------------------
// PBlockResult
// ---------------------------------------------------------------------------

#[pyclass]
pub struct PBlockResult {
    sv: Vec<C>,
    num_qubits: usize,
    phys_qubits: Vec<usize>,
    cbits: HashMap<usize, i32>,
}

#[pymethods]
impl PBlockResult {
    /// Raw complex amplitudes as a NumPy array. Local bit i = physical qubit phys_qubits[i].
    #[getter]
    fn statevector<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<C>> {
        self.sv.clone().into_pyarray_bound(py)
    }

    #[getter]
    fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Physical qubit indices present in the result, sorted ascending.
    /// Local bit i in the statevector corresponds to physical qubit phys_qubits[i].
    #[getter]
    fn physical_qubits(&self) -> Vec<usize> {
        self.phys_qubits.clone()
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

    /// Probability distribution over physical qubits.
    ///
    /// `qubits`: physical qubit indices, qubits[0] = MSB of the output key.
    /// Default: all physical qubits in descending order, so bit 0 of the key
    /// corresponds to physical qubit 0 — matching StatevectorSimulator.probabilities().
    #[pyo3(signature = (qubits=None))]
    fn probabilities(&self, py: Python, qubits: Option<Vec<usize>>) -> PyObject {
        let phys: Vec<usize> =
            qubits.unwrap_or_else(|| self.phys_qubits.iter().rev().cloned().collect());
        let local: Vec<usize> = phys
            .iter()
            .map(|&q| {
                self.phys_qubits
                    .iter()
                    .position(|&pq| pq == q)
                    .unwrap_or_else(|| panic!("qubit {} not in result", q))
            })
            .collect();
        let probs = marginal_probs(&self.sv, self.num_qubits, &local);
        let d = PyDict::new_bound(py);
        for (j, p) in probs.iter().enumerate() {
            if *p > 0.0 {
                d.set_item(j, p).unwrap();
            }
        }
        d.into()
    }
}

// ---------------------------------------------------------------------------
// PBlockSimulator
// ---------------------------------------------------------------------------

#[pyclass]
pub struct PBlockSimulator {
    seed: Option<u64>,
}

#[pymethods]
impl PBlockSimulator {
    #[new]
    #[pyo3(signature = (seed=None))]
    pub fn new(seed: Option<u64>) -> Self {
        Self { seed }
    }

    /// Run the distributed circuit `shots` times independently, yielding a true
    /// per-shot distribution that respects mid-circuit measurements and classical feedback.
    /// The expensive Python/JSON extraction happens once; only the block simulation loops.
    #[pyo3(signature = (distributed, shots=1000))]
    pub fn simulate_shots(
        &self,
        py: Python,
        distributed: &Bound<PyAny>,
        shots: usize,
    ) -> PyResult<PyObject> {
        // ── One-time setup (identical to simulate()) ──────────────────────────
        let instr_index_py = distributed.getattr("_instruction_index")?;
        let instr_index_dict = instr_index_py.downcast::<PyDict>()?;
        let mut instr_index: HashMap<usize, i64> = HashMap::new();
        for (k, v) in instr_index_dict.iter() {
            instr_index.insert(k.extract()?, v.extract()?);
        }

        let qpn: HashMap<usize, Vec<usize>> = distributed.getattr("qubits_per_node")?.extract()?;

        let circuits_py = distributed.getattr("circuits")?;
        let circuits_dict = circuits_py.downcast::<PyDict>()?;

        let mut nodes: Vec<usize> = circuits_dict
            .iter()
            .map(|(k, _)| k.extract::<usize>().unwrap())
            .collect();
        nodes.sort();

        let mut seen_ids: HashSet<usize> = HashSet::new();
        let mut entries: Vec<(i64, usize, usize)> = Vec::new();
        let mut fallback_order: i64 = i64::MAX / 2;
        let mut node_circuit_jsons: HashMap<usize, String> = HashMap::new();

        for &node in &nodes {
            let circuit_py = circuits_dict
                .get_item(node)?
                .expect("node not in circuits dict");
            let json: String = circuit_py.call_method0("model_dump_json")?.extract()?;
            node_circuit_jsons.insert(node, json);

            let instructions_list = circuit_py
                .getattr("instructions")?
                .downcast::<PyList>()?
                .to_owned();

            for (local_idx, inst_py) in instructions_list.iter().enumerate() {
                let py_id = inst_py.as_ptr() as usize;
                if seen_ids.contains(&py_id) {
                    continue;
                }
                seen_ids.insert(py_id);
                let order = instr_index.get(&py_id).copied().unwrap_or_else(|| {
                    let o = fallback_order;
                    fallback_order += 1;
                    o
                });
                entries.push((order, node, local_idx));
            }
        }
        entries.sort_by_key(|e| e.0);

        let node_circuits: HashMap<usize, Circuit> = node_circuit_jsons
            .iter()
            .map(|(&node, json)| {
                let c: Circuit = serde_json::from_str(json).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "Circuit JSON parse error for node {node}: {e}"
                    ))
                })?;
                Ok((node, c))
            })
            .collect::<PyResult<_>>()?;

        let num_cbits = node_circuits
            .values()
            .map(|c| c.num_cbits())
            .max()
            .unwrap_or(0);
        let base_seed = self.seed.unwrap_or_else(|| rand::thread_rng().gen());
        let n = qpn.values().flatten().copied().max().map_or(0, |q| q + 1);

        // Pre-fuse consecutive single-qubit gates across the globally-sorted stream.
        let fused_entries = fuse_pblock_entries(&entries, &node_circuits);

        // ── Shot loop — parallel across shots via rayon ───────────────────────
        let shot_loop_profiler = ShotLoopProfiler::start("pblock", n, shots, fused_entries.len());
        let counts = py
            .allow_threads(|| -> Result<HashMap<String, usize>, String> {
                (0..shots)
                    .into_par_iter()
                    .map(|i| -> Result<String, String> {
                        let mut pool = BlockPool::new(&qpn);
                        let mut cbits: HashMap<usize, i32> = HashMap::new();
                        let mut rng = ChaCha8Rng::seed_from_u64(base_seed.wrapping_add(i as u64));

                        for entry in &fused_entries {
                            match entry {
                                FusedPBlockEntry::Fused1Q { qubit, matrix } => {
                                    let block_idx = pool.ensure_single_block(&[*qubit]);
                                    let block = pool.blocks[block_idx].as_mut().unwrap();
                                    let local_q = block.local(*qubit);
                                    let n = block.qubits.len();
                                    apply_one_qubit_seq(&mut block.state, matrix, local_q, n, &[]);
                                }
                                FusedPBlockEntry::Original { node, local_idx } => {
                                    let inst = &node_circuits[node].instructions[*local_idx];
                                    let qubits = inst.qubits();
                                    if qubits.is_empty() {
                                        continue;
                                    }

                                    let block_idx = pool.ensure_single_block(&qubits);
                                    let block = pool.blocks[block_idx].as_mut().unwrap();
                                    dispatch_par(block, inst, &mut cbits, &mut rng)?;
                                }
                            }
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

    /// Simulate a DistributedCircuit using PBlock statevectors.
    ///
    /// Starts with one statevector per node. Merges blocks on demand when
    /// cross-node gates are encountered. Never calls as_monolithic_circuit().
    pub fn simulate(&self, _py: Python, distributed: &Bound<PyAny>) -> PyResult<PBlockResult> {
        // ── 1. Extract _instruction_index: dict[int(py_id), int(order)] ──────
        let instr_index_py = distributed.getattr("_instruction_index")?;
        let instr_index_dict = instr_index_py.downcast::<PyDict>()?;
        let mut instr_index: HashMap<usize, i64> = HashMap::new();
        for (k, v) in instr_index_dict.iter() {
            let id: usize = k.extract()?;
            let order: i64 = v.extract()?;
            instr_index.insert(id, order);
        }

        // ── 2. Extract qubits_per_node: dict[int, list[int]] ─────────────────
        let qpn: HashMap<usize, Vec<usize>> = distributed.getattr("qubits_per_node")?.extract()?;

        // ── 3. Iterate node circuits in sorted order ──────────────────────────
        let circuits_py = distributed.getattr("circuits")?;
        let circuits_dict = circuits_py.downcast::<PyDict>()?;

        let mut nodes: Vec<usize> = circuits_dict
            .iter()
            .map(|(k, _)| k.extract::<usize>().unwrap())
            .collect();
        nodes.sort();

        // entries: (global_order, node_id, local_instruction_index)
        // Deduplication by Python object pointer handles shared cross-node gates.
        let mut seen_ids: HashSet<usize> = HashSet::new();
        let mut entries: Vec<(i64, usize, usize)> = Vec::new();
        let mut fallback_order: i64 = i64::MAX / 2;
        let mut node_circuit_jsons: HashMap<usize, String> = HashMap::new();

        for &node in &nodes {
            let circuit_py = circuits_dict
                .get_item(node)?
                .expect("node not in circuits dict");

            let json: String = circuit_py.call_method0("model_dump_json")?.extract()?;
            node_circuit_jsons.insert(node, json);

            let instructions_list = circuit_py
                .getattr("instructions")?
                .downcast::<PyList>()?
                .to_owned();

            for (local_idx, inst_py) in instructions_list.iter().enumerate() {
                let py_id = inst_py.as_ptr() as usize;

                if seen_ids.contains(&py_id) {
                    continue;
                }
                seen_ids.insert(py_id);

                let order = instr_index.get(&py_id).copied().unwrap_or_else(|| {
                    let o = fallback_order;
                    fallback_order += 1;
                    o
                });
                entries.push((order, node, local_idx));
            }
        }

        entries.sort_by_key(|e| e.0);

        // ── 4. Deserialize all node circuits ─────────────────────────────────
        let node_circuits: HashMap<usize, Circuit> = node_circuit_jsons
            .iter()
            .map(|(&node, json)| {
                let c: Circuit = serde_json::from_str(json).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "Circuit JSON parse error for node {node}: {e}"
                    ))
                })?;
                Ok((node, c))
            })
            .collect::<PyResult<_>>()?;

        // ── 5. Initialise block pool (one block per node) ─────────────────────
        let mut pool = BlockPool::new(&qpn);
        let mut cbits: HashMap<usize, i32> = HashMap::new();
        let seed = self.seed.unwrap_or_else(|| rand::thread_rng().gen());
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        // ── 6. Execute instructions in global order (epoch-parallel) ─────────

        let mut cursor = 0;
        while cursor < entries.len() {
            // Greedily build an epoch: consecutive entries on distinct single blocks
            let mut epoch: Vec<(usize, usize)> = vec![]; // (entry_idx, block_idx)
            let mut epoch_block_set: HashSet<usize> = HashSet::new();
            let mut j = cursor;

            loop {
                if j >= entries.len() {
                    break;
                }
                let (_, node, local_idx) = entries[j];
                let inst = &node_circuits[&node].instructions[local_idx];
                let qubits = inst.qubits();
                if qubits.is_empty() {
                    j += 1;
                    continue;
                }

                // Which blocks does this instruction touch?
                let inst_blocks: Vec<usize> = qubits
                    .iter()
                    .filter_map(|q| pool.qubit_to_block.get(q).copied())
                    .collect::<HashSet<_>>()
                    .into_iter()
                    .collect();

                if inst_blocks.len() != 1 {
                    break;
                } // needs merge
                let blk = inst_blocks[0];
                if epoch_block_set.contains(&blk) {
                    break;
                } // same block twice

                epoch.push((j, blk));
                epoch_block_set.insert(blk);
                j += 1;
            }

            if epoch.len() > 1 {
                // Generate deterministic seeds for each parallel gate from the main rng
                let gate_seeds: Vec<u64> = (0..epoch.len()).map(|_| rng.gen()).collect();
                let mut per_cbits: Vec<HashMap<usize, i32>> =
                    (0..epoch.len()).map(|_| HashMap::new()).collect();

                // SAFETY: each iteration accesses a unique block_idx, so no aliasing.
                // We split `per_cbits` entries and pair each with a unique &mut Block.
                let raw = pool.blocks.as_mut_ptr();
                type EpochTask<'a> = (
                    &'a mut Option<Block>,
                    &'a mut HashMap<usize, i32>,
                    &'a Instruction,
                    u64,
                );
                let mut tasks: Vec<EpochTask<'_>> = per_cbits
                    .iter_mut()
                    .enumerate()
                    .map(|(k, local_cbits)| {
                        let (entry_idx, block_idx) = epoch[k];
                        let gate_seed = gate_seeds[k];
                        let (_, node, local_idx) = entries[entry_idx];
                        let inst = &node_circuits[&node].instructions[local_idx];
                        // SAFETY: epoch guarantees distinct block_idx for each k
                        let opt_block: &mut Option<Block> = unsafe { &mut *raw.add(block_idx) };
                        (opt_block, local_cbits, inst, gate_seed)
                    })
                    .collect();

                tasks
                    .par_iter_mut()
                    .for_each(|(opt_block, local_cbits, inst, gate_seed)| {
                        let block = opt_block.as_mut().expect("block should exist");
                        let mut gate_rng = ChaCha8Rng::seed_from_u64(*gate_seed);
                        dispatch_par(block, inst, local_cbits, &mut gate_rng).unwrap();
                    });

                for local in per_cbits {
                    cbits.extend(local);
                }
                cursor = j;
            } else {
                // Sequential: handle current instruction (may need a merge)
                let (_, node, local_idx) = entries[cursor];
                let inst = &node_circuits[&node].instructions[local_idx];
                let qubits = inst.qubits();
                if !qubits.is_empty() {
                    let block_idx = pool.ensure_single_block(&qubits);
                    let block = pool.blocks[block_idx].as_mut().unwrap();
                    dispatch(block, inst, &mut cbits, &mut rng)?;
                }
                cursor += 1;
            }
        }

        // ── 7. Combine all remaining blocks into final result ─────────────────
        let final_block = pool.merge_all();
        let num_qubits = final_block.qubits.len();
        let phys_qubits = final_block.qubits.clone();

        Ok(PBlockResult {
            sv: final_block.state,
            num_qubits,
            phys_qubits,
            cbits,
        })
    }
}

// ---------------------------------------------------------------------------
// Instruction dispatcher
// ---------------------------------------------------------------------------

fn dispatch(
    block: &mut Block,
    inst: &Instruction,
    cbits: &mut HashMap<usize, i32>,
    rng: &mut impl Rng,
) -> PyResult<()> {
    let n = block.qubits.len();

    match inst {
        Instruction::Id { .. } | Instruction::U0 { .. } => {}

        Instruction::X { qubit } => {
            let q = block.local(*qubit);
            apply_one_qubit(&mut block.state, &gates::X, q, n, &[]);
        }
        Instruction::Y { qubit } => {
            let q = block.local(*qubit);
            apply_one_qubit(&mut block.state, &gates::Y, q, n, &[]);
        }
        Instruction::Z { qubit } => {
            let q = block.local(*qubit);
            apply_one_qubit(&mut block.state, &gates::Z, q, n, &[]);
        }
        Instruction::H { qubit } => {
            let q = block.local(*qubit);
            apply_one_qubit(&mut block.state, &gates::h(), q, n, &[]);
        }
        Instruction::S { qubit } => {
            let q = block.local(*qubit);
            apply_one_qubit(&mut block.state, &gates::s_gate(), q, n, &[]);
        }
        Instruction::Sdg { qubit } => {
            let q = block.local(*qubit);
            apply_one_qubit(&mut block.state, &gates::sdg(), q, n, &[]);
        }
        Instruction::T { qubit } => {
            let q = block.local(*qubit);
            apply_one_qubit(&mut block.state, &gates::t_gate(), q, n, &[]);
        }
        Instruction::Tdg { qubit } => {
            let q = block.local(*qubit);
            apply_one_qubit(&mut block.state, &gates::tdg(), q, n, &[]);
        }
        Instruction::Sx { qubit } => {
            let q = block.local(*qubit);
            apply_one_qubit(&mut block.state, &gates::sx(), q, n, &[]);
        }
        Instruction::Sxdg { qubit } => {
            let q = block.local(*qubit);
            apply_one_qubit(&mut block.state, &gates::sxdg(), q, n, &[]);
        }

        Instruction::U3 {
            qubit,
            theta,
            phi,
            lam,
        } => {
            let q = block.local(*qubit);
            apply_one_qubit(&mut block.state, &gates::u3(*theta, *phi, *lam), q, n, &[]);
        }
        Instruction::U2 { qubit, phi, lam } => {
            let q = block.local(*qubit);
            apply_one_qubit(&mut block.state, &gates::u2(*phi, *lam), q, n, &[]);
        }
        Instruction::U1 { qubit, lam } => {
            let q = block.local(*qubit);
            apply_one_qubit(&mut block.state, &gates::u1(*lam), q, n, &[]);
        }
        Instruction::U {
            qubit,
            theta,
            phi,
            lam,
        } => {
            let q = block.local(*qubit);
            apply_one_qubit(&mut block.state, &gates::u(*theta, *phi, *lam), q, n, &[]);
        }
        Instruction::P { qubit, lam } => {
            let q = block.local(*qubit);
            apply_one_qubit(&mut block.state, &gates::p(*lam), q, n, &[]);
        }
        Instruction::Rx { qubit, theta } => {
            let q = block.local(*qubit);
            apply_one_qubit(&mut block.state, &gates::rx(*theta), q, n, &[]);
        }
        Instruction::Ry { qubit, theta } => {
            let q = block.local(*qubit);
            apply_one_qubit(&mut block.state, &gates::ry(*theta), q, n, &[]);
        }
        Instruction::Rz { qubit, phi } => {
            let q = block.local(*qubit);
            apply_one_qubit(&mut block.state, &gates::rz(*phi), q, n, &[]);
        }

        Instruction::Cx { control, target } => {
            let qs = [block.local(*control), block.local(*target)];
            apply_n_qubit(&mut block.state, &m4(gates::cnot()), &qs, n);
        }
        Instruction::Cz { control, target } => {
            let qs = [block.local(*control), block.local(*target)];
            apply_n_qubit(&mut block.state, &m4(gates::cz()), &qs, n);
        }
        Instruction::Cy { control, target } => {
            let qs = [block.local(*control), block.local(*target)];
            apply_n_qubit(&mut block.state, &m4(gates::cy()), &qs, n);
        }
        Instruction::Ch { control, target } => {
            let qs = [block.local(*control), block.local(*target)];
            apply_n_qubit(&mut block.state, &m4(gates::ch()), &qs, n);
        }
        Instruction::Swap { a, b } => {
            let qs = [block.local(*a), block.local(*b)];
            apply_n_qubit(&mut block.state, &m4(gates::swap()), &qs, n);
        }
        Instruction::Csx { control, target } => {
            let qs = [block.local(*control), block.local(*target)];
            apply_n_qubit(&mut block.state, &m4(gates::csx()), &qs, n);
        }
        Instruction::Crx {
            control,
            target,
            theta,
        } => {
            let qs = [block.local(*control), block.local(*target)];
            apply_n_qubit(&mut block.state, &m4(gates::crx(*theta)), &qs, n);
        }
        Instruction::Cry {
            control,
            target,
            theta,
        } => {
            let qs = [block.local(*control), block.local(*target)];
            apply_n_qubit(&mut block.state, &m4(gates::cry(*theta)), &qs, n);
        }
        Instruction::Crz {
            control,
            target,
            lam,
        } => {
            let qs = [block.local(*control), block.local(*target)];
            apply_n_qubit(&mut block.state, &m4(gates::crz(*lam)), &qs, n);
        }
        Instruction::Cu1 {
            control,
            target,
            lam,
        } => {
            let qs = [block.local(*control), block.local(*target)];
            apply_n_qubit(&mut block.state, &m4(gates::cu1(*lam)), &qs, n);
        }
        Instruction::Cp {
            control,
            target,
            lam,
        } => {
            let qs = [block.local(*control), block.local(*target)];
            apply_n_qubit(&mut block.state, &m4(gates::cp(*lam)), &qs, n);
        }
        Instruction::Cu3 {
            control,
            target,
            theta,
            phi,
            lam,
        } => {
            let qs = [block.local(*control), block.local(*target)];
            apply_n_qubit(
                &mut block.state,
                &m4(gates::cu3(*theta, *phi, *lam)),
                &qs,
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
            let qs = [block.local(*control), block.local(*target)];
            apply_n_qubit(
                &mut block.state,
                &m4(gates::cu(*theta, *phi, *lam, *gamma)),
                &qs,
                n,
            );
        }
        Instruction::Rxx { a, b, theta } => {
            let qs = [block.local(*a), block.local(*b)];
            apply_n_qubit(&mut block.state, &m4(gates::rxx(*theta)), &qs, n);
        }
        Instruction::Rzz { a, b, theta } => {
            let qs = [block.local(*a), block.local(*b)];
            apply_n_qubit(&mut block.state, &m4(gates::rzz(*theta)), &qs, n);
        }

        Instruction::Ccx {
            control1,
            control2,
            target,
        } => {
            let qs = [
                block.local(*control1),
                block.local(*control2),
                block.local(*target),
            ];
            apply_n_qubit(&mut block.state, &m8(gates::ccx()), &qs, n);
        }
        Instruction::Cswap {
            control,
            target1,
            target2,
        } => {
            let qs = [
                block.local(*control),
                block.local(*target1),
                block.local(*target2),
            ];
            apply_n_qubit(&mut block.state, &m8(gates::cswap()), &qs, n);
        }
        Instruction::Rccx {
            control1,
            control2,
            target,
        } => {
            let qs = [
                block.local(*control1),
                block.local(*control2),
                block.local(*target),
            ];
            apply_n_qubit(&mut block.state, &m8(gates::rccx()), &qs, n);
        }
        Instruction::Rc3x {
            control1,
            control2,
            control3,
            target,
        } => {
            let qs = [
                block.local(*control1),
                block.local(*control2),
                block.local(*control3),
                block.local(*target),
            ];
            apply_n_qubit(&mut block.state, &m16(gates::rc3x()), &qs, n);
        }
        Instruction::C3x {
            control1,
            control2,
            control3,
            target,
        } => {
            let qs = [
                block.local(*control1),
                block.local(*control2),
                block.local(*control3),
                block.local(*target),
            ];
            apply_n_qubit(&mut block.state, &m16(gates::c3x()), &qs, n);
        }
        Instruction::C3sqrtx {
            control1,
            control2,
            control3,
            target,
        } => {
            let qs = [
                block.local(*control1),
                block.local(*control2),
                block.local(*control3),
                block.local(*target),
            ];
            apply_n_qubit(&mut block.state, &m16(gates::c3sqrtx()), &qs, n);
        }
        Instruction::C4x {
            control1,
            control2,
            control3,
            control4,
            target,
        } => {
            let qs = [
                block.local(*control1),
                block.local(*control2),
                block.local(*control3),
                block.local(*control4),
                block.local(*target),
            ];
            apply_n_qubit(&mut block.state, &m32(gates::c4x()), &qs, n);
        }

        Instruction::Gate { name, qubits, .. } => {
            let lqs: Vec<usize> = qubits.iter().map(|&q| block.local(q)).collect();
            match name.to_lowercase().as_str() {
                "remote_link_phi_plus" => {
                    apply_n_qubit(&mut block.state, &m4(gates::phi_plus()), &lqs, n);
                }
                "remote_link_psi_minus" => {
                    apply_n_qubit(&mut block.state, &m4(gates::psi_minus()), &lqs, n);
                }
                "remote_link_psi_plus" => {
                    apply_n_qubit(&mut block.state, &m4(gates::psi_plus()), &lqs, n);
                }
                "nonlocal_cz" | "remote_cz" => {
                    apply_n_qubit(&mut block.state, &m4(gates::cz()), &lqs, n);
                }
                "remote_cx" => {
                    apply_n_qubit(&mut block.state, &m4(gates::cnot()), &lqs, n);
                }
                "remote_barrier" | "remote_cu1" => {
                    // remote_barrier is a no-op; remote_cu1 is opaque with no params
                }
                "remote_epr" => {
                    apply_n_qubit(&mut block.state, &m4(gates::phi_plus()), &lqs, n);
                }
                other if other.starts_with("circuit-") => {
                    // opaque Qiskit subcircuit — no-op for performance benchmarking
                }
                "teleport" => {
                    return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                        "Symbolic 'teleport' gate cannot be simulated natively. \
                         Distribute with lowered=True to get a decomposed circuit.",
                    ));
                }
                other => {
                    return Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
                        "Unsupported gate in pblock simulator: {other:?}"
                    )));
                }
            }
        }

        Instruction::Measure { qubit, cbit } => {
            let q = block.local(*qubit);
            let outcome = measure_qubit(&mut block.state, q, n, rng);
            cbits.insert(*cbit, outcome as i32);
        }

        Instruction::Conditional { condition, op } => {
            let mut actual: u64 = 0;
            for bit in 0..condition.creg_size {
                let val = *cbits.get(&(condition.creg_base + bit)).unwrap_or(&0) as u64;
                actual |= val << bit;
            }
            if actual == condition.creg_value {
                dispatch(block, op, cbits, rng)?;
            }
        }

        Instruction::Reset { qubit } => {
            let q = block.local(*qubit);
            let outcome = measure_qubit(&mut block.state, q, n, rng);
            if outcome == 1 {
                apply_one_qubit(&mut block.state, &gates::X, q, n, &[]);
            }
        }

        Instruction::Barrier | Instruction::Classical { .. } => {}
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// GIL-free instruction dispatcher (used inside py.allow_threads for parallel shots
// and intra-shot epoch parallelism)
// ---------------------------------------------------------------------------

fn dispatch_par(
    block: &mut Block,
    inst: &Instruction,
    cbits: &mut HashMap<usize, i32>,
    rng: &mut impl Rng,
) -> Result<(), String> {
    let n = block.qubits.len();

    match inst {
        Instruction::Id { .. } | Instruction::U0 { .. } => {}

        Instruction::X { qubit } => {
            let q = block.local(*qubit);
            apply_one_qubit_seq(&mut block.state, &gates::X, q, n, &[]);
        }
        Instruction::Y { qubit } => {
            let q = block.local(*qubit);
            apply_one_qubit_seq(&mut block.state, &gates::Y, q, n, &[]);
        }
        Instruction::Z { qubit } => {
            let q = block.local(*qubit);
            apply_one_qubit_seq(&mut block.state, &gates::Z, q, n, &[]);
        }
        Instruction::H { qubit } => {
            let q = block.local(*qubit);
            apply_one_qubit_seq(&mut block.state, &gates::h(), q, n, &[]);
        }
        Instruction::S { qubit } => {
            let q = block.local(*qubit);
            apply_one_qubit_seq(&mut block.state, &gates::s_gate(), q, n, &[]);
        }
        Instruction::Sdg { qubit } => {
            let q = block.local(*qubit);
            apply_one_qubit_seq(&mut block.state, &gates::sdg(), q, n, &[]);
        }
        Instruction::T { qubit } => {
            let q = block.local(*qubit);
            apply_one_qubit_seq(&mut block.state, &gates::t_gate(), q, n, &[]);
        }
        Instruction::Tdg { qubit } => {
            let q = block.local(*qubit);
            apply_one_qubit_seq(&mut block.state, &gates::tdg(), q, n, &[]);
        }
        Instruction::Sx { qubit } => {
            let q = block.local(*qubit);
            apply_one_qubit_seq(&mut block.state, &gates::sx(), q, n, &[]);
        }
        Instruction::Sxdg { qubit } => {
            let q = block.local(*qubit);
            apply_one_qubit_seq(&mut block.state, &gates::sxdg(), q, n, &[]);
        }

        Instruction::U3 {
            qubit,
            theta,
            phi,
            lam,
        } => {
            let q = block.local(*qubit);
            apply_one_qubit_seq(&mut block.state, &gates::u3(*theta, *phi, *lam), q, n, &[]);
        }
        Instruction::U2 { qubit, phi, lam } => {
            let q = block.local(*qubit);
            apply_one_qubit_seq(&mut block.state, &gates::u2(*phi, *lam), q, n, &[]);
        }
        Instruction::U1 { qubit, lam } => {
            let q = block.local(*qubit);
            apply_one_qubit_seq(&mut block.state, &gates::u1(*lam), q, n, &[]);
        }
        Instruction::U {
            qubit,
            theta,
            phi,
            lam,
        } => {
            let q = block.local(*qubit);
            apply_one_qubit_seq(&mut block.state, &gates::u(*theta, *phi, *lam), q, n, &[]);
        }
        Instruction::P { qubit, lam } => {
            let q = block.local(*qubit);
            apply_one_qubit_seq(&mut block.state, &gates::p(*lam), q, n, &[]);
        }
        Instruction::Rx { qubit, theta } => {
            let q = block.local(*qubit);
            apply_one_qubit_seq(&mut block.state, &gates::rx(*theta), q, n, &[]);
        }
        Instruction::Ry { qubit, theta } => {
            let q = block.local(*qubit);
            apply_one_qubit_seq(&mut block.state, &gates::ry(*theta), q, n, &[]);
        }
        Instruction::Rz { qubit, phi } => {
            let q = block.local(*qubit);
            apply_one_qubit_seq(&mut block.state, &gates::rz(*phi), q, n, &[]);
        }

        Instruction::Cx { control, target } => {
            let qs = [block.local(*control), block.local(*target)];
            apply_n_qubit_seq(&mut block.state, &m4(gates::cnot()), &qs, n);
        }
        Instruction::Cz { control, target } => {
            let qs = [block.local(*control), block.local(*target)];
            apply_n_qubit_seq(&mut block.state, &m4(gates::cz()), &qs, n);
        }
        Instruction::Cy { control, target } => {
            let qs = [block.local(*control), block.local(*target)];
            apply_n_qubit_seq(&mut block.state, &m4(gates::cy()), &qs, n);
        }
        Instruction::Ch { control, target } => {
            let qs = [block.local(*control), block.local(*target)];
            apply_n_qubit_seq(&mut block.state, &m4(gates::ch()), &qs, n);
        }
        Instruction::Swap { a, b } => {
            let qs = [block.local(*a), block.local(*b)];
            apply_n_qubit_seq(&mut block.state, &m4(gates::swap()), &qs, n);
        }
        Instruction::Csx { control, target } => {
            let qs = [block.local(*control), block.local(*target)];
            apply_n_qubit_seq(&mut block.state, &m4(gates::csx()), &qs, n);
        }
        Instruction::Crx {
            control,
            target,
            theta,
        } => {
            let qs = [block.local(*control), block.local(*target)];
            apply_n_qubit_seq(&mut block.state, &m4(gates::crx(*theta)), &qs, n);
        }
        Instruction::Cry {
            control,
            target,
            theta,
        } => {
            let qs = [block.local(*control), block.local(*target)];
            apply_n_qubit_seq(&mut block.state, &m4(gates::cry(*theta)), &qs, n);
        }
        Instruction::Crz {
            control,
            target,
            lam,
        } => {
            let qs = [block.local(*control), block.local(*target)];
            apply_n_qubit_seq(&mut block.state, &m4(gates::crz(*lam)), &qs, n);
        }
        Instruction::Cu1 {
            control,
            target,
            lam,
        } => {
            let qs = [block.local(*control), block.local(*target)];
            apply_n_qubit_seq(&mut block.state, &m4(gates::cu1(*lam)), &qs, n);
        }
        Instruction::Cp {
            control,
            target,
            lam,
        } => {
            let qs = [block.local(*control), block.local(*target)];
            apply_n_qubit_seq(&mut block.state, &m4(gates::cp(*lam)), &qs, n);
        }
        Instruction::Cu3 {
            control,
            target,
            theta,
            phi,
            lam,
        } => {
            let qs = [block.local(*control), block.local(*target)];
            apply_n_qubit_seq(
                &mut block.state,
                &m4(gates::cu3(*theta, *phi, *lam)),
                &qs,
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
            let qs = [block.local(*control), block.local(*target)];
            apply_n_qubit_seq(
                &mut block.state,
                &m4(gates::cu(*theta, *phi, *lam, *gamma)),
                &qs,
                n,
            );
        }
        Instruction::Rxx { a, b, theta } => {
            let qs = [block.local(*a), block.local(*b)];
            apply_n_qubit_seq(&mut block.state, &m4(gates::rxx(*theta)), &qs, n);
        }
        Instruction::Rzz { a, b, theta } => {
            let qs = [block.local(*a), block.local(*b)];
            apply_n_qubit_seq(&mut block.state, &m4(gates::rzz(*theta)), &qs, n);
        }

        Instruction::Ccx {
            control1,
            control2,
            target,
        } => {
            let qs = [
                block.local(*control1),
                block.local(*control2),
                block.local(*target),
            ];
            apply_n_qubit_seq(&mut block.state, &m8(gates::ccx()), &qs, n);
        }
        Instruction::Cswap {
            control,
            target1,
            target2,
        } => {
            let qs = [
                block.local(*control),
                block.local(*target1),
                block.local(*target2),
            ];
            apply_n_qubit_seq(&mut block.state, &m8(gates::cswap()), &qs, n);
        }
        Instruction::Rccx {
            control1,
            control2,
            target,
        } => {
            let qs = [
                block.local(*control1),
                block.local(*control2),
                block.local(*target),
            ];
            apply_n_qubit_seq(&mut block.state, &m8(gates::rccx()), &qs, n);
        }
        Instruction::Rc3x {
            control1,
            control2,
            control3,
            target,
        } => {
            let qs = [
                block.local(*control1),
                block.local(*control2),
                block.local(*control3),
                block.local(*target),
            ];
            apply_n_qubit_seq(&mut block.state, &m16(gates::rc3x()), &qs, n);
        }
        Instruction::C3x {
            control1,
            control2,
            control3,
            target,
        } => {
            let qs = [
                block.local(*control1),
                block.local(*control2),
                block.local(*control3),
                block.local(*target),
            ];
            apply_n_qubit_seq(&mut block.state, &m16(gates::c3x()), &qs, n);
        }
        Instruction::C3sqrtx {
            control1,
            control2,
            control3,
            target,
        } => {
            let qs = [
                block.local(*control1),
                block.local(*control2),
                block.local(*control3),
                block.local(*target),
            ];
            apply_n_qubit_seq(&mut block.state, &m16(gates::c3sqrtx()), &qs, n);
        }
        Instruction::C4x {
            control1,
            control2,
            control3,
            control4,
            target,
        } => {
            let qs = [
                block.local(*control1),
                block.local(*control2),
                block.local(*control3),
                block.local(*control4),
                block.local(*target),
            ];
            apply_n_qubit_seq(&mut block.state, &m32(gates::c4x()), &qs, n);
        }

        Instruction::Gate { name, qubits, .. } => {
            let lqs: Vec<usize> = qubits.iter().map(|&q| block.local(q)).collect();
            match name.to_lowercase().as_str() {
                "remote_link_phi_plus" => {
                    apply_n_qubit_seq(&mut block.state, &m4(gates::phi_plus()), &lqs, n);
                }
                "remote_link_psi_minus" => {
                    apply_n_qubit_seq(&mut block.state, &m4(gates::psi_minus()), &lqs, n);
                }
                "remote_link_psi_plus" => {
                    apply_n_qubit_seq(&mut block.state, &m4(gates::psi_plus()), &lqs, n);
                }
                "nonlocal_cz" | "remote_cz" => {
                    apply_n_qubit_seq(&mut block.state, &m4(gates::cz()), &lqs, n);
                }
                "remote_cx" => {
                    apply_n_qubit_seq(&mut block.state, &m4(gates::cnot()), &lqs, n);
                }
                "remote_barrier" | "remote_cu1" => {
                    // remote_barrier is a no-op; remote_cu1 is opaque with no params
                }
                "remote_epr" => {
                    apply_n_qubit_seq(&mut block.state, &m4(gates::phi_plus()), &lqs, n);
                }
                other if other.starts_with("circuit-") => {
                    // opaque Qiskit subcircuit — no-op for performance benchmarking
                }
                "teleport" => {
                    return Err("Symbolic 'teleport' gate cannot be simulated natively. \
                         Distribute with lowered=True to get a decomposed circuit."
                        .to_string());
                }
                other => {
                    return Err(format!("Unsupported gate in pblock simulator: {other:?}"));
                }
            }
        }

        Instruction::Measure { qubit, cbit } => {
            let q = block.local(*qubit);
            let outcome = measure_qubit_seq(&mut block.state, q, n, rng);
            cbits.insert(*cbit, outcome as i32);
        }

        Instruction::Conditional { condition, op } => {
            let mut actual: u64 = 0;
            for bit in 0..condition.creg_size {
                let val = *cbits.get(&(condition.creg_base + bit)).unwrap_or(&0) as u64;
                actual |= val << bit;
            }
            if actual == condition.creg_value {
                dispatch_par(block, op, cbits, rng)?;
            }
        }

        Instruction::Reset { qubit } => {
            let q = block.local(*qubit);
            let outcome = measure_qubit_seq(&mut block.state, q, n, rng);
            if outcome == 1 {
                apply_one_qubit_seq(&mut block.state, &gates::X, q, n, &[]);
            }
        }

        Instruction::Barrier | Instruction::Classical { .. } => {}
    }
    Ok(())
}
