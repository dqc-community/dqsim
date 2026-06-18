use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::codecs::{
    parse_distributed_mode, parse_monolithic_mode, parse_mps_options, reject_monolithic_options,
    DistributedSimulationMode, MonolithicSimulationMode,
};
use crate::distributed::pblock::PBlockSimulator;
use crate::monolithic::mps::MpsSimulator;
use crate::monolithic::statevector::StatevectorSimulator;

#[pyfunction]
#[pyo3(signature = (circuit, mode="state_vector", seed=None, profile=false, **options))]
pub fn simulate_monolithic(
    py: Python,
    circuit: &Bound<PyAny>, // todo: tighten this
    mode: &str,
    seed: Option<u64>,
    profile: bool,
    options: Option<&Bound<PyDict>>,
) -> PyResult<PyObject> {
    match parse_monolithic_mode(mode)? {
        MonolithicSimulationMode::StateVector => {
            reject_monolithic_options(options, mode)?;
            let sim = StatevectorSimulator::new(seed, profile);
            Ok(sim.simulate(py, circuit)?.into_py(py))
        }
        MonolithicSimulationMode::Mps => {
            let options = parse_mps_options(options)?;
            let sim = MpsSimulator::new(
                seed,
                options.max_bond_dimension,
                options.truncation_threshold,
            );
            Ok(sim.simulate(py, circuit)?.into_py(py))
        }
    }
}

#[pyfunction]
#[pyo3(signature = (distributed, mode="p_block", seed=None))]
pub fn simulate_distributed(
    py: Python,
    distributed: &Bound<PyAny>,  // todo: tighten this
    mode: &str,
    seed: Option<u64>,
) -> PyResult<PyObject> {
    match parse_distributed_mode(mode)? {
        DistributedSimulationMode::PBlock => {
            let sim = PBlockSimulator::new(seed);
            Ok(sim.simulate(py, distributed)?.into_py(py))
        }
    }
}

#[pyfunction]
#[pyo3(signature = (circuit, mode="state_vector", shots=1000, seed=None, **options))]
pub fn simulate_monolithic_shots(
    py: Python,
    circuit: &Bound<PyAny>, // todo: tighten this
    mode: &str,
    shots: usize,
    seed: Option<u64>,
    options: Option<&Bound<PyDict>>,
) -> PyResult<PyObject> {
    match parse_monolithic_mode(mode)? {
        MonolithicSimulationMode::StateVector => {
            reject_monolithic_options(options, mode)?;
            let sim = StatevectorSimulator::new(seed, false);
            sim.simulate_shots(py, circuit, shots, false)
        }
        MonolithicSimulationMode::Mps => {
            let options = parse_mps_options(options)?;
            let sim = MpsSimulator::new(
                seed,
                options.max_bond_dimension,
                options.truncation_threshold,
            );
            sim.simulate_shots(py, circuit, shots)
        }
    }
}

#[pyfunction]
#[pyo3(signature = (distributed, mode="p_block", shots=1000, seed=None))]
pub fn simulate_distributed_shots(
    py: Python,
    distributed: &Bound<PyAny>, // todo: tighten this
    mode: &str,
    shots: usize,
    seed: Option<u64>,
) -> PyResult<PyObject> {
    match parse_distributed_mode(mode)? {
        DistributedSimulationMode::PBlock => {
            let sim = PBlockSimulator::new(seed);
            sim.simulate_shots(py, distributed, shots)
        }
    }
}
