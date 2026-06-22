#![allow(clippy::useless_conversion)]

mod codecs;
mod distributed;
mod engine;
mod gates;
mod monolithic;
mod profiling;
mod simulator;
mod types;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<monolithic::statevector::StatevectorSimulator>()?;
    m.add_class::<monolithic::mps::MpsSimulator>()?;
    m.add_class::<monolithic::statevector::SimulationResult>()?;
    m.add_class::<monolithic::statevector::SimulationProfile>()?;
    m.add_class::<monolithic::stabilizer::StabilizerSimulator>()?;
    m.add_class::<distributed::pblock::PBlockSimulator>()?;
    m.add_class::<distributed::pblock::PBlockResult>()?;
    m.add_function(wrap_pyfunction!(simulator::simulate_monolithic, m)?)?;
    m.add_function(wrap_pyfunction!(simulator::simulate_distributed, m)?)?;
    m.add_function(wrap_pyfunction!(simulator::simulate_monolithic_shots, m)?)?;
    m.add_function(wrap_pyfunction!(simulator::simulate_distributed_shots, m)?)?;
    Ok(())
}
