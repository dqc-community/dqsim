use pyo3::prelude::*;
use pyo3::types::PyDict;

pub(crate) enum MonolithicSimulationMode {
    StateVector,
    Mps,
    Stabilizer,
}

pub(crate) enum DistributedSimulationMode {
    PBlock,
}

pub(crate) struct MpsOptions {
    pub(crate) max_bond_dimension: Option<usize>,
    pub(crate) truncation_threshold: f64,
}

pub(crate) fn parse_monolithic_mode(mode: &str) -> PyResult<MonolithicSimulationMode> {
    match mode.trim().to_ascii_lowercase().replace('-', "_").as_str() {
        "state_vector" | "statevector" | "sv" => Ok(MonolithicSimulationMode::StateVector),
        "mps" | "matrix_product_state" => Ok(MonolithicSimulationMode::Mps),
        "stabilizer" | "stab" => Ok(MonolithicSimulationMode::Stabilizer),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unsupported monolithic simulation mode {other:?}; expected 'state_vector' or 'mps'"
        ))),
    }
}

pub(crate) fn parse_distributed_mode(mode: &str) -> PyResult<DistributedSimulationMode> {
    match mode.trim().to_ascii_lowercase().replace('-', "_").as_str() {
        "p_block" => Ok(DistributedSimulationMode::PBlock),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unsupported distributed simulation mode {other:?}; expected 'p_block'"
        ))),
    }
}

pub(crate) fn parse_mps_options(options: Option<&Bound<PyDict>>) -> PyResult<MpsOptions> {
    let mut max_bond_dimension = None;
    let mut truncation_threshold: f64 = 1e-12;

    if let Some(options) = options {
        for (key, value) in options.iter() {
            let key: String = key.extract()?;
            match key.as_str() {
                "max_bond_dimension" => {
                    max_bond_dimension = parse_max_bond_dimension(&value)?;
                }
                "truncation_threshold" => {
                    truncation_threshold = parse_truncation_threshold(&value)?;
                }
                other => {
                    return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                        "Unsupported MPS option {other:?}"
                    )));
                }
            }
        }
    }

    Ok(MpsOptions {
        max_bond_dimension,
        truncation_threshold,
    })
}

pub(crate) fn reject_monolithic_options(
    options: Option<&Bound<PyDict>>,
    mode: &str,
) -> PyResult<()> {
    if let Some(options) = options {
        if !options.is_empty() {
            let keys: Vec<String> = options
                .iter()
                .map(|(key, _)| key.extract())
                .collect::<PyResult<_>>()?;
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "Options {keys:?} are not supported for monolithic mode {mode:?}"
            )));
        }
    }
    Ok(())
}

fn parse_max_bond_dimension(value: &Bound<PyAny>) -> PyResult<Option<usize>> {
    if value.is_none() {
        return Ok(None);
    }

    let max_bond: usize = value.extract()?;
    if max_bond == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "max_bond_dimension must be positive or None",
        ));
    }
    Ok(Some(max_bond))
}

fn parse_truncation_threshold(value: &Bound<PyAny>) -> PyResult<f64> {
    let truncation_threshold: f64 = value.extract()?;
    if !truncation_threshold.is_finite() || truncation_threshold < 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "truncation_threshold must be a finite non-negative float",
        ));
    }
    Ok(truncation_threshold)
}
