#[cfg(unix)]
use std::fs::{self, File};
#[cfg(unix)]
use std::path::PathBuf;
#[cfg(unix)]
use std::sync::atomic::{AtomicUsize, Ordering};

#[cfg(unix)]
use pprof::{ProfilerGuard, ProfilerGuardBuilder};
use serde::{Deserialize, Serialize};

#[cfg(unix)]
static PROFILE_COUNTER: AtomicUsize = AtomicUsize::new(1);

#[cfg(unix)]
pub struct ShotLoopProfiler {
    guard: ProfilerGuard<'static>,
    output_path: PathBuf,
}

#[cfg(unix)]
impl ShotLoopProfiler {
    pub fn start(mode: &str, qubits: usize, shots: usize, instructions: usize) -> Option<Self> {
        let dir = output_dir()?;
        if let Err(err) = fs::create_dir_all(&dir) {
            eprintln!(
                "[dqsim flamegraph] could not create {}: {err}",
                dir.display()
            );
            return None;
        }

        let frequency = std::env::var("DQSIM_FLAMEGRAPH_HZ")
            .ok()
            .and_then(|value| value.parse::<i32>().ok())
            .unwrap_or(997);

        let guard = match ProfilerGuardBuilder::default()
            .frequency(frequency)
            .blocklist(&["libc", "libgcc", "pthread", "vdso"])
            .build()
        {
            Ok(guard) => guard,
            Err(err) => {
                eprintln!("[dqsim flamegraph] could not start profiler: {err}");
                return None;
            }
        };

        let index = PROFILE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let output_path = dir.join(format!(
            "{index:03}-{}-q{qubits}-shots{shots}-inst{instructions}.svg",
            sanitize(mode),
        ));

        Some(Self { guard, output_path })
    }

    pub fn finish(self) {
        let report = match self.guard.report().build() {
            Ok(report) => report,
            Err(err) => {
                eprintln!("[dqsim flamegraph] could not build report: {err}");
                return;
            }
        };

        let mut file = match File::create(&self.output_path) {
            Ok(file) => file,
            Err(err) => {
                eprintln!(
                    "[dqsim flamegraph] could not create {}: {err}",
                    self.output_path.display()
                );
                return;
            }
        };

        if let Err(err) = report.flamegraph(&mut file) {
            eprintln!(
                "[dqsim flamegraph] could not write {}: {err}",
                self.output_path.display()
            );
            return;
        }

        eprintln!("[dqsim flamegraph] wrote {}", self.output_path.display());
    }
}

#[cfg(not(unix))]
pub struct ShotLoopProfiler;

#[cfg(not(unix))]
impl ShotLoopProfiler {
    pub fn start(_mode: &str, _qubits: usize, _shots: usize, _instructions: usize) -> Option<Self> {
        None
    }

    pub fn finish(self) {}
}

#[cfg(unix)]
fn output_dir() -> Option<PathBuf> {
    std::env::var_os("DQSIM_FLAMEGRAPH_DIR")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

#[cfg(unix)]
fn sanitize(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '-'
            }
        })
        .collect()
}
// ---------------------------------------------------------------------------
// Shot-level Profiling Data Structure
// ---------------------------------------------------------------------------

/// Statistics for a single shot execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShotStats {
    /// Shot index (0..num_shots)
    pub shot_id: usize,
    /// Execution time for this shot in milliseconds
    pub execution_time_ms: f64,
}

/// Comprehensive profiling data for simulate_shots execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShotsProfile {
    /// Total number of shots executed
    pub num_shots: usize,
    /// Number of qubits in the circuit
    pub num_qubits: usize,
    /// Number of instructions in the circuit
    pub num_instructions: usize,
    /// Time to parse and deserialize circuit in milliseconds
    pub preprocessing_ms: f64,
    /// Time to perform gate fusion in milliseconds
    pub gate_fusion_ms: f64,
    /// Total execution time for all shots in parallel in milliseconds
    pub parallel_execution_ms: f64,
    /// Per-shot execution statistics (optional, only if detailed=true)
    pub per_shot_stats: Option<Vec<ShotStats>>,
    /// Whether the statevector SIMD path was active for this profile.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub statevector_simd_enabled: Option<bool>,
    /// SIMD backend used by statevector, when active.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub statevector_simd_backend: Option<String>,
    /// Whether this statevector profile was large enough to use the SIMD gate path.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub statevector_simd_used: Option<bool>,
    /// Minimum qubit count required before the default SIMD path is used.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub statevector_simd_min_qubits: Option<usize>,
    /// Whether specialized fixed two-qubit statevector kernels were requested.
    #[serde(
        rename = "statevector_2q_kernels_enabled",
        skip_serializing_if = "Option::is_none"
    )]
    pub statevector_two_qubit_kernels_enabled: Option<bool>,
    /// Whether this statevector profile contained gates handled by the specialized kernels.
    #[serde(
        rename = "statevector_2q_kernels_used",
        skip_serializing_if = "Option::is_none"
    )]
    pub statevector_two_qubit_kernels_used: Option<bool>,
    /// Gate names covered by the specialized fixed two-qubit kernels.
    #[serde(
        rename = "statevector_2q_kernel_gates",
        skip_serializing_if = "Option::is_none"
    )]
    pub statevector_two_qubit_kernel_gates: Option<String>,
    /// Whether statevector shot branching/batching was requested and available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub statevector_shot_branching_enabled: Option<bool>,
    /// Whether this profile actually used the shot branching/batching path.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub statevector_shot_branching_used: Option<bool>,
    /// Branching strategy used by the statevector shot loop.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub statevector_shot_branching_strategy: Option<String>,
    /// Whether MPS shot branching/batching was requested and available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mps_shot_branching_enabled: Option<bool>,
    /// Whether this profile actually used the MPS shot branching/batching path.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mps_shot_branching_used: Option<bool>,
    /// Branching strategy used by the MPS shot loop.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mps_shot_branching_strategy: Option<String>,
    /// Whether lazy MPS qubit ordering was enabled for this profile.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mps_lazy_qubit_ordering_enabled: Option<bool>,
    /// MPS routing strategy used for non-adjacent two-qubit gates.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mps_routing_mode: Option<String>,
    /// Logical two-qubit gates processed by the MPS simulator.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mps_logical_2q_gates: Option<usize>,
    /// Adjacent routing SWAPs inserted by the MPS simulator.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mps_routing_swaps: Option<usize>,
    /// Total adjacent two-qubit tensor applications, including routing swaps.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mps_adjacent_2q_applications: Option<usize>,
    /// Number of SVD decompositions performed by MPS two-qubit applications.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mps_svd_count: Option<usize>,
    /// Time spent in MPS SVD calls, in milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mps_svd_time_ms: Option<f64>,
    /// Whether optimized MPS adjacent two-qubit kernels were enabled.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mps_2q_fast_kernels_enabled: Option<bool>,
    /// MPS two-qubit fast-kernel selection mode: off, auto, or force.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mps_2q_fast_kernel_mode: Option<String>,
    /// Whether any optimized MPS adjacent two-qubit kernel path was used.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mps_2q_fast_kernels_used: Option<bool>,
    /// Adjacent two-qubit applications handled by the optimized MPS assembly path.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mps_2q_fast_kernel_applications: Option<usize>,
    /// Adjacent two-qubit applications kept on the baseline path by auto mode.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mps_2q_fast_kernel_auto_skipped_applications: Option<usize>,
    /// Optimized MPS diagonal two-qubit kernel applications.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mps_2q_diagonal_kernel_applications: Option<usize>,
    /// Optimized MPS permutation two-qubit kernel applications.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mps_2q_permutation_kernel_applications: Option<usize>,
    /// Optimized MPS dense two-qubit kernel applications.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mps_2q_dense_kernel_applications: Option<usize>,
    /// Two-site updates that stayed rank-1 and skipped SVD.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mps_rank1_factorizations: Option<usize>,
    /// Number of MPS SVD calls skipped by exact rank-1 factorization.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mps_svd_skipped_count: Option<usize>,
    /// Maximum bond dimension observed after MPS decompositions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mps_max_observed_bond_dimension: Option<usize>,
    /// Average physical distance between logical two-qubit operands before routing.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mps_average_routed_distance: Option<f64>,
    /// Maximum physical distance between logical two-qubit operands before routing.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mps_max_routed_distance: Option<usize>,
    /// Number of lookahead routing choices made.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mps_lookahead_decisions: Option<usize>,
    /// Number of lookahead choices that selected the non-default route direction.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mps_lookahead_flipped_routes: Option<usize>,
    /// Whether statevector qubit truncation/remapping was requested.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub statevector_qubit_truncation_enabled: Option<bool>,
    /// Whether this profile actually removed one or more qubits.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub statevector_qubit_truncation_used: Option<bool>,
    /// Truncation strategy used by the statevector shot loop.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub statevector_qubit_truncation_strategy: Option<String>,
    /// Original circuit qubit count before truncation/remapping.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub statevector_original_num_qubits: Option<usize>,
    /// Effective simulated qubit count after truncation/remapping.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub statevector_effective_num_qubits: Option<usize>,
    /// Original qubit indices removed by truncation/remapping.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub statevector_removed_qubits: Option<Vec<usize>>,
    /// Total time including preprocessing and fusion in milliseconds
    pub total_time_ms: f64,
}

impl ShotsProfile {
    pub fn to_json_string(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}
