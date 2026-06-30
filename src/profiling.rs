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
    /// Total time including preprocessing and fusion in milliseconds
    pub total_time_ms: f64,
}

impl ShotsProfile {
    pub fn to_json_string(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}
