use std::fs::{self, File};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use pprof::{ProfilerGuard, ProfilerGuardBuilder};

static PROFILE_COUNTER: AtomicUsize = AtomicUsize::new(1);

pub struct ShotLoopProfiler {
    guard: ProfilerGuard<'static>,
    output_path: PathBuf,
}

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

fn output_dir() -> Option<PathBuf> {
    std::env::var_os("DQSIM_FLAMEGRAPH_DIR")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

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
