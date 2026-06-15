use std::fmt::Write;
use std::ops::AddAssign;
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Default)]
pub struct Counter {
    pub calls: u64,
    pub elapsed: Duration,
}

impl Counter {
    #[inline]
    pub fn add(&mut self, elapsed: Duration) {
        self.calls += 1;
        self.elapsed += elapsed;
    }
}

impl AddAssign for Counter {
    fn add_assign(&mut self, rhs: Self) {
        self.calls += rhs.calls;
        self.elapsed += rhs.elapsed;
    }
}

#[derive(Clone, Default)]
pub struct ShotStats {
    pub state_clone: Counter,
    pub block_pool_create: Counter,
    pub ensure_block: Counter,
    pub fused_one_qubit: Counter,
    pub original_dispatch: Counter,
    pub single_qubit: Counter,
    pub multi_qubit: Counter,
    pub measurement: Counter,
    pub reset: Counter,
    pub conditional: Counter,
    pub format_counts_key: Counter,
}

impl AddAssign for ShotStats {
    fn add_assign(&mut self, rhs: Self) {
        self.state_clone += rhs.state_clone;
        self.block_pool_create += rhs.block_pool_create;
        self.ensure_block += rhs.ensure_block;
        self.fused_one_qubit += rhs.fused_one_qubit;
        self.original_dispatch += rhs.original_dispatch;
        self.single_qubit += rhs.single_qubit;
        self.multi_qubit += rhs.multi_qubit;
        self.measurement += rhs.measurement;
        self.reset += rhs.reset;
        self.conditional += rhs.conditional;
        self.format_counts_key += rhs.format_counts_key;
    }
}

#[derive(Default)]
pub struct ShotsProfile {
    pub mode: &'static str,
    pub shots: usize,
    pub qubits: usize,
    pub instructions: usize,
    pub fused_instructions: usize,
    pub py_extract: Duration,
    pub deserialize: Duration,
    pub setup: Duration,
    pub fusion: Duration,
    pub shot_loop_wall: Duration,
    pub result_dict: Duration,
    pub shot_stats: ShotStats,
}

pub fn enabled() -> bool {
    std::env::var_os("DQSIM_PROFILE_SHOTS").is_some()
}

#[inline]
pub fn elapsed_since(start: Instant) -> Duration {
    start.elapsed()
}

pub fn print(profile: &ShotsProfile) {
    eprintln!("{}", format(profile));
}

fn format(profile: &ShotsProfile) -> String {
    let total = profile.py_extract
        + profile.deserialize
        + profile.setup
        + profile.fusion
        + profile.shot_loop_wall
        + profile.result_dict;

    let mut out = String::new();
    let _ = writeln!(
        out,
        "\n[dqsim profile] mode={} qubits={} shots={} instructions={} fused={}",
        profile.mode,
        profile.qubits,
        profile.shots,
        profile.instructions,
        profile.fused_instructions
    );
    let _ = writeln!(out, "  top-level wall phases:");
    write_phase(&mut out, "python/extract", profile.py_extract, total, None);
    write_phase(&mut out, "deserialize", profile.deserialize, total, None);
    write_phase(&mut out, "setup", profile.setup, total, None);
    write_phase(&mut out, "fusion", profile.fusion, total, None);
    write_phase(
        &mut out,
        "shot_loop_wall",
        profile.shot_loop_wall,
        total,
        None,
    );
    write_phase(&mut out, "result_dict", profile.result_dict, total, None);

    let shot_total = profile.shot_stats.state_clone.elapsed
        + profile.shot_stats.block_pool_create.elapsed
        + profile.shot_stats.ensure_block.elapsed
        + profile.shot_stats.fused_one_qubit.elapsed
        + profile.shot_stats.single_qubit.elapsed
        + profile.shot_stats.multi_qubit.elapsed
        + profile.shot_stats.measurement.elapsed
        + profile.shot_stats.reset.elapsed
        + profile.shot_stats.conditional.elapsed
        + profile.shot_stats.format_counts_key.elapsed;

    let _ = writeln!(
        out,
        "  per-shot accumulated work across Rayon workers (can exceed wall time):"
    );
    write_phase(
        &mut out,
        "state_clone",
        profile.shot_stats.state_clone.elapsed,
        shot_total,
        Some(profile.shot_stats.state_clone.calls),
    );
    write_phase(
        &mut out,
        "block_pool_create",
        profile.shot_stats.block_pool_create.elapsed,
        shot_total,
        Some(profile.shot_stats.block_pool_create.calls),
    );
    write_phase(
        &mut out,
        "ensure_block",
        profile.shot_stats.ensure_block.elapsed,
        shot_total,
        Some(profile.shot_stats.ensure_block.calls),
    );
    write_phase(
        &mut out,
        "fused_one_qubit",
        profile.shot_stats.fused_one_qubit.elapsed,
        shot_total,
        Some(profile.shot_stats.fused_one_qubit.calls),
    );
    write_phase(
        &mut out,
        "single_qubit",
        profile.shot_stats.single_qubit.elapsed,
        shot_total,
        Some(profile.shot_stats.single_qubit.calls),
    );
    write_phase(
        &mut out,
        "multi_qubit",
        profile.shot_stats.multi_qubit.elapsed,
        shot_total,
        Some(profile.shot_stats.multi_qubit.calls),
    );
    write_phase(
        &mut out,
        "measurement",
        profile.shot_stats.measurement.elapsed,
        shot_total,
        Some(profile.shot_stats.measurement.calls),
    );
    write_phase(
        &mut out,
        "reset",
        profile.shot_stats.reset.elapsed,
        shot_total,
        Some(profile.shot_stats.reset.calls),
    );
    write_phase(
        &mut out,
        "conditional",
        profile.shot_stats.conditional.elapsed,
        shot_total,
        Some(profile.shot_stats.conditional.calls),
    );
    write_phase(
        &mut out,
        "format_key",
        profile.shot_stats.format_counts_key.elapsed,
        shot_total,
        Some(profile.shot_stats.format_counts_key.calls),
    );
    out
}

fn write_phase(
    out: &mut String,
    label: &str,
    elapsed: Duration,
    total: Duration,
    calls: Option<u64>,
) {
    let ms = elapsed.as_secs_f64() * 1_000.0;
    let pct = if total.is_zero() {
        0.0
    } else {
        elapsed.as_secs_f64() * 100.0 / total.as_secs_f64()
    };
    match calls {
        Some(calls) => {
            let per_call_us = if calls == 0 {
                0.0
            } else {
                elapsed.as_secs_f64() * 1_000_000.0 / calls as f64
            };
            let _ = writeln!(
                out,
                "    {label:<18} {ms:>10.3} ms {pct:>6.1}% {calls:>10} calls {per_call_us:>10.3} us/call"
            );
        }
        None => {
            let _ = writeln!(out, "    {label:<18} {ms:>10.3} ms {pct:>6.1}%");
        }
    }
}
