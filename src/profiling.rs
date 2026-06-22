use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;

/// Writes a `simulate_shots` profile to `./dqsim_profiles/<prefix>_<unix_ns>.json`,
/// creating the directory if needed. Used by the `profile=true` opt-in path on each
/// simulator's `simulate_shots`; never called on the default (profile=false) path.
pub fn write_shots_profile<T: Serialize>(prefix: &str, profile: &T) -> std::io::Result<()> {
    let dir = std::path::Path::new("dqsim_profiles");
    std::fs::create_dir_all(dir)?;
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let path = dir.join(format!("{prefix}_shots_profile_{ts}.json"));
    let json = serde_json::to_string_pretty(profile).map_err(std::io::Error::other)?;
    std::fs::write(path, json)
}
