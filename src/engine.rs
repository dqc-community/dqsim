use std::sync::OnceLock;

#[cfg(target_arch = "x86")]
use std::arch::x86::{
    __m256d, _mm256_add_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_permute_pd, _mm256_set1_pd,
    _mm256_setr_pd, _mm256_storeu_pd,
};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    __m256d, _mm256_add_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_permute_pd, _mm256_set1_pd,
    _mm256_setr_pd, _mm256_storeu_pd,
};

use num_complex::Complex64;
use rand::Rng;
use rayon::prelude::*;

type C = Complex64;

/// Parallelize when the statevector has at least 2^PAR_THRESHOLD amplitudes.
const PAR_THRESHOLD: usize = 12;

/// Maximum gate arity (C4x = 5 qubits → dim = 32).
const MAX_DIM: usize = 32;
const SIMD_PAR_CHUNK_COMPLEX: usize = 4096;
const DEFAULT_SIMD_MIN_QUBITS: usize = 12;
const DEFAULT_MULTI_CONTROL_KERNEL_MIN_QUBITS: usize = 8;

#[derive(Clone, Copy)]
struct OneQubitCoeffs {
    u00: C,
    u01: C,
    u10: C,
    u11: C,
}

impl OneQubitCoeffs {
    fn from_matrix(u: &[[C; 2]; 2]) -> Self {
        Self {
            u00: u[0][0],
            u01: u[0][1],
            u10: u[1][0],
            u11: u[1][1],
        }
    }
}

/// Returns true when the opt-in statevector SIMD path is active for this process.
///
/// SIMD is intentionally off by default so existing benchmark behavior is unchanged.
/// Set `DQSIM_STATEVECTOR_SIMD=1` to enable it for larger statevectors. Set
/// `DQSIM_STATEVECTOR_SIMD=force` to use it for every eligible one-qubit gate.
/// Unsupported CPUs/architectures fall back to the scalar implementation.
pub fn statevector_simd_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| statevector_simd_requested() && statevector_simd_supported())
}

pub fn statevector_simd_backend() -> Option<&'static str> {
    statevector_simd_enabled().then_some("avx")
}

fn statevector_simd_allowed_for_qubits(n: usize) -> bool {
    statevector_simd_enabled() && n >= statevector_simd_min_qubits()
}

pub fn statevector_simd_min_qubits() -> usize {
    static MIN_QUBITS: OnceLock<usize> = OnceLock::new();
    *MIN_QUBITS.get_or_init(|| {
        if statevector_simd_forced() {
            return 0;
        }
        std::env::var("DQSIM_STATEVECTOR_SIMD_MIN_QUBITS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(DEFAULT_SIMD_MIN_QUBITS)
    })
}

fn statevector_simd_requested() -> bool {
    let Ok(value) = std::env::var("DQSIM_STATEVECTOR_SIMD") else {
        return false;
    };
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on" | "auto" | "avx" | "avx2" | "force" | "always"
    )
}

fn statevector_simd_forced() -> bool {
    static FORCED: OnceLock<bool> = OnceLock::new();
    *FORCED.get_or_init(|| {
        let Ok(value) = std::env::var("DQSIM_STATEVECTOR_SIMD") else {
            return false;
        };
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "force" | "always"
        )
    })
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn statevector_simd_supported() -> bool {
    std::is_x86_feature_detected!("avx")
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn statevector_simd_supported() -> bool {
    false
}

/// Returns true when the opt-in specialized two-qubit statevector kernels are active.
pub fn statevector_two_qubit_kernels_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        let Ok(value) = std::env::var("DQSIM_STATEVECTOR_2Q_KERNELS") else {
            return false;
        };
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on" | "auto" | "force" | "always"
        )
    })
}

pub fn statevector_two_qubit_kernel_gates() -> &'static str {
    "cx,cz,cy,ch,swap,csx,crx,cry,crz,cu1,cp,cu3,cu,ccx,cswap,c3x,c3sqrtx,c4x"
}

pub fn statevector_multi_control_kernels_enabled_for(n: usize) -> bool {
    statevector_two_qubit_kernels_enabled() && n >= statevector_multi_control_kernel_min_qubits()
}

pub fn statevector_multi_control_kernel_min_qubits() -> usize {
    static MIN_QUBITS: OnceLock<usize> = OnceLock::new();
    *MIN_QUBITS.get_or_init(|| {
        std::env::var("DQSIM_STATEVECTOR_MULTI_CONTROL_MIN_QUBITS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(DEFAULT_MULTI_CONTROL_KERNEL_MIN_QUBITS)
    })
}

// ---------------------------------------------------------------------------
// apply_one_qubit
// Apply a 2×2 unitary to `target` qubit in-place.
// `controls` is a slice of (wire, must_be_one) pairs.
//
// Parallelism strategy: split state into blocks of `2*half` (each block holds
// exactly one (low, high) pair).  The outer par_chunks_mut handles many blocks
// (small target qubit); the inner par_iter_mut zip handles large blocks (large
// target qubit).  This keeps access contiguous at both levels.
// ---------------------------------------------------------------------------

pub fn apply_one_qubit(
    state: &mut [C],
    u: &[[C; 2]; 2],
    target: usize,
    n: usize,
    controls: &[(usize, bool)],
) {
    let inclusion_mask: usize = controls.iter().fold(0, |m, &(w, _)| m | (1 << w));
    let desired_mask: usize = controls
        .iter()
        .fold(0, |m, &(w, flag)| if flag { m | (1 << w) } else { m });
    let half = 1 << target;
    let block = 2 * half;
    let dim = 1 << n;

    let u00 = u[0][0];
    let u01 = u[0][1];
    let u10 = u[1][0];
    let u11 = u[1][1];

    if inclusion_mask == 0 && statevector_simd_allowed_for_qubits(n) {
        apply_one_qubit_uncontrolled_simd(state, OneQubitCoeffs::from_matrix(u), target, n);
        return;
    }

    if n >= PAR_THRESHOLD {
        // Split state into contiguous blocks of size `block = 2*half`.
        // Each block's lower half maps to gate input 0, upper half to input 1.
        // Outer par_chunks_mut gives parallelism for small target qubits (many blocks).
        // Inner par_iter_mut zip gives parallelism for large target qubits (large blocks).
        state
            .par_chunks_mut(block)
            .enumerate()
            .for_each(|(ci, chunk)| {
                let base_i = ci * block;
                let (lo, hi) = chunk.split_at_mut(half);

                if lo.len() >= 1024 {
                    // Large block — too few outer chunks for good parallelism, use inner.
                    if inclusion_mask == 0 {
                        lo.par_iter_mut()
                            .zip(hi.par_iter_mut())
                            .for_each(|(a_r, b_r)| {
                                let a = *a_r;
                                let b = *b_r;
                                *a_r = u00 * a + u01 * b;
                                *b_r = u10 * a + u11 * b;
                            });
                    } else {
                        lo.par_iter_mut()
                            .zip(hi.par_iter_mut())
                            .enumerate()
                            .for_each(|(k, (a_r, b_r))| {
                                let i = base_i + k;
                                if (i & inclusion_mask) == desired_mask {
                                    let a = *a_r;
                                    let b = *b_r;
                                    *a_r = u00 * a + u01 * b;
                                    *b_r = u10 * a + u11 * b;
                                }
                            });
                    }
                } else {
                    // Small block — outer loop provides enough parallelism; sequential inner.
                    for (k, (a_r, b_r)) in lo.iter_mut().zip(hi.iter_mut()).enumerate() {
                        let i = base_i + k;
                        if inclusion_mask == 0 || (i & inclusion_mask) == desired_mask {
                            let a = *a_r;
                            let b = *b_r;
                            *a_r = u00 * a + u01 * b;
                            *b_r = u10 * a + u11 * b;
                        }
                    }
                }
            });
    } else {
        let mut i = 0;
        while i < dim {
            if (i & half) == 0 && (inclusion_mask == 0 || (i & inclusion_mask) == desired_mask) {
                let j = i | half;
                let a = state[i];
                let b = state[j];
                state[i] = u00 * a + u01 * b;
                state[j] = u10 * a + u11 * b;
            }
            i += 1;
        }
    }
}

fn apply_one_qubit_uncontrolled_simd(
    state: &mut [C],
    coeffs: OneQubitCoeffs,
    target: usize,
    n: usize,
) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        let half = 1 << target;
        let block = 2 * half;

        if n >= PAR_THRESHOLD {
            state.par_chunks_mut(block).for_each(|chunk| {
                let (lo, hi) = chunk.split_at_mut(half);
                if lo.len() >= SIMD_PAR_CHUNK_COMPLEX {
                    lo.par_chunks_mut(SIMD_PAR_CHUNK_COMPLEX)
                        .zip(hi.par_chunks_mut(SIMD_PAR_CHUNK_COMPLEX))
                        .for_each(|(lo_chunk, hi_chunk)| unsafe {
                            apply_one_qubit_slices_avx(lo_chunk, hi_chunk, coeffs);
                        });
                } else {
                    unsafe { apply_one_qubit_slices_avx(lo, hi, coeffs) };
                }
            });
        } else {
            for chunk in state.chunks_mut(block) {
                let (lo, hi) = chunk.split_at_mut(half);
                unsafe { apply_one_qubit_slices_avx(lo, hi, coeffs) };
            }
        }
        return;
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        let half = 1 << target;
        let block = 2 * half;
        for chunk in state.chunks_mut(block) {
            let (lo, hi) = chunk.split_at_mut(half);
            apply_one_qubit_slices_scalar(lo, hi, coeffs);
        }
    }
}

fn apply_one_qubit_uncontrolled_simd_seq(state: &mut [C], coeffs: OneQubitCoeffs, target: usize) {
    let half = 1 << target;
    let block = 2 * half;

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        for chunk in state.chunks_mut(block) {
            let (lo, hi) = chunk.split_at_mut(half);
            unsafe { apply_one_qubit_slices_avx(lo, hi, coeffs) };
        }
        return;
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        for chunk in state.chunks_mut(block) {
            let (lo, hi) = chunk.split_at_mut(half);
            apply_one_qubit_slices_scalar(lo, hi, coeffs);
        }
    }
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn apply_one_qubit_slices_scalar(lo: &mut [C], hi: &mut [C], coeffs: OneQubitCoeffs) {
    for (a_r, b_r) in lo.iter_mut().zip(hi.iter_mut()) {
        let a = *a_r;
        let b = *b_r;
        *a_r = coeffs.u00 * a + coeffs.u01 * b;
        *b_r = coeffs.u10 * a + coeffs.u11 * b;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
unsafe fn apply_one_qubit_slices_avx(lo: &mut [C], hi: &mut [C], coeffs: OneQubitCoeffs) {
    let len = lo.len().min(hi.len());
    let simd_len = len & !1;
    let mut k = 0;

    let u00_re = _mm256_set1_pd(coeffs.u00.re);
    let u00_im = _mm256_setr_pd(-coeffs.u00.im, coeffs.u00.im, -coeffs.u00.im, coeffs.u00.im);
    let u01_re = _mm256_set1_pd(coeffs.u01.re);
    let u01_im = _mm256_setr_pd(-coeffs.u01.im, coeffs.u01.im, -coeffs.u01.im, coeffs.u01.im);
    let u10_re = _mm256_set1_pd(coeffs.u10.re);
    let u10_im = _mm256_setr_pd(-coeffs.u10.im, coeffs.u10.im, -coeffs.u10.im, coeffs.u10.im);
    let u11_re = _mm256_set1_pd(coeffs.u11.re);
    let u11_im = _mm256_setr_pd(-coeffs.u11.im, coeffs.u11.im, -coeffs.u11.im, coeffs.u11.im);

    while k < simd_len {
        let a = _mm256_loadu_pd(lo.as_ptr().add(k).cast::<f64>());
        let b = _mm256_loadu_pd(hi.as_ptr().add(k).cast::<f64>());

        let next_lo = _mm256_add_pd(
            complex_mul_scalar_avx(a, u00_re, u00_im),
            complex_mul_scalar_avx(b, u01_re, u01_im),
        );
        let next_hi = _mm256_add_pd(
            complex_mul_scalar_avx(a, u10_re, u10_im),
            complex_mul_scalar_avx(b, u11_re, u11_im),
        );

        _mm256_storeu_pd(lo.as_mut_ptr().add(k).cast::<f64>(), next_lo);
        _mm256_storeu_pd(hi.as_mut_ptr().add(k).cast::<f64>(), next_hi);
        k += 2;
    }

    while k < len {
        let a = lo[k];
        let b = hi[k];
        lo[k] = coeffs.u00 * a + coeffs.u01 * b;
        hi[k] = coeffs.u10 * a + coeffs.u11 * b;
        k += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
unsafe fn complex_mul_scalar_avx(z: __m256d, re: __m256d, im: __m256d) -> __m256d {
    let real_part = _mm256_mul_pd(z, re);
    let swapped = _mm256_permute_pd(z, 0b0101);
    let imag_part = _mm256_mul_pd(swapped, im);
    _mm256_add_pd(real_part, imag_part)
}

// ---------------------------------------------------------------------------
// apply_n_qubit
// Apply a 2^k × 2^k unitary to k arbitrary qubits in-place.
// qubits[0] = MSB of the gate's index space, qubits[k-1] = LSB.
// ---------------------------------------------------------------------------

pub fn apply_n_qubit(state: &mut [C], u: &[Vec<C>], qubits: &[usize], n: usize) {
    let k = qubits.len();
    let dim = 1 << k;
    let n_states = 1 << n;
    let mask: usize = qubits.iter().fold(0, |m, &q| m | (1 << q));

    // Pre-compute target-bit offsets once — the same for every base index.
    let mut offsets = [0usize; MAX_DIM];
    for (j, offset_slot) in offsets.iter_mut().enumerate().take(dim) {
        let mut offset = 0usize;
        for (bit, &q) in qubits.iter().enumerate() {
            if (j >> (k - 1 - bit)) & 1 == 1 {
                offset |= 1 << q;
            }
        }
        *offset_slot = offset;
    }

    if n >= PAR_THRESHOLD {
        // Safety: different base values produce fully disjoint index sets.
        // Each base has all target-qubit bits = 0; offsets fill in all 2^k
        // combinations of those bits. Two distinct bases differ in a non-target
        // bit, so their index sets cannot overlap.
        let ptr = state.as_mut_ptr() as usize;
        (0..n_states)
            .into_par_iter()
            .filter(|&i| (i & mask) == 0)
            .for_each(|base| {
                let p = ptr as *mut C;
                let mut v = [C::new(0.0, 0.0); MAX_DIM];
                let mut w = [C::new(0.0, 0.0); MAX_DIM];
                let mut idx = [0usize; MAX_DIM];

                for j in 0..dim {
                    idx[j] = base + offsets[j];
                    v[j] = unsafe { *p.add(idx[j]) };
                }
                for row in 0..dim {
                    let mut acc = C::new(0.0, 0.0);
                    for col in 0..dim {
                        acc += u[row][col] * v[col];
                    }
                    w[row] = acc;
                }
                for j in 0..dim {
                    unsafe { *p.add(idx[j]) = w[j] };
                }
            });
    } else {
        let mut v = vec![C::new(0.0, 0.0); dim];
        let mut w = vec![C::new(0.0, 0.0); dim];
        let mut idx = vec![0usize; dim];

        for base in (0..n_states).filter(|&i| (i & mask) == 0) {
            for j in 0..dim {
                idx[j] = base + offsets[j];
            }
            for j in 0..dim {
                v[j] = state[idx[j]];
            }
            for row in 0..dim {
                let mut acc = C::new(0.0, 0.0);
                for col in 0..dim {
                    acc += u[row][col] * v[col];
                }
                w[row] = acc;
            }
            for j in 0..dim {
                state[idx[j]] = w[j];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// measure_qubit
// Collapse the statevector on `qubit`, renormalise, return outcome (0 or 1).
//
// Parallelism strategy: elements with (i & bit) != 0 form contiguous runs of
// length `bit` interleaved with runs of equal length where the bit is clear.
// par_chunks over blocks of `2*bit` lets us sum/zero/scale each run without
// any scatter/filter overhead and with fully contiguous memory access.
// ---------------------------------------------------------------------------

pub fn measure_qubit<R: Rng>(state: &mut [C], qubit: usize, n: usize, rng: &mut R) -> u8 {
    let n_states = 1 << n;
    let bit = 1 << qubit;

    let p1: f64 = if n >= PAR_THRESHOLD {
        // Each block of 2*bit elements has its upper half with the bit set.
        state
            .par_chunks(2 * bit)
            .map(|chunk| chunk[bit..].iter().map(|c| c.norm_sqr()).sum::<f64>())
            .sum()
    } else {
        (0..n_states)
            .filter(|&i| (i & bit) != 0)
            .map(|i| state[i].norm_sqr())
            .sum()
    };

    let outcome = if rng.gen::<f64>() < p1 { 1u8 } else { 0u8 };

    if n >= PAR_THRESHOLD {
        if outcome == 1 {
            let norm = p1.sqrt().max(1e-15);
            state.par_chunks_mut(2 * bit).for_each(|chunk| {
                let (lo, hi) = chunk.split_at_mut(bit);
                for c in lo {
                    *c = C::new(0.0, 0.0);
                }
                for c in hi {
                    *c /= norm;
                }
            });
        } else {
            let norm = (1.0 - p1).max(0.0).sqrt().max(1e-15);
            state.par_chunks_mut(2 * bit).for_each(|chunk| {
                let (lo, hi) = chunk.split_at_mut(bit);
                for c in hi {
                    *c = C::new(0.0, 0.0);
                }
                for c in lo {
                    *c /= norm;
                }
            });
        }
    } else {
        if outcome == 1 {
            let norm = p1.sqrt().max(1e-15);
            for i in (0..n_states).filter(|&i| (i & bit) == 0) {
                state[i] = C::new(0.0, 0.0);
            }
            for i in (0..n_states).filter(|&i| (i & bit) != 0) {
                state[i] /= norm;
            }
        } else {
            let norm = (1.0 - p1).max(0.0).sqrt().max(1e-15);
            for i in (0..n_states).filter(|&i| (i & bit) != 0) {
                state[i] = C::new(0.0, 0.0);
            }
            for i in (0..n_states).filter(|&i| (i & bit) == 0) {
                state[i] /= norm;
            }
        }
    }

    outcome
}

// ---------------------------------------------------------------------------
// Sequential variants — always single-threaded, for use inside parallel shot loops
// to avoid nested Rayon thread pool contention.
// ---------------------------------------------------------------------------

/// Sequential apply_one_qubit — always single-threaded, for use inside parallel shot loops.
pub fn apply_one_qubit_seq(
    state: &mut [C],
    u: &[[C; 2]; 2],
    target: usize,
    n: usize,
    controls: &[(usize, bool)],
) {
    let inclusion_mask: usize = controls.iter().fold(0, |m, &(w, _)| m | (1 << w));
    let desired_mask: usize = controls
        .iter()
        .fold(0, |m, &(w, flag)| if flag { m | (1 << w) } else { m });
    let half = 1 << target;
    let dim = 1 << n;
    let u00 = u[0][0];
    let u01 = u[0][1];
    let u10 = u[1][0];
    let u11 = u[1][1];

    if inclusion_mask == 0 && statevector_simd_allowed_for_qubits(n) {
        apply_one_qubit_uncontrolled_simd_seq(state, OneQubitCoeffs::from_matrix(u), target);
        return;
    }

    let mut i = 0;
    while i < dim {
        if (i & half) == 0 && (inclusion_mask == 0 || (i & inclusion_mask) == desired_mask) {
            let j = i | half;
            let a = state[i];
            let b = state[j];
            state[i] = u00 * a + u01 * b;
            state[j] = u10 * a + u11 * b;
        }
        i += 1;
    }
}

/// Sequential apply_n_qubit — always single-threaded.
pub fn apply_n_qubit_seq(state: &mut [C], u: &[Vec<C>], qubits: &[usize], n: usize) {
    let k = qubits.len();
    let dim = 1 << k;
    let n_states = 1 << n;
    let mask: usize = qubits.iter().fold(0, |m, &q| m | (1 << q));
    let mut offsets = [0usize; MAX_DIM];
    for (j, offset_slot) in offsets.iter_mut().enumerate().take(dim) {
        let mut offset = 0usize;
        for (bit, &q) in qubits.iter().enumerate() {
            if (j >> (k - 1 - bit)) & 1 == 1 {
                offset |= 1 << q;
            }
        }
        *offset_slot = offset;
    }
    let mut v = vec![C::new(0.0, 0.0); dim];
    let mut w = vec![C::new(0.0, 0.0); dim];
    let mut idx = vec![0usize; dim];
    for base in (0..n_states).filter(|&i| (i & mask) == 0) {
        for j in 0..dim {
            idx[j] = base + offsets[j];
        }
        for j in 0..dim {
            v[j] = state[idx[j]];
        }
        for row in 0..dim {
            let mut acc = C::new(0.0, 0.0);
            for col in 0..dim {
                acc += u[row][col] * v[col];
            }
            w[row] = acc;
        }
        for j in 0..dim {
            state[idx[j]] = w[j];
        }
    }
}

/// Specialized multi-controlled X kernel. Swaps target-bit amplitude pairs only
/// when every control bit is set.
pub fn apply_multi_controlled_x_kernel(
    state: &mut [C],
    controls: &[usize],
    target: usize,
    n: usize,
) {
    if controls.iter().any(|&control| control == target) {
        return;
    }

    if n >= PAR_THRESHOLD {
        let control_mask = controls
            .iter()
            .fold(0usize, |mask, &q| mask | (1usize << q));
        let target_bit = 1usize << target;
        let n_states = 1usize << n;
        let ptr = state.as_mut_ptr() as usize;

        (0..n_states)
            .into_par_iter()
            .filter(|&i| (i & target_bit) == 0 && (i & control_mask) == control_mask)
            .for_each(|i| unsafe {
                let p = ptr as *mut C;
                std::ptr::swap(p.add(i), p.add(i | target_bit));
            });
    } else {
        apply_multi_controlled_x_kernel_seq(state, controls, target, n);
    }
}

pub fn apply_multi_controlled_x_kernel_seq(
    state: &mut [C],
    controls: &[usize],
    target: usize,
    n: usize,
) {
    if controls.iter().any(|&control| control == target) {
        return;
    }

    let control_mask = controls
        .iter()
        .fold(0usize, |mask, &q| mask | (1usize << q));
    let target_bit = 1usize << target;
    let n_states = 1usize << n;

    for i in 0..n_states {
        if (i & target_bit) == 0 && (i & control_mask) == control_mask {
            state.swap(i, i | target_bit);
        }
    }
}

/// Specialized CNOT kernel. Swaps target-bit amplitude pairs only when control is 1.
pub fn apply_cx_kernel(state: &mut [C], control: usize, target: usize, n: usize) {
    if control == target {
        return;
    }
    if n >= PAR_THRESHOLD {
        let control_bit = 1usize << control;
        let target_bit = 1usize << target;
        let n_states = 1usize << n;
        let ptr = state.as_mut_ptr() as usize;

        (0..n_states)
            .into_par_iter()
            .filter(|&i| (i & control_bit) != 0 && (i & target_bit) == 0)
            .for_each(|i| unsafe {
                let p = ptr as *mut C;
                std::ptr::swap(p.add(i), p.add(i | target_bit));
            });
    } else {
        apply_cx_kernel_seq(state, control, target, n);
    }
}

pub fn apply_cx_kernel_seq(state: &mut [C], control: usize, target: usize, n: usize) {
    if control == target {
        return;
    }
    let control_bit = 1usize << control;
    let target_bit = 1usize << target;
    let n_states = 1usize << n;

    for i in 0..n_states {
        if (i & control_bit) != 0 && (i & target_bit) == 0 {
            state.swap(i, i | target_bit);
        }
    }
}

/// Specialized CZ kernel. Only the |11> subspace picks up a minus sign.
pub fn apply_cz_kernel(state: &mut [C], control: usize, target: usize, n: usize) {
    if control == target {
        return;
    }
    let control_bit = 1usize << control;
    let target_bit = 1usize << target;
    let both_bits = control_bit | target_bit;

    if n >= PAR_THRESHOLD {
        state.par_iter_mut().enumerate().for_each(|(i, amp)| {
            if (i & both_bits) == both_bits {
                *amp = -*amp;
            }
        });
    } else {
        apply_cz_kernel_seq(state, control, target, n);
    }
}

pub fn apply_cz_kernel_seq(state: &mut [C], control: usize, target: usize, n: usize) {
    if control == target {
        return;
    }
    let control_bit = 1usize << control;
    let target_bit = 1usize << target;
    let both_bits = control_bit | target_bit;
    let n_states = 1usize << n;

    for (i, amp) in state.iter_mut().enumerate().take(n_states) {
        if (i & both_bits) == both_bits {
            *amp = -*amp;
        }
    }
}

/// Specialized SWAP kernel. Exchanges amplitudes whose two target bits differ.
pub fn apply_swap_kernel(state: &mut [C], a: usize, b: usize, n: usize) {
    if a == b {
        return;
    }
    if n >= PAR_THRESHOLD {
        let a_bit = 1usize << a;
        let b_bit = 1usize << b;
        let n_states = 1usize << n;
        let ptr = state.as_mut_ptr() as usize;

        (0..n_states)
            .into_par_iter()
            .filter(|&i| (i & a_bit) == 0 && (i & b_bit) != 0)
            .for_each(|i| unsafe {
                let p = ptr as *mut C;
                let j = (i | a_bit) & !b_bit;
                std::ptr::swap(p.add(i), p.add(j));
            });
    } else {
        apply_swap_kernel_seq(state, a, b, n);
    }
}

pub fn apply_swap_kernel_seq(state: &mut [C], a: usize, b: usize, n: usize) {
    if a == b {
        return;
    }
    let a_bit = 1usize << a;
    let b_bit = 1usize << b;
    let n_states = 1usize << n;

    for i in 0..n_states {
        if (i & a_bit) == 0 && (i & b_bit) != 0 {
            let j = (i | a_bit) & !b_bit;
            state.swap(i, j);
        }
    }
}

/// Specialized controlled-SWAP kernel. Exchanges target amplitudes only when
/// the control bit is set.
pub fn apply_cswap_kernel(state: &mut [C], control: usize, a: usize, b: usize, n: usize) {
    if control == a || control == b || a == b {
        return;
    }
    if n >= PAR_THRESHOLD {
        let control_bit = 1usize << control;
        let a_bit = 1usize << a;
        let b_bit = 1usize << b;
        let n_states = 1usize << n;
        let ptr = state.as_mut_ptr() as usize;

        (0..n_states)
            .into_par_iter()
            .filter(|&i| (i & control_bit) != 0 && (i & a_bit) == 0 && (i & b_bit) != 0)
            .for_each(|i| unsafe {
                let p = ptr as *mut C;
                let j = (i | a_bit) & !b_bit;
                std::ptr::swap(p.add(i), p.add(j));
            });
    } else {
        apply_cswap_kernel_seq(state, control, a, b, n);
    }
}

pub fn apply_cswap_kernel_seq(state: &mut [C], control: usize, a: usize, b: usize, n: usize) {
    if control == a || control == b || a == b {
        return;
    }
    let control_bit = 1usize << control;
    let a_bit = 1usize << a;
    let b_bit = 1usize << b;
    let n_states = 1usize << n;

    for i in 0..n_states {
        if (i & control_bit) != 0 && (i & a_bit) == 0 && (i & b_bit) != 0 {
            let j = (i | a_bit) & !b_bit;
            state.swap(i, j);
        }
    }
}
/// Sequential measure_qubit — always single-threaded.
pub fn measure_qubit_seq<R: Rng>(state: &mut [C], qubit: usize, n: usize, rng: &mut R) -> u8 {
    let n_states = 1 << n;
    let bit = 1 << qubit;
    let p1: f64 = (0..n_states)
        .filter(|&i| (i & bit) != 0)
        .map(|i| state[i].norm_sqr())
        .sum();
    let outcome = if rng.gen::<f64>() < p1 { 1u8 } else { 0u8 };
    if outcome == 1 {
        let norm = p1.sqrt().max(1e-15);
        for i in (0..n_states).filter(|&i| (i & bit) == 0) {
            state[i] = C::new(0.0, 0.0);
        }
        for i in (0..n_states).filter(|&i| (i & bit) != 0) {
            state[i] /= norm;
        }
    } else {
        let norm = (1.0 - p1).max(0.0).sqrt().max(1e-15);
        for i in (0..n_states).filter(|&i| (i & bit) != 0) {
            state[i] = C::new(0.0, 0.0);
        }
        for i in (0..n_states).filter(|&i| (i & bit) == 0) {
            state[i] /= norm;
        }
    }
    outcome
}

// ---------------------------------------------------------------------------
// marginal_probs
// Returns a Vec of length 2^k giving probabilities for each basis state of
// the specified qubits. qubits[0] = MSB of output index.
// ---------------------------------------------------------------------------

pub fn marginal_probs(state: &[C], n: usize, qubits: &[usize]) -> Vec<f64> {
    let k = qubits.len();
    let dim = 1 << k;
    let n_states = (1usize << n).min(state.len());
    let mut probs = vec![0.0f64; dim];

    for (basis, amp) in state.iter().take(n_states).enumerate() {
        let mut outcome = 0usize;
        for (bit, &q) in qubits.iter().enumerate() {
            if ((basis >> q) & 1) == 1 {
                outcome |= 1 << (k - 1 - bit);
            }
        }
        probs[outcome] += amp.norm_sqr();
    }

    probs
}

// ---------------------------------------------------------------------------
// sample_counts
// Sample `shots` outcomes from the state distribution.
// Returns a HashMap of (bitstring, count) pairs.
// ---------------------------------------------------------------------------

pub fn sample_counts<R: Rng>(
    state: &[C],
    n: usize,
    shots: usize,
    rng: &mut R,
    qubits: Option<&[usize]>,
) -> std::collections::HashMap<String, usize> {
    let all_qubits: Vec<usize>;
    let q = match qubits {
        Some(qs) => qs,
        None => {
            all_qubits = (0..n).rev().collect();
            &all_qubits
        }
    };

    let k = q.len();
    let probs = marginal_probs(state, n, q);

    let mut cdf = vec![0.0f64; probs.len()];
    let mut acc = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        acc += p;
        cdf[i] = acc;
    }
    *cdf.last_mut().unwrap() = 1.0;

    let mut sampled = vec![0usize; cdf.len()];
    for _ in 0..shots {
        let r: f64 = rng.gen();
        let idx = cdf.partition_point(|&c| c < r).min((1 << k) - 1);
        sampled[idx] += 1;
    }

    let mut counts = std::collections::HashMap::new();
    for (idx, count) in sampled.into_iter().enumerate() {
        if count == 0 {
            continue;
        }
        let bits = format!("{:0>width$b}", idx, width = k);
        counts.insert(bits, count);
    }
    counts
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    fn assert_close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}");
    }

    fn mat_to_vec<const N: usize>(m: [[C; N]; N]) -> Vec<Vec<C>> {
        m.into_iter().map(|row| row.to_vec()).collect()
    }

    fn sample_state(n: usize) -> Vec<C> {
        (0..(1usize << n))
            .map(|i| C::new((i as f64 + 1.0) / 100.0, ((i % 7) as f64 - 3.0) / 100.0))
            .collect()
    }

    fn assert_state_close(actual: &[C], expected: &[C]) {
        assert_eq!(actual.len(), expected.len());
        for (i, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            let diff = (*actual - *expected).norm();
            assert!(
                diff < 1e-12,
                "state[{i}] differs: {actual:?} != {expected:?}"
            );
        }
    }

    #[test]
    fn marginal_probs_accumulates_in_one_pass_order() {
        let mut state = vec![C::new(0.0, 0.0); 8];
        state[0b000] = C::new(0.1_f64.sqrt(), 0.0);
        state[0b101] = C::new(0.2_f64.sqrt(), 0.0);
        state[0b011] = C::new(0.3_f64.sqrt(), 0.0);
        state[0b111] = C::new(0.4_f64.sqrt(), 0.0);

        let probs = marginal_probs(&state, 3, &[2, 0]);

        assert_eq!(probs.len(), 4);
        assert_close(probs[0b00], 0.1);
        assert_close(probs[0b01], 0.3);
        assert_close(probs[0b10], 0.0);
        assert_close(probs[0b11], 0.6);
    }

    #[test]
    fn multi_controlled_x_kernels_match_generic_matrices() {
        let n = 5;

        let mut fast = sample_state(n);
        let mut generic = fast.clone();
        apply_multi_controlled_x_kernel_seq(&mut fast, &[3, 1], 4, n);
        apply_n_qubit_seq(
            &mut generic,
            &mat_to_vec(crate::gates::ccx()),
            &[3, 1, 4],
            n,
        );
        assert_state_close(&fast, &generic);

        let mut fast = sample_state(n);
        let mut generic = fast.clone();
        apply_multi_controlled_x_kernel_seq(&mut fast, &[0, 2, 3], 4, n);
        apply_n_qubit_seq(
            &mut generic,
            &mat_to_vec(crate::gates::c3x()),
            &[0, 2, 3, 4],
            n,
        );
        assert_state_close(&fast, &generic);

        let mut fast = sample_state(n);
        let mut generic = fast.clone();
        apply_multi_controlled_x_kernel_seq(&mut fast, &[0, 1, 2, 3], 4, n);
        apply_n_qubit_seq(
            &mut generic,
            &mat_to_vec(crate::gates::c4x()),
            &[0, 1, 2, 3, 4],
            n,
        );
        assert_state_close(&fast, &generic);
    }

    #[test]
    fn c3sqrtx_kernel_matches_generic_matrix() {
        let n = 5;
        let sx = crate::gates::sx();
        let mut fast = sample_state(n);
        let mut generic = fast.clone();

        apply_one_qubit_seq(&mut fast, &sx, 4, n, &[(0, true), (2, true), (3, true)]);
        apply_n_qubit_seq(
            &mut generic,
            &mat_to_vec(crate::gates::c3sqrtx()),
            &[0, 2, 3, 4],
            n,
        );

        assert_state_close(&fast, &generic);
    }

    #[test]
    fn controlled_swap_kernel_matches_generic_matrix() {
        let n = PAR_THRESHOLD;
        let mut fast = sample_state(n);
        let mut generic = fast.clone();

        apply_cswap_kernel(&mut fast, 9, 1, 10, n);
        apply_n_qubit_seq(
            &mut generic,
            &mat_to_vec(crate::gates::cswap()),
            &[9, 1, 10],
            n,
        );

        assert_state_close(&fast, &generic);
    }
}
