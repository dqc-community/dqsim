use num_complex::Complex64;
use std::f64::consts::{FRAC_1_SQRT_2, PI};

pub type C = Complex64;
pub type Mat2 = [[C; 2]; 2];
pub type Mat4 = [[C; 4]; 4];
pub type Mat8 = [[C; 8]; 8];
pub type Mat16 = [[C; 16]; 16];
pub type Mat32 = [[C; 32]; 32];

#[inline]
pub const fn c(re: f64, im: f64) -> C {
    C::new(re, im)
}
#[inline]
pub const fn r(re: f64) -> C {
    C::new(re, 0.0)
}

// ---------------------------------------------------------------------------
// Fixed single-qubit gates
// ---------------------------------------------------------------------------

pub const X: Mat2 = [[r(0.0), r(1.0)], [r(1.0), r(0.0)]];

pub const Y: Mat2 = [[r(0.0), c(0.0, -1.0)], [c(0.0, 1.0), r(0.0)]];

pub const Z: Mat2 = [[r(1.0), r(0.0)], [r(0.0), r(-1.0)]];

pub fn h() -> Mat2 {
    let s = FRAC_1_SQRT_2;
    [[r(s), r(s)], [r(s), r(-s)]]
}

pub fn s_gate() -> Mat2 {
    [[r(1.0), r(0.0)], [r(0.0), c(0.0, 1.0)]]
}
pub fn sdg() -> Mat2 {
    [[r(1.0), r(0.0)], [r(0.0), c(0.0, -1.0)]]
}
pub fn t_gate() -> Mat2 {
    // T = diag(1, e^{iπ/4}) = diag(1, (1+i)/√2)
    let s = FRAC_1_SQRT_2;
    [[r(1.0), r(0.0)], [r(0.0), c(s, s)]]
}
pub fn tdg() -> Mat2 {
    // T† = diag(1, e^{-iπ/4}) = diag(1, (1-i)/√2)
    let s = FRAC_1_SQRT_2;
    [[r(1.0), r(0.0)], [r(0.0), c(s, -s)]]
}
pub fn sx() -> Mat2 {
    let a = c(0.5, 0.5);
    let b = c(0.5, -0.5);
    [[a, b], [b, a]]
}
pub fn sxdg() -> Mat2 {
    let a = c(0.5, -0.5);
    let b = c(0.5, 0.5);
    [[a, b], [b, a]]
}

// ---------------------------------------------------------------------------
// Parametric single-qubit gates
// ---------------------------------------------------------------------------

pub fn u3(theta: f64, phi: f64, lam: f64) -> Mat2 {
    let (ct, st) = ((theta / 2.0).cos(), (theta / 2.0).sin());
    [
        [r(ct), -(c(lam.cos(), lam.sin()) * r(st))],
        [
            c(phi.cos(), phi.sin()) * r(st),
            c((phi + lam).cos(), (phi + lam).sin()) * r(ct),
        ],
    ]
}
pub fn u2(phi: f64, lam: f64) -> Mat2 {
    u3(PI / 2.0, phi, lam)
}
pub fn u1(lam: f64) -> Mat2 {
    [[r(1.0), r(0.0)], [r(0.0), c(lam.cos(), lam.sin())]]
}
pub fn u(theta: f64, phi: f64, lam: f64) -> Mat2 {
    u3(theta, phi, lam)
}
pub fn p(lam: f64) -> Mat2 {
    u1(lam)
}

pub fn rx(theta: f64) -> Mat2 {
    let (c2, s2) = ((theta / 2.0).cos(), (theta / 2.0).sin());
    [[r(c2), c(0.0, -s2)], [c(0.0, -s2), r(c2)]]
}
pub fn ry(theta: f64) -> Mat2 {
    let (c2, s2) = ((theta / 2.0).cos(), (theta / 2.0).sin());
    [[r(c2), r(-s2)], [r(s2), r(c2)]]
}
pub fn rz(phi: f64) -> Mat2 {
    [
        [c((-phi / 2.0).cos(), (-phi / 2.0).sin()), r(0.0)],
        [r(0.0), c((phi / 2.0).cos(), (phi / 2.0).sin())],
    ]
}

// ---------------------------------------------------------------------------
// Fixed two-qubit gates (4×4, row-major, qubit[0]=MSB)
// ---------------------------------------------------------------------------

pub fn cnot() -> Mat4 {
    let o = r(0.0);
    let i = r(1.0);
    [[i, o, o, o], [o, i, o, o], [o, o, o, i], [o, o, i, o]]
}
pub fn cz() -> Mat4 {
    let o = r(0.0);
    let i = r(1.0);
    let n = r(-1.0);
    [[i, o, o, o], [o, i, o, o], [o, o, i, o], [o, o, o, n]]
}
pub fn cy() -> Mat4 {
    let o = r(0.0);
    let i = r(1.0);
    [
        [i, o, o, o],
        [o, i, o, o],
        [o, o, o, c(0.0, -1.0)],
        [o, o, c(0.0, 1.0), o],
    ]
}
pub fn ch() -> Mat4 {
    let s = FRAC_1_SQRT_2;
    let o = r(0.0);
    let i = r(1.0);
    [
        [i, o, o, o],
        [o, i, o, o],
        [o, o, r(s), r(s)],
        [o, o, r(s), r(-s)],
    ]
}
pub fn swap() -> Mat4 {
    let o = r(0.0);
    let i = r(1.0);
    [[i, o, o, o], [o, o, i, o], [o, i, o, o], [o, o, o, i]]
}
pub fn csx() -> Mat4 {
    let o = r(0.0);
    let i = r(1.0);
    let a = c(0.5, 0.5);
    let b = c(0.5, -0.5);
    [[i, o, o, o], [o, i, o, o], [o, o, a, b], [o, o, b, a]]
}

// ---------------------------------------------------------------------------
// Parametric two-qubit gates
// ---------------------------------------------------------------------------

fn controlled(u: Mat2) -> Mat4 {
    let o = r(0.0);
    let i = r(1.0);
    [
        [i, o, o, o],
        [o, i, o, o],
        [o, o, u[0][0], u[0][1]],
        [o, o, u[1][0], u[1][1]],
    ]
}

pub fn crx(theta: f64) -> Mat4 {
    controlled(rx(theta))
}
pub fn cry(theta: f64) -> Mat4 {
    controlled(ry(theta))
}
pub fn crz(lam: f64) -> Mat4 {
    controlled(rz(lam))
}
pub fn cu1(lam: f64) -> Mat4 {
    controlled(u1(lam))
}
pub fn cp(lam: f64) -> Mat4 {
    controlled(p(lam))
}
pub fn cu3(theta: f64, phi: f64, lam: f64) -> Mat4 {
    controlled(u3(theta, phi, lam))
}
pub fn cu(theta: f64, phi: f64, lam: f64, gamma: f64) -> Mat4 {
    let phase = c(gamma.cos(), gamma.sin());
    let inner = u3(theta, phi, lam);
    let phased = [
        [phase * inner[0][0], phase * inner[0][1]],
        [phase * inner[1][0], phase * inner[1][1]],
    ];
    controlled(phased)
}

pub fn rxx(theta: f64) -> Mat4 {
    let (cv, sv) = ((theta / 2.0).cos(), (theta / 2.0).sin());
    let o = r(0.0);
    [
        [r(cv), o, o, c(0.0, -sv)],
        [o, r(cv), c(0.0, -sv), o],
        [o, c(0.0, -sv), r(cv), o],
        [c(0.0, -sv), o, o, r(cv)],
    ]
}
pub fn rzz(theta: f64) -> Mat4 {
    let ep = c((theta / 2.0).cos(), (theta / 2.0).sin());
    let em = c((theta / 2.0).cos(), -(theta / 2.0).sin());
    let o = r(0.0);
    [[em, o, o, o], [o, ep, o, o], [o, o, ep, o], [o, o, o, em]]
}

// ---------------------------------------------------------------------------
// Fixed three-qubit gates (8×8)
// ---------------------------------------------------------------------------

pub fn ccx() -> Mat8 {
    let mut m = identity8();
    m[6][6] = r(0.0);
    m[7][7] = r(0.0);
    m[6][7] = r(1.0);
    m[7][6] = r(1.0);
    m
}
pub fn cswap() -> Mat8 {
    let mut m = identity8();
    m[5][5] = r(0.0);
    m[6][6] = r(0.0);
    m[5][6] = r(1.0);
    m[6][5] = r(1.0);
    m
}
pub fn rccx() -> Mat8 {
    let mut m = identity8();
    m[6][6] = r(0.0);
    m[7][7] = r(0.0);
    m[6][7] = c(0.0, 1.0);
    m[7][6] = c(0.0, -1.0);
    m
}

// ---------------------------------------------------------------------------
// Fixed four-qubit gates (16×16)
// ---------------------------------------------------------------------------

pub fn rc3x() -> Mat16 {
    let mut m = identity16();
    m[14][14] = c(0.0, 1.0);
    m[14][15] = r(0.0);
    m[15][14] = r(0.0);
    m[15][15] = c(0.0, -1.0);
    m
}
pub fn c3x() -> Mat16 {
    let mut m = identity16();
    m[14][14] = r(0.0);
    m[15][15] = r(0.0);
    m[14][15] = r(1.0);
    m[15][14] = r(1.0);
    m
}
pub fn c3sqrtx() -> Mat16 {
    let mut m = identity16();
    let sx = sx();
    m[14][14] = sx[0][0];
    m[14][15] = sx[0][1];
    m[15][14] = sx[1][0];
    m[15][15] = sx[1][1];
    m
}

// ---------------------------------------------------------------------------
// Fixed five-qubit gates (32×32)
// ---------------------------------------------------------------------------

pub fn c4x() -> Mat32 {
    let mut m = identity32();
    m[30][30] = r(0.0);
    m[31][31] = r(0.0);
    m[30][31] = r(1.0);
    m[31][30] = r(1.0);
    m
}

// ---------------------------------------------------------------------------
// Cross-node gates (hardcoded from RemoteLinkGatePsiMinus/Plus._REMOTE_MATRIX)
// ---------------------------------------------------------------------------

pub fn psi_minus() -> Mat4 {
    let s = FRAC_1_SQRT_2;
    [
        [r(0.0), r(s), r(0.0), r(s)],
        [r(s), r(0.0), r(s), r(0.0)],
        [r(-s), r(0.0), r(s), r(0.0)],
        [r(0.0), r(-s), r(0.0), r(s)],
    ]
}

pub fn psi_plus() -> Mat4 {
    // diag([1,-1]) ⊗ I₂  @  psi_minus
    let pm = psi_minus();
    let mut out = [[r(0.0); 4]; 4];
    // rows 0,1 unchanged; rows 2,3 negated
    for j in 0..4 {
        out[0][j] = pm[0][j];
        out[1][j] = pm[1][j];
        out[2][j] = -pm[2][j];
        out[3][j] = -pm[3][j];
    }
    out
}

pub fn phi_plus() -> Mat4 {
    // Bell state |Φ+⟩ preparation: (H⊗I) · CNOT applied to |00⟩
    // Matches bell_pair_phi_plus_matrix() in protocols.py
    let s = FRAC_1_SQRT_2;
    [
        [r(s), r(0.0), r(s), r(0.0)],
        [r(0.0), r(s), r(0.0), r(s)],
        [r(0.0), r(s), r(0.0), r(-s)],
        [r(s), r(0.0), r(-s), r(0.0)],
    ]
}

#[allow(dead_code)]
pub const NONLOCAL_CZ: Mat4 = {
    let o = r(0.0);
    let i = r(1.0);
    let n = r(-1.0);
    [[i, o, o, o], [o, i, o, o], [o, o, i, o], [o, o, o, n]]
};

// ---------------------------------------------------------------------------
// Identity helpers
// ---------------------------------------------------------------------------

fn identity8() -> Mat8 {
    let mut m = [[r(0.0); 8]; 8];
    for (i, row) in m.iter_mut().enumerate() {
        row[i] = r(1.0);
    }
    m
}
fn identity16() -> Mat16 {
    let mut m = [[r(0.0); 16]; 16];
    for (i, row) in m.iter_mut().enumerate() {
        row[i] = r(1.0);
    }
    m
}
fn identity32() -> Mat32 {
    let mut m = [[r(0.0); 32]; 32];
    for (i, row) in m.iter_mut().enumerate() {
        row[i] = r(1.0);
    }
    m
}
