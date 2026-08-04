use std::collections::HashMap;
use serde::Deserialize;
use num_complex::Complex64;

type C = Complex64;

#[derive(Deserialize, Clone)]
pub struct Register {
    #[allow(dead_code)]
    pub name: String,
    pub size: usize,
    pub base: usize,
}

#[derive(Deserialize, Clone)]
pub struct Circuit {
    pub qregs: HashMap<String, Register>,
    #[serde(default)]
    pub cregs: HashMap<String, Register>,
    pub instructions: Vec<Instruction>,
}

impl Circuit {
    pub fn num_qubits(&self) -> usize {
        self.qregs
            .values()
            .map(|r| r.base + r.size)
            .max()
            .unwrap_or(0)
    }

    pub fn num_cbits(&self) -> usize {
        self.cregs
            .values()
            .map(|r| r.base + r.size)
            .max()
            .unwrap_or(0)
    }
}

/// Format a cbits map as a bitstring with MSB first (Qiskit convention).
/// Bits not written during simulation default to 0.
pub fn format_cbits(cbits: &HashMap<usize, i32>, num_cbits: usize) -> String {
    if num_cbits == 0 {
        return String::new();
    }
    (0..num_cbits)
        .rev()
        .map(|i| cbits.get(&i).copied().unwrap_or(0).to_string())
        .collect()
}

#[derive(Deserialize, Clone)]
pub struct Condition {
    pub creg_base: usize,
    pub creg_size: usize,
    pub creg_value: u64,
}

#[derive(Deserialize, Clone)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Instruction {
    // -----------------------------------------------------------------------
    // Single-qubit fixed
    // -----------------------------------------------------------------------
    Id     { qubit: usize },
    X      { qubit: usize },
    Y      { qubit: usize },
    Z      { qubit: usize },
    H      { qubit: usize },
    S      { qubit: usize },
    Sdg    { qubit: usize },
    T      { qubit: usize },
    Tdg    { qubit: usize },
    Sx     { qubit: usize },
    Sxdg   { qubit: usize },
    // -----------------------------------------------------------------------
    // Single-qubit parametric
    // -----------------------------------------------------------------------
    U3  { qubit: usize, theta: f64, phi: f64, lam: f64 },
    U2  { qubit: usize, phi: f64, lam: f64 },
    U1  { qubit: usize, lam: f64 },
    U   { qubit: usize, theta: f64, phi: f64, lam: f64 },
    P   { qubit: usize, lam: f64 },
    Rx  { qubit: usize, theta: f64 },
    Ry  { qubit: usize, theta: f64 },
    Rz  { qubit: usize, phi: f64 },
    U0  { qubit: usize },
    // -----------------------------------------------------------------------
    // Two-qubit fixed
    // -----------------------------------------------------------------------
    Cx   { control: usize, target: usize },
    Cz   { control: usize, target: usize },
    Cy   { control: usize, target: usize },
    Ch   { control: usize, target: usize },
    Swap { a: usize, b: usize },
    Csx  { control: usize, target: usize },
    // -----------------------------------------------------------------------
    // Two-qubit parametric
    // -----------------------------------------------------------------------
    Crx { control: usize, target: usize, theta: f64 },
    Cry { control: usize, target: usize, theta: f64 },
    Crz { control: usize, target: usize, lam: f64 },
    Cu1 { control: usize, target: usize, lam: f64 },
    Cp  { control: usize, target: usize, lam: f64 },
    Cu3 { control: usize, target: usize, theta: f64, phi: f64, lam: f64 },
    Cu  { control: usize, target: usize, theta: f64, phi: f64, lam: f64, gamma: f64 },
    Rxx { a: usize, b: usize, theta: f64 },
    Rzz { a: usize, b: usize, theta: f64 },
    // -----------------------------------------------------------------------
    // Three-qubit
    // -----------------------------------------------------------------------
    Ccx     { control1: usize, control2: usize, target: usize },
    Cswap   { control: usize, target1: usize, target2: usize },
    Rccx    { control1: usize, control2: usize, target: usize },
    Rc3x    { control1: usize, control2: usize, control3: usize, target: usize },
    C3x     { control1: usize, control2: usize, control3: usize, target: usize },
    C3sqrtx { control1: usize, control2: usize, control3: usize, target: usize },
    C4x     { control1: usize, control2: usize, control3: usize, control4: usize, target: usize },
    // -----------------------------------------------------------------------
    // Generic / cross-node
    // -----------------------------------------------------------------------
    Gate { name: String, #[allow(dead_code)] params: Vec<f64>, qubits: Vec<usize> },
    // -----------------------------------------------------------------------
    // Measurement and classical control
    // -----------------------------------------------------------------------
    Measure    { qubit: usize, cbit: usize },
    Reset      { qubit: usize },
    Conditional { condition: Condition, op: Box<Instruction> },
    // -----------------------------------------------------------------------
    // No-ops
    // -----------------------------------------------------------------------
    Barrier,
    Classical  { #[allow(dead_code)] name: String },
}

// ---------------------------------------------------------------------------
// Gate fusion
// ---------------------------------------------------------------------------

/// A fused instruction: either an original instruction (by reference) or a
/// pre-computed single-qubit matrix resulting from fusing consecutive gates.
pub enum FusedInstruction<'a> {
    Original(&'a Instruction),
    Fused1Q { qubit: usize, matrix: [[C; 2]; 2] },
}

#[inline]
fn matmul2x2(a: [[C; 2]; 2], b: [[C; 2]; 2]) -> [[C; 2]; 2] {
    [
        [
            a[0][0] * b[0][0] + a[0][1] * b[1][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1],
        ],
        [
            a[1][0] * b[0][0] + a[1][1] * b[1][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1],
        ],
    ]
}

/// Extract the 2×2 matrix and target qubit for a fuseable single-qubit gate.
/// Returns None if the instruction is not a plain (no controls) single-qubit gate.
fn gate_matrix_1q(inst: &Instruction) -> Option<(usize, [[C; 2]; 2])> {
    use std::f64::consts::{FRAC_1_SQRT_2, PI};
    #[inline] fn c(re: f64, im: f64) -> C { C::new(re, im) }
    #[inline] fn r(re: f64) -> C { C::new(re, 0.0) }

    let identity = [[r(1.0), r(0.0)], [r(0.0), r(1.0)]];

    match inst {
        Instruction::Id { qubit } | Instruction::U0 { qubit } =>
            Some((*qubit, identity)),

        Instruction::X { qubit } =>
            Some((*qubit, [[r(0.0), r(1.0)], [r(1.0), r(0.0)]])),
        Instruction::Y { qubit } =>
            Some((*qubit, [[r(0.0), c(0.0, -1.0)], [c(0.0, 1.0), r(0.0)]])),
        Instruction::Z { qubit } =>
            Some((*qubit, [[r(1.0), r(0.0)], [r(0.0), r(-1.0)]])),
        Instruction::H { qubit } => {
            let s = FRAC_1_SQRT_2;
            Some((*qubit, [[r(s), r(s)], [r(s), r(-s)]]))
        }
        Instruction::S { qubit } =>
            Some((*qubit, [[r(1.0), r(0.0)], [r(0.0), c(0.0, 1.0)]])),
        Instruction::Sdg { qubit } =>
            Some((*qubit, [[r(1.0), r(0.0)], [r(0.0), c(0.0, -1.0)]])),
        Instruction::T { qubit } => {
            let s = FRAC_1_SQRT_2;
            Some((*qubit, [[r(1.0), r(0.0)], [r(0.0), c(s, s)]]))
        }
        Instruction::Tdg { qubit } => {
            let s = FRAC_1_SQRT_2;
            Some((*qubit, [[r(1.0), r(0.0)], [r(0.0), c(s, -s)]]))
        }
        Instruction::Sx { qubit } =>
            Some((*qubit, [[c(0.5, 0.5), c(0.5, -0.5)], [c(0.5, -0.5), c(0.5, 0.5)]])),
        Instruction::Sxdg { qubit } =>
            Some((*qubit, [[c(0.5, -0.5), c(0.5, 0.5)], [c(0.5, 0.5), c(0.5, -0.5)]])),

        Instruction::U3 { qubit, theta, phi, lam } | Instruction::U { qubit, theta, phi, lam } => {
            let (ct, st) = ((theta / 2.0).cos(), (theta / 2.0).sin());
            Some((*qubit, [
                [r(ct), -(c(lam.cos(), lam.sin()) * r(st))],
                [c(phi.cos(), phi.sin()) * r(st), c((phi + lam).cos(), (phi + lam).sin()) * r(ct)],
            ]))
        }
        Instruction::U2 { qubit, phi, lam } => {
            let theta = PI / 2.0;
            let (ct, st) = ((theta / 2.0).cos(), (theta / 2.0).sin());
            Some((*qubit, [
                [r(ct), -(c(lam.cos(), lam.sin()) * r(st))],
                [c(phi.cos(), phi.sin()) * r(st), c((phi + lam).cos(), (phi + lam).sin()) * r(ct)],
            ]))
        }
        Instruction::U1 { qubit, lam } | Instruction::P { qubit, lam } =>
            Some((*qubit, [[r(1.0), r(0.0)], [r(0.0), c(lam.cos(), lam.sin())]])),
        Instruction::Rx { qubit, theta } => {
            let (cv, sv) = ((theta / 2.0).cos(), (theta / 2.0).sin());
            Some((*qubit, [[r(cv), c(0.0, -sv)], [c(0.0, -sv), r(cv)]]))
        }
        Instruction::Ry { qubit, theta } => {
            let (cv, sv) = ((theta / 2.0).cos(), (theta / 2.0).sin());
            Some((*qubit, [[r(cv), r(-sv)], [r(sv), r(cv)]]))
        }
        Instruction::Rz { qubit, phi } => {
            Some((*qubit, [
                [c((-phi / 2.0).cos(), (-phi / 2.0).sin()), r(0.0)],
                [r(0.0), c((phi / 2.0).cos(), (phi / 2.0).sin())],
            ]))
        }

        // All multi-qubit and non-unitary instructions cannot be fused
        _ => None,
    }
}

fn is_identity_2x2(m: &[[C; 2]; 2]) -> bool {
    (m[0][0] - C::new(1.0, 0.0)).norm() < 1e-10
        && m[0][1].norm() < 1e-10
        && m[1][0].norm() < 1e-10
        && (m[1][1] - C::new(1.0, 0.0)).norm() < 1e-10
}

#[inline]
fn identity_2x2() -> [[C; 2]; 2] {
    [
        [C::new(1.0, 0.0), C::new(0.0, 0.0)],
        [C::new(0.0, 0.0), C::new(1.0, 0.0)],
    ]
}

type Pending1Q = Vec<Option<[[C; 2]; 2]>>;

#[inline]
fn ensure_pending_qubit(pending: &mut Pending1Q, qubit: usize) {
    if qubit >= pending.len() {
        pending.resize(qubit + 1, None);
    }
}

#[inline]
fn take_pending_qubit(pending: &mut Pending1Q, qubit: usize) -> Option<[[C; 2]; 2]> {
    pending.get_mut(qubit).and_then(Option::take)
}

#[inline]
fn flush_fused_instruction<'a>(
    qubit: usize,
    pending: &mut Pending1Q,
    out: &mut Vec<FusedInstruction<'a>>,
) {
    if let Some(matrix) = take_pending_qubit(pending, qubit) {
        if !is_identity_2x2(&matrix) {
            out.push(FusedInstruction::Fused1Q { qubit, matrix });
        }
    }
}

#[inline]
fn flush_fused_pblock_entry(
    qubit: usize,
    pending: &mut Pending1Q,
    out: &mut Vec<FusedPBlockEntry>,
) {
    if let Some(matrix) = take_pending_qubit(pending, qubit) {
        if !is_identity_2x2(&matrix) {
            out.push(FusedPBlockEntry::Fused1Q { qubit, matrix });
        }
    }
}

/// Fuse consecutive single-qubit gates on the same qubit into a single matrix.
/// Returns a Vec of FusedInstruction that covers the same logical circuit.
pub fn fuse_circuit<'a>(instructions: &'a [Instruction]) -> Vec<FusedInstruction<'a>> {
    let mut pending: Pending1Q = Vec::new();
    let mut out: Vec<FusedInstruction<'a>> = Vec::with_capacity(instructions.len());

    for inst in instructions {
        if let Some((qubit, mat)) = gate_matrix_1q(inst) {
            // Accumulate: new_mat = mat * pending  (mat applied after pending)
            ensure_pending_qubit(&mut pending, qubit);
            let acc = pending[qubit].get_or_insert_with(identity_2x2);
            *acc = matmul2x2(mat, *acc);
        } else {
            // Non-fuseable instruction: flush all qubits it touches, then emit it.
            inst.for_each_qubit(|q| flush_fused_instruction(q, &mut pending, &mut out));
            // Special case: Conditional / Reset also touch their inner qubit already
            // via inst.for_each_qubit(), so no extra handling needed.
            out.push(FusedInstruction::Original(inst));
        }
    }

    // Flush any remaining pending gates.
    for q in 0..pending.len() {
        flush_fused_instruction(q, &mut pending, &mut out);
    }

    out
}

/// Owned fused entry for the pblock simulator shot loop.
/// Unlike FusedInstruction<'a>, this is 'static and Send.
pub enum FusedPBlockEntry {
    /// Reference back into node_circuits by (node, local_idx).
    Original { node: usize, local_idx: usize },
    /// Pre-computed single-qubit matrix (no circuit reference needed).
    Fused1Q { qubit: usize, matrix: [[C; 2]; 2] },
}

/// Fuse consecutive single-qubit gates in the globally-sorted pblock entry stream.
/// `entries` is `&[(order, node, local_idx)]` already sorted by order.
pub fn fuse_pblock_entries(
    entries: &[(i64, usize, usize)],
    node_circuits: &HashMap<usize, Circuit>,
) -> Vec<FusedPBlockEntry> {
    let mut pending: Pending1Q = Vec::new();
    let mut out: Vec<FusedPBlockEntry> = Vec::with_capacity(entries.len());

    for &(_, node, local_idx) in entries {
        let inst = &node_circuits[&node].instructions[local_idx];
        if let Some((qubit, mat)) = gate_matrix_1q(inst) {
            ensure_pending_qubit(&mut pending, qubit);
            let acc = pending[qubit].get_or_insert_with(identity_2x2);
            *acc = matmul2x2(mat, *acc);
        } else {
            inst.for_each_qubit(|q| flush_fused_pblock_entry(q, &mut pending, &mut out));
            out.push(FusedPBlockEntry::Original { node, local_idx });
        }
    }

    for q in 0..pending.len() {
        flush_fused_pblock_entry(q, &mut pending, &mut out);
    }
    out
}

impl Instruction {
    pub fn for_each_qubit<F>(&self, mut f: F)
    where
        F: FnMut(usize),
    {
        self.visit_qubits(&mut f);
    }

    fn visit_qubits(&self, f: &mut dyn FnMut(usize)) {
        match self {
            Instruction::X { qubit }
            | Instruction::Y { qubit }
            | Instruction::Z { qubit }
            | Instruction::H { qubit }
            | Instruction::S { qubit }
            | Instruction::Sdg { qubit }
            | Instruction::T { qubit }
            | Instruction::Tdg { qubit }
            | Instruction::Sx { qubit }
            | Instruction::Sxdg { qubit }
            | Instruction::U0 { qubit }
            | Instruction::Id { qubit }
            | Instruction::Reset { qubit } => f(*qubit),

            Instruction::U3 { qubit, .. }
            | Instruction::U2 { qubit, .. }
            | Instruction::U1 { qubit, .. }
            | Instruction::U { qubit, .. }
            | Instruction::P { qubit, .. }
            | Instruction::Rx { qubit, .. }
            | Instruction::Ry { qubit, .. }
            | Instruction::Rz { qubit, .. } => f(*qubit),

            Instruction::Measure { qubit, .. } => f(*qubit),

            Instruction::Cx { control, target }
            | Instruction::Cz { control, target }
            | Instruction::Cy { control, target }
            | Instruction::Ch { control, target }
            | Instruction::Csx { control, target }
            | Instruction::Crx { control, target, .. }
            | Instruction::Cry { control, target, .. }
            | Instruction::Crz { control, target, .. }
            | Instruction::Cu1 { control, target, .. }
            | Instruction::Cp { control, target, .. }
            | Instruction::Cu3 { control, target, .. }
            | Instruction::Cu { control, target, .. } => {
                f(*control);
                f(*target);
            }

            Instruction::Swap { a, b }
            | Instruction::Rxx { a, b, .. }
            | Instruction::Rzz { a, b, .. } => {
                f(*a);
                f(*b);
            }

            Instruction::Ccx { control1, control2, target }
            | Instruction::Rccx { control1, control2, target } => {
                f(*control1);
                f(*control2);
                f(*target);
            }

            Instruction::Cswap { control, target1, target2 } => {
                f(*control);
                f(*target1);
                f(*target2);
            }

            Instruction::Rc3x { control1, control2, control3, target }
            | Instruction::C3x { control1, control2, control3, target }
            | Instruction::C3sqrtx { control1, control2, control3, target } => {
                f(*control1);
                f(*control2);
                f(*control3);
                f(*target);
            }

            Instruction::C4x { control1, control2, control3, control4, target } => {
                f(*control1);
                f(*control2);
                f(*control3);
                f(*control4);
                f(*target);
            }

            Instruction::Gate { qubits, .. } => {
                for &qubit in qubits {
                    f(qubit);
                }
            }

            Instruction::Conditional { op, .. } => op.visit_qubits(f),

            Instruction::Barrier | Instruction::Classical { .. } => {}
        }
    }

    pub fn qubits(&self) -> Vec<usize> {
        let mut qubits = Vec::new();
        self.for_each_qubit(|qubit| qubits.push(qubit));
        qubits
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_matrix_close(actual: &[[C; 2]; 2], expected: &[[C; 2]; 2]) {
        for row in 0..2 {
            for col in 0..2 {
                assert!(
                    (actual[row][col] - expected[row][col]).norm() < 1e-10,
                    "matrix mismatch at [{row}][{col}]: {:?} != {:?}",
                    actual[row][col],
                    expected[row][col]
                );
            }
        }
    }

    #[test]
    fn fuse_circuit_elides_identity_single_qubit_run() {
        let instructions = vec![Instruction::X { qubit: 0 }, Instruction::X { qubit: 0 }];

        let fused = fuse_circuit(&instructions);

        assert_eq!(fused.len(), 0);
    }

    #[test]
    fn fuse_circuit_flushes_before_touching_gate() {
        let instructions = vec![
            Instruction::H { qubit: 0 },
            Instruction::X { qubit: 0 },
            Instruction::Cx {
                control: 0,
                target: 1,
            },
            Instruction::Z { qubit: 1 },
        ];

        let fused = fuse_circuit(&instructions);

        assert_eq!(fused.len(), 3);
        match &fused[0] {
            FusedInstruction::Fused1Q { qubit, matrix } => {
                assert_eq!(*qubit, 0);
                let (_, h) = gate_matrix_1q(&instructions[0]).unwrap();
                let (_, x) = gate_matrix_1q(&instructions[1]).unwrap();
                assert_matrix_close(matrix, &matmul2x2(x, h));
            }
            _ => panic!("expected fused 1Q instruction before CX"),
        }
        assert!(matches!(
            fused[1],
            FusedInstruction::Original(Instruction::Cx { .. })
        ));
        assert!(matches!(
            fused[2],
            FusedInstruction::Fused1Q { qubit: 1, .. }
        ));
    }
}