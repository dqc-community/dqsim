use std::collections::HashMap;
use serde::Deserialize;
use num_complex::Complex64;

type C = Complex64;

#[derive(Deserialize, Clone, Debug)]
pub struct Register {
    #[allow(dead_code)]
    pub name: String,
    pub size: usize,
    pub base: usize,
}

#[derive(Deserialize, Debug)]
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

#[derive(Deserialize, Clone, Debug)]
pub struct Condition {
    pub creg_base: usize,
    pub creg_size: usize,
    pub creg_value: u64,
}

#[derive(Deserialize, Clone, Debug)]
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

/// Fuse consecutive single-qubit gates on the same qubit into a single matrix.
/// Returns a Vec of FusedInstruction that covers the same logical circuit.
pub fn fuse_circuit<'a>(instructions: &'a [Instruction]) -> Vec<FusedInstruction<'a>> {
    // pending[qubit] = (accumulated_matrix, first_original_index_unused)
    let mut pending: HashMap<usize, [[C; 2]; 2]> = HashMap::new();
    let mut out: Vec<FusedInstruction<'a>> = Vec::with_capacity(instructions.len());

    let identity: [[C; 2]; 2] = [
        [C::new(1.0, 0.0), C::new(0.0, 0.0)],
        [C::new(0.0, 0.0), C::new(1.0, 0.0)],
    ];

    // Flush accumulated matrix for a qubit into `out`.
    let flush_qubit = |q: usize,
                       pending: &mut HashMap<usize, [[C; 2]; 2]>,
                       out: &mut Vec<FusedInstruction<'a>>| {
        if let Some(m) = pending.remove(&q) {
            if !is_identity_2x2(&m) {
                out.push(FusedInstruction::Fused1Q { qubit: q, matrix: m });
            }
        }
    };

    for inst in instructions {
        if let Some((qubit, mat)) = gate_matrix_1q(inst) {
            // Accumulate: new_mat = mat * pending  (mat applied after pending)
            let acc = pending.entry(qubit).or_insert(identity);
            *acc = matmul2x2(mat, *acc);
        } else {
            // Non-fuseable instruction: flush all qubits it touches, then emit it.
            let touched = inst.qubits();
            for q in &touched {
                flush_qubit(*q, &mut pending, &mut out);
            }
            // Special case: Conditional / Reset also touch their inner qubit already
            // via inst.qubits(), so no extra handling needed.
            out.push(FusedInstruction::Original(inst));
        }
    }

    // Flush any remaining pending gates.
    let remaining_qubits: Vec<usize> = pending.keys().copied().collect();
    for q in remaining_qubits {
        flush_qubit(q, &mut pending, &mut out);
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
    let mut pending: HashMap<usize, [[C; 2]; 2]> = HashMap::new();
    let mut out: Vec<FusedPBlockEntry> = Vec::with_capacity(entries.len());

    let identity: [[C; 2]; 2] = [
        [C::new(1.0, 0.0), C::new(0.0, 0.0)],
        [C::new(0.0, 0.0), C::new(1.0, 0.0)],
    ];

    let flush_qubit = |q: usize, pending: &mut HashMap<usize, [[C; 2]; 2]>, out: &mut Vec<FusedPBlockEntry>| {
        if let Some(m) = pending.remove(&q) {
            if !is_identity_2x2(&m) {
                out.push(FusedPBlockEntry::Fused1Q { qubit: q, matrix: m });
            }
        }
    };

    for &(_, node, local_idx) in entries {
        let inst = &node_circuits[&node].instructions[local_idx];
        if let Some((qubit, mat)) = gate_matrix_1q(inst) {
            let acc = pending.entry(qubit).or_insert(identity);
            *acc = matmul2x2(mat, *acc);
        } else {
            let touched = inst.qubits();
            for q in &touched {
                flush_qubit(*q, &mut pending, &mut out);
            }
            out.push(FusedPBlockEntry::Original { node, local_idx });
        }
    }

    let remaining: Vec<usize> = pending.keys().copied().collect();
    for q in remaining {
        flush_qubit(q, &mut pending, &mut out);
    }
    out
}

impl Instruction {
    pub fn qubits(&self) -> Vec<usize> {
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
            | Instruction::Reset { qubit } => vec![*qubit],

            Instruction::U3 { qubit, .. }
            | Instruction::U2 { qubit, .. }
            | Instruction::U1 { qubit, .. }
            | Instruction::U { qubit, .. }
            | Instruction::P { qubit, .. }
            | Instruction::Rx { qubit, .. }
            | Instruction::Ry { qubit, .. }
            | Instruction::Rz { qubit, .. } => vec![*qubit],

            Instruction::Measure { qubit, .. } => vec![*qubit],

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
            | Instruction::Cu { control, target, .. } => vec![*control, *target],

            Instruction::Swap { a, b }
            | Instruction::Rxx { a, b, .. }
            | Instruction::Rzz { a, b, .. } => vec![*a, *b],

            Instruction::Ccx { control1, control2, target }
            | Instruction::Rccx { control1, control2, target } => {
                vec![*control1, *control2, *target]
            }

            Instruction::Cswap { control, target1, target2 } => {
                vec![*control, *target1, *target2]
            }

            Instruction::Rc3x { control1, control2, control3, target }
            | Instruction::C3x { control1, control2, control3, target }
            | Instruction::C3sqrtx { control1, control2, control3, target } => {
                vec![*control1, *control2, *control3, *target]
            }

            Instruction::C4x { control1, control2, control3, control4, target } => {
                vec![*control1, *control2, *control3, *control4, *target]
            }

            Instruction::Gate { qubits, .. } => qubits.clone(),

            Instruction::Conditional { op, .. } => op.qubits(),

            Instruction::Barrier | Instruction::Classical { .. } => vec![],
        }
    }
}
