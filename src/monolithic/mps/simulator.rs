use std::collections::HashMap;
use std::sync::OnceLock;
use std::time::Instant;

use nalgebra::DMatrix;
use num_complex::Complex64;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

use crate::gates;
use crate::monolithic::statevector::SimulationResult;
use crate::profiling::{ShotLoopProfiler, ShotsProfile};
use crate::types::{format_cbits, Circuit, Instruction};

type C = Complex64;

#[derive(Clone)]
struct Tensor {
    left: usize,
    right: usize,
    data: Vec<C>,
}

impl Tensor {
    fn zero(left: usize, right: usize) -> Self {
        Self {
            left,
            right,
            data: vec![C::new(0.0, 0.0); left * 2 * right],
        }
    }

    #[inline]
    fn idx(&self, left: usize, state: usize, right: usize) -> usize {
        (left * 2 + state) * self.right + right
    }

    #[inline]
    fn get(&self, left: usize, state: usize, right: usize) -> C {
        self.data[self.idx(left, state, right)]
    }

    #[inline]
    fn set(&mut self, left: usize, state: usize, right: usize, value: C) {
        let idx = self.idx(left, state, right);
        self.data[idx] = value;
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MpsRoutingMode {
    Restore,
    Lazy,
    Lookahead,
}

impl MpsRoutingMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::Restore => "restore",
            Self::Lazy => "lazy",
            Self::Lookahead => "lookahead",
        }
    }

    fn uses_lazy_ordering(self) -> bool {
        !matches!(self, Self::Restore)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RoutingDirection {
    MoveHighLeft,
    MoveLowRight,
}

#[derive(Clone, Copy, Debug)]
enum MpsTwoQKernel {
    Diagonal([C; 4]),
    Permutation([(usize, C); 4]),
    Dense,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MpsTwoQKernelMode {
    Off,
    Auto,
    Force,
}

impl MpsTwoQKernelMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Auto => "auto",
            Self::Force => "force",
        }
    }

    fn enabled(self) -> bool {
        !matches!(self, Self::Off)
    }
}

fn classify_diagonal_2q(mat: &[[C; 4]; 4]) -> Option<[C; 4]> {
    let zero = C::new(0.0, 0.0);
    let tolerance = mps_fast_kernel_zero_tolerance();
    let mut diagonal = [zero; 4];
    for row in 0..4 {
        diagonal[row] = mat[row][row];
        for col in 0..4 {
            if row != col && !is_effectively_zero(mat[row][col], tolerance) {
                return None;
            }
        }
    }
    Some(diagonal)
}

fn reverse_basis_index(idx: usize) -> usize {
    ((idx & 1) << 1) | ((idx >> 1) & 1)
}

fn reverse_2q_kernel_order(kernel: MpsTwoQKernel) -> MpsTwoQKernel {
    match kernel {
        MpsTwoQKernel::Diagonal(diagonal) => {
            let mut reversed = [C::new(0.0, 0.0); 4];
            for row in 0..4 {
                reversed[row] = diagonal[reverse_basis_index(row)];
            }
            MpsTwoQKernel::Diagonal(reversed)
        }
        MpsTwoQKernel::Permutation(mapping) => {
            let mut reversed = [(0usize, C::new(0.0, 0.0)); 4];
            for row in 0..4 {
                let old_row = reverse_basis_index(row);
                let (old_col, scale) = mapping[old_row];
                reversed[row] = (reverse_basis_index(old_col), scale);
            }
            MpsTwoQKernel::Permutation(reversed)
        }
        MpsTwoQKernel::Dense => MpsTwoQKernel::Dense,
    }
}

fn cnot_kernel() -> MpsTwoQKernel {
    let one = C::new(1.0, 0.0);
    MpsTwoQKernel::Permutation([(0, one), (1, one), (3, one), (2, one)])
}

fn cy_kernel() -> MpsTwoQKernel {
    let one = C::new(1.0, 0.0);
    let i = C::new(0.0, 1.0);
    MpsTwoQKernel::Permutation([(0, one), (1, one), (3, -i), (2, i)])
}

fn cz_kernel() -> MpsTwoQKernel {
    MpsTwoQKernel::Diagonal([
        C::new(1.0, 0.0),
        C::new(1.0, 0.0),
        C::new(1.0, 0.0),
        C::new(-1.0, 0.0),
    ])
}

fn swap_kernel() -> MpsTwoQKernel {
    let one = C::new(1.0, 0.0);
    MpsTwoQKernel::Permutation([(0, one), (2, one), (1, one), (3, one)])
}

impl MpsTwoQKernel {
    fn classify(mat: &[[C; 4]; 4]) -> Self {
        let zero = C::new(0.0, 0.0);
        if let Some(diagonal) = classify_diagonal_2q(mat) {
            return Self::Diagonal(diagonal);
        }

        let tolerance = mps_fast_kernel_zero_tolerance();
        let mut mapping = [(0usize, zero); 4];
        let mut used_columns = [false; 4];
        for row in 0..4 {
            let mut nonzero = None;
            for col in 0..4 {
                let value = mat[row][col];
                if is_effectively_zero(value, tolerance) {
                    continue;
                }
                if nonzero.is_some() {
                    return Self::Dense;
                }
                nonzero = Some((col, value));
            }

            let Some((col, value)) = nonzero else {
                return Self::Dense;
            };
            if used_columns[col] {
                return Self::Dense;
            }
            used_columns[col] = true;
            mapping[row] = (col, value);
        }

        Self::Permutation(mapping)
    }
}

fn apply_mps_2q_kernel(kernel: MpsTwoQKernel, mat: &[[C; 4]; 4], input: [C; 4]) -> [C; 4] {
    match kernel {
        MpsTwoQKernel::Diagonal(diagonal) => [
            diagonal[0] * input[0],
            diagonal[1] * input[1],
            diagonal[2] * input[2],
            diagonal[3] * input[3],
        ],
        MpsTwoQKernel::Permutation(mapping) => {
            let mut output = [C::new(0.0, 0.0); 4];
            for out_idx in 0..4 {
                let (in_idx, scale) = mapping[out_idx];
                output[out_idx] = scale * input[in_idx];
            }
            output
        }
        MpsTwoQKernel::Dense => {
            let mut output = [C::new(0.0, 0.0); 4];
            for out_idx in 0..4 {
                output[out_idx] = mat[out_idx][0] * input[0]
                    + mat[out_idx][1] * input[1]
                    + mat[out_idx][2] * input[2]
                    + mat[out_idx][3] * input[3];
            }
            output
        }
    }
}

fn factor_product_state_2x2(output: [C; 4]) -> Option<([C; 2], [C; 2])> {
    let tolerance = mps_rank1_factorization_tolerance();
    let mut pivot = None;
    for row in 0..2 {
        for col in 0..2 {
            let value = output[row * 2 + col];
            if !is_effectively_zero(value, tolerance) {
                pivot = Some((row, col, value));
                break;
            }
        }
        if pivot.is_some() {
            break;
        }
    }

    let Some((pivot_row, pivot_col, pivot_value)) = pivot else {
        return None;
    };

    for row in 0..2 {
        for col in 0..2 {
            let lhs = output[row * 2 + col] * pivot_value;
            let rhs = output[row * 2 + pivot_col] * output[pivot_row * 2 + col];
            let scale = output[row * 2 + col]
                .norm()
                .max(output[row * 2 + pivot_col].norm())
                .max(output[pivot_row * 2 + col].norm())
                .max(pivot_value.norm())
                .max(1.0);
            if (lhs - rhs).norm() > tolerance * scale * scale {
                return None;
            }
        }
    }

    let mut left = [C::new(0.0, 0.0); 2];
    let mut right = [C::new(0.0, 0.0); 2];
    for row in 0..2 {
        left[row] = output[row * 2 + pivot_col] / pivot_value;
    }
    for col in 0..2 {
        right[col] = output[pivot_row * 2 + col];
    }
    Some((left, right))
}

#[derive(Clone, Debug)]
struct MpsProfileCounters {
    logical_2q_gates: usize,
    routing_swaps: usize,
    adjacent_2q_applications: usize,
    svd_count: usize,
    svd_time_ms: f64,
    fast_kernel_applications: usize,
    fast_kernel_auto_skipped_applications: usize,
    diagonal_kernel_applications: usize,
    permutation_kernel_applications: usize,
    dense_kernel_applications: usize,
    rank1_factorizations: usize,
    svd_skipped_count: usize,
    max_observed_bond_dimension: usize,
    total_routed_distance: usize,
    max_routed_distance: usize,
    lookahead_decisions: usize,
    lookahead_flipped_routes: usize,
}

impl Default for MpsProfileCounters {
    fn default() -> Self {
        Self {
            logical_2q_gates: 0,
            routing_swaps: 0,
            adjacent_2q_applications: 0,
            svd_count: 0,
            svd_time_ms: 0.0,
            fast_kernel_applications: 0,
            fast_kernel_auto_skipped_applications: 0,
            diagonal_kernel_applications: 0,
            permutation_kernel_applications: 0,
            dense_kernel_applications: 0,
            rank1_factorizations: 0,
            svd_skipped_count: 0,
            max_observed_bond_dimension: 1,
            total_routed_distance: 0,
            max_routed_distance: 0,
            lookahead_decisions: 0,
            lookahead_flipped_routes: 0,
        }
    }
}

impl MpsProfileCounters {
    fn merge(&mut self, other: &Self) {
        self.logical_2q_gates += other.logical_2q_gates;
        self.routing_swaps += other.routing_swaps;
        self.adjacent_2q_applications += other.adjacent_2q_applications;
        self.svd_count += other.svd_count;
        self.svd_time_ms += other.svd_time_ms;
        self.fast_kernel_applications += other.fast_kernel_applications;
        self.fast_kernel_auto_skipped_applications += other.fast_kernel_auto_skipped_applications;
        self.diagonal_kernel_applications += other.diagonal_kernel_applications;
        self.permutation_kernel_applications += other.permutation_kernel_applications;
        self.dense_kernel_applications += other.dense_kernel_applications;
        self.rank1_factorizations += other.rank1_factorizations;
        self.svd_skipped_count += other.svd_skipped_count;
        self.max_observed_bond_dimension = self
            .max_observed_bond_dimension
            .max(other.max_observed_bond_dimension);
        self.total_routed_distance += other.total_routed_distance;
        self.max_routed_distance = self.max_routed_distance.max(other.max_routed_distance);
        self.lookahead_decisions += other.lookahead_decisions;
        self.lookahead_flipped_routes += other.lookahead_flipped_routes;
    }

    fn average_routed_distance(&self) -> f64 {
        if self.logical_2q_gates == 0 {
            0.0
        } else {
            self.total_routed_distance as f64 / self.logical_2q_gates as f64
        }
    }
}

#[derive(Debug, Default)]
struct MpsRunSummary {
    counts: HashMap<String, usize>,
    counters: MpsProfileCounters,
}

impl MpsRunSummary {
    fn merge(&mut self, other: Self) {
        for (key, value) in other.counts {
            *self.counts.entry(key).or_insert(0) += value;
        }
        self.counters.merge(&other.counters);
    }
}

struct Mps {
    tensors: Vec<Tensor>,
    logical_to_position: Vec<usize>,
    position_to_logical: Vec<usize>,
    max_bond_dimension: Option<usize>,
    truncation_threshold: f64,
    routing_mode: MpsRoutingMode,
    fast_kernel_mode: MpsTwoQKernelMode,
    total_shots: usize,
    counters: MpsProfileCounters,
}

impl Mps {
    fn new(
        num_qubits: usize,
        max_bond_dimension: Option<usize>,
        truncation_threshold: f64,
        total_shots: usize,
    ) -> Self {
        let mut tensors = Vec::with_capacity(num_qubits);
        for _ in 0..num_qubits {
            let mut tensor = Tensor::zero(1, 1);
            tensor.set(0, 0, 0, C::new(1.0, 0.0));
            tensors.push(tensor);
        }
        let logical_to_position: Vec<usize> = (0..num_qubits).collect();
        let position_to_logical = logical_to_position.clone();
        Self {
            tensors,
            logical_to_position,
            position_to_logical,
            max_bond_dimension,
            truncation_threshold,
            routing_mode: mps_routing_mode(),
            fast_kernel_mode: mps_two_qubit_fast_kernel_mode(),
            total_shots: total_shots.max(1),
            counters: MpsProfileCounters::default(),
        }
    }

    fn apply_1q(&mut self, qubit: usize, mat: &[[C; 2]; 2]) {
        let position = self.logical_to_position[qubit];
        let old = self.tensors[position].clone();
        let mut new = Tensor::zero(old.left, old.right);
        for left in 0..old.left {
            for right in 0..old.right {
                for (out, row) in mat.iter().enumerate() {
                    let mut acc = C::new(0.0, 0.0);
                    for (input, gate_element) in row.iter().enumerate() {
                        acc += *gate_element * old.get(left, input, right);
                    }
                    new.set(left, out, right, acc);
                }
            }
        }
        self.tensors[position] = new;
    }

    fn apply_2q(
        &mut self,
        a: usize,
        b: usize,
        mat: &[[C; 4]; 4],
        lookahead: &[Instruction],
    ) -> PyResult<()> {
        self.apply_2q_with_kernel_hint(a, b, mat, lookahead, None)
    }

    fn apply_2q_with_kernel_hint(
        &mut self,
        a: usize,
        b: usize,
        mat: &[[C; 4]; 4],
        lookahead: &[Instruction],
        kernel_hint: Option<MpsTwoQKernel>,
    ) -> PyResult<()> {
        if a == b {
            return Ok(());
        }
        let pos_a = self.logical_to_position[a];
        let pos_b = self.logical_to_position[b];
        let routed_distance = pos_a.abs_diff(pos_b);
        self.counters.logical_2q_gates += 1;
        self.counters.total_routed_distance += routed_distance;
        self.counters.max_routed_distance = self.counters.max_routed_distance.max(routed_distance);

        match self.routing_mode {
            MpsRoutingMode::Restore => {
                self.apply_2q_restoring_order(pos_a, pos_b, mat, kernel_hint)
            }
            MpsRoutingMode::Lazy => self.apply_2q_lazy(
                pos_a,
                pos_b,
                mat,
                RoutingDirection::MoveHighLeft,
                kernel_hint,
            ),
            MpsRoutingMode::Lookahead => {
                self.apply_2q_lookahead(pos_a, pos_b, mat, lookahead, kernel_hint)
            }
        }
    }

    fn apply_2q_restoring_order(
        &mut self,
        pos_a: usize,
        pos_b: usize,
        mat: &[[C; 4]; 4],
        kernel_hint: Option<MpsTwoQKernel>,
    ) -> PyResult<()> {
        let (lo, hi, reversed) = if pos_a < pos_b {
            (pos_a, pos_b, false)
        } else {
            (pos_b, pos_a, true)
        };

        for pos in ((lo + 1)..hi).rev() {
            self.apply_physical_routing_swap(pos)?;
        }

        let gate = if reversed {
            reverse_2q_order(mat)
        } else {
            *mat
        };
        let adjacent_kernel_hint = if reversed {
            kernel_hint.map(reverse_2q_kernel_order)
        } else {
            kernel_hint
        };
        self.apply_adjacent_2q(lo, &gate, adjacent_kernel_hint)?;

        for pos in (lo + 1)..hi {
            self.apply_physical_routing_swap(pos)?;
        }
        Ok(())
    }

    fn apply_2q_lookahead(
        &mut self,
        pos_a: usize,
        pos_b: usize,
        mat: &[[C; 4]; 4],
        lookahead: &[Instruction],
        kernel_hint: Option<MpsTwoQKernel>,
    ) -> PyResult<()> {
        if pos_a.abs_diff(pos_b) <= 1 {
            return self.apply_2q_lazy(
                pos_a,
                pos_b,
                mat,
                RoutingDirection::MoveHighLeft,
                kernel_hint,
            );
        }

        self.counters.lookahead_decisions += 1;
        let high_left_score =
            self.score_routing_candidate(pos_a, pos_b, RoutingDirection::MoveHighLeft, lookahead);
        let low_right_score =
            self.score_routing_candidate(pos_a, pos_b, RoutingDirection::MoveLowRight, lookahead);
        let direction = if low_right_score < high_left_score {
            self.counters.lookahead_flipped_routes += 1;
            RoutingDirection::MoveLowRight
        } else {
            RoutingDirection::MoveHighLeft
        };

        self.apply_2q_lazy(pos_a, pos_b, mat, direction, kernel_hint)
    }

    fn apply_2q_lazy(
        &mut self,
        mut pos_a: usize,
        mut pos_b: usize,
        mat: &[[C; 4]; 4],
        direction: RoutingDirection,
        kernel_hint: Option<MpsTwoQKernel>,
    ) -> PyResult<()> {
        match direction {
            RoutingDirection::MoveHighLeft => {
                if pos_a < pos_b {
                    while pos_b > pos_a + 1 {
                        self.apply_routing_swap(pos_b - 1)?;
                        pos_b -= 1;
                    }
                    self.apply_adjacent_2q(pos_a, mat, kernel_hint)
                } else {
                    while pos_a > pos_b + 1 {
                        self.apply_routing_swap(pos_a - 1)?;
                        pos_a -= 1;
                    }
                    let gate = reverse_2q_order(mat);
                    self.apply_adjacent_2q(pos_b, &gate, kernel_hint.map(reverse_2q_kernel_order))
                }
            }
            RoutingDirection::MoveLowRight => {
                if pos_a < pos_b {
                    while pos_a + 1 < pos_b {
                        self.apply_routing_swap(pos_a)?;
                        pos_a += 1;
                    }
                    self.apply_adjacent_2q(pos_a, mat, kernel_hint)
                } else {
                    while pos_b + 1 < pos_a {
                        self.apply_routing_swap(pos_b)?;
                        pos_b += 1;
                    }
                    let gate = reverse_2q_order(mat);
                    self.apply_adjacent_2q(pos_b, &gate, kernel_hint.map(reverse_2q_kernel_order))
                }
            }
        }
    }

    fn score_routing_candidate(
        &self,
        pos_a: usize,
        pos_b: usize,
        direction: RoutingDirection,
        lookahead: &[Instruction],
    ) -> usize {
        let mut logical_to_position = self.logical_to_position.clone();
        let mut position_to_logical = self.position_to_logical.clone();
        simulate_routing_direction(
            &mut logical_to_position,
            &mut position_to_logical,
            pos_a,
            pos_b,
            direction,
        );
        score_routing_lookahead(&logical_to_position, lookahead)
    }

    fn apply_physical_routing_swap(&mut self, position: usize) -> PyResult<()> {
        self.counters.routing_swaps += 1;
        self.apply_adjacent_2q(position, &gates::swap(), Some(swap_kernel()))
    }

    fn apply_routing_swap(&mut self, position: usize) -> PyResult<()> {
        self.apply_physical_routing_swap(position)?;
        self.position_to_logical.swap(position, position + 1);
        let left_logical = self.position_to_logical[position];
        let right_logical = self.position_to_logical[position + 1];
        self.logical_to_position[left_logical] = position;
        self.logical_to_position[right_logical] = position + 1;
        Ok(())
    }

    fn apply_adjacent_2q(
        &mut self,
        q: usize,
        mat: &[[C; 4]; 4],
        kernel_hint: Option<MpsTwoQKernel>,
    ) -> PyResult<()> {
        self.counters.adjacent_2q_applications += 1;
        match self.fast_kernel_mode {
            MpsTwoQKernelMode::Off => self.apply_adjacent_2q_baseline(q, mat),
            MpsTwoQKernelMode::Force => {
                let kernel = kernel_hint.unwrap_or_else(|| MpsTwoQKernel::classify(mat));
                self.apply_adjacent_2q_fast(q, mat, kernel)
            }
            MpsTwoQKernelMode::Auto => {
                if let Some(kernel) = kernel_hint {
                    if self.should_skip_high_shot_product_pair_with_kernel(q, kernel)? {
                        self.counters.fast_kernel_auto_skipped_applications += 1;
                        return self.apply_adjacent_2q_baseline(q, mat);
                    }
                    if self.should_use_adjacent_2q_fast(q, kernel)? {
                        return self.apply_adjacent_2q_fast(q, mat, kernel);
                    }
                    self.counters.fast_kernel_auto_skipped_applications += 1;
                    return self.apply_adjacent_2q_baseline(q, mat);
                }

                if self.should_skip_high_shot_product_pair_without_classification(q, mat)? {
                    self.counters.fast_kernel_auto_skipped_applications += 1;
                    return self.apply_adjacent_2q_baseline(q, mat);
                }

                let kernel = MpsTwoQKernel::classify(mat);
                if self.should_use_adjacent_2q_fast(q, kernel)? {
                    self.apply_adjacent_2q_fast(q, mat, kernel)
                } else {
                    self.counters.fast_kernel_auto_skipped_applications += 1;
                    self.apply_adjacent_2q_baseline(q, mat)
                }
            }
        }
    }

    fn apply_adjacent_2q_baseline(&mut self, q: usize, mat: &[[C; 4]; 4]) -> PyResult<()> {
        let left_tensor = self.tensors[q].clone();
        let right_tensor = self.tensors[q + 1].clone();
        if left_tensor.right != right_tensor.left {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Invalid MPS bond dimensions",
            ));
        }

        let left_dim = left_tensor.left;
        let bond_dim = left_tensor.right;
        let right_dim = right_tensor.right;
        let mut theta = DMatrix::<C>::zeros(left_dim * 2, 2 * right_dim);

        for left in 0..left_dim {
            for right in 0..right_dim {
                for out0 in 0..2 {
                    for out1 in 0..2 {
                        let mut acc = C::new(0.0, 0.0);
                        let out_idx = out0 * 2 + out1;
                        for in0 in 0..2 {
                            for in1 in 0..2 {
                                let in_idx = in0 * 2 + in1;
                                let mut input_amp = C::new(0.0, 0.0);
                                for bond in 0..bond_dim {
                                    input_amp += left_tensor.get(left, in0, bond)
                                        * right_tensor.get(bond, in1, right);
                                }
                                acc += mat[out_idx][in_idx] * input_amp;
                            }
                        }
                        theta[(left * 2 + out0, out1 * right_dim + right)] = acc;
                    }
                }
            }
        }

        self.finish_adjacent_2q(q, left_dim, right_dim, theta, false)
    }

    fn apply_adjacent_2q_fast(
        &mut self,
        q: usize,
        mat: &[[C; 4]; 4],
        kernel: MpsTwoQKernel,
    ) -> PyResult<()> {
        self.counters.fast_kernel_applications += 1;
        match kernel {
            MpsTwoQKernel::Diagonal(_) => self.counters.diagonal_kernel_applications += 1,
            MpsTwoQKernel::Permutation(_) => self.counters.permutation_kernel_applications += 1,
            MpsTwoQKernel::Dense => self.counters.dense_kernel_applications += 1,
        }

        if let Some((left_values, right_values)) = self.product_pair_update(q, kernel, mat)? {
            self.finish_product_pair(q, left_values, right_values);
            return Ok(());
        }

        let (left_dim, right_dim, theta) = {
            let left_tensor = &self.tensors[q];
            let right_tensor = &self.tensors[q + 1];
            if left_tensor.right != right_tensor.left {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "Invalid MPS bond dimensions",
                ));
            }

            let left_dim = left_tensor.left;
            let bond_dim = left_tensor.right;
            let right_dim = right_tensor.right;
            let mut theta = DMatrix::<C>::zeros(left_dim * 2, 2 * right_dim);

            for left in 0..left_dim {
                for right in 0..right_dim {
                    let mut input = [C::new(0.0, 0.0); 4];
                    for bond in 0..bond_dim {
                        let left0 = left_tensor.get(left, 0, bond);
                        let left1 = left_tensor.get(left, 1, bond);
                        let right0 = right_tensor.get(bond, 0, right);
                        let right1 = right_tensor.get(bond, 1, right);
                        input[0] += left0 * right0;
                        input[1] += left0 * right1;
                        input[2] += left1 * right0;
                        input[3] += left1 * right1;
                    }

                    let output = apply_mps_2q_kernel(kernel, mat, input);

                    for (out_idx, value) in output.iter().enumerate() {
                        let out0 = out_idx / 2;
                        let out1 = out_idx % 2;
                        theta[(left * 2 + out0, out1 * right_dim + right)] = *value;
                    }
                }
            }

            (left_dim, right_dim, theta)
        };

        self.finish_adjacent_2q(q, left_dim, right_dim, theta, true)
    }

    fn is_product_pair(&self, q: usize) -> PyResult<bool> {
        let left_tensor = &self.tensors[q];
        let right_tensor = &self.tensors[q + 1];
        if left_tensor.right != right_tensor.left {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Invalid MPS bond dimensions",
            ));
        }
        Ok(left_tensor.left == 1 && left_tensor.right == 1 && right_tensor.right == 1)
    }

    fn should_skip_high_shot_product_pair_with_kernel(
        &self,
        q: usize,
        kernel: MpsTwoQKernel,
    ) -> PyResult<bool> {
        if self.total_shots <= mps_product_permutation_fast_max_shots() {
            return Ok(false);
        }
        if matches!(kernel, MpsTwoQKernel::Diagonal(_)) {
            return Ok(false);
        }
        self.is_product_pair(q)
    }

    fn should_skip_high_shot_product_pair_without_classification(
        &self,
        q: usize,
        mat: &[[C; 4]; 4],
    ) -> PyResult<bool> {
        if self.total_shots <= mps_product_permutation_fast_max_shots() {
            return Ok(false);
        }
        if classify_diagonal_2q(mat).is_some() {
            return Ok(false);
        }

        self.is_product_pair(q)
    }

    fn should_use_adjacent_2q_fast(&self, q: usize, kernel: MpsTwoQKernel) -> PyResult<bool> {
        let left_tensor = &self.tensors[q];
        let right_tensor = &self.tensors[q + 1];
        if left_tensor.right != right_tensor.left {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Invalid MPS bond dimensions",
            ));
        }

        let left_dim = left_tensor.left;
        let bond_dim = left_tensor.right;
        let right_dim = right_tensor.right;
        let is_product_pair = left_dim == 1 && bond_dim == 1 && right_dim == 1;
        let work_items = left_dim.saturating_mul(bond_dim).saturating_mul(right_dim);

        Ok(match kernel {
            MpsTwoQKernel::Diagonal(_) => true,
            MpsTwoQKernel::Dense => {
                !is_product_pair || work_items >= mps_dense_fast_min_work_items()
            }
            MpsTwoQKernel::Permutation(_) => {
                !is_product_pair || self.total_shots <= mps_product_permutation_fast_max_shots()
            }
        })
    }

    fn product_pair_update(
        &self,
        q: usize,
        kernel: MpsTwoQKernel,
        mat: &[[C; 4]; 4],
    ) -> PyResult<Option<([C; 2], [C; 2])>> {
        let left_tensor = &self.tensors[q];
        let right_tensor = &self.tensors[q + 1];
        if left_tensor.right != right_tensor.left {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Invalid MPS bond dimensions",
            ));
        }
        if left_tensor.left != 1 || left_tensor.right != 1 || right_tensor.right != 1 {
            return Ok(None);
        }

        let left0 = left_tensor.get(0, 0, 0);
        let left1 = left_tensor.get(0, 1, 0);
        let right0 = right_tensor.get(0, 0, 0);
        let right1 = right_tensor.get(0, 1, 0);
        let input = [
            left0 * right0,
            left0 * right1,
            left1 * right0,
            left1 * right1,
        ];
        Ok(factor_product_state_2x2(apply_mps_2q_kernel(
            kernel, mat, input,
        )))
    }

    fn finish_product_pair(&mut self, q: usize, left_values: [C; 2], right_values: [C; 2]) {
        let mut new_left = Tensor::zero(1, 1);
        let mut new_right = Tensor::zero(1, 1);
        for state in 0..2 {
            new_left.set(0, state, 0, left_values[state]);
            new_right.set(0, state, 0, right_values[state]);
        }
        self.counters.rank1_factorizations += 1;
        self.counters.svd_skipped_count += 1;
        self.counters.max_observed_bond_dimension =
            self.counters.max_observed_bond_dimension.max(1);
        self.tensors[q] = new_left;
        self.tensors[q + 1] = new_right;
    }

    fn finish_adjacent_2q(
        &mut self,
        q: usize,
        left_dim: usize,
        right_dim: usize,
        theta: DMatrix<C>,
        allow_rank1_factorization: bool,
    ) -> PyResult<()> {
        if allow_rank1_factorization
            && left_dim == 1
            && right_dim == 1
            && self.try_rank1_factorization(q, left_dim, right_dim, &theta)
        {
            return Ok(());
        }

        self.counters.svd_count += 1;
        let svd_t0 = Instant::now();
        let svd = theta.svd(true, true);
        self.counters.svd_time_ms += svd_t0.elapsed().as_secs_f64() * 1000.0;
        let u = svd
            .u
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("MPS SVD did not return U"))?;
        let vt = svd.v_t.ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("MPS SVD did not return Vt")
        })?;

        let mut keep = svd
            .singular_values
            .iter()
            .filter(|&&s| s > self.truncation_threshold)
            .count()
            .max(1);
        if let Some(max_bond) = self.max_bond_dimension {
            keep = keep.min(max_bond.max(1));
        }
        self.counters.max_observed_bond_dimension =
            self.counters.max_observed_bond_dimension.max(keep);
        let kept_norm = svd
            .singular_values
            .iter()
            .take(keep)
            .map(|s| s * s)
            .sum::<f64>()
            .sqrt()
            .max(1e-15);
        let total_norm = svd
            .singular_values
            .iter()
            .map(|s| s * s)
            .sum::<f64>()
            .sqrt();
        let discarded_weight = svd
            .singular_values
            .iter()
            .skip(keep)
            .map(|s| s * s)
            .sum::<f64>();
        let truncation_scale = if discarded_weight > 0.0 {
            total_norm / kept_norm
        } else {
            1.0
        };

        let mut new_left = Tensor::zero(left_dim, keep);
        let mut new_right = Tensor::zero(keep, right_dim);
        for left in 0..left_dim {
            for state in 0..2 {
                let row = left * 2 + state;
                for bond in 0..keep {
                    new_left.set(left, state, bond, u[(row, bond)]);
                }
            }
        }
        for bond in 0..keep {
            let sigma = C::new(svd.singular_values[bond] * truncation_scale, 0.0);
            for state in 0..2 {
                for right in 0..right_dim {
                    let col = state * right_dim + right;
                    new_right.set(bond, state, right, sigma * vt[(bond, col)]);
                }
            }
        }

        self.tensors[q] = new_left;
        self.tensors[q + 1] = new_right;
        Ok(())
    }

    fn try_rank1_factorization(
        &mut self,
        q: usize,
        left_dim: usize,
        right_dim: usize,
        theta: &DMatrix<C>,
    ) -> bool {
        let rows = left_dim * 2;
        let cols = right_dim * 2;
        let tolerance = mps_rank1_factorization_tolerance();
        let mut pivot = None;

        for row in 0..rows {
            for col in 0..cols {
                let value = theta[(row, col)];
                if !is_effectively_zero(value, tolerance) {
                    pivot = Some((row, col, value));
                    break;
                }
            }
            if pivot.is_some() {
                break;
            }
        }

        let Some((pivot_row, pivot_col, pivot_value)) = pivot else {
            return false;
        };

        for row in 0..rows {
            for col in 0..cols {
                let lhs = theta[(row, col)] * pivot_value;
                let rhs = theta[(row, pivot_col)] * theta[(pivot_row, col)];
                let scale = theta[(row, col)]
                    .norm()
                    .max(theta[(row, pivot_col)].norm())
                    .max(theta[(pivot_row, col)].norm())
                    .max(pivot_value.norm())
                    .max(1.0);
                if (lhs - rhs).norm() > tolerance * scale * scale {
                    return false;
                }
            }
        }

        let mut new_left = Tensor::zero(left_dim, 1);
        let mut new_right = Tensor::zero(1, right_dim);
        for left in 0..left_dim {
            for state in 0..2 {
                let row = left * 2 + state;
                new_left.set(left, state, 0, theta[(row, pivot_col)] / pivot_value);
            }
        }
        for state in 0..2 {
            for right in 0..right_dim {
                let col = state * right_dim + right;
                new_right.set(0, state, right, theta[(pivot_row, col)]);
            }
        }

        self.counters.rank1_factorizations += 1;
        self.counters.svd_skipped_count += 1;
        self.counters.max_observed_bond_dimension =
            self.counters.max_observed_bond_dimension.max(1);
        self.tensors[q] = new_left;
        self.tensors[q + 1] = new_right;
        true
    }

    fn to_statevector(&self) -> Vec<C> {
        let num_qubits = self.tensors.len();
        let mut state = vec![C::new(0.0, 0.0); 1 << num_qubits];
        for (basis, amp) in state.iter_mut().enumerate() {
            let mut work = vec![C::new(1.0, 0.0)];
            for (position, tensor) in self.tensors.iter().enumerate() {
                let logical_qubit = self.position_to_logical[position];
                let bit = (basis >> logical_qubit) & 1;
                let mut next = vec![C::new(0.0, 0.0); tensor.right];
                for (left, left_amp) in work.iter().enumerate().take(tensor.left) {
                    for (right, next_amp) in next.iter_mut().enumerate().take(tensor.right) {
                        *next_amp += *left_amp * tensor.get(left, bit, right);
                    }
                }
                work = next;
            }
            *amp = work[0];
        }
        state
    }

    fn qubit_outcome_probability(&self, qubit: usize, outcome: usize) -> f64 {
        let mut env_dim = 1usize;
        let mut env = vec![C::new(1.0, 0.0)];

        for (idx, tensor) in self.tensors.iter().enumerate() {
            debug_assert_eq!(env_dim, tensor.left);
            let mut next = vec![C::new(0.0, 0.0); tensor.right * tensor.right];

            if idx == qubit {
                contract_tensor_physical_state(tensor, &env, env_dim, &mut next, outcome);
            } else {
                contract_tensor_physical_state(tensor, &env, env_dim, &mut next, 0);
                contract_tensor_physical_state(tensor, &env, env_dim, &mut next, 1);
            }

            env = next;
            env_dim = tensor.right;
        }

        env.first().map(|value| value.re.max(0.0)).unwrap_or(0.0)
    }

    fn measure(&mut self, qubit: usize, rng: &mut impl Rng) -> usize {
        let position = self.logical_to_position[qubit];
        let p0_raw = self.qubit_outcome_probability(position, 0);
        let p1_raw = self.qubit_outcome_probability(position, 1);
        let total = p0_raw + p1_raw;
        let p1 = if total > 0.0 {
            (p1_raw / total).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let outcome = if rng.gen::<f64>() < p1 { 1 } else { 0 };
        let raw_prob = if outcome == 1 { p1_raw } else { p0_raw };
        self.project_qubit(position, outcome, raw_prob);
        outcome
    }

    fn project_qubit(&mut self, qubit: usize, outcome: usize, prob: f64) {
        let scale = if prob > 0.0 { 1.0 / prob.sqrt() } else { 0.0 };
        let tensor = &mut self.tensors[qubit];
        for left in 0..tensor.left {
            for state in 0..2 {
                for right in 0..tensor.right {
                    let value = if state == outcome {
                        tensor.get(left, state, right) * C::new(scale, 0.0)
                    } else {
                        C::new(0.0, 0.0)
                    };
                    tensor.set(left, state, right, value);
                }
            }
        }
    }
}

fn contract_tensor_physical_state(
    tensor: &Tensor,
    env: &[C],
    env_dim: usize,
    next: &mut [C],
    state: usize,
) {
    let zero = C::new(0.0, 0.0);
    for left in 0..tensor.left {
        for left_prime in 0..tensor.left {
            let env_value = env[left * env_dim + left_prime];
            if env_value == zero {
                continue;
            }
            for right in 0..tensor.right {
                let amp = tensor.get(left, state, right);
                if amp == zero {
                    continue;
                }
                let scaled = env_value * amp;
                for right_prime in 0..tensor.right {
                    let bra_amp = tensor.get(left_prime, state, right_prime).conj();
                    next[right * tensor.right + right_prime] += scaled * bra_amp;
                }
            }
        }
    }
}

#[pyclass]
pub struct MpsSimulator {
    seed: Option<u64>,
    max_bond_dimension: Option<usize>,
    truncation_threshold: f64,
}

#[pymethods]
impl MpsSimulator {
    #[new]
    #[pyo3(signature = (seed=None, max_bond_dimension=None, truncation_threshold=1e-12))]
    pub fn new(
        seed: Option<u64>,
        max_bond_dimension: Option<usize>,
        truncation_threshold: f64,
    ) -> Self {
        Self {
            seed,
            max_bond_dimension,
            truncation_threshold,
        }
    }

    pub fn simulate(&self, _py: Python, circuit: &Bound<PyAny>) -> PyResult<SimulationResult> {
        let json_str: String = circuit.call_method0("model_dump_json")?.extract()?;
        let rust_circuit: Circuit = serde_json::from_str(&json_str).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Circuit JSON parse error: {e}"))
        })?;

        let mut mps = Mps::new(
            rust_circuit.num_qubits(),
            self.max_bond_dimension,
            self.truncation_threshold,
            1,
        );
        let mut cbits: HashMap<usize, i32> = HashMap::new();
        let seed = self.seed.unwrap_or_else(|| rand::thread_rng().gen());
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        for (idx, inst) in rust_circuit.instructions.iter().enumerate() {
            let lookahead = &rust_circuit.instructions[idx + 1..];
            run_instruction(&mut mps, inst, lookahead, &mut cbits, &mut rng)?;
        }

        Ok(SimulationResult::new(
            mps.to_statevector(),
            rust_circuit.num_qubits(),
            cbits,
            None,
        ))
    }

    #[pyo3(signature = (circuit, shots=1000, collect_profile=false))]
    pub fn simulate_shots(
        &self,
        py: Python,
        circuit: &Bound<PyAny>,
        shots: usize,
        collect_profile: bool,
    ) -> PyResult<PyObject> {
        let total_t0 = Instant::now();
        let preprocessing_t0 = Instant::now();
        let json_str: String = circuit.call_method0("model_dump_json")?.extract()?;
        let rust_circuit: Circuit = serde_json::from_str(&json_str).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Circuit JSON parse error: {e}"))
        })?;

        let num_qubits = rust_circuit.num_qubits();
        let num_instructions = rust_circuit.instructions.len();
        let num_cbits = rust_circuit.num_cbits();
        let base_seed = self.seed.unwrap_or_else(|| rand::thread_rng().gen());
        let preprocessing_ms = preprocessing_t0.elapsed().as_secs_f64() * 1000.0;

        let shot_loop_profiler =
            ShotLoopProfiler::start("mps", num_qubits, shots, num_instructions);
        let shot_branching_enabled = mps_shot_branching_enabled();
        let shot_branching_plan = if shot_branching_enabled {
            mps_terminal_measurement_plan(&rust_circuit.instructions)
        } else {
            None
        };
        let shot_branching_used = shot_branching_plan.is_some();
        let shot_branching_strategy =
            shot_branching_used.then_some("terminal_measurement_full_state_batch".to_string());
        let routing_mode = mps_routing_mode();
        let lazy_qubit_ordering_enabled = mps_lazy_qubit_ordering_enabled();
        let two_qubit_fast_kernel_mode = mps_two_qubit_fast_kernel_mode();
        let two_qubit_fast_kernels_enabled = two_qubit_fast_kernel_mode.enabled();
        let max_bond_dimension = self.max_bond_dimension;
        let truncation_threshold = self.truncation_threshold;

        let exec_t0 = Instant::now();
        let summary = py
            .allow_threads(|| -> Result<MpsRunSummary, String> {
                if let Some(plan) = &shot_branching_plan {
                    run_terminal_measurement_branching(
                        &rust_circuit,
                        plan,
                        num_qubits,
                        num_cbits,
                        shots,
                        base_seed,
                        max_bond_dimension,
                        truncation_threshold,
                    )
                } else {
                    run_independent_shots(
                        &rust_circuit,
                        num_qubits,
                        num_cbits,
                        shots,
                        base_seed,
                        max_bond_dimension,
                        truncation_threshold,
                    )
                }
            })
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
        let parallel_execution_ms = exec_t0.elapsed().as_secs_f64() * 1000.0;
        if let Some(profiler) = shot_loop_profiler {
            profiler.finish();
        }

        let d = PyDict::new_bound(py);
        for (key, value) in &summary.counts {
            d.set_item(key, value)?;
        }

        if collect_profile {
            let profile = ShotsProfile {
                num_shots: shots,
                num_qubits,
                num_instructions,
                preprocessing_ms,
                gate_fusion_ms: 0.0,
                parallel_execution_ms,
                per_shot_stats: None,
                statevector_simd_enabled: None,
                statevector_simd_backend: None,
                statevector_simd_used: None,
                statevector_simd_min_qubits: None,
                statevector_two_qubit_kernels_enabled: None,
                statevector_two_qubit_kernels_used: None,
                statevector_two_qubit_kernel_gates: None,
                statevector_shot_branching_enabled: None,
                statevector_shot_branching_used: None,
                statevector_shot_branching_strategy: None,
                mps_shot_branching_enabled: Some(shot_branching_enabled),
                mps_shot_branching_used: Some(shot_branching_used),
                mps_shot_branching_strategy: shot_branching_strategy,
                mps_lazy_qubit_ordering_enabled: Some(lazy_qubit_ordering_enabled),
                mps_routing_mode: Some(routing_mode.as_str().to_string()),
                mps_logical_2q_gates: Some(summary.counters.logical_2q_gates),
                mps_routing_swaps: Some(summary.counters.routing_swaps),
                mps_adjacent_2q_applications: Some(summary.counters.adjacent_2q_applications),
                mps_svd_count: Some(summary.counters.svd_count),
                mps_svd_time_ms: Some(summary.counters.svd_time_ms),
                mps_2q_fast_kernels_enabled: Some(two_qubit_fast_kernels_enabled),
                mps_2q_fast_kernel_mode: Some(two_qubit_fast_kernel_mode.as_str().to_string()),
                mps_2q_fast_kernels_used: Some(summary.counters.fast_kernel_applications > 0),
                mps_2q_fast_kernel_applications: Some(summary.counters.fast_kernel_applications),
                mps_2q_fast_kernel_auto_skipped_applications: Some(
                    summary.counters.fast_kernel_auto_skipped_applications,
                ),
                mps_2q_diagonal_kernel_applications: Some(
                    summary.counters.diagonal_kernel_applications,
                ),
                mps_2q_permutation_kernel_applications: Some(
                    summary.counters.permutation_kernel_applications,
                ),
                mps_2q_dense_kernel_applications: Some(summary.counters.dense_kernel_applications),
                mps_rank1_factorizations: Some(summary.counters.rank1_factorizations),
                mps_svd_skipped_count: Some(summary.counters.svd_skipped_count),
                mps_max_observed_bond_dimension: Some(summary.counters.max_observed_bond_dimension),
                mps_average_routed_distance: Some(summary.counters.average_routed_distance()),
                mps_max_routed_distance: Some(summary.counters.max_routed_distance),
                mps_lookahead_decisions: Some(summary.counters.lookahead_decisions),
                mps_lookahead_flipped_routes: Some(summary.counters.lookahead_flipped_routes),
                statevector_qubit_truncation_enabled: None,
                statevector_qubit_truncation_used: None,
                statevector_qubit_truncation_strategy: None,
                statevector_original_num_qubits: None,
                statevector_effective_num_qubits: None,
                statevector_removed_qubits: None,
                total_time_ms: total_t0.elapsed().as_secs_f64() * 1000.0,
            };

            let profile_json_str = profile.to_json_string().map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Profile serialization error: {e}"
                ))
            })?;
            let json_module = py.import_bound("json")?;
            let profile_dict = json_module.call_method1("loads", (&profile_json_str,))?;

            let result_dict = PyDict::new_bound(py);
            result_dict.set_item("counts", &d)?;
            result_dict.set_item("profile", &profile_dict)?;
            return Ok(result_dict.into());
        }

        Ok(d.into())
    }
}

#[derive(Clone, Copy)]
struct TerminalMeasurement {
    qubit: usize,
    cbit: usize,
}

struct TerminalMeasurementPlan {
    measurements: Vec<TerminalMeasurement>,
}

fn mps_two_qubit_fast_kernel_mode() -> MpsTwoQKernelMode {
    static MODE: OnceLock<MpsTwoQKernelMode> = OnceLock::new();
    *MODE.get_or_init(|| {
        for key in [
            "DQSIM_MPS_2Q_FAST_KERNELS",
            "DQSIM_MPS_TWO_QUBIT_FAST_KERNELS",
        ] {
            let Ok(value) = std::env::var(key) else {
                continue;
            };
            return match value.trim().to_ascii_lowercase().as_str() {
                "0" | "false" | "no" | "off" | "baseline" => MpsTwoQKernelMode::Off,
                "1" | "true" | "yes" | "on" | "force" | "forced" => MpsTwoQKernelMode::Force,
                "auto" | "selective" | "heuristic" => MpsTwoQKernelMode::Auto,
                _ => MpsTwoQKernelMode::Auto,
            };
        }
        MpsTwoQKernelMode::Auto
    })
}

fn mps_product_permutation_fast_max_shots() -> usize {
    static MAX_SHOTS: OnceLock<usize> = OnceLock::new();
    *MAX_SHOTS.get_or_init(|| {
        std::env::var("DQSIM_MPS_PRODUCT_PERM_FAST_MAX_SHOTS")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(128)
    })
}

fn mps_dense_fast_min_work_items() -> usize {
    static MIN_WORK: OnceLock<usize> = OnceLock::new();
    *MIN_WORK.get_or_init(|| {
        std::env::var("DQSIM_MPS_DENSE_FAST_MIN_WORK_ITEMS")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(4)
    })
}

fn mps_fast_kernel_zero_tolerance() -> f64 {
    1e-14
}

fn mps_rank1_factorization_tolerance() -> f64 {
    1e-12
}

#[inline]
fn is_effectively_zero(value: C, tolerance: f64) -> bool {
    value.norm_sqr() <= tolerance * tolerance
}

fn mps_shot_branching_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        let Ok(value) = std::env::var("DQSIM_MPS_SHOT_BRANCHING") else {
            return false;
        };
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on" | "terminal" | "terminal_measurement"
        )
    })
}

fn mps_routing_mode() -> MpsRoutingMode {
    static MODE: OnceLock<MpsRoutingMode> = OnceLock::new();
    *MODE.get_or_init(|| {
        if let Ok(value) = std::env::var("DQSIM_MPS_ROUTING") {
            match value.trim().to_ascii_lowercase().as_str() {
                "restore" | "restoring" | "0" | "false" | "off" => return MpsRoutingMode::Restore,
                "lazy" | "1" | "true" | "yes" | "on" => return MpsRoutingMode::Lazy,
                "lookahead" | "look-ahead" | "ahead" => return MpsRoutingMode::Lookahead,
                _ => {}
            }
        }

        if mps_legacy_lazy_qubit_ordering_enabled() {
            MpsRoutingMode::Lazy
        } else {
            MpsRoutingMode::Restore
        }
    })
}

fn mps_lazy_qubit_ordering_enabled() -> bool {
    mps_routing_mode().uses_lazy_ordering()
}

fn mps_legacy_lazy_qubit_ordering_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        let Ok(value) = std::env::var("DQSIM_MPS_LAZY_QUBIT_ORDERING") else {
            return false;
        };
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on" | "lazy"
        )
    })
}

fn mps_lookahead_depth() -> usize {
    static DEPTH: OnceLock<usize> = OnceLock::new();
    *DEPTH.get_or_init(|| {
        std::env::var("DQSIM_MPS_LOOKAHEAD_GATES")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(8)
    })
}

fn simulate_routing_direction(
    logical_to_position: &mut [usize],
    position_to_logical: &mut [usize],
    mut pos_a: usize,
    mut pos_b: usize,
    direction: RoutingDirection,
) {
    match direction {
        RoutingDirection::MoveHighLeft => {
            if pos_a < pos_b {
                while pos_b > pos_a + 1 {
                    simulate_logical_routing_swap(
                        logical_to_position,
                        position_to_logical,
                        pos_b - 1,
                    );
                    pos_b -= 1;
                }
            } else {
                while pos_a > pos_b + 1 {
                    simulate_logical_routing_swap(
                        logical_to_position,
                        position_to_logical,
                        pos_a - 1,
                    );
                    pos_a -= 1;
                }
            }
        }
        RoutingDirection::MoveLowRight => {
            if pos_a < pos_b {
                while pos_a + 1 < pos_b {
                    simulate_logical_routing_swap(logical_to_position, position_to_logical, pos_a);
                    pos_a += 1;
                }
            } else {
                while pos_b + 1 < pos_a {
                    simulate_logical_routing_swap(logical_to_position, position_to_logical, pos_b);
                    pos_b += 1;
                }
            }
        }
    }
}

fn simulate_logical_routing_swap(
    logical_to_position: &mut [usize],
    position_to_logical: &mut [usize],
    position: usize,
) {
    position_to_logical.swap(position, position + 1);
    let left_logical = position_to_logical[position];
    let right_logical = position_to_logical[position + 1];
    logical_to_position[left_logical] = position;
    logical_to_position[right_logical] = position + 1;
}

fn score_routing_lookahead(logical_to_position: &[usize], lookahead: &[Instruction]) -> usize {
    let depth = mps_lookahead_depth();
    if depth == 0 {
        return 0;
    }

    let mut score = 0usize;
    let mut seen = 0usize;
    for inst in lookahead {
        let Some((left, right)) = mps_instruction_qubit_pair(inst) else {
            continue;
        };
        if left >= logical_to_position.len() || right >= logical_to_position.len() {
            continue;
        }
        let distance = logical_to_position[left].abs_diff(logical_to_position[right]);
        score = score.saturating_add(distance.saturating_sub(1));
        seen += 1;
        if seen >= depth {
            break;
        }
    }
    score
}

fn mps_instruction_qubit_pair(inst: &Instruction) -> Option<(usize, usize)> {
    match inst {
        Instruction::Cx { control, target }
        | Instruction::Cz { control, target }
        | Instruction::Cy { control, target }
        | Instruction::Ch { control, target }
        | Instruction::Csx { control, target }
        | Instruction::Crx {
            control, target, ..
        }
        | Instruction::Cry {
            control, target, ..
        }
        | Instruction::Crz {
            control, target, ..
        }
        | Instruction::Cu1 {
            control, target, ..
        }
        | Instruction::Cp {
            control, target, ..
        }
        | Instruction::Cu3 {
            control, target, ..
        }
        | Instruction::Cu {
            control, target, ..
        } => Some((*control, *target)),
        Instruction::Swap { a, b }
        | Instruction::Rxx { a, b, .. }
        | Instruction::Rzz { a, b, .. } => Some((*a, *b)),
        Instruction::Gate { name, qubits, .. } => match name.to_lowercase().as_str() {
            "remote_link_phi_plus"
            | "remote_epr"
            | "epr"
            | "remote_link_psi_minus"
            | "remote_link_psi_plus"
            | "nonlocal_cz"
            | "remote_cz"
            | "remote_cx"
                if qubits.len() >= 2 =>
            {
                Some((qubits[0], qubits[1]))
            }
            _ => None,
        },
        Instruction::Conditional { op, .. } => mps_instruction_qubit_pair(op),
        _ => None,
    }
}

fn mps_terminal_measurement_plan(instructions: &[Instruction]) -> Option<TerminalMeasurementPlan> {
    let mut measurements = Vec::new();
    let mut seen_cbits = Vec::new();
    let mut saw_terminal_measurement = false;

    for inst in instructions {
        match inst {
            Instruction::Measure { qubit, cbit } => {
                saw_terminal_measurement = true;
                if seen_cbits.contains(cbit) {
                    return None;
                }
                seen_cbits.push(*cbit);
                measurements.push(TerminalMeasurement {
                    qubit: *qubit,
                    cbit: *cbit,
                });
            }
            Instruction::Barrier | Instruction::Classical { .. } => {}
            Instruction::Reset { .. } | Instruction::Conditional { .. } => return None,
            _ if saw_terminal_measurement => return None,
            _ => {}
        }
    }

    if measurements.is_empty() {
        return None;
    }

    Some(TerminalMeasurementPlan { measurements })
}

fn run_independent_shots(
    circuit: &Circuit,
    num_qubits: usize,
    num_cbits: usize,
    shots: usize,
    base_seed: u64,
    max_bond_dimension: Option<usize>,
    truncation_threshold: f64,
) -> Result<MpsRunSummary, String> {
    (0..shots)
        .into_par_iter()
        .map(|shot| -> Result<MpsRunSummary, String> {
            let mut mps = Mps::new(num_qubits, max_bond_dimension, truncation_threshold, shots);
            let mut cbits: HashMap<usize, i32> = HashMap::new();
            let mut rng = ChaCha8Rng::seed_from_u64(base_seed.wrapping_add(shot as u64));
            for (idx, inst) in circuit.instructions.iter().enumerate() {
                let lookahead = &circuit.instructions[idx + 1..];
                run_instruction(&mut mps, inst, lookahead, &mut cbits, &mut rng)
                    .map_err(|err| err.to_string())?;
            }

            let mut summary = MpsRunSummary::default();
            *summary
                .counts
                .entry(format_cbits(&cbits, num_cbits))
                .or_insert(0) += 1;
            summary.counters.merge(&mps.counters);
            Ok(summary)
        })
        .try_reduce(MpsRunSummary::default, |mut left, right| {
            left.merge(right);
            Ok(left)
        })
}

fn run_terminal_measurement_branching(
    circuit: &Circuit,
    plan: &TerminalMeasurementPlan,
    num_qubits: usize,
    num_cbits: usize,
    shots: usize,
    base_seed: u64,
    max_bond_dimension: Option<usize>,
    truncation_threshold: f64,
) -> Result<MpsRunSummary, String> {
    let mut mps = Mps::new(num_qubits, max_bond_dimension, truncation_threshold, shots);
    let mut cbits: HashMap<usize, i32> = HashMap::new();
    let mut rng = ChaCha8Rng::seed_from_u64(base_seed);

    for (idx, inst) in circuit.instructions.iter().enumerate() {
        if matches!(inst, Instruction::Measure { .. }) {
            continue;
        }
        let lookahead = &circuit.instructions[idx + 1..];
        run_instruction(&mut mps, inst, lookahead, &mut cbits, &mut rng)
            .map_err(|err| err.to_string())?;
    }

    let state = mps.to_statevector();
    let mut summary = MpsRunSummary {
        counts: sample_terminal_sequential_full_state(&state, num_cbits, plan, shots, base_seed),
        counters: MpsProfileCounters::default(),
    };
    summary.counters.merge(&mps.counters);
    Ok(summary)
}

fn sample_terminal_sequential_full_state(
    state: &[C],
    num_cbits: usize,
    plan: &TerminalMeasurementPlan,
    shots: usize,
    base_seed: u64,
) -> HashMap<String, usize> {
    let probabilities: Vec<f64> = state.iter().map(|amp| amp.norm_sqr()).collect();
    let mut counts = HashMap::new();

    for shot in 0..shots {
        let mut rng = ChaCha8Rng::seed_from_u64(base_seed.wrapping_add(shot as u64));
        let mut fixed_mask = 0usize;
        let mut fixed_value = 0usize;
        let mut bits = vec![b'0'; num_cbits];

        for measurement in &plan.measurements {
            let qubit_mask = 1usize << measurement.qubit;
            let mut p0 = 0.0;
            let mut p1 = 0.0;

            for (basis, probability) in probabilities.iter().enumerate() {
                if (basis & fixed_mask) != fixed_value {
                    continue;
                }
                if (basis & qubit_mask) == 0 {
                    p0 += *probability;
                } else {
                    p1 += *probability;
                }
            }

            let total = p0 + p1;
            let p1 = if total > 0.0 {
                (p1 / total).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let outcome = usize::from(rng.gen::<f64>() < p1);

            fixed_mask |= qubit_mask;
            if outcome == 1 {
                fixed_value |= qubit_mask;
            } else {
                fixed_value &= !qubit_mask;
            }

            if measurement.cbit < num_cbits {
                bits[num_cbits - 1 - measurement.cbit] = if outcome == 1 { b'1' } else { b'0' };
            }
        }

        let key = String::from_utf8(bits).expect("terminal measurement keys are ASCII bits");
        *counts.entry(key).or_insert(0) += 1;
    }

    counts
}
fn run_instruction(
    mps: &mut Mps,
    inst: &Instruction,
    lookahead: &[Instruction],
    cbits: &mut HashMap<usize, i32>,
    rng: &mut impl Rng,
) -> PyResult<()> {
    match inst {
        Instruction::Id { .. }
        | Instruction::U0 { .. }
        | Instruction::Barrier
        | Instruction::Classical { .. } => {}
        Instruction::X { qubit } => mps.apply_1q(*qubit, &gates::X),
        Instruction::Y { qubit } => mps.apply_1q(*qubit, &gates::Y),
        Instruction::Z { qubit } => mps.apply_1q(*qubit, &gates::Z),
        Instruction::H { qubit } => mps.apply_1q(*qubit, &gates::h()),
        Instruction::S { qubit } => mps.apply_1q(*qubit, &gates::s_gate()),
        Instruction::Sdg { qubit } => mps.apply_1q(*qubit, &gates::sdg()),
        Instruction::T { qubit } => mps.apply_1q(*qubit, &gates::t_gate()),
        Instruction::Tdg { qubit } => mps.apply_1q(*qubit, &gates::tdg()),
        Instruction::Sx { qubit } => mps.apply_1q(*qubit, &gates::sx()),
        Instruction::Sxdg { qubit } => mps.apply_1q(*qubit, &gates::sxdg()),
        Instruction::U3 {
            qubit,
            theta,
            phi,
            lam,
        }
        | Instruction::U {
            qubit,
            theta,
            phi,
            lam,
        } => mps.apply_1q(*qubit, &gates::u3(*theta, *phi, *lam)),
        Instruction::U2 { qubit, phi, lam } => mps.apply_1q(*qubit, &gates::u2(*phi, *lam)),
        Instruction::U1 { qubit, lam } | Instruction::P { qubit, lam } => {
            mps.apply_1q(*qubit, &gates::u1(*lam));
        }
        Instruction::Rx { qubit, theta } => mps.apply_1q(*qubit, &gates::rx(*theta)),
        Instruction::Ry { qubit, theta } => mps.apply_1q(*qubit, &gates::ry(*theta)),
        Instruction::Rz { qubit, phi } => mps.apply_1q(*qubit, &gates::rz(*phi)),
        Instruction::Cx { control, target } => mps.apply_2q_with_kernel_hint(
            *control,
            *target,
            &gates::cnot(),
            lookahead,
            Some(cnot_kernel()),
        )?,
        Instruction::Cz { control, target } => mps.apply_2q_with_kernel_hint(
            *control,
            *target,
            &gates::cz(),
            lookahead,
            Some(cz_kernel()),
        )?,
        Instruction::Cy { control, target } => mps.apply_2q_with_kernel_hint(
            *control,
            *target,
            &gates::cy(),
            lookahead,
            Some(cy_kernel()),
        )?,
        Instruction::Ch { control, target } => {
            mps.apply_2q(*control, *target, &gates::ch(), lookahead)?
        }
        Instruction::Swap { a, b } => {
            mps.apply_2q_with_kernel_hint(*a, *b, &gates::swap(), lookahead, Some(swap_kernel()))?
        }
        Instruction::Csx { control, target } => {
            mps.apply_2q(*control, *target, &gates::csx(), lookahead)?
        }
        Instruction::Crx {
            control,
            target,
            theta,
        } => mps.apply_2q(*control, *target, &gates::crx(*theta), lookahead)?,
        Instruction::Cry {
            control,
            target,
            theta,
        } => mps.apply_2q(*control, *target, &gates::cry(*theta), lookahead)?,
        Instruction::Crz {
            control,
            target,
            lam,
        } => mps.apply_2q(*control, *target, &gates::crz(*lam), lookahead)?,
        Instruction::Cu1 {
            control,
            target,
            lam,
        }
        | Instruction::Cp {
            control,
            target,
            lam,
        } => mps.apply_2q(*control, *target, &gates::cu1(*lam), lookahead)?,
        Instruction::Cu3 {
            control,
            target,
            theta,
            phi,
            lam,
        } => mps.apply_2q(
            *control,
            *target,
            &gates::cu3(*theta, *phi, *lam),
            lookahead,
        )?,
        Instruction::Cu {
            control,
            target,
            theta,
            phi,
            lam,
            gamma,
        } => mps.apply_2q(
            *control,
            *target,
            &gates::cu(*theta, *phi, *lam, *gamma),
            lookahead,
        )?,
        Instruction::Rxx { a, b, theta } => mps.apply_2q(*a, *b, &gates::rxx(*theta), lookahead)?,
        Instruction::Rzz { a, b, theta } => mps.apply_2q(*a, *b, &gates::rzz(*theta), lookahead)?,
        Instruction::Gate { name, qubits, .. } => match name.to_lowercase().as_str() {
            "remote_link_phi_plus" | "remote_epr" | "epr" => {
                mps.apply_2q(qubits[0], qubits[1], &gates::phi_plus(), lookahead)?;
            }
            "remote_link_psi_minus" => {
                mps.apply_2q(qubits[0], qubits[1], &gates::psi_minus(), lookahead)?;
            }
            "remote_link_psi_plus" => {
                mps.apply_2q(qubits[0], qubits[1], &gates::psi_plus(), lookahead)?;
            }
            "nonlocal_cz" | "remote_cz" => {
                mps.apply_2q_with_kernel_hint(
                    qubits[0],
                    qubits[1],
                    &gates::cz(),
                    lookahead,
                    Some(cz_kernel()),
                )?;
            }
            "remote_cx" => {
                mps.apply_2q_with_kernel_hint(
                    qubits[0],
                    qubits[1],
                    &gates::cnot(),
                    lookahead,
                    Some(cnot_kernel()),
                )?;
            }
            "remote_barrier" => {}
            "remote_cu1" => {
                return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                    "Symbolic 'remote_cu1' cannot be simulated natively. Distribute with lowered=True.",
                ));
            }
            other if other.starts_with("circuit-") => {
                return Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
                    "Opaque symbolic subcircuit {other:?} cannot be simulated natively. Distribute with lowered=True."
                )));
            }
            other => {
                return Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
                    "MPS simulator does not support generic gate {other:?}"
                )));
            }
        },
        Instruction::Measure { qubit, cbit } => {
            let outcome = mps.measure(*qubit, rng);
            cbits.insert(*cbit, outcome as i32);
        }
        Instruction::Reset { qubit } => {
            let outcome = mps.measure(*qubit, rng);
            if outcome == 1 {
                mps.apply_1q(*qubit, &gates::X);
            }
        }
        Instruction::Conditional { condition, op } => {
            let mut actual: u64 = 0;
            for bit in 0..condition.creg_size {
                let val = *cbits.get(&(condition.creg_base + bit)).unwrap_or(&0) as u64;
                actual |= val << bit;
            }
            if actual == condition.creg_value {
                run_instruction(mps, op, &[], cbits, rng)?;
            }
        }
        _ => {
            return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "MPS simulator only supports one- and two-qubit unitary gates plus measurement, reset, and conditionals",
            ));
        }
    }
    Ok(())
}

fn reverse_2q_order(mat: &[[C; 4]; 4]) -> [[C; 4]; 4] {
    let mut out = [[C::new(0.0, 0.0); 4]; 4];
    for a_out in 0..2 {
        for b_out in 0..2 {
            for a_in in 0..2 {
                for b_in in 0..2 {
                    let row = a_out * 2 + b_out;
                    let col = a_in * 2 + b_in;
                    let swapped_row = b_out * 2 + a_out;
                    let swapped_col = b_in * 2 + a_in;
                    out[row][col] = mat[swapped_row][swapped_col];
                }
            }
        }
    }
    out
}
