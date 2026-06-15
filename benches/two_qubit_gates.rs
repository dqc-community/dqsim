use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use num_complex::Complex64;

#[path = "../src/engine.rs"]
#[allow(dead_code)]
mod engine;
#[path = "../src/gates.rs"]
#[allow(dead_code)]
mod gates;

type C = Complex64;
type Mat4Factory = fn() -> [[C; 4]; 4];

const TWO_QUBIT_GATES: &[(&str, Mat4Factory)] = &[
    ("cx", gates::cnot),
    ("cz", gates::cz),
    ("cy", gates::cy),
    ("ch", gates::ch),
    ("swap", gates::swap),
    ("csx", gates::csx),
    ("crx", || gates::crx(0.37)),
    ("cry", || gates::cry(0.37)),
    ("crz", || gates::crz(0.37)),
    ("cu1", || gates::cu1(0.37)),
    ("cp", || gates::cp(0.37)),
    ("cu3", || gates::cu3(0.37, 0.23, 0.11)),
    ("cu", || gates::cu(0.37, 0.23, 0.11, 0.07)),
    ("rxx", || gates::rxx(0.37)),
    ("rzz", || gates::rzz(0.37)),
    ("remote_link_psi_minus", gates::psi_minus),
    ("remote_link_psi_plus", gates::psi_plus),
    ("remote_link_phi_plus", gates::phi_plus),
];

fn zero_state(n: usize) -> Vec<C> {
    let mut state = vec![C::new(0.0, 0.0); 1 << n];
    state[0] = C::new(1.0, 0.0);
    state
}

fn mat4(m: [[C; 4]; 4]) -> Vec<Vec<C>> {
    m.iter().map(|row| row.to_vec()).collect()
}

fn bench_two_qubit_gates(c: &mut Criterion) {
    let mut group = c.benchmark_group("two_qubit_gate_total");
    group.warm_up_time(Duration::from_millis(100));
    group.measurement_time(Duration::from_millis(400));
    group.sample_size(20);

    for n in [4usize, 8, 12, 16] {
        let qubits = [0usize, n - 1];
        for &(name, gate) in TWO_QUBIT_GATES {
            group.bench_with_input(BenchmarkId::new(name, format!("n={n}")), &n, |b, &n| {
                b.iter_batched(
                    || zero_state(n),
                    |mut state| {
                        let matrix = mat4(gate());
                        engine::apply_n_qubit_seq(
                            black_box(&mut state),
                            black_box(&matrix),
                            black_box(&qubits),
                            black_box(n),
                        );
                        black_box(state);
                    },
                    criterion::BatchSize::SmallInput,
                );
            });
        }
    }

    group.finish();
}

criterion_group!(benches, bench_two_qubit_gates);
criterion_main!(benches);
