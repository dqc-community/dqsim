use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use num_complex::Complex64;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

#[path = "../src/engine.rs"]
#[allow(dead_code)]
mod engine;
#[path = "../src/gates.rs"]
#[allow(dead_code)]
mod gates;

use engine::{apply_n_qubit_seq, apply_one_qubit_seq, measure_qubit_seq};

type C = Complex64;

fn zero_state(n: usize) -> Vec<C> {
    let mut state = vec![C::new(0.0, 0.0); 1 << n];
    state[0] = C::new(1.0, 0.0);
    state
}

fn mat4(m: [[C; 4]; 4]) -> Vec<Vec<C>> {
    m.iter().map(|row| row.to_vec()).collect()
}

fn bench_apply_one_qubit_seq(c: &mut Criterion) {
    let mut group = c.benchmark_group("apply_one_qubit_seq");
    for n in [8usize, 12, 16] {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            let gate = gates::h();
            b.iter_batched(
                || zero_state(n),
                |mut state| {
                    apply_one_qubit_seq(
                        black_box(&mut state),
                        black_box(&gate),
                        black_box(n / 2),
                        black_box(n),
                        black_box(&[]),
                    );
                    black_box(state);
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

fn bench_apply_n_qubit_seq(c: &mut Criterion) {
    let mut group = c.benchmark_group("apply_n_qubit_seq_cnot");
    let gate = mat4(gates::cnot());
    for n in [8usize, 12, 16] {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter_batched(
                || zero_state(n),
                |mut state| {
                    apply_n_qubit_seq(
                        black_box(&mut state),
                        black_box(&gate),
                        black_box(&[0, n - 1]),
                        black_box(n),
                    );
                    black_box(state);
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

fn bench_measure_qubit_seq(c: &mut Criterion) {
    let mut group = c.benchmark_group("measure_qubit_seq");
    for n in [8usize, 12, 16] {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter_batched(
                || {
                    let mut state = zero_state(n);
                    apply_one_qubit_seq(&mut state, &gates::h(), n / 2, n, &[]);
                    (state, ChaCha8Rng::seed_from_u64(1234))
                },
                |(mut state, mut rng)| {
                    let outcome = measure_qubit_seq(
                        black_box(&mut state),
                        black_box(n / 2),
                        black_box(n),
                        black_box(&mut rng),
                    );
                    black_box((state, outcome));
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_apply_one_qubit_seq,
    bench_apply_n_qubit_seq,
    bench_measure_qubit_seq
);
criterion_main!(benches);
