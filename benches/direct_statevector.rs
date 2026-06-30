// benches/direct_statevector.rs
use _core::engine::apply_one_qubit;
use _core::gates;
use _core::types::{Circuit, Instruction, Register};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use num_complex::Complex64 as C;
use std::collections::HashMap;
use std::time::Instant;

// ============================================================================
// Test Circuit Builders
// ============================================================================

/// Deutsch circuit: H, CX, H (2 qubits)
fn circuit_deutsch() -> Circuit {
    let mut qregs = HashMap::new();
    qregs.insert(
        "q".to_string(),
        Register {
            name: "q".to_string(),
            size: 2,
            base: 0,
        },
    );

    let instructions = vec![
        Instruction::H { qubit: 0 },
        Instruction::H { qubit: 1 },
        Instruction::Cx {
            control: 0,
            target: 1,
        },
        Instruction::H { qubit: 0 },
    ];

    Circuit {
        qregs,
        cregs: HashMap::new(),
        instructions,
    }
}

/// Toffoli circuit: multi-control structure (3 qubits)
fn circuit_toffoli() -> Circuit {
    let mut qregs = HashMap::new();
    qregs.insert(
        "q".to_string(),
        Register {
            name: "q".to_string(),
            size: 3,
            base: 0,
        },
    );

    let instructions = vec![
        Instruction::H { qubit: 0 },
        Instruction::H { qubit: 1 },
        Instruction::H { qubit: 2 },
        Instruction::Ccx {
            control1: 0,
            control2: 1,
            target: 2,
        },
        Instruction::H { qubit: 0 },
        Instruction::H { qubit: 1 },
        Instruction::H { qubit: 2 },
    ];

    Circuit {
        qregs,
        cregs: HashMap::new(),
        instructions,
    }
}

/// QFT circuit: quantum Fourier transform on 4 qubits
fn circuit_qft() -> Circuit {
    let mut qregs = HashMap::new();
    qregs.insert(
        "q".to_string(),
        Register {
            name: "q".to_string(),
            size: 4,
            base: 0,
        },
    );

    let mut instructions = vec![
        Instruction::H { qubit: 0 },
        Instruction::Cp {
            control: 1,
            target: 0,
            lam: std::f64::consts::PI / 2.0,
        },
        Instruction::Cp {
            control: 2,
            target: 0,
            lam: std::f64::consts::PI / 4.0,
        },
        Instruction::Cp {
            control: 3,
            target: 0,
            lam: std::f64::consts::PI / 8.0,
        },
    ];

    // Add more QFT-like structure
    for i in 1..4 {
        instructions.push(Instruction::H { qubit: i });
        for j in (i + 1)..4 {
            instructions.push(Instruction::Cp {
                control: j,
                target: i,
                lam: std::f64::consts::PI / (1 << (j - i)) as f64,
            });
        }
    }

    Circuit {
        qregs,
        cregs: HashMap::new(),
        instructions,
    }
}

/// Adder circuit: quantum adder pattern (4 qubits)
fn circuit_adder() -> Circuit {
    let mut qregs = HashMap::new();
    qregs.insert(
        "q".to_string(),
        Register {
            name: "q".to_string(),
            size: 4,
            base: 0,
        },
    );

    let instructions = vec![
        Instruction::H { qubit: 0 },
        Instruction::Cx {
            control: 0,
            target: 1,
        },
        Instruction::H { qubit: 2 },
        Instruction::Cx {
            control: 2,
            target: 3,
        },
        Instruction::Ccx {
            control1: 1,
            control2: 3,
            target: 0,
        },
        Instruction::Cx {
            control: 1,
            target: 3,
        },
        Instruction::Cx {
            control: 0,
            target: 1,
        },
    ];

    Circuit {
        qregs,
        cregs: HashMap::new(),
        instructions,
    }
}

// ============================================================================
// Simulation Runners
// ============================================================================

/// Run a circuit using direct gate application (our benchmark method)
fn run_circuit_direct(circuit: &Circuit) {
    let n = circuit.num_qubits();
    let mut state = vec![C::new(0.0, 0.0); 1 << n];
    state[0] = C::new(1.0, 0.0);

    for inst in &circuit.instructions {
        match inst {
            Instruction::H { qubit } => {
                apply_one_qubit(&mut state, &gates::h(), *qubit, n, &[]);
            }
            Instruction::X { qubit } => {
                apply_one_qubit(&mut state, &gates::X, *qubit, n, &[]);
            }
            Instruction::Y { qubit } => {
                apply_one_qubit(&mut state, &gates::Y, *qubit, n, &[]);
            }
            Instruction::Z { qubit } => {
                apply_one_qubit(&mut state, &gates::Z, *qubit, n, &[]);
            }
            Instruction::S { qubit } => {
                apply_one_qubit(&mut state, &gates::s_gate(), *qubit, n, &[]);
            }
            Instruction::T { qubit } => {
                apply_one_qubit(&mut state, &gates::t_gate(), *qubit, n, &[]);
            }
            Instruction::Rx { qubit, theta } => {
                apply_one_qubit(&mut state, &gates::rx(*theta), *qubit, n, &[]);
            }
            Instruction::Ry { qubit, theta } => {
                apply_one_qubit(&mut state, &gates::ry(*theta), *qubit, n, &[]);
            }
            Instruction::Rz { qubit, phi } => {
                apply_one_qubit(&mut state, &gates::rz(*phi), *qubit, n, &[]);
            }
            Instruction::Cx { control, target } => {
                apply_one_qubit(&mut state, &gates::X, *target, n, &[(*control, true)]);
            }
            Instruction::Cz { control, target } => {
                apply_one_qubit(&mut state, &gates::Z, *target, n, &[(*control, true)]);
            }
            Instruction::Cy { control, target } => {
                apply_one_qubit(&mut state, &gates::Y, *target, n, &[(*control, true)]);
            }
            Instruction::Ch { control, target } => {
                apply_one_qubit(&mut state, &gates::h(), *target, n, &[(*control, true)]);
            }
            Instruction::Ccx {
                control1,
                control2,
                target,
            } => {
                apply_one_qubit(
                    &mut state,
                    &gates::X,
                    *target,
                    n,
                    &[(*control1, true), (*control2, true)],
                );
            }
            Instruction::Cp {
                control,
                target,
                lam,
            } => {
                apply_one_qubit(&mut state, &gates::p(*lam), *target, n, &[(*control, true)]);
            }
            _ => {} // Skip other instructions for this benchmark
        }
    }

    let _result = black_box(state);
}

fn bench_statevector_direct(c: &mut Criterion) {
    let mut group = c.benchmark_group("test_circuits");

    // Reduce sample size to generate fewer flamegraphs
    group.sample_size(10);

    // Define test circuits from benchmarking suite
    let test_cases = vec![
        ("deutsch", circuit_deutsch()),
        ("toffoli", circuit_toffoli()),
        ("qft", circuit_qft()),
        ("adder", circuit_adder()),
    ];

    for (name, circuit) in test_cases {
        let qubits = circuit.num_qubits();
        let circuit_clone = circuit.clone();

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_q{}", name, qubits)),
            &name,
            |b, _| {
                b.iter_custom(|iters| {
                    #[cfg(unix)]
                    let profiler = pprof::ProfilerGuard::new(100).ok();

                    let start = Instant::now();
                    for _ in 0..iters {
                        run_circuit_direct(black_box(&circuit_clone));
                    }
                    let duration = start.elapsed();

                    #[cfg(unix)]
                    if let Some(profiler) = profiler {
                        if let Ok(report) = profiler.report().build() {
                            let flamegraph_path = format!("flamegraph_{}_{}.svg", name, qubits);
                            let file = std::fs::File::create(&flamegraph_path).unwrap();
                            let _ = report.flamegraph(file);
                            println!("Wrote flamegraph: {}", flamegraph_path);
                        }
                    }

                    duration
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_statevector_direct);
criterion_main!(benches);
