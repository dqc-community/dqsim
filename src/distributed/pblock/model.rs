use std::collections::HashMap;

use num_complex::Complex64;

pub(super) type C = Complex64;

pub(super) struct Block {
    pub(super) state: Vec<C>,
    pub(super) qubits: Vec<usize>,
}

impl Block {
    pub(super) fn new(mut qubits: Vec<usize>) -> Self {
        qubits.sort();
        qubits.dedup();
        let n = qubits.len();
        let mut state = vec![C::new(0.0, 0.0); 1 << n];
        state[0] = C::new(1.0, 0.0);
        Block { state, qubits }
    }

    pub(super) fn local(&self, phys: usize) -> usize {
        self.qubits
            .iter()
            .position(|&q| q == phys)
            .unwrap_or_else(|| panic!("qubit {} not in block {:?}", phys, self.qubits))
    }
}

pub(super) fn merge_blocks(a: Block, b: Block) -> Block {
    let mut qubits: Vec<usize> = a.qubits.iter().chain(b.qubits.iter()).cloned().collect();
    qubits.sort();
    qubits.dedup();

    let n = qubits.len();
    let dim = 1 << n;
    let mut state = vec![C::new(0.0, 0.0); dim];

    for (i, amp) in state.iter_mut().enumerate() {
        let mut a_local = 0usize;
        for (bit, &aq) in a.qubits.iter().enumerate() {
            let merged_bit = qubits.iter().position(|&q| q == aq).unwrap();
            if (i >> merged_bit) & 1 == 1 {
                a_local |= 1 << bit;
            }
        }
        let mut b_local = 0usize;
        for (bit, &bq) in b.qubits.iter().enumerate() {
            let merged_bit = qubits.iter().position(|&q| q == bq).unwrap();
            if (i >> merged_bit) & 1 == 1 {
                b_local |= 1 << bit;
            }
        }
        *amp = a.state[a_local] * b.state[b_local];
    }

    Block { state, qubits }
}

pub(super) struct BlockPool {
    pub(super) blocks: Vec<Option<Block>>,
    pub(super) qubit_to_block: HashMap<usize, usize>,
}

impl BlockPool {
    pub(super) fn new(qubits_per_node: &HashMap<usize, Vec<usize>>) -> Self {
        let mut nodes: Vec<usize> = qubits_per_node.keys().cloned().collect();
        nodes.sort();

        let mut blocks = Vec::new();
        let mut qubit_to_block = HashMap::new();

        for node in nodes {
            let qubits = qubits_per_node[&node].clone();
            let block_idx = blocks.len();
            for &q in &qubits {
                qubit_to_block.insert(q, block_idx);
            }
            blocks.push(Some(Block::new(qubits)));
        }

        BlockPool {
            blocks,
            qubit_to_block,
        }
    }

    fn block_of(&self, phys: usize) -> usize {
        *self
            .qubit_to_block
            .get(&phys)
            .unwrap_or_else(|| panic!("qubit {} is not assigned to any block", phys))
    }

    fn merge(&mut self, keep_idx: usize, drop_idx: usize) {
        if keep_idx == drop_idx {
            return;
        }
        let a = self.blocks[keep_idx].take().unwrap();
        let b = self.blocks[drop_idx].take().unwrap();
        for &q in &b.qubits {
            self.qubit_to_block.insert(q, keep_idx);
        }
        self.blocks[keep_idx] = Some(merge_blocks(a, b));
    }

    pub(super) fn ensure_single_block(&mut self, qubits: &[usize]) -> usize {
        let first = match qubits.first() {
            Some(&q) => q,
            None => return 0,
        };
        let target = self.block_of(first);
        for &q in &qubits[1..] {
            let bq = self.block_of(q);
            if bq != target {
                self.merge(target, bq);
            }
        }
        target
    }

    pub(super) fn merge_all(mut self) -> Block {
        let mut result: Option<Block> = None;
        for block in self.blocks.drain(..).flatten() {
            result = Some(match result {
                None => block,
                Some(acc) => merge_blocks(acc, block),
            });
        }
        result.unwrap_or_else(|| Block::new(vec![]))
    }
}

pub(super) fn m4(m: [[C; 4]; 4]) -> Vec<Vec<C>> {
    m.iter().map(|row| row.to_vec()).collect()
}
pub(super) fn m8(m: [[C; 8]; 8]) -> Vec<Vec<C>> {
    m.iter().map(|row| row.to_vec()).collect()
}
pub(super) fn m16(m: [[C; 16]; 16]) -> Vec<Vec<C>> {
    m.iter().map(|row| row.to_vec()).collect()
}

pub(super) fn m32(m: [[C; 32]; 32]) -> Vec<Vec<C>> {
    m.iter().map(|row| row.to_vec()).collect()
}
