//! algorithms/pesa_ii/evolutionary.rs
//! Implements the first phase of the algorithm (Genetic algorithm)
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2024 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::algorithms::pesa_ii::{hypergrid, HyperBox, Solution};
use crate::operators::*;

use rayon::prelude::*;
use rustc_hash::FxBuildHasher;
use std::collections::HashMap;
use std::sync::Arc;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::graph::{Graph, Partition};
use crate::utils::args::AGArgs;

pub const MAX_ARCHIVE_SIZE: usize = 100;

/// Thread-safe random number generator management
struct SafeRng {
    seed_counter: AtomicU64,
}

impl SafeRng {
    fn new() -> Self {
        Self {
            seed_counter: AtomicU64::new(0),
        }
    }

    fn get_rng(&self) -> ChaCha8Rng {
        let seed = self.seed_counter.fetch_add(1, Ordering::SeqCst);
        ChaCha8Rng::seed_from_u64(seed)
    }
}

/// Parallel population generation using PESA-II selection and reproduction
fn generate_new_population(
    hyperboxes: &[HyperBox],
    args: &AGArgs,
    graph: &Graph,
) -> Vec<Partition> {
    // Create thread-safe RNG
    let safe_rng = Arc::new(SafeRng::new());

    // Calculate chunk size based on available threads
    let chunk_size = (args.pop_size / rayon::current_num_threads()).max(1);

    // Create chunks for parallel processing
    let chunks: Vec<_> = (0..args.pop_size)
        .collect::<Vec<_>>()
        .chunks(chunk_size)
        .map(|c| c.len())
        .collect();

    // Process chunks in parallel
    let results: Vec<Partition> = chunks
        .par_iter()
        .flat_map(|&chunk_size| {
            let mut local_children = Vec::with_capacity(chunk_size);
            let safe_rng = Arc::clone(&safe_rng);

            // Create a thread-local RNG
            let mut local_rng = safe_rng.get_rng();

            // Process each chunk
            for _ in 0..chunk_size {
                // Thread-safe selection of parents
                let parent1 = hypergrid::select(hyperboxes, &mut local_rng);
                let parent2 = hypergrid::select(hyperboxes, &mut local_rng);

                // Perform crossover and mutation
                let mut child = crossover(&parent1.partition, &parent2.partition, args.cross_rate);
                mutation(&mut child, graph, args.mut_rate);
                local_children.push(child);
            }

            local_children
        })
        .collect();

    results
}

pub fn evolutionary_phase(
    graph: &Graph,
    args: &AGArgs,
    degrees: &HashMap<i32, usize, FxBuildHasher>,
) -> (Vec<Solution>, Vec<f64>) {
    let mut archive: Vec<Solution> = Vec::with_capacity(args.pop_size);
    let mut population = generate_population(graph, args.pop_size);
    let mut best_fitness_history: Vec<f64> = Vec::with_capacity(args.num_gens);

    let mut max_local: ConvergenceCriteria = ConvergenceCriteria::default();

    for generation in 0..args.num_gens {
        // Evaluate current population and update archive
        let solutions: Vec<Solution> = population
            .par_chunks(population.len() / rayon::current_num_threads())
            .flat_map(|chunk| {
                chunk
                    .iter()
                    .map(|partition| {
                        let metrics = get_fitness(graph, partition, degrees, true);
                        Solution {
                            partition: partition.clone(),
                            objectives: vec![metrics.modularity, metrics.inter, metrics.intra],
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        // Update Pareto archive
        for solution in solutions {
            if !archive.iter().any(|archived| archived.dominates(&solution)) {
                archive.retain(|archived| !solution.dominates(archived));
                archive.push(solution);
            }
        }

        if archive.len() > MAX_ARCHIVE_SIZE {
            hypergrid::truncate_archive(&mut archive, MAX_ARCHIVE_SIZE);
        }

        // Create hyperboxes from archive
        let hyperboxes: Vec<HyperBox> = hypergrid::create(&archive, hypergrid::GRID_DIVISIONS);

        // Record the best fitness (using first objective as an example)
        let best_fitness = archive
            .iter()
            .map(|s| s.objectives[0])
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        best_fitness_history.push(best_fitness);

        // Generate new population in parallel
        population = generate_new_population(&hyperboxes, args, graph);

        // Early stopping
        if max_local.has_converged(best_fitness) {
            if args.debug { println!("[evolutionary_phase]: Converged!"); }
            break;
        }

        if args.debug {
            println!(
                "\x1b[1A\x1b[2K[evolutionary_phase]: gen: {} | bf: {:.4} | pop/arch: {}/{} | bA: {:.4} |",
                generation,
                best_fitness,
                population.len(),
                archive.len(),
                max_local.get_best_fitness(),
            );
        }
    }

    (archive, best_fitness_history)
}
