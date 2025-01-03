//! algorithms/pesa_ii/evolutionary.rs
//! Implements the first phase of the algorithm (Genetic algorithm)
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2024 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::algorithms::pesa_ii::{hypergrid, HyperBox, Solution};
use crate::operators::crossover;
use crate::operators::generate_population;
use crate::operators::get_fitness;
use crate::operators::mutation;

use rayon::prelude::*;
use rustc_hash::FxBuildHasher;
use std::collections::HashMap;

use crate::graph::Graph;
use crate::utils::args::AGArgs;

pub const MAX_ARCHIVE_SIZE: usize = 100;

#[derive(Debug)]
struct BestFitnessGlobal {
    value: f64,        // Current best global value
    count: usize,      // Count of generations with the same value
    exhaustion: usize, // Max of generations with the same value
}

impl Default for BestFitnessGlobal {
    fn default() -> Self {
        BestFitnessGlobal {
            value: f64::MIN,
            count: 0,
            exhaustion: 100,
        }
    }
}

impl BestFitnessGlobal {
    fn verify_convergence(&mut self, best_local_fitness: f64) -> bool {
        if self.value < best_local_fitness {
            self.value = best_local_fitness;
            self.count = 0;
            return false;
        }

        self.count += 1;
        if self.count > self.exhaustion {
            self.count = 0;
            return true;
        }
        false
    }
}

pub fn evolutionary_phase(
    graph: &Graph,
    args: &AGArgs,
    degrees: &HashMap<i32, usize, FxBuildHasher>,
) -> (Vec<Solution>, Vec<f64>) {
    let mut rng = rand::thread_rng();
    let mut archive: Vec<Solution> = Vec::with_capacity(args.pop_size);
    let mut population = generate_population(graph, args.pop_size);
    let mut best_fitness_history: Vec<f64> = Vec::with_capacity(args.num_gens);

    let mut max_local: BestFitnessGlobal = BestFitnessGlobal::default();

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

        // PESA-II Selection + Reproduction
        let mut new_population = Vec::with_capacity(args.pop_size);
        while new_population.len() < args.pop_size {
            let parent1 = hypergrid::select(&hyperboxes, &mut rng);
            let parent2 = hypergrid::select(&hyperboxes, &mut rng);

            let mut child = crossover(&parent1.partition, &parent2.partition);
            mutation(&mut child, graph, args.mut_rate);
            new_population.push(child);
        }

        population = new_population;

        // Early stopping
        if max_local.verify_convergence(best_fitness) && args.debug {
            println!("[evolutionary_phase]: Converged!");
            break;
        }

        if args.debug {
            println!(
                "\x1b[1A\x1b[2K[evolutionary_phase]: gen: {} | bf: {:.4} | pop/arch: {}/{} | ba: {:.4} |",
                generation,
                best_fitness,
                population.len(),
                archive.len(),
                max_local.value,
            );
        }
    }

    (archive, best_fitness_history)
}
