//! algorithms/pesa_ii.rs
//! Implements the Pareto Envelope-based Selection Algorithm II (PESA-II)
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2024 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rayon::prelude::*;
use rustc_hash::FxBuildHasher;
use std::collections::HashMap;

use crate::graph::{Graph, Partition};
use crate::utils::args::AGArgs;

use crate::operators::crossover;
use crate::operators::generate_population;
use crate::operators::get_fitness;
use crate::operators::mutation;

mod hypergrid;
use hypergrid::{HyperBox, Solution};

#[derive(Debug)]
struct BestFitnessGlobal {
    value: f64,        // Current best global value
    count: usize,      // Count of generations with the same value
    exhaustion: usize, // Max of generations with the same value
    epsilon: f64,
}

impl Default for BestFitnessGlobal {
    fn default() -> Self {
        BestFitnessGlobal {
            value: f64::MIN,
            count: 0,
            exhaustion: 15,
            epsilon: 1e-6,
        }
    }
}

impl BestFitnessGlobal {
    fn verify_exhaustion(&mut self, best_local_fitness: f64) -> bool {
        if (self.value - best_local_fitness).abs() > self.epsilon {
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

pub fn run(graph: &Graph, args: AGArgs) -> (Partition, Vec<f64>, f64) {
    let mut rng = rand::thread_rng();
    let mut archive: Vec<Solution> = Vec::with_capacity(args.pop_size);
    let mut population = generate_population(graph, args.pop_size);
    let mut best_fitness_history: Vec<f64> = Vec::with_capacity(args.num_gens);
    let degrees: HashMap<i32, usize, FxBuildHasher> = graph.precompute_degress();

    let mut max_local: BestFitnessGlobal = BestFitnessGlobal::default();

    for generation in 0..args.num_gens {
        let solutions: Vec<Solution> = population
            .par_chunks(population.len() / rayon::current_num_threads())
            .flat_map(|chunk| {
                chunk
                    .iter()
                    .map(|partition| {
                        let metrics = get_fitness(graph, partition, &degrees, true);
                        hypergrid::Solution {
                            partition: partition.clone(),
                            objectives: vec![metrics.modularity, metrics.inter, metrics.intra],
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        // Update archive with non-dominated solutions
        // TODO: Optimize archive
        for solution in solutions {
            if !archive
                .iter()
                .any(|archived: &Solution| archived.dominates(&solution))
            {
                // Remove solutions from archive that are dominated by the new solution
                archive.retain(|archived: &hypergrid::Solution| !solution.dominates(archived));
                archive.push(solution);
            }
        }

        let hyperboxes: Vec<HyperBox> = hypergrid::create(&archive, hypergrid::GRID_DIVISIONS);

        // Record best fitness (using modularity as primary objective)
        let best_fitness = archive
            .iter()
            .map(|s| s.objectives[0])
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        best_fitness_history.push(best_fitness);

        // Selection and reproduction
        let mut new_population = Vec::with_capacity(args.pop_size);
        while new_population.len() < args.pop_size {
            // Select parents using PESA-II selection method

            let parent1: &Solution = hypergrid::select(&hyperboxes, &mut rng);
            let parent2: &Solution = hypergrid::select(&hyperboxes, &mut rng);

            let mut child = crossover(&parent1.partition, &parent2.partition);

            mutation(&mut child, graph, args.mut_rate);
            new_population.push(child);
        }

        population = new_population;

        // Early stopping condition
        if max_local.verify_exhaustion(best_fitness) && args.debug {
            println!("[algorithms/pesa_ii.rs]: Converged, breaking...");
            break;
        }

        if args.debug {
            // cursor clear
            println!(
                "\x1b[1A\x1b[2K[algorithms/pesa_ii.rs]: gen: {} | bf: {:.4} | pop/arch: {}/{} |",
                generation,
                best_fitness,
                population.len(),
                archive.len()
            );
        }
    }

    // Find best solution from archive (using modularity as primary objective)
    let best_solution = archive
        .par_iter()
        .max_by(|a, b| a.objectives[0].partial_cmp(&b.objectives[0]).unwrap())
        .unwrap();

    (
        best_solution.partition.clone(),
        best_fitness_history,
        best_solution.objectives[0],
    )
}
