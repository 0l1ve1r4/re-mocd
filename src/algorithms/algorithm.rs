//! algorithms/algorithm.rs
//! Genetic algorithm without PESA-II Implementation
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2024 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use core::f64;
use rand::seq::SliceRandom;
use rayon::prelude::*;

use crate::graph::{Graph, Partition};
use crate::utils::args::AGArgs;

use crate::operators::crossover;
use crate::operators::generate_population;
use crate::operators::get_fitness;
use crate::operators::metrics::Metrics;
use crate::operators::mutation;
use crate::operators::selection;

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
            exhaustion: 5,
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
    let mut population = generate_population(graph, args.pop_size);
    let mut best_fitness_history = Vec::with_capacity(args.num_gens);
    let degress = graph.precompute_degress();

    let mut max_local: BestFitnessGlobal = BestFitnessGlobal::default();

    /*  1. Evolution */
    for generation in 0..args.num_gens {
        let fitnesses: Vec<Metrics> = if args.parallelism {
            population
                .par_iter()
                .map(|partition| get_fitness(graph, partition, &degress, true))
                .collect()
        } else {
            population
                .iter()
                .map(|partition| get_fitness(graph, partition, &degress, false))
                .collect()
        };

        let best_fitness = fitnesses
            .iter()
            .map(|m| m.modularity)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        best_fitness_history.push(best_fitness);

        // 1.1. Selection
        population = selection(population, fitnesses, args.pop_size, 2);

        // 1.2. New population
        let mut new_population = Vec::with_capacity(args.pop_size);
        while new_population.len() < args.pop_size {
            let parent1 = population.choose(&mut rng).unwrap();
            let parent2 = population.choose(&mut rng).unwrap();
            let mut child = crossover(parent1, parent2);

            mutation(&mut child, graph, args.mut_rate);
            new_population.push(child);
        }
        population = new_population;

        if max_local.verify_exhaustion(best_fitness) && args.debug {
            println!("[Optimization]: Converged, breaking...");
            break;
        }

        if args.debug {
            // cursor clear
            print!("\x1b[1A\x1b[2K");
            println!(
                "[algorithms/pesa_ii.rs]: gen: {} | best fitness: {:.4} | pop.len: {}",
                generation,
                best_fitness,
                population.len()
            );
        }
    }

    // Find best partition
    let best_partition = population
        .into_par_iter()
        .max_by_key(|partition| {
            let metrics = get_fitness(graph, partition, &degress, args.parallelism);
            (metrics.modularity * 1000.0) as i64
        })
        .unwrap();

    let max_modularity = best_fitness_history
        .iter()
        .fold(None, |max, &val| match max {
            None => Some(val),
            Some(max_val) if val > max_val && !val.is_nan() => Some(val),
            Some(max_val) => Some(max_val),
        });

    (
        best_partition,
        best_fitness_history,
        max_modularity.unwrap(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_calculate_objectives() {
        let graph: Graph = Graph::new();
        let partition: Partition = Partition::new();

        assert_eq!(
            get_fitness(&graph, &partition, &graph.precompute_degress(), true),
            Metrics {
                inter: 0.0,
                intra: 0.0,
                modularity: 0.0,
            }
        );
    }

    #[test]
    #[should_panic]
    fn test_panic_ga() {
        let graph: Graph = Graph::new();
        run(&graph, AGArgs::default());
    }
}
