//! operators/mod.rs
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2024 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::graph::{Graph, Partition};
use metrics::Metrics;
use rustc_hash::FxBuildHasher;
use std::collections::HashMap;

pub mod metrics;

mod crossover;
mod mutation;
mod objective;
mod population;
mod selection;

/// Tracks the convergence status of a genetic algorithm by monitoring fitness improvements
/// over successive generations.
#[derive(Debug, Clone)]
pub struct ConvergenceTracker {
    best_fitness: f64,                  // The highest fitness value observed so far
    stagnant_generations: usize,        // Number of consecutive generations without improvement
    convergence_threshold: usize,       // Maximum allowed generations without improvement
    last_improvement_generation: usize, // Stores the generation number when the best fitness was last updated
    current_generation: usize,          // Current generation number
}

impl ConvergenceTracker {
    pub fn new(convergence_threshold: usize) -> Self {
        Self {
            best_fitness: f64::NEG_INFINITY,
            stagnant_generations: 0,
            convergence_threshold,
            last_improvement_generation: 0,
            current_generation: 0,
        }
    }

    /// Updates the tracker with the current generation's best fitness value
    pub fn update(&mut self, current_fitness: f64) -> bool {
        self.current_generation += 1;

        if current_fitness > self.best_fitness {
            self.best_fitness = current_fitness;
            self.stagnant_generations = 0;
            self.last_improvement_generation = self.current_generation;
            return false;
        }

        self.stagnant_generations += 1;
        self.stagnant_generations >= self.convergence_threshold
    }

    pub fn generations_since_improvement(&self) -> usize {
        self.current_generation - self.last_improvement_generation
    }

    pub fn best_fitness(&self) -> f64 {
        self.best_fitness
    }

    /// Returns the current convergence progress as a percentage
    pub fn convergence_progress(&self) -> f64 {
        (self.stagnant_generations as f64 / self.convergence_threshold as f64) * 100.0
    }

    /// Resets the tracker while maintaining the convergence threshold
    pub fn reset(&mut self) {
        self.best_fitness = f64::NEG_INFINITY;
        self.stagnant_generations = 0;
        self.last_improvement_generation = 0;
        self.current_generation = 0;
    }
}

impl Default for ConvergenceTracker {
    fn default() -> Self {
        Self::new(100) // Default convergence threshold of 100 generations
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convergence_detection() {
        let mut tracker = ConvergenceTracker::new(3);

        assert!(!tracker.update(1.0)); // First value
        assert!(!tracker.update(2.0)); // Improvement
        assert!(!tracker.update(2.0)); // No improvement, count = 1
        assert!(!tracker.update(2.0)); // No improvement, count = 2
        assert!(tracker.update(2.0)); // No improvement, count = 3, converged
    }

    #[test]
    fn test_progress_tracking() {
        let mut tracker = ConvergenceTracker::new(4);

        tracker.update(1.0);
        tracker.update(1.0);

        assert_eq!(tracker.convergence_progress(), 50.0); // 2/4 * 100
        assert_eq!(tracker.generations_since_improvement(), 2);
        assert_eq!(tracker.best_fitness(), 1.0);
    }

    #[test]
    fn test_reset() {
        let mut tracker = ConvergenceTracker::new(3);

        tracker.update(1.0);
        tracker.update(2.0);
        tracker.reset();

        assert_eq!(tracker.best_fitness(), f64::NEG_INFINITY);
        assert_eq!(tracker.stagnant_generations, 0);
        assert_eq!(tracker.current_generation, 0);
    }
}

pub fn crossover(parent1: &Partition, parent2: &Partition, rate: f64) -> Partition {
    crossover::optimized_crossover(parent1, parent2, rate)
}

pub fn mutation(partition: &mut Partition, graph: &Graph, mutation_rate: f64) {
    mutation::optimized_mutate(partition, graph, mutation_rate);
}

pub fn selection(
    population: Vec<Partition>,
    fitnesses: Vec<metrics::Metrics>,
    pop_size: usize,
    tournament_size: usize,
) -> Vec<Partition> {
    selection::optimized_selection(population, fitnesses, pop_size, tournament_size)
}

pub fn get_fitness(
    graph: &Graph,
    partition: &Partition,
    degrees: &HashMap<i32, usize, FxBuildHasher>,
    parallel: bool,
) -> metrics::Metrics {
    objective::calculate_objectives(graph, partition, degrees, parallel)
}

pub fn generate_population(graph: &Graph, population_size: usize) -> Vec<Partition> {
    population::generate_optimized_population(graph, population_size)
}

#[allow(dead_code)]
pub fn get_modularity_from_partition(partition: &Partition, graph: &Graph) -> f64 {
    let metrics: Metrics =
        objective::calculate_objectives(graph, partition, &graph.precompute_degress(), false);

    metrics.get_modularity()
}
