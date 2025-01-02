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

pub fn crossover(parent1: &Partition, parent2: &Partition) -> Partition {
    crossover::optimized_crossover(parent1, parent2)
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
        objective::calculate_objectives(&graph, &partition, &graph.precompute_degress(), false);

    metrics.get_modularity()
}
