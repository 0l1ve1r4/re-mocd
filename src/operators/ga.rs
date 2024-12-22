/// This Source Code Form is subject to the terms of The GNU General Public License v3.0
/// Copyright 2024 - Guilherme Santos. If a copy of the GPL3 was not distributed with this
/// file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html
use petgraph::graph::{Graph, NodeIndex};
use petgraph::Undirected;
use rand::prelude::*;

use crate::operators::Partition;

/// Generates the initial population of random partitions.
pub fn generate_initial_population(
    graph: &Graph<(), (), Undirected>,
    population_size: usize,
) -> Vec<Partition> {
    let mut rng = thread_rng();
    let nodes: Vec<NodeIndex> = graph.node_indices().collect();
    let mut population = Vec::with_capacity(population_size);

    for _ in 0..population_size {
        let partition: Partition = nodes
            .iter()
            .map(|&node| (node, rng.gen_range(0..nodes.len())))
            .collect();
        population.push(partition);
    }

    population
}

/// Performs two-point crossover between two parents.
pub fn crossover(parent1: &Partition, parent2: &Partition) -> Partition {
    let mut rng = thread_rng();
    let keys: Vec<&NodeIndex> = parent1.keys().collect();
    if keys.len() < 2 {
        return parent1.clone();
    }

    let len = keys.len();
    let mut idxs = (0..len).collect::<Vec<_>>();
    idxs.shuffle(&mut rng);
    let i1 = idxs[0];
    let i2 = idxs[1];
    let idx1 = i1.min(i2);
    let idx2 = i1.max(i2);

    let mut child = parent1.clone();
    for &key in &keys[idx1..=idx2] {
        child.insert(*key, parent2[key]);
    }

    child
}

/// Mutates a partition by changing a node's community to that of a neighbor.
pub fn mutate(partition: &mut Partition, graph: &Graph<(), (), Undirected>) {
    let mut rng = thread_rng();
    let nodes: Vec<&NodeIndex> = partition.keys().collect();
    if nodes.is_empty() {
        return;
    }

    let node = nodes.choose(&mut rng).unwrap();
    let neighbors: Vec<NodeIndex> = graph.neighbors(**node).collect();
    if !neighbors.is_empty() {
        let neighbor = neighbors.choose(&mut rng).unwrap();
        partition.insert(**node, *partition.get(neighbor).unwrap());
    }
}

/// Selects the top half of the population based on modularity.
pub fn selection(population: &[Partition], fitnesses: &[(f64, f64, f64)]) -> Vec<Partition> {
    let mut pop_fitness: Vec<(&Partition, f64)> = population
        .iter()
        .zip(fitnesses.iter().map(|f| f.0))
        .collect();

    // Sort by modularity descending
    pop_fitness.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Select top half
    pop_fitness
        .iter()
        .take(population.len() / 2)
        .map(|&(partition, _)| partition.clone())
        .collect()
}
