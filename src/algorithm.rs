// This Source Code Form is subject to the terms of The GNU General Public License v3.0
// Copyright 2024 - Guilherme Santos. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use petgraph::graph::{Graph, NodeIndex};
use petgraph::Undirected;
use rand::prelude::*;
use rayon::prelude::*;
use rustc_hash::FxHashMap as HashMap;
use std::f64::INFINITY;

type Partition = HashMap<NodeIndex, usize>;

/// Precompute degrees for each node. This function returns a vector indexed by NodeIndex.index().
fn compute_node_degrees(graph: &Graph<(), (), Undirected>) -> Vec<f64> {
    let mut degrees = vec![0.0; graph.node_count()];
    for node in graph.node_indices() {
        degrees[node.index()] = graph.neighbors(node).count() as f64;
    }
    degrees
}

/// Calculates the modularity, intra-community edges, and inter-community edges.
fn calculate_objectives(
    graph: &Graph<(), (), Undirected>,
    partition: &Partition,
    node_degrees: &[f64]
) -> (f64, f64, f64) {
    let total_edges: f64 = graph.edge_count() as f64;
    if total_edges == 0.0 {
        return (0.0, 0.0, 0.0);
    }
    let total_edges_doubled: f64 = 2.0 * total_edges;

    // Convert partition to a Vec-based membership for O(1) lookups:
    // membership[node.index()] = community_id
    let mut membership = vec![0; graph.node_count()];
    for (&node, &comm) in partition {
        membership[node.index()] = comm;
    }

    // Build sets of nodes by community
    let mut communities: HashMap<usize, Vec<NodeIndex>> = HashMap::default();
    for (&node, &community) in partition.iter() {
        communities.entry(community).or_default().push(node);
    }

    let community_vec: Vec<_> = communities.values().collect();

    // Parallel processing of communities
    let (intra_sum, inter_sum) = community_vec
        .par_iter()
        .map(|community_nodes| {
            let mut community_edges = 0.0;
            let mut community_degree_sum = 0.0;

            // Summation for edges within the same community
            for &node in community_nodes.iter() {
                community_degree_sum += node_degrees[node.index()];
                for neighbor in graph.neighbors(node) {
                    // Only count edge if neighbor is in the same community
                    if membership[neighbor.index()] == membership[node.index()] {
                        community_edges += 1.0;
                    }
                }
            }

            // Adjust for undirected graph (each edge counted twice)
            community_edges /= 2.0;
            let normalized_degree = community_degree_sum / total_edges_doubled;
            let inter = normalized_degree * normalized_degree;

            (community_edges, inter)
        })
        .reduce(
            || (0.0, 0.0),
            |(sum_edges1, sum_inter1), (sum_edges2, sum_inter2)| {
                (sum_edges1 + sum_edges2, sum_inter1 + sum_inter2)
            },
        );

    let intra = 1.0 - (intra_sum / total_edges);
    let mut modularity = 1.0 - intra - inter_sum;
    modularity = modularity.clamp(-1.0, 1.0);

    (modularity, intra, inter_sum)
}

/// Generates the initial population of random partitions.
fn generate_initial_population(
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
fn crossover(parent1: &Partition, parent2: &Partition) -> Partition {
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
fn mutate(partition: &mut Partition, graph: &Graph<(), (), Undirected>) {
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
fn selection(
    population: &[Partition],
    fitnesses: &[(f64, f64, f64)],
) -> Vec<Partition> {
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

/// Calculates the distance between two fitness tuples (for the final Max-Min selection).
fn calculate_distance(fitness1: &(f64, f64, f64), fitness2: &(f64, f64, f64)) -> f64 {
    let intra_diff = fitness1.1 - fitness2.1;
    let inter_diff = fitness1.2 - fitness2.2;
    (intra_diff * intra_diff + inter_diff * inter_diff).sqrt()
}

/// Generates a random graph by adding edges between random node pairs.
/// Optional optimization: check if an edge already exists before adding to avoid duplicates.
fn generate_random_graph(node_count: usize, edge_count: usize) -> Graph<(), (), Undirected> {
    let mut graph = Graph::<(), (), Undirected>::new_undirected();
    let mut rng = thread_rng();

    // Add nodes
    for _ in 0..node_count {
        graph.add_node(());
    }

    // Add random edges (optional: track inserted edges to avoid duplicates).
    for _ in 0..edge_count {
        let a = rng.gen_range(0..node_count);
        let b = rng.gen_range(0..node_count);
        if a != b {
            // If you want to avoid duplicates:
            // if !graph.contains_edge((a as u32).into(), (b as u32).into()) {
            //     graph.add_edge((a as u32).into(), (b as u32).into(), ());
            // }
            graph.add_edge((a as u32).into(), (b as u32).into(), ());
        }
    }

    graph
}

/// The main genetic algorithm function with parallel objective computation and caching.
pub fn genetic_algorithm(graph: &Graph<(), (), Undirected>,
    generations: usize,
    population_size: usize,) -> (
    Partition,
    Vec<(Partition, (f64, f64, f64), f64)>,
    Vec<(f64, f64, f64)>,
    Vec<(f64, f64, f64)>,
    Vec<f64>,
    Vec<f64>,
) {

    // Precompute degrees once for efficiency
    let node_degrees = compute_node_degrees(graph);
    let mut best_fitness_history = Vec::with_capacity(generations);
    let mut avg_fitness_history = Vec::with_capacity(generations);
    
    // 1. Generate initial population
    let mut population = generate_initial_population(graph, population_size);

    // 2. Evolution loop
    for _ in 0..generations {
        let fitnesses: Vec<(f64, f64, f64)> = population
            .par_iter()
            .map(|partition| calculate_objectives(graph, partition, &node_degrees))
            .collect();

        let modularity_values: Vec<f64> = fitnesses.iter().map(|f| f.0).collect();
        let best_fitness = modularity_values
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let avg_fitness = modularity_values.iter().sum::<f64>() / modularity_values.len() as f64;

        best_fitness_history.push(best_fitness);
        avg_fitness_history.push(avg_fitness);

        // Selection
        population = selection(&population, &fitnesses);

        // Generate new population
        let mut rng = thread_rng();
        let mut new_population = Vec::with_capacity(population_size);
        while new_population.len() < population_size {
            let parents: Vec<&Partition> = population.choose_multiple(&mut rng, 2).collect();
            if parents.len() < 2 {
                // If we cannot find two parents, reuse existing population
                new_population.extend_from_slice(&population);
                break;
            }
            let mut child = crossover(parents[0], parents[1]);
            mutate(&mut child, graph);
            new_population.push(child);
        }
        population = new_population;
    }

    // Final evaluation for real network
    let fitnesses: Vec<(f64, f64, f64)> = population
        .par_iter()
        .map(|partition| calculate_objectives(graph, partition, &node_degrees))
        .collect();

    let best_modularity = fitnesses
        .iter()
        .map(|f| f.0)
        .fold(f64::NEG_INFINITY, f64::max);

    let pareto_front: Vec<&Partition> = population
        .iter()
        .zip(fitnesses.iter())
        .filter(|(_, fitness)| fitness.0 == best_modularity)
        .map(|(p, _)| p)
        .collect();

    // 3. Run on random network of same size
    let node_count = graph.node_count();
    let edge_count = graph.edge_count();
    let random_graph = generate_random_graph(node_count, edge_count);

    // Precompute degrees for random graph
    let random_degrees = compute_node_degrees(&random_graph);

    let mut random_population = generate_initial_population(&random_graph, population_size);
    for _ in 0..generations {
        let fitnesses: Vec<(f64, f64, f64)> = random_population
            .par_iter()
            .map(|partition| calculate_objectives(&random_graph, partition, &random_degrees))
            .collect();

        random_population = selection(&random_population, &fitnesses);

        let mut rng = thread_rng();
        let mut new_population = Vec::with_capacity(population_size);
        while new_population.len() < population_size {
            let parents: Vec<&Partition> = random_population.choose_multiple(&mut rng, 2).collect();
            if parents.len() < 2 {
                new_population.extend_from_slice(&random_population);
                break;
            }
            let mut child = crossover(parents[0], parents[1]);
            mutate(&mut child, &random_graph);
            new_population.push(child);
        }
        random_population = new_population;
    }

    // Final evaluation for random network
    let random_fitnesses: Vec<(f64, f64, f64)> = random_population
        .par_iter()
        .map(|partition| calculate_objectives(&random_graph, partition, &random_degrees))
        .collect();

    let random_best_modularity = random_fitnesses
        .iter()
        .map(|f| f.0)
        .fold(f64::NEG_INFINITY, f64::max);

    let random_pareto_front: Vec<&(f64, f64, f64)> = random_fitnesses
        .iter()
        .filter(|fitness| fitness.0 == random_best_modularity)
        .collect();

    // 4. Max-Min Distance Selection
    let mut max_deviation = -1.0;
    let mut best_partition = None;
    let mut deviations = Vec::new();

    for (partition, fitness) in pareto_front.iter().zip(fitnesses.iter()) {
        let min_distance = random_pareto_front
            .iter()
            .map(|random_fitness| calculate_distance(fitness, random_fitness))
            .fold(INFINITY, f64::min);

        deviations.push(((*partition).clone(), *fitness, min_distance));

        if min_distance > max_deviation {
            max_deviation = min_distance;
            best_partition = Some((*partition).clone());
        }
    }

    (
        best_partition.unwrap(),
        deviations,
        fitnesses,
        random_fitnesses,
        best_fitness_history,
        avg_fitness_history,
    )
}