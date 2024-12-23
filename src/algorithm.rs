// This Source Code Form is subject to the terms of The GNU General Public License v3.0
// Copyright 2024 - Guilherme Santos. If a copy of the GPL3 was not distributed with this
// file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use petgraph::graph::Graph;
use petgraph::Undirected;
use rand::prelude::*;
use rayon::prelude::*;
use std::f64::INFINITY;
use std::fs::File;
use std::io::Write;

use crate::operators;
use crate::operators::Partition;

const OUTPUT_PATH: &str = "src/graphs/output/output.json";

/// Calculates the distance between two fitness tuples (for the final Max-Min selection).
fn calculate_distance(fitness1: &(f64, f64, f64), fitness2: &(f64, f64, f64)) -> f64 {
    let intra_diff = fitness1.1 - fitness2.1;
    let inter_diff = fitness1.2 - fitness2.2;
    (intra_diff * intra_diff + inter_diff * inter_diff).sqrt()
}

/// Generates a random graph by adding edges between random node pairs.
fn generate_random_graph(node_count: usize, edge_count: usize) -> Graph<(), (), Undirected> {
    let mut graph = Graph::<(), (), Undirected>::new_undirected();
    let mut rng = thread_rng();

    for _ in 0..node_count {
        graph.add_node(());
    }

    // Add random edges
    for _ in 0..edge_count {
        let a = rng.gen_range(0..node_count);
        let b = rng.gen_range(0..node_count);
        if a != b {
            graph.add_edge((a as u32).into(), (b as u32).into(), ());
        }
    }

    graph
}

fn debug(generation: usize, best_fitness: f64, avg_fitness: f64) {
    println!(
        "[Debug Mode]: | Generation {:.4}\t | B.Fitness: {:.4} | Avg.Fitness: {:.4}",
        generation, best_fitness, avg_fitness
    );
}
pub fn genetic_algorithm(
    graph: &Graph<(), (), Undirected>,
    generations: usize,
    population_size: usize,
    debug_mode: bool,
) -> (
    Vec<(Partition, (f64, f64, f64), f64)>,
    Vec<(f64, f64, f64)>,
    Vec<(f64, f64, f64)>,
    Vec<f64>,
    Vec<f64>,
) {
    // Precompute degrees once for efficiency
    let node_degrees: Vec<f64> = operators::compute_node_degrees(graph);
    let mut best_fitness_history = Vec::with_capacity(generations);
    let mut avg_fitness_history = Vec::with_capacity(generations);

    // 1. Generate initial population
    let mut population = operators::ga::generate_initial_population(graph, population_size);

    // 2. Evolution loop
    for generation in 0..generations {
        let fitnesses: Vec<(f64, f64, f64)> = population
            .par_iter()
            .map(|partition| {
                operators::modularity::calculate_objectives(graph, partition, &node_degrees)
            })
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
        population = operators::ga::selection(&population, &fitnesses);

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
            let mut child = operators::ga::crossover(parents[0], parents[1]);
            operators::ga::mutate(&mut child, graph);
            new_population.push(child);
        }
        population = new_population;

        if debug_mode {
            debug(
                generation, 
                best_fitness, 
                avg_fitness
            );
        }

    }

    // Final evaluation for real network
    let fitnesses: Vec<(f64, f64, f64)> = population
        .par_iter()
        .map(|partition| {
            operators::modularity::calculate_objectives(graph, partition, &node_degrees)
        })
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
    let random_degrees = operators::compute_node_degrees(&random_graph);

    let mut random_population =
        operators::ga::generate_initial_population(&random_graph, population_size);
    for _ in 0..generations {
        let fitnesses: Vec<(f64, f64, f64)> = random_population
            .par_iter()
            .map(|partition| {
                operators::modularity::calculate_objectives(
                    &random_graph,
                    partition,
                    &random_degrees,
                )
            })
            .collect();

        random_population = operators::ga::selection(&random_population, &fitnesses);

        let mut rng = thread_rng();
        let mut new_population = Vec::with_capacity(population_size);
        while new_population.len() < population_size {
            let parents: Vec<&Partition> = random_population.choose_multiple(&mut rng, 2).collect();
            if parents.len() < 2 {
                new_population.extend_from_slice(&random_population);
                break;
            }
            let mut child = operators::ga::crossover(parents[0], parents[1]);
            operators::ga::mutate(&mut child, &random_graph);
            new_population.push(child);
        }
        random_population = new_population;
    }

    // Final evaluation for random network
    let random_fitnesses: Vec<(f64, f64, f64)> = random_population
        .par_iter()
        .map(|partition| {
            operators::modularity::calculate_objectives(&random_graph, partition, &random_degrees)
        })
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

    // Save the best partition to a file
    if let Some(best_partition) = best_partition {
        let json_string = operators::partition_to_json(&best_partition);
        let mut file = File::create(OUTPUT_PATH).expect("Unable to create file");
        write!(file, "{}", json_string).expect("Unable to write data");
    }

    // Return results
    (
        deviations,
        fitnesses,
        random_fitnesses,
        best_fitness_history,
        avg_fitness_history,
    )
}