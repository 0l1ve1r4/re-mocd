// This Source Code Form is subject to the terms of The GNU General Public License v3.0
// Copyright 2024 - Guilherme Santos. If a copy of the GPL3 was not distributed with this
// file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use petgraph::graph::Graph;
use petgraph::Undirected;
use rand::prelude::*;
use rayon::prelude::*;
use std::fs::File;
use std::io::Write;

use crate::operators;
use crate::operators::Partition;

const OUTPUT_PATH: &str = "src/graphs/output/output.json";

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
        let avg_fitness = modularity_values.iter().sum::<f64>() / fitnesses.len() as f64;

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

    // Select the best partition based on highest modularity
    let (best_partition, best_fitness) = population
        .iter()
        .zip(fitnesses.iter())
        .max_by(|a, b| a.1 .0.partial_cmp(&b.1 .0).unwrap())
        .map(|(p, f)| ((*p).clone(), f.0))
        .expect("Population is empty");

    // Save the best partition to a file
    let json_string = operators::partition_to_json(&best_partition);
    let mut file = File::create(OUTPUT_PATH).expect("Unable to create file");
    write!(file, "{}", json_string).expect("Unable to write data");

    (
        vec![(best_partition, fitnesses.into_iter().next().unwrap_or((0.0, 0.0, 0.0)), best_fitness)],
        best_fitness_history,
        avg_fitness_history,
    )
}