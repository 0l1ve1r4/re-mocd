use rayon::prelude::*;
use std::collections::HashMap;
use rustc_hash::FxBuildHasher;

use crate::operators;
use crate::args::AGArgs;
use crate::graph::{Graph, Partition};

#[derive(Clone, Debug)]
struct Solution {
    partition: Partition,
    objectives: Vec<f64>,
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
struct HyperBox {
    solutions: Vec<Solution>,
    coordinates: Vec<usize>,
}

impl Solution {
    fn dominates(&self, other: &Solution) -> bool {
        let mut has_better = false;
        for (self_obj, other_obj) in self.objectives.iter().zip(other.objectives.iter()) {
            if self_obj < other_obj {
                return false;
            }
            if self_obj > other_obj {
                has_better = true;
            }
        }
        has_better
    }
}

const GRID_DIVISIONS: usize = 8;

pub fn genetic_algorithm(
    graph: &Graph,
    args: AGArgs,
) -> (Partition, Vec<f64>, f64) {
    let mut rng = rand::thread_rng();
    let mut archive: Vec<Solution> = Vec::with_capacity(args.pop_size);
    let mut population: Vec<std::collections::BTreeMap<i32, i32>> = operators::
                                                                    generate_initial_population(graph,
                                                                        args.pop_size);
    let mut best_fitness_history: Vec<f64> = Vec::with_capacity(args.num_gens);
    let degrees: HashMap<i32, usize, FxBuildHasher> = graph.precompute_degress();
        
    for generation in 0..args.num_gens {
        let solutions: Vec<Solution> = population
        .par_chunks(population.len() / rayon::current_num_threads())
        .flat_map(|chunk| {
            chunk.iter().map(|partition| {
                let metrics = operators::calculate_objectives(graph, partition, &degrees, true);
                Solution {
                    partition: partition.clone(),
                    objectives: vec![metrics.modularity, metrics.inter, metrics.intra],
                }
            }).collect::<Vec<_>>()
        })
        .collect();

        // Update archive with non-dominated solutions
        // TODO: Optimize archive
        for solution in solutions {
        if !archive.iter().any(|archived: &Solution| archived.dominates(&solution)) {
            // Remove solutions from archive that are dominated by the new solution
            archive.retain(|archived: &Solution| !solution.dominates(archived));
            archive.push(solution);
        }

        }

        let hyperboxes: Vec<HyperBox> = create_hypergrid_parallel(&archive, GRID_DIVISIONS);
        
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
            let parent1: &Solution;
            let parent2: &Solution;
            
            parent1 = select_from_hypergrid_parallel(&hyperboxes,&mut rng);
            parent2 = select_from_hypergrid_parallel(&hyperboxes,&mut rng);
            
            let mut child: std::collections::BTreeMap<i32, i32> = operators::crossover(&parent1.partition, &parent2.partition);
            operators::mutate(&mut child, graph);
            new_population.push(child);
        }
        
        population = new_population;

        // Early stopping condition
        if operators::last_x_same(&best_fitness_history) {
            if args.debug {
                println!("[Optimization]: Max Local, breaking...");
            }
            break;
        }

        if args.debug {
            println!(
                "Generation: {} \t | Best Fitness: {} | Archive size: {}",
                generation, best_fitness, archive.len()
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

fn create_hypergrid_parallel(solutions: &[Solution], divisions: usize) -> Vec<HyperBox> {
    if solutions.is_empty() {
        return Vec::new();
    }

    // Calculate min/max values in parallel
    let obj_len = solutions[0].objectives.len();
    let (min_values, max_values) = rayon::join(
        || {
            (0..obj_len).into_par_iter().map(|i| {
                solutions.par_iter()
                    .map(|s| s.objectives[i])
                    .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap()
            }).collect::<Vec<_>>()
        },
        || {
            (0..obj_len).into_par_iter().map(|i| {
                solutions.par_iter()
                    .map(|s| s.objectives[i])
                    .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap()
            }).collect::<Vec<_>>()
        }
    );

    // Use a concurrent HashMap for grouping solutions
    use dashmap::DashMap;
    let hyperbox_map = DashMap::new();

    solutions.par_iter().for_each(|solution| {
        let coordinates = solution.objectives.iter()
            .enumerate()
            .map(|(i, &obj)| {
                let normalized = if (max_values[i] - min_values[i]).abs() < f64::EPSILON {
                    0.0
                } else {
                    (obj - min_values[i]) / (max_values[i] - min_values[i])
                };
                (normalized * divisions as f64).min((divisions - 1) as f64) as usize
            })
            .collect::<Vec<_>>();

        hyperbox_map.entry(coordinates.clone())
            .and_modify(|solutions: &mut Vec<Solution>| solutions.push(solution.clone()))
            .or_insert_with(|| vec![solution.clone()]);
    });

    // Convert DashMap to Vec<HyperBox>
    hyperbox_map.into_iter()
        .map(|(coordinates, solutions)| HyperBox { solutions, coordinates })
        .collect()
}

/// Parallel version of select_from_hypergrid
fn select_from_hypergrid_parallel<'a>(hyperboxes: &'a [HyperBox], rng: &mut impl rand::Rng) -> &'a Solution {
    // Compute total weight in parallel
    let total_weight: f64 = hyperboxes
        .par_iter()
        .map(|hb| 1.0 / (hb.solutions.len() as f64))
        .sum();
    
    let mut random_value = rng.gen::<f64>() * total_weight;
    
    // Selection remains sequential to handle the cumulative weights
    for hyperbox in hyperboxes {
        let weight = 1.0 / (hyperbox.solutions.len() as f64);
        if random_value <= weight {
            // Randomly select a solution from the chosen hyperbox
            return &hyperbox.solutions[rng.gen_range(0..hyperbox.solutions.len())];
        }
        random_value -= weight;
    }
    
    // Fallback to last hyperbox
    let last_box = hyperboxes.last().unwrap();
    &last_box.solutions[rng.gen_range(0..last_box.solutions.len())]
}