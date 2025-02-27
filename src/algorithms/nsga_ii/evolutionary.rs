//! algorithms/pesa_ii/evolutionary.rs
//! Implements the first phase of the algorithm (Genetic algorithm)
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2024 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::operators::*;
use rand::seq::SliceRandom;
use rustc_hash::FxBuildHasher;
use std::collections::HashMap;

use rand::thread_rng;

use crate::graph::{Graph, Partition};
use crate::utils::args::AGArgs;

pub const MAX_ARCHIVE_SIZE: usize = 100;

// Update Solution struct definition
#[derive(Clone, PartialEq)]
pub struct Solution {
    pub partition: Partition,
    pub objectives: Vec<f64>,
    pub crowding_distance: f64,
}

impl Solution {
    // Add default implementation for new field
    fn new(partition: Partition, objectives: Vec<f64>) -> Self {
        Solution {
            partition,
            objectives,
            crowding_distance: 0.0,
        }
    }

    pub fn dominates(&self, other: &Solution) -> bool {
        let mut at_least_one_better = false;
        
        for i in 0..self.objectives.len() {
            if self.objectives[i] > other.objectives[i] {
                return false;
            }
            if self.objectives[i] < other.objectives[i] {
                at_least_one_better = true;
            }
        }
        
        at_least_one_better
    }

}


pub fn evolutionary_phase(
    graph: &Graph,
    args: &AGArgs,
    degrees: &HashMap<i32, usize, FxBuildHasher>,
) -> (Vec<Solution>, Vec<f64>) {
    // Validate graph
    if graph.nodes.is_empty() || graph.edges.is_empty() {
        println!("[evolutionary_phase]: Empty graph detected");
        return (Vec::new(), Vec::new());
    }

    // Debug print graph information
    if args.debug {
        println!(
            "[evolutionary_phase]: Starting with graph - nodes: {}, edges: {}",
            graph.nodes.len(),
            graph.edges.len()
        );
    }

    // Initialize archive for returning Pareto optimal solutions
    let mut archive: Vec<Solution> = Vec::new();
    
    // Generate and validate initial population
    let mut population = generate_population(graph, args.pop_size);
    if population.is_empty() {
        println!("[evolutionary_phase]: Failed to generate initial population");
        return (Vec::new(), Vec::new());
    }

    if args.debug {
        println!(
            "[evolutionary_phase]: Initial population size: {}",
            population.len()
        );
    }

    let mut solutions: Vec<Solution> = Vec::with_capacity(population.len());
    let mut best_fitness_history: Vec<f64> = Vec::with_capacity(args.num_gens);
    let mut max_local: ConvergenceCriteria = ConvergenceCriteria::default();

    // Evaluate initial population
    for partition in &population {
        let metrics = get_fitness(graph, partition, degrees, true);
        solutions.push(Solution {
            partition: partition.clone(),
            objectives: vec![metrics.inter, metrics.intra],
            crowding_distance: 0.0,
        });
    }

    // Main evolutionary loop
    for generation in 0..args.num_gens {
        // Validate population size before processing
        if solutions.is_empty() {
            println!("[evolutionary_phase]: Empty solution set");
            break;
        }

        // NSGA-II: Perform non-dominated sorting
        let fronts = fast_non_dominated_sort(&solutions);
        
        if fronts.is_empty() {
            println!("[evolutionary_phase]: No valid fronts generated");
            break;
        }

        // Calculate crowding distance for each front
        let mut ranked_solutions: Vec<Solution> = Vec::new();
        for front in &fronts {
            let mut front_solutions: Vec<Solution> = front.iter()
                .map(|&idx| solutions[idx].clone())
                .collect();
            
            // Calculate crowding distance for this front
            calculate_crowding_distance(&mut front_solutions);
            
            // Add solutions from this front to the ranked list
            ranked_solutions.extend(front_solutions);
        }

        // Select the best solutions based on rank and crowding distance
        ranked_solutions.sort_by(|a, b| {
            let a_rank = fronts.iter().position(|front| front.contains(&solutions.iter().position(|s| s == a).unwrap())).unwrap();
            let b_rank = fronts.iter().position(|front| front.contains(&solutions.iter().position(|s| s == b).unwrap())).unwrap();
            
            match a_rank.cmp(&b_rank) {
                std::cmp::Ordering::Equal => b.crowding_distance.partial_cmp(&a.crowding_distance).unwrap(),
                other => other,
            }
        });

        // Truncate to population size
        if ranked_solutions.len() > args.pop_size {
            ranked_solutions.truncate(args.pop_size);
        }

        // Update archive with first Pareto front
        if !fronts.is_empty() {
            for &idx in &fronts[0] {
                let solution = solutions[idx].clone();
                
                // Update the archive using non-dominated solutions
                if !archive.iter().any(|archived| archived.dominates(&solution)) {
                    archive.retain(|archived| !solution.dominates(archived));
                    archive.push(solution);
                }
            }
        }

        // Limit archive size if needed
        if archive.len() > MAX_ARCHIVE_SIZE {
            // Sort by crowding distance and keep the most diverse solutions
            calculate_crowding_distance(&mut archive);
            archive.sort_by(|a, b| b.crowding_distance.partial_cmp(&a.crowding_distance).unwrap());
            archive.truncate(MAX_ARCHIVE_SIZE);
        }

        // Safely compute best fitness
        let best_fitness = archive
            .iter()
            .map(|s| 1.0 - s.objectives[0] - s.objectives[1])
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(f64::NEG_INFINITY);

        best_fitness_history.push(best_fitness);

        // Generate new population through selection, crossover, and mutation
        let mut offspring = Vec::with_capacity(args.pop_size);
        
        while offspring.len() < args.pop_size {
            // Selection: Tournament selection based on rank and crowding distance
            let parent1 = tournament_selection(&ranked_solutions, 2);
            let parent2 = tournament_selection(&ranked_solutions, 2);
            
            let mut child = crossover(&parent1.partition, &parent2.partition, args.cross_rate);
            mutation(&mut child, graph, args.mut_rate);
            offspring.push(child);                        
        }

        // Replace current population with offspring
        population = offspring;
        
        // Evaluate new population
        solutions.clear();
        for partition in &population {
            let metrics = get_fitness(graph, partition, degrees, true);
            solutions.push(Solution {
                partition: partition.clone(),
                objectives: vec![metrics.inter, metrics.intra],
                crowding_distance: 0.0, // Reset for next iteration
            });
        }

        // Early stopping
        if max_local.has_converged(best_fitness) {
            if args.debug {
                println!("[evolutionary_phase]: Converged!");
            }
            break;
        }

        if args.debug {
            println!(
                "\x1b[1A\x1b[2K[evolutionary_phase]: gen: {} | bf: {:.4} | pop/arch: {}/{} | bA: {:.4} |",
                generation,
                best_fitness,
                population.len(),
                archive.len(),
                max_local.get_best_fitness(),
            );
        }
    }

    // Return empty results if archive is empty
    if archive.is_empty() {
        return (Vec::new(), best_fitness_history);
    }

    (archive, best_fitness_history)
}

// Helper functions for NSGA-II

// Fast Non-dominated Sort to divide solutions into Pareto fronts
fn fast_non_dominated_sort(solutions: &[Solution]) -> Vec<Vec<usize>> {
    let n = solutions.len();
    let mut fronts: Vec<Vec<usize>> = Vec::new();
    let mut domination_count: Vec<usize> = vec![0; n];
    let mut dominated_by: Vec<Vec<usize>> = vec![Vec::new(); n];
    
    // First front
    let mut current_front: Vec<usize> = Vec::new();
    
    // Calculate domination relationships
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            
            if solutions[i].dominates(&solutions[j]) {
                dominated_by[i].push(j);
            } else if solutions[j].dominates(&solutions[i]) {
                domination_count[i] += 1;
            }
        }
        
        // If not dominated by anyone, add to first front
        if domination_count[i] == 0 {
            current_front.push(i);
        }
    }
    
    // Identify fronts
    fronts.push(current_front.clone());
    
    let mut front_index = 0;
    while !fronts[front_index].is_empty() {
        let mut next_front: Vec<usize> = Vec::new();
        
        for &i in &fronts[front_index] {
            for &j in &dominated_by[i] {
                domination_count[j] -= 1;
                if domination_count[j] == 0 {
                    next_front.push(j);
                }
            }
        }
        
        front_index += 1;
        if !next_front.is_empty() {
            fronts.push(next_front);
        }
    }
    
    fronts
}

// Calculate crowding distance for a set of solutions
fn calculate_crowding_distance(solutions: &mut Vec<Solution>) {
    let n = solutions.len();
    if n <= 2 {
        // For very small sets, assign maximum crowding distance
        for solution in solutions.iter_mut() {
            solution.crowding_distance = f64::INFINITY;
        }
        return;
    }
    
    // Initialize all crowding distances to 0
    for solution in solutions.iter_mut() {
        solution.crowding_distance = 0.0;
    }
    
    // For each objective
    let num_objectives = solutions[0].objectives.len();
    for m in 0..num_objectives {
        // Sort by current objective
        solutions.sort_by(|a, b| a.objectives[m].partial_cmp(&b.objectives[m]).unwrap());
        
        // Set boundary points to infinity
        solutions[0].crowding_distance = f64::INFINITY;
        solutions[n - 1].crowding_distance = f64::INFINITY;
        
        // Calculate crowding distance for intermediate points
        let obj_range = solutions[n-1].objectives[m] - solutions[0].objectives[m];
        if obj_range > 1e-10 { // Avoid division by zero
            for i in 1..n-1 {
                solutions[i].crowding_distance += (solutions[i+1].objectives[m] - solutions[i-1].objectives[m]) / obj_range;
            }
        }
    }
}

// Tournament selection based on rank and crowding distance
fn tournament_selection(ranked_solutions: &[Solution], tournament_size: usize) -> &Solution {
    let n = ranked_solutions.len();
    let mut best_idx = rand::random::<usize>() % n;
    
    for _ in 1..tournament_size {
        let idx = rand::random::<usize>() % n;
        let current = &ranked_solutions[idx];
        let best = &ranked_solutions[best_idx];
        
        // Compare based on rank, then crowding distance
        let current_rank = current.objectives[0]; // assuming first objective is rank
        let best_rank = best.objectives[0];
        
        if current_rank < best_rank || (current_rank == best_rank && current.crowding_distance > best.crowding_distance) {
            best_idx = idx;
        }
    }
    
    &ranked_solutions[best_idx]
}

/// Generates multiple random networks and combines their solutions
fn generate_random_networks(original: &Graph, num_networks: usize) -> Vec<Graph> {
    (0..num_networks)
        .map(|_| {
            let mut random_graph = Graph {
                nodes: original.nodes.clone(),
                ..Default::default()
            };

            let node_vec: Vec<_> = random_graph.nodes.iter().cloned().collect();
            let num_nodes = node_vec.len();
            let num_edges = original.edges.len();
            let mut rng = thread_rng();
            let mut possible_pairs = Vec::with_capacity(num_nodes * (num_nodes - 1) / 2);

            for i in 0..num_nodes {
                for j in (i + 1)..num_nodes {
                    possible_pairs.push((node_vec[i], node_vec[j]));
                }
            }

            possible_pairs.shuffle(&mut rng);
            let selected_edges = possible_pairs
                .into_iter()
                .take(num_edges)
                .collect::<Vec<_>>();

            for (src, dst) in &selected_edges {
                random_graph.edges.push((*src, *dst));
            }

            for node in &random_graph.nodes {
                random_graph.adjacency_list.insert(*node, Vec::new());
            }

            for (src, dst) in &random_graph.edges {
                random_graph.adjacency_list.get_mut(src).unwrap().push(*dst);
                random_graph.adjacency_list.get_mut(dst).unwrap().push(*src);
            }

            random_graph
        })
        .collect()
}
