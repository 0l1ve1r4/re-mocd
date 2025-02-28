//! algorithms/nsga_ii/evolutionary.rs
//! Implements the Non-Dominated Sorting Genetic Algorithm 2 (NSGA-II) to community detection
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
    #[allow(dead_code)]
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
            // For maximization: check if self is LOWER than other
            if self.objectives[i] < other.objectives[i] {
                return false;
            }
            // For maximization: check if self is HIGHER than other
            if self.objectives[i] > other.objectives[i] {
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
            objectives: vec![metrics.modularity, metrics.inter, metrics.intra],
            crowding_distance: 0.0,
        });
    }

    if args.debug {
        println!("[evolutionary_phase]: Evaluated initial population. Solution count: {}", solutions.len());
        println!("[evolutionary_phase]: First solution objectives: {:?}", 
                 solutions.first().map(|s| s.objectives.clone()).unwrap_or_default());
    }

    // Main evolutionary loop
    for generation in 0..args.num_gens {
        // Validate population size before processing
        if solutions.is_empty() {
            println!("[evolutionary_phase]: Empty solution set");
            break;
        }

        if args.debug {
            println!("[evolutionary_phase]: Generation {}, solution count: {}", generation, solutions.len());
        }

        // NSGA-II: Perform non-dominated sorting
        let fronts = fast_non_dominated_sort(&solutions);
        
        if fronts.is_empty() {
            println!("[evolutionary_phase]: No valid fronts generated");
            break;
        }

        if args.debug {
            println!("[evolutionary_phase]: Non-dominated sorting created {} fronts", fronts.len());
            for (i, front) in fronts.iter().enumerate() {
                println!("[evolutionary_phase]: Front {} has {} solutions", i, front.len());
            }
        }

        // Calculate crowding distance for each front
        let mut ranked_solutions: Vec<Solution> = Vec::new();
        for (front_idx, front) in fronts.iter().enumerate() {
            if args.debug {
                println!("[evolutionary_phase]: Processing front {} with {} solutions", front_idx, front.len());
            }
            
            let mut front_solutions: Vec<Solution> = Vec::new();
            for &idx in front {
                if idx < solutions.len() {
                    front_solutions.push(solutions[idx].clone());
                } else {
                    println!("[evolutionary_phase]: ERROR: Invalid solution index {} (max: {})", 
                             idx, solutions.len() - 1);
                    // Skip this invalid index
                    continue;
                }
            }
            
            if args.debug {
                println!("[evolutionary_phase]: Calculating crowding distance for front {}, {} solutions", 
                         front_idx, front_solutions.len());
            }
            
            // Calculate crowding distance for this front
            calculate_crowding_distance(&mut front_solutions, args);
            
            // Add solutions from this front to the ranked list
            ranked_solutions.extend(front_solutions);
        }

        if args.debug {
            println!("[evolutionary_phase]: Ranked solutions: {}", ranked_solutions.len());
        }

        // Select the best solutions based on rank and crowding distance
        if ranked_solutions.len() > 1 {
            ranked_solutions.sort_by(|a, b| {
                let a_rank = find_solution_rank(&solutions, a, &fronts);
                let b_rank = find_solution_rank(&solutions, b, &fronts);
                
                if args.debug && (a_rank.is_none() || b_rank.is_none()) {
                    println!("[evolutionary_phase]: WARNING: Could not find rank for one or more solutions");
                }
                
                let a_rank = a_rank.unwrap_or(usize::MAX);
                let b_rank = b_rank.unwrap_or(usize::MAX);
                
                match a_rank.cmp(&b_rank) {
                    std::cmp::Ordering::Equal => b.crowding_distance.partial_cmp(&a.crowding_distance)
                        .unwrap_or(std::cmp::Ordering::Equal),
                    other => other,
                }
            });
        }

        // Truncate to population size
        if ranked_solutions.len() > args.pop_size {
            if args.debug {
                println!("[evolutionary_phase]: Truncating ranked solutions from {} to {}", 
                         ranked_solutions.len(), args.pop_size);
            }
            ranked_solutions.truncate(args.pop_size);
        }

        // Update archive with first Pareto front
        if !fronts.is_empty() && !fronts[0].is_empty() {
            for &idx in &fronts[0] {
                if idx < solutions.len() {
                    let solution = solutions[idx].clone();
                    
                    // Update the archive using non-dominated solutions
                    if !archive.iter().any(|archived| archived.dominates(&solution)) {
                        archive.retain(|archived| !solution.dominates(archived));
                        archive.push(solution);
                    }
                } else {
                    println!("[evolutionary_phase]: ERROR: Invalid solution index {} when updating archive", idx);
                }
            }
            
            if args.debug {
                println!("[evolutionary_phase]: Updated archive, now contains {} solutions", archive.len());
            }
        }

        // Limit archive size if needed
        if archive.len() > MAX_ARCHIVE_SIZE {
            // Sort by crowding distance and keep the most diverse solutions
            if args.debug {
                println!("[evolutionary_phase]: Archive exceeds max size ({}), calculating crowding distances", 
                         MAX_ARCHIVE_SIZE);
            }
            calculate_crowding_distance(&mut archive, args);
            archive.sort_by(|a, b| b.crowding_distance.partial_cmp(&a.crowding_distance)
                .unwrap_or(std::cmp::Ordering::Equal));
            archive.truncate(MAX_ARCHIVE_SIZE);
        }

        // Safely compute best fitness - using modularity as the primary objective for maximization
        let best_fitness = if !archive.is_empty() {
            archive
                .iter()
                .map(|s| s.objectives[0]) // Use modularity as the primary fitness measure
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(f64::NEG_INFINITY)
        } else {
            f64::NEG_INFINITY
        };

        best_fitness_history.push(best_fitness);

        // Generate new population through selection, crossover, and mutation
        let mut offspring = Vec::with_capacity(args.pop_size);
        
        if args.debug {
            println!("[evolutionary_phase]: Generating offspring population");
        }
        
        // Safety check for tournament selection
        if ranked_solutions.is_empty() {
            println!("[evolutionary_phase]: ERROR: Cannot perform selection with empty population");
            break;
        }
        
        while offspring.len() < args.pop_size {
            // Selection: Tournament selection based on rank and crowding distance
            let parent1 = safe_tournament_selection(&ranked_solutions, 2);
            let parent2 = safe_tournament_selection(&ranked_solutions, 2);
            
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
                objectives: vec![metrics.modularity, metrics.inter, metrics.intra],
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
        if args.debug {
            println!("[evolutionary_phase]: WARNING: Empty archive at end of evolution");
        }
        return (Vec::new(), best_fitness_history);
    }

    (archive, best_fitness_history)
}

// Helper function to find a solution's rank
fn find_solution_rank(solutions: &[Solution], solution: &Solution, fronts: &[Vec<usize>]) -> Option<usize> {
    for (front_idx, front) in fronts.iter().enumerate() {
        for &sol_idx in front {
            if sol_idx < solutions.len() && solutions[sol_idx] == *solution {
                return Some(front_idx);
            }
        }
    }
    None
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
        } else {
            break; // Exit the loop if no more fronts can be formed
        }
    }
    
    fronts
}

// Calculate crowding distance for a set of solutions
fn calculate_crowding_distance(solutions: &mut Vec<Solution>, args: &AGArgs) {
    let n = solutions.len();
    
    if args.debug {
        println!("[calculate_crowding_distance]: Processing {} solutions", n);
    }
    
    if n <= 2 {
        // For very small sets, assign maximum crowding distance
        for solution in solutions.iter_mut() {
            solution.crowding_distance = f64::INFINITY;
        }
        
        if args.debug {
            println!("[calculate_crowding_distance]: Small solution set ({}), assigned infinite crowding distance", n);
        }
        
        return;
    }
    
    // Initialize all crowding distances to 0
    for solution in solutions.iter_mut() {
        solution.crowding_distance = 0.0;
    }
    
    // For each objective
    let num_objectives = if let Some(first) = solutions.first() {
        first.objectives.len()
    } else {
        if args.debug {
            println!("[calculate_crowding_distance]: WARNING: Empty solution vector");
        }
        return;
    };
    
    if args.debug {
        println!("[calculate_crowding_distance]: Processing {} objectives", num_objectives);
    }
    
    for m in 0..num_objectives {
        if args.debug {
            println!("[calculate_crowding_distance]: Processing objective {}", m);
        }
        
        // Sort by current objective
        solutions.sort_by(|a, b| {
            if m < a.objectives.len() && m < b.objectives.len() {
                a.objectives[m].partial_cmp(&b.objectives[m]).unwrap_or(std::cmp::Ordering::Equal)
            } else {
                if args.debug {
                    println!("[calculate_crowding_distance]: ERROR: Objective index {} out of bounds (sizes: {}, {})",
                             m, a.objectives.len(), b.objectives.len());
                }
                std::cmp::Ordering::Equal
            }
        });
        
        // Set boundary points to infinity
        if !solutions.is_empty() {
            solutions[0].crowding_distance = f64::INFINITY;
            
            if n > 1 {
                solutions[n - 1].crowding_distance = f64::INFINITY;
            }
        }
        
        // Calculate crowding distance for intermediate points
        if n > 2 && m < solutions[0].objectives.len() && m < solutions[n-1].objectives.len() {
            let obj_range = solutions[n-1].objectives[m] - solutions[0].objectives[m];
            
            if args.debug {
                println!("[calculate_crowding_distance]: Objective {} range: {}", m, obj_range);
            }
            
            if obj_range > 1e-10 { // Avoid division by zero
                for i in 1..n-1 {
                    if m < solutions[i-1].objectives.len() && 
                       m < solutions[i].objectives.len() && 
                       m < solutions[i+1].objectives.len() {
                        
                        solutions[i].crowding_distance += 
                            (solutions[i+1].objectives[m] - solutions[i-1].objectives[m]) / obj_range;
                        
                    } else {
                        if args.debug {
                            println!("[calculate_crowding_distance]: ERROR: Index out of bounds at solution {}", i);
                        }
                    }
                }
            } else if args.debug {
                println!("[calculate_crowding_distance]: Objective {} has zero range, skipping", m);
            }
        }
    }
    
    if args.debug {
        println!("[calculate_crowding_distance]: Crowding distance calculation complete");
    }
}

// A safer tournament selection implementation
fn safe_tournament_selection(ranked_solutions: &[Solution], tournament_size: usize) -> &Solution {
    let n = ranked_solutions.len();
    
    // Handle edge cases
    if n == 0 {
        panic!("[safe_tournament_selection]: Empty solution set");
    }
    
    if n == 1 || tournament_size <= 1 {
        return &ranked_solutions[0];
    }
    
    // Generate random indices for the tournament
    let mut rng = thread_rng();
    let mut indices: Vec<usize> = (0..n).collect();
    indices.shuffle(&mut rng);
    
    // Take at most tournament_size or n, whichever is smaller
    let actual_tournament_size = tournament_size.min(n);
    let tournament_indices = &indices[0..actual_tournament_size];
    
    // Find the best solution in the tournament
    let mut best_idx = tournament_indices[0];
    for &idx in &tournament_indices[1..] {
        // In NSGA-II, solutions are already ranked
        // If the crowding distance of the current solution is greater, select it
        if ranked_solutions[idx].crowding_distance > ranked_solutions[best_idx].crowding_distance {
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