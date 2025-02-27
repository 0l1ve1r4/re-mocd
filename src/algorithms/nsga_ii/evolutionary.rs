//! algorithms/pesa_ii/evolutionary.rs
//! Implements the first phase of the algorithm (Genetic algorithm)
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2024 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::operators::*;

use rayon::prelude::*;
use rustc_hash::FxBuildHasher;
use std::collections::HashMap;
use std::sync::Arc;

use rand_chacha::ChaCha8Rng;
use crate::graph::{Graph, Partition};
use crate::utils::args::AGArgs;

pub const MAX_ARCHIVE_SIZE: usize = 100;

pub struct Solution {
    pub partition: Partition,
    pub objectives: Vec<f64>,
    pub rank: usize,
    pub crowding_distance: f64,
}

fn tournament_selection(parents: &[Solution], rng: &mut ChaCha8Rng) -> &Solution {
    let i = rng.gen_range(0..parents.len());
    let j = rng.gen_range(0..parents.len());
    let a = &parents[i];
    let b = &parents[j];
    if a.rank < b.rank || (a.rank == b.rank && a.crowding_distance > b.crowding_distance) {
        a
    } else {
        b
    }
}

fn generate_offspring(
    parent_solutions: &[Solution],
    args: &AGArgs,
    graph: &Graph,
    safe_rng: Arc<SafeRng>,
) -> Vec<Partition> {
    let chunk_size = (args.pop_size / rayon::current_num_threads()).max(1);
    let chunks: Vec<_> = (0..args.pop_size)
        .collect::<Vec<_>>()
        .chunks(chunk_size)
        .map(|c| c.len())
        .collect();

    let results: Vec<Partition> = chunks
        .par_iter()
        .flat_map(|&chunk_size| {
            let mut local_children = Vec::with_capacity(chunk_size);
            let mut local_rng = safe_rng.get_rng();
            for _ in 0..chunk_size {
                let parent1 = tournament_selection(parent_solutions, &mut local_rng);
                let parent2 = tournament_selection(parent_solutions, &mut local_rng);
                let mut child = crossover(&parent1.partition, &parent2.partition, args.cross_rate);
                mutation(&mut child, graph, args.mut_rate);
                local_children.push(child);
            }
            local_children
        })
        .collect();
    results
}

fn non_dominated_sort(solutions: &mut [Solution]) -> Vec<Vec<Solution>> {
    let mut fronts = Vec::new();
    let mut current_front = Vec::new();
    let mut solution_info = vec![(0, Vec::new(), 0); solutions.len()];

    for i in 0..solutions.len() {
        let mut dominated = Vec::new();
        let mut count = 0;
        for j in 0..solutions.len() {
            if i == j { continue; }
            if solutions[i].dominates(&solutions[j]) {
                dominated.push(j);
            } else if solutions[j].dominates(&solutions[i]) {
                count += 1;
            }
        }
        solution_info[i] = (count, dominated, 0);
        if count == 0 {
            current_front.push(i);
        }
    }

    let mut rank = 0;
    while !current_front.is_empty() {
        let mut next_front = Vec::new();
        for &i in &current_front {
            solution_info[i].2 = rank;
            for &j in &solution_info[i].1 {
                solution_info[j].0 -= 1;
                if solution_info[j].0 == 0 {
                    next_front.push(j);
                }
            }
        }
        let mut front = current_front.iter().map(|&i| solutions[i].clone()).collect();
        fronts.push(front);
        current_front = next_front;
        rank += 1;
    }

    for (i, info) in solution_info.iter().enumerate() {
        solutions[i].rank = info.2;
    }

    fronts
}

fn calculate_crowding_distance(front: &mut [Solution]) {
    let len = front.len();
    if len == 0 { return; }

    let num_objs = front[0].objectives.len();
    for s in front.iter_mut() {
        s.crowding_distance = 0.0;
    }

    for obj_idx in 0..num_objs {
        front.sort_by(|a, b| a.objectives[obj_idx].partial_cmp(&b.objectives[obj_idx]).unwrap());
        front[0].crowding_distance = f64::INFINITY;
        front[len - 1].crowding_distance = f64::INFINITY;

        let min = front[0].objectives[obj_idx];
        let max = front[len - 1].objectives[obj_idx];
        let range = max - min;
        if range <= f64::EPSILON { continue; }

        for i in 1..len-1 {
            front[i].crowding_distance += (front[i+1].objectives[obj_idx] - front[i-1].objectives[obj_idx]) / range;
        }
    }
}

fn select_next_population(fronts: Vec<Vec<Solution>>, pop_size: usize) -> Vec<Solution> {
    let mut selected = Vec::with_capacity(pop_size);
    let mut remaining = pop_size;

    for mut front in fronts {
        if front.is_empty() { continue; }
        if selected.len() + front.len() <= pop_size {
            selected.append(&mut front);
        } else {
            front.sort_by(|a, b| b.crowding_distance.partial_cmp(&a.crowding_distance).unwrap());
            let take = remaining.min(front.len());
            selected.extend(front.into_iter().take(take));
            break;
        }
        remaining = pop_size - selected.len();
        if remaining == 0 { break; }
    }

    selected
}

pub fn evolutionary_phase(
    graph: &Graph,
    args: &AGArgs,
    degrees: &HashMap<i32, usize, FxBuildHasher>,
) -> (Vec<Solution>, Vec<f64>) {
    // Initial population and evaluation
    let mut population = generate_population(graph, args.pop_size);
    let mut current_solutions = evaluate_population(&population, graph, degrees);
    let safe_rng = Arc::new(SafeRng::new());
    let mut best_fitness_history = Vec::new();

    for generation in 0..args.num_gens {
        // Generate offspring
        let offspring_partitions = generate_offspring(t_solutions, args, graph, Arc::clone(&safe_rng));
        let offspring_solutions = evaluate_population(&offspring_partitions, graph, degrees);

        // Combine and sort
        let mut combined = [current_solutions.clone(), offspring_solutions].concat();
        let mut fronts = non_dominated_sort(&mut combined);
        for front in &mut fronts {
            calculate_crowding_distance(front);
        }

        // Select next generation
        current_solutions = select_next_population(fronts, args.pop_size);
        population = current_solutions.iter().map(|s| s.partition.clone()).collect();

        // Track best fitness
        let best_fitness = current_solutions.iter()
            .map(|s| 1.0 - s.objectives[0] - s.objectives[1])
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);
        best_fitness_history.push(best_fitness);

        // Convergence check (existing code)
        // ...
    }

    (current_solutions, best_fitness_history)
}