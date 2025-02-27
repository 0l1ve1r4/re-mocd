use crate::graph::{Graph, Partition};
use crate::utils::args::AGArgs;
use super::evolutionary::evolutionary_phase;

use rustc_hash::FxBuildHasher;
use std::collections::HashMap;
use std::cmp::Ordering;

pub fn nsga_ii(graph: &Graph, args: AGArgs) -> (Partition, Vec<f64>, f64) {
    // Precompute degrees (or any other helper information)
    let degrees: HashMap<i32, usize, FxBuildHasher> = graph.precompute_degress();

    // Phase 1: Run the evolutionary algorithm to get the Pareto frontier and fitness history.
    let (final_solutions, best_fitness_history) = evolutionary_phase(graph, &args, &degrees);

    // Select the best solution.
    // Note: Since NSGA-II is multiobjective, there is no inherent single "best" solution.
    // Here, we choose the one with the lowest first objective value.
    let best_solution = final_solutions
        .into_iter()
        .min_by(|a, b| {
            a.objectives[0]
                .partial_cmp(&b.objectives[0])
                .unwrap_or(Ordering::Equal)
        })
        .expect("No solution found");

    (
        best_solution.partition.clone(), // Return the partition (e.g., a Vec of size 64)
        best_fitness_history,              // Return the fitness history vector
        best_solution.objectives[0],       // Return the best solution's first objective value
    )
}