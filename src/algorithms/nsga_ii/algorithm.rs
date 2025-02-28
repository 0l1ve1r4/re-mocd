use crate::graph::{Graph, Partition};
use crate::utils::args::AGArgs;
use super::evolutionary::evolutionary_phase;

use rustc_hash::FxBuildHasher;
use std::collections::HashMap;
use std::cmp::Ordering;

pub fn nsga_ii(graph: &Graph, args: AGArgs) -> (Partition, Vec<f64>, f64) {
    if args.debug >= 2 {
        println!("[algorithms/nsga_ii/algorithm.rs]: Starting..")
    }

    let degrees: HashMap<i32, usize, FxBuildHasher> = graph.precompute_degress();
    let (final_solutions, best_fitness_history) = evolutionary_phase(graph, &args, &degrees);

    let best_solution = final_solutions
        .into_iter()
        .max_by(|a, b| {
            a.objectives[0]
                .partial_cmp(&b.objectives[0])
                .unwrap_or(Ordering::Equal)
        })
        .expect("No solution found");

    (
        best_solution.partition.clone(),
        best_fitness_history,
        best_solution.objectives[0],
    )
}