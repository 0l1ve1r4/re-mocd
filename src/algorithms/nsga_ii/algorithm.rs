const NUM_RANDOM_NETWORKS: usize = 1;

use crate::graph::{self, Graph, Partition, CommunityId, NodeId};
use crate::utils::args::AGArgs;
use super::evolutionary::evolutionary_phase;

use rustc_hash::FxBuildHasher;
use std::collections::{HashMap, BTreeMap};
use rand::thread_rng;
use rand::seq::SliceRandom as _;

#[derive(Default)]
struct GraphLevel {
    graph: Graph,
    partition: Option<Partition>,  // Alterado para Option
    mapping: BTreeMap<CommunityId, Vec<NodeId>>,
}

/// Main run function that creates both the real and random fronts, then
/// selects the best solution via the chosen criterion.
pub fn nsga_ii(graph: &Graph, args: AGArgs) -> (Partition, Vec<f64>, f64) {
    let degrees: HashMap<i32, usize, FxBuildHasher> = graph.precompute_degress();

    // Phase 1: Evolutionary algorithm returns the Pareto frontier for the real network
    let (final_solutions, best_fitness_history) = evolutionary_phase(graph, &args, &degrees);

    // Phase 2: Selection Model, best solution based on strategy
    let best_solution = if NUM_RANDOM_NETWORKS == 0 {
        // Use Max Q selection
        max_q_selection(&archive)
    } else {
        // Generate multiple random networks and their archives
        let random_networks = generate_random_networks(graph, NUM_RANDOM_NETWORKS);
        let random_archives: Vec<Vec<Solution>> = random_networks
            .iter()
            .map(|random_graph| {
                let random_degrees = random_graph.precompute_degress();
                let (random_archive, _) = evolutionary_phase(random_graph, &args, &random_degrees);
                random_archive
            })
            .collect();

        // Use Min-Max selection with random archives
        min_max_selection(&archive, &random_archives)
    };

    (
        best_solution.partition.clone(),
        best_fitness_history,
        best_solution.objectives[0],
    )
}
