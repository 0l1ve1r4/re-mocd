//! algorithms/pesa_ii.rs
//! Implements the Pareto Envelope-based Selection Algorithm II (PESA-II)
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2024 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

mod evolutionary;
mod hypergrid;
mod model_selection;

use crate::graph::{self, Graph, Partition};
use crate::utils::args::AGArgs;
use evolutionary::evolutionary_phase;
use hypergrid::{HyperBox, Solution};
use model_selection::{max_q_selection, min_max_selection};
use rand::seq::SliceRandom as _;
use rand::thread_rng;
use rustc_hash::FxBuildHasher;
use std::collections::HashMap;

/// Generates a random network of the same size as the original `Graph`,
/// maintaining the same number of nodes and edges while randomizing connections.
/// "In order to avoid the random factors, multiple random networks are applied 
/// (three random Pareto fronts are generated in the following experiments).""
const NUM_RANDOM_NETWORKS: usize = 3;

/// Generates multiple random networks and combines their solutions
fn generate_random_networks(original: &Graph, num_networks: usize) -> Vec<Graph> {
    (0..num_networks)
        .map(|_| {
            let mut random_graph = graph::Graph {
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

/// Main run function that creates both the real and random fronts, then
/// selects the best solution via the chosen criterion.
pub fn run(graph: &Graph, args: AGArgs) -> (Partition, Vec<f64>, f64) {
    let degrees: HashMap<i32, usize, FxBuildHasher> = graph.precompute_degress();

    // Phase 1: Evolutionary algorithm returns the Pareto frontier for the real network
    let (archive, best_fitness_history) = evolutionary_phase(graph, &args, &degrees);

    // Phase 2: Selection Model, best solution based on strategy
    let best_solution = if NUM_RANDOM_NETWORKS == 0 {
        // Use Max Q selection
        max_q_selection(&archive)
    } else {
        // Generate multiple random networks and their archives
        let random_networks: Vec<Graph> = generate_random_networks(graph, NUM_RANDOM_NETWORKS);
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
