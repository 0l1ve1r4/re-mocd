//! operators/objective.rs
//! Genetic Algorithm fitness function
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2024 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::graph::{CommunityId, Graph, NodeId, Partition};
use crate::operators::metrics::Metrics;

use rayon::iter::*;
use rustc_hash::FxHashMap as HashMap;

pub fn calculate_objectives(
    graph: &Graph,
    partition: &Partition,
    degrees: &HashMap<i32, usize>,
    parallel: bool,
) -> Metrics {
    let total_edges = graph.edges.len() as f64;
    if total_edges == 0.0 {
        return Metrics::default();
    }

    // Build communities from the partition
    let mut communities: HashMap<CommunityId, Vec<NodeId>> = HashMap::default();
    for (&node, &community) in partition {
        communities.entry(community).or_default().push(node);
    }

    let total_edges_doubled = 2.0 * total_edges;
    let folder = |(mut intra_acc, mut inter_acc), (_, community_nodes): (&i32, &Vec<i32>)| {
        let mut community_edges = 0.0;
        let mut community_degree = 0.0;

        for &node in community_nodes {
            // Use precomputed degree
            let node_degree = degrees.get(&node).copied().unwrap_or(0) as f64;
            community_degree += node_degree;

            // Iterate through neighbors once
            for neighbor in graph.neighbors(&node) {
                if community_nodes.binary_search(neighbor).is_ok() {
                    community_edges += 1.0;
                }
            }
        }

        // Avoid double counting by dividing by 2
        community_edges /= 2.0;
        intra_acc += community_edges;

        // Calculate normalized degree
        let normalized_degree = community_degree / total_edges_doubled;
        inter_acc += normalized_degree.powi(2);

        (intra_acc, inter_acc)
    };
    let (intra_sum, inter) = if parallel {
        communities
            .par_iter()
            .fold(
                || (0.0, 0.0), // Initialize accumulators for each thread
                folder,
            )
            .reduce(
                || (0.0, 0.0),                 // Initialize accumulators for reduction
                |a, b| (a.0 + b.0, a.1 + b.1), // Combine results from different threads
            )
    } else {
        communities.iter().fold((0.0, 0.0), folder)
    };

    let intra = 1.0 - (intra_sum / total_edges);
    let mut modularity = 1.0 - intra - inter;
    modularity = modularity.clamp(-1.0, 1.0);

    Metrics {
        modularity,
        intra,
        inter,
    }
}
