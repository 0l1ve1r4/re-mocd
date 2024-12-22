/// This Source Code Form is subject to the terms of The GNU General Public License v3.0
/// Copyright 2024 - Guilherme Santos. If a copy of the GPL3 was not distributed with this
/// file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html
use crate::operators::Partition;
use petgraph::graph::{Graph, NodeIndex};
use petgraph::Undirected;
use rayon::prelude::*;
use rustc_hash::FxHashMap as HashMap;

/// Calculates modularity along with intra- and inter-community metrics.
pub fn calculate_objectives(
    graph: &Graph<(), (), Undirected>,
    partition: &Partition,
    node_degrees: &[f64],
) -> (f64, f64, f64) {
    let total_edges = graph.edge_count() as f64;
    if total_edges == 0.0 {
        return (0.0, 0.0, 0.0);
    }

    let total_edges_doubled: f64 = 2.0 * total_edges;

    // Create a quick membership lookup so membership[node.index()] = community_id. O(1)
    let mut membership = vec![0; graph.node_count()];
    for (&node_idx, &community_id) in partition.iter() {
        membership[node_idx.index()] = community_id;
    }

    // Group nodes by community for parallel iteration.
    let mut communities: HashMap<usize, Vec<NodeIndex>> = HashMap::default();
    for (&node_idx, &community_id) in partition.iter() {
        communities.entry(community_id).or_default().push(node_idx);
    }
    let community_vectors: Vec<_> = communities.values().collect();

    // Compute intra-community edges and sum of degrees per community in parallel.
    let (sum_intra_edges, sum_inter_value) = community_vectors
        .par_iter()
        .map(|nodes_in_comm| {
            let mut community_edges = 0.0;
            let mut degree_sum = 0.0;

            for &node_idx in nodes_in_comm.iter() {
                degree_sum += node_degrees[node_idx.index()];
                for neighbor_idx in graph.neighbors(node_idx) {
                    // Only count edges within the same community.
                    if membership[neighbor_idx.index()] == membership[node_idx.index()] {
                        community_edges += 1.0;
                    }
                }
            }

            // Divide by 2.0 because edges are undirected (counted twice).
            community_edges *= 0.5; // Multiplication is less cpu costly.

            // Compute a portion for "inter" (inter-community measure).
            let normalized_deg = degree_sum / total_edges_doubled;
            let inter_part = normalized_deg * normalized_deg;

            (community_edges, inter_part)
        })
        .reduce(
            // Identity function that starts sums at (0.0, 0.0).
            || (0.0, 0.0),
            // Accumulate partial results into a total.
            |(edges_acc, inter_acc), (edges_part, inter_part)| {
                (edges_acc + edges_part, inter_acc + inter_part)
            },
        );

    let intra_value: f64 = 1.0 - (sum_intra_edges / total_edges);

    // Modularity is capped between -1.0 and 1.0.
    let mut modularity: f64 = 1.0 - intra_value - sum_inter_value;
    modularity = modularity.clamp(-1.0, 1.0);

    (modularity, intra_value, sum_inter_value)
}