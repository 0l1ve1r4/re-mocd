/// This Source Code Form is subject to the terms of The GNU General Public License v3.0
/// Copyright 2024 - Guilherme Santos. If a copy of the GPL3 was not distributed with this
/// file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use petgraph::graph::{Graph, NodeIndex};
use petgraph::Undirected;

use crate::operators::Partition;
use rustc_hash::FxHashMap as HashMap;
use rayon::prelude::*;

pub fn calculate_objectives(
    graph: &Graph<(), (), Undirected>,
    partition: &Partition,
    node_degrees: &[f64]
) -> (f64, f64, f64) {
    let total_edges: f64 = graph.edge_count() as f64;
    if total_edges == 0.0 {
        return (0.0, 0.0, 0.0);
    }
    let total_edges_doubled: f64 = 2.0 * total_edges;

    // Convert partition to a Vec-based membership for O(1) lookups:
    // membership[node.index()] = community_id
    let mut membership = vec![0; graph.node_count()];
    for (&node, &comm) in partition {
        membership[node.index()] = comm;
    }

    // Build sets of nodes by community
    let mut communities: HashMap<usize, Vec<NodeIndex>> = HashMap::default();
    for (&node, &community) in partition.iter() {
        communities.entry(community).or_default().push(node);
    }

    let community_vec: Vec<_> = communities.values().collect();

    // Parallel processing of communities
    let (intra_sum, inter_sum) = community_vec
        .par_iter()
        .map(|community_nodes| {
            let mut community_edges = 0.0;
            let mut community_degree_sum = 0.0;

            // Summation for edges within the same community
            for &node in community_nodes.iter() {
                community_degree_sum += node_degrees[node.index()];
                for neighbor in graph.neighbors(node) {
                    // Only count edge if neighbor is in the same community
                    if membership[neighbor.index()] == membership[node.index()] {
                        community_edges += 1.0;
                    }
                }
            }

            // Adjust for undirected graph (each edge counted twice)
            community_edges /= 2.0;
            let normalized_degree = community_degree_sum / total_edges_doubled;
            let inter = normalized_degree * normalized_degree;

            (community_edges, inter)
        })
        .reduce(
            || (0.0, 0.0),
            |(sum_edges1, sum_inter1), (sum_edges2, sum_inter2)| {
                (sum_edges1 + sum_edges2, sum_inter1 + sum_inter2)
            },
        );

    let intra = 1.0 - (intra_sum / total_edges);
    let mut modularity = 1.0 - intra - inter_sum;
    modularity = modularity.clamp(-1.0, 1.0);

    (modularity, intra, inter_sum)
}