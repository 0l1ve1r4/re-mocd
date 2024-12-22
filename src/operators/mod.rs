/// This Source Code Form is subject to the terms of The GNU General Public License v3.0
/// Copyright 2024 - Guilherme Santos. If a copy of the GPL3 was not distributed with this
/// file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html
/// 
pub mod ga;
pub mod modularity;

use rustc_hash::FxHashMap as HashMap;
use petgraph::graph::NodeIndex;

use petgraph::graph::Graph;
use petgraph::Undirected;

pub type Partition = HashMap<NodeIndex, usize>;

/// Precompute degrees for each node. This function returns a vector indexed by NodeIndex.index().
pub fn compute_node_degrees(graph: &Graph<(), (), Undirected>) -> Vec<f64> {
    let mut degrees = vec![0.0; graph.node_count()];
    for node in graph.node_indices() {
        degrees[node.index()] = graph.neighbors(node).count() as f64;
    }
    degrees
}