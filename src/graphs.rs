use petgraph::stable_graph::{StableGraph, NodeIndex};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Partition {
    pub communities: HashMap<usize, Vec<NodeIndex>>, // community_id -> nodes
    
}

// Helper function to create a test partition
fn create_partition(communities: Vec<Vec<NodeIndex>>) -> Partition {
    let mut partition = Partition {
        communities: HashMap::new(),
    };
    for (i, community) in communities.into_iter().enumerate() {
        partition.communities.insert(i, community);
    }
    partition
}

// Create a graph with two perfectly separated communities
// Community 1: 0-1-2 (fully connected)
// Community 2: 3-4-5 (fully connected)
// No edges between communities
pub fn perfectly_communities() -> (StableGraph<(), ()>, Partition) {

    let mut graph = StableGraph::new();
    let nodes: Vec<NodeIndex> = (0..6).map(|_| graph.add_node(())).collect();
        
    // Community 1
    graph.add_edge(nodes[0], nodes[1], ());
    graph.add_edge(nodes[1], nodes[2], ());
    graph.add_edge(nodes[0], nodes[2], ());
        
    // Community 2
    graph.add_edge(nodes[3], nodes[4], ());
    graph.add_edge(nodes[4], nodes[5], ());
    graph.add_edge(nodes[3], nodes[5], ());

    let partition = create_partition(vec![
        vec![nodes[0], nodes[1], nodes[2]],
        vec![nodes[3], nodes[4], nodes[5]]
    ]);

    (graph, partition)

}

/// Creates a graph with complete mixing between communities.
/// 
/// # Structure
/// * Four nodes connected in a cycle: n1-n2-n3-n4-n1
/// * Communities are interleaved:
///   - Community 1: n1, n3
///   - Community 2: n2, n4
/// 
pub fn create_complete_mixing() -> (StableGraph<(), ()>, Partition) {
    let mut graph = StableGraph::new();
    let n1 = graph.add_node(());
    let n2 = graph.add_node(());
    let n3 = graph.add_node(());
    let n4 = graph.add_node(());

    // Create complete mixing by connecting all nodes
    graph.add_edge(n1, n2, ());
    graph.add_edge(n2, n3, ());
    graph.add_edge(n3, n4, ());
    graph.add_edge(n4, n1, ());

    let mut partition = Partition {
        communities: HashMap::new(),
    };

    // Split nodes evenly between two communities
    partition.communities.insert(0, vec![n1, n3]);
    partition.communities.insert(1, vec![n2, n4]);

    (graph, partition)
}

/// Creates a graph with a single fully connected community.
/// 
/// # Structure
/// * Three nodes forming a triangle: n1-n2-n3-n1
/// * All nodes belong to the same community
/// 
pub fn create_single_community() -> (StableGraph<(), ()>, Partition) {
    let mut graph = StableGraph::new();
    let n1 = graph.add_node(());
    let n2 = graph.add_node(());
    let n3 = graph.add_node(());

    // Create a single fully connected community
    graph.add_edge(n1, n2, ());
    graph.add_edge(n2, n3, ());
    graph.add_edge(n3, n1, ());

    let mut partition = Partition {
        communities: HashMap::new(),
    };

    // Put all nodes in one community
    partition.communities.insert(0, vec![n1, n2, n3]);

    (graph, partition)
}

/// Creates a graph with two sparsely connected communities.
/// 
/// # Structure
/// * Community 1: line graph 0-1-2
/// * Community 2: line graph 3-4-5
/// * One bri
pub fn create_sparse_communities() -> (StableGraph<(), ()>, Partition) {
    let mut graph = StableGraph::new();
    let nodes: Vec<NodeIndex> = (0..6).map(|_| graph.add_node(())).collect();

    // Community 1: line graph 0-1-2
    graph.add_edge(nodes[0], nodes[1], ());
    graph.add_edge(nodes[1], nodes[2], ());

    // Community 2: line graph 3-4-5
    graph.add_edge(nodes[3], nodes[4], ());
    graph.add_edge(nodes[4], nodes[5], ());

    // One edge between communities
    graph.add_edge(nodes[2], nodes[3], ());

    let partition = create_partition(vec![
        vec![nodes[0], nodes[1], nodes[2]],
        vec![nodes[3], nodes[4], nodes[5]],
    ]);

    (graph, partition)
}

/// Creates an empty graph with no nodes or edges.
/// 
/// # Structure
/// * Empty graph with no nodes
/// * Empty partition with no communities
/// 
pub fn create_empty_graph() -> (StableGraph<(), ()>, Partition) {
    let graph = StableGraph::<(), ()>::new();
    let partition = create_partition(vec![]);

    (graph, partition)
}