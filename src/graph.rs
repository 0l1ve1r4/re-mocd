use petgraph::stable_graph::{StableGraph, NodeIndex};
use std::collections::HashMap;

/// Representation of a community partition
#[derive(Debug, Clone)]
pub struct Partition {
    pub communities: HashMap<usize, Vec<NodeIndex>>, // community_id -> nodes
}

pub fn calculate_objectives(graph: &StableGraph<(), ()>, partition: &Partition) -> f64 {
    let total_edges = graph.edge_count();
    let mut intra = 0.0;
    let mut inter = 0.0;

    for community in partition.communities.values() {
        let mut intra_edges = 0;
        let mut total_degree = 0;

        for &node in community {
            total_degree += graph.neighbors(node).count();
            for neighbor in graph.neighbors(node) {
                if community.contains(&neighbor) {
                    intra_edges += 1;
                }
            }
        }

        intra += intra_edges as f64 / (2.0 * total_edges as f64);
        inter += (total_degree as f64 / (2.0 * total_edges as f64)).powi(2);
    }

    intra /= 2.0;

    println!("Modularity: {}, Intra: {}, Inter: {}", 1.0 - intra - inter, intra, inter);

    1.0 - intra - inter
}

// Example graph where intra-objective = 1 (no intra edges)
pub fn high_intra_graph() -> StableGraph<(), ()> {
    let mut graph_intra = StableGraph::<(), ()>::new();
    let n0 = graph_intra.add_node(());
    let n1 = graph_intra.add_node(());
    let n2 = graph_intra.add_node(());
    let n3 = graph_intra.add_node(());

    graph_intra.add_edge(n0, n1, ());
    graph_intra.add_edge(n2, n3, ());

    let partition_intra = Partition {
        communities: HashMap::from([(0, vec![n0, n2]), (1, vec![n1, n3])]),
    };

    let modularity = calculate_objectives(&graph_intra, &partition_intra);

    println!("Graph with intra-objective = 1");
    println!("Modularity: {:.4}", modularity);

    graph_intra
}

pub fn high_inter_graph() -> StableGraph<(), ()> {
    // Example graph where inter-objective = 1 (perfectly balanced degrees)
    let mut graph_inter = StableGraph::<(), ()>::new();
    let n0 = graph_inter.add_node(());
    let n1 = graph_inter.add_node(());
    let n2 = graph_inter.add_node(());
    let n3 = graph_inter.add_node(());

    graph_inter.add_edge(n0, n1, ());
    graph_inter.add_edge(n0, n2, ());
    graph_inter.add_edge(n0, n3, ());
    graph_inter.add_edge(n1, n2, ());
    graph_inter.add_edge(n1, n3, ());
    graph_inter.add_edge(n2, n3, ());

    let partition_inter = Partition {
        communities: HashMap::from([(0, vec![n0, n1, n2, n3])]),
    };

    let modularity = calculate_objectives(&graph_inter, &partition_inter);

    println!("Graph with inter-objective = 1");
    println!("Modularity: {:.4}", modularity);

    graph_inter
}
