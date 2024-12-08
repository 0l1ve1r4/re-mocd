use petgraph::stable_graph::StableGraph;
use crate::graphs::Partition;

pub fn calculate_objectives(graph: &StableGraph<(), ()>, partition: &Partition) -> f64 {
    let total_edges = graph.edge_count() as f64;
    if total_edges == 0.0 {
        return 0.0;
    }
    
    let mut intra_sum = 0.0;
    let mut inter = 0.0;
    let total_edges_doubled = 2.0 * total_edges;  // Add this for normalization

    // Calculate both metrics for each community
    for community in partition.communities.values() {
        let mut community_edges = 0;
        let mut community_degree = 0.0;

        // Count internal edges and total degree
        for &node in community {
            let node_degree = graph.neighbors(node).count() as f64;
            community_degree += node_degree;
            
            // Count edges within the community
            for neighbor in graph.neighbors(node) {
                if community.contains(&neighbor) {
                    community_edges += 1;
                }
            }
        }
        
        // Handle edge counting for undirected graphs
        if !graph.is_directed() {
            community_edges /= 2;
        }
        
        // Add to intra-link sum
        intra_sum += community_edges as f64;
        
        // Fix inter-link calculation
        let normalized_degree = community_degree / total_edges_doubled;
        inter += normalized_degree.powi(2);
    }

    // Calculate final intra value
    let intra = 1.0 - (intra_sum / total_edges);

    // Calculate final modularity with proper scaling
    let modularity = 1.0 - intra - inter;

    println!("Modularity: {} | Intra: {} | Inter: {}", modularity, intra, inter);

    // Ensure modularity is in valid range [-1, 1]
    modularity.max(-1.0).min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graphs::{
        perfectly_communities,
        create_complete_mixing,
        create_empty_graph,
        create_single_community,
        create_sparse_communities,        
    };

    #[test]
    fn test_perfect_separation() {
        let (graph, partition) = perfectly_communities();
        let modularity = calculate_objectives(&graph, &partition);
        // High modularity expected as communities are perfectly separated
        assert!(modularity > 0.7, "Modularity should be high for perfect separation");
    }

    #[test]
    fn test_complete_mixing() {
        let (graph, partition) = create_complete_mixing();
        let modularity = calculate_objectives(&graph, &partition);
        assert!(modularity.abs() < 0.2); // Should be close to 0 for complete mixing
    }

    #[test]
    fn test_single_community() {
        let (graph, partition) = create_single_community();
        let modularity = calculate_objectives(&graph, &partition);
        assert!(modularity > 0.0); // Should be positive for a single well-defined community
    }

    #[test]
    fn test_sparse_communities() {
        let (graph, partition) = create_sparse_communities();
        let modularity = calculate_objectives(&graph, &partition);
        // Moderate modularity expected due to sparse internal connections
        assert!(modularity > 0.2 && modularity < 0.7, 
            "Modularity should be moderate for sparse communities");
    }

    #[test]
    fn test_empty_graph() {
        let (graph, partition) = create_empty_graph();
        let modularity = calculate_objectives(&graph, &partition);
        assert_eq!(modularity, 0.0, "Modularity should be 0.0 for empty graph");
    }
}
