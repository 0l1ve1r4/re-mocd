use crate::graph::{Graph, Partition, NodeId, CommunityId};
use std::collections::{HashMap, BTreeMap};

/// Reduces a graph by combining nodes according to their communities in the partition
pub fn expand_partition(
    reduced_partition: &Partition,
    community_mapping: &BTreeMap<CommunityId, Vec<NodeId>>,
) -> Partition {
    let mut expanded_partition = Partition::new();
    
    for (&reduced_node, &final_community) in reduced_partition {
        if let Some(original_nodes) = community_mapping.get(&reduced_node) {
            for &original_node in original_nodes {
                expanded_partition.insert(original_node, final_community);
            }
        }
    }
    
    for (&community_id, original_nodes) in community_mapping {
        if !reduced_partition.contains_key(&community_id) {
            for &original_node in original_nodes {
                if !expanded_partition.contains_key(&original_node) {
                    expanded_partition.insert(original_node, community_id);
                }
            }
        }
    }
    
    for &original_node in community_mapping.values().flat_map(|v| v) {
        if !expanded_partition.contains_key(&original_node) {
            expanded_partition.insert(original_node, original_node);
        }
    }
    
    expanded_partition
}

pub fn reduce_graph(original_graph: &Graph, partition: &Partition) -> (Graph, BTreeMap<CommunityId, Vec<NodeId>>) {
    let mut community_mapping: BTreeMap<CommunityId, Vec<NodeId>> = BTreeMap::new();
    let mut edge_counts: HashMap<(CommunityId, CommunityId), usize> = HashMap::default();
    
    // Step 1: Map nodes to their communities
    for &node in &original_graph.nodes {
        let community = partition.get(&node).copied().unwrap_or(node);
        community_mapping
            .entry(community)
            .or_insert_with(Vec::new)
            .push(node);
    }
    
    // Step 2: Count edges between communities
    for &(node1, node2) in &original_graph.edges {
        let comm1 = partition.get(&node1).copied().unwrap_or(node1);
        let comm2 = partition.get(&node2).copied().unwrap_or(node2);
        
        let comm_pair = if comm1 < comm2 {
            (comm1, comm2)
        } else {
            (comm2, comm1)
        };
        *edge_counts.entry(comm_pair).or_insert(0) += 1;
    }
    
    // Step 3: Create the reduced graph
    let mut reduced_graph = Graph::new();
    
    // Add all communities as nodes
    for &community in community_mapping.keys() {
        reduced_graph.nodes.insert(community);
        reduced_graph.adjacency_list.insert(community, Vec::new());
    }
    
    // Add edges between communities
    for ((comm1, comm2), _count) in edge_counts {
        reduced_graph.add_edge(comm1, comm2);
    }
    
    (reduced_graph, community_mapping)
}