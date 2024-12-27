use rustc_hash::FxHashMap as HashMap;
use rustc_hash::FxHashSet as HashSet;

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

pub type NodeId = i32;
pub type CommunityId = i32;
pub type Partition = BTreeMap<NodeId, CommunityId>;

#[derive(Debug)]
pub struct Graph {
    pub edges: Vec<(NodeId, NodeId)>,
    pub nodes: HashSet<NodeId>,
    pub adjacency_list: HashMap<NodeId, Vec<NodeId>>,
}

impl Graph {
    pub fn new() -> Self {
        Graph {
            edges: Vec::new(),
            nodes: HashSet::default(),
            adjacency_list: HashMap::default(),
        }
    }

    pub fn add_edge(&mut self, from: NodeId, to: NodeId) {
        self.edges.push((from, to));
        self.nodes.insert(from);
        self.nodes.insert(to);
        
        // Update adjacency list
        self.adjacency_list
            .entry(from)
            .or_insert_with(Vec::new)
            .push(to);
        self.adjacency_list
            .entry(to)
            .or_insert_with(Vec::new)
            .push(from);
    }

    pub fn from_edgelist(path: &Path) -> Result<Self, std::io::Error> {
        let mut graph = Graph::new();
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        for line in reader.lines() {
            let line = line?;
            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() >= 2 {
                let from: NodeId = parts[0].parse().unwrap();
                let to: NodeId = parts[1].parse().unwrap();
                graph.add_edge(from, to);
            }
        }
        Ok(graph)
    }

    pub fn neighbors(&self, node: &NodeId) -> &[NodeId] {
        self.adjacency_list
            .get(node)
            .map_or(&[], |x| x)
    }

    #[allow(dead_code)]
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    #[allow(dead_code)]
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    pub fn precompute_degress(&self) -> HashMap<i32, usize> {
        let degrees: HashMap<NodeId, usize> = self
            .nodes
            .iter()
            .map(|&node| (node, self.neighbors(&node).len()))
            .collect();
    
        degrees
    }

}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_graph_num_nodes() {
        let mut graph: Graph = Graph::new();
        graph.add_edge(0, 1);
        graph.add_edge(0, 2);
        graph.add_edge(0, 4);

        assert_eq!(graph.num_nodes(), 4);
    }

    #[test]
    fn test_graph_num_edges() {
        let mut graph: Graph = Graph::new();
        graph.add_edge(0, 1);
        graph.add_edge(0, 2);
        graph.add_edge(0, 4);

        assert_eq!(graph.num_edges(), 3);
    }
}