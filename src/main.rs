use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;
use rand::prelude::*;
use serde_json::json;
use std::sync::{Arc, Mutex};
use rand::seq::IteratorRandom; // Import the trait for choose
use rayon::prelude::*;

#[derive(Debug, Clone)]
struct Graph {
    edges: Vec<(usize, usize)>,
    nodes: HashSet<usize>,
}

impl Graph {
    fn from_edgelist(file_path: &str) -> Self {
        let content = fs::read_to_string(file_path).expect("Unable to read file");
        let mut edges = Vec::new();
        let mut nodes = HashSet::new();

        for line in content.lines() {
            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() >= 2 {
                let u: usize = parts[0].trim().parse().expect("Invalid node format");
                let v: usize = parts[1].trim().parse().expect("Invalid node format");
                edges.push((u, v));
                nodes.insert(u);
                nodes.insert(v);
            }
        }

        Graph { edges, nodes }
    }

    fn neighbors(&self, node: usize) -> Vec<usize> {
        self.edges
            .iter()
            .filter_map(|&(u, v)| {
                if u == node {
                    Some(v)
                } else if v == node {
                    Some(u)
                } else {
                    None
                }
            })
            .collect()
    }

    fn number_of_edges(&self) -> usize {
        self.edges.len()
    }

    fn number_of_nodes(&self) -> usize {
        self.nodes.len()
    }
}

fn calculate_objectives(graph: &Graph, partition: &HashMap<usize, usize>) -> (f64, f64, f64) {
    let total_edges = graph.number_of_edges();
    if total_edges == 0 {
        return (0.0, 0.0, 0.0);
    }

    let mut intra_sum = 0.0;
    let mut inter = 0.0;
    let total_edges_doubled = 2.0 * total_edges as f64;

    let mut communities: HashMap<usize, HashSet<usize>> = HashMap::new();
    for (&node, &community) in partition {
        communities.entry(community).or_default().insert(node);
    }

    for community_nodes in communities.values() {
        let mut community_edges = 0;
        let mut community_degree = 0.0;

        for &node in community_nodes {
            let node_degree = graph.neighbors(node).len() as f64;
            community_degree += node_degree;

            for neighbor in graph.neighbors(node) {
                if community_nodes.contains(&neighbor) {
                    community_edges += 1;
                }
            }
        }

        let intra_community_edges = community_edges as f64 / 2.0;
        intra_sum += intra_community_edges;
        let normalized_degree = community_degree / total_edges_doubled;
        inter += normalized_degree.powi(2);
    }

    let intra = 1.0 - (intra_sum / total_edges as f64);
    let modularity = (1.0 - intra - inter).clamp(-1.0, 1.0);

    (modularity, intra, inter)
}

fn generate_initial_population(graph: &Graph, population_size: usize) -> Vec<HashMap<usize, usize>> {
    let mut population = Vec::new();
    let nodes: Vec<usize> = graph.nodes.iter().cloned().collect();
    let mut rng = thread_rng();

    for _ in 0..population_size {
        let mut partition = HashMap::new();
        for &node in &nodes {
            partition.insert(node, rng.gen_range(0..nodes.len()));
        }
        population.push(partition);
    }
    population
}

fn crossover(parent1: &HashMap<usize, usize>, parent2: &HashMap<usize, usize>) -> HashMap<usize, usize> {
    let keys: Vec<&usize> = parent1.keys().collect();
    let mut rng = thread_rng();
    let (idx1, idx2) = {
        let mut indices = [rng.gen_range(0..keys.len()), rng.gen_range(0..keys.len())];
        indices.sort_unstable();
        (indices[0], indices[1])
    };

    let mut child = parent1.clone();
    for &key in &keys[idx1..idx2] {
        child.insert(*key, parent2[key]);
    }
    child
}


fn selection(population: &[HashMap<usize, usize>], fitnesses: &[f64]) -> Vec<HashMap<usize, usize>> {
    let mut sorted: Vec<_> = population.iter().zip(fitnesses).collect();
    sorted.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    sorted.iter().take(population.len() / 2).map(|(p, _)| (*p).clone()).collect()
}

fn genetic_algorithm(graph: &Graph, generations: usize, population_size: usize) {
    let mut population = generate_initial_population(graph, population_size);
    let mut best_partition = None;
    let mut best_modularity = f64::MIN;

    for _ in 0..generations {
        let fitnesses: Vec<f64> = population
            .par_iter()
            .map(|partition| calculate_objectives(graph, partition).0)
            .collect();

        let selected_population = selection(&population, &fitnesses);

        let mut new_population = Vec::new();
        let mut rng = thread_rng();

        while new_population.len() < population_size {
            let parents: Vec<_> = selected_population
                .choose_multiple(&mut rng, 2)
                .cloned()
                .collect();

            let mut child = crossover(&parents[0], &parents[1]);
            new_population.push(child);
        }

        population = new_population;

        if let Some((idx, &modularity)) = fitnesses
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        {
            if modularity > best_modularity {
                best_modularity = modularity;
                best_partition = Some(population[idx].clone());
            }
        }
    }

    if let Some(partition) = best_partition {
        let output = json!({ "best_partition": partition });
        fs::write("best_partition.json", serde_json::to_string_pretty(&output).unwrap()).expect("Unable to write file");
    }
}

fn main() {
    let graph = Graph::from_edgelist("src/graphs/artificials/article.edgelist");
    genetic_algorithm(&graph, 800, 100);
}
