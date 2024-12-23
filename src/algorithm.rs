use std::collections::{HashMap, HashSet};
use rand::Rng;
use rand::seq::SliceRandom;
use rayon::prelude::*;


use crate::graph::{
    Graph,
    Partition,
    CommunityId,
    NodeId
};

#[derive(Debug)]
#[allow(dead_code)]
pub struct Metrics {
    pub modularity: f64,
    intra: f64,
    inter: f64,
}

pub fn calculate_objectives(graph: &Graph, partition: &Partition) -> Metrics {
    let total_edges = graph.edges.len() as f64;
    if total_edges == 0.0 {
        return Metrics {
            modularity: 0.0,
            intra: 0.0,
            inter: 0.0,
        };
    }

    let mut communities: HashMap<CommunityId, HashSet<NodeId>> = HashMap::new();
    for (node, community) in partition {
        communities
            .entry(*community)
            .or_insert_with(HashSet::new)
            .insert(*node);
    }

    let mut intra_sum = 0.0;
    let mut inter = 0.0;
    let total_edges_doubled = 2.0 * total_edges;

    for community_nodes in communities.values() {
        let mut community_edges = 0.0;
        let mut community_degree = 0.0;

        for &node in community_nodes {
            let node_degree = graph.neighbors(&node).len() as f64;
            community_degree += node_degree;

            for neighbor in graph.neighbors(&node) {
                if community_nodes.contains(&neighbor) {
                    community_edges += 1.0;
                }
            }
        }

        community_edges /= 2.0;
        intra_sum += community_edges;
        let normalized_degree = community_degree / total_edges_doubled;
        inter += normalized_degree.powi(2);
    }

    let intra = 1.0 - (intra_sum / total_edges);
    let mut modularity = 1.0 - intra - inter;
    modularity = modularity.max(-1.0).min(1.0);

    Metrics {
        modularity,
        intra,
        inter,
    }
}

fn generate_initial_population(graph: &Graph, population_size: usize) -> Vec<Partition> {
    let mut rng = rand::thread_rng();
    let nodes: Vec<NodeId> = graph.nodes.iter().cloned().collect();
    let num_nodes = nodes.len();

    (0..population_size)
        .map(|_| {
            nodes
                .iter()
                .map(|&node| (node, rng.gen_range(0..num_nodes) as CommunityId))
                .collect()
        })
        .collect()
}

fn crossover(parent1: &Partition, parent2: &Partition) -> Partition {
    let mut rng = rand::thread_rng();
    let keys: Vec<NodeId> = parent1.keys().cloned().collect();
    let len = keys.len();
    let (idx1, idx2) = {
        let mut points = vec![rng.gen_range(0..len), rng.gen_range(0..len)];
        points.sort();
        (points[0], points[1])
    };

    let mut child = parent1.clone();
    for i in idx1..idx2 {
        if let Some(&community) = parent2.get(&keys[i]) {
            child.insert(keys[i], community);
        }
    }
    child
}

fn mutate(partition: &mut Partition, graph: &Graph) {
    let mut rng = rand::thread_rng();
    let nodes: Vec<NodeId> = partition.keys().cloned().collect();
    let node = nodes.choose(&mut rng).unwrap();
    let neighbors = graph.neighbors(node);
    
    if let Some(&neighbor) = neighbors.choose(&mut rng) {
        if let Some(&neighbor_community) = partition.get(&neighbor) {
            partition.insert(*node, neighbor_community);
        }
    }
}

pub fn genetic_algorithm(graph: &Graph, generations: usize, population_size: usize, debug: bool) -> (Partition, Vec<f64>) {
    let mut rng = rand::thread_rng();
    let mut population = generate_initial_population(graph, population_size);
    let mut best_fitness_history = Vec::with_capacity(generations);

    for generation in 0..generations {
        // Evaluate fitness in parallel
        let fitnesses: Vec<Metrics> = population
            .par_iter()
            .map(|partition| calculate_objectives(graph, partition))
            .collect();

        // Record best fitness
        let best_fitness = fitnesses
            .iter()
            .map(|m| m.modularity)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        best_fitness_history.push(best_fitness);

        // Selection
        let mut population_with_fitness: Vec<_> = population
            .into_iter()
            .zip(fitnesses)
            .collect();
        population_with_fitness.sort_by(|(_, a), (_, b)| {
            b.modularity.partial_cmp(&a.modularity).unwrap()
        });
        population = population_with_fitness
            .into_iter()
            .take(population_size / 2)
            .map(|(p, _)| p)
            .collect();

        // Create new population
        let mut new_population = Vec::with_capacity(population_size);
        while new_population.len() < population_size {
            let parent1 = population.choose(&mut rng).unwrap();
            let parent2 = population.choose(&mut rng).unwrap();
            let mut child = crossover(parent1, parent2);
            mutate(&mut child, graph);
            new_population.push(child);
        }
        population = new_population;

        if debug{
            println!("Generation: {} \t | Best Fitness: {}", 
            generation, 
            best_fitness
            );
        }

    }

    // Find best partition
    let best_partition = population
        .into_iter()
        .max_by_key(|partition| {
            let metrics = calculate_objectives(graph, partition);
            (metrics.modularity * 1000.0) as i64
        })
        .unwrap();

    (best_partition, best_fitness_history)
}
