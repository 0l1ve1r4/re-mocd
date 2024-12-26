use rand::seq::SliceRandom;
use rand::Rng;
use rayon::prelude::*;
use rustc_hash::FxHashMap as HashMap;

use crate::graph::{CommunityId, Graph, NodeId, Partition};
use crate::args::AGArgs;

#[derive(Debug)]
#[derive(PartialEq)]
#[allow(dead_code)]
pub struct Metrics {
    pub modularity: f64,
    intra: f64,
    inter: f64,
}

impl Metrics {
    pub fn default() -> Self {
        return Metrics {
            modularity: 0.0,
            intra: 0.0,
            inter: 0.0,
        };
    }
}

// Maximum number of generations with unchanged fitness
const MAX_GENERATIONS_WITH_SAME_FITNESS: usize = 20;

// Tolerance for floating-point fitness comparisons
const FITNESS_COMPARISON_EPSILON: f64 = 0.001;

pub fn calculate_objectives(
    graph: &Graph,
    partition: &Partition,
    degrees: &HashMap<i32, usize>,
    parallel: bool,
) -> Metrics {
    let total_edges = graph.edges.len() as f64;
    if total_edges == 0.0 {
        return Metrics::default();
    }

    // Build communities from the partition
    let mut communities: HashMap<CommunityId, Vec<NodeId>> = HashMap::default();
    for (&node, &community) in partition {
        communities
            .entry(community)
            .or_default()
            .push(node);
    }

    let total_edges_doubled = 2.0 * total_edges;
    let folder = |(mut intra_acc, mut inter_acc), (_, community_nodes): (&i32, &Vec<i32>)| {
        let mut community_edges = 0.0;
        let mut community_degree = 0.0;

        for &node in community_nodes {
            // Use precomputed degree
            let node_degree = degrees.get(&node).copied().unwrap_or(0) as f64;
            community_degree += node_degree;

            // Iterate through neighbors once
            for neighbor in graph.neighbors(&node) {
                if community_nodes.binary_search(&neighbor).is_ok() {
                    community_edges += 1.0;
                }
            }
        }

        // Avoid double counting by dividing by 2
        community_edges /= 2.0;
        intra_acc += community_edges;

        // Calculate normalized degree
        let normalized_degree = community_degree / total_edges_doubled;
        inter_acc += normalized_degree.powi(2);

        (intra_acc, inter_acc)
    };
    let (intra_sum, inter) = if parallel {
        communities
            .par_iter()
            .fold(
                || (0.0, 0.0), // Initialize accumulators for each thread
                folder,
            )
            .reduce(
                || (0.0, 0.0),                 // Initialize accumulators for reduction
                |a, b| (a.0 + b.0, a.1 + b.1), // Combine results from different threads
            )
    } else {
        communities.iter().fold((0.0, 0.0), folder)
    };

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
    let nodes: Vec<NodeId> = graph.nodes.iter().copied().collect();
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
    let keys: Vec<NodeId> = parent1.keys().copied().collect();
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
    let nodes: Vec<NodeId> = partition.keys().copied().collect();
    let node = nodes.choose(&mut rng).unwrap();
    let neighbors = graph.neighbors(node);

    if let Some(&neighbor) = neighbors.choose(&mut rng) {
        if let Some(&neighbor_community) = partition.get(&neighbor) {
            partition.insert(*node, neighbor_community);
        }
    }
}

fn last_x_same(vec: &Vec<f64>, x: usize, epsilon: f64) -> bool {
    if vec.len() < x {
        return false;
    }

    let last_x = &vec[vec.len() - x..];
    let first_value = last_x[0];

    for &value in &last_x[1..] {
        if (value - first_value).abs() > epsilon {
            return false;
        }
    }
    true
}

pub fn genetic_algorithm(
    graph: &Graph,
    args: AGArgs,
) -> (Partition, Vec<f64>, f64) {
    let mut rng = rand::thread_rng();
    let mut population = generate_initial_population(graph, args.pop_size);
    let mut best_fitness_history = Vec::with_capacity(args.num_gens);
    let degress = graph.precompute_degress();

    for generation in 0..args.num_gens {
        let fitnesses: Vec<Metrics> = if args.parallelism {
            population
                .par_iter()
                .map(|partition| calculate_objectives(graph, partition, &degress, true))
                .collect()
        } else {
            population
                .iter()
                .map(|partition| calculate_objectives(graph, partition, &degress, false))
                .collect()
        };

        // Record best fitness
        let best_fitness = fitnesses
            .iter()
            .map(|m| m.modularity)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        best_fitness_history.push(best_fitness);

        // Selection
        let mut population_with_fitness: Vec<_> = population.into_iter().zip(fitnesses).collect();
        population_with_fitness
            .sort_by(|(_, a), (_, b)| b.modularity.partial_cmp(&a.modularity).unwrap());
        population = population_with_fitness
            .into_iter()
            .take(args.pop_size / 2)
            .map(|(p, _)| p)
            .collect();

        // Create new population
        let mut new_population = Vec::with_capacity(args.pop_size);
        while new_population.len() < args.pop_size {
            let parent1 = population.choose(&mut rng).unwrap();
            let parent2 = population.choose(&mut rng).unwrap();
            let mut child = crossover(parent1, parent2);
            mutate(&mut child, graph);
            new_population.push(child);
        }
        population = new_population;

        if args.debug {
            println!(
                "Generation: {} \t | Best Fitness: {}",
                generation, best_fitness
            );
        }
    
        if last_x_same(&best_fitness_history, MAX_GENERATIONS_WITH_SAME_FITNESS, FITNESS_COMPARISON_EPSILON){
            if args.debug {
                println!("[Optimization]: Max Local, breaking...");
            }
            break;
        }
        
    }

    // Find best partition
    let best_partition = population
        .into_iter()
        .max_by_key(|partition| {
            let metrics = calculate_objectives(graph, partition, &degress, args.parallelism);
            (metrics.modularity * 1000.0) as i64
        })
        .unwrap();

    let max_modularity = best_fitness_history
        .iter()
        .fold(None, |max, &val| match max {
            None => Some(val),
            Some(max_val) if val > max_val && !val.is_nan() => Some(val),
            Some(max_val) => Some(max_val),
        });

    (
        best_partition,
        best_fitness_history,
        max_modularity.unwrap(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_calculate_objectives() {
        let graph: Graph = Graph::new();
        let partition: Partition = Partition::new();

        assert_eq!(calculate_objectives(&graph, 
            &partition, 
            &graph.precompute_degress(),
            true
        ), Metrics {
            inter: 0.0,
            intra: 0.0,
            modularity: 0.0,
        });
    }

    #[test]
    #[should_panic]
    fn test_panic_ga() {
        let graph: Graph = Graph::new();
        genetic_algorithm(&graph, AGArgs::default());
    }


}