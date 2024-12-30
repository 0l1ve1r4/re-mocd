use crate::graph::{CommunityId, Graph, NodeId, Partition};
use rand::seq::SliceRandom;
use rand::Rng;
use rayon::prelude::*;
use rustc_hash::FxBuildHasher;
use rustc_hash::FxHashMap as HashMap;
use rustc_hash::FxHashSet as HashSet;
use std::collections::BTreeMap;

// Maximum number of generations with unchanged fitness
pub const MAX_GENERATIONS_WITH_SAME_FITNESS: usize = 10;

// Tolerance for floating-point fitness comparisons
pub const FITNESS_COMPARISON_EPSILON: f64 = 1e-4;

#[derive(Debug, PartialEq)]
#[allow(dead_code)]
pub struct Metrics {
    pub modularity: f64,
    pub intra: f64,
    pub inter: f64,
}

impl Metrics {
    pub fn default() -> Self {
        Metrics {
            modularity: 0.0,
            intra: 0.0,
            inter: 0.0,
        }
    }
}

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
        communities.entry(community).or_default().push(node);
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
                if community_nodes.binary_search(neighbor).is_ok() {
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
    modularity = modularity.clamp(-1.0, 1.0);

    Metrics {
        modularity,
        intra,
        inter,
    }
}

pub fn optimized_initial_population(graph: &Graph, population_size: usize) -> Vec<Partition> {
    let mut rng = rand::thread_rng();
    let nodes: Vec<NodeId> = graph.nodes.iter().copied().collect();
    let num_nodes = nodes.len();
    let mut population = Vec::with_capacity(population_size);

    // 1. Random Partitions (33% of population)
    for _ in 0..population_size / 3 {
        let mut partition = BTreeMap::new();
        for &node in &nodes {
            partition.insert(node, rng.gen_range(0..=(num_nodes / 2)) as CommunityId);
        }
        population.push(partition);
    }

    // 2. Neighbor-based Partitions (33% of population)
    for _ in 0..population_size / 3 {
        let mut partition = BTreeMap::new();
        let mut unassigned: HashSet<_> = nodes.iter().copied().collect();
        let mut current_community: CommunityId = 0;

        while !unassigned.is_empty() {
            let start_node = *unassigned.iter().next().unwrap();
            let mut to_process = vec![start_node];

            while let Some(node) = to_process.pop() {
                if unassigned.remove(&node) {
                    partition.insert(node, current_community);

                    // Add some neighbors with 70% probability
                    for &neighbor in graph.neighbors(&node) {
                        if unassigned.contains(&neighbor) && rng.gen_bool(0.7) {
                            to_process.push(neighbor);
                        }
                    }
                }
            }
            current_community += 1;
        }
        population.push(partition);
    }

    // 3. Single-community and Small-community Partitions (remaining population)
    for i in 2 * population_size / 3..population_size {
        let mut partition = BTreeMap::new();
        if i % 2 == 0 {
            // Single community
            for &node in &nodes {
                partition.insert(node, 0);
            }
        } else {
            // Two communities based on degree
            for &node in &nodes {
                let degree = graph.neighbors(&node).len();
                partition.insert(node, if degree > nodes.len() / 2 { 0 } else { 1 });
            }
        }
        population.push(partition);
    }

    population
}

pub fn optimized_crossover(parent1: &Partition, parent2: &Partition) -> Partition {
    let mut rng = rand::thread_rng();

    // Use Vec for faster sequential access
    let keys: Vec<NodeId> = parent1.keys().copied().collect();
    let len = keys.len();

    // Optimize crossover point selection
    let crossover_points: (usize, usize) = {
        let point1: usize = rng.gen_range(0..len);
        let point2: usize = (point1 + rng.gen_range(1..len / 2)).min(len - 1);
        (point1, point2)
    };

    // Pre-allocate with capacity
    let mut child: BTreeMap<i32, i32> = Partition::new();

    // Copy elements before crossover point from parent1
    keys.iter().take(crossover_points.0).for_each(|&key| {
        if let Some(&community) = parent1.get(&key) {
            child.insert(key, community);
        }
    });

    // Copy elements in crossover region from parent2
    keys.iter()
        .skip(crossover_points.0)
        .take(crossover_points.1 - crossover_points.0)
        .for_each(|&key| {
            if let Some(&community) = parent2.get(&key) {
                child.insert(key, community);
            }
        });

    // Copy remaining elements from parent1
    keys.iter().skip(crossover_points.1).for_each(|&key| {
        if let Some(&community) = parent1.get(&key) {
            child.insert(key, community);
        }
    });

    child
}

pub fn optimized_mutate(partition: &mut Partition, graph: &Graph, mutation_rate: f64) {
    let mut rng = rand::thread_rng();

    // Convert BTreeMap to a faster hash map for the duration of the mutation
    let partition_size = partition.len();
    let mut fast_partition: HashMap<NodeId, CommunityId> =
        HashMap::with_capacity_and_hasher(partition_size, Default::default());
    fast_partition.extend(partition.iter().map(|(&k, &v)| (k, v)));

    // Pre-calculate nodes to mutate
    let nodes: Vec<NodeId> = fast_partition
        .keys()
        .copied()
        .filter(|_| rng.gen_bool(mutation_rate))
        .collect();

    // Pre-allocate community cache with expected size
    let mut community_cache: HashMap<NodeId, HashMap<CommunityId, usize>> =
        HashMap::with_capacity_and_hasher(nodes.len(), FxBuildHasher::default());

    // Process nodes in batches for better cache locality
    const BATCH_SIZE: usize = 64;
    for node_chunk in nodes.chunks(BATCH_SIZE) {
        for &node in node_chunk {
            let neighbor_communities = community_cache.entry(node).or_insert_with(|| {
                let mut freq = HashMap::default();
                if let Some(neighbors) = graph.adjacency_list.get(&node) {
                    freq.reserve(neighbors.len());
                    for &neighbor in neighbors {
                        if let Some(&community) = fast_partition.get(&neighbor) {
                            *freq.entry(community).or_insert(0) += 1;
                        }
                    }
                }
                freq
            });

            if let Some((&new_community, _)) =
                neighbor_communities.iter().max_by_key(|&(_, count)| count)
            {
                fast_partition.insert(node, new_community);
            }
        }
    }
    // Update
    partition.clear();
    partition.extend(fast_partition.into_iter());
}

pub fn generate_initial_population(graph: &Graph, population_size: usize) -> Vec<Partition> {
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

pub fn crossover(parent1: &Partition, parent2: &Partition) -> Partition {
    let mut rng = rand::thread_rng();
    let keys: Vec<NodeId> = parent1.keys().copied().collect();
    let len = keys.len();
    let (idx1, idx2) = {
        let mut points = [rng.gen_range(0..len), rng.gen_range(0..len)];
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

pub fn mutate(partition: &mut Partition, graph: &Graph) {
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

pub fn last_x_same(vec: &[f64]) -> bool {
    if vec.len() < MAX_GENERATIONS_WITH_SAME_FITNESS {
        return false;
    }

    let last_x = &vec[vec.len() - MAX_GENERATIONS_WITH_SAME_FITNESS..];
    let first_value = last_x[0];

    for &value in &last_x[1..] {
        if (value - first_value).abs() > FITNESS_COMPARISON_EPSILON {
            return false;
        }
    }
    true
}
