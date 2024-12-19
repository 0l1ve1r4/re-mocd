use petgraph::graph::{Graph, NodeIndex};
use petgraph::Undirected;
use rand::prelude::*;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::f64::INFINITY;

// Type alias
type Partition = HashMap<NodeIndex, usize>;

/// Calculates the modularity, intra-community edges, and inter-community edges.
fn calculate_objectives(
    graph: &Graph<(), (), Undirected>,
    partition: &Partition,
) -> (f64, f64, f64) {
    let total_edges = graph.edge_count() as f64;
    if total_edges == 0.0 {
        return (0.0, 0.0, 0.0);
    }

    let mut intra_sum = 0.0;
    let mut inter = 0.0;
    let total_edges_doubled = 2.0 * total_edges;

    let mut communities: HashMap<usize, HashSet<NodeIndex>> = HashMap::new();
    for (&node, &community) in partition.iter() {
        communities
            .entry(community)
            .or_insert_with(HashSet::new)
            .insert(node);
    }

    for community_nodes in communities.values() {
        let mut community_edges = 0.0;
        let mut community_degree = 0.0;

        for &node in community_nodes {
            let neighbors = graph.neighbors(node);
            let node_degree = neighbors.clone().count() as f64;
            community_degree += node_degree;

            for neighbor in neighbors {
                if community_nodes.contains(&neighbor) {
                    community_edges += 1.0;
                }
            }
        }

        // Adjust for undirected graph
        community_edges /= 2.0;
        intra_sum += community_edges;

        let normalized_degree = community_degree / total_edges_doubled;
        inter += normalized_degree * normalized_degree;
    }

    let intra = 1.0 - (intra_sum / total_edges);
    let mut modularity = 1.0 - intra - inter;
    modularity = modularity.clamp(-1.0, 1.0);

    (modularity, intra, inter)
}

/// Generates the initial population of random partitions.
fn generate_initial_population(
    graph: &Graph<(), (), Undirected>,
    population_size: usize,
) -> Vec<Partition> {
    let mut rng = thread_rng();
    let nodes: Vec<NodeIndex> = graph.node_indices().collect();
    let mut population = Vec::with_capacity(population_size);

    for _ in 0..population_size {
        let partition: Partition = nodes
            .iter()
            .map(|&node| (node, rng.gen_range(0..nodes.len())))
            .collect();
        population.push(partition);
    }

    population
}

/// Performs two-point crossover between two parents.
fn crossover(parent1: &Partition, parent2: &Partition) -> Partition {
    let mut rng = thread_rng();
    let keys: Vec<&NodeIndex> = parent1.keys().collect();
    let len = keys.len();
    let mut idxs = (0..len).collect::<Vec<_>>();
    idxs.shuffle(&mut rng);
    let idx1 = idxs[0].min(idxs[1]);
    let idx2 = idxs[0].max(idxs[1]);

    let mut child = parent1.clone();
    for &key in &keys[idx1..idx2] {
        child.insert(*key, parent2[key]);
    }

    child
}

/// Mutates a partition by changing a node's community to that of a neighbor.
fn mutate(partition: &mut Partition, graph: &Graph<(), (), Undirected>) {
    let mut rng = thread_rng();
    let nodes: Vec<&NodeIndex> = partition.keys().collect();
    let node = nodes.choose(&mut rng).unwrap();

    let neighbors: Vec<NodeIndex> = graph.neighbors(**node).collect();
    if !neighbors.is_empty() {
        let neighbor = neighbors.choose(&mut rng).unwrap();
        partition.insert(**node, *partition.get(neighbor).unwrap());
    }
}

/// Selects the top half of the population based on modularity.
fn selection(
    population: &[Partition],
    fitnesses: &[(f64, f64, f64)],
) -> Vec<Partition> {
    let mut pop_fitness: Vec<(&Partition, f64)> = population
        .iter()
        .zip(fitnesses.iter().map(|f| f.0))
        .collect();

    // Sort by modularity descending
    pop_fitness.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Select top half
    pop_fitness
        .iter()
        .take(population.len() / 2)
        .map(|&(partition, _)| partition.clone())
        .collect()
}

/// Calculates the distance between two fitness tuples.
fn calculate_distance(fitness1: &(f64, f64, f64), fitness2: &(f64, f64, f64)) -> f64 {
    let intra_diff = fitness1.1 - fitness2.1;
    let inter_diff = fitness1.2 - fitness2.2;
    (intra_diff * intra_diff + inter_diff * inter_diff).sqrt()
}

fn generate_random_graph(node_count: usize, edge_count: usize) -> Graph<(), (), Undirected> {
    let mut graph = Graph::<(), (), Undirected>::new_undirected();
    let mut rng = thread_rng();

    // Adiciona os nós
    for _ in 0..node_count {
        graph.add_node(());
    }

    // Adiciona arestas aleatórias
    for _ in 0..edge_count {
        let a = rng.gen_range(0..node_count);
        let b = rng.gen_range(0..node_count);
        if a != b {
            graph.add_edge((a as u32).into(), (b as u32).into(), ());
        }
    }

    graph
}

/// The main genetic algorithm function.
pub fn genetic_algorithm(
    graph: &Graph<(), (), Undirected>,
    generations: usize,
    population_size: usize,
) -> (
    Partition,
    Vec<(Partition, (f64, f64, f64), f64)>,
    Vec<(f64, f64, f64)>,
    Vec<(f64, f64, f64)>,
    Vec<f64>,
    Vec<f64>,
) {
    let mut best_fitness_history = Vec::with_capacity(generations);
    let mut avg_fitness_history = Vec::with_capacity(generations);

    // Generate initial population
    let mut population = generate_initial_population(graph, population_size);

    for _ in 0..generations {
        // Evaluate fitnesses in parallel
        let fitnesses: Vec<(f64, f64, f64)> = population
            .par_iter()
            .map(|partition| calculate_objectives(graph, partition))
            .collect();

        // Record best and average fitness
        let modularity_values: Vec<f64> = fitnesses.iter().map(|f| f.0).collect();
        let best_fitness = modularity_values
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let avg_fitness = modularity_values.iter().sum::<f64>() / modularity_values.len() as f64;
        best_fitness_history.push(best_fitness);
        avg_fitness_history.push(avg_fitness);

        // Selection
        population = selection(&population, &fitnesses);

        // Generate new population
        let mut rng = thread_rng();
        let mut new_population = Vec::with_capacity(population_size);
        while new_population.len() < population_size {
            let parents: Vec<&Partition> = population.choose_multiple(&mut rng, 2).collect();
            let mut child = crossover(parents[0], parents[1]);
            mutate(&mut child, graph);
            new_population.push(child);
        }
        population = new_population;
    }

    // Final evaluation for real network
    let fitnesses: Vec<(f64, f64, f64)> = population
        .par_iter()
        .map(|partition| calculate_objectives(graph, partition))
        .collect();

        let best_modularity = fitnesses
        .iter()
        .map(|f| f.0)
        .fold(f64::NEG_INFINITY, |a, b| a.max(b));

    let pareto_front: Vec<&Partition> = population
        .iter()
        .zip(fitnesses.iter())
        .filter(|(_, fitness)| fitness.0 == best_modularity)
        .map(|(p, _)| p)
        .collect();

    // Run on random network
    let node_count = graph.node_count();
    let edge_count = graph.edge_count();

    let random_graph = generate_random_graph(node_count, edge_count);

    let mut random_population = generate_initial_population(&random_graph, population_size);

    for _ in 0..generations {
        // Evaluate fitness
        let fitnesses: Vec<(f64, f64, f64)> = random_population
            .par_iter()
            .map(|partition| calculate_objectives(&random_graph, partition))
            .collect();

        // Selection
        random_population = selection(&random_population, &fitnesses);

        // Generate new population
        let mut new_population = Vec::with_capacity(population_size);
        let mut rng = thread_rng();
        while new_population.len() < population_size {
            let parents: Vec<&Partition> = random_population.choose_multiple(&mut rng, 2).collect();
            let mut child = crossover(parents[0], parents[1]);
            mutate(&mut child, &random_graph);
            new_population.push(child);
        }
        random_population = new_population;
    }

    // Final evaluation for random network
    let random_fitnesses: Vec<(f64, f64, f64)> = random_population
        .par_iter()
        .map(|partition| calculate_objectives(&random_graph, partition))
        .collect();

    let random_best_modularity = random_fitnesses
        .iter()
        .map(|f| f.0)
        .fold(f64::NEG_INFINITY, |a, b| a.max(b));

    let random_pareto_front: Vec<&(f64, f64, f64)> = random_fitnesses
        .iter()
        .filter(|fitness| fitness.0 == random_best_modularity)
        .collect();

    // Max-Min Distance Selection
    let mut max_deviation = -1.0;
    let mut best_partition = None;
    let mut deviations = Vec::new();

    for (partition, fitness) in pareto_front.iter().zip(fitnesses.iter()) {
        let min_distance = random_pareto_front
            .iter()
            .map(|random_fitness| calculate_distance(fitness, random_fitness))
            .fold(INFINITY, f64::min);

        deviations.push(((*partition).clone(), *fitness, min_distance));

        if min_distance > max_deviation {
            max_deviation = min_distance;
            best_partition = Some((*partition).clone());
        }
    }

    (
        best_partition.unwrap(),
        deviations,
        fitnesses,
        random_fitnesses,
        best_fitness_history,
        avg_fitness_history,
    )
}