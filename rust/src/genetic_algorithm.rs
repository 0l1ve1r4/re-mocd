use petgraph::graph::{Graph, NodeIndex};
use petgraph::Undirected;
use rand::prelude::*;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

/// Calculates the modularity, intra-community connectivity, and inter-community connectivity.
fn calculate_objectives(
    graph: &Graph<(), (), Undirected>,
    partition: &HashMap<NodeIndex, usize>,
) -> (f64, f64, f64) {
    let total_edges = graph.edge_count() as f64;
    if total_edges == 0.0 {
        return (0.0, 0.0, 0.0);
    }

    let mut intra_sum = 0.0;
    let mut inter = 0.0;
    let total_edges_doubled = 2.0 * total_edges;

    // Group nodes into communities
    let mut communities: HashMap<usize, HashSet<NodeIndex>> = HashMap::new();
    for (&node, &community) in partition.iter() {
        communities.entry(community).or_default().insert(node);
    }

    for community_nodes in communities.values() {
        let mut community_edges = 0.0;
        let mut community_degree = 0.0;

        for &node in community_nodes {
            let neighbors: Vec<_> = graph.neighbors(node).collect();
            let node_degree = neighbors.len() as f64;
            community_degree += node_degree;

            for neighbor in neighbors {
                if community_nodes.contains(&neighbor) {
                    community_edges += 1.0;
                }
            }
        }

        // Since the graph is undirected, avoid double-counting
        community_edges /= 2.0;
        intra_sum += community_edges;
        let normalized_degree = community_degree / total_edges_doubled;
        inter += normalized_degree.powi(2);
    }

    let intra = 1.0 - (intra_sum / total_edges);
    let mut modularity = 1.0 - intra - inter;
    modularity = modularity.clamp(-1.0, 1.0);

    (modularity, intra, inter)
}

/// Generates the initial population of partitions.
fn generate_initial_population(
    graph: &Graph<(), (), Undirected>,
    population_size: usize,
) -> Vec<HashMap<NodeIndex, usize>> {
    let mut population = Vec::new();
    let nodes: Vec<_> = graph.node_indices().collect();
    let mut rng = thread_rng();

    for _ in 0..population_size {
        let mut partition = HashMap::new();
        for &node in nodes.iter() {
            let community = rng.gen_range(0..nodes.len());
            partition.insert(node, community);
        }
        population.push(partition);
    }
    population
}

/// Performs two-point crossover between two parent partitions.
fn crossover(
    parent1: &HashMap<NodeIndex, usize>,
    parent2: &HashMap<NodeIndex, usize>,
) -> HashMap<NodeIndex, usize> {
    let keys: Vec<_> = parent1.keys().cloned().collect();
    let mut rng = thread_rng();
    let idxs = rand::seq::index::sample(&mut rng, keys.len(), 2).into_vec();
    let (idx1, idx2) = (idxs[0].min(idxs[1]), idxs[0].max(idxs[1]));
    let mut child = parent1.clone();

    for i in idx1..idx2 {
        if let Some(&community) = parent2.get(&keys[i]) {
            child.insert(keys[i], community);
        }
    }
    child
}

/// Mutates a partition by assigning a node to the community of a random neighbor.
fn mutate(
    partition: &mut HashMap<NodeIndex, usize>,
    graph: &Graph<(), (), Undirected>,
) {
    let mut rng = thread_rng();
    let nodes: Vec<_> = partition.keys().cloned().collect();
    let node = nodes.choose(&mut rng).unwrap();
    let neighbors: Vec<_> = graph.neighbors(*node).collect();

    if !neighbors.is_empty() {
        let neighbor = neighbors.choose(&mut rng).unwrap();
        if let Some(&community) = partition.get(neighbor) {
            partition.insert(*node, community);
        }
    }
}

/// Selects the top half of the population based on modularity.
fn selection(
    population: &[HashMap<NodeIndex, usize>],
    fitnesses: &[(f64, f64, f64)],
) -> Vec<HashMap<NodeIndex, usize>> {
    let mut combined: Vec<(&(f64, f64, f64), &HashMap<NodeIndex, usize>)> =
        fitnesses.iter().zip(population.iter()).collect();
    combined.sort_by(|a, b| b.0 .0.partial_cmp(&a.0 .0).unwrap());
    combined
        .iter()
        .take(population.len() / 2)
        .map(|&(_, partition)| partition.clone())
        .collect()
}

/// Checks if solution1 dominates solution2.
fn dominates(solution1: &(f64, f64, f64), solution2: &(f64, f64, f64)) -> bool {
    let s1 = [solution1.0, solution1.1, solution1.2];
    let s2 = [solution2.0, solution2.1, solution2.2];
    s1.iter().zip(s2.iter()).all(|(a, b)| a <= b) && s1.iter().zip(s2.iter()).any(|(a, b)| a < b)
}

/// Calculates the crowding distances for a Pareto front.
fn compute_crowding_distance(front: &[(f64, f64, f64)]) -> Vec<f64> {
    let size = front.len();
    if size == 0 {
        return vec![];
    }

    let mut distances = vec![0.0; size];
    let num_objectives = 3;

    for i in 0..num_objectives {
        let mut idxs: Vec<usize> = (0..size).collect();
        idxs.sort_by(|&a, &b| front[a][i].partial_cmp(&front[b][i]).unwrap());
        let min_value = front[idxs[0]][i];
        let max_value = front[idxs[size - 1]][i];

        if (max_value - min_value).abs() < std::f64::EPSILON {
            continue;
        }

        distances[idxs[0]] = std::f64::INFINITY;
        distances[idxs[size - 1]] = std::f64::INFINITY;

        for j in 1..size - 1 {
            distances[idxs[j]] +=
                (front[idxs[j + 1]][i] - front[idxs[j - 1]][i]) / (max_value - min_value);
        }
    }
    distances
}

/// Selects individuals using Pareto dominance and crowding distance.
fn fitness_based_selection(
    population: &[HashMap<NodeIndex, usize>],
    fitnesses: &[(f64, f64, f64)],
    num_selected: usize,
) -> Vec<HashMap<NodeIndex, usize>> {
    let mut selected = Vec::new();
    let mut remaining: Vec<usize> = (0..population.len()).collect();

    while !remaining.is_empty() && selected.len() < num_selected {
        let mut current_front = Vec::new();
        for &i in &remaining {
            let mut dominated = false;
            for &j in &remaining {
                if i != j && dominates(&fitnesses[j], &fitnesses[i]) {
                    dominated = true;
                    break;
                }
            }
            if !dominated {
                current_front.push(i);
            }
        }

        if selected.len() + current_front.len() <= num_selected {
            selected.extend(current_front.iter().cloned());
        } else {
            let front_fitnesses: Vec<_> = current_front.iter().map(|&i| fitnesses[i]).collect();
            let crowding_distances = compute_crowding_distance(&front_fitnesses);
            let mut idx_distances: Vec<_> = current_front
                .iter()
                .zip(crowding_distances.iter())
                .collect();
            idx_distances.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
            let slots_remaining = num_selected - selected.len();
            selected.extend(
                idx_distances
                    .iter()
                    .take(slots_remaining)
                    .map(|(&&idx, _)| idx),
            );
            break;
        }
        remaining.retain(|i| !current_front.contains(i));
    }

    selected
        .iter()
        .map(|&i| population[i].clone())
        .collect()
}

/// Calculates the distance between two solutions.
fn calculate_distance(fitness1: &(f64, f64, f64), fitness2: &(f64, f64, f64)) -> f64 {
    let intra_diff = fitness1.1 - fitness2.1;
    let inter_diff = fitness1.2 - fitness2.2;
    (intra_diff.powi(2) + inter_diff.powi(2)).sqrt()
}

/// The main genetic algorithm function with Max-Min Distance selection.
fn genetic_algorithm(
    graph: &Graph<(), (), Undirected>,
    generations: usize,
    population_size: usize,
) -> (
    HashMap<NodeIndex, usize>,
    Vec<(HashMap<NodeIndex, usize>, (f64, f64, f64), f64)>,
    Vec<(f64, f64, f64)>,
    Vec<(f64, f64, f64)>,
    Vec<f64>,
    Vec<f64>,
) {
    let mut best_fitness_history = Vec::new();
    let mut avg_fitness_history = Vec::new();

    // Real network
    let mut real_population = generate_initial_population(graph, population_size);

    for generation in 0..generations {
        // Evaluate fitnesses in parallel
        let fitnesses: Vec<_> = real_population
            .par_iter()
            .map(|partition| calculate_objectives(graph, partition))
            .collect();

        let modularity_values: Vec<_> = fitnesses.iter().map(|fitness| fitness.0).collect();
        let best_fitness = modularity_values
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let avg_fitness = modularity_values.iter().sum::<f64>() / modularity_values.len() as f64;
        best_fitness_history.push(best_fitness);
        avg_fitness_history.push(avg_fitness);

        // Selection
        real_population = selection(&real_population, &fitnesses);

        // Generate new population
        let mut new_population = Vec::new();
        let mut rng = thread_rng();
        while new_population.len() < population_size {
            let parents: Vec<_> = real_population.choose_multiple(&mut rng, 2).collect();
            let mut child = crossover(parents[0], parents[1]);
            mutate(&mut child, graph);
            new_population.push(child);
        }
        real_population = new_population;

        println!("[Generation]: {}", generation);
    }

    // Final evaluation
    let real_fitnesses: Vec<_> = real_population
        .iter()
        .map(|partition| calculate_objectives(graph, partition))
        .collect();

    let max_modularity = real_fitnesses
        .iter()
        .map(|fitness| fitness.0)
        .fold(f64::NEG_INFINITY, f64::max);

    let real_pareto_front: Vec<_> = real_population
        .iter()
        .zip(real_fitnesses.iter())
        .filter(|(_, fitness)| fitness.0 == max_modularity)
        .map(|(partition, _)| partition.clone())
        .collect();

    // Random network
    use petgraph::rand::random_graph;
    let node_count = graph.node_count();
    let edge_count = graph.edge_count();
    let random_graph = random_graph::<Undirected>(node_count, edge_count);

    let mut random_population = generate_initial_population(&random_graph, population_size);

    for _ in 0..generations {
        let fitnesses: Vec<_> = random_population
            .iter()
            .map(|partition| calculate_objectives(&random_graph, partition))
            .collect();

        random_population = selection(&random_population, &fitnesses);

        // Generate new population
        let mut new_population = Vec::new();
        let mut rng = thread_rng();
        while new_population.len() < population_size {
            let parents: Vec<_> = random_population.choose_multiple(&mut rng, 2).collect();
            let mut child = crossover(parents[0], parents[1]);
            mutate(&mut child, &random_graph);
            new_population.push(child);
        }
        random_population = new_population;
    }

    // Final evaluation for random network
    let random_fitnesses: Vec<_> = random_population
        .iter()
        .map(|partition| calculate_objectives(&random_graph, partition))
        .collect();

    let max_modularity_random = random_fitnesses
        .iter()
        .map(|fitness| fitness.0)
        .fold(f64::NEG_INFINITY, f64::max);

    let random_pareto_front_fitnesses: Vec<_> = random_fitnesses
        .iter()
        .filter(|fitness| fitness.0 == max_modularity_random)
        .cloned()
        .collect();

    // Max-Min Distance Selection
    let mut max_deviation = -1.0;
    let mut best_partition = None;
    let mut deviations = Vec::new();

    for (real_partition, real_fitness) in real_population.iter().zip(real_fitnesses.iter()) {
        let min_distance = random_pareto_front_fitnesses
            .iter()
            .map(|random_fitness| calculate_distance(real_fitness, random_fitness))
            .fold(f64::INFINITY, f64::min);

        deviations.push((
            real_partition.clone(),
            *real_fitness,
            min_distance,
        ));

        if min_distance > max_deviation {
            max_deviation = min_distance;
            best_partition = Some(real_partition.clone());
        }
    }

    (
        best_partition.unwrap(),
        deviations,
        real_fitnesses,
        random_fitnesses,
        best_fitness_history,
        avg_fitness_history,
    )
}