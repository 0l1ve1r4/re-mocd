use crate::graph::{CommunityId, Graph, Partition};
use crate::operators::*;

use rand::prelude::*;
use rustc_hash::FxHashMap as HashMap;
use rustc_hash::FxHashSet as HashSet;
use std::time::Instant;
use std::cmp::Ordering;

// Constants for the algorithm
const POPULATION_SIZE: usize = 100;
const MAX_GENERATIONS: usize = 1000;
const CROSSOVER_RATE: f64 = 0.9;
const MUTATION_RATE: f64 = 0.1;
const TOURNAMENT_SIZE: usize = 2;


#[derive(Clone, Debug)]
struct Individual {
    partition: Partition,
    objectives: Vec<f64>,
    rank: usize,
    crowding_distance: f64,
}

impl Individual {
    fn new(partition: Partition) -> Self {
        Individual {
            partition,
            objectives: vec![0.0, 0.0],
            rank: 0,
            crowding_distance: 0.0,
        }
    }

    // Check if this individual dominates another
    fn dominates(&self, other: &Individual) -> bool {
        let mut at_least_one_better = false;
        
        for i in 0..self.objectives.len() {
            if self.objectives[i] > other.objectives[i] {
                return false;
            }
            if self.objectives[i] < other.objectives[i] {
                at_least_one_better = true;
            }
        }
        
        at_least_one_better
    }
}

// Convergence criteria tracking
#[derive(Default)]
pub struct ConvergenceCriteria {
    best_fitness: f64,
    unchanged_count: usize,
    threshold: usize,
}

impl ConvergenceCriteria {
    pub fn new(threshold: usize) -> Self {
        Self {
            best_fitness: f64::NEG_INFINITY,
            unchanged_count: 0,
            threshold,
        }
    }

    pub fn has_converged(&mut self, fitness: f64) -> bool {
        let epsilon = 1e-6;
        let improved = fitness > self.best_fitness + epsilon;
        let same = (fitness - self.best_fitness).abs() < epsilon;

        if improved {
            self.best_fitness = fitness;
            self.unchanged_count = 0;
            false
        } else if same {
            self.unchanged_count += 1;
            self.unchanged_count >= self.threshold
        } else {
            self.unchanged_count = 0;
            false
        }
    }
}

// Generate initial random population
fn generate_population(graph: &Graph, size: usize) -> Vec<Partition> {
    let mut rng = rand::thread_rng();
    let mut population = Vec::with_capacity(size);

    for _ in 0..size {
        let mut partition = Partition::new();
        let community_count = rng.gen_range(2..=graph.nodes.len().min(20));
        
        for &node in &graph.nodes {
            let community = rng.gen_range(0..community_count) as CommunityId;
            partition.insert(node, community);
        }
        
        population.push(partition);
    }

    population
}

// Tournament selection
fn tournament_selection<'a>(population: &'a [Individual], tournament_size: usize) -> &'a Individual {
    let mut rng = rand::thread_rng();
    let mut best = &population[rng.gen_range(0..population.len())];
    
    for _ in 1..tournament_size {
        let candidate = &population[rng.gen_range(0..population.len())];
        if candidate.rank < best.rank || 
           (candidate.rank == best.rank && candidate.crowding_distance > best.crowding_distance) {
            best = candidate;
        }
    }
    
    best
}

// Crossover operator
fn crossover(parent1: &Partition, parent2: &Partition, rate: f64) -> Partition {
    let mut rng = rand::thread_rng();
    
    if rng.gen::<f64>() > rate {
        return parent1.clone();
    }
    
    let mut child = Partition::new();
    let mut community_mapping1: HashMap<CommunityId, CommunityId> = HashMap::default();
    let mut community_mapping2: HashMap<CommunityId, CommunityId> = HashMap::default();
    let mut next_community_id = 0;
    
    // For each node, randomly choose from which parent to inherit
    for &node in parent1.keys() {
        if rng.gen::<bool>() {
            // Inherit from parent1
            let parent_comm = *parent1.get(&node).unwrap();
            let mapped_comm = *community_mapping1.entry(parent_comm).or_insert_with(|| {
                let id = next_community_id;
                next_community_id += 1;
                id
            });
            child.insert(node, mapped_comm);
        } else {
            // Inherit from parent2
            let parent_comm = *parent2.get(&node).unwrap();
            let mapped_comm = *community_mapping2.entry(parent_comm).or_insert_with(|| {
                let id = next_community_id;
                next_community_id += 1;
                id
            });
            child.insert(node, mapped_comm);
        }
    }
    
    child
}

// Mutation operator
fn mutation(partition: &mut Partition, graph: &Graph, rate: f64) {
    let mut rng = rand::thread_rng();
    
    // Count the number of communities
    let mut communities: HashSet<CommunityId> = HashSet::default();
    for &comm in partition.values() {
        communities.insert(comm);
    }
    let community_count = communities.len();
    
    // Mutate nodes with probability based on rate
    for &node in &graph.nodes {
        if rng.gen::<f64>() < rate {
            let community = if rng.gen::<bool>() || community_count < 2 {
                // Assign to an existing random community
                *communities.iter().nth(rng.gen_range(0..communities.len())).unwrap()
            } else {
                // Create a new community
                community_count as CommunityId
            };
            
            partition.insert(node, community);
        }
    }
}

// Fast non-dominated sort
fn fast_non_dominated_sort(population: &mut [Individual]) {
    // Clear previous ranks
    for ind in population.iter_mut() {
        ind.rank = 0;
    }
    
    // Track domination relationships
    let mut fronts: Vec<Vec<usize>> = vec![Vec::new()];
    let mut dominated_by: Vec<Vec<usize>> = vec![Vec::new(); population.len()];
    let mut domination_count: Vec<usize> = vec![0; population.len()];
    
    // Determine domination relationships
    for i in 0..population.len() {
        for j in 0..population.len() {
            if i == j {
                continue;
            }
            
            if population[i].dominates(&population[j]) {
                dominated_by[i].push(j);
            } else if population[j].dominates(&population[i]) {
                domination_count[i] += 1;
            }
        }
        
        // First front
        if domination_count[i] == 0 {
            population[i].rank = 1;
            fronts[0].push(i);
        }
    }
    
    // Find subsequent fronts
    let mut front_idx = 0;
    while !fronts[front_idx].is_empty() {
        let current_front = fronts[front_idx].clone();
        let mut next_front = Vec::new();
        
        for &i in &current_front {
            for &j in &dominated_by[i] {
                domination_count[j] -= 1;
                if domination_count[j] == 0 {
                    population[j].rank = front_idx + 2;
                    next_front.push(j);
                }
            }
        }
        
        front_idx += 1;
        if !next_front.is_empty() {
            fronts.push(next_front);
        } else {
            break;
        }
    }
}

// Calculate crowding distance
fn calculate_crowding_distance(population: &mut [Individual]) {
    if population.is_empty() {
        return;
    }
    
    let n_obj = population[0].objectives.len();
    
    // Initialize crowding distances
    for ind in population.iter_mut() {
        ind.crowding_distance = 0.0;
    }
    
    // Group individuals by rank
    let mut rank_groups: HashMap<usize, Vec<usize>> = HashMap::default();
    for (idx, ind) in population.iter().enumerate() {
        rank_groups.entry(ind.rank).or_insert_with(Vec::new).push(idx);
    }
    
    // Calculate crowding distance for each rank
    for (_rank, indices) in rank_groups {
        if indices.len() <= 1 {
            for &i in &indices {
                population[i].crowding_distance = f64::INFINITY;
            }
            continue;
        }
        
        // Process each objective
        for obj_idx in 0..n_obj {
            // Sort indices by objective value
            let mut sorted = indices.clone();
            sorted.sort_by(|&a, &b| {
                population[a].objectives[obj_idx]
                    .partial_cmp(&population[b].objectives[obj_idx])
                    .unwrap_or(Ordering::Equal)
            });
            
            // Set boundary points to infinity
            population[sorted[0]].crowding_distance = f64::INFINITY;
            population[sorted[sorted.len() - 1]].crowding_distance = f64::INFINITY;
            
            // Calculate distance for interior points
            let obj_min = population[sorted[0]].objectives[obj_idx];
            let obj_max = population[sorted[sorted.len() - 1]].objectives[obj_idx];
            
            if (obj_max - obj_min).abs() > 1e-10 {
                for i in 1..sorted.len() - 1 {
                    let prev_obj = population[sorted[i - 1]].objectives[obj_idx];
                    let next_obj = population[sorted[i + 1]].objectives[obj_idx];
                    
                    population[sorted[i]].crowding_distance += 
                        (next_obj - prev_obj) / (obj_max - obj_min);
                }
            }
        }
    }
}

// Generate offspring population
fn create_offspring(
    population: &[Individual], 
    graph: &Graph,
    crossover_rate: f64,
    mutation_rate: f64,
    tournament_size: usize
) -> Vec<Individual> {
    let mut offspring = Vec::with_capacity(population.len());
    
    while offspring.len() < population.len() {
        let parent1 = tournament_selection(population, tournament_size);
        let parent2 = tournament_selection(population, tournament_size);
        
        let mut child = crossover(&parent1.partition, &parent2.partition, crossover_rate);
        
        mutation(&mut child, graph, mutation_rate);
        
        offspring.push(Individual::new(child));
        
        // If needed, add another child to meet population size
        if offspring.len() < population.len() {
            let mut child2 = crossover(&parent2.partition, &parent1.partition, crossover_rate);
            mutation(&mut child2, graph, mutation_rate);
            offspring.push(Individual::new(child2));
        }
    }
    
    offspring
}

// Max-Q selection criterion
fn max_q_selection(population: &[Individual]) -> &Individual {
    population.iter()
        .max_by(|a, b| {
            let fitness_a = 1.0 - a.objectives[0] - a.objectives[1];
            let fitness_b = 1.0 - b.objectives[0] - b.objectives[1];
            fitness_a.partial_cmp(&fitness_b).unwrap_or(Ordering::Equal)
        })
        .unwrap_or(&population[0])
}

fn max_min_selection<'a>(
    real_front: &'a [Individual],
    random_fronts: &[Vec<Individual>],
) -> &'a Individual {
    let mut best_index = 0;
    let mut max_min_distance = f64::NEG_INFINITY;
    
    for (i, real_sol) in real_front.iter().enumerate() {
        let mut min_distance = f64::INFINITY;
        
        // Find minimum distance to any solution in random fronts
        for random_front in random_fronts {
            for random_sol in random_front {
                let mut dist_squared = 0.0;
                for j in 0..real_sol.objectives.len() {
                    let diff = real_sol.objectives[j] - random_sol.objectives[j];
                    dist_squared += diff * diff;
                }
                let dist = dist_squared.sqrt();
                min_distance = min_distance.min(dist);
            }
        }
        
        // Update best if this solution has larger minimum distance
        if min_distance > max_min_distance {
            max_min_distance = min_distance;
            best_index = i;
        }
    }
    
    &real_front[best_index]
}

// Create a random network with same degree distribution
fn generate_random_network(original: &Graph) -> Graph {
    let mut random_graph = Graph::new();
    
    // Add all nodes
    for &node in &original.nodes {
        random_graph.nodes.insert(node);
    }
    
    // Create stubs list based on node degrees
    let mut stubs = Vec::new();
    let degrees = original.precompute_degress();
    
    for (&node, &degree) in &degrees {
        for _ in 0..degree {
            stubs.push(node);
        }
    }
    
    // Shuffle stubs
    let mut rng = rand::thread_rng();
    stubs.shuffle(&mut rng);
    
    // Connect stubs to form edges
    let mut edge_count = 0;
    for i in (0..stubs.len()).step_by(2) {
        if i + 1 < stubs.len() {
            let src = stubs[i];
            let dst = stubs[i + 1];
            
            // Avoid self-loops
            if src != dst {
                random_graph.add_edge(src, dst);
                edge_count += 1;
            }
        }
    }
    
    // Initialize adjacency lists
    for node in &random_graph.nodes {
        random_graph.adjacency_list.insert(*node, Vec::new());
    }
    
    // Populate adjacency lists
    for (src, dst) in &random_graph.edges {
        random_graph.adjacency_list.get_mut(src).unwrap().push(*dst);
        random_graph.adjacency_list.get_mut(dst).unwrap().push(*src);
    }
    
    if edge_count < original.edges.len() / 2 {
        println!("Warning: Random network has fewer edges than original: {}/{}", 
                 edge_count, original.edges.len() / 2);
    }
    
    random_graph
}

// Main NSGA-II algorithm function
pub fn nsga_ii(graph: &Graph, debug_level: i8) -> (Partition, Vec<f64>) {
    let start_time = Instant::now();
    
    // Precompute node degrees
    let degrees = graph.precompute_degress();
    
    // Generate initial population
    let population = generate_population(graph, POPULATION_SIZE);
    let mut individuals: Vec<Individual> = population
        .into_iter()
        .map(Individual::new)
        .collect();
    
    // Evaluate initial population
    for ind in &mut individuals {
        let metrics = get_fitness(graph, &ind.partition, &degrees, true);
        ind.objectives = vec![metrics.intra, metrics.inter];
    }
    
    // Initialize convergence tracking
    let mut convergence = ConvergenceCriteria::new(10);
    let mut best_fitness_history = Vec::with_capacity(MAX_GENERATIONS);
    
    // Main generational loop
    for generation in 0..MAX_GENERATIONS {
        // Sort by rank and crowding distance
        fast_non_dominated_sort(&mut individuals);
        calculate_crowding_distance(&mut individuals);
        
        // Create offspring population
        let mut offspring = create_offspring(
            &individuals, 
            graph, 
            CROSSOVER_RATE,
            MUTATION_RATE,
            TOURNAMENT_SIZE
        );
        
        // Evaluate offspring
        for ind in &mut offspring {
            let metrics = get_fitness(graph, &ind.partition, &degrees, true);
            ind.objectives = vec![metrics.intra, metrics.inter];
        }
        
        // Combine populations
        individuals.extend(offspring);
        
        // Apply selection (environmental selection)
        fast_non_dominated_sort(&mut individuals);
        calculate_crowding_distance(&mut individuals);
        
        // Sort by rank and crowding distance
        individuals.sort_by(|a, b| {
            a.rank.cmp(&b.rank).then_with(|| {
                b.crowding_distance.partial_cmp(&a.crowding_distance).unwrap_or(Ordering::Equal)
            })
        });
        
        // Reduce to population size
        individuals.truncate(POPULATION_SIZE);
        
        // Track best fitness
        let best_fitness = individuals
            .iter()
            .map(|ind| 1.0 - ind.objectives[0] - ind.objectives[1])
            .fold(f64::NEG_INFINITY, f64::max);
        
        best_fitness_history.push(best_fitness);
        
        // Check for convergence
        if convergence.has_converged(best_fitness) {
            if debug_level >= 1 {
                println!("NSGA-II converged after {} generations", generation + 1);
            }
            break;
        }
        
        if debug_level >= 1 && (generation % 10 == 0 || generation == MAX_GENERATIONS - 1) {
            println!(
                "NSGA-II: Gen {} | Best fitness: {:.4} | First front size: {} | Pop size: {}",
                generation,
                best_fitness,
                individuals.iter().filter(|ind| ind.rank == 1).count(),
                individuals.len()
            );
        }
    }
    
    // Generate random networks for model selection
    let random_fronts = if debug_level >= 1 {
        println!("Generating random networks for model selection...");
        
        let mut fronts = Vec::new();
        for i in 0..3 {  // Generate 3 random networks
            if debug_level >= 2 {
                println!("Generating random network {}/3", i+1);
            }
            
            let random_graph = generate_random_network(graph);
            
            // Run NSGA-II on random graph (with fewer generations)
            let random_population = generate_population(&random_graph, POPULATION_SIZE);
            let mut random_individuals: Vec<Individual> = random_population
                .into_iter()
                .map(Individual::new)
                .collect();
            
            let random_degrees = random_graph.precompute_degress();
            
            for ind in &mut random_individuals {
                let metrics = get_fitness(&random_graph, &ind.partition, &random_degrees, true);
                ind.objectives = vec![metrics.intra, metrics.inter];
            }
            
            // Run for fewer generations on random networks
            for _ in 0..MAX_GENERATIONS {
                fast_non_dominated_sort(&mut random_individuals);
                calculate_crowding_distance(&mut random_individuals);
                
                let mut offspring = create_offspring(
                    &random_individuals, 
                    &random_graph, 
                    CROSSOVER_RATE,
                    MUTATION_RATE,
                    TOURNAMENT_SIZE
                );
                
                for ind in &mut offspring {
                    let metrics = get_fitness(&random_graph, &ind.partition, &random_degrees, true);
                    ind.objectives = vec![metrics.intra, metrics.inter];
                }
                
                random_individuals.extend(offspring);
                fast_non_dominated_sort(&mut random_individuals);
                calculate_crowding_distance(&mut random_individuals);
                
                random_individuals.sort_by(|a, b| {
                    a.rank.cmp(&b.rank).then_with(|| {
                        b.crowding_distance.partial_cmp(&a.crowding_distance).unwrap_or(Ordering::Equal)
                    })
                });
                
                random_individuals.truncate(POPULATION_SIZE);
            }
            
            // Keep only first front
            random_individuals.retain(|ind| ind.rank == 1);
            fronts.push(random_individuals);
        }
        
        fronts
    } else {
        Vec::new()
    };
    
    let first_front: Vec<Individual> = individuals
    .iter()
    .filter(|ind| ind.rank == 1)
    .cloned() // Requires Individual: Clone
    .collect();

    // Now you can borrow the slice from first_front
    let best_solution = if !random_fronts.is_empty() {
        max_min_selection(&first_front, &random_fronts)
    } else {
        max_q_selection(&individuals)
    };

    
    let elapsed = start_time.elapsed();
    if debug_level >= 1 {
        println!("NSGA-II completed in {:.2?}", elapsed);
    }
    
    (best_solution.partition.clone(), best_fitness_history)
}

// Main entry point
pub fn detect_communities(graph: &Graph, debug_level: i8) -> Partition {
    let (partition, _history) = nsga_ii(graph, debug_level);
    partition
}