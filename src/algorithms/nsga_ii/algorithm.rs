use crate::graph::{Graph, Partition};
use crate::operators;

use rand::prelude::*;
use rayon::prelude::*;
use rustc_hash::FxHashMap as HashMap;
use rustc_hash::FxHashSet as HashSet;
use std::time::Instant;
use std::cmp::Ordering;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

// Constants for the algorithm
const POPULATION_SIZE: usize = 100;
const MAX_GENERATIONS: usize = 500;
const CROSSOVER_RATE: f64 = 0.9;
const MUTATION_RATE: f64 = 0.1;
const TOURNAMENT_SIZE: usize = 2;
const ENSEMBLE_SIZE: usize = 4;
const CONVERGENCE_THRESHOLD: usize = 50; // Number of generations with no improvement

#[derive(Clone, Debug)]
struct Individual {
    partition: Partition,
    objectives: Vec<f64>,
    rank: usize,
    crowding_distance: f64,
    fitness: f64, // Pre-computed fitness to avoid recalculation
}

impl Individual {
    fn new(partition: Partition) -> Self {
        Individual {
            partition,
            objectives: vec![0.0, 0.0],
            rank: 0,
            crowding_distance: 0.0,
            fitness: f64::NEG_INFINITY,
        }
    }

    // Check if this individual dominates another
    #[inline(always)]
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

    #[inline(always)]
    fn calculate_fitness(&mut self) {
        self.fitness = 1.0 - self.objectives[0] - self.objectives[1];
    }
}

// Tournament selection with early return
#[inline]
fn tournament_selection<'a>(population: &'a [Individual], tournament_size: usize) -> &'a Individual {
    let mut rng: ThreadRng = thread_rng(); 
    let best_idx: usize = rng.gen_range(0..population.len());
    let mut best: &Individual = &population[best_idx];
    
    for _ in 1..tournament_size {
        let candidate_idx: usize = rng.gen_range(0..population.len());
        let candidate: &Individual = &population[candidate_idx];
        
        if candidate.rank < best.rank || 
           (candidate.rank == best.rank && candidate.crowding_distance > best.crowding_distance) {
            best = candidate;
        }
    }
    
    best
}

// Fast non-dominated sort with optimized data structures and parallelism
fn fast_non_dominated_sort(population: &mut [Individual]) {
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let pop_size = population.len();
    
    // Preallocate fronts
    let mut fronts: Vec<Vec<usize>> = Vec::with_capacity(pop_size / 2);
    fronts.push(Vec::with_capacity(pop_size / 2));
    
    // Store dominated indices in a contiguous buffer with ranges
    let mut dominated_data = Vec::new();
    let mut dominated_indices = Vec::with_capacity(pop_size);
    
    // Use atomic counters for parallel front processing
    let domination_count: Vec<AtomicUsize> = (0..pop_size)
        .map(|_| AtomicUsize::new(0))
        .collect();
    
    // Parallel computation of domination relationships
    let domination_relations: Vec<_> = (0..pop_size)
        .into_par_iter()
        .map(|i| {
            let mut dominated = Vec::with_capacity(20); // Increased initial capacity
            let mut count = 0;
            
            for j in 0..pop_size {
                if i == j { continue; }
                
                if population[i].dominates(&population[j]) {
                    dominated.push(j);
                } else if population[j].dominates(&population[i]) {
                    count += 1;
                }
            }
            
            (dominated, count)
        })
        .collect();
    
    // Build contiguous dominated data and indices
    for (i, (dominated, count)) in domination_relations.into_iter().enumerate() {
        let start = dominated_data.len();
        dominated_data.extend(dominated);
        dominated_indices.push(start..dominated_data.len());
        domination_count[i].store(count, Ordering::Relaxed);
        
        if count == 0 {
            population[i].rank = 1;
            fronts[0].push(i);
        }
    }
    
    // Process fronts in parallel using atomic operations
    let mut front_idx = 0;
    while !fronts[front_idx].is_empty() {
        let current_front = &fronts[front_idx];
        let next_front: Vec<usize> = current_front
            .par_iter()
            .fold(Vec::new, |mut acc, &i| {
                let range = &dominated_indices[i];
                for &j in &dominated_data[range.start..range.end] {
                    // Atomic decrement and check for transition to 0
                    let prev = domination_count[j].fetch_sub(1, Ordering::Relaxed);
                    if prev == 1 {
                        acc.push(j);
                    }
                }
                acc
            })
            .reduce(Vec::new, |mut a, mut b| {
                a.append(&mut b);
                a
            });
        
        front_idx += 1;
        if !next_front.is_empty() {
            // Assign ranks and store the next front
            for &j in &next_front {
                population[j].rank = front_idx + 1;
            }
            fronts.push(next_front);
        } else {
            break;
        }
    }
}

// Calculate crowding distance with optimized memory usage
fn calculate_crowding_distance(population: &mut [Individual]) {
    if population.is_empty() {
        return;
    }
    
    let n_obj = population[0].objectives.len();
    
    // Reset crowding distances
    for ind in population.iter_mut() {
        ind.crowding_distance = 0.0;
    }
    
    // Group individuals by rank - preallocate with reasonable size
    let mut rank_groups: HashMap<usize, Vec<usize>> = HashMap::with_capacity_and_hasher(10, Default::default());
    for (idx, ind) in population.iter().enumerate() {
        rank_groups
            .entry(ind.rank)
            .or_insert_with(|| Vec::with_capacity(population.len() / 4))
            .push(idx);
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
            sorted.sort_unstable_by(|&a, &b| {
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
                let scale = 1.0 / (obj_max - obj_min);
                for i in 1..sorted.len() - 1 {
                    let prev_obj = population[sorted[i - 1]].objectives[obj_idx];
                    let next_obj = population[sorted[i + 1]].objectives[obj_idx];
                    
                    population[sorted[i]].crowding_distance += (next_obj - prev_obj) * scale;
                }
            }
        }
    }
}

// Create offspring with better parallelization
fn create_offspring(
    population: &[Individual], 
    graph: &Graph,
    crossover_rate: f64,
    mutation_rate: f64,
    tournament_size: usize
) -> Vec<Individual> {
    let pop_size = population.len();
    let mut offspring = Vec::with_capacity(pop_size);
    let num_threads = rayon::current_num_threads();
    let chunk_size = (pop_size + num_threads - 1) / num_threads;
    
    // Use atomic counter for better load balancing
    let offspring_counter = AtomicUsize::new(0);
    
    let thread_offsprings: Vec<Vec<Individual>> = (0..num_threads)
        .into_par_iter()
        .map(|_| {
            let mut local_rng = thread_rng();
            let mut local_offspring = Vec::with_capacity(chunk_size);
            
            while offspring_counter.fetch_add(1, AtomicOrdering::Relaxed) < pop_size {
                // Select unique parents
                let mut parents = Vec::with_capacity(ENSEMBLE_SIZE);
                let mut selected_ids = HashSet::with_capacity_and_hasher(ENSEMBLE_SIZE, Default::default());
                
                let mut attempts = 0;
                while parents.len() < ENSEMBLE_SIZE && attempts < 50 {
                    let parent = tournament_selection(population, tournament_size);
                    if selected_ids.insert(parent.rank) {
                        parents.push(parent);
                    }
                    attempts += 1;
                }
                
                // Fill remaining slots if needed
                while parents.len() < ENSEMBLE_SIZE {
                    parents.push(tournament_selection(population, tournament_size));
                }

                let parent_partitions: Vec<Partition> = parents.iter()
                    .map(|p| p.partition.clone())
                    .collect();
                
                let parent_slice: &[Partition] = &parent_partitions;   
                let should_crossover = local_rng.gen::<f64>() < crossover_rate;
                     
                let mut child = if should_crossover {
                    operators::ensemble_crossover(parent_slice, 1.0)
                } else {
                    parent_partitions[0].clone()
                };
                
                operators::mutation(&mut child, graph, mutation_rate);
                local_offspring.push(Individual::new(child));
            }
            
            local_offspring
        })
        .collect();
    
    // Combine results, only taking what we need
    let mut remaining = pop_size;
    for mut thread_offspring in thread_offsprings {
        let to_take = remaining.min(thread_offspring.len());
        offspring.extend(thread_offspring.drain(..to_take));
        remaining -= to_take;
        if remaining == 0 {
            break;
        }
    }
    
    offspring
}


// Main NSGA-II algorithm function with max-Q selection from Pareto front
pub fn nsga_ii(graph: &Graph, debug_level: i8) -> (Partition, Vec<f64>) {
    let start_time = Instant::now();
    
    // Precompute node degrees once
    let degrees = graph.precompute_degress();
    
    // Generate initial population
    let mut individuals: Vec<Individual> = operators::generate_population(graph, POPULATION_SIZE)
        .into_par_iter()
        .map(Individual::new)
        .collect();
    
    // Evaluate initial population in parallel
    individuals.par_iter_mut().for_each(|ind| {
        let metrics = operators::get_fitness(graph, &ind.partition, &degrees, true);
        ind.objectives = vec![metrics.intra, metrics.inter];
        ind.calculate_fitness();
    });
    
    // Initialize convergence tracking
    let mut stagnation_counter = 0;
    let mut best_fitness = individuals.par_iter().map(|ind| ind.fitness).max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal)).unwrap_or(f64::NEG_INFINITY);
    let mut prev_best_fitness = best_fitness;
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
        
        // Evaluate offspring in parallel
        offspring.par_iter_mut().for_each(|ind| {
            let metrics = operators::get_fitness(graph, &ind.partition, &degrees, true);
            ind.objectives = vec![metrics.intra, metrics.inter];
            ind.calculate_fitness();
        });
        
        // Reserve capacity before extending
        let combined_size = individuals.len() + offspring.len();
        if individuals.capacity() < combined_size {
            individuals.reserve(combined_size - individuals.capacity());
        }
        
        // Combine populations
        individuals.extend(offspring);
        
        // Apply selection (environmental selection)
        fast_non_dominated_sort(&mut individuals);
        calculate_crowding_distance(&mut individuals);
        
        // Sort by rank and crowding distance with unstable sort for speed
        individuals.sort_unstable_by(|a, b| {
            a.rank.cmp(&b.rank).then_with(|| {
                b.crowding_distance.partial_cmp(&a.crowding_distance).unwrap_or(Ordering::Equal)
            })
        });
        
        // Reduce to population size
        individuals.truncate(POPULATION_SIZE);
        
        // Track best fitness more efficiently
        best_fitness = individuals.par_iter()
            .map(|ind| ind.fitness)
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .unwrap_or(f64::NEG_INFINITY);
        
        best_fitness_history.push(best_fitness);
        
        // Check for convergence - improved stagnation detection
        if (best_fitness - prev_best_fitness).abs() < 1e-6 {
            stagnation_counter += 1;
        } else {
            stagnation_counter = 0;
            prev_best_fitness = best_fitness;
        }
        
        if stagnation_counter >= CONVERGENCE_THRESHOLD {
            if debug_level >= 1 {
                println!("NSGA-II converged after {} generations (stagnation for {} generations)", 
                    generation + 1, stagnation_counter);
            }
            break;
        }
        
        if debug_level >= 1 && (generation % 10 == 0 || generation == MAX_GENERATIONS - 1) {
            let first_front_size = individuals.iter().filter(|ind| ind.rank == 1).count();
            println!(
                "NSGA-II: Gen {} | Best fitness: {:.4} | First front size: {} | Pop size: {}",
                generation,
                best_fitness,
                first_front_size,
                individuals.len()
            );
        }
    }
    
    // Extract Pareto front (first non-dominated front)
    let first_front: Vec<Individual> = individuals.iter()
        .filter(|ind| ind.rank == 1)
        .cloned()
        .collect();
    
    // Select solution with maximum Q value from Pareto front
    let best_solution = if !first_front.is_empty() {
        max_q_selection(&first_front)
    } else {
        // Fallback to best in population if no Pareto front found (shouldn't happen)
        println!("Pareto front not found, this should'nt happen, make a issue");
        max_q_selection(&individuals)
    };
    
    let elapsed = start_time.elapsed();
    if debug_level >= 1 {
        println!("NSGA-II completed in {:.2?}", elapsed);
    }
    
    (best_solution.partition.clone(), best_fitness_history)
}


#[inline]
fn max_q_selection(population: &[Individual]) -> &Individual {
    population.iter()
        .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap_or(Ordering::Equal))
        .expect("Empty population in max_q_selection")
}

// Main entry point
pub fn detect_communities(graph: &Graph, debug_level: i8) -> Partition {
    let (partition, _history) = nsga_ii(graph, debug_level);
    partition
}