use crate::args::AGArgs;
use rayon::prelude::*;
use std::time::Instant;
use crate::graph::{Graph, Partition};

use crate::operators;

#[derive(Clone, Debug)]
struct Solution {
    partition: Partition,
    objectives: Vec<f64>,
}

// hypergrid region
#[derive(Clone, Debug)]
struct HyperBox {
    solutions: Vec<Solution>,
    coordinates: Vec<usize>,
}

impl Solution {
    fn dominates(&self, other: &Solution) -> bool {
        let mut has_better = false;
        for (self_obj, other_obj) in self.objectives.iter().zip(other.objectives.iter()) {
            if self_obj < other_obj {
                return false;
            }
            if self_obj > other_obj {
                has_better = true;
            }
        }
        has_better
    }
}

pub fn pesa2_genetic_algorithm(
    graph: &Graph,
    args: AGArgs,
) -> (Partition, Vec<f64>, f64) {
    let mut rng = rand::thread_rng();
    let mut archive = Vec::new();  // Archive of non-dominated solutions
    let mut population = operators::generate_initial_population(graph, args.pop_size);
    let mut best_fitness_history = Vec::with_capacity(args.num_gens);
    let degrees = graph.precompute_degress();
    
    const GRID_DIVISIONS: usize = 8;
    const MAX_ARCHIVE_SIZE: usize = 100; 

    for generation in 0..args.num_gens {
        let generation_start = Instant::now();

        // Calculate objectives for current population
        let solutions: Vec<Solution> = if args.parallelism {
            population
                .par_iter()
                .map(|partition| {
                    let metrics = operators::calculate_objectives(graph, partition, &degrees, true);
                    Solution {
                        partition: partition.clone(),
                        objectives: vec![metrics.modularity, metrics.inter, metrics.intra],
                    }
                })
                .collect()
        } else {
            population
                .iter()
                .map(|partition| {
                    let metrics = operators::calculate_objectives(graph, partition, &degrees, false);
                    Solution {
                        partition: partition.clone(),
                        objectives: vec![metrics.modularity, metrics.inter, metrics.intra],
                    }
                })
                .collect()
        };

        if archive.len() >= MAX_ARCHIVE_SIZE {
            let len = archive.len();
            archive = archive.split_at(len - MAX_ARCHIVE_SIZE).1.to_vec();

        }

        update_archive_parallel(solutions, &mut archive);

        let hyperboxes: Vec<HyperBox>;
        if args.parallelism {
            hyperboxes = create_hypergrid_parallel(&archive, GRID_DIVISIONS);
        }

        else {
            hyperboxes = create_hypergrid(&archive, GRID_DIVISIONS)
        }
        
        // Record best fitness (using modularity as primary objective)
        let best_fitness = archive
            .iter()
            .map(|s| s.objectives[0])
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        best_fitness_history.push(best_fitness);

        // Selection and reproduction
        let mut new_population = Vec::with_capacity(args.pop_size);
        while new_population.len() < args.pop_size {
            // Select parents using PESA-II selection method
            let parent1: &Solution;
            let parent2: &Solution;
            
            if args.parallelism {
                parent1 = select_from_hypergrid_parallel(&hyperboxes,&mut rng);
                parent2 = select_from_hypergrid_parallel(&hyperboxes,&mut rng);

            }

            else {
                parent1 = select_from_hypergrid(&hyperboxes, &mut rng);
                parent2 = select_from_hypergrid(&hyperboxes, &mut rng);              
            }
            
            // Crossover and mutation
            let mut child = operators::crossover(&parent1.partition, &parent2.partition);
            operators::mutate(&mut child, graph);
            new_population.push(child);
        }
        
        population = new_population;

        // Early stopping condition
        if operators::last_x_same(&best_fitness_history) {
            if args.debug {
                println!("[Optimization]: Max Local, breaking...");
            }
            break;
        }
        let generation_duration = generation_start.elapsed();

        if args.debug {
            println!(
                "Generation: {} | Best Fitness: {} | Duration: {:?} | Archive Size: {}",
                generation,
                best_fitness,
                generation_duration,
                archive.len()
            );
}
    }

    // Find best solution from archive (using modularity as primary objective)
    let best_solution = archive
        .iter()
        .max_by(|a, b| a.objectives[0].partial_cmp(&b.objectives[0]).unwrap())
        .unwrap();

    (
        best_solution.partition.clone(),
        best_fitness_history,
        best_solution.objectives[0],
    )
}

fn create_hypergrid(solutions: &[Solution], divisions: usize) -> Vec<HyperBox> {
    let mut hyperboxes: Vec<HyperBox> = Vec::new();
    
    // Skip if solutions is empty
    if solutions.is_empty() {
        return hyperboxes;
    }
    
    // Find min and max values for each objective
    let mut min_values = vec![f64::MAX; solutions[0].objectives.len()];
    let mut max_values = vec![f64::MIN; solutions[0].objectives.len()];
    
    for solution in solutions {
        for (i, &obj) in solution.objectives.iter().enumerate() {
            min_values[i] = min_values[i].min(obj);
            max_values[i] = max_values[i].max(obj);
        }
    }

    // Process each solution
    for solution in solutions {
        let coordinates: Vec<usize> = solution
            .objectives
            .iter()
            .enumerate()
            .map(|(i, &obj)| {
                let normalized = if (max_values[i] - min_values[i]).abs() < f64::EPSILON {
                    0.0
                } else {
                    (obj - min_values[i]) / (max_values[i] - min_values[i])
                };
                (normalized * divisions as f64).min((divisions - 1) as f64) as usize
            })
            .collect();

        // Find existing hyperbox or create new one
        match hyperboxes.iter_mut().find(|hb| hb.coordinates == coordinates) {
            Some(hyperbox) => {
                hyperbox.solutions.push(solution.clone());
            }
            None => {
                hyperboxes.push(HyperBox {
                    solutions: vec![solution.clone()],
                    coordinates: coordinates,
                });
            }
        }
    }

    hyperboxes
}

fn select_from_hypergrid<'a>(hyperboxes: &'a [HyperBox], rng: &mut impl rand::Rng) -> &'a Solution {
    // Select hyperbox with probability inversely proportional to its crowding
    let total_weight: f64 = hyperboxes.iter()
        .map(|hb| 1.0 / (hb.solutions.len() as f64))
        .sum();
    
    let mut random_value = rng.gen::<f64>() * total_weight;
    
    for hyperbox in hyperboxes {
        let weight = 1.0 / (hyperbox.solutions.len() as f64);
        if random_value <= weight {
            // Randomly select a solution from the chosen hyperbox
            return &hyperbox.solutions[rng.gen_range(0..hyperbox.solutions.len())];
        }
        random_value -= weight;
    }
    
    // Fallback to last hyperbox if something goes wrong
    let last_box = hyperboxes.last().unwrap();
    &last_box.solutions[rng.gen_range(0..last_box.solutions.len())]
}

fn create_hypergrid_parallel(solutions: &[Solution], divisions: usize) -> Vec<HyperBox> {
    let mut hyperboxes: Vec<HyperBox> = Vec::new();
    
    // Skip if solutions is empty
    if solutions.is_empty() {
        return hyperboxes;
    }
    
    // Find min and max values for each objective (done sequentially here)
    let mut min_values = vec![f64::MAX; solutions[0].objectives.len()];
    let mut max_values = vec![f64::MIN; solutions[0].objectives.len()];
    
    for solution in solutions {
        for (i, &obj) in solution.objectives.iter().enumerate() {
            min_values[i] = min_values[i].min(obj);
            max_values[i] = max_values[i].max(obj);
        }
    }

    // First, compute all coordinates in parallel:
    // Each solution is mapped to (coordinates, solution) in parallel.
    let coords_with_solutions: Vec<(Vec<usize>, Solution)> = solutions
        .par_iter()
        .map(|solution| {
            let coordinates: Vec<usize> = solution
                .objectives
                .iter() // can also use .par_iter() if objectives are large
                .enumerate()
                .map(|(i, &obj)| {
                    let normalized = if (max_values[i] - min_values[i]).abs() < f64::EPSILON {
                        0.0
                    } else {
                        (obj - min_values[i]) / (max_values[i] - min_values[i])
                    };
                    (normalized * divisions as f64)
                        .min((divisions - 1) as f64)
                        .round() as usize
                })
                .collect();
            (coordinates, solution.clone())
        })
        .collect();
    
    // Sequentially group solutions by their coordinates
    for (coordinates, sol) in coords_with_solutions {
        match hyperboxes.iter_mut().find(|hb| hb.coordinates == coordinates) {
            Some(hyperbox) => hyperbox.solutions.push(sol),
            None => {
                hyperboxes.push(HyperBox {
                    solutions: vec![sol],
                    coordinates,
                });
            }
        }
    }

    hyperboxes
}

fn update_archive_parallel(
    new_solutions: Vec<Solution>,
    archive: &mut Vec<Solution>,
) {
    let chunks = new_solutions.par_chunks(100);  // Process in chunks
    let mut updates: Vec<Solution> = chunks
        .flat_map(|chunk| {
            chunk
                .par_iter()
                .filter(|&solution| {
                    !archive.par_iter().any(|archived| archived.dominates(solution))
                })
                .cloned()
                .collect::<Vec<_>>()
        })
        .collect();

    // Batch process the updates
    if !updates.is_empty() {
        archive.retain(|archived| {
            !updates.par_iter().any(|update| update.dominates(archived))
        });
        archive.extend(updates);
    }
}

/// Parallel version of select_from_hypergrid
fn select_from_hypergrid_parallel<'a>(hyperboxes: &'a [HyperBox], rng: &mut impl rand::Rng) -> &'a Solution {
    // Compute total weight in parallel
    let total_weight: f64 = hyperboxes
        .par_iter()
        .map(|hb| 1.0 / (hb.solutions.len() as f64))
        .sum();
    
    let mut random_value = rng.gen::<f64>() * total_weight;
    
    // Selection remains sequential to handle the cumulative weights
    for hyperbox in hyperboxes {
        let weight = 1.0 / (hyperbox.solutions.len() as f64);
        if random_value <= weight {
            // Randomly select a solution from the chosen hyperbox
            return &hyperbox.solutions[rng.gen_range(0..hyperbox.solutions.len())];
        }
        random_value -= weight;
    }
    
    // Fallback to last hyperbox
    let last_box = hyperboxes.last().unwrap();
    &last_box.solutions[rng.gen_range(0..last_box.solutions.len())]
}