use rand::seq::SliceRandom;
use rayon::prelude::*;

use crate::graph::{Graph, Partition};
use crate::args::AGArgs;
use crate::operators;
use operators::Metrics;

pub fn genetic_algorithm (
    graph: &Graph,
    args: AGArgs,
) -> (Partition, Vec<f64>, f64) {
    let mut rng = rand::thread_rng();
    let mut population = operators::generate_initial_population(graph, args.pop_size);
    let mut best_fitness_history = Vec::with_capacity(args.num_gens);
    let degress = graph.precompute_degress();

    for generation in 0..args.num_gens {
        let fitnesses: Vec<Metrics> = if args.parallelism {
            population
                .par_iter()
                .map(|partition| operators::calculate_objectives(graph, partition, &degress, true))
                .collect()
        } else {
            population
                .iter()
                .map(|partition| operators::calculate_objectives(graph, partition, &degress, false))
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
            let mut child = operators::crossover(parent1, parent2);
            operators::mutate(&mut child, graph);
            new_population.push(child);
        }
        population = new_population;

        if args.debug {
            println!(
                "Generation: {} \t | Best Fitness: {}",
                generation, best_fitness
            );
        }
    
        if operators::last_x_same(&best_fitness_history){
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
            let metrics = operators::calculate_objectives(graph, partition, &degress, args.parallelism);
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

        assert_eq!(operators::calculate_objectives(&graph, 
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