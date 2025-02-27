const NUM_RANDOM_NETWORKS: usize = 1;

use crate::graph::{self, Graph, Partition, CommunityId, NodeId};
use crate::utils::args::AGArgs;
use super::evolutionary::evolutionary_phase;
use super::hypergrid::Solution;
use super::model_selection::{max_q_selection, min_max_selection};
use super::reduction::{expand_partition, reduce_graph};

use rustc_hash::FxBuildHasher;
use std::collections::{HashMap, BTreeMap};
use rand::thread_rng;
use rand::seq::SliceRandom as _;

#[derive(Default)]
struct GraphLevel {
    graph: Graph,
    partition: Option<Partition>,  // Alterado para Option
    mapping: BTreeMap<CommunityId, Vec<NodeId>>,
}

/// Generates multiple random networks and combines their solutions
fn generate_random_networks(original: &Graph, num_networks: usize) -> Vec<Graph> {
    (0..num_networks)
        .map(|_| {
            let mut random_graph = graph::Graph {
                nodes: original.nodes.clone(),
                ..Default::default()
            };

            let node_vec: Vec<_> = random_graph.nodes.iter().cloned().collect();
            let num_nodes = node_vec.len();
            let num_edges = original.edges.len();
            let mut rng = thread_rng();
            let mut possible_pairs = Vec::with_capacity(num_nodes * (num_nodes - 1) / 2);

            for i in 0..num_nodes {
                for j in (i + 1)..num_nodes {
                    possible_pairs.push((node_vec[i], node_vec[j]));
                }
            }

            possible_pairs.shuffle(&mut rng);
            let selected_edges = possible_pairs
                .into_iter()
                .take(num_edges)
                .collect::<Vec<_>>();

            for (src, dst) in &selected_edges {
                random_graph.edges.push((*src, *dst));
            }

            for node in &random_graph.nodes {
                random_graph.adjacency_list.insert(*node, Vec::new());
            }

            for (src, dst) in &random_graph.edges {
                random_graph.adjacency_list.get_mut(src).unwrap().push(*dst);
                random_graph.adjacency_list.get_mut(dst).unwrap().push(*src);
            }

            random_graph
        })
        .collect()
}

/// Executes the multi-level community detection algorithm, inspired by the Leiden method.
/// The algorithm consists of three main phases:
///
/// 1. Hierarchical Reduction:
///    - The original graph is progressively coarsened into multiple levels
///    - Each level aggregates nodes into communities based on found partitions
///    - The process continues until reaching a maximum number of levels or when
///      reduction becomes insignificant (< 5% reduction)
///
/// 2. Optimization at Each Level:
///    - At each level, runs the PESA-II evolutionary algorithm
///    - PESA-II simultaneously optimizes:
///      * Internal community modularity 
///      * Inter-community connectivity
///    - Uses a population of solutions and a Pareto archive to maintain best partitions
///    - Considers both real and random networks for comparison
///
/// 3. Hierarchical Refinement:
///    - After finding communities in the most reduced graph
///    - Partitions are progressively expanded back to the original graph
///    - Maintains mapping between communities at different levels
///    - Ensures all nodes are properly assigned to communities
///
/// # Arguments
/// * `graph` - Reference to the original input graph
/// * `args` - Genetic algorithm configuration including:
///   - pop_size: Population size for evolutionary algorithm
///   - num_gens: Maximum number of generations
///   - cross_rate: Crossover rate for genetic operations
///   - mut_rate: Mutation rate for genetic operations
///   - debug: Enable debug output
///
/// # Returns
/// * `Partition` - Final community assignments (node ID -> community ID mapping)
/// * `Vec<f64>` - History of best fitness values across generations
/// * `f64` - Final best fitness value achieved
///
/// # Technical Details
/// - Uses PESA-II for multi-objective optimization
/// - Maintains Pareto archive of non-dominated solutions
/// - Implements adaptive grid-based selection
/// - Handles both real and random network comparisons
/// - Supports early convergence detection
pub fn multi_level_evolutionary(graph: &Graph, args: AGArgs) -> (Partition, Vec<f64>, f64) {
    let mut levels: Vec<GraphLevel> = Vec::new();
    
    let mut current_level = GraphLevel {
        graph: graph.clone(),
        partition: None,
        mapping: BTreeMap::new(),
    };
    
    let mut best_fitness_history = Vec::new();
    let mut final_fitness = 0.0;
    let max_levels = 2;
    
    if args.debug { println!("[run]: Starting hierarchical reduction"); }

    for level in 0..max_levels {
        let degrees = current_level.graph.precompute_degress();
        if args.debug {
            println!("[run]: Level {} - Processing graph with {} nodes and {} edges",
                level, current_level.graph.nodes.len(), current_level.graph.edges.len());
        }

        if current_level.graph.nodes.len() <= 3 {
            if args.debug {
                println!("[run]: Level {} - Hasnt enought nodes.",
                    level);
            }
            break;
        };

        let (archive, level_fitness_history) = evolutionary_phase(&current_level.graph, &args, &degrees);
        
        if archive.is_empty() {
            if args.debug {
                println!("[run]: Empty archive at level {}", level);
            }
            break;
        }
        
        let best_solution = if NUM_RANDOM_NETWORKS == 0 {
            max_q_selection(&archive)
        } else {
            let random_networks = generate_random_networks(&current_level.graph, NUM_RANDOM_NETWORKS);
            let random_archives: Vec<Vec<Solution>> = random_networks
                .iter()
                .map(|random_graph| {
                    let random_degrees = random_graph.precompute_degress();
                    let (random_archive, _) = evolutionary_phase(random_graph, &args, &random_degrees);
                    random_archive
                })
                .collect();
            min_max_selection(&archive, &random_archives)
        };
        
        best_fitness_history.extend(level_fitness_history);
        final_fitness = best_solution.objectives[0];
        
        // Guarda a partição encontrada para este nível
        current_level.partition = Some(best_solution.partition.clone());
        
        if level < max_levels - 1 {
            let current_partition = current_level.partition.as_ref().unwrap();
            let (reduced_graph, community_mapping) = reduce_graph(&current_level.graph, current_partition);
            if args.debug {
            println!("[run]: Level {} - Reduced to {} nodes and {} edges",
                level, reduced_graph.nodes.len(), reduced_graph.edges.len());
            }
            if reduced_graph.num_nodes() >= (current_level.graph.num_nodes()/10) * 9 {
                if args.debug {
                    println!("[run]: Insignificant reduction at level {}, stopping", level);
                }
                break;
            }
            
            levels.push(GraphLevel {
                graph: current_level.graph,
                partition: current_level.partition,
                mapping: community_mapping,
            });
            
            current_level = GraphLevel {
                graph: reduced_graph,
                partition: None,
                mapping: BTreeMap::new(),
            };
        }
    }
    if args.debug {
        println!("[run]: Starting hierarchical expansion");
    }
    let mut final_partition = current_level.partition.unwrap_or_else(|| {
        if args.debug {
            println!("[run]: Warning - Using fallback partition for current level");
        }

        let mut part = Partition::new();
        for &node in &current_level.graph.nodes {
            part.insert(node, node);
        }
        part
    });
    
    if args.debug {
        println!("[run]: Initial partition size: {}", final_partition.len());
    }

    for (i, level) in levels.iter().rev().enumerate() {
        if args.debug {
            println!("[run]: Expanding level {} - Current partition size: {}", i, final_partition.len());
        }
        final_partition = expand_partition(&final_partition, &level.mapping);
        if args.debug {
            println!("[run]: After expansion - Partition size: {}", final_partition.len());
        }
    }
    
    if final_partition.len() != graph.nodes.len() {
        if args.debug {
            println!("[run]: WARNING - Final partition size ({}) doesn't match original graph size ({})",
                final_partition.len(), graph.nodes.len());
        }
        for &node in &graph.nodes {
            if !final_partition.contains_key(&node) {
                final_partition.insert(node, node);
            }
        }
    }
    
    (final_partition, best_fitness_history, final_fitness)
}

/// Main run function that creates both the real and random fronts, then
/// selects the best solution via the chosen criterion.
pub fn single_level(graph: &Graph, args: AGArgs) -> (Partition, Vec<f64>, f64) {
    let degrees: HashMap<i32, usize, FxBuildHasher> = graph.precompute_degress();

    // Phase 1: Evolutionary algorithm returns the Pareto frontier for the real network
    let (archive, best_fitness_history) = evolutionary_phase(graph, &args, &degrees);

    // Phase 2: Selection Model, best solution based on strategy
    let best_solution = if NUM_RANDOM_NETWORKS == 0 {
        // Use Max Q selection
        max_q_selection(&archive)
    } else {
        // Generate multiple random networks and their archives
        let random_networks = generate_random_networks(graph, NUM_RANDOM_NETWORKS);
        let random_archives: Vec<Vec<Solution>> = random_networks
            .iter()
            .map(|random_graph| {
                let random_degrees = random_graph.precompute_degress();
                let (random_archive, _) = evolutionary_phase(random_graph, &args, &random_degrees);
                random_archive
            })
            .collect();

        // Use Min-Max selection with random archives
        min_max_selection(&archive, &random_archives)
    };

    (
        best_solution.partition.clone(),
        best_fitness_history,
        best_solution.objectives[0],
    )
}
