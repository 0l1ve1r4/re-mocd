//! algorithms/mod.rs
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2024 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

mod algorithm;
mod pesa_ii;

use std::collections::BTreeMap;
use std::collections::HashMap;

use crate::graph::{Graph, Partition};
use crate::utils::args::AGArgs;

const PARALLELISM_MIN_LEN: usize = 150;
const PESA_II_MIN_LEN: usize = 0; // Default algorithm deactivated.

/// Algorithm "smart" selection, based on the graph structure.
pub fn select(graph: &Graph, mut args: AGArgs) -> (Partition, Vec<f64>, f64) {
    if args.debug {
        println!();
        println!(
            "[algorithms/mod.rs]: graph n/e: {}/{}",
            graph.nodes.len(),
            graph.edges.len(),
        );
    }

    match graph.nodes.len() > PARALLELISM_MIN_LEN {
        true => {
            if args.debug {
                println!("[algorithms/mod.rs]: args.parallelism set to true");
            }
            args.parallelism = true;
        }
        false => {
            if args.debug {
                println!("[algorithms/mod.rs]: args.parallelism set to false");
            }
            args.parallelism = false
        }
    }

    match graph.nodes.len() > PESA_II_MIN_LEN {
        true => {
            if args.debug {
                println!("[algorithms/mod.rs]: running algorithm with pesa_ii\n");
            }
            let (best_solution, best_fitness_history, highest_modularity) =
                pesa_ii::run(graph, args);
            (
                normalize_community_ids(best_solution),
                best_fitness_history,
                highest_modularity,
            )
        }
        false => {
            if args.debug {
                println!("[algorithms/mod.rs]: running default algorithm\n");
            }
            let (best_solution, best_fitness_history, highest_modularity) =
                algorithm::run(graph, args);
            (
                normalize_community_ids(best_solution),
                best_fitness_history,
                highest_modularity,
            )
        }
    }
}

fn normalize_community_ids(partition: Partition) -> BTreeMap<i32, i32> {
    let mut new_partition = Partition::new();
    let mut id_mapping = HashMap::new();
    let mut next_id = 0;

    // Create a new mapping for community IDs
    for (node_id, &community_id) in partition.iter() {
        if !id_mapping.contains_key(&community_id) {
            id_mapping.insert(community_id, next_id);
            next_id += 1;
        }
        new_partition.insert(*node_id, *id_mapping.get(&community_id).unwrap());
    }

    new_partition
}
