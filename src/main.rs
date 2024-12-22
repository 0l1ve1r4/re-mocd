// This Source Code Form is subject to the terms of The GNU General Public License v3.0
// Copyright 2024 - Guilherme Santos. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use petgraph::Undirected;
use petgraph::graph::Graph;

use std::env;
use std::fs::exists;
use std::time::{Duration, Instant};

mod algorithm;
mod helper;
mod consts;
mod operators;

fn main() -> () {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        println!("Usage: <program> <file_path> [ -d for debug ]");
        println!("Example: ./my_program ../graphs/artificial/karate.edgelist -d");
        return;
    }

    let file_path: &str = &args[1];
    if !exists(file_path).unwrap_or_else(|err| {
        eprintln!("Failed to check file existence: {}", err);
        false
    }) {
        println!("Graph .edgelist file not found.");
        return;
    }



    let debug_mode: bool = args.len() > 2 && &args[2] == "-d";
    if debug_mode {
        println!("Debug mode: {}", debug_mode);
        println!("File found, proceed with reading...");
    }


    let start: Instant = Instant::now();
    let graph: Graph<(), (), Undirected> = helper::read_graph(file_path);
    let reading_time: Duration = start.elapsed();

    let start: Instant = Instant::now();
    let (
        _, 
        _, 
        _,
        best_history,
        avg_history
    ) = algorithm::genetic_algorithm(&graph, consts::NUM_GENERATIONS, consts::POPULATION_SIZE,);

    let algorithm_time: Duration = start.elapsed();

    if debug_mode {
        helper::debug(reading_time, algorithm_time);
    }

    helper::save_data(best_history, avg_history,algorithm_time, consts::NUM_GENERATIONS);
   
}