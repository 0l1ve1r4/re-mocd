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

fn parse_args(args: &Vec<String>) -> (&str, bool) {
    if args.len() < 2 {
        println!("Usage: <program> <file_path> [ -d for debug ]");
        println!("Example: ./my_program ../graphs/artificial/karate.edgelist -d");
        return ("", false);
    }

    let file_path: &str = &args[1];
    if exists(file_path).is_err() {
        println!("Graph .edgelist file not found.");
        return ("", false);
    }

    let debug_mode: bool = args.len() > 2 && &args[2] == "-d";
    if debug_mode {
        println!("[Warning] Debug mode: {} | This may increase algorithm time", debug_mode);
    }

    (file_path, debug_mode)
}

fn main() -> () {
    let args: Vec<String> = env::args().collect();
    
    let (file_path, 
        debug_mode
    ) = parse_args(&args);

    if file_path.is_empty() {
        return;
    }

    let start: Instant = Instant::now();
    let graph: Graph<(), (), Undirected> = helper::read_graph(file_path);
    let reading_time: Duration = start.elapsed();

    let start: Instant = Instant::now();
    let (
        _, 
        best_history, 
        avg_history
    ) = algorithm::genetic_algorithm(
        &graph, 
        consts::NUM_GENERATIONS, 
        consts::POPULATION_SIZE,
        debug_mode
    );

    let algorithm_time: Duration = start.elapsed();
    
    // Debug time elapsed even with debug mode false
    helper::debug(reading_time, algorithm_time);
    helper::save_data(best_history, avg_history, algorithm_time, consts::NUM_GENERATIONS);
}
