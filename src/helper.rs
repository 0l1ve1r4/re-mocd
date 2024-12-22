// This Source Code Form is subject to the terms of The GNU General Public License v3.0
// Copyright 2024 - Guilherme Santos. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use std::path::Path;
use std::io::{self, BufRead, Write};

use petgraph::Undirected;
use petgraph::graph::{Graph, UnGraph};

use std::fs::File;
use std::time::Duration;

const SAVE_FILE_NAME:  &str = "mocd_output.csv";

pub fn read_graph(file_path: &str) -> UnGraph<(), ()> {
    let mut graph: Graph<(), (), Undirected> = Graph::new_undirected();
    let mut node_indices = std::collections::HashMap::new();

    // Open the file and iterate over its lines
    if let Ok(lines) = read_lines(file_path) {
        for line in lines {
            if let Ok(entry) = line {
                let parts: Vec<&str> = entry.split(',').collect();
                if parts.len() >= 2 {
                    // Safely parse node indices
                    if let (Ok(source), Ok(target)) = (parts[0].parse::<usize>(), parts[1].parse::<usize>()) {
                        // Add nodes to the graph if they don't exist
                        let src_index = *node_indices.entry(source).or_insert_with(|| graph.add_node(()));
                        let tgt_index = *node_indices.entry(target).or_insert_with(|| graph.add_node(()));
                        // Add an edge between the nodes
                        graph.add_edge(src_index, tgt_index, ());
                    }
                }
            }
        }
    }
    graph
}

fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where
    P: AsRef<Path>,
{
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}

pub fn save_data(best_history: Vec<f64>, 
        avg_history: Vec<f64>, 
        algorithm_time: Duration,
        num_generations: usize,
    ) {
    let mut data: Vec<String> = Vec::with_capacity(num_generations);
    for generation in 0..num_generations {
     data.push(format!(
         "{},{},{},{:?}",
         generation,
         best_history[generation],
         avg_history[generation],
         algorithm_time
     ));
 }
 
    let mut file = 
        File::create(SAVE_FILE_NAME)
        .expect("Unable to create file");
    
    writeln!(file, "generation,best_fitness,avg,time").unwrap();
    for line in data {
         writeln!(file, "{}", line).unwrap();
    }
    
    println!("Data saved to generations_data.csv");
}

pub fn debug(reading_time: Duration, algorithm_time: Duration) {
    println!("[Debug Mode]:");
    println!("- Reading time: {:?}", reading_time);
    println!("- Algorithm time: {:?}", algorithm_time);

}

