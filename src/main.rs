use petgraph::graph::{Graph, UnGraph};
use petgraph::Undirected;

use std::env;
use std::fs::{File, exists};
use std::path::Path;
use std::time::{Duration, Instant};
use std::io::{self, BufRead, Write};

mod algorithm;

const NUM_GENERATIONS: usize = 400;
const POPULATION_SIZE: usize = 100;

fn read_graph(file_path: &str) -> UnGraph<(), ()> {
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

fn debug(reading_time: Duration, algorithm_time: Duration) {
    println!("[Debug Mode]:");
    println!("- Reading time: {:?}", reading_time);
    println!("- Algorithm time: {:?}", algorithm_time);

}

fn save_data(best_history: Vec<f64>, avg_history: Vec<f64>) {
    let mut data: Vec<String> = Vec::with_capacity(NUM_GENERATIONS);
    for generation in 0..NUM_GENERATIONS {
     data.push(format!(
         "{},{},{}",
         generation,
         best_history[generation],
         avg_history[generation]
     ));
 }
 
    let mut file = 
        File::create("generations_data.csv")
        .expect("Unable to create file");
    
    writeln!(file, "generation,best_fitness,avg").unwrap();
    for line in data {
         writeln!(file, "{}", line).unwrap();
    }
    
    println!("Data saved to generations_data.csv");
}

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
    let graph: Graph<(), (), Undirected> = read_graph(file_path);
    let reading_time: Duration = start.elapsed();

    let start: Instant = Instant::now();
    let (_, 
        _, 
        _, 
        _,
        best_history,
        avg_history
    ) = algorithm::genetic_algorithm(&graph, NUM_GENERATIONS, POPULATION_SIZE,);

    let algorithm_time: Duration = start.elapsed();

    if debug_mode {
        debug(reading_time, algorithm_time);
    }

    save_data(best_history, avg_history);

   
}