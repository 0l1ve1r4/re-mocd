use serde_json;
use std::env;
use std::fs::exists;
use std::fs::{self};
use std::path::Path;
use std::time::{Duration, Instant};

use std::fs::OpenOptions;
use std::io::Write;

mod algorithm;
mod graph;

use crate::algorithm::genetic_algorithm;
use crate::graph::Graph;

const OUTPUT_PATH: &str = "src/graphs/output/output.json";
const OUTPUT_CSV: &str = "src/graphs/output/mocd_output.csv";

fn parse_args(args: &Vec<String>) -> (&str, bool, bool) {
    if args.len() < 2 {
        println!("Usage: <program> <file_path> [ -d for debug ] [ -s for serial processing ]");
        println!("Example: ./my_program ../graphs/artificial/karate.edgelist -d");
        return ("", false, false);
    }

    let file_path: &str = &args[1];
    if exists(file_path).is_err() {
        println!("Graph .edgelist file not found.");
        return ("", false, false);
    }

    let debug_mode: bool = args.len() > 2 && args[2..].iter().any(|a| a == "-d");
    if debug_mode {
        println!(
            "[Warning] Debug mode: {} | This may increase algorithm time",
            debug_mode
        );
    }

    let serial = args.len() > 2 && args[2..].iter().any(|a| a == "-s");

    (file_path, debug_mode, !serial)
}

fn save_csv(time_taken: Instant, num_nodes: usize, num_edges: usize, modularity: f64) {
    // Calculate the elapsed time in seconds
    let elapsed_time = time_taken.elapsed().as_secs_f64();

    // Open the file in append mode, creating it if it doesn't exist
    let mut file = OpenOptions::new()
        .write(true)
        .append(true)
        .create(true)
        .open(OUTPUT_CSV)
        .expect("Failed to open or create the output CSV file");

    // Write the metrics as a new row in the CSV file
    writeln!(
        file,
        "{:.4},{},{},{:.4}",
        elapsed_time, num_nodes, num_edges, modularity
    )
    .expect("Failed to write to the CSV file");
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    let (file_path, debug_mode, parallel) = parse_args(&args);

    let start: Instant = Instant::now();
    let graph = Graph::from_edgelist(Path::new(file_path))?;
    let reading_time: Duration = start.elapsed();

    let (best_partition, _fitness_history, modularity) =
        genetic_algorithm(&graph, 800, 200, debug_mode, parallel);

    let json = serde_json::to_string_pretty(&best_partition)?;
    fs::write(OUTPUT_PATH, json)?;
    println!(
        "Algorithm time {:?} | Reading time: {:?} | Nodes: {:?} | Edges: {:?}",
        start.elapsed(),
        reading_time,
        graph.num_nodes(),
        graph.num_edges()
    );

    save_csv(start, graph.num_nodes(), graph.num_edges(), modularity);

    Ok(())
}
