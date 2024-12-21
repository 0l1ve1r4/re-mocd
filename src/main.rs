use petgraph::Undirected;
use petgraph::graph::{Graph, UnGraph};

use std::fs::File;
use std::path::Path;
use std::time::Instant;
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
                // Parse the line into source and target
                let parts: Vec<&str> = entry.split(',').collect();
                if parts.len() >= 2 {
                    let source = parts[0].parse::<usize>().unwrap();
                    let target = parts[1].parse::<usize>().unwrap();

                    // Add nodes to the graph if they don't exist
                    let src_index = *node_indices.entry(source).or_insert_with(|| graph.add_node(()));
                    let tgt_index = *node_indices.entry(target).or_insert_with(|| graph.add_node(()));

                    // Add an edge between the nodes
                    graph.add_edge(src_index, tgt_index, ());
                }
            }
        }
    }
    graph
}

// Helper function to read lines from a file
fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where
    P: AsRef<Path>,
{
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}

fn main() {
    let start: Instant = std::time::Instant::now();

    let file_path = "/home/ol1ve1r4/Desktop/mocd/src/graphs/artificials/karate.edgelist";
    let mut data: Vec<String> = Vec::new();
    let graph: Graph<(), (), Undirected> = 
    read_graph(&file_path);
        
    println!("Reading elapsed: {:.2?}", start.elapsed());

    let start: Instant = std::time::Instant::now();

    let (
        _,
        _,
        _,
        _,
        best_history,
        avg_history,
    ) = algorithm::genetic_algorithm(&graph, NUM_GENERATIONS, POPULATION_SIZE);

    let elapsed = start.elapsed();
    println!("Elapsed: {:.2?}", elapsed);

        // Save in DataFrame
    for generation in 0..NUM_GENERATIONS {
        data.push(format!(
            "{},{},{}",
            generation,
            best_history[generation],
            avg_history[generation],
        ));
    }
    
    let mut file = File::create("generations_data.csv").expect("Unable to create file");
    writeln!(
        file,
        "generation,best_fitness,avg"
    )
    .unwrap();
    for line in data {
        writeln!(file, "{}", line).unwrap();
    }
    println!("Data saved to generations_data.csv");
}
