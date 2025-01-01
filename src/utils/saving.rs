use std::fs::OpenOptions;
use std::fs::{self};
use std::io::Write;

use crate::graph::Partition;

const OUTPUT_PATH: &str = "output.json";
const OUTPUT_CSV: &str = "mocd_output.csv";

pub fn to_csv(elapsed_time: f64, num_nodes: usize, num_edges: usize, modularity: f64) -> () {
    let mut file = OpenOptions::new()
        .append(true)
        .create(true)
        .open(OUTPUT_CSV)
        .expect("Failed to open or create the output CSV file");

    writeln!(
        file,
        "{:.4},{},{},{:.4}",
        elapsed_time, num_nodes, num_edges, modularity
    )
    .expect("Failed to write to the CSV file");
}

pub fn to_json(best_partition: Partition) -> Result<(), std::io::Error> {
    let json = serde_json::to_string_pretty(&best_partition)?;
    fs::write(OUTPUT_PATH, json)?;

    Ok(())
}
