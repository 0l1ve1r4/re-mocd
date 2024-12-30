// This Source Code Form is subject to the terms of The GNU General Public License v3.0
// Copyright 2024 - Guilherme Santos. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use args::AGArgs;
use std::collections::BTreeMap;
use std::env;
use std::fs::{self};
use std::io::Write;
use std::path::Path;
use std::time::{Duration, Instant};

use std::fs::OpenOptions;

mod algorithm;
mod args;
mod graph;
mod operators;
mod pesa_ii;

use crate::graph::Graph;

const OUTPUT_PATH: &str = "output.json";
const OUTPUT_CSV: &str = "mocd_output.csv";

fn save_csv(time_taken: Instant, num_nodes: usize, num_edges: usize, modularity: f64) -> () {
    let elapsed_time = time_taken.elapsed().as_secs_f64();
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: AGArgs = AGArgs::parse(&(env::args().collect()));
    let start: Instant = Instant::now();
    let graph = Graph::from_edgelist(Path::new(&args.file_path))?;
    let reading_time: Duration = start.elapsed();
    let final_output: bool = args.debug;
    let save: bool = args.save_csv;

    let best_partition: BTreeMap<i32, i32>;
    let modularity: f64;
    match args.pesa_ii {
        true => {
            (best_partition, _, modularity) = pesa_ii::genetic_algorithm(&graph, args);
        }
        false => {
            (best_partition, _, modularity) = algorithm::genetic_algorithm(&graph, args);
        }
    }

    let json = serde_json::to_string_pretty(&best_partition)?;
    fs::write(OUTPUT_PATH, json)?;

    if final_output {
        println!(
            "Elapsed AG/Red. {:?}/{:?} | Nodes/Edges: {:?}/{:?}",
            start.elapsed(),
            reading_time,
            graph.num_nodes(),
            graph.num_edges()
        );
    }

    if save {
        save_csv(start, graph.num_nodes(), graph.num_edges(), modularity);
    }

    Ok(())
}
