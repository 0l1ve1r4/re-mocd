//! main.rs
//! Implements the algorithm to be run by command line interface
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2024 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use std::env;
use std::path::Path;
use std::time::Instant;
mod algorithms;
mod graph;
mod operators;
mod utils;

use graph::Graph;
use utils::args::AGArgs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: AGArgs = AGArgs::parse(&(env::args().collect()));
    let start: Instant = Instant::now();
    let graph: Graph = Graph::from_edgelist(Path::new(&args.file_path))?;
    let final_output: bool = args.debug;

    let (best_partition, _, modularity) = algorithms::select(&graph, args);

    if final_output {
        println!("[main.rs] Algorithm Time (s) {:.2?}!", start.elapsed(),);
    }

    utils::saving::to_csv(
        start.elapsed().as_secs_f64(),
        graph.num_nodes(),
        graph.num_edges(),
        modularity,
    );
    _ = utils::saving::to_json(best_partition);

    Ok(())
}
