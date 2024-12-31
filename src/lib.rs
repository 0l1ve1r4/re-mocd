//! lib.rs
//! Implements the algorithm to be run as a PyPI python library 
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2024 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html


use pyo3::prelude::*;
use std::collections::BTreeMap;
use std::path::Path;

mod algorithms;
mod graph;
pub mod operators;
mod utils;

use algorithms::algorithm;
use algorithms::pesa_ii;
use graph::Graph;
use utils::args::AGArgs;

#[pyfunction(
    signature = (
        file_path,
        pesa_ii = false,
        parallelism = true,
        infinity = false,
        debug = false,
    )
)]
fn run(
    file_path: String,
    pesa_ii: bool,
    parallelism: bool,
    infinity: bool,
    debug: bool,
) -> PyResult<BTreeMap<i32, i32>> {
    let mut args_vec: Vec<String> = vec!["--library-".to_string(), file_path];
    if infinity {
        args_vec.push("-i".to_string());
    }
    if !parallelism {
        args_vec.push("-s".to_string());
    }
    if debug {
        args_vec.push("-d".to_string());
    }

    let args: AGArgs = AGArgs::parse(&args_vec);
    if args.debug {
        println!("[DEBUG | ArgsVe]: {:?}", args_vec);
        println!("[DEBUG | AGArgs]: {:?}", args);
    }

    let graph: Graph = Graph::from_edgelist(Path::new(&args.file_path))?;
    let best_partition: BTreeMap<i32, i32>;

    match pesa_ii {
        false => {
            (best_partition, _, _) = algorithm::run(&graph, args);
        }
        true => {
            (best_partition, _, _) = pesa_ii::run(&graph, args);
        }
    }

    Ok(best_partition)
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn re_mocd(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run, m)?)?;
    Ok(())
}
