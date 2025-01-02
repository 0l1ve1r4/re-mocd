//! lib.rs
//! Implements the algorithm to be run as a PyPI python library
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2024 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use std::collections::BTreeMap;
use std::path::Path;

mod algorithms;
mod graph;
pub mod operators;
mod utils;

use graph::{CommunityId, Graph, NodeId, Partition};
use utils::args::AGArgs;

#[pyfunction(
    signature = (
        file_path,
        infinity = false,
        debug = false,
    )
)]
fn from_edglist(file_path: String, infinity: bool, debug: bool) -> PyResult<BTreeMap<i32, i32>> {
    let mut args_vec: Vec<String> = vec!["--library-".to_string(), file_path];
    if infinity {
        args_vec.push("-i".to_string());
    }

    if debug {
        args_vec.push("-d".to_string());
    }

    let args: AGArgs = AGArgs::parse(&args_vec);
    if args.debug {
        println!("[lib.rs]: {:?}", args_vec);
        println!("[lib.rs]: {:?}", args);
    }

    let graph: Graph = Graph::from_edgelist(Path::new(&args.file_path))?;

    let (best_partition, _, _) = algorithms::select(&graph, args);

    Ok(best_partition)
}

#[pyfunction]
fn from_nx(graph: &Bound<'_, PyAny>) -> PyResult<BTreeMap<i32, i32>> {
    let mut graph_struct = Graph::new();

    // Convert EdgeView to list first
    let edges_view = graph.call_method0("edges")?;
    let edges_list = edges_view.call_method0("__iter__")?;

    // Iterate over the edges
    for edge_item in edges_list.try_iter()? {
        let edge = edge_item?;
        let from: NodeId = match edge.get_item(0) {
            Ok(item) => item.extract()?,
            Err(e) => {
                println!("Error getting 'from' node: {:?}", e);
                continue;
            }
        };

        let to: NodeId = match edge.get_item(1) {
            Ok(item) => item.extract()?,
            Err(e) => {
                println!("Error getting 'to' node: {:?}", e);
                continue;
            }
        };

        graph_struct.add_edge(from, to);
    }

    let args: AGArgs = AGArgs::lib_args();
    let (best_partition, _, _) = algorithms::select(&graph_struct, args);

    Ok(best_partition)
}

fn convert_partition(py_partition: &Bound<'_, PyDict>) -> PyResult<Partition> {
    let mut partition = BTreeMap::new();

    for (key, value) in py_partition.iter() {
        let node: NodeId = key.extract()?;
        let community: CommunityId = value.extract()?;

        // Insert into the BTreeMap
        partition.insert(node, community);
    }

    Ok(partition)
}

#[pyfunction]
fn get_modularity(graph: &Bound<'_, PyAny>, partition: &Bound<'_, PyDict>) -> PyResult<f64> {
    let mut graph_struct = Graph::new();

    // Convert EdgeView to list first
    let edges_view = graph.call_method0("edges")?;
    let edges_list = edges_view.call_method0("__iter__")?;

    // Iterate over the edges
    for edge_item in edges_list.try_iter()? {
        let edge = edge_item?;
        let from: NodeId = match edge.get_item(0) {
            Ok(item) => item.extract()?,
            Err(e) => {
                println!("Error getting 'from' node: {:?}", e);
                continue;
            }
        };

        let to: NodeId = match edge.get_item(1) {
            Ok(item) => item.extract()?,
            Err(e) => {
                println!("Error getting 'to' node: {:?}", e);
                continue;
            }
        };

        graph_struct.add_edge(from, to);
    }

    Ok(operators::get_modularity_from_partition(
        &convert_partition(partition).unwrap(),
        &graph_struct,
    ))
}

#[pymodule]
fn re_mocd(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(from_nx, m)?)?;
    m.add_function(wrap_pyfunction!(from_edglist, m)?)?;
    m.add_function(wrap_pyfunction!(get_modularity, m)?)?;
    Ok(())
}
