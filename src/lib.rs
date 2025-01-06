//! lib.rs
//! Implements the algorithm to be run as a PyPI python library
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2024 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use std::collections::BTreeMap;
use std::time::Instant;

mod algorithms;
mod graph;
pub mod operators;
mod utils;

use graph::{CommunityId, Graph, NodeId, Partition};
use utils::args::AGArgs;

/// `from_nx`
/// Takes a `networkx.Graph` as input and performs community detection on it.
///
/// ---
/// ### Parameters:
/// - `graph` (networkx.Graph): The graph on which community detection will be performed.
/// - `verbose` (bool, optional): Enables verbose output for debugging and monitoring. Defaults to `False`.
///
#[pyfunction(name = "rmocd")]
#[pyo3(signature = (graph, verbose = false))]
fn rmocd(
    py: Python<'_>,
    graph: &Bound<'_, PyAny>,
    verbose: bool,
) -> PyResult<BTreeMap<i32, i32>> {
    // First get all the data we need while holding the GIL
    let mut edges = Vec::new();
    let start: Instant = Instant::now();

    // Convert EdgeView to list first
    let edges_view = graph.call_method0("edges")?;
    let edges_list = edges_view.call_method0("__iter__")?;

    // Collect edges while we have GIL access
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

        edges.push((from, to));
    }

    let args: AGArgs = AGArgs::lib_args(verbose);
    if args.debug {
        println!("{:?}", args);
    }
    let time_debug: bool = args.debug;

    // Release the GIL
    py.allow_threads(|| {
        let mut graph_struct = Graph::new();

        for (from, to) in edges {
            graph_struct.add_edge(from, to);
        }

        let (best_partition, _, _) = algorithms::rmocd(&graph_struct, args);

        if time_debug {
            println!("[lib.rs] Algorithm Time (s) {:.2?}!", start.elapsed(),);
        }

        Ok(best_partition)
    })
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

/// `get_modularity`
/// Calculates the modularity score of a given graph and its community partitioning based on (Shi, 2012) multi-objective modularity equation.
///
/// ---
/// ### Parameters:
/// - `graph` (networkx.Graph): The graph for which the modularity is to be computed.
/// - `partition` (dict [int, int]): A dictionary mapping nodes to their respective community IDs.
///
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
    m.add_function(wrap_pyfunction!(rmocd, m)?)?;
    m.add_function(wrap_pyfunction!(get_modularity, m)?)?;
    Ok(())
}
