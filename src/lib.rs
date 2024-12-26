use args::AGArgs;
use pyo3::prelude::*;
use std::collections::BTreeMap;
use std::path::Path;

mod operators;
mod algorithm;
mod pesa_ii;
mod graph;
mod args;

#[pyfunction]
fn run_rmocd(
    file_path: String, 
    parallelism: bool, 
    infinity: bool, 
    single_obj: bool
) -> PyResult<(BTreeMap<i32, i32>, f64)> {
    
    let args: AGArgs = AGArgs {
        file_path,
        num_gens: 100,
        pop_size: 800,
        mut_rate: 0.1,
        cross_rate: 0.9,
        parallelism,
        debug: false,
        infinity,
        single_obj,
    };

    let graph = graph::Graph::from_edgelist(Path::new(&args.file_path))?;

    let (best_partition, modularity) = if args.single_obj {
        let (best_partition, _, modularity) = algorithm::genetic_algorithm(&graph, args);
        (best_partition, modularity)
    } else {
        let (best_partition, _, modularity) = pesa_ii::pesa2_genetic_algorithm(&graph, args);
        (best_partition, modularity)
    };

    Ok((best_partition, modularity))
}



/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn rmocd(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_rmocd, m)?)?;
    Ok(())
}