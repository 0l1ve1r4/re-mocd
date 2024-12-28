use args::AGArgs;
use std::path::Path;
use pyo3::prelude::*;
use std::collections::BTreeMap;

mod operators;
mod algorithm;
mod pesa_ii;
mod graph;
mod args;

#[pyfunction(
    signature = (
        file_path,
        pesa_ii,
        parallelism = true,
        infinity = false,
        debug = false,
    )
)]
fn run (
    file_path: String, 
    pesa_ii: bool,
    parallelism: bool, 
    infinity: bool, 
    debug:  bool,
) -> PyResult<(BTreeMap<i32, i32>, f64)> {
    
    let mut args_vec: Vec<String> = vec!["--library-".to_string(), file_path];
    if infinity { 
        args_vec.push("-i".to_string()); 
    } if !parallelism { 
        args_vec.push("-s".to_string()); 
    } if debug { 
        args_vec.push("-d".to_string()); 
    }

    let args: AGArgs = args::AGArgs::parse(&args_vec);
    if args.debug {
        println!("[DEBUG | ArgsVe]: {:?}", args_vec);
        println!("[DEBUG | AGArgs]: {:?}", args);
    }

    let graph: graph::Graph = graph::Graph::from_edgelist(Path::new(&args.file_path))?;
    let best_partition: BTreeMap<i32, i32>;
    let modularity: f64;

    match pesa_ii {
        false => {(best_partition, _, modularity) = algorithm::genetic_algorithm(&graph, args);}
        true => {(best_partition, _, modularity) = pesa_ii::genetic_algorithm(&graph, args);}
    }
              
    Ok((best_partition, modularity))
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn rmocd(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run, m)?)?;
    Ok(())
}