use args::AGArgs;
use pyo3::prelude::*;
use std::collections::BTreeMap;
use std::path::Path;

mod operators;
mod algorithm;
mod graph;
mod args;

use algorithm::genetic_algorithm;
use graph::Graph;

/// Run the RMOCD algorithm on a given file
///
/// Args:
///     file_path (str): Path to the input file to process
///     parallelism (bool, optional): Enable parallel processing. Defaults to True.
///     infinity (bool, optional): Continue until local maximum is reached. Defaults to False.
///     single_obj (bool, optional): Use single objective fitness function. Defaults to False.
///     debug (bool, optional): Enable debug output. Defaults to False.
///
/// Returns:
///     tuple: A tuple containing:
///         - Dict[int, int]: Mapping of results
///         - float: Final computation value
///
/// Raises:
///     PyValueError: If the file_path is invalid
///     PyRuntimeError: If the algorithm fails to execute
#[pyfunction(
    signature = (
        file_path,
        parallelism = true,
        infinity = false,
        debug = false,
    )
)]
fn run (
    file_path: String, 
    parallelism: bool, 
    infinity: bool, 
    debug: bool,
) -> PyResult<(BTreeMap<i32, i32>, f64)> {
    

    let mut args_vec: Vec<String> = vec!["--library-".to_string()];
    args_vec.push(file_path);

    if infinity {
        args_vec.push("-i".to_string());
    }
    if !parallelism {
        args_vec.push("-s".to_string());
    }
    if debug {
        args_vec.push("-d".to_string());
    }

    let args: AGArgs = args::AGArgs::parse(&args_vec);

    if args.debug {
        println!("[DEBUG | ArgsVe]: {:?}", args_vec);
        println!("[DEBUG | AGArgs]: {:?}", args);
    }

    let graph = Graph::from_edgelist(Path::new(&args.file_path))?;
    let best_partition: BTreeMap<i32, i32>;
    let modularity: f64;

    (best_partition, _, modularity) = genetic_algorithm(&graph, args);
              
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