//! args.rs
//! Parse arguments from cli or from lib to be a AGArgs struct
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2024 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use std::fs::metadata;

const INFINITY_POP_SIZE: usize = 0x3E8;
const INFINITY_GENERATIONS: usize = 0x2710;

#[derive(Debug)]
#[allow(dead_code)]
pub struct AGArgs {
    pub file_path: String,
    pub num_gens: usize,
    pub pop_size: usize,
    pub mut_rate: f64,
    pub cross_rate: f64,

    pub parallelism: bool,
    pub debug: bool,
    pub infinity: bool,
    pub save_csv: bool,
}

impl AGArgs {
    #[allow(dead_code)]
    pub fn default() -> AGArgs {
        AGArgs {
            file_path: "default.edgelist".to_string(),
            num_gens: 0x1000,
            pop_size: 0x0110,
            mut_rate: 0.6,
            cross_rate: 0.9,
            parallelism: false,
            debug: false,
            infinity: false,
            save_csv: false,
        }
    }

    #[allow(dead_code)]
    pub fn lib_args() -> Self {
        AGArgs {
            file_path: "".to_string(),
            num_gens: 0x3E8,
            pop_size: 0x164,
            mut_rate: 0.3,
            cross_rate: 0.9,
            parallelism: true,
            debug: true,
            infinity: false,
            save_csv: false,
        }
    }

    #[allow(clippy::ptr_arg)]
    pub fn parse(args: &Vec<String>) -> AGArgs {
        if args.len() < 2 && args[0] != "--library"
            || args.iter().any(|a| a == "-h" || a == "--help")
        {
            eprintln!("Usage:");
            eprintln!("\t re_mocd [file_path] [arguments]\n");

            eprintln!("Options:");
            eprintln!("\t -h, --help                Show this message;");
            eprintln!("\t -d, --debug               Show debugs (May increase time running);");
            eprintln!("\t -s, --serial              Serial processing (Disable Parallelism);");
            eprintln!("\t -i, --infinity            Stop the algorithm only when reach a local max (slower, but potentially more accurate);");
            eprintln!("\t -c, --save-csv            Save elapsed Time, graph info and modularity in a .csv;");
            eprintln!("\t -p, --pesa-ii             Run with a PESA-II pareto Front;");
            eprintln!();
            panic!();
        }

        let file_path = &args[1];
        if metadata(file_path).is_err() {
            panic!("Graph .edgelist file not found: {}", file_path);
        }

        let parallelism: bool = !(args.iter().any(|a| a == "-s" || a == "--serial"));
        let debug: bool = args.iter().any(|a| a == "-d" || a == "--debug");
        let infinity: bool = args.iter().any(|a| a == "-i" || a == "--infinity");
        let save_csv: bool = args.iter().any(|a| a == "-c" || a == "--save-csv");

        AGArgs {
            file_path: file_path.to_string(),
            num_gens: if infinity {
                INFINITY_GENERATIONS
            } else {
                0x3E8
            },
            pop_size: if infinity { INFINITY_POP_SIZE } else { 0x164 },
            mut_rate: 0.3,
            cross_rate: 0.9,
            parallelism,
            debug,
            infinity,
            save_csv,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::AGArgs;

    #[test]
    #[should_panic]
    fn test_missing_file() {
        let args = vec!["re_mocd".to_string(), "missing.edgelist".to_string()];
        AGArgs::parse(&args);
    }
}
