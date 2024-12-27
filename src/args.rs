use std::fs::metadata;

const INFINITY_POP_SIZE: usize = 1000;
const INFINITY_GENERATIONS: usize = 10000;

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
    pub single_obj: bool,
}

impl AGArgs {
    #[allow(dead_code)]
    pub fn default() -> AGArgs {
        AGArgs {
            file_path: "default.edgelist".to_string(),
            num_gens: 800,
            pop_size: 1000,
            mut_rate: 0.6,
            cross_rate: 0.9,
            parallelism: false,
            debug: false,
            infinity: false,
            single_obj: false,
        }
    }

    pub fn parse(args: &Vec<String>) -> AGArgs {
        if  args.len() < 2 && args[0] != "--library-usage" || 
            args.iter().any(|a| a == "-h" || a == "--help") {
            eprintln!("Usage:");
            eprintln!("\t rmocd [file_path] [arguments]\n");

            eprintln!("Options:");
            eprintln!("\t -h, --help                Show this message;");
            eprintln!("\t -d, --debug               Show debugs (May increase time running);");
            eprintln!("\t -s, --serial              Serial processing (Disable Parallelism);");
            eprintln!("\t -i, --infinity            Stop the algorithm only when reach a local max (slower, but potentially more accurate);");
            eprintln!("\t -so --single-objective    Run only with a single objective fitness function (faster, but potentially less accurate);");
            eprintln!();
            panic!();
        }

        let file_path = &args[1];
        if metadata(file_path).is_err() {
            panic!("Graph .edgelist file not found: {}", file_path);
        }

        let parallelism = !(args.iter().any(|a| a == "-s" || a == "--serial"));
        let debug = args.iter().any(|a| a == "-d" || a == "--debug");
        let infinity = args.iter().any(|a| a == "-i" || a == "--infinity");
        let single_obj =  args.iter().any(|a| a == "-so" || a == "--single-objective");

        AGArgs {
            file_path: file_path.to_string(),
            num_gens: if infinity { INFINITY_GENERATIONS } else { 800 },
            pop_size: if infinity { INFINITY_POP_SIZE } else { 100 },
            mut_rate: 0.6,
            cross_rate: 0.9,
            parallelism,
            debug,
            infinity,
            single_obj,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::args::INFINITY_GENERATIONS;

    use super::AGArgs;

    const TEST_FILE_PATH: &str = "res/graphs/artificials/article.edgelist";

    #[test]
    fn test_parse_args() {
        let args = vec!["rmocd".to_string(), TEST_FILE_PATH.to_string(), "-s".to_string(), "--debug".to_string()];
        let parsed = AGArgs::parse(&args);
        assert_eq!(parsed.file_path, TEST_FILE_PATH);
        assert!(!parsed.parallelism);
        assert!(parsed.debug);
        assert!(!parsed.infinity);
    }

    #[test]
    #[should_panic]
    fn test_missing_file() {
        let args = vec!["rmocd".to_string(), "missing.edgelist".to_string()];
        AGArgs::parse(&args);
    }

    #[test]
    fn test_infinity_mode() {
        let args = vec!["rmocd".to_string(), TEST_FILE_PATH.to_string(), "-i".to_string()];
        let parsed = AGArgs::parse(&args);
        assert!(parsed.infinity);
        assert_eq!(parsed.num_gens, INFINITY_GENERATIONS);        
        println!("{:?}", parsed);
    
    }
}