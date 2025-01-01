import argparse
import benchmarks.known_structure as ks
import benchmarks.mu as mu
import benchmarks.single as single
import benchmarks.ring as ring
from example import show_example_plot

def main():
    parser = argparse.ArgumentParser(
        description="A tool for evaluating community detection with re-mocd in various scenarios.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    subparsers = parser.add_subparsers(
        title="Commands",
        dest="command",
        description="Available benchmarking options"
    )

    known_parser = subparsers.add_parser(
        "known",
        help="Run benchmarks using predefined community structures."
    )
    known_parser.add_argument(
        "--use-subprocess",
        action="store_true",
        help="Execute benchmarks using a subprocess to ensure isolation."
    )

    mu_parser = subparsers.add_parser(
        "mu",
        help="Run benchmarks to analyze the impact of mu values on NMI compared to standard algorithms."
    )
    mu_parser.add_argument(
        "--use-subprocess",
        action="store_true",
        help="Execute benchmarks using a subprocess to ensure isolation."
    )

    # Single-file benchmark
    single_parser = subparsers.add_parser(
        "single",
        help="Run benchmarks on a specific input file against standard algorithms."
    )
    single_parser.add_argument(
        "filepath",
        type=str,
        help="Path to the input file for the benchmark."
    )

    # Example: Simple graph demonstration
    example_parser = subparsers.add_parser(
        "example",
        help="Visualize how the algorithm detects communities in a simple graph."
    )

    # Ring benchmark
    ring_parser = subparsers.add_parser(
        "ring",
        help="Demonstrate the algorithm's performance on the 'ring' structure, a challenge for modularity metrics."
    )

    args = parser.parse_args()

    if args.command == "known":
        ks.run(args.use_subprocess)
    elif args.command == "mu":
        mu.run(args.use_subprocess)
    elif args.command == "single":
        single.run(args.filepath)
    elif args.command == "example":
        show_example_plot()
    elif args.command == "ring":
        ring.run()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
