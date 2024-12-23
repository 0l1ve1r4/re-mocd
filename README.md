# Multi-Objective Community Detection

> **Warning:**  
>
> This project is in its early stages, and the algorithm is still being refined. Performance and results may not be optimal yet.

## Overview

This project aims to develop a high-performance genetic algorithm to detect communities in a graph. The algorithm is implemented in **Rust**, a systems programming language known for its speed and efficiency. The goal is to optimize the process of community detection, which can be computationally intensive, and leverage Rust's capabilities to handle large-scale graphs efficiently.

#### Why Rust?

Rust is one of the fastest programming languages available, offering high-performance execution and memory safety without a garbage collector. With a rich ecosystem of libraries, Rust is ideal for computationally heavy tasks like genetic algorithms, where performance is critical.

## Requirements

Before running the algorithm, you'll need an edge list file formatted as follows:

```plaintext
0,1,{'weight': 4}
0,2,{'weight': 5}
0,3,{'weight': 3}
...
0,10,{'weight': 2}
```

The **weight** attribute is optional. If not provided, it can be represented by an empty dictionary: `{}`.

## Running the Algorithm

### Build and Run

1. Clone the repository and navigate to the project folder:
   ```bash
   git clone <repository_url>
   cd <project_directory>
   ```

2. Compile and run the algorithm with your edge list file:
   ```bash
   cargo run --release mygraph.edgelist
   ```

### Debug Mode

To run the algorithm in debug mode, use the `-d` flag:
```bash
cargo run --release mygraph.edgelist -d
```

This will provide additional debug output, useful for troubleshooting and monitoring the algorithm's progress.

## Future Improvements

- Algorithm optimization for better community detection results.
- More advanced graph input/output options.
- Additional configuration and tuning parameters for flexibility.