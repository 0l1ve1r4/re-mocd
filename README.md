<div align="center">
  <img src="res/logo.png" alt="logo" style="width: 40%;"> 

   <strong>rapid evolutionary multi-objective community detection</strong>

![PyPI - Implementation](https://img.shields.io/pypi/implementation/re_mocd)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/re_mocd)
![PyPI - Downloads](https://img.shields.io/pypi/dm/re_mocd)
[![PyPI - Stats](https://img.shields.io/badge/More%20Info-F58025?logo=PyPi)](https://pypistats.org/packages/re_mocd)

<hr>

</div>

> [!NOTE]  
> **This project is in its early stages.** Performance and results may not be optimal yet.

## Overview  

**re-mocd** is a Rust-based library designed for efficient and high-performance community detection in graphs. By leveraging the speed and memory safety of Rust, the project aims to handle large-scale graphs while addressing limitations in traditional algorithms, such as Louvain.   

---

## Installation  

### Via PyPI  

Install the library using pip:  
```bash
pip install re-mocd
```

---

## Usage  

### From `networkx.Graph()`  

Using **re-mocd** with a `networkx.Graph()` is simple. For example:  
```python
import networkx as nx 
import re_mocd

# Create a graph
G = nx.Graph([
    (0, 1), (0, 3), (0, 7), 
    (1, 2), (1, 3), (1, 5), 
    (2, 3), 
    (3, 6), 
    (4, 5), (4, 6), 
    (5, 6), 
    (7, 8)
])

# Detect communities
partition = re_mocd.from_nx(G)

# Check modularity
mod = re_mocd.get_modularity(G, partition)
```

### From an Edge List File  

Prepare an edge list file formatted as:  
```plaintext
0,1,{'weight': 4}
0,2,{'weight': 5}
0,3,{'weight': 3}
...
0,10,{'weight': 2}
```

The `weight` attribute is optional and can be omitted (`{}`). Save your `networkx` graph as an edge list:  
```python
nx.write_edgelist(G, file_path, delimiter=",", data=False)
```

Run the algorithm:  
```python
import re_mocd

edgelist_file = "my.edgelist"
partition = re_mocd.from_edglist(edgelist_file)
```

### Examples  

- [Plotting Example](tests/python/example.py)  
- [Comparison with Other Algorithms](tests/python/main.py)  
- [Modularity ring problem](tests/python/benchmarks/ring.py)
- [Single file test](tests/python/benchmarks/single.py)

---

<center>  
<img src="res/example.png" alt="Example Plot" width="600">  
</center>  

---

## Running from Scratch  

### Build and Run  

1. Clone the repository:  
   ```bash
   git clone https://github.com/0l1ve1r4/re_mocd
   cd re_mocd
   ```

2. Compile and execute the algorithm:  
   ```bash
   cargo run --release mygraph.edgelist
   ```

### Debug Mode  

Use the `-d` flag for additional debug output:  
```bash
cargo run --release mygraph.edgelist -d
```

Debug mode helps troubleshoot and monitor the algorithm's progress effectively.  

---

### Contributing  

Contributions are welcome! Feel free to submit issues, feature requests, or pull requests to improve the project.  

**License:** GPL-3.0 or later  
**Author:** [Guilherme Santos](https://github.com/0l1ve1r4)  
