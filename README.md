<div align="center">
  <img src="res/logo.png" alt="logo" style="width: 40%;"> 

   <strong>Rapid Enhanced Multi-objective Community Detection Algorithm</strong>

![PyPI - Implementation](https://img.shields.io/pypi/implementation/re_mocd)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/re_mocd)
![PyPI - Downloads](https://img.shields.io/pypi/dm/re_mocd)
[![PyPI - Stats](https://img.shields.io/badge/More%20Info-F58025?logo=PyPi)](https://pypistats.org/packages/re_mocd)

</div>

> [!IMPORTANT]  
> **re-mocd** is an open source Rust-based library designed to provide a simple and easy-to-use multi-objective algorithm for efficient and high-performance community detection on graphs. You can use it to make tests on your own graphs, or to make comparisons, be free ☺


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

# Random networks help validate the detection of overlapping communities 
# by serving as a baseline for comparison. These structures appear as significant 
# deviations from the expected behavior in unorganized networks, allowing the method
#  to highlight more complex patterns, such as overlapping communities. 
# However, generating random networks and their Pareto fronts increases the runtime. 
# Higher values ​​are recommended for large numbers of overlapping communities
random_networks = 3

# The main function that will perform the search for communities in the graph. 
# If you want a fast search, keep the number of random networks low.
partition = re_mocd.rmocd(G, random_networks)

# You can see its fitness function using the function below. 
# (check section "Fitness Function" to see how it is calculated).
mod = re_mocd.modularity(G, partition)
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

2. Rename main (it is like this to avoid unused warnings):
   ```bash
   mv cli.rs main.rs
   ```

3. Compile and execute the algorithm:  
   ```bash
   cargo run --release mygraph.edgelist
   ```

### Debug Mode  

Use the `-d` flag for additional debug output:  
```bash
cargo run --release mygraph.edgelist -d
```

---

### Fitness Function

1. **Intra Objective:** Maximize the density of connections within communities:

   $$\text{intra}(C) = 1 - \frac{\sum_{c \in C} |E(c)|}{m}$$

2. **Inter Objective:** Minimize the strength of connections between communities:

   $$\text{inter}(C) = \sum_{c \in C} \left( \frac{\sum_{v \in c} \text{deg}(v)}{2m} \right)^2$$

3. **Modularity Function:** Combines both:

   $$Q(C) = 1 - \text{intra}(C) - \text{inter}(C)$$


These two conflicting objectives balance the density of internal connections and the sparsity of external connections. They are optimized simultaneously.


### Contributing  

Contributions are welcome! Feel free to submit issues, feature requests, or pull requests to improve the project.  

**License:** GPL-3.0 or later  
**Author:** [Guilherme Santos](https://github.com/0l1ve1r4)  