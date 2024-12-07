use petgraph::graph::UnGraph;

// Define a wrapper struct
pub struct Graph<N, E> {
    graph: UnGraph<N, E>,
}

impl<N, E> Graph<N, E> {
    pub fn new() -> Self {
        Graph {
            graph: UnGraph::new_undirected(),
        }
    }

    pub fn print(&self) {
        println!("{:?}", &self);
    }

    pub fn generate_example_graph(&mut self) {
        let g = UnGraph::<i32, ()>::from_edges(&[
            (1, 2), (2, 3), (3, 4),
            (1, 4)
        ]);

        // Set the graph inside the wrapper
        self.graph = g;
    }
}
