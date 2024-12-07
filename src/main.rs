use petgraph::graph::UnGraph;


fn main() {
// Create an undirected graph with `i32` nodes and edges with `()` associated data.
let g = UnGraph::<i32, ()>::from_edges(&[
    (1, 2), (2, 3), (3, 4),
    (1, 4)]);

// Output the tree to `graphviz` `DOT` format
println!("{:?}", &g);
// graph {
//     0 [label="\"0\""]
//     1 [label="\"0\""]
//     2 [label="\"0\""]
//     3 [label="\"0\""]
//     1 -- 2
//     3 -- 4
//     2 -- 3
// }
}
