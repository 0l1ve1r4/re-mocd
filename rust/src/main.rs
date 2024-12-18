use petgraph::graph::{Graph, };

fn main() {
    // Create a sample graph
    let mut graph = Graph::<(), ()>::new();
    let a = graph.add_node(());
    let b = graph.add_node(());
    let c = graph.add_node(());
    let d = graph.add_node(());
    graph.extend_with_edges(&[(a, b), (b, c), (c, d), (d, a)]);

    // Run genetic algorithm
    let (
        best_partition,
        deviations,
        real_fitnesses,
        random_fitnesses,
        best_fitness_history,
        avg_fitness_history,
    ) = genetic_algorithm(&graph, 80, 100);

    // Output the best partition
    println!("Best Partition: {:?}", best_partition);
}