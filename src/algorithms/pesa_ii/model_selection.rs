use crate::algorithms::pesa_ii::Solution;

/// Calculate Euclidean distance between two solutions.
fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

/// Selects a solution from the real Pareto front based on the "max-min" distance criterion.
///
/// - `real_front`: A vector of non-dominated (Pareto) solutions from the real network.
/// - `random_front`: A vector of non-dominated (Pareto) solutions from a random network of the same size.
///
pub fn model_selection_phase<'a>(
    real_front: &'a [Solution],
    random_front: &[Solution],
) -> &'a Solution {
    // Keep track of the best solution and its distance to the random front
    let mut best_solution: Option<&Solution> = None;
    let mut best_distance = f64::MIN;

    // For each solution in the real front, compute the minimum distance
    // to any solution in the random front
    for real_sol in real_front {
        let min_dist_to_random = random_front
            .iter()
            .map(|rand_sol| euclidean_distance(&real_sol.objectives, &rand_sol.objectives))
            .fold(f64::MAX, |acc, val| acc.min(val));

        // If this minimum distance is higher than our current best_distance,
        // update best_solution
        if min_dist_to_random > best_distance {
            best_solution = Some(real_sol);
            best_distance = min_dist_to_random;
        }
    }

    // Return the solution that maximizes the minimum distance
    best_solution.expect("Real Pareto front is empty.")
}
