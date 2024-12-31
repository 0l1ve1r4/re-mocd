//! operators/crossover.rs
//! Genetic Algorithm crossover functions
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2024 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::graph::{NodeId, Partition};

use rand::Rng;
use std::collections::BTreeMap;

pub fn optimized_crossover(parent1: &Partition, parent2: &Partition) -> Partition {
    let mut rng = rand::thread_rng();

    // Use Vec for faster sequential access
    let keys: Vec<NodeId> = parent1.keys().copied().collect();
    let len = keys.len();

    // Optimize crossover point selection
    let crossover_points: (usize, usize) = {
        let point1: usize = rng.gen_range(0..len);
        let point2: usize = (point1 + rng.gen_range(1..len / 2)).min(len - 1);
        (point1, point2)
    };

    // Pre-allocate with capacity
    let mut child: BTreeMap<i32, i32> = Partition::new();

    // Copy elements before crossover point from parent1
    keys.iter().take(crossover_points.0).for_each(|&key| {
        if let Some(&community) = parent1.get(&key) {
            child.insert(key, community);
        }
    });

    // Copy elements in crossover region from parent2
    keys.iter()
        .skip(crossover_points.0)
        .take(crossover_points.1 - crossover_points.0)
        .for_each(|&key| {
            if let Some(&community) = parent2.get(&key) {
                child.insert(key, community);
            }
        });

    // Copy remaining elements from parent1
    keys.iter().skip(crossover_points.1).for_each(|&key| {
        if let Some(&community) = parent1.get(&key) {
            child.insert(key, community);
        }
    });

    child
}

#[allow(dead_code)]
pub fn crossover(parent1: &Partition, parent2: &Partition) -> Partition {
    let mut rng = rand::thread_rng();
    let keys: Vec<NodeId> = parent1.keys().copied().collect();
    let len = keys.len();
    let (idx1, idx2) = {
        let mut points = [rng.gen_range(0..len), rng.gen_range(0..len)];
        points.sort();
        (points[0], points[1])
    };

    let mut child = parent1.clone();
    for key in keys.iter().skip(idx1).take(idx2 - idx1) {
        if let Some(&community) = parent2.get(key) {
            child.insert(*key, community);
        }
    }
    child
}
