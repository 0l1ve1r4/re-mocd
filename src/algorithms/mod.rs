//! algorithms/mod.rs
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2024 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

pub mod algorithm;
pub mod pesa_ii;

/// Make a smart selection of the algorithm, based on the algorithm structure,[
/// e.g. if x nodes > max, apply parallelism, if has y edges, use pesa_ii.
pub fn select_algorithm() -> () {
    todo!()

}