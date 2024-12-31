//! operators/mod.rs
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2024 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

pub mod crossover;
pub mod metrics;
pub mod mutation;
pub mod objective;
pub mod population;
pub mod selection;

// TODO: Set these mods private and create pub funcs

pub fn crossover() {}
pub fn mutation() {}
pub fn selection() {}

pub fn get_fitness() {}
pub fn get_population() {}