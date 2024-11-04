//========================================================================
//    This file is part of mocd (Multi-objective Community Detection).
//    Copyright (C) 2024 Guilherme Oliveira Santos
//    This is free software: you can redistribute it and/or modify it
//    under the terms of the GNU GPL3 or any later version.
//========================================================================

#include <stdint.h>

typedef struct GaParams {
    float mut_rate;
    float cross_rate;
    uint32_t pop_size;
} GaParams;

typedef struct GeneticAlgorithm {
    GaParams params;
    int ** pareto;
    int ** population;
} GeneticAlgorithm;
