//========================================================================
//    This file is part of mocd (Multi-objective Community Detection).
//    Copyright (C) 2024 Guilherme Oliveira Santos
//    This is free software: you can redistribute it and/or modify it
//    under the terms of the GNU GPL3 or any later version.
//========================================================================

#include "../include/graph.h"
#include "../include/gui.h"

#include <stdbool.h>

static uint32_t num_vertices = 1000;
static uint32_t num_nodes = 3000;
static uint32_t community_size = 10;            // Define the number of nodes per community
static uint32_t inter_community_link_prob = 10; // Probability (in %) of inter-community edges
static bool is_directed = false;
static int window_width = 1900;
static int windown_height = 900;

int main(void){
    drawGraph(createDenseGraph(num_nodes, num_nodes, community_size,
            inter_community_link_prob, is_directed), window_width,
            windown_height);
}
