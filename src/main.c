//========================================================================
//    This file is part of mocd (Multi-objective Community Detection).
//    Copyright (C) 2024 Guilherme Oliveira Santos
//    This is free software: you can redistribute it and/or modify it
//    under the terms of the GNU GPL3 or any later version.
//========================================================================

#include "../include/graph.h"
#include "../include/gui.h"

#include <stdbool.h>
#include <stdlib.h>

static uint32_t num_nodes = 100;
static bool is_directed = false;
static uint32_t community_size = 10;  // Define the number of nodes per community
static uint32_t inter_community_link_prob = 10; // Probability (in %) of inter-community edges

int main(){
    Graph * graph = createGraph(num_nodes, is_directed);

    // Create dense intra-community connections
    for (uint32_t i = 0; i < num_nodes; i++) {
        uint32_t community_start = (i / community_size) * community_size;

        for (uint32_t j = community_start; j < community_start + community_size; j++) {
            if (j != i && (rand() % 100) < 80) { // 80% chance of connecting within the community
                graph->addEdge(graph, i, j);
            }
        }

        if ((rand() % 100) < inter_community_link_prob) {
            uint32_t other_community_start = ((rand() % (num_nodes / community_size)) * community_size);
            uint32_t random_node = other_community_start + (rand() % community_size);
            graph->addEdge(graph, i, random_node);
        }
    }

    drawGraph(graph, 1900, 1100);
}
