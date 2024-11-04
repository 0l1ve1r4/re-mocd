//========================================================================
//    This file is part of mocd (Multi-objective Community Detection).
//    Copyright (C) 2024 Guilherme Oliveira Santos
//    This is free software: you can redistribute it and/or modify it
//    under the terms of the GNU GPL3 or any later version.
//========================================================================

#include "../include/graph.h"
#include "../include/gui.h"

#include <stdbool.h>

static uint32_t num_nodes = 10000;
static bool is_directed = false;

int main(){
    Graph * graph = createGraph(num_nodes, is_directed);

    for (uint32_t i = 0; i< num_nodes; i++)
        if (i%23 == 0)
            graph->addEdge(graph, i/3, i);

    drawGraph(graph, 9000, 9000);

}
