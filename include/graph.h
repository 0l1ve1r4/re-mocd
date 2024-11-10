//========================================================================
//    This file is part of mocd (Multi-objective Community Detection).
//    Copyright (C) 2024 Guilherme Oliveira Santos
//    This is free software: you can redistribute it and/or modify it
//    under the terms of the GNU GPL3 or any later version.
//========================================================================

/*
reference:
    https://www.geeksforgeeks.org/c-program-to-implement-adjacency-list/
    https://www.geeksforgeeks.org/graph-data-structure-and-algorithms/
    */

#ifndef GRAPH_H
#define GRAPH_H

#include <stdint.h>
#include <stdbool.h>

typedef struct Node {
    uint32_t vertex;
    struct Node * next;
} Node;

typedef struct Graph {
    uint32_t num_vertices;
    struct Node ** adj_lists;
    bool is_directed;

    void (*addEdge)(struct Graph *, uint32_t src, uint32_t dest);
    void (*print)(struct Graph *);
} Graph;

Graph * createGraph(uint32_t num_vertices, bool is_directed);

Graph * createDenseGraph(uint32_t num_vertices, uint32_t num_nodes,
                        uint32_t community_size,
                        uint32_t inter_community_link_prob,
                        bool is_directed);

#endif // GRAPH_H
