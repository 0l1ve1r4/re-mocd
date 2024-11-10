//========================================================================
//    This file is part of mocd (Multi-objective Community Detection).
//    Copyright (C) 2024 Guilherme Oliveira Santos
//    This is free software: you can redistribute it and/or modify it
//    under the terms of the GNU GPL3 or any later version.
//========================================================================

#include "../include/graph.h"

#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <stdint.h>

#define CONNECTING_COMMUNITY_CHANCE 80

static Node * createNode(uint32_t v) {
    Node * node = (Node*)malloc(sizeof(Node));
    node->vertex = v;
    node->next = NULL;
    return node;
}

static void addEdge(Graph* graph, uint32_t src, uint32_t dest) {
    /* Add a edge from src to dest */
    Node * new_node = createNode(dest);
    new_node->next = graph->adj_lists[src];
    graph->adj_lists[src] = new_node;

    /* If is undirected, add an edge from dest to src */
    if(!graph->is_directed) {
        new_node = createNode(src);
        new_node->next = graph->adj_lists[dest];
        graph->adj_lists[dest] = new_node;
    }
}

static void printGraph(Graph* graph) {
    if (graph->is_directed) printf("DIRECTED GRAPH:\n");
    else printf("UNDIRECTED GRAPH:\n");

    printf("Vertex ---> Adjacency List\n");
    for (uint32_t v = 0; v < graph->num_vertices; v++) {
        struct Node* temp = graph->adj_lists[v];
        printf("%d --->", v);
        while (temp) {
            printf(" %d ->", temp->vertex);
            temp = temp->next;
        }
        printf(" NULL\n");
    }
}

Graph * createGraph(uint32_t num_vertices, bool is_directed) {
    struct Graph* graph = malloc(sizeof(struct Graph));
    graph->num_vertices = num_vertices;
    graph->is_directed = is_directed;

    /* array of adjacency lists */
    graph->adj_lists = malloc(num_vertices * sizeof(struct Node*));

    /* Initialize each adjacency list as empty */
    for (uint32_t i = 0; i < num_vertices; i++) {
        graph->adj_lists[i] = NULL;
    }

    graph->addEdge = addEdge;
    graph->print = printGraph;

    return graph;
}

Graph * createDenseGraph(uint32_t num_vertices, uint32_t num_nodes,
                        uint32_t community_size, uint32_t inter_prob,
                        bool is_directed) {
    Graph * graph = createGraph(num_vertices, is_directed);
    for (uint32_t i = 0; i < num_nodes; i++) {
        uint32_t community_start = (i / community_size) * community_size;

        for (uint32_t j = community_start; j <
            community_start + community_size; j++) {
            if (j != i && (rand() % 100) < CONNECTING_COMMUNITY_CHANCE) {
                graph->addEdge(graph, i, j);
            }
        }

        if ((rand() % 100) < inter_prob) {
            uint32_t other_community_start = ((rand() % (num_nodes / community_size)) * community_size);
            uint32_t random_node = other_community_start + (rand() % community_size);
            graph->addEdge(graph, i, random_node);
        }
    }
    return graph;
}
