//========================================================================
//    This file is part of mocd (Multi-objective Community Detection).
//    Copyright (C) 2024 Guilherme Oliveira Santos
//    This is free software: you can redistribute it and/or modify it
//    under the terms of the GNU GPL3 or any later version.
//========================================================================

//========================================================================
// INCLUDES
//========================================================================

#include "../include/gui.h"

//#include <cstdlib>
#include <stdio.h>      // snprintf
#include <stdlib.h>     // malloc
#include <string.h>     // memset
#include <raylib.h>
#include <rlgl.h>
#include <raymath.h>
#include <pthread.h>    // calculateRepulsion threads
#include <sys/time.h>   // calculateRepulsion time spent

//========================================================================
// DEFINES
//========================================================================

#define WINDOW_TITLE "MOCD - Graph Output"
#define NODE_RADIUS 30
#define MAX_FORCE 0.05f
#define REPULSION_DISTANCE 100.0f
#define ATTRACTION_DISTANCE 50.0f
#define NUM_THREADS 8
#define GRID_WIDTH_MULTIPLIER NODE_RADIUS/2
#define GRID_HEIGHT_MULTIPLIER NODE_RADIUS/2
#define REPULSION_STRENGTH 10000.0f

/* Make this variables changes just one time (optimization) */
static Vector2 * global_positions = NULL;
static int selectedNode = -1; // -1 means no node is selected
static int grid_width = 0;
static int grid_height = 0;
Camera2D camera = { 0 };
//========================================================================
// STRUCTS
//========================================================================

typedef struct {
    Vector2 *positions;
    uint32_t start_idx;
    uint32_t end_idx;
    uint32_t num_vertices;
    float repulsion_strength;
} ThreadData;

//========================================================================
// FUNCIONS
//========================================================================

static void _drawGraph(Graph * graph, int screen_width,
            int screen_height);

void drawGraph(Graph * graph, int width, int height){
    const int screen_width = width;
    const int screen_height = height;

        InitWindow(screen_width, screen_height, WINDOW_TITLE);

        camera.zoom = 1.0f;

        int zoom_mode = 0;

        SetTargetFPS(60);
        while (!WindowShouldClose())
        {
            // Update
            //------------------------------------------------------------
            // Translate based on mouse right click
            if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
                Vector2 delta = GetMouseDelta();
                delta = Vector2Scale(delta, -1.0f/camera.zoom);
                camera.target = Vector2Add(camera.target, delta);
            }

            if (zoom_mode == 0) {
                float wheel = GetMouseWheelMove();
                if (wheel != 0) {
                    Vector2 mouseWorldPos =
                        GetScreenToWorld2D(GetMousePosition(), camera);

                    camera.offset = GetMousePosition();
                    camera.target = mouseWorldPos;

                    float scaleFactor = 1.0f + (0.25f*fabsf(wheel));
                    if (wheel < 0) scaleFactor = 1.0f/scaleFactor;
                    camera.zoom = Clamp(camera.zoom*scaleFactor,
                            0.125f, 64.0f);
                }
            }
            //------------------------------------------------------------
            // Draw
            //------------------------------------------------------------
            BeginDrawing();
                ClearBackground(RAYWHITE);

                BeginMode2D(camera);

                _drawGraph(graph, screen_width, screen_height);

                EndMode2D();

                DrawText("Mouse left button drag to move, wheel to zoom",
                        20, 0, 20, DARKGRAY);
                DrawText(TextFormat("FPS: %i", GetFPS()), 20, 25, 20, DARKGRAY);

            EndDrawing();
        }

        CloseWindow();        // Close window and OpenGL context
}

void *calculateRepulsion(void *arg) {
    ThreadData *data = (ThreadData *)arg;
    for (uint32_t i = data->start_idx; i < data->end_idx; i++) {
        Vector2 force = {0.0f, 0.0f};
        for (uint32_t j = 0; j < data->num_vertices; j++) {
            if (i != j) {
                Vector2 diff = Vector2Subtract(data->positions[i], data->positions[j]);
                float dist = Vector2Length(diff) + 0.1f;
                float repulsion = data->repulsion_strength / (dist * dist);
                force = Vector2Add(force, Vector2Scale(Vector2Normalize(diff), repulsion));
            }
        }
        data->positions[i] = Vector2Add(data->positions[i], Vector2Scale(force, 0.01f));
    }
    return NULL;
}

static Vector2 * getNodesPosThread(Graph * graph) {
    float radius = fminf(grid_width, grid_height) * 0.4f;
    Vector2 *positions = (Vector2 *)malloc(graph->num_vertices * sizeof(Vector2));
    if (!positions) {
        fprintf(stderr, "[getNodesPos]: Memory Allocation Failed");
        return NULL;
    }

    srand(time(NULL));

    // Initialize random positions within the grid
    for (uint32_t i = 0; i < graph->num_vertices; i++) {
        float randomX = (float)(rand() % grid_width);
        float randomY = (float)(rand() % grid_height);
        positions[i] = (Vector2){ randomX, randomY };
    }

    // Setup threading for repulsion effect
    pthread_t threads[NUM_THREADS];
    ThreadData thread_data[NUM_THREADS];

    // Main repulsion iterations
    for (uint32_t iter = 0; iter < 100; iter++) {
        uint32_t nodes_per_thread = graph->num_vertices / NUM_THREADS;
        for (int t = 0; t < NUM_THREADS; t++) {
            thread_data[t].positions = positions;
            thread_data[t].start_idx = t * nodes_per_thread;
            thread_data[t].end_idx = (t == NUM_THREADS - 1) ? graph->num_vertices : (t + 1) * nodes_per_thread;
            thread_data[t].num_vertices = graph->num_vertices;
            thread_data[t].repulsion_strength = REPULSION_STRENGTH;
            pthread_create(&threads[t], NULL, calculateRepulsion, &thread_data[t]);
        }

        // Join threads
        for (int t = 0; t < NUM_THREADS; t++) {
            pthread_join(threads[t], NULL);
        }
    }

    return positions;
}

static Vector2 * getNodesPos(Graph * graph){
    float radius = fminf(grid_width, grid_height) * 0.4f;
    Vector2 * positions = (Vector2 *)malloc(graph->num_vertices * sizeof(Vector2));
    if (!positions) {
        fprintf(stderr, "[getNodesPos]: Memory Allocation Failed");
        return NULL;
    }

    // Seed random for initial node positions
    srand(time(NULL));

    for (uint32_t i = 0; i < graph->num_vertices; i++) {
        float randomX = grid_width / 2 + ((rand() % (int)(2 * radius)) - radius);
        float randomY = grid_height / 2 + ((rand() % (int)(2 * radius)) - radius);
        positions[i] = (Vector2){ randomX, randomY };
    }

    for (uint32_t iter = 0; iter < 100; iter++) { // Increase for finer dispersion
        for (uint32_t i = 0; i < graph->num_vertices; i++) {
            Vector2 force = { 0.0f, 0.0f };
            for (uint32_t j = 0; j < graph->num_vertices; j++) {
                if (i != j) {
                    Vector2 diff = Vector2Subtract(positions[i], positions[j]);
                    float dist = Vector2Length(diff) + 0.1f;
                    float repulsion = REPULSION_STRENGTH / (dist * dist); // Inverse square law
                    force = Vector2Add(force, Vector2Scale(Vector2Normalize(diff), repulsion));
                }
            }
            positions[i] = Vector2Add(positions[i], Vector2Scale(force, 0.01f)); // Move node by calculated force
        }
    }

    return positions;
}

static void drawNodes(Graph* graph, Vector2 * positions) {
    Vector2 mousePos = GetScreenToWorld2D(GetMousePosition(), camera);

    if (IsMouseButtonDown(MOUSE_BUTTON_RIGHT)) selectedNode = -1;

    // Check for mouse click on a node
    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        for (uint32_t i = 0; i < graph->num_vertices; i++) {
            if (CheckCollisionPointCircle(mousePos, positions[i], NODE_RADIUS)) {
                selectedNode = i;
                break;
            }
        }
    }

    // Array to keep track of nodes connected to the selected node
    bool *highlightedNodes = (bool *)calloc(graph->num_vertices, sizeof(bool));

    if (selectedNode != -1) {
        highlightedNodes[selectedNode] = true;

        struct Node *temp = graph->adj_lists[selectedNode];
        while (temp) {
            highlightedNodes[temp->vertex] = true;
            temp = temp->next;
        }
    }

    for (uint32_t i = 0; i < graph->num_vertices; i++) {
        Color nodeColor = (selectedNode == -1 || highlightedNodes[i]) ? MAROON : (Color){128, 0, 0, 64};

        // Draw current node
        DrawCircleV(positions[i], NODE_RADIUS, nodeColor);

        char label[10];
        snprintf(label, sizeof(label), "%d", i);
        int textWidth = MeasureText(label, 20);
        DrawText(label, positions[i].x - textWidth / 2, positions[i].y - 10, 20, WHITE);

        // Draw edges with lower opacity if neither end is highlighted
        struct Node* temp = graph->adj_lists[i];
        while (temp) {
            int j = temp->vertex;
            Color edgeColor = (selectedNode == -1 || highlightedNodes[i] || highlightedNodes[j]) ? DARKGRAY : (Color){169, 169, 169, 128};

            Vector2 start = positions[i];
            Vector2 end = positions[j];
            Vector2 direction = Vector2Normalize(Vector2Subtract(end, start));
            Vector2 arrowStart = Vector2Add(start, Vector2Scale(direction, 30)); // Move out of the node
            Vector2 arrowEnd = Vector2Subtract(end, Vector2Scale(direction, 30)); // Move into the destination node

            DrawLineV(arrowStart, arrowEnd, edgeColor);

            if (graph->is_directed) {
                float arrowSize = 10.0f;
                Vector2 perp = (Vector2){-direction.y, direction.x};
                Vector2 left = Vector2Subtract(arrowEnd, Vector2Scale(direction, arrowSize));
                Vector2 right = Vector2Add(left, Vector2Scale(perp, arrowSize / 2));
                left = Vector2Subtract(left, Vector2Scale(perp, arrowSize / 2));

                DrawTriangle(arrowEnd, left, right, edgeColor);
            }
            temp = temp->next;
        }
    }

    free(highlightedNodes); // Clean up
}


static void _drawGraph(Graph * graph, int screen_width, int screen_height){


    if (global_positions == NULL) {
        struct timeval start, stop;
        double secs = 0;
        gettimeofday(&start, NULL);
        // ===============================================================
        grid_width = graph->num_vertices * GRID_WIDTH_MULTIPLIER;
        grid_height = graph->num_vertices * GRID_HEIGHT_MULTIPLIER;
        global_positions = getNodesPosThread(graph); // 10-20x faster.
        // global_positions = getNodesPos(graph);
        // ===============================================================
        gettimeofday(&stop, NULL);
        secs = (double)(stop.tv_usec - start.tv_usec) / 1000000 + (double)(stop.tv_sec - start.tv_sec);
        printf("[_drawGraph]: Time taken %f\n",secs);
    }
    drawNodes(graph, global_positions);
}
