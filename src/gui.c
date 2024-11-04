#include "../include/gui.h"

#include <stdio.h>  // snprintf
#include <stdlib.h> // malloc
#include <string.h> //memset
#include <raylib.h>
#include <rlgl.h>
#include <raymath.h>
#include <pthread.h>


#define WINDOW_TITLE "MOCD - Graph Output"
#define NODE_RADIUS 30
#define MAX_FORCE 0.05f
#define REPULSION_DISTANCE 100.0f
#define ATTRACTION_DISTANCE 50.0f

/* Get positions just one time (optimization) */
static Vector2 * global_positions = NULL;

// static int grid_width, grid_height;

static void drawGrid(void);
static void _drawGraph(Graph * graph, int screen_width,
            int screen_height);

void drawGraph(Graph * graph, int width, int height){
    const int screen_width = width;
    const int screen_height = height;

        InitWindow(screen_width, screen_height, WINDOW_TITLE);

        Camera2D camera = { 0 };
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

                drawGrid();
                _drawGraph(graph, screen_width, screen_height);

                EndMode2D();

                DrawText("Mouse left button drag to move, wheel to zoom",
                        20, 0, 20, DARKGRAY);
                DrawText(TextFormat("FPS: %i", GetFPS()), 20, 25, 20, DARKGRAY);

            EndDrawing();
        }

        CloseWindow();        // Close window and OpenGL context
}

static void drawGrid(void) {
    rlPushMatrix();
        rlTranslatef(0, 25 * 50, 0);
        rlRotatef(90, 1, 0, 0);
        DrawGrid(100, 50);
    rlPopMatrix();
}

static Vector2 * getNodesPos(Graph * graph, int screen_width, int screen_height) {
    Vector2 *positions = (Vector2 *)malloc(graph->num_vertices * sizeof(Vector2));
    if (!positions) {
        fprintf(stderr, "[getNodesPos]: Memory Allocation Failed");
        return NULL;
    }

    for (uint32_t i = 0; i < graph->num_vertices; i++) {
        positions[i] = (Vector2) {
            rand() % screen_width,
            rand() % screen_height
        };
    }

    Vector2 *forces = (Vector2 *)calloc(graph->num_vertices, sizeof(Vector2));
    if (!forces) {
        fprintf(stderr, "[getNodesPos]: Memory Allocation for Forces Failed");
        free(positions);
        return NULL;
    }

    for (int iteration = 0; iteration < 1000; iteration++) {
        // Reset forces for each iteration
        memset(forces, 0, graph->num_vertices * sizeof(Vector2));

        // Calculate repulsive forces
        for (uint32_t i = 0; i < graph->num_vertices; i++) {
            Vector2 pos_i = positions[i];
            for (uint32_t j = i + 1; j < graph->num_vertices; j++) {
                Vector2 pos_j = positions[j];
                Vector2 direction = Vector2Subtract(pos_j, pos_i);
                float distance = Vector2Length(direction);

                if (distance < REPULSION_DISTANCE) {
                    direction = Vector2Normalize(direction);
                    float forceMagnitude = MAX_FORCE * (REPULSION_DISTANCE - distance);
                    forces[i] = Vector2Add(forces[i], Vector2Scale(direction, forceMagnitude));
                    forces[j] = Vector2Subtract(forces[j], Vector2Scale(direction, forceMagnitude));
                }
            }
        }

        // Calculate attractive forces
        for (uint32_t i = 0; i < graph->num_vertices; i++) {
            struct Node* temp = graph->adj_lists[i];
            Vector2 pos_i = positions[i];

            while (temp) {
                int j = temp->vertex;
                Vector2 pos_j = positions[j];
                Vector2 direction = Vector2Subtract(pos_j, pos_i);
                float distance = Vector2Length(direction);

                if (distance < ATTRACTION_DISTANCE) {
                    direction = Vector2Normalize(direction);
                    float forceMagnitude = MAX_FORCE * (ATTRACTION_DISTANCE - distance);
                    forces[i] = Vector2Subtract(forces[i], Vector2Scale(direction, forceMagnitude));
                }

                temp = temp->next;
            }
        }

        for (uint32_t i = 0; i < graph->num_vertices; i++) {
            positions[i] = Vector2Add(positions[i], forces[i]);

            positions[i].x = fminf(fmaxf(positions[i].x, NODE_RADIUS), screen_width - NODE_RADIUS);
            positions[i].y = fminf(fmaxf(positions[i].y, NODE_RADIUS), screen_height - NODE_RADIUS);
        }
    }

    free(forces);
    return positions;
}

static void drawNodes(Graph* graph, Vector2 * positions){
    for (uint32_t i = 0; i < graph->num_vertices; i++) {
        DrawCircleV(positions[i], NODE_RADIUS, MAROON);

        char label[10];
        snprintf(label, sizeof(label), "%d", i);
        int textWidth = MeasureText(label, 20);
        DrawText(label, positions[i].x - textWidth / 2, positions[i].y - 10, 20, WHITE);

        struct Node* temp = graph->adj_lists[i];
        while (temp) {
            int j = temp->vertex;

            Vector2 start = positions[i];
            Vector2 end = positions[j];
            Vector2 direction = Vector2Normalize(Vector2Subtract(end, start));
            Vector2 arrowStart = Vector2Add(start, Vector2Scale(direction, 30)); // Move out of the node
            Vector2 arrowEnd = Vector2Subtract(end, Vector2Scale(direction, 30)); // Move into the destination node

            DrawLineV(arrowStart, arrowEnd, DARKGRAY);

            if (graph->is_directed) {
                float arrowSize = 10.0f;
                Vector2 perp = (Vector2){-direction.y, direction.x};
                Vector2 left = Vector2Subtract(arrowEnd, Vector2Scale(direction, arrowSize));
                Vector2 right = Vector2Add(left, Vector2Scale(perp, arrowSize / 2));
                left = Vector2Subtract(left, Vector2Scale(perp, arrowSize / 2));

                DrawTriangle(arrowEnd, left, right, DARKGRAY);
            }
            temp = temp->next;
        }
    }
}

static void _drawGraph(Graph * graph, int screen_width, int screen_height){
    if (global_positions == NULL) global_positions = getNodesPos(graph, screen_width, screen_height);
    drawNodes(graph, global_positions);
}
