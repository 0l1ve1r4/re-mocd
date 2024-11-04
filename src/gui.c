#include "../include/gui.h"

#include <stdio.h>  // snprintf
#include <stdlib.h> // malloc
#include <raylib.h>
#include <rlgl.h>
#include <raymath.h>

#define WINDOW_TITLE "MOCD - Graph Output"
#define NODE_RADIUS 30

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

static Vector2 * drawNodes(Graph * graph, int screen_width, int screen_height){
    float radius = fminf(screen_width, screen_height) * 0.4f;
    float angleIncrement = 2 * PI / graph->num_vertices;
    Vector2 * positions = (Vector2 *)malloc(graph->num_vertices * sizeof(Vector2));
    if (!positions) {
        fprintf(stderr, "[drawNodes]: Memory Allocation Failed");
        return NULL;
    }

    // Pass 1: Calculate node positions
    for (uint32_t i = 0; i < graph->num_vertices; i++) {
        float angle = i * angleIncrement;
        positions[i] = (Vector2){
            screen_width / 2 + cosf(angle) * radius,
            screen_height / 2 + sinf(angle) * radius
        };
    }

    return positions;

}

static void drawEdges(Graph* graph, Vector2 * positions){
    for (uint32_t i = 0; i < graph->num_vertices; i++) {
        // Draw node circle
        DrawCircleV(positions[i], NODE_RADIUS, MAROON);

        // Draw node label
        char label[10];
        snprintf(label, sizeof(label), "%d", i);
        int textWidth = MeasureText(label, 20);
        DrawText(label, positions[i].x - textWidth / 2, positions[i].y - 10, 20, WHITE);

        // Draw edges with arrows
        struct Node* temp = graph->adj_lists[i];
        while (temp) {
            int j = temp->vertex;  // Destination node index

            // Calculate direction and positions for arrow
            Vector2 start = positions[i];
            Vector2 end = positions[j];
            Vector2 direction = Vector2Normalize(Vector2Subtract(end, start));
            Vector2 arrowStart = Vector2Add(start, Vector2Scale(direction, 30)); // Move out of the node
            Vector2 arrowEnd = Vector2Subtract(end, Vector2Scale(direction, 30)); // Move into the destination node

            // Draw line for edge
            DrawLineV(arrowStart, arrowEnd, DARKGRAY);

            // Draw arrowhead if directed graph
            if (graph->is_directed) {
                float arrowSize = 10.0f;
                Vector2 perp = (Vector2){-direction.y, direction.x};
                Vector2 left = Vector2Subtract(arrowEnd, Vector2Scale(direction, arrowSize));
                Vector2 right = Vector2Add(left, Vector2Scale(perp, arrowSize / 2));
                left = Vector2Subtract(left, Vector2Scale(perp, arrowSize / 2));

                // Draw arrowhead triangle
                DrawTriangle(arrowEnd, left, right, DARKGRAY);
            }
            temp = temp->next;
        }
    }
}

static void _drawGraph(Graph * graph, int screen_width, int screen_height){
    Vector2 * positions = drawNodes(graph, screen_width, screen_height);
    drawEdges(graph, positions);

    if (positions != NULL) free(positions);
}
