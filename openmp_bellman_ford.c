#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>
#include <omp.h>
#include <sys/stat.h>
#include <string.h>

#define INF 1000000

typedef struct {
    int id;
} node;

typedef struct {
    int src;
    int dest;
    int weight;
} edge;

typedef struct {
    int n; // number of nodes
    int m; // number of edges
    node *nodes;
    edge *edges;
} graph;

void abort_with_error_message(const char* msg) {
    fprintf(stderr, "%s\n", msg);
    exit(1);
}

void read_graph(const char* filename, graph *G) {
    FILE *inputf = fopen(filename, "r");
    if (inputf == NULL) {
        abort_with_error_message("ERROR OCCURRED WHILE READING INPUT FILE");
    }

    fscanf(inputf, "%d %d", &G->n, &G->m);
    G->nodes = (node *)malloc(G->n * sizeof(node));
    G->edges = (edge *)malloc(G->m * sizeof(edge));

    for (int i = 0; i < G->n; i++) {
        G->nodes[i].id = i; // Initialize node ID
    }

    for (int i = 0; i < G->m; i++) {
        int src, dest, weight;
        fscanf(inputf, "%d %d %d", &src, &dest, &weight);
        G->edges[i].src = src;
        G->edges[i].dest = dest;
        G->edges[i].weight = weight;
    }

    fclose(inputf);
}

void write_result(const char *filename, int *dist, int n, bool has_negative_cycle) {
    FILE *outputf = fopen(filename, "w");
    if (outputf == NULL) {
        abort_with_error_message("ERROR OCCURRED WHILE OPENING OUTPUT FILE");
    }
    if (!has_negative_cycle) {
        for (int i = 0; i < n; i++) {
            if (i == n - 1) {
                fprintf(outputf, "%d", dist[i] == INF ? INF : dist[i]); // No space after last number
            } else {
                fprintf(outputf, "%d ", dist[i] == INF ? INF : dist[i]); // Add space after each number
            }
        }
        fprintf(outputf, "\n"); // Newline at the end
    } else {
        fprintf(outputf, "FOUND NEGATIVE CYCLE!\n");
    }
    fclose(outputf);
}

void set_schedule_type(const char *schedule_type, int chunk_size) {
    // Set OpenMP runtime scheduling policy
    if (strcmp(schedule_type, "static") == 0) {
        omp_set_schedule(omp_sched_static, chunk_size); // Static scheduling
    } else if (strcmp(schedule_type, "dynamic") == 0) {
        omp_set_schedule(omp_sched_dynamic, chunk_size); // Dynamic scheduling
    } else if (strcmp(schedule_type, "auto") == 0) {
        omp_set_schedule(omp_sched_auto, 0); // Auto scheduling
    } else {
        abort_with_error_message("INVALID SCHEDULE TYPE! Use static, dynamic, or auto.");
    }
}

void bellman_ford(graph *G, int *dist, bool *has_negative_cycle, int num_threads, const char *schedule_type, int chunk_size) {
    int *local_dist = (int *)malloc(G->n * sizeof(int));  // Allocate local_dist
    *has_negative_cycle = false;

    double start, end;
    double exec_time = 0.0;

    // Initialize dist and local_dist
    for (int i = 0; i < G->n; i++) {
        dist[i] = INF;
        local_dist[i] = INF;
    }
    dist[0] = 0; // Assuming the source node is 0
    local_dist[0] = 0;
    
    set_schedule_type(schedule_type, chunk_size);

    start = omp_get_wtime(); // Start timing

    // Perform exactly n - 1 iterations
    for (int iter = 0; iter < G->n - 1; iter++) {
        // Parallelize edge relaxation with OpenMP
        #pragma omp parallel for num_threads(num_threads) schedule(runtime)
        for (int e = 0; e < G->m; e++) {
            edge current_edge = G->edges[e];
            if (dist[current_edge.src] < INF) {
                int new_dist = dist[current_edge.src] + current_edge.weight;
                if (new_dist < local_dist[current_edge.dest]) {
                    local_dist[current_edge.dest] = new_dist;
                }
            }
        }

        // Copy local_dist to dist after each iteration
        #pragma omp parallel for num_threads(num_threads)
        for (int i = 0; i < G->n; i++) {
            if (local_dist[i] < dist[i]) {
                dist[i] = local_dist[i];
            }
        }
    }

    end = omp_get_wtime(); // End timing
    exec_time = end - start; // Calculate elapsed time
    printf("OpenMP execution time: %.6f seconds\n", exec_time);

    // Perform extra iteration to check for negative weight cycles (SERIAL)
    for (int e = 0; e < G->m; e++) {
        edge current_edge = G->edges[e];
        if (dist[current_edge.src] < INF) {
            int new_dist = dist[current_edge.src] + current_edge.weight;
            if (new_dist < dist[current_edge.dest]) {
                *has_negative_cycle = true;
                break;  // If we find a negative cycle, we can stop early
            }
        }
    }

    free(local_dist);  // Free the local_dist array
}



int main(int argc, char **argv) {
    if (argc < 5) {
        abort_with_error_message("INPUT FILE, NUMBER OF THREADS, SCHEDULE TYPE, AND CHUNK SIZE WERE NOT FOUND!");
    }

    const char* filename = argv[1];
    int num_threads = atoi(argv[2]); // Read number of threads from command line
    const char *schedule_type = argv[3]; // Read scheduling type from command line
    int chunk_size = atoi(argv[4]); // Read chunk size from command line

    if (num_threads <= 0 || chunk_size < 0) {
        abort_with_error_message("INVALID NUMBER OF THREADS OR CHUNK SIZE!");
    }

    graph G;
    read_graph(filename, &G);

    int *dist = (int *)malloc(G.n * sizeof(int));
    bool has_negative_cycle = false;


    // Run the Bellman-Ford algorithm with OpenMP parallelism and user-defined scheduling
    printf("Number of threads: %d\n", num_threads);
    printf("Scheduling type: %s\n", schedule_type);
    printf("Chunk size: %d\n", chunk_size);

    bellman_ford(&G, dist, &has_negative_cycle, num_threads, schedule_type, chunk_size);

    mkdir("openmp_bellman_ford_results", 0755);

    char output_filename[256];
    snprintf(output_filename, sizeof(output_filename), "openmp_bellman_ford_results/output%s", strrchr(filename, '/') + 6); // Extract file number from input filename
    write_result(output_filename, dist, G.n, has_negative_cycle);

    free(dist);
    free(G.nodes);
    free(G.edges);
    return 0;
}
