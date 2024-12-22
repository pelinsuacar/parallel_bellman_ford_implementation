#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <sys/stat.h>

#define INF 1000000

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                          \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);               \
        fprintf(stderr, "code: %d, reason: %s\n", error,                     \
                cudaGetErrorString(error));                                  \
        exit(1);                                                               \
    }                                                                          \
}

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

__global__ void bellman_ford_one_iter(int n, int m, edge *d_edges, int *d_dist, bool *d_has_next) {
    int global_tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (global_tid < m) {
        edge e = d_edges[global_tid];
        if (d_dist[e.src] < INF) {
            int new_dist = d_dist[e.src] + e.weight;
            if (new_dist < d_dist[e.dest]) {
                d_dist[e.dest] = new_dist;
                *d_has_next = true;
            }
        }
    }
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
void bellman_ford(graph *G, int *dist, bool *has_negative_cycle, int blocksPerGrid, int threadsPerBlock) {
    int *d_dist;
    edge *d_edges;
    bool *d_has_next, h_has_next;

    struct timeval start, end;
    double alloc_time, exec_time = 0.0, h2d_time = 0.0, d2h_time = 0.0;

    // Memory allocation/setup timing
    gettimeofday(&start, NULL);

    cudaMalloc(&d_edges, G->m * sizeof(edge));
    cudaMalloc(&d_dist, sizeof(int) * G->n);
    cudaMalloc(&d_has_next, sizeof(bool));

    *has_negative_cycle = false;

    for (int i = 0; i < G->n; i++) {
        dist[i] = INF;
    }
    dist[0] = 0; // Assuming the source node is 0

    gettimeofday(&end, NULL);
    alloc_time = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;


    // Host to Device copy timing
    gettimeofday(&start, NULL);
    cudaMemcpy(d_edges, G->edges, sizeof(edge) * G->m, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dist, dist, sizeof(int) * G->n, cudaMemcpyHostToDevice);
    gettimeofday(&end, NULL);
    h2d_time += (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;

    // Run the algorithm for exactly n - 1 iterations 
    for (int i = 0; i < G->n - 1; i++) {
        h_has_next = false;

        gettimeofday(&start, NULL);
        cudaMemcpy(d_has_next, &h_has_next, sizeof(bool), cudaMemcpyHostToDevice);
        gettimeofday(&end, NULL);
        h2d_time += (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;

        // Run one iteration of Bellman-Ford
        gettimeofday(&start, NULL);
        bellman_ford_one_iter<<<blocksPerGrid, threadsPerBlock>>>(G->n, G->m, d_edges, d_dist, d_has_next);
        gettimeofday(&end, NULL);
        CHECK(cudaDeviceSynchronize());
        exec_time += (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;

        gettimeofday(&start, NULL);
        cudaMemcpy(&h_has_next, d_has_next, sizeof(bool), cudaMemcpyDeviceToHost);
        gettimeofday(&end, NULL);
        d2h_time += (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;
    }

    // Copy the distances back to host for negative cycle check
    gettimeofday(&start, NULL);
    cudaMemcpy(dist, d_dist, sizeof(int) * G->n, cudaMemcpyDeviceToHost);
    gettimeofday(&end, NULL);
    d2h_time += (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;

    // CPU negative cycle detection
    for (int i = 0; i < G->m; i++) {
        edge e = G->edges[i];
        if (dist[e.src] < INF && dist[e.src] + e.weight < dist[e.dest]) {
            *has_negative_cycle = true;
            break;
        }
    }

    cudaFree(d_edges);
    cudaFree(d_dist);
    cudaFree(d_has_next);

    printf("Memory allocation/setup time: %.6f seconds\n", alloc_time);
    printf("Host to Device copy time: %.6f seconds\n", h2d_time);
    printf("Device to Host copy time: %.6f seconds\n", d2h_time);
    printf("Total memory copy time: %.6f seconds\n", h2d_time + d2h_time);
    printf("CUDA execution time: %.6f seconds\n", exec_time);
}


int main(int argc, char **argv) {
    if (argc <= 2) {
        abort_with_error_message("INPUT FILE AND NUMBER OF THREADS WERE NOT FOUND!");
    }

    const char* filename = argv[1];
    int threadsPerBlock = atoi(argv[2]); // Read number of threads from command line
    if (threadsPerBlock <= 0) {
        abort_with_error_message("INVALID NUMBER OF THREADS!");
    }

    graph G;
    read_graph(filename, &G);

    int *dist = (int *)calloc(G.n, sizeof(int));
    bool has_negative_cycle = false;

    cudaDeviceReset();

    int blocksPerGrid = (G.m + threadsPerBlock - 1) / threadsPerBlock;
    printf("Number of blocks: %d\n", blocksPerGrid);
    printf("Number of threads: %d\n", threadsPerBlock);

    bellman_ford(&G, dist, &has_negative_cycle, blocksPerGrid, threadsPerBlock);

    mkdir("cuda_bellman_ford_results", 0755);

    char output_filename[256];
    snprintf(output_filename, sizeof(output_filename), "cuda_bellman_ford_results/output%s", strrchr(filename, '/') + 6); // Extract file number from input filename
    write_result(output_filename, dist, G.n, has_negative_cycle);

    free(dist);
    free(G.nodes);
    free(G.edges);
    return 0;
}
