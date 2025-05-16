#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <fstream>
#include <cuda_runtime.h>
#include <climits>
#include "ECLgraph.h"
using namespace std;

struct Edge {
    int u, v, w, idx;
};

class DisjointSet {
private:
    int V;
    std::vector<int> parent, rank;
public:
    DisjointSet(int V) : V(V), parent(V), rank(V, 0) {
        for (int i = 0; i < V; i++) parent[i] = i;
    }
    int find(int v) {
        if (parent[v] == v) return v;
        return parent[v] = find(parent[v]);
    }
    bool merge(int u, int v) {
        int a = find(u), b = find(v);
        if (a == b) return false;
        if (rank[a] > rank[b]) parent[b] = a;
        else if (rank[a] < rank[b]) parent[a] = b;
        else {
            parent[a] = b;
            rank[b]++;
        }
        return true;
    }
};

long long mst_cpu(std::vector<Edge>& edges, int numNodes) {
    long long mst_weight = 0;
    int trees = numNodes;
    DisjointSet ds(numNodes);
    std::vector<Edge> cheapest(numNodes, {-1, -1, -1, -1});

    while (trees > 1) {
        for (auto &e : edges) {
            int u_set = ds.find(e.u);
            int v_set = ds.find(e.v);
            if (u_set == v_set) continue;

            if (cheapest[u_set].w == -1 || cheapest[u_set].w > e.w) cheapest[u_set] = e;
            if (cheapest[v_set].w == -1 || cheapest[v_set].w > e.w) cheapest[v_set] = e;
        }

        bool merged = false;
        for (int i = 0; i < numNodes; i++) {
            if (cheapest[i].w != -1) {
                Edge &e = cheapest[i];
                if (ds.merge(e.u, e.v)) {
                    mst_weight += e.w;
                    trees--;
                    merged = true;
                }
            }
            cheapest[i] = {-1, -1, -1, -1};
        }
        if (!merged) break;
    }

    return mst_weight;
}

// CUDA Device functions and kernels

__device__ int d_find(int* parent, int v) {
    while (v != parent[v]) {
        parent[v] = parent[parent[v]];  // Path compression
        v = parent[v];
    }
    return v;
}

__device__ unsigned long long packEdge(int w, int idx) {
    return ((unsigned long long)w << 32) | (unsigned int)idx;
}

__device__ void unpackEdge(unsigned long long val, int &w, int &idx) {
    w = (int)(val >> 32);
    idx = (int)(val & 0xFFFFFFFF);
}

__device__ unsigned long long atomicMinULL(unsigned long long* address, unsigned long long val) {
    unsigned long long old = *address, assumed;
    do {
        assumed = old;
        if (assumed <= val) break;
        old = atomicCAS((unsigned long long*)address, assumed, val);
    } while (assumed != old);
    return old;
}

__global__ void pointer_jump_kernel(int* parent, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        while (parent[idx] != parent[parent[idx]]) {
        parent[idx] = parent[parent[idx]];
    }
    }
}

__global__ void find_cheapest_edges_kernel(
    Edge* edges, int* parent, unsigned long long* cheapest, int* in_mst, int numEdges) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numEdges || in_mst[idx] == 0) return;

    Edge e = edges[idx];
    int u_root = d_find(parent, e.u);
    int v_root = d_find(parent, e.v);
    if (u_root == v_root) return;

    unsigned long long packed = packEdge(e.w, idx);

    atomicMinULL(&cheapest[u_root], packed);
    atomicMinULL(&cheapest[v_root], packed);
}

__global__ void link_components_kernel(
    Edge* edges, int* parent, unsigned long long* cheapest, int* in_mst, int* mst_weight, int numNodes, int* changed) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numNodes) return;

    unsigned long long val = cheapest[idx];
    if (val == ULLONG_MAX) return;

    int w, edge_idx;
    unpackEdge(val, w, edge_idx);
    if (in_mst[edge_idx] == 0) return;

    Edge e = edges[edge_idx];
    int u_root = d_find(parent, e.u);
    int v_root = d_find(parent, e.v);
    if (u_root == v_root) return;

    int high = max(u_root, v_root);
    int low = min(u_root, v_root);

    if (atomicCAS(&parent[high], high, low) == high) {
        atomicAdd(mst_weight, w);
        atomicExch(&in_mst[edge_idx], 0);
        *changed = 1;
    }
}

long long mst_gpu(Edge* h_edges, int numNodes, int totalEdges) {
    Edge* d_edges;
    int *d_parent, *d_in_mst, *d_mst_weight, *d_changed;
    unsigned long long* d_cheapest;

    cudaMalloc(&d_edges, totalEdges * sizeof(Edge));
    cudaMalloc(&d_parent, numNodes * sizeof(int));
    cudaMalloc(&d_in_mst, totalEdges * sizeof(int));
    cudaMalloc(&d_cheapest, numNodes * sizeof(unsigned long long));
    cudaMalloc(&d_changed, sizeof(int));
    cudaMalloc(&d_mst_weight, sizeof(int));

    cudaMemcpy(d_edges, h_edges, totalEdges * sizeof(Edge), cudaMemcpyHostToDevice);

    vector<int> h_parent(numNodes);
    vector<int> h_in_mst(totalEdges, 1);
    for (int i = 0; i < numNodes; i++) h_parent[i] = i;

    cudaMemcpy(d_parent, h_parent.data(), numNodes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in_mst, h_in_mst.data(), totalEdges * sizeof(int), cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int blocks_nodes = (numNodes + threads_per_block - 1) / threads_per_block;
    int blocks_edges = (totalEdges + threads_per_block - 1) / threads_per_block;

    int h_changed = 1;
    long long h_mst_weight = 0;

    cudaMemset(d_mst_weight, 0, sizeof(int));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    while (h_changed) {
        h_changed = 0;
        cudaMemcpy(d_changed, &h_changed, sizeof(int), cudaMemcpyHostToDevice);

        // Initialize cheapest array to max value
        cudaMemset(d_cheapest, 0xFF, numNodes * sizeof(unsigned long long)); // all bits 1 = ULLONG_MAX

        // Compress parents fully by running pointer_jump multiple times

            pointer_jump_kernel<<<blocks_nodes, threads_per_block>>>(d_parent, numNodes);
            cudaDeviceSynchronize();


        find_cheapest_edges_kernel<<<blocks_edges, threads_per_block>>>(d_edges, d_parent, d_cheapest, d_in_mst, totalEdges);
        cudaDeviceSynchronize();

        link_components_kernel<<<blocks_nodes, threads_per_block>>>(d_edges, d_parent, d_cheapest, d_in_mst, d_mst_weight, numNodes, d_changed);
        cudaDeviceSynchronize();

        cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop);
    std::cout << "GPU MST time: " << gpu_time << " ms\n";

    cudaMemcpy(&h_mst_weight, d_mst_weight, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_edges);
    cudaFree(d_parent);
    cudaFree(d_in_mst);
    cudaFree(d_cheapest);
    cudaFree(d_changed);
    cudaFree(d_mst_weight);

    return h_mst_weight;
}

int main() {

    int numNodes = 0;
    int totalEdges = 0;
    const char* filename = "2d-2e20.sym.egr";
    ECLgraph g = readECLgraph(filename);

    std::cout << "Graph loaded: " << g.nodes << " nodes, " << g.edges << " edges\n";
    numNodes = g.nodes;
    totalEdges = g.edges;

    std::vector<Edge> edges(totalEdges);

    // Print or save edge list
    for (int u = 0; u < g.nodes; u++) {
        for (int i = g.nindex[u]; i < g.nindex[u + 1]; i++) {
            int v = g.nlist[i];
            int w = g.eweight ? g.eweight[i] : 1; // default weight = 1
            edges[i] = {u, v, w, i};
        }
    }
    freeECLgraph(g);
    std::cout << "Nodes: " << numNodes << ", Edges: " << totalEdges << "\n";

    auto cpu_start = std::chrono::high_resolution_clock::now();
    long long cpu_weight = mst_cpu(edges, numNodes);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = cpu_end - cpu_start;

    std::cout << "CPU MST weight: " << cpu_weight << "\n";
    std::cout << "CPU time: " << cpu_duration.count() << " ms\n";

    long long gpu_weight = mst_gpu(edges.data(), numNodes, totalEdges);
    std::cout << "GPU MST weight: " << gpu_weight << "\n";

    return 0;
}

