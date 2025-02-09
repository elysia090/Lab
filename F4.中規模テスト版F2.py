#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Refactored PyCUDA Implementation for a Medium-Scale Graph (n=64)
with 64-bit dominator set representation, randomized control-flow graph,
optimized fixed-point iteration kernel, and benchmark (ms unit).

Note:
- dominator set is represented as a 64-bit integer.
- A random control-flow graph with n nodes is generated:
    * Node 0 is the unique entry.
    * For each node v (v>=1), a random number (between min_preds and max_preds)
      of predecessors are chosen from among nodes 0 to v-1 (ensuring acyclicity).
- Since the generated graph stores for each node its predecessors,
  we compute the successors separately to perform DFS order computation.
- GPU kernels use "unsigned long long" for 64-bit operations.
"""

import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import time
import random

# ===============================
# 1. Graph Generation (Random CFG)
# ===============================
def generate_random_cfg(n, min_preds=1, max_preds=3):
    """
    Generate a random control-flow graph with n nodes.
    Node 0 is the unique entry.
    For each node v >= 1, choose a random number of predecessors (between min_preds and max_preds)
    from among nodes 0 to v-1 (ensuring acyclicity).
    """
    graph = {i: [] for i in range(n)}
    for v in range(1, n):
        num_preds = random.randint(min_preds, max_preds)
        preds = random.sample(range(v), min(num_preds, v))
        graph[v] = sorted(preds)
    graph[0] = []
    return graph

# ===============================
# 2. Compute Successors (for DFS Order)
# ===============================
def compute_successors(graph):
    """
    Given a graph (where each node stores its predecessors), compute the successors.
    """
    successors = {v: [] for v in graph}
    for v in graph:
        for u in graph[v]:
            successors[u].append(v)
    return successors

# ===============================
# 3. Host-side Preprocessing: Prepare Graph Arrays (Predecessors)
# ===============================
def compute_predecessors(graph):
    """For our purpose, the input graph already is the predecessor list."""
    return graph

def prepare_graph_arrays(graph):
    """
    Convert each node's predecessor list into a flat array and an offsets array.
    Returns: (pred_list_np, pred_offsets_np)
    """
    n = len(graph)
    pred = compute_predecessors(graph)
    pred_list = []
    pred_offsets = [0]
    for v in range(n):
        plist = pred.get(v, [])
        pred_list.extend(plist)
        pred_offsets.append(len(pred_list))
    # Allocate pinned memory for faster transfer
    pred_list_np = cuda.pagelocked_empty(len(pred_list), dtype=np.int32)
    pred_list_np[:] = np.array(pred_list, dtype=np.int32)
    pred_offsets_np = cuda.pagelocked_empty(len(pred_offsets), dtype=np.int32)
    pred_offsets_np[:] = np.array(pred_offsets, dtype=np.int32)
    return pred_list_np, pred_offsets_np

# ===============================
# 4. GPU Preprocessing Kernel: Init dominator set and GrayCode (64-bit)
# ===============================
init_kernel_code = r'''
extern "C"
__global__ void init_preproc(int n, int root, unsigned long long U,
                             unsigned long long *dom, int *gray)
{
    int v = blockDim.x * blockIdx.x + threadIdx.x;
    if (v >= n) return;
    // GrayCode: v XOR (v >> 1)
    gray[v] = v ^ (v >> 1);
    // Initialize dominator set: for root, only itself; others, U (all bits 1)
    if (v == root)
        dom[v] = (1ULL << v);
    else
        dom[v] = U;
}
'''

mod_init = SourceModule(init_kernel_code, options=["--use_fast_math"])
init_preproc = mod_init.get_function("init_preproc")

# ===============================
# 5. Optimized Fixed-Point Iteration Kernel (64-bit, using Shared Memory)
# ===============================
update_kernel_code = r'''
extern "C"
__global__ void update_dom_shared_opt(int n, unsigned long long U, unsigned long long *dom,
                                      const int *pred_offsets, const int *pred_list,
                                      int root, int *changed)
{
    // Determine block's vertex range
    int block_start = blockIdx.x * blockDim.x;
    int block_end = block_start + blockDim.x;
    if (block_end > n) block_end = n;
    int local_idx = threadIdx.x;
    int global_idx = block_start + local_idx;
    
    extern __shared__ unsigned long long s_dom[];
    if (global_idx < block_end)
        s_dom[local_idx] = dom[global_idx];
    __syncthreads();
    
    __shared__ int block_changed;
    if (local_idx == 0)
        block_changed = 0;
    __syncthreads();
    
    for (int iter = 0; iter < 1000; iter++) {
        bool local_changed = false;
        if (global_idx < block_end && global_idx != root) {
            unsigned long long newDom = U;
            int start = pred_offsets[global_idx];
            int end = pred_offsets[global_idx + 1];
            for (int i = start; i < end; i++) {
                int u = pred_list[i];
                if (u >= block_start && u < block_end)
                    newDom &= s_dom[u - block_start];
                else
                    newDom &= dom[u];
            }
            newDom |= (1ULL << global_idx);
            if (newDom != s_dom[local_idx]) {
                s_dom[local_idx] = newDom;
                local_changed = true;
            }
        }
        __syncthreads();
        if (local_changed)
            atomicOr(&block_changed, 1);
        __syncthreads();
        if (block_changed == 0)
            break;
        if (local_idx == 0)
            block_changed = 0;
        __syncthreads();
    }
    if (global_idx < block_end)
        dom[global_idx] = s_dom[local_idx];
    
    if (local_idx == 0 && block_changed)
        atomicExch(changed, 1);
}
'''

mod_update = SourceModule(update_kernel_code, options=["--use_fast_math"])
update_dom_shared_opt = mod_update.get_function("update_dom_shared_opt")

# ===============================
# 6. Host-side Postprocessing: DFS Order and Immediate Dominators
# ===============================
def dfs_order(successors, root):
    """Compute DFS order (using successors) starting from root"""
    order = {}
    visited = set()
    counter = 0
    def dfs(v):
        nonlocal counter
        visited.add(v)
        order[v] = counter
        counter += 1
        for w in successors[v]:
            if w not in visited:
                dfs(w)
    dfs(root)
    return order

def compute_immediate_dominators(dom, order, root, n):
    """
    For each vertex v (v != root), select the immediate dominator from dom[v]
    (excluding v itself) as the candidate with the maximum DFS order.
    Note: Use 64-bit shift: (np.uint64(1) << np.uint64(d)).
    """
    idom = {}
    for v in range(n):
        if v == root:
            continue
        candidates = []
        for d in range(n):
            if d == v:
                continue
            if dom[v] & (np.uint64(1) << np.uint64(d)):
                candidates.append(d)
        if candidates:
            idom[v] = max(candidates, key=lambda d: order[d])
        else:
            idom[v] = None
    return idom

def bitmask_to_set(mask):
    """Convert a bitmask to a set of indices (for debugging)"""
    s = set()
    mask_int = int(mask)  # キャストして Python の int 型に変換
    i = 0
    while mask_int:
        if mask_int & 1:
            s.add(i)
        mask_int >>= 1
        i += 1
    return s

# ===============================
# 7. Main Processing and Benchmark (excluding printing time)
# ===============================
def main():
    # Generate a medium-scale graph (n = 64)
    n = 64
    graph = generate_random_cfg(n, min_preds=1, max_preds=3)
    root = 0
    
    # For DFS order, compute successors from the predecessor graph.
    # Since our graph is stored as predecessors, compute successors.
    successors = {v: [] for v in graph}
    for v in graph:
        for u in graph[v]:
            successors[u].append(v)
    
    results = {}  # Store compute times (ms)
    
    # Host-side preprocessing: prepare graph arrays
    t0 = time.time()
    pred_list_np, pred_offsets_np = prepare_graph_arrays(graph)
    t1 = time.time()
    results["Host Graph Preprocessing"] = (t1 - t0)*1000  # ms
    
    # GPU initialization: use pinned memory
    U = np.uint64((1 << n) - 1)  # 64-bit all 1's
    dom_host = cuda.pagelocked_empty(n, dtype=np.uint64)
    gray_host = cuda.pagelocked_empty(n, dtype=np.int32)
    changed_host = cuda.pagelocked_empty(1, dtype=np.int32)
    
    # Allocate GPU buffers
    dom_gpu = cuda.mem_alloc(dom_host.nbytes)
    gray_gpu = cuda.mem_alloc(gray_host.nbytes)
    changed_gpu = cuda.mem_alloc(changed_host.nbytes)
    
    # Transfer predecessor info to GPU
    pred_list_gpu = cuda.mem_alloc(pred_list_np.nbytes)
    cuda.memcpy_htod(pred_list_gpu, pred_list_np)
    pred_offsets_gpu = cuda.mem_alloc(pred_offsets_np.nbytes)
    cuda.memcpy_htod(pred_offsets_gpu, pred_offsets_np)
    
    # Create an asynchronous stream
    stream = cuda.Stream()
    
    # GPU preprocessing kernel (asynchronous)
    block_size_init = 128
    grid_size_init = (n + block_size_init - 1) // block_size_init
    start_event = cuda.Event()
    end_event = cuda.Event()
    start_event.record(stream)
    init_preproc(np.int32(n), np.int32(root), np.uint64(U),
                 dom_gpu, gray_gpu,
                 block=(block_size_init, 1, 1), grid=(grid_size_init, 1), stream=stream)
    end_event.record(stream)
    stream.synchronize()
    results["GPU Preprocessing Kernel"] = start_event.time_till(end_event)  # ms
    
    # GPU fixed-point iteration loop (optimized kernel, asynchronous)
    block_size = 128
    grid_size = (n + block_size - 1) // block_size
    shared_mem_size = block_size * np.uint64(0).nbytes
    iteration = 0
    compute_start = time.time()
    while True:
        iteration += 1
        changed_host[0] = 0
        cuda.memcpy_htod_async(changed_gpu, changed_host, stream)
        update_dom_shared_opt(np.int32(n), np.uint64(U), dom_gpu,
                              pred_offsets_gpu, pred_list_gpu,
                              np.int32(root), changed_gpu,
                              block=(block_size, 1, 1), grid=(grid_size, 1),
                              shared=shared_mem_size, stream=stream)
        cuda.memcpy_dtoh_async(changed_host, changed_gpu, stream)
        stream.synchronize()
        if changed_host[0] == 0:
            break
    compute_end = time.time()
    results["GPU Fixed-Point Iteration Loop"] = (compute_end - compute_start)*1000
    results["Iteration Count"] = iteration
    
    # Retrieve results from GPU (asynchronous)
    cuda.memcpy_dtoh_async(dom_host, dom_gpu, stream)
    stream.synchronize()
    
    # Host-side DFS Order Calculation (using successors)
    t2 = time.time()
    order = dfs_order(successors, root)
    t3 = time.time()
    results["Host DFS Order Calculation"] = (t3 - t2)*1000
    
    # Host-side Immediate Dominator Calculation
    t4 = time.time()
    idom = compute_immediate_dominators(dom_host, order, root, n)
    t5 = time.time()
    results["Host Immediate Dominator Calculation"] = (t5 - t4)*1000
    
    # Retrieve GrayCode results from GPU (asynchronous)
    cuda.memcpy_dtoh_async(gray_host, gray_gpu, stream)
    stream.synchronize()
    
    total_compute_time = (time.time() - t0)*1000
    
    # 結果出力（printは計測に含めない）
    print("\n==== Execution Time Results (ms) ====")
    for key, val in results.items():
        if key == "Iteration Count":
            print(f"{key}: {val}")
        else:
            print(f"{key}: {val:.3f} ms")
    print("Total Compute Time (excl. printing): {:.3f} ms".format(total_compute_time))
    
    print("\n==== Dominator Sets (Dom) ====")
    for v in range(n):
        print("  v = {:3d}: {}".format(v, sorted(bitmask_to_set(dom_host[v]))))
    
    print("\n==== DFS Order ====")
    print(order)
    
    print("\n==== Immediate Dominators (idom) ====")
    for v in sorted(idom.keys()):
        print("  v = {:3d}: idom = {}".format(v, idom[v]))
    
    print("\n==== GrayCode Labels ====")
    for v in range(n):
        print("  v = {:3d}: gray = {}".format(v, gray_host[v]))

if __name__ == '__main__':
    main()
