#!/usr/bin/env python3
"""
Optimized PoC for CFG-based dominator computation with GPU acceleration.
"""

import ast
import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Set, Optional

import numpy as np
import cupy as cp
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

# ========================================================
# 1. CFG Construction (AST → CFG)
# ========================================================
@dataclass
class Block:
    """
    Represents a basic block in the Control Flow Graph (CFG).

    Attributes:
        id (int): A unique identifier for the block.
        successors (List[int]): A list of IDs of the successor blocks.
    """
    id: int
    successors: List[int] = field(default_factory=list)

class CFG:
    """
    Represents the Control Flow Graph (CFG).

    Attributes:
        keys (Dict[int, Block]): A dictionary mapping block IDs to Block objects.  Using 'keys'
                                  here emphasizes that the dictionary keys are the primary way to
                                  access and iterate through the blocks.
    """
    def __init__(self) -> None:
        self.keys: Dict[int, Block] = {}  # Using 'keys' to represent the blocks

    def add_block(self, block: Block) -> None:
        """Adds a block to the CFG."""
        self.keys[block.id] = block

    def get_graph(self) -> Dict[int, List[int]]:
        """Returns the CFG as an adjacency list (dictionary)."""
        return {block.id: block.successors for block in self.keys.values()}

class CFGBuilder:
    """
    Builds a Control Flow Graph (CFG) from an Abstract Syntax Tree (AST).
    """
    def __init__(self) -> None:
        self.cfg = CFG()
        self.next_block_id = 0  # Counter for assigning unique block IDs

    def new_block(self) -> Block:
        """Creates a new block, adds it to the CFG, and returns it."""
        block = Block(self.next_block_id)
        self.cfg.add_block(block)
        self.next_block_id += 1
        return block

    def _connect_blocks(self, from_block: Optional[Block], to_block: Block) -> None:
        """Connects two blocks in the CFG (from_block -> to_block)."""
        if from_block:
            from_block.successors.append(to_block.id)

    def build_sequence(self, stmts: List[ast.stmt]) -> Tuple[Block, Optional[Block]]:
        """
        Recursively builds a CFG for a sequence of AST statements.

        Args:
            stmts: A list of AST statements.

        Returns:
            A tuple containing the entry and exit blocks of the generated CFG.
        """
        if not stmts:
            empty_block = self.new_block()
            return empty_block, empty_block

        first_block = None  # The first block in the sequence
        prev_block = None   # The previously processed block

        for stmt in stmts:
            if isinstance(stmt, ast.If):
                # Create a block for the if condition
                cond_block = self.new_block()
                self._connect_blocks(prev_block, cond_block)

                # Recursively build CFGs for the then and else branches
                then_entry, then_exit = self.build_sequence(stmt.body)
                else_entry, else_exit = self.build_sequence(stmt.orelse) if stmt.orelse else (self.new_block(), self.new_block())

                # Connect the condition block to the entry blocks of the branches
                cond_block.successors.extend([then_entry.id, else_entry.id])

                # Create a join block where the branches merge
                join_block = self.new_block()
                if then_exit:
                    self._connect_blocks(then_exit, join_block)
                if else_exit:
                    self._connect_blocks(else_exit, join_block)

                prev_block = join_block  # The join block becomes the previous block for the next statement
                if not first_block:
                    first_block = cond_block

            elif isinstance(stmt, (ast.While, ast.For)):
                # Create a block for the loop condition/iterator
                cond_block = self.new_block()
                self._connect_blocks(prev_block, cond_block)

                # Recursively build the CFG for the loop body
                body_entry, body_exit = self.build_sequence(stmt.body)

                # Connect the condition block to the body entry and the exit block
                cond_block.successors.append(body_entry.id)
                if body_exit:
                    self._connect_blocks(body_exit, cond_block)  # Loop back edge

                exit_block = self.new_block()
                cond_block.successors.append(exit_block.id)
                prev_block = exit_block  # The exit block becomes the previous block
                if not first_block:
                    first_block = cond_block

            else:
                # Create a block for a simple statement
                simple_block = self.new_block()
                self._connect_blocks(prev_block, simple_block)
                prev_block = simple_block
                if not first_block:
                    first_block = simple_block

        return first_block, prev_block

    def build(self, stmts: List[ast.stmt]) -> CFG:
        """Builds the complete CFG from a list of AST statements."""
        entry, exit_blk = self.build_sequence(stmts)
        final_exit = self.new_block()
        if exit_blk:
            self._connect_blocks(exit_blk, final_exit)
        return self.cfg

def ast_to_cfg(source: str) -> Dict[int, List[int]]:
    """Converts Python source code to a CFG (adjacency list representation)."""
    tree = ast.parse(source)
    builder = CFGBuilder()
    cfg_obj = builder.build(tree.body)
    return cfg_obj.get_graph()

# ========================================================
# 2. Graph Preprocessing (Optimized Predecessor Construction)
# ========================================================
class GraphPreprocessor:
    """
    Preprocesses the CFG to create fixed-size predecessor data for GPU computation.
    """
    @staticmethod
    def build_predecessor_fixed(graph: Dict[int, List[int]], num_nodes: int) -> Tuple[cp.ndarray, cp.ndarray, int]:
        """
        Builds fixed-size predecessor arrays for all nodes in the graph.

        Args:
            graph: The CFG as an adjacency list (dictionary).
            num_nodes: The total number of nodes in the CFG.

        Returns:
            A tuple containing:
                - pred_counts_gpu: A CuPy array storing the number of predecessors for each node.
                - pred_indices_gpu: A CuPy array storing the IDs of predecessors for each node
                                   (flattened into a 1D array).
                - max_pred: The maximum number of predecessors for any node.
        """
        pred_lists = [[] for _ in range(num_nodes)]  # List of predecessors for each node
        for src, succs in graph.items():
            for dst in succs:
                if 0 <= dst < num_nodes:
                    pred_lists[dst].append(src)

        max_pred = max((len(lst) for lst in pred_lists), default=0)  # Maximum number of predecessors

        # Create a fixed-size NumPy array and fill it with predecessor IDs
        pred_indices = np.full((num_nodes, max_pred), -1, dtype=np.int32)  # -1 indicates invalid node ID
        for i, lst in enumerate(pred_lists):
            pred_indices[i, :len(lst)] = lst

        pred_counts = np.array([len(lst) for lst in pred_lists], dtype=np.int32) # Number of predecessors for each node

        # Transfer the arrays to the GPU using CuPy
        pred_counts_gpu = cp.asarray(pred_counts)
        pred_indices_gpu = cp.asarray(pred_indices.flatten())  # Flatten for efficient GPU kernel access

        return pred_counts_gpu, pred_indices_gpu, max_pred

# ========================================================
# 3. GPU Dominator Computation (Optimized Kernel)
# ========================================================
class DominatorCalculator:
    """
    Computes dominators and immediate dominators on the GPU using a double-buffered,
    persistent kernel.
    """

    COMBINED_KERNEL_CODE = r'''
    #define MAX_PRED %d          // Maximum number of predecessors (filled in during compilation)
    #define INVALID_NODE -1      // Represents an invalid node ID
    #define INT_BITS 32          // Number of bits in an unsigned integer

    extern "C"
    __global__ void compute_doms_and_idom(const short n, const short root, const unsigned int U,
                                          const int * __restrict__ pred_counts,
                                          const int * __restrict__ pred_indices,
                                          unsigned int * __restrict__ dom,
                                          short * __restrict__ idom,
                                          const int iterations)
    {
        // Use shared memory for double buffering (dom_in and dom_out)
        extern __shared__ unsigned int shared_mem[];
        unsigned int* dom_in = shared_mem;        // Input buffer (from previous iteration)
        unsigned int* dom_out = shared_mem + n;    // Output buffer (for current iteration)

        int tid = threadIdx.x; // Thread ID (corresponds to node ID)

        // Initialize dom_in with the initial dominator sets
        if (tid < n)
            dom_in[tid] = dom[tid];
        __syncthreads();

        // Iterate until convergence (or for a fixed number of iterations)
        for (int it = 0; it < iterations; it++) {
            unsigned int newDom; // The new dominator set for the current node

            // The root node dominates only itself
            if (tid == root) {
                newDom = (1u << tid);
            } else {
                newDom = U; // Start with the universal set
                int count = __ldg(&pred_counts[tid]); // Number of predecessors
                int base_idx = tid * MAX_PRED;       // Base index for accessing predecessors

                // Iterate over predecessors and intersect their dominator sets
                #pragma unroll
                for (int j = 0; j < MAX_PRED; j++) {
                    if (j < count) { // Check if a predecessor exists at this index
                        int p = __ldg(&pred_indices[base_idx + j]); // Predecessor node ID
                        if (p != INVALID_NODE) {  // should check p >=0, but INVALID_NODE is safer
                            newDom &= dom_in[p];
                        }
                    }
                }
                newDom |= (1u << tid); // Add the current node to its dominator set
            }

            // Write the new dominator set to the output buffer
            dom_out[tid] = newDom;
            __syncthreads(); // Wait for all threads to finish writing

            // Swap the input and output buffers
            unsigned int* temp = dom_in;
            dom_in = dom_out;
            dom_out = temp;
            __syncthreads(); // Wait for all threads to acknowledge the swap
        }

        // Copy the final dominator set to global memory and compute the immediate dominator
        if (tid < n) {
            dom[tid] = dom_in[tid]; // The final result is in dom_in
            if (tid == root) {
                idom[tid] = INVALID_NODE; // The root node has no immediate dominator
            } else {
                unsigned int mask = dom_in[tid] & ~(1u << tid); // Remove the current node from its dominator set
                idom[tid] = (mask != 0) ? (INT_BITS - 1 - __clz(mask)) : INVALID_NODE; // Find immediate dominator
            }
        }
    }
    '''

    def __init__(self, num_nodes: int, root: int, U: int, max_pred: int) -> None:
        """
        Initializes the DominatorCalculator.

        Args:
            num_nodes: The number of nodes in the CFG.
            root: The ID of the root node.
            U: The universal set (bitmask representing all nodes).
            max_pred: The maximum number of predecessors for any node.
        """
        self.num_nodes = num_nodes
        self.root = root
        self.U = U
        self.max_pred = max_pred
        self.block_size = min(256, num_nodes) # Limit block size to 256 for better occupancy
        self.grid_size = (num_nodes + self.block_size - 1) // self.block_size
        kernel_source = self.COMBINED_KERNEL_CODE % self.max_pred
        self.module = SourceModule(kernel_source, options=["--use_fast_math"])
        self.combined_kernel = self.module.get_function("compute_doms_and_idom")

    def run(self, pred_counts_gpu: cp.ndarray, pred_indices_gpu: cp.ndarray,
            dom_init: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs the GPU-accelerated dominator computation.

        Args:
            pred_counts_gpu:  Number of predecessors for each node (on GPU).
            pred_indices_gpu: Flattened array of predecessor IDs (on GPU).
            dom_init: Initial dominator sets (on CPU).

        Returns:
            A tuple containing:
                - dom_host:  Dominator sets for each node (on CPU).
                - idom_host: Immediate dominators for each node (on CPU).
        """
        dom_host = cuda.pagelocked_empty(self.num_nodes, dtype=np.uint32)
        idom_host = cuda.pagelocked_empty(self.num_nodes, dtype=np.int16)
        dom_gpu = cuda.mem_alloc(dom_init.nbytes)
        idom_gpu = cuda.mem_alloc(idom_host.nbytes)
        cuda.memcpy_htod(dom_gpu, dom_init) # Transfer initial dominator sets to GPU
        iterations = np.int32(self.num_nodes) # Maximum iterations (can be tuned)
        shared_mem_size = self.block_size * np.dtype(np.uint32).itemsize * 2 # Double buffering

        self.combined_kernel(np.int16(self.num_nodes), np.int16(self.root), np.uint32(self.U),
                             np.intp(int(pred_counts_gpu.data.ptr)),
                             np.intp(int(pred_indices_gpu.data.ptr)),
                             dom_gpu, idom_gpu, iterations,
                             block=(self.block_size, 1, 1),
                             grid=(self.grid_size, 1),
                             shared=shared_mem_size)
        cuda.Context.synchronize() # Ensure kernel completion
        cuda.memcpy_dtoh(dom_host, dom_gpu)    # Transfer results back to CPU
        cuda.memcpy_dtoh(idom_host, idom_gpu)
        return dom_host, idom_host

# ========================================================
# 4. Helper Functions
# ========================================================
def bitmask_to_set(mask: int) -> Set[int]:
    """Converts a bitmask to a set of integers."""
    result = set()
    index = 0
    while mask:
        if mask & 1:
            result.add(index)
        mask >>= 1
        index += 1
    return result

# ========================================================
# 5. Precomputation
# ========================================================
def precompute_data(source_code: str) -> Tuple[
        Dict[int, List[int]], int, int,
        cp.ndarray, cp.ndarray, int, int, np.ndarray]:
    """
    Performs precomputation steps: CFG construction, preprocessing, and initialization.
    """
    graph = ast_to_cfg(source_code)
    num_nodes = len(graph)
    root = 0  # Always assume node 0 is the root
    pred_counts_gpu, pred_indices_gpu, max_pred = GraphPreprocessor.build_predecessor_fixed(graph, num_nodes)
    U = (1 << num_nodes) - 1  # Universal set (bitmask)
    dom_init = np.empty(num_nodes, dtype=np.uint32)
    for v in range(num_nodes):
        dom_init[v] = (1 << v) if v == root else U  # Initialize dominator sets
    return graph, num_nodes, root, pred_counts_gpu, pred_indices_gpu, max_pred, U, dom_init

# ========================================================
# 6. End-to-End Pipeline (With Timing)
# ========================================================
def run_dominator_computation(source_code: str) -> Tuple[np.ndarray, np.ndarray, float, float, Dict[int, List[int]]]:
    """
    Runs the complete dominator computation pipeline.
    """
    start_total = time.time()
    graph, num_nodes, root, pred_counts_gpu, pred_indices_gpu, max_pred, U, dom_init = precompute_data(source_code)
    dominator_calc = DominatorCalculator(num_nodes, root, U, max_pred)
    start_gpu = time.time()
    dom_host, idom_host = dominator_calc.run(pred_counts_gpu, pred_indices_gpu, dom_init)
    gpu_time = (time.time() - start_gpu) * 1000  # ms
    comp_time = time.time() - start_total  # s

    return dom_host, idom_host, comp_time, gpu_time, graph

def print_results(graph: Dict[int, List[int]], dom_host: np.ndarray, idom_host: np.ndarray) -> None:
    """Prints the CFG, dominator sets, and immediate dominators."""
    print("\n==== Constructed CFG ====")
    for node_id in sorted(graph.keys()):
        print(f"{node_id}: -> {graph[node_id]}")

    print("\n==== Dominator Sets (Dom) ====")
    for v in range(len(dom_host)):
        dom_set = sorted(bitmask_to_set(int(dom_host[v])))
        print(f"v = {v:2d}: {dom_set}")

    print("\n==== Immediate Dominators (idom) ====")
    for v in range(len(idom_host)):
        print(f"v = {v:2d}: idom = {idom_host[v]}")

# ========================================================
# 7. Benchmarking
# ========================================================
def benchmark_pipeline(sample_code: str, iterations: int = 10) -> None:
    """
    Runs the dominator computation pipeline multiple times and reports average timings.
    """
    total_times = []
    gpu_times = []
    for _ in range(iterations):
        dom_host, idom_host, comp_time, gpu_time, graph = run_dominator_computation(sample_code)
        total_times.append(comp_time)
        gpu_times.append(gpu_time)
    avg_total = np.mean(total_times) * 1000  # ms
    avg_gpu = np.mean(gpu_times)
    print(f"\n=== Benchmark Results over {iterations} iterations ===")
    print(f"Average Total Pipeline Time (without print): {avg_total:.3f} ms")
    print(f"Average GPU Kernel Time: {avg_gpu:.3f} ms")
    print_results(graph, dom_host, idom_host)

# ========================================================
# 8. Main Execution
# ========================================================
def main() -> None:
    sample_code = """
a = 1
if a > 0:
    print("positive")
    a = a - 1
else:
    print("non-positive")
    a = a + 1
while a < 10:
    a += 1
print(a)
"""
    print("=== Dominator Computation Pipeline ===")
    # Run once and print detailed results
    dom_host, idom_host, comp_time, gpu_time, graph = run_dominator_computation(sample_code)
    print_results(graph, dom_host, idom_host)
    print(f"\nTotal Computation Time: {comp_time*1000:.3f} ms")

    # Run benchmark
    benchmark_pipeline(sample_code, iterations=10)

if __name__ == '__main__':
    main()
