"""
GPU-Accelerated Dominator Analysis for Control Flow Graphs

This module implements control flow graph (CFG) construction and 
dominator computation using CUDA GPU acceleration.
"""

import ast
import time
from typing import Dict, List, Tuple, Any, Optional, Set, Union
from collections import defaultdict

import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import graphviz  # For CFG visualization

# Configuration constants
class Config:
    """Configuration settings for dominator computation."""
    BITS_PER_MASK: int = 64
    MAX_MASK_COUNT: int = 10
    DEFAULT_BLOCK_SIZE: int = 256
    MAX_ITERATIONS: int = 1000  # Safety limit for convergence


# --------------------------------------------------
# CFG Construction Module
# --------------------------------------------------
class CFGBuilder(ast.NodeVisitor):
    """
    Control Flow Graph Builder that traverses Python AST.
    
    Builds a control flow graph by visiting AST nodes and tracking edges
    between basic blocks.
    """
    
    def __init__(self) -> None:
        """Initialize a new CFG builder."""
        self.cfg: Dict[int, List[int]] = defaultdict(list)
        self.predecessors: Dict[int, List[int]] = defaultdict(list)
        self.current_node: int = 0
        self.node_count: int = 0
        self.labels: Dict[int, Any] = {}  # Maps node IDs to AST nodes or labels
        
    def add_edge(self, from_node: int, to_node: int) -> None:
        """
        Add a directed edge from one node to another in the CFG.
        
        Args:
            from_node: Source node ID
            to_node: Destination node ID
        """
        self.cfg[from_node].append(to_node)
        self.predecessors[to_node].append(from_node)
        
    def new_node(self) -> int:
        """
        Create a new node in the CFG.
        
        Returns:
            The ID of the newly created node
        """
        self.node_count += 1
        return self.node_count - 1
        
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """
        Visit a function definition node and create entry point in CFG.
        
        Args:
            node: The FunctionDef AST node
        """
        self.current_node = 0
        self.node_count = 1
        self.labels[0] = node
        self.generic_visit(node)
        
    def visit_If(self, node: ast.If) -> None:
        """
        Visit an if statement node and create appropriate CFG branches.
        
        Args:
            node: The If AST node
        """
        current_node = self.current_node
        
        # Create then branch
        then_node = self.new_node()
        self.labels[then_node] = node
        self.add_edge(current_node, then_node)
        self.current_node = then_node
        
        for stmt in node.body:
            self.visit(stmt)
            
        if_end_node = self.current_node
        
        # Create else branch if it exists
        if node.orelse:
            else_node = self.new_node()
            self.labels[else_node] = node.orelse
            self.add_edge(current_node, else_node)
            self.current_node = else_node
            
            for stmt in node.orelse:
                self.visit(stmt)
                
            # Create merge point after if-else
            next_node = self.new_node()
            self.labels[next_node] = f"next_{node.lineno}"
            self.add_edge(if_end_node, next_node)
            self.add_edge(self.current_node, next_node)
            self.current_node = next_node
        else:
            # Create merge point after if (no else)
            next_node = self.new_node()
            self.labels[next_node] = f"next_{node.lineno}"
            self.add_edge(current_node, next_node)
            self.add_edge(if_end_node, next_node)
            self.current_node = next_node
            
    def visit_Return(self, node: ast.Return) -> None:
        """
        Visit a return statement node and create corresponding CFG node.
        
        Args:
            node: The Return AST node
        """
        return_node = self.new_node()
        self.labels[return_node] = node
        self.add_edge(self.current_node, return_node)
        self.current_node = return_node
        
    def generic_visit(self, node: ast.AST) -> None:
        """
        Visit nodes not handled by specific visit methods.
        
        Args:
            node: Any AST node
        """
        if isinstance(node, ast.FunctionDef):
            for stmt in node.body:
                self.visit(stmt)
        elif isinstance(node, list):
            for item in node:
                if isinstance(item, ast.AST):
                    self.visit(item)
        else:
            current_node = self.new_node()
            self.labels[current_node] = type(node).__name__
            self.add_edge(self.current_node, current_node)
            self.current_node = current_node
            
            for field, value in ast.iter_fields(node):
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, ast.AST):
                            self.visit(item)
                elif isinstance(value, ast.AST):
                    self.visit(value)


def build_cfg_from_source(source_code: str) -> CFGBuilder:
    """
    Build a control flow graph from Python source code.
    
    Args:
        source_code: Python source code as a string
        
    Returns:
        A CFGBuilder instance with the constructed graph
        
    Raises:
        SyntaxError: If the source code contains syntax errors
    """
    try:
        tree = ast.parse(source_code)
        builder = CFGBuilder()
        builder.visit(tree)
        return builder
    except SyntaxError as e:
        raise SyntaxError(f"Failed to parse source code: {e}")


# --------------------------------------------------
# Data Preparation Functions
# --------------------------------------------------
def prepare_predecessor_data(cfg_builder: CFGBuilder) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare predecessor data for GPU processing.
    
    Convert the graph representation to flat arrays suitable for GPU processing.
    
    Args:
        cfg_builder: A CFG builder with graph data
        
    Returns:
        A tuple of (predecessors array, offsets array)
    """
    predecessors = cfg_builder.predecessors
    num_nodes = cfg_builder.node_count
    
    pred_list: List[int] = []
    pred_offsets: List[int] = [0]
    offset = 0
    
    for node in range(num_nodes):
        preds = predecessors[node]
        pred_list.extend(preds)
        offset += len(preds)
        pred_offsets.append(offset)
        
    return (
        np.array(pred_list, dtype=np.int32),
        np.array(pred_offsets, dtype=np.int32)
    )


def build_dominator_tree(dom: np.ndarray, num_nodes: int, mask_count: int) -> Dict[int, List[int]]:
    """
    Build a dominator tree from dominator bit masks.
    
    Args:
        dom: Computed dominator bit masks
        num_nodes: Number of nodes in the CFG
        mask_count: Number of 64-bit masks per node
        
    Returns:
        A dictionary representing the dominator tree (parent -> [children])
    """
    tree: Dict[int, List[int]] = defaultdict(list)
    
    for node in range(1, num_nodes):
        idom: Optional[int] = None
        
        # Find immediate dominator (closest dominator)
        for i in range(num_nodes):
            mask_index = i // Config.BITS_PER_MASK
            bit_index = i % Config.BITS_PER_MASK
            
            if mask_index < mask_count:
                dom_value = np.uint64(dom[node][mask_index])
                bit_value = np.uint64(1 << bit_index)
                
                # Check if i dominates node
                if (dom_value & bit_value) != 0 and i != node:
                    # Check if i is closer than current idom
                    if idom is None or (np.uint64(dom[idom][mask_index]) & bit_value) == 0:
                        idom = i
                        
        if idom is not None:
            tree[idom].append(node)
            
    return tree


# --------------------------------------------------
# GPU Dominator Computation
# --------------------------------------------------
def create_cuda_module() -> SourceModule:
    """
    Create CUDA module with the dominator computation kernel.
    
    Returns:
        Compiled CUDA module
    """
    cuda_code = f"""
    #define BITS_PER_MASK {Config.BITS_PER_MASK}
    #define MAX_MASK_COUNT {Config.MAX_MASK_COUNT}

    __global__ void compute_dominator(
        unsigned long long *dom,
        int *predecessors,
        int *pred_offsets,
        int num_nodes,
        int num_preds,
        int mask_count,
        int *d_changed)
    {{
        int node = blockIdx.x * blockDim.x + threadIdx.x;
        if (node >= num_nodes) return;
        if (mask_count > MAX_MASK_COUNT) return;

        // Special case for entry node (0)
        if (node == 0) {{
            dom[node * mask_count] = 1ULL << (node % BITS_PER_MASK);
            for (int i = 1; i < mask_count; i++) {{
                dom[node * mask_count + i] = 0;
            }}
            return;
        }}

        // Compute intersection of all predecessors
        unsigned long long intersection[MAX_MASK_COUNT];
        for (int i = 0; i < mask_count; i++) {{
            intersection[i] = ~0ULL;  // Initialize to all 1s
        }}
        
        // Get predecessor range for this node
        int start = pred_offsets[node];
        int end = (node + 1 < num_nodes) ? pred_offsets[node + 1] : num_preds;
        
        // Compute intersection of dominator sets of all predecessors
        for (int i = start; i < end; i++) {{
            int pred = predecessors[i];
            for (int j = 0; j < mask_count; j++) {{
                intersection[j] &= dom[pred * mask_count + j];
            }}
        }}
        
        // Add node itself to its dominator set
        int mask_index = node / BITS_PER_MASK;
        int bit_index = node % BITS_PER_MASK;
        if (mask_index < mask_count) {{
            intersection[mask_index] |= (1ULL << bit_index);
        }}

        // Check if anything changed
        bool changed = false;
        for (int i = 0; i < mask_count; i++) {{
            if (dom[node * mask_count + i] != intersection[i]) {{
                changed = true;
                break;
            }}
        }}
        
        // Update dominator set if changed
        if (changed) {{
            for (int i = 0; i < mask_count; i++) {{
                dom[node * mask_count + i] = intersection[i];
            }}
            atomicExch(&d_changed[0], 1);
        }}
    }}
    """
    
    try:
        return SourceModule(cuda_code)
    except Exception as e:
        raise RuntimeError(f"CUDA module compilation failed: {e}")


def compute_dominators_gpu(cfg_builder: CFGBuilder, block_size: int = Config.DEFAULT_BLOCK_SIZE) -> Tuple[np.ndarray, Dict[int, List[int]], float]:
    """
    Compute dominators using GPU acceleration.
    
    Args:
        cfg_builder: CFG builder with the graph
        block_size: CUDA thread block size
        
    Returns:
        Tuple of (dominator bit masks, dominator tree, execution time in ms)
        
    Raises:
        ValueError: If mask count exceeds MAX_MASK_COUNT
        RuntimeError: For CUDA-related errors
    """
    num_nodes = cfg_builder.node_count
    
    # Prepare data for GPU processing
    predecessors, pred_offsets = prepare_predecessor_data(cfg_builder)
    num_preds = len(predecessors)
    
    # Calculate mask count (number of 64-bit masks per node)
    mask_count = (num_nodes + Config.BITS_PER_MASK - 1) // Config.BITS_PER_MASK
    if mask_count > Config.MAX_MASK_COUNT:
        raise ValueError(f"Node count ({num_nodes}) requires {mask_count} masks, exceeding MAX_MASK_COUNT ({Config.MAX_MASK_COUNT})")
    
    # Initialize dominator sets
    dom = np.zeros((num_nodes, mask_count), dtype=np.uint64)
    dom[0, 0] = 1  # Entry node dominates itself
    
    # Start timing
    start_time = time.time()
    
    # Allocate GPU memory
    try:
        predecessors_gpu = cuda.to_device(predecessors)
        pred_offsets_gpu = cuda.to_device(pred_offsets)
        dom_gpu = cuda.to_device(dom)
        
        # Allocate change flag
        d_changed = cuda.mem_alloc(np.zeros(1, dtype=np.int32).nbytes)
        
        # Compile and get kernel function
        module = create_cuda_module()
        compute_dominator = module.get_function("compute_dominator")
        
        # Calculate grid dimensions
        grid_size = (num_nodes + block_size - 1) // block_size
        
        # Create CUDA stream for asynchronous operations
        stream = cuda.Stream()
        
        # Convergence loop
        iteration_count = 0
        while iteration_count < Config.MAX_ITERATIONS:
            iteration_count += 1
            
            # Reset change flag
            changed_host = np.zeros(1, dtype=np.int32)
            cuda.memcpy_htod_async(d_changed, changed_host, stream)
            
            # Launch kernel
            compute_dominator(
                dom_gpu, predecessors_gpu, pred_offsets_gpu,
                np.int32(num_nodes), np.int32(num_preds), np.int32(mask_count),
                d_changed,
                block=(block_size, 1, 1), grid=(grid_size, 1), stream=stream
            )
            
            # Synchronize and check for changes
            stream.synchronize()
            cuda.memcpy_dtoh(changed_host, d_changed)
            
            if changed_host[0] == 0:
                break
                
        # Copy final result back to host
        cuda.memcpy_dtoh(dom, dom_gpu)
        
        # Free GPU memory
        predecessors_gpu.free()
        pred_offsets_gpu.free()
        dom_gpu.free()
        d_changed.free()
        
    except cuda.Error as e:
        raise RuntimeError(f"CUDA error: {e}")
    
    # End timing
    end_time = time.time()
    execution_time_ms = (end_time - start_time) * 1000
    
    # Build dominator tree
    dom_tree = build_dominator_tree(dom, num_nodes, mask_count)
    
    return dom, dom_tree, execution_time_ms


# --------------------------------------------------
# Visualization Functions
# --------------------------------------------------
def visualize_cfg(cfg_builder: CFGBuilder, filename: str = 'cfg') -> None:
    """
    Visualize the control flow graph using Graphviz.
    
    Args:
        cfg_builder: CFG builder with the graph
        filename: Base filename for the output image
    """
    dot = graphviz.Digraph(comment='Control Flow Graph')
    
    # Add nodes
    for node_id in sorted(cfg_builder.labels.keys()):
        label = cfg_builder.labels[node_id]
        if isinstance(label, ast.AST):
            # For AST nodes, use the class name
            node_label = f"{node_id}: {type(label).__name__}"
            if isinstance(label, ast.FunctionDef):
                node_label += f"\n{label.name}"
        else:
            # For string labels
            node_label = f"{node_id}: {label}"
            
        dot.node(str(node_id), node_label)
    
    # Add edges
    for from_node, to_nodes in cfg_builder.cfg.items():
        for to_node in to_nodes:
            dot.edge(str(from_node), str(to_node))
    
    # Render the graph
    try:
        dot.render(filename, format='png', view=False)
        print(f"CFG saved to {filename}.png")
    except Exception as e:
        print(f"Failed to visualize CFG: {e}")


def visualize_dominator_tree(dom_tree: Dict[int, List[int]], filename: str = 'dominator_tree') -> None:
    """
    Visualize the dominator tree using Graphviz.
    
    Args:
        dom_tree: Dominator tree as a dict (parent -> [children])
        filename: Base filename for the output image
    """
    dot = graphviz.Digraph(comment='Dominator Tree')
    
    # Add all nodes that appear in the tree
    nodes = set(dom_tree.keys())
    for children in dom_tree.values():
        nodes.update(children)
    
    for node in sorted(nodes):
        dot.node(str(node), str(node))
    
    # Add edges
    for parent, children in dom_tree.items():
        for child in children:
            dot.edge(str(parent), str(child))
    
    # Render the graph
    try:
        dot.render(filename, format='png', view=False)
        print(f"Dominator tree saved to {filename}.png")
    except Exception as e:
        print(f"Failed to visualize dominator tree: {e}")


# --------------------------------------------------
# Utility Functions
# --------------------------------------------------
def format_dominator_results(dom: np.ndarray, num_nodes: int) -> List[Set[int]]:
    """
    Convert bit mask representation to sets of dominators for easier reading.
    
    Args:
        dom: Dominator bit masks
        num_nodes: Number of nodes in the CFG
        
    Returns:
        List of sets, where each set contains the dominators of a node
    """
    mask_count = dom.shape[1]
    result: List[Set[int]] = []
    
    for node in range(num_nodes):
        dominators = set()
        for i in range(num_nodes):
            mask_index = i // Config.BITS_PER_MASK
            bit_index = i % Config.BITS_PER_MASK
            
            if mask_index < mask_count:
                dom_value = np.uint64(dom[node][mask_index])
                bit_value = np.uint64(1 << bit_index)
                
                if (dom_value & bit_value) != 0:
                    dominators.add(i)
                    
        result.append(dominators)
        
    return result


def benchmark_dominators(source_code: str, num_runs: int = 10) -> float:
    """
    Benchmark dominator computation on a given source code.
    
    Args:
        source_code: Python source code to analyze
        num_runs: Number of benchmark iterations
        
    Returns:
        Average execution time in milliseconds
    """
    cfg_builder = build_cfg_from_source(source_code)
    
    total_time = 0
    for _ in range(num_runs):
        _, _, time_ms = compute_dominators_gpu(cfg_builder)
        total_time += time_ms
        
    return total_time / num_runs


# --------------------------------------------------
# Main Application
# --------------------------------------------------
def main() -> None:
    """Main entry point for dominator analysis."""
    # Test code example
    source_code_nested = """
def nested_example(a, b, c):
    if a > b:
        if b > c:
            x = a + b
        else:
            x = a - b
    else:
        if a > c:
            y = b + c
        else:
            if c > 10:
                y = 100
            else:
                y = -100
    return x + y
"""

    print("Building CFG...")
    cfg_builder = build_cfg_from_source(source_code_nested)
    
    print("Computing dominators...")
    dominators, dom_tree, execution_time = compute_dominators_gpu(cfg_builder)
    
    print(f"Dominator computation completed in {execution_time:.3f} ms")
    
    # Display results
    print("\nDominator Results:")
    dom_sets = format_dominator_results(dominators, cfg_builder.node_count)
    for i, doms in enumerate(dom_sets):
        print(f"Node {i} is dominated by: {sorted(doms)}")
    
    print("\nDominator Tree:")
    for parent, children in sorted(dom_tree.items()):
        print(f"Node {parent} immediately dominates: {sorted(children)}")
    
    # Visualize the graphs
    visualize_cfg(cfg_builder, filename='cfg_nested_example')
    visualize_dominator_tree(dom_tree, filename='dom_tree_nested_example')
    
    # Run benchmark
    print("\nRunning benchmark...")
    num_runs = 100
    avg_time = benchmark_dominators(source_code_nested, num_runs)
    print(f"Average execution time over {num_runs} runs: {avg_time:.3f} ms")


if __name__ == "__main__":
    main()
