"""
Python's ast is an abstract syntax tree specifically for Python code, so when dealing with C/C++ code,
you can use Clang's Python bindings (clang.cindex) to generate and traverse the abstract syntax tree.

In Clang AST, the node types are
CursorKind.FUNCTION_DECL,
CursorKind.IF_STMT,
CursorKind.FOR_STMT,
CursorKind.WHILE_STMT, etc., and 
they need to be mapped to Python's ast.If / ast.For, etc.

The CFG construction logic (node ​​generation, edge addition, etc.)
and the Dominator calculation part (bit vector operations on the GPU, etc.) 
can basically be shared between Python AST and Clang AST.
source：[https://libclang.readthedocs.io/en/latest/]

The only thing that changes is "which nodes are traversed and how."
"""

import ast
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from collections import defaultdict
from typing import List, Dict, Tuple
import time
import graphviz
import concurrent.futures

# Constants
BITS_PER_MASK = 64
MAX_MASK_COUNT = 10

class OptimizedCFGBuilder(ast.NodeVisitor):
    """Optimized Control Flow Graph Builder."""
    def __init__(self):
        self.cfg = defaultdict(list)
        self.predecessors = defaultdict(list)
        self.current_node = 0
        self.node_count = 0
        self.labels = {}
        self.return_nodes = set()

    def add_edge(self, from_node: int, to_node: int) -> None:
        if to_node not in self.cfg[from_node]:
            self.cfg[from_node].append(to_node)
            self.predecessors[to_node].append(from_node)

    def new_node(self) -> int:
        self.node_count += 1
        return self.node_count - 1

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.current_node = 0
        self.node_count = 1
        self.labels[0] = node
        
        entry_node = self.new_node()
        self.labels[entry_node] = f"entry_{node.name}"
        self.add_edge(0, entry_node)
        self.current_node = entry_node
        
        for stmt in node.body:
            self.visit(stmt)

    def visit_If(self, node: ast.If) -> None:
        condition_node = self.current_node
        
        # Then branch
        then_node = self.new_node()
        self.labels[then_node] = f"then_{node.lineno}"
        self.add_edge(condition_node, then_node)
        self.current_node = then_node
        
        for stmt in node.body:
            self.visit(stmt)
            
        then_end = self.current_node
        
        # Else branch
        if node.orelse:
            else_node = self.new_node()
            self.labels[else_node] = f"else_{node.lineno}"
            self.add_edge(condition_node, else_node)
            self.current_node = else_node
            
            for stmt in node.orelse:
                self.visit(stmt)
                
            else_end = self.current_node
            
            # Merge point - only if there are no returns
            if then_end not in self.return_nodes and else_end not in self.return_nodes:
                merge_node = self.new_node()
                self.labels[merge_node] = f"merge_{node.lineno}"
                self.add_edge(then_end, merge_node)
                self.add_edge(else_end, merge_node)
                self.current_node = merge_node
        else:
            # Create merge point if no return
            if then_end not in self.return_nodes:
                merge_node = self.new_node()
                self.labels[merge_node] = f"merge_{node.lineno}"
                self.add_edge(then_end, merge_node)
                self.add_edge(condition_node, merge_node)
                self.current_node = merge_node

    def visit_Return(self, node: ast.Return) -> None:
        return_node = self.new_node()
        self.labels[return_node] = node
        self.add_edge(self.current_node, return_node)
        self.current_node = return_node
        self.return_nodes.add(return_node)

    def _process_loop(self, node, loop_type):
        """Handle common loop processing for For and While"""
        header_node = self.new_node()
        self.labels[header_node] = f"{loop_type}_header_{node.lineno}"
        self.add_edge(self.current_node, header_node)
        self.current_node = header_node
        
        body_node = self.new_node()
        self.labels[body_node] = f"{loop_type}_body_{node.lineno}"
        self.add_edge(header_node, body_node)
        self.current_node = body_node
        
        for stmt in node.body:
            self.visit(stmt)
            
        self.add_edge(self.current_node, header_node)
        
        exit_node = self.new_node()
        self.labels[exit_node] = f"{loop_type}_exit_{node.lineno}"
        self.add_edge(header_node, exit_node)
        self.current_node = exit_node
        
        if node.orelse:
            else_node = self.new_node()
            self.labels[else_node] = f"{loop_type}_else_{node.lineno}"
            self.add_edge(exit_node, else_node)
            self.current_node = else_node
            
            for stmt in node.orelse:
                self.visit(stmt)

    def visit_For(self, node: ast.For) -> None:
        self._process_loop(node, "for")

    def visit_While(self, node: ast.While) -> None:
        self._process_loop(node, "while")

    def generic_visit(self, node: ast.AST) -> None:
        if isinstance(node, (ast.FunctionDef, ast.If, ast.For, ast.While, ast.Return)):
            pass  # Already have specific visit methods
        elif isinstance(node, list):
            for item in node:
                if isinstance(item, ast.AST):
                    self.visit(item)
        else:
            next_node = self.new_node()
            self.labels[next_node] = type(node).__name__
            self.add_edge(self.current_node, next_node)
            self.current_node = next_node
            
            for field, value in ast.iter_fields(node):
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, ast.AST):
                            self.visit(item)
                elif isinstance(value, ast.AST):
                    self.visit(value)

def compute_dominators_optimized(source_code: str) -> Tuple[np.ndarray, Dict[int, List[int]], float]:
    # Build CFG
    tree = ast.parse(source_code)
    builder = OptimizedCFGBuilder()
    builder.visit(tree)
    num_nodes = builder.node_count
    
    # Prepare predecessor data
    pred_list = []
    pred_offsets = [0]
    
    for node in range(num_nodes):
        pred_list.extend(builder.predecessors[node])
        pred_offsets.append(len(pred_list))
    
    predecessors = np.array(pred_list, dtype=np.int32)
    pred_offsets = np.array(pred_offsets, dtype=np.int32)
    num_preds = len(predecessors)
    
    # Calculate masks needed
    mask_count = (num_nodes + BITS_PER_MASK - 1) // BITS_PER_MASK
    if mask_count > MAX_MASK_COUNT:
        raise ValueError(f"mask_count ({mask_count}) exceeds MAX_MASK_COUNT ({MAX_MASK_COUNT})")
    
    # Initialize dominator sets
    dom = np.zeros((num_nodes, mask_count), dtype=np.uint64)
    dom[0, 0] = 1  # Entry node dominates itself
    
    # GPU computation setup
    start_time = time.time()
    
    # Using pinned memory for faster transfers
    predecessors_gpu = cuda.to_device(predecessors)
    pred_offsets_gpu = cuda.to_device(pred_offsets)
    dom_gpu = cuda.to_device(dom)
    
    # Change flag
    d_changed = cuda.mem_alloc(np.zeros(1, dtype=np.int32).nbytes)
    
    # CUDA kernel
    module = SourceModule("""
    #define BITS_PER_MASK 64
    #define MAX_MASK_COUNT 10
    
    __global__ void compute_dominator_optimized(
        unsigned long long *dom,
        int *predecessors,
        int *pred_offsets,
        int num_nodes,
        int num_preds,
        int mask_count,
        int *d_changed)
    {
        int node = blockIdx.x * blockDim.x + threadIdx.x;
        if (node >= num_nodes) return;
        if (mask_count > MAX_MASK_COUNT) return;
        
        if (node == 0) {
            // Root node dominates only itself
            dom[0] = 1ULL;
            for (int i = 1; i < mask_count; i++) {
                dom[i] = 0;
            }
            return;
        }
        
        __shared__ unsigned long long shared_intersection[MAX_MASK_COUNT * 32];
        int warp_id = threadIdx.x / 32;
        int lane_id = threadIdx.x % 32;
        unsigned long long *my_intersection = &shared_intersection[warp_id * MAX_MASK_COUNT];
        
        // Initialize intersection to all 1s
        #pragma unroll
        for (int i = 0; i < mask_count; i++) {
            if (lane_id == 0) {
                my_intersection[i] = ~0ULL;
            }
        }
        __syncwarp();
        
        // Compute intersection of all predecessors
        int start = pred_offsets[node];
        int end = (node + 1 < num_nodes) ? pred_offsets[node + 1] : num_preds;
        
        for (int i = start; i < end; i++) {
            int pred = predecessors[i];
            #pragma unroll
            for (int j = 0; j < mask_count; j++) {
                my_intersection[j] &= dom[pred * mask_count + j];
            }
        }
        
        // Add node itself to its dominators
        int mask_index = node / BITS_PER_MASK;
        int bit_index = node % BITS_PER_MASK;
        if (mask_index < mask_count) {
            my_intersection[mask_index] |= (1ULL << bit_index);
        }
        
        // Check for changes
        bool changed = false;
        #pragma unroll
        for (int i = 0; i < mask_count; i++) {
            if (dom[node * mask_count + i] != my_intersection[i]) {
                changed = true;
                break;
            }
        }
        
        // Update if changed
        if (changed) {
            #pragma unroll
            for (int i = 0; i < mask_count; i++) {
                dom[node * mask_count + i] = my_intersection[i];
            }
            atomicExch(d_changed, 1);
        }
    }
    """)
    
    compute_dominator = module.get_function("compute_dominator_optimized")
    
    # Optimize thread block size
    block_size = 256
    grid_size = (num_nodes + block_size - 1) // block_size
    
    # Convergence loop
    iterations = 0
    max_iterations = 100
    
    while iterations < max_iterations:
        iterations += 1
        changed_host = np.zeros(1, dtype=np.int32)
        cuda.memcpy_htod(d_changed, changed_host)
        
        # Execute kernel
        compute_dominator(
            dom_gpu, predecessors_gpu, pred_offsets_gpu,
            np.int32(num_nodes), np.int32(num_preds), np.int32(mask_count),
            d_changed,
            block=(block_size, 1, 1), grid=(grid_size, 1)
        )
        
        # Check for convergence
        cuda.memcpy_dtoh(changed_host, d_changed)
        if changed_host[0] == 0:
            break
    
    # Get results
    cuda.memcpy_dtoh(dom, dom_gpu)
    
    end_time = time.time()
    elapsed_time = (end_time - start_time) * 1000  # Milliseconds
    
    # Build dominator tree
    tree = defaultdict(list)
    
    # Find immediate dominators
    for node in range(1, num_nodes):
        idom = None
        for i in range(num_nodes):
            if i == node: continue
            
            # Check if i dominates node
            mask_index = i // BITS_PER_MASK
            bit_index = i % BITS_PER_MASK
            
            if mask_index >= mask_count: continue
            
            dom_value = np.uint64(dom[node][mask_index])
            bit_value = np.uint64(1 << bit_index)
            
            if (dom_value & bit_value) != 0:
                # Find closest dominator
                if idom is None or all(
                    (dom[idom][j] & dom[i][j]) == dom[i][j]
                    for j in range(mask_count)
                ):
                    idom = i
        
        if idom is not None:
            tree[idom].append(node)
    
    return dom, tree, elapsed_time

def visualize_cfg(cfg, labels, filename='cfg', format='png'):
    dot = graphviz.Digraph(comment='Control Flow Graph', format=format)
    dot.attr(rankdir='TB', size='8,8', dpi='300')
    
    # Node styling
    node_styles = {
        'function': {'shape': 'oval', 'fillcolor': '#FFDDDD'},
        'conditional': {'shape': 'diamond', 'fillcolor': '#DDFFDD'},
        'loop': {'shape': 'hexagon', 'fillcolor': '#DDDDFF'},
        'return': {'shape': 'box', 'fillcolor': '#FFDDFF'},
        'default': {'shape': 'box', 'fillcolor': 'white'}
    }
    
    for node_id in sorted(labels.keys()):
        label = str(labels[node_id])
        
        # Determine node type
        node_type = 'default'
        if 'FunctionDef' in label:
            node_type = 'function'
        elif any(x in label for x in ['if', 'then', 'else']):
            node_type = 'conditional'
        elif any(x in label for x in ['for', 'while']):
            node_type = 'loop'
        elif 'Return' in label:
            node_type = 'return'
        
        # Apply style
        style = node_styles[node_type]
        dot.node(
            str(node_id),
            label=f"{node_id}: {label}",
            shape=style['shape'],
            style='filled',
            fillcolor=style['fillcolor']
        )
    
    # Add edges
    for from_node, to_nodes in cfg.items():
        for to_node in to_nodes:
            dot.edge(str(from_node), str(to_node))
    
    # Render
    dot.render(filename, view=False)
    return dot

if __name__ == "__main__":
    source_code = """
def complex_example(a, b, c, d, e):
    x = 0
    y = 0

    if a > b:
        if b > c:
            x = a + b
            while d > 0:
                x += 1
                d -= 1
        else:
            x = a - b
            for i in range(10):
                if i % 2 == 0:
                    x += i
    else:
        if a > c:
            y = b + c
            for j in range(e):
                if j > 5:
                    break
                y += j
        else:
            if c > 10:
                y = 100
                while e > 0:
                    y -= 1
                    e -= 1
            else:
                y = -100

    if x > y:
        return x
    else:
        return y
"""

    # Build and visualize CFG
    builder = OptimizedCFGBuilder()
    tree = ast.parse(source_code)
    builder.visit(tree)
    visualize_cfg(builder.cfg, builder.labels, filename='optimized_cfg')
    
    # Compute dominators
    dom, dom_tree, elapsed_time = compute_dominators_optimized(source_code)
    print("\nDominator Tree:")
    for parent, children in dom_tree.items():
        print(f"Node {parent} → {sorted(children)}")
    
    print(f"\nDominator computation time: {elapsed_time:.3f} ms")
