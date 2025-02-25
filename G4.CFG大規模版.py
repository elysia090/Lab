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

# --------------------------------------------------
# Optimized CFG Construction Module
# --------------------------------------------------
class OptimizedCFGBuilder(ast.NodeVisitor):
    """
    Optimized Control Flow Graph Builder.
    Improves upon the basic CFG construction by:
    - Avoiding duplicate edges.
    - Separating function signature node and the first node of the body.
    - Creating merge points only when there are no return statements in the branches.
    - Tracking return nodes.
    """
    def __init__(self):
        self.cfg = defaultdict(list)  # Successor nodes
        self.predecessors = defaultdict(list)  # Predecessor nodes
        self.current_node = 0
        self.node_count = 0
        self.labels = {}  # Node labels
        self.return_nodes = set()  # Track return nodes

    def add_edge(self, from_node: int, to_node: int) -> None:
        """Adds an edge to the CFG, preventing duplicate edges."""
        if to_node not in self.cfg[from_node]:  # Prevent duplicate edges
            self.cfg[from_node].append(to_node)
            self.predecessors[to_node].append(from_node)

    def new_node(self) -> int:
        """Creates a new node and returns its ID."""
        self.node_count += 1
        return self.node_count - 1

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Handles function definitions."""
        self.current_node = 0
        self.node_count = 1
        self.labels[0] = node

        # Separate signature node and first node of body
        entry_node = self.new_node()
        self.labels[entry_node] = f"entry_{node.name}"
        self.add_edge(0, entry_node)
        self.current_node = entry_node

        for stmt in node.body:
            self.visit(stmt)

    def visit_If(self, node: ast.If) -> None:
        """Handles if statements."""
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
            # Create merge point even with no else - only if no return
            if then_end not in self.return_nodes:
                merge_node = self.new_node()
                self.labels[merge_node] = f"merge_{node.lineno}"
                self.add_edge(then_end, merge_node)
                self.add_edge(condition_node, merge_node)  # Direct edge from condition to merge
                self.current_node = merge_node

    def visit_Return(self, node: ast.Return) -> None:
        """Handles return statements."""
        return_node = self.new_node()
        self.labels[return_node] = node
        self.add_edge(self.current_node, return_node)
        self.current_node = return_node
        self.return_nodes.add(return_node)  # Record return node

    def visit_For(self, node: ast.For) -> None:
        """Handles for loops."""
        # Loop header
        header_node = self.new_node()
        self.labels[header_node] = f"for_header_{node.lineno}"
        self.add_edge(self.current_node, header_node)
        self.current_node = header_node

        # Loop body
        body_node = self.new_node()
        self.labels[body_node] = f"for_body_{node.lineno}"
        self.add_edge(header_node, body_node)
        self.current_node = body_node

        for stmt in node.body:
            self.visit(stmt)

        # Loop back
        self.add_edge(self.current_node, header_node)

        # Loop exit
        exit_node = self.new_node()
        self.labels[exit_node] = f"for_exit_{node.lineno}"
        self.add_edge(header_node, exit_node)
        self.current_node = exit_node

        # Else block
        if node.orelse:
            else_node = self.new_node()
            self.labels[else_node] = f"for_else_{node.lineno}"
            self.add_edge(exit_node, else_node)
            self.current_node = else_node

            for stmt in node.orelse:
                self.visit(stmt)


    def visit_While(self, node: ast.While) -> None:
        """Handles while loops."""
        # Loop header
        header_node = self.new_node()
        self.labels[header_node] = f"while_header_{node.lineno}"
        self.add_edge(self.current_node, header_node)
        self.current_node = header_node

        # Loop body
        body_node = self.new_node()
        self.labels[body_node] = f"while_body_{node.lineno}"
        self.add_edge(header_node, body_node)
        self.current_node = body_node

        for stmt in node.body:
            self.visit(stmt)

        # Loop back
        self.add_edge(self.current_node, header_node)

        # Loop exit
        exit_node = self.new_node()
        self.labels[exit_node] = f"while_exit_{node.lineno}"
        self.add_edge(header_node, exit_node)
        self.current_node = exit_node

        # Else block
        if node.orelse:
            else_node = self.new_node()
            self.labels[else_node] = f"while_else_{node.lineno}"
            self.add_edge(exit_node, else_node)
            self.current_node = else_node

            for stmt in node.orelse:
                self.visit(stmt)

    def generic_visit(self, node: ast.AST) -> None:
        """Handles other AST node types."""
        if isinstance(node, ast.FunctionDef):
            for stmt in node.body:
                self.visit(stmt)
        elif isinstance(node, (ast.If, ast.For, ast.While, ast.Return)):
            # Already have specific visit methods, so do nothing
            pass
        elif isinstance(node, list):
            for item in node:
                if isinstance(item, ast.AST):
                    self.visit(item)

        else:
            current = self.current_node
            next_node = self.new_node()
            self.labels[next_node] = type(node).__name__
            self.add_edge(current, next_node)
            self.current_node = next_node

            for field, value in ast.iter_fields(node):
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, ast.AST):
                            self.visit(item)
                elif isinstance(value, ast.AST):
                    self.visit(value)

# --------------------------------------------------
# Optimized GPU Computation
# --------------------------------------------------

def prepare_predecessor_data_optimized(cfg, predecessors, num_nodes):
    """Prepares predecessor data for GPU computation (parallel version)."""
    pred_list = []
    pred_offsets = [0]

    # Build predecessor list in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_node = {
            executor.submit(lambda n: (n, predecessors[n]), node): node
            for node in range(num_nodes)
        }

        for future in concurrent.futures.as_completed(future_to_node):
            node, preds = future.result()
            pred_list.extend(preds)
            pred_offsets.append(len(pred_list))

    return np.array(pred_list, dtype=np.int32), np.array(pred_offsets, dtype=np.int32)

def compute_dominators_optimized(source_code: str) -> Tuple[np.ndarray, Dict[int, List[int]], float]:
    """Optimized dominator computation."""
    # Build CFG
    tree = ast.parse(source_code)
    builder = OptimizedCFGBuilder()
    builder.visit(tree)
    num_nodes = builder.node_count

    # Prepare predecessor data
    predecessors, pred_offsets = prepare_predecessor_data_optimized(builder.cfg, builder.predecessors, num_nodes)
    num_preds = len(predecessors)

    # Calculate number of masks
    mask_count = (num_nodes + BITS_PER_MASK - 1) // BITS_PER_MASK
    if mask_count > MAX_MASK_COUNT:
        raise ValueError(f"mask_count ({mask_count}) exceeds MAX_MASK_COUNT ({MAX_MASK_COUNT})")

    # Initialize dominator sets
    dom = np.zeros((num_nodes, mask_count), dtype=np.uint64)
    dom[0, 0] = 1  # Entry node dominates itself

    # Allocate GPU memory
    start_time = time.time()

    # Use pinned memory for faster host-device transfers
    pinned_predecessors = cuda.pagelocked_empty_like(predecessors)
    pinned_predecessors[:] = predecessors
    pinned_pred_offsets = cuda.pagelocked_empty_like(pred_offsets)
    pinned_pred_offsets[:] = pred_offsets
    pinned_dom = cuda.pagelocked_empty_like(dom)
    pinned_dom[:] = dom

    # Transfer to device memory
    predecessors_gpu = cuda.to_device(pinned_predecessors)
    pred_offsets_gpu = cuda.to_device(pinned_pred_offsets)
    dom_gpu = cuda.to_device(pinned_dom)

    # Change flag
    d_changed = cuda.mem_alloc(np.zeros(1, dtype=np.int32).nbytes)

    # Compile kernel
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
            dom[0] = 1ULL;
            for (int i = 1; i < mask_count; i++) {
                dom[i] = 0;
            }
            return;
        }

        // Use shared memory
        __shared__ unsigned long long shared_intersection[MAX_MASK_COUNT * 32];
        int warp_id = threadIdx.x / 32;
        int lane_id = threadIdx.x % 32;
        unsigned long long *my_intersection = &shared_intersection[warp_id * MAX_MASK_COUNT];

        // Initialize
        #pragma unroll
        for (int i = 0; i < MAX_MASK_COUNT; i++) {
            if (lane_id == 0) {
                my_intersection[i] = ~0ULL;
            }
        }
        __syncwarp();

        int start = pred_offsets[node];
        int end = (node + 1 < num_nodes) ? pred_offsets[node + 1] : num_preds;

        for (int i = start; i < end; i++) {
            int pred = predecessors[i];
            #pragma unroll
            for (int j = 0; j < mask_count; j++) {
                my_intersection[j] &= dom[pred * mask_count + j];
            }
        }

        int mask_index = node / BITS_PER_MASK;
        int bit_index = node % BITS_PER_MASK;
        if (mask_index < mask_count) {
            my_intersection[mask_index] |= (1ULL << bit_index);
        }

        bool changed = false;
        #pragma unroll
        for (int i = 0; i < mask_count; i++) {
            if (dom[node * mask_count + i] != my_intersection[i]) {
                changed = true;
                break;
            }
        }

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

    # Optimize block and grid sizes
    block_size = 256  # Threads per block
    grid_size = (num_nodes + block_size - 1) // block_size # Blocks

    # Use multiple streams for parallelism
    stream1 = cuda.Stream()
    stream2 = cuda.Stream()

    # Track number of iterations
    iterations = 0
    max_iterations = 100  # For safety

    # Convergence loop
    while iterations < max_iterations:
        iterations += 1
        changed_host = np.zeros(1, dtype=np.int32)
        cuda.memcpy_htod_async(d_changed, changed_host, stream1)

        # Kernel execution
        compute_dominator(
            dom_gpu, predecessors_gpu, pred_offsets_gpu,
            np.int32(num_nodes), np.int32(num_preds), np.int32(mask_count),
            d_changed,
            block=(block_size, 1, 1), grid=(grid_size, 1), stream=stream1
        )

        # Check change flag
        cuda.memcpy_dtoh_async(changed_host, d_changed, stream2)
        stream1.synchronize()
        stream2.synchronize()

        if changed_host[0] == 0:
            break

    # Copy final results to host
    cuda.memcpy_dtoh(pinned_dom, dom_gpu)
    dom = np.array(pinned_dom)  # Copy from pinned memory

    end_time = time.time()
    elapsed_time = (end_time - start_time) * 1000  # Milliseconds

    # Build dominator tree
    dom_tree = build_dominator_tree_optimized(dom, num_nodes, mask_count)

    return dom, dom_tree, elapsed_time

def build_dominator_tree_optimized(dom, num_nodes, mask_count):
    """Optimized dominator tree construction."""
    tree = defaultdict(list)

    # Use parallelism
    def find_idom(node):
        if node == 0:
            return None, None  # Root node has no dominator

        idom = None
        for i in range(num_nodes):
            mask_index = i // BITS_PER_MASK
            bit_index = i % BITS_PER_MASK
            if mask_index < mask_count:
                dom_value = np.uint64(dom[node][mask_index])
                bit_value = np.uint64(1 << bit_index)
                if (dom_value & bit_value) != 0 and i != node:
                    # Choose closest dominator among candidates
                    if idom is None or all(
                        (np.uint64(dom[idom][j]) & np.uint64(dom[i][j])) == np.uint64(dom[i][j])
                        for j in range(mask_count)
                    ):
                        idom = i
        return node, idom

    # Calculate idom in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for node, idom in executor.map(find_idom, range(1, num_nodes)):
            if idom is not None:
                tree[idom].append(node)

    return tree

# --------------------------------------------------
# Optimized Visualization
# --------------------------------------------------
def visualize_cfg_optimized(cfg, labels, filename='cfg', format='png'):
    """Optimized CFG visualization function."""
    dot = graphviz.Digraph(comment='Control Flow Graph', format=format)
    dot.attr(rankdir='TB', size='8,8', dpi='300')  # Top-to-bottom layout

    # Set colors based on node type
    for node_id in sorted(labels.keys()):
        label = str(labels[node_id])
        shape = 'box'
        color = 'black'
        style = 'filled'
        fillcolor = 'white'

        if 'FunctionDef' in label:
            shape = 'oval'  # Function definition
            fillcolor = '#FFDDDD' # Light red
        elif 'if' in label or 'then' in label or 'else' in label:
            shape = 'diamond'  # Conditional
            fillcolor = '#DDFFDD' # Light green
        elif 'for' in label or 'while' in label:
            shape = 'hexagon'  # Loop
            fillcolor = '#DDDDFF' # Light blue
        elif 'Return' in label:
            shape = 'box'      # Return statement
            fillcolor = '#FFDDFF'# Light purple

        dot.node(
            str(node_id),
            label=f"{node_id}: {label}",
            shape=shape,
            color=color,
            style=style,
            fillcolor=fillcolor
        )

    # Add edges
    for from_node, to_nodes in cfg.items():
        for to_node in to_nodes:
            dot.edge(str(from_node), str(to_node))

    # Render (file extension is automatically added)
    dot.render(filename, view=False)
    print(f"Saved CFG to {filename}.{format}")

    return dot


# --------------------------------------------------
# Main Function
# --------------------------------------------------
if __name__ == "__main__":
    source_code_complex = """
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


    # Optimized CFG Visualization
    opt_builder = OptimizedCFGBuilder()
    tree = ast.parse(source_code_complex)
    opt_builder.visit(tree)
    visualize_cfg_optimized(opt_builder.cfg, opt_builder.labels, filename='optimized_complex_cfg')

    # Compute dominators (Optimized)
    dom, dom_tree, elapsed_time = compute_dominators_optimized(source_code_complex)
    print("\nDominator Tree:")
    for parent, children in dom_tree.items():
        print(f"Node {parent} → {sorted(children)}")

    print(f"\nDominator computation time: {elapsed_time:.3f} ms")
