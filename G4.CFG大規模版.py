"""
Control Flow Graph Analysis with GPU-Accelerated Dominator Computation

This module provides tools to:
1. Build a Control Flow Graph (CFG) from Python source code
2. Compute dominators on GPU using CUDA
3. Visualize the CFG and dominator tree
"""

import ast
import time
from collections import defaultdict
from typing import Dict, List, Tuple, Union, Optional, Set

import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import graphviz

# =========================================================
# Constants
# =========================================================
BITS_PER_MASK = 64
MAX_MASK_COUNT = 10
DEFAULT_BLOCK_SIZE = 256
MAX_ITERATIONS = 100

# =========================================================
# Data Structures
# =========================================================
class ControlFlowGraph:
    """Represents a Control Flow Graph with nodes and edges."""
    
    def __init__(self):
        self.successors: Dict[int, List[int]] = defaultdict(list)
        self.predecessors: Dict[int, List[int]] = defaultdict(list)
        self.node_labels: Dict[int, Union[ast.AST, str]] = {}
        self.node_count: int = 0
    
    def add_node(self, label: Union[ast.AST, str]) -> int:
        """Add a node to the CFG and return its ID."""
        node_id = self.node_count
        self.node_count += 1
        self.node_labels[node_id] = label
        return node_id
    
    def add_edge(self, from_node: int, to_node: int) -> None:
        """Add a directed edge between nodes in the CFG."""
        if to_node not in self.successors[from_node]:
            self.successors[from_node].append(to_node)
        if from_node not in self.predecessors[to_node]:
            self.predecessors[to_node].append(from_node)
    
    def visualize(self, filename: str = 'cfg') -> None:
        """Generate a visual representation of the CFG using Graphviz."""
        dot = graphviz.Digraph(comment='Control Flow Graph')
        
        # Add nodes with labels
        for node_id, label in self.node_labels.items():
            if isinstance(label, ast.AST):
                node_label = f"{node_id}: {type(label).__name__}"
                # Add line number if available
                if hasattr(label, 'lineno'):
                    node_label += f" (line {label.lineno})"
            else:
                node_label = f"{node_id}: {label}"
            
            dot.node(str(node_id), node_label)
        
        # Add edges
        for from_node, to_nodes in self.successors.items():
            for to_node in to_nodes:
                dot.edge(str(from_node), str(to_node))
        
        # Render the graph
        dot.render(filename, format='png', view=False)
        print(f"CFG visualization saved to {filename}.png")
    
    def to_gpu_format(self) -> Tuple[np.ndarray, np.ndarray]:
        """Convert graph to format suitable for GPU processing."""
        pred_list = []
        pred_offsets = [0]
        offset = 0
        
        for node in range(self.node_count):
            preds = self.predecessors[node]
            pred_list.extend(preds)
            offset += len(preds)
            pred_offsets.append(offset)
            
        return (
            np.array(pred_list, dtype=np.int32),
            np.array(pred_offsets, dtype=np.int32)
        )

# =========================================================
# CFG Construction
# =========================================================
class CFGBuilder(ast.NodeVisitor):
    """Constructs a Control Flow Graph from Python AST."""
    
    def __init__(self):
        self.cfg = ControlFlowGraph()
        self.current_node: int = -1
    
    def build(self, source_code: str) -> ControlFlowGraph:
        """Build a CFG from source code."""
        tree = ast.parse(source_code)
        self.cfg = ControlFlowGraph()
        self.current_node = self.cfg.add_node("Entry")
        self.visit(tree)
        return self.cfg
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Process a function definition."""
        # Mark the current node as the function entry
        self.cfg.node_labels[self.current_node] = node
        
        # Visit function body
        for stmt in node.body:
            self.visit(stmt)
    
    def visit_If(self, node: ast.If) -> None:
        """Process an if statement with then/else branches."""
        condition_node = self.cfg.add_node(node)
        self.cfg.add_edge(self.current_node, condition_node)
        prev_node = self.current_node
        self.current_node = condition_node
        
        # Then branch
        then_node = self.cfg.add_node(f"then_branch_{node.lineno}")
        self.cfg.add_edge(condition_node, then_node)
        self.current_node = then_node
        
        for stmt in node.body:
            self.visit(stmt)
        
        then_exit = self.current_node
        
        # Else branch (if present)
        if node.orelse:
            else_node = self.cfg.add_node(f"else_branch_{node.lineno}")
            self.cfg.add_edge(condition_node, else_node)
            self.current_node = else_node
            
            for stmt in node.orelse:
                self.visit(stmt)
            
            else_exit = self.current_node
            
            # Create merge point
            merge_node = self.cfg.add_node(f"if_merge_{node.lineno}")
            self.cfg.add_edge(then_exit, merge_node)
            self.cfg.add_edge(else_exit, merge_node)
            self.current_node = merge_node
        else:
            # No else branch - direct path from condition to merge
            merge_node = self.cfg.add_node(f"if_merge_{node.lineno}")
            self.cfg.add_edge(then_exit, merge_node)
            self.cfg.add_edge(condition_node, merge_node)
            self.current_node = merge_node
    
    def visit_While(self, node: ast.While) -> None:
        """Process a while loop."""
        # Create condition node
        condition_node = self.cfg.add_node(node)
        self.cfg.add_edge(self.current_node, condition_node)
        self.current_node = condition_node
        
        # Create body entry node
        body_node = self.cfg.add_node(f"while_body_{node.lineno}")
        self.cfg.add_edge(condition_node, body_node)
        self.current_node = body_node
        
        # Process body statements
        for stmt in node.body:
            self.visit(stmt)
        
        # Loop back to condition
        self.cfg.add_edge(self.current_node, condition_node)
        
        # Create exit node
        exit_node = self.cfg.add_node(f"while_exit_{node.lineno}")
        self.cfg.add_edge(condition_node, exit_node)
        self.current_node = exit_node
        
        # Process else clause if present
        if node.orelse:
            else_node = self.cfg.add_node(f"while_else_{node.lineno}")
            self.cfg.add_edge(exit_node, else_node)
            self.current_node = else_node
            
            for stmt in node.orelse:
                self.visit(stmt)
    
    def visit_For(self, node: ast.For) -> None:
        """Process a for loop."""
        # Create initialization node
        init_node = self.cfg.add_node(node)
        self.cfg.add_edge(self.current_node, init_node)
        self.current_node = init_node
        
        # Create body entry node
        body_node = self.cfg.add_node(f"for_body_{node.lineno}")
        self.cfg.add_edge(init_node, body_node)
        self.current_node = body_node
        
        # Process body statements
        for stmt in node.body:
            self.visit(stmt)
        
        # Loop back to init for next iteration
        self.cfg.add_edge(self.current_node, init_node)
        
        # Create exit node for when loop terminates
        exit_node = self.cfg.add_node(f"for_exit_{node.lineno}")
        self.cfg.add_edge(init_node, exit_node)
        self.current_node = exit_node
        
        # Process else clause if present
        if node.orelse:
            else_node = self.cfg.add_node(f"for_else_{node.lineno}")
            self.cfg.add_edge(exit_node, else_node)
            self.current_node = else_node
            
            for stmt in node.orelse:
                self.visit(stmt)
    
    def visit_Return(self, node: ast.Return) -> None:
        """Process a return statement."""
        return_node = self.cfg.add_node(node)
        self.cfg.add_edge(self.current_node, return_node)
        self.current_node = return_node
    
    def generic_visit(self, node: ast.AST) -> None:
        """Process other AST node types."""
        # For statement-like nodes that aren't specially handled
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.Expr, ast.Assert)):
            stmt_node = self.cfg.add_node(node)
            self.cfg.add_edge(self.current_node, stmt_node)
            self.current_node = stmt_node
        else:
            # Default behavior
            super().generic_visit(node)

# =========================================================
# GPU Dominator Computation
# =========================================================
class DominatorAnalyzer:
    """Computes dominator sets and trees using GPU acceleration."""
    
    def __init__(self):
        self.cuda_code = self._generate_cuda_code()
        self.module = SourceModule(self.cuda_code)
        self.compute_dominator = self.module.get_function("compute_dominator")
    
    def _generate_cuda_code(self) -> str:
        """Generate CUDA C code for dominator computation."""
        return f"""
        #define BITS_PER_MASK {BITS_PER_MASK}
        #define MAX_MASK_COUNT {MAX_MASK_COUNT}

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

            // Entry node (0) dominates only itself
            if (node == 0) {{
                dom[node * mask_count] = 1ULL << (node % BITS_PER_MASK);
                for (int i = 1; i < mask_count; i++) {{
                    dom[node * mask_count + i] = 0;
                }}
                return;
            }}

            // Initialize intersection to universal set (all bits set)
            unsigned long long intersection[MAX_MASK_COUNT];
            for (int i = 0; i < mask_count; i++) {{
                intersection[i] = ~0ULL;
            }}
            
            // Compute intersection of predecessors' dominator sets
            int start = pred_offsets[node];
            int end = (node + 1 < num_nodes) ? pred_offsets[node + 1] : num_preds;
            
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

            // Check if dominator set has changed
            bool changed = false;
            for (int i = 0; i < mask_count; i++) {{
                if (dom[node * mask_count + i] != intersection[i]) {{
                    changed = true;
                    break;
                }}
            }}
            
            // Update dominator set if it changed
            if (changed) {{
                for (int i = 0; i < mask_count; i++) {{
                    dom[node * mask_count + i] = intersection[i];
                }}
                atomicExch(&d_changed[0], 1);  // Signal that something changed
            }}
        }}
        """
    
    def compute(self, cfg: ControlFlowGraph) -> Tuple[np.ndarray, Dict[int, List[int]], float]:
        """
        Compute dominators for a CFG using GPU.
        
        Returns:
            Tuple of (dominator_sets, dominator_tree, execution_time_ms)
        """
        num_nodes = cfg.node_count
        if num_nodes == 0:
            raise ValueError("Empty CFG - no nodes found")
        
        # Prepare data for GPU
        pred_list, pred_offsets = cfg.to_gpu_format()
        num_preds = len(pred_list)
        
        # Calculate number of bit masks needed
        mask_count = (num_nodes + BITS_PER_MASK - 1) // BITS_PER_MASK
        if mask_count > MAX_MASK_COUNT:
            raise ValueError(f"Too many nodes: mask_count ({mask_count}) exceeds MAX_MASK_COUNT ({MAX_MASK_COUNT})")
        
        # Start timing
        start_time = time.time()
        
        # Initialize dominator sets
        dom = np.zeros((num_nodes, mask_count), dtype=np.uint64)
        dom[0, 0] = 1  # Entry node dominates itself
        
        # Allocate GPU memory
        try:
            predecessors_gpu = cuda.to_device(pred_list)
            pred_offsets_gpu = cuda.to_device(pred_offsets)
            dom_gpu = cuda.to_device(dom)
            d_changed = cuda.mem_alloc(np.zeros(1, dtype=np.int32).nbytes)
            
            # Configure kernel execution
            block_size = DEFAULT_BLOCK_SIZE
            grid_size = (num_nodes + block_size - 1) // block_size
            stream = cuda.Stream()
            
            # Fixed-point iteration
            iteration_count = 0
            while iteration_count < MAX_ITERATIONS:
                iteration_count += 1
                
                # Reset changed flag
                changed_host = np.zeros(1, dtype=np.int32)
                cuda.memcpy_htod_async(d_changed, changed_host, stream)
                
                # Run kernel
                self.compute_dominator(
                    dom_gpu, predecessors_gpu, pred_offsets_gpu,
                    np.int32(num_nodes), np.int32(num_preds), np.int32(mask_count),
                    d_changed,
                    block=(block_size, 1, 1), grid=(grid_size, 1), stream=stream
                )
                stream.synchronize()
                
                # Check if anything changed
                cuda.memcpy_dtoh(changed_host, d_changed)
                if changed_host[0] == 0:
                    break
            
            # Get final result
            cuda.memcpy_dtoh(dom, dom_gpu)
            
        finally:
            # Clean up GPU resources
            try:
                predecessors_gpu.free()
                pred_offsets_gpu.free()
                dom_gpu.free()
                d_changed.free()
            except:
                pass
        
        end_time = time.time()
        elapsed_ms = (end_time - start_time) * 1000
        
        # Build dominator tree
        dom_tree = self._build_dominator_tree(dom, num_nodes, mask_count)
        
        return dom, dom_tree, elapsed_ms
    
    def _build_dominator_tree(self, dom: np.ndarray, num_nodes: int, mask_count: int) -> Dict[int, List[int]]:
        """Build a dominator tree from dominator sets."""
        tree = defaultdict(list)
        
        # For each node (except entry), find its immediate dominator
        for node in range(1, num_nodes):
            idom = None
            
            for i in range(num_nodes):
                mask_index = i // BITS_PER_MASK
                bit_index = i % BITS_PER_MASK
                
                if mask_index >= mask_count:
                    continue
                
                dom_value = np.uint64(dom[node][mask_index])
                bit_value = np.uint64(1 << bit_index)
                
                # Check if i dominates node and could be immediate dominator
                if (dom_value & bit_value) != 0 and i != node:
                    # Check if i is more specific than current idom
                    if idom is None:
                        idom = i
                    else:
                        # Check if i is dominated by current idom
                        idom_mask_index = idom // BITS_PER_MASK
                        idom_bit_index = idom % BITS_PER_MASK
                        
                        if idom_mask_index < mask_count:
                            if (np.uint64(dom[i][idom_mask_index]) & 
                                np.uint64(1 << idom_bit_index)) != 0:
                                # i is dominated by idom, so i is more specific
                                idom = i
            
            if idom is not None:
                tree[idom].append(node)
        
        return tree

# =========================================================
# Main Analysis API
# =========================================================
class CFGAnalyzer:
    """Main API for CFG analysis."""
    
    def __init__(self):
        self.builder = CFGBuilder()
        self.dominator_analyzer = DominatorAnalyzer()
    
    def analyze(self, source_code: str, visualize: bool = True) -> Dict[str, object]:
        """
        Analyze Python source code to compute CFG and dominators.
        
        Args:
            source_code: Python source code to analyze
            visualize: Whether to generate a visualization
            
        Returns:
            Dictionary with analysis results
        """
        # Build CFG
        cfg = self.builder.build(source_code)
        
        # Compute dominators
        dom_sets, dom_tree, elapsed_time = self.dominator_analyzer.compute(cfg)
        
        # Visualize if requested
        if visualize:
            cfg.visualize(filename='cfg_analysis')
        
        # Return results
        return {
            'cfg': cfg,
            'dominator_sets': dom_sets,
            'dominator_tree': dom_tree,
            'execution_time_ms': elapsed_time
        }
    
    def print_analysis_summary(self, results: Dict[str, object]) -> None:
        """Print a summary of the analysis results."""
        dom_tree = results['dominator_tree']
        dom_sets = results['dominator_sets']
        cfg = results['cfg']
        
        print(f"\nControl Flow Graph: {cfg.node_count} nodes")
        
        print("\nDominator Tree:")
        for parent, children in sorted(dom_tree.items()):
            print(f"Node {parent} immediately dominates: {sorted(children)}")
        
        print(f"\nAnalysis completed in {results['execution_time_ms']:.2f} ms")

# =========================================================
# Testing and Benchmarking
# =========================================================
def run_benchmark(analyzer: CFGAnalyzer, source_code: str, runs: int = 10) -> float:
    """Run a benchmark on the given source code."""
    total_time = 0
    
    for i in range(runs):
        results = analyzer.analyze(source_code, visualize=(i == 0))
        total_time += results['execution_time_ms']
    
    return total_time / runs

def main():
    """Main entry point for demonstration."""
    # Test code samples
    nested_conditionals = """
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

    loops_example = """
def loop_example(n, threshold):
    result = 0
    for i in range(n):
        if i % 2 == 0:
            result += i
        else:
            result -= i
            
    count = 0
    while result > threshold:
        result = result // 2
        count += 1
        
    return result, count
"""

    # Create analyzer
    analyzer = CFGAnalyzer()
    
    # Analyze nested conditionals
    print("Analyzing code with nested conditionals...")
    results_nested = analyzer.analyze(nested_conditionals, visualize=True)
    analyzer.print_analysis_summary(results_nested)
    
    # Analyze loops
    print("\nAnalyzing code with loops...")
    results_loops = analyzer.analyze(loops_example, visualize=True)
    analyzer.print_analysis_summary(results_loops)
    
    # Run benchmarks
    print("\nRunning benchmarks...")
    avg_time_nested = run_benchmark(analyzer, nested_conditionals, runs=10)
    avg_time_loops = run_benchmark(analyzer, loops_example, runs=10)
    
    print(f"Average execution time (nested conditionals): {avg_time_nested:.3f} ms")
    print(f"Average execution time (loops): {avg_time_loops:.3f} ms")

if __name__ == "__main__":
    main()
