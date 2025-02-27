import unittest
import llvmlite.binding as llvm
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import time
from typing import Dict, List, Tuple, Any, Optional, Union
from functools import lru_cache
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize LLVM
llvm.initialize()
llvm.initialize_all_targets()
llvm.initialize_all_asmprinters()

# Constants
BITS_PER_MASK = 64
MAX_MASK_COUNT = 10

class LLVMToCudaTranspiler:
    """LLVM IR to CUDA transpiler with end-to-end execution capabilities."""

    # Type and operation mappings
    TYPE_MAP = {
        "i1": "bool", "i8": "char", "i16": "short", "i32": "int", "i64": "long long",
        "float": "float", "double": "double", "void": "void"
    }
    
    OP_MAPS = {
        # Binary operations
        "add": "+", "fadd": "+", "sub": "-", "fsub": "-", 
        "mul": "*", "fmul": "*", "sdiv": "/", "udiv": "/", "fdiv": "/",
        # Comparison predicates
        "eq": "==", "ne": "!=", "sgt": ">", "sge": ">=", "slt": "<", "sle": "<=", 
        "ugt": ">", "uge": ">=", "ult": "<", "ule": "<="
    }

    def __init__(self, target_triple="nvptx64-nvidia-cuda", data_layout="e-i64:64-i128:128-v16:16-v32:32-n16:32:64",
                 gpu_arch="sm_70", opt_level=2):
        """Initialize the transpiler with target configuration."""
        self.target_triple = target_triple
        self.data_layout = data_layout
        self.gpu_arch = gpu_arch
        self.opt_level = opt_level
        self.kernel_cache = {}
        logger.info(f"Initialized transpiler with target: {target_triple}, GPU arch: {gpu_arch}")

    @lru_cache(maxsize=32)
    def parse_module(self, llvm_ir: str):
        """Parse LLVM IR into a module (with caching)."""
        mod = llvm.parse_assembly(llvm_ir)
        mod.verify()
        
        # Set target triple and data layout if not already set
        if not mod.triple: mod.triple = self.target_triple
        if not mod.data_layout: mod.data_layout = self.data_layout
        return mod

    def compute_dominators(self, llvm_ir: str):
        """Compute dominators from LLVM IR using CUDA parallelization."""
        # Parse LLVM IR and extract function
        mod = self.parse_module(llvm_ir)
        func = next((f for f in mod.functions if not f.is_declaration), None)
        if not func: raise ValueError("No non-declaration function found in IR")

        # Build CFG and extract blocks
        cfg_succ, block_names = self._extract_cfg(func)
        num_nodes = len(block_names)

        # Process predecessors for dominator computation
        pred_index_list, pred_offsets = self._build_predecessors(cfg_succ, block_names)

        # Set up bit vector representation
        mask_count = (num_nodes + BITS_PER_MASK - 1) // BITS_PER_MASK
        if mask_count > MAX_MASK_COUNT:
            raise ValueError(f"mask_count ({mask_count}) exceeds MAX_MASK_COUNT ({MAX_MASK_COUNT})")

        # Prepare data for CUDA processing
        dom_matrix, _ = self._run_dominator_kernel(num_nodes, pred_index_list, pred_offsets, mask_count)
        
        # Convert bit vector representation to dominator sets
        dom_sets = {}
        for i, node_name in enumerate(block_names):
            dom_set = set()
            for j in range(num_nodes):
                mask_idx = j // BITS_PER_MASK
                bit_idx = j % BITS_PER_MASK
                if (dom_matrix[i, mask_idx] & (1 << bit_idx)) != 0:
                    dom_set.add(block_names[j])
            dom_sets[node_name] = dom_set
            
        return dom_sets, block_names

    def _extract_cfg(self, func):
        """Extract control flow graph from function."""
        cfg_succ = {}
        for block in func.blocks:
            name = block.name
            succ_list = []
            instructions = list(block.instructions)
            if not instructions: continue

            term = instructions[-1]  # terminator instruction
            if term.opcode == "br":
                ops = list(term.operands)
                if len(ops) == 3:  # conditional branch
                    succ_list.extend(op.name for op in ops[1:])
                elif len(ops) == 1:  # unconditional branch
                    succ_list.append(ops[0].name)
            elif term.opcode == "switch":
                ops = list(term.operands)
                if len(ops) >= 2:
                    succ_list.append(ops[1].name)  # default target
                    # add case targets
                    for i in range(2, len(ops), 2):
                        if i+1 < len(ops): succ_list.append(ops[i+1].name)

            cfg_succ[name] = succ_list

        block_names = list(cfg_succ.keys())
        return cfg_succ, block_names

    def _build_predecessors(self, cfg_succ, block_names):
        """Build predecessor lists for dominator computation."""
        index_of = {name: i for i, name in enumerate(block_names)}
        num_nodes = len(block_names)
        preds = [[] for _ in range(num_nodes)]

        for src, succ_list in cfg_succ.items():
            for tgt in succ_list:
                preds[index_of[tgt]].append(index_of[src])

        # Convert preds lists to flattened array + offsets
        pred_index_list, pred_offsets = [], []
        for i in range(num_nodes):
            pred_offsets.append(len(pred_index_list))
            pred_index_list.extend(preds[i])

        return np.array(pred_index_list, dtype=np.int32), np.array(pred_offsets, dtype=np.int32)

    def _run_dominator_kernel(self, num_nodes, predecessors, pred_offsets, mask_count):
        """Run CUDA kernel for dominator computation."""
        # Initialize dominator matrix
        dom = np.zeros((num_nodes, mask_count), dtype=np.uint64)
        dom[0, 0] = 1 << 0  # entry node dominates itself

        # Prepare device memory
        pinned_offs = cuda.pagelocked_empty_like(pred_offsets)
        pinned_offs[:] = pred_offsets
        d_offs = cuda.to_device(pinned_offs)

        pinned_dom = cuda.pagelocked_empty_like(dom)
        pinned_dom[:] = dom
        d_dom = cuda.to_device(pinned_dom)

        d_changed = cuda.mem_alloc(np.int32(0).nbytes)

        # Handle the case when there are no predecessors
        num_preds = len(predecessors)
        if num_preds > 0:
            pinned_pred = cuda.pagelocked_empty_like(predecessors)
            pinned_pred[:] = predecessors
            d_pred = cuda.to_device(pinned_pred)
        else:
            d_pred = cuda.to_device(np.array([0], dtype=np.int32))
            num_preds = 1

        # Get or create kernel
        if "dominator_kernel" not in self.kernel_cache:
            module = self._create_dominator_kernel_module()
            self.kernel_cache["dominator_kernel"] = module.get_function("compute_dominator_optimized")

        kernel = self.kernel_cache["dominator_kernel"]

        # Launch configuration
        block_size = 256
        grid_size = (num_nodes + block_size - 1) // block_size

        # Iterate until convergence
        for iteration in range(1000):
            host_changed = np.zeros(1, dtype=np.int32)
            cuda.memcpy_htod(d_changed, host_changed)

            kernel(d_dom, d_pred, d_offs,
                  np.int32(num_nodes), np.int32(num_preds), np.int32(mask_count),
                  d_changed,
                  block=(block_size,1,1), grid=(grid_size,1,1))

            cuda.memcpy_dtoh(host_changed, d_changed)
            if host_changed[0] == 0:
                logger.info(f"Dominator computation converged after {iteration+1} iterations")
                break

        # Get results
        cuda.memcpy_dtoh(pinned_dom, d_dom)
        return np.array(pinned_dom, copy=True), block_names

    def _create_dominator_kernel_module(self):
        """Create CUDA kernel module for dominator computation."""
        cuda_code = f"""
        #define BITS_PER_MASK {BITS_PER_MASK}
        #define MAX_MASK_COUNT {MAX_MASK_COUNT}

        __global__ void compute_dominator_optimized(unsigned long long *dom,
                                                const int *preds, const int *pred_offs,
                                                int num_nodes, int num_preds, int mask_count,
                                                int *d_changed) {{
            int node = blockIdx.x * blockDim.x + threadIdx.x;
            if (node >= num_nodes) return;
            if (mask_count > MAX_MASK_COUNT) return;

            // Entry node special case
            if (node == 0) {{
                for(int j=0; j<mask_count; ++j) dom[j] = 0ULL;
                dom[0] = 1ULL << 0;
                return;
            }}

            // Shared memory for warp-level reduction
            __shared__ unsigned long long shared_intersection[MAX_MASK_COUNT * 32];
            int warp_id = threadIdx.x / 32;
            int lane_id = threadIdx.x % 32;
            unsigned long long *my_intersection = &shared_intersection[warp_id * mask_count];

            // Initialize intersection to all 1s
            for(int j=0; j<mask_count; ++j) {{
                if(lane_id == 0) my_intersection[j] = 0xFFFFFFFFFFFFFFFFULL;
            }}
            __syncwarp();

            // Intersect all predecessors' dominator sets
            int start = pred_offs[node];
            int end = (node + 1 < num_nodes) ? pred_offs[node + 1] : num_preds;

            for(int i = start; i < end; ++i) {{
                int pred = preds[i];
                for(int j=0; j<mask_count; ++j) {{
                    my_intersection[j] &= dom[pred * mask_count + j];
                }}
            }}

            // A node always dominates itself
            int mask_index = node / BITS_PER_MASK;
            int bit_index = node % BITS_PER_MASK;
            if(mask_index < mask_count) {{
                my_intersection[mask_index] |= (1ULL << bit_index);
            }}

            // Update dominator set if changed
            bool changed = false;
            for(int j=0; j<mask_count; ++j) {{
                unsigned long long newval = my_intersection[j];
                if(dom[node * mask_count + j] != newval) {{
                    dom[node * mask_count + j] = newval;
                    changed = true;
                }}
            }}

            if(changed) atomicExch(d_changed, 1);
        }}
        """
        return SourceModule(cuda_code, no_extern_c=True)

    def llvm_to_ptx(self, llvm_ir: str) -> str:
        """Convert LLVM IR to PTX assembly for CUDA execution."""
        mod = self.parse_module(llvm_ir)
        target = llvm.Target.from_triple(self.target_triple)
        target_machine = target.create_target_machine(
            triple=self.target_triple, cpu=self.gpu_arch, features="",
            opt=self.opt_level, reloc="pic", code_model="default"
        )
        return target_machine.emit_assembly(mod)

    def extract_kernels(self, llvm_ir: str) -> Dict[str, str]:
        """Extract kernel functions from LLVM IR."""
        mod = self.parse_module(llvm_ir)
        kernels = {}
        for func in mod.functions:
            if func.is_declaration: continue
            is_kernel = any("kernel" in str(attr).lower() or "global" in str(attr).lower()
                          for attr in func.attributes)
            if is_kernel: kernels[func.name] = str(func)
        return kernels

    def transpile_to_cuda(self, llvm_ir: str) -> str:
        """Transpile LLVM IR to CUDA C code."""
        mod = self.parse_module(llvm_ir)
        cuda_code = ["// Auto-generated CUDA code from LLVM IR",
                    "#include <cuda_runtime.h>", ""]

        # Process functions
        for func in mod.functions:
            if func.is_declaration: continue

            is_kernel = any("kernel" in str(attr).lower() or "global" in str(attr).lower()
                           for attr in func.attributes)

            return_type = self._convert_llvm_type(func.return_type)
            args = [f"{self._convert_llvm_type(arg.type)} {arg.name}" for arg in func.arguments]

            prefix = "__global__" if is_kernel else ""
            cuda_code.append(f"{prefix} {return_type} {func.name}({', '.join(args)}) {{")
            self._process_function_body(func, cuda_code)
            cuda_code.append("}")
            cuda_code.append("")

        return "\n".join(cuda_code)

    def _process_function_body(self, func, cuda_code):
        """Process function body and convert to CUDA."""
        block_labels = {block.name: f"block_{i}" for i, block in enumerate(func.blocks)}
        for i, block in enumerate(func.blocks):
            if i > 0: cuda_code.append(f"  {block_labels[block.name]}:")
            for instr in block.instructions:
                cuda_instr = self._convert_instruction(instr, block_labels)
                if cuda_instr: cuda_code.append(f"  {cuda_instr}")

    @lru_cache(maxsize=64)
    def _convert_llvm_type(self, llvm_type) -> str:
        """Convert LLVM types to CUDA C types with caching."""
        type_str = str(llvm_type)

        # Check basic type mapping
        for llvm_t, cuda_t in self.TYPE_MAP.items():
            if llvm_t in type_str: return cuda_t

        # Handle pointers
        if "*" in type_str:
            base_type = self._convert_llvm_type(llvm_type.pointee)
            return f"{base_type}*"

        # Vector types
        if "x" in type_str and "[" in type_str:
            try:
                vec_match = type_str.split("x")[0]
                count = int(vec_match)
                elem_type = self._convert_llvm_type(llvm_type.element)
                return f"{elem_type}{count}"
            except (ValueError, AttributeError): pass

        return "void*"  # Default fallback

    def _convert_instruction(self, instr, block_labels=None) -> str:
        """Convert a single LLVM instruction to CUDA C code."""
        if block_labels is None: block_labels = {}
        opcode = instr.opcode

        # Return instruction
        if opcode == "ret":
            return f"return {instr.operands[0].name};" if len(instr.operands) > 0 else "return;"

        # Binary operations
        if opcode in self.OP_MAPS and len(instr.operands) >= 2:
            return f"{instr.name} = {instr.operands[0].name} {self.OP_MAPS[opcode]} {instr.operands[1].name};"

        # Comparisons
        elif opcode in ["icmp", "fcmp"] and len(instr.operands) >= 2:
            instr_str = str(instr)
            for pred, op in self.OP_MAPS.items():
                if pred in instr_str:
                    return f"{instr.name} = {instr.operands[0].name} {op} {instr.operands[1].name};"

        # Control flow
        elif opcode == "br":
            if len(instr.operands) == 1:  # Unconditional
                label = block_labels.get(instr.operands[0].name, instr.operands[0].name)
                return f"goto {label};"
            elif len(instr.operands) == 3:  # Conditional
                cond = instr.operands[0].name
                true_label = block_labels.get(instr.operands[1].name, instr.operands[1].name)
                false_label = block_labels.get(instr.operands[2].name, instr.operands[2].name)
                return f"if ({cond}) goto {true_label}; else goto {false_label};"

        # Memory operations
        elif opcode == "alloca":
            type_name = self._convert_llvm_type(instr.type.pointee)
            return f"{type_name} {instr.name};"
        elif opcode == "load" and len(instr.operands) >= 1:
            return f"{instr.name} = *{instr.operands[0].name};"
        elif opcode == "store" and len(instr.operands) >= 2:
            return f"*{instr.operands[1].name} = {instr.operands[0].name};"
        elif opcode == "getelementptr" and len(instr.operands) >= 2:
            base_ptr = instr.operands[0].name
            indices = [op.name for op in instr.operands[1:]]
            index_expr = " + ".join(indices) if indices else "0"
            return f"{instr.name} = {base_ptr} + {index_expr};"

        # Function calls
        elif opcode == "call" and len(instr.operands) >= 1:
            func_name = instr.operands[-1].name
            args = [op.name for op in instr.operands[:-1]]
            return (f"{instr.name} = {func_name}({', '.join(args)});" 
                   if str(instr.type) != "void" else f"{func_name}({', '.join(args)});")

        # Type conversions
        elif opcode in ["trunc", "zext", "sext", "fptrunc", "fpext", "bitcast", 
                      "inttoptr", "ptrtoint"] and len(instr.operands) >= 1:
            target_type = self._convert_llvm_type(instr.type)
            return f"{instr.name} = ({target_type}){instr.operands[0].name};"

        # PHI nodes
        elif opcode == "phi" and len(instr.operands) >= 2:
            val = instr.operands[0].name
            return f"{instr.name} = {val}; /* Simplified PHI node */"

        return f"/* Unsupported: {opcode} */"

    def compile_and_load_ptx(self, ptx_code: str, function_name: str):
        """Compile PTX code and load the specified function."""
        if function_name in self.kernel_cache:
            return self.kernel_cache[function_name]

        try:
            module = SourceModule(ptx_code, no_extern_c=True)
            kernel = module.get_function(function_name)
            self.kernel_cache[function_name] = kernel
            return kernel
        except Exception as e:
            logger.error(f"Failed to compile PTX: {e}")
            raise

    def execute_kernel(self, kernel_func, *args, grid=(1,1,1), block=(32,1,1)):
        """Execute a CUDA kernel with the specified arguments."""
        try:
            kernel_func(*args, grid=grid, block=block)
            cuda.Context.synchronize()
        except Exception as e:
            logger.error(f"Kernel execution failed: {e}")
            raise

    def end_to_end_execute(self, llvm_ir: str, input_data=None, output_type=np.int32):
        """Execute LLVM IR end-to-end on GPU."""
        try:
            # Convert to PTX
            ptx_code = self.llvm_to_ptx(llvm_ir)
            
            # Get kernel name
            mod = self.parse_module(llvm_ir)
            kernel_name = next((f.name for f in mod.functions if not f.is_declaration), None)
            if not kernel_name:
                raise ValueError("No kernel function found")

            # Compile and load kernel
            kernel_func = self.compile_and_load_ptx(ptx_code, kernel_name)
            
            # Prepare data
            if input_data is not None:
                if isinstance(input_data, int):
                    d_input = cuda.to_device(np.array([input_data], dtype=np.int32))
                    d_output = cuda.mem_alloc(np.dtype(output_type).itemsize)
                elif isinstance(input_data, np.ndarray):
                    d_input = cuda.to_device(input_data)
                    d_output = cuda.mem_alloc(np.dtype(output_type).itemsize * 
                                             (input_data.size if output_type != np.void else 1))
                else:
                    raise TypeError(f"Unsupported input data type: {type(input_data)}")
            else:
                d_input = None
                d_output = cuda.mem_alloc(np.dtype(output_type).itemsize)
            
            # Execute kernel
            start_time = time.time()
            if d_input is not None:
                self.execute_kernel(
                    kernel_func, d_input, d_output, 
                    block=(256, 1, 1), grid=((input_data.size if isinstance(input_data, np.ndarray) else 1 + 255) // 256, 1, 1)
                )
            else:
                self.execute_kernel(
                    kernel_func, d_output,
                    block=(256, 1, 1), grid=(1, 1, 1)
                )
            execution_time = time.time() - start_time
            
            # Retrieve result
            if output_type == np.void:
                return None, execution_time
                
            if isinstance(input_data, np.ndarray):
                result = np.zeros(input_data.shape, dtype=output_type)
            else:
                result = np.zeros(1, dtype=output_type)
                
            cuda.memcpy_dtoh(result, d_output)
            
            return result[0] if result.size == 1 else result, execution_time
            
        except Exception as e:
            logger.error(f"End-to-end execution failed: {e}")
            raise
            
    def execute_cpu_for_validation(self, llvm_ir: str, input_data=None):
        """Execute on CPU for validation purposes."""
        try:
            import ctypes
            # Parse and optimize module
            mod = self.parse_module(llvm_ir)
            
            # Create execution engine
            target = llvm.Target.from_default_triple()
            target_machine = target.create_target_machine()
            engine = llvm.create_mcjit_compiler(mod, target_machine)
            
            # Get function
            func_name = next((f.name for f in mod.functions if not f.is_declaration), None)
            if not func_name:
                raise ValueError("No function found to execute")
            
            func_ptr = engine.get_function_address(func_name)
            
            # Create callable function based on input data
            if input_data is not None:
                if isinstance(input_data, int):
                    cfunc = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int)(func_ptr)
                    result = cfunc(input_data)
                elif isinstance(input_data, np.ndarray):
                    cfunc = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)(func_ptr)
                    result = cfunc(input_data.ctypes.data)
                else:
                    raise TypeError(f"Unsupported input type: {type(input_data)}")
            else:
                cfunc = ctypes.CFUNCTYPE(ctypes.c_int)(func_ptr)
                result = cfunc()
                
            return result
        except Exception as e:
            logger.error(f"CPU execution failed: {e}")
            raise

class TestLLVMToCudaTranspiler(unittest.TestCase):

    def setUp(self):
        self.transpiler = LLVMToCudaTranspiler()

    def test_initialization(self):
        self.assertIsInstance(self.transpiler, LLVMToCudaTranspiler)
        self.assertEqual(self.transpiler.target_triple, "nvptx64-nvidia-cuda")
        self.assertEqual(self.transpiler.gpu_arch, "sm_70")

    def test_parse_module_valid_ir(self):
        llvm_ir_code = """
        define i32 @add(i32 %a, i32 %b) {
          %result = add i32 %a, %b
          ret i32 %result
        }
        """
        mod = self.transpiler.parse_module(llvm_ir_code)
        self.assertIsInstance(mod, llvm.ModuleRef)

    def test_parse_module_invalid_ir(self):
        llvm_ir_code = """
        This is not valid LLVM IR
        """
        with self.assertRaises(llvm.ParseError):
            self.transpiler.parse_module(llvm_ir_code)

    def test_llvm_to_ptx_simple_kernel(self):
        llvm_ir_code = """
        define void @kernel.add(float* nocapture %out, float* nocapture %in1, float* nocapture %in2) {
          %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
          %in1.ptr = getelementptr float, float* %in1, i32 %tid
          %in2.ptr = getelementptr float, float* %in2, i32 %tid
          %out.ptr = getelementptr float, float* %out, i32 %tid
          %val1 = load float, float* %in1.ptr
          %val2 = load float, float* %in2.ptr
          %add = fadd float %val1, %val2
          store float %add, float* %out.ptr
          ret void
        }

        declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() nounwind readnone

        attributes #0 = { nounwind }
        """
        ptx_code = self.transpiler.llvm_to_ptx(llvm_ir_code)
        self.assertIsInstance(ptx_code, str)
        self.assertTrue("%kernel.add" in ptx_code)

    def test_extract_kernels_single_kernel(self):
        llvm_ir_code = """
        define void @kernel.add(float* nocapture %out, float* nocapture %in1, float* nocapture %in2) {
          %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
          ret void
        }

        define i32 @host_function(i32 %a) {
          ret i32 %a
        }
        """
        kernels = self.transpiler.extract_kernels(llvm_ir_code)
        self.assertIn("kernel.add", kernels)
        self.assertNotIn("host_function", kernels)

    def test_transpile_to_cuda_simple_kernel(self):
        llvm_ir_code = """
        define void @kernel.add(float* nocapture %out, float* nocapture %in1, float* nocapture %in2) {
          %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
          %in1.ptr = getelementptr float, float* %in1, i32 %tid
          %val1 = load float, float* %in1.ptr
          ret void
        }

        declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() nounwind readnone
        """
        cuda_code = self.transpiler.transpile_to_cuda(llvm_ir_code)
        self.assertIsInstance(cuda_code, str)
        self.assertTrue("__global__ void kernel_add" in cuda_code)
        self.assertTrue("/* Unsupported: call */" in cuda_code) # llvm.nvvm.read.ptx.sreg.tid.x is not directly transpilable

    def test_convert_llvm_type(self):
        test_types = [
            ("i32", "int"),
            ("float*", "float*"),
            ("[4 x float]", "float4"),
            ("void", "void"),
            ("i1", "bool")
        ]
        for llvm_type_str, expected_cuda_type in test_types:
            llvm_type = llvm.Type.parse(llvm_type_str)
            cuda_type = self.transpiler._convert_llvm_type(llvm_type)
            self.assertEqual(cuda_type, expected_cuda_type)

    def test_convert_instruction_binary_op(self):
        llvm_ir_code = """
        define i32 @test_func(i32 %a, i32 %b) {
          %add_result = add i32 %a, %b
          ret i32 %add_result
        }
        """
        mod = self.transpiler.parse_module(llvm_ir_code)
        func = mod.functions[0]
        add_instr = func.blocks[0].instructions[0] # Get the add instruction
        cuda_instr = self.transpiler._convert_instruction(add_instr)
        self.assertEqual(cuda_instr.strip(), "add_result = a + b;")

    def test_convert_instruction_ret(self):
        llvm_ir_code = """
        define i32 @test_func(i32 %a) {
          ret i32 %a
        }
        """
        mod = self.transpiler.parse_module(llvm_ir_code)
        func = mod.functions[0]
        ret_instr = func.blocks[0].instructions[0]
        cuda_instr = self.transpiler._convert_instruction(ret_instr)
        self.assertEqual(cuda_instr.strip(), "return a;")

    def test_end_to_end_execute_simple_add(self):
        llvm_ir_code = """
        define i32 @kernel_add(i32 %a) {
          %result = add i32 %a, 10
          ret i32 %result
        }
        """
        input_data = 5
        result, _ = self.transpiler.end_to_end_execute(llvm_ir_code, input_data=input_data)
        self.assertEqual(result, 15)

    def test_end_to_end_execute_array_add(self):
        llvm_ir_code = """
        define void @kernel_array_add(float* %out, float* %in) {
          %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
          %in.ptr = getelementptr float, float* %in, i32 %tid
          %out.ptr = getelementptr float, float* %out, i32 %tid
          %val = load float, float* %in.ptr
          %add = fadd float %val, 1.0
          store float %add, float* %out.ptr
          ret void
        }
        declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() nounwind readnone
        """
        input_array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result, _ = self.transpiler.end_to_end_execute(llvm_ir_code, input_data=input_array, output_type=np.float32)
        expected_output = np.array([2.0, 3.0, 4.0], dtype=np.float32)
        np.testing.assert_array_equal(result, expected_output)

    def test_execute_cpu_for_validation_simple_add(self):
        llvm_ir_code = """
        define i32 @kernel_add(i32 %a) {
          %result = add i32 %a, 10
          ret i32 %result
        }
        """
        input_data = 5
        result_cpu = self.transpiler.execute_cpu_for_validation(llvm_ir_code, input_data=input_data)
        self.assertEqual(result_cpu, 15)

    def test_compute_dominators_simple_cfg(self):
        llvm_ir_code = """
        define void @func() {
        entry:
          br label %block1

        block1:                                           ; preds = %entry
          br label %block2

        block2:                                           ; preds = %block1
          br label %block3

        block3:                                           ; preds = %block2
          ret void
        }
        """
        dom_sets, block_names = self.transpiler.compute_dominators(llvm_ir_code)
        expected_dom_sets = {
            "entry": {"entry"},
            "block1": {"entry", "block1"},
            "block2": {"entry", "block1", "block2"},
            "block3": {"entry", "block1", "block2", "block3"}
        }
        self.assertEqual(dom_sets, expected_dom_sets)

    def test_compute_dominators_if_else_cfg(self):
        llvm_ir_code = """
        define void @func(i1 %cond) {
        entry:
          br i1 %cond, label %then_block, label %else_block

        then_block:                                       ; preds = %entry
          br label %merge_block

        else_block:                                       ; preds = %entry
          br label %merge_block

        merge_block:                                      ; preds = %then_block, %else_block
          ret void
        }
        """
        dom_sets, block_names = self.transpiler.compute_dominators(llvm_ir_code)
        expected_dom_sets = {
            "entry": {"entry"},
            "then_block": {"entry", "then_block"},
            "else_block": {"entry", "else_block"},
            "merge_block": {"entry", "merge_block"} # entry dominates merge in this simplified version
        }
        self.assertEqual(dom_sets, expected_dom_sets)

    def test_execute_kernel_compilation_failure(self):
        invalid_ptx_code = "This is not valid PTX code"
        function_name = "invalid_kernel"
        with self.assertRaises(Exception) as context:
            self.transpiler.compile_and_load_ptx(invalid_ptx_code, function_name)
        self.assertIn("Failed to compile PTX", str(context.exception))

    def test_end_to_end_execution_kernel_not_found(self):
        llvm_ir_code_no_kernel = """
        define i32 @host_function(i32 %a) {
          ret i32 %a
        }
        """
        with self.assertRaises(ValueError) as context:
            self.transpiler.end_to_end_execute(llvm_ir_code_no_kernel)
        self.assertIn("No kernel function found", str(context.exception))

if __name__ == '__main__':
    unittest.main()
