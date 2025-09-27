import llvmlite.binding as llvm
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

# Initialize LLVM targets (for NVPTX backend usage if needed)
llvm.initialize()
llvm.initialize_all_targets()
llvm.initialize_all_asmprinters()

# Constants for dominator bit masks
BITS_PER_MASK = 64
MAX_MASK_COUNT = 10

def compute_dominators_from_ir(llvm_ir: str):
    # Parse LLVM IR
    mod = llvm.parse_assembly(llvm_ir)
    mod.verify()
    # Assume a single function of interest in the module
    func = next(f for f in mod.functions if not f.is_declaration)
    # Build CFG edges by scanning terminators
    cfg_succ = {}  # dict: block_name -> list of successor block_names
    for block in func.blocks:
        name = block.name
        succ_list = []
        # Get terminator instruction (last instruction in block)
        term = list(block.instructions)[-1]
        if term.opcode == "br":
            ops = list(term.operands)
            if len(ops) == 3:
                # Conditional branch: operands[0]=cond, [1]=falseDest, [2]=trueDest] (or vice versa)
                for op in ops[1:]:
                    succ_list.append(op.name)
            elif len(ops) == 1:
                # Unconditional branch
                succ_list.append(ops[0].name)
        elif term.opcode == "switch":
            ops = list(term.operands)
            # operands[0] = value, operands[1] = default dest, remaining are value/dest pairs
            if len(ops) >= 2:
                succ_list.append(ops[1].name)  # default target
                # add all case targets (even indices starting from 2 are case values, odd are targets)
                for i in range(2, len(ops), 2):
                    if i+1 < len(ops):
                        succ_list.append(ops[i+1].name)
        # (Other terminators like 'ret' produce no successors)
        cfg_succ[name] = succ_list

    # Build predecessor lists from successor dict
    block_names = list(cfg_succ.keys())
    index_of = {name: i for i, name in enumerate(block_names)}
    num_nodes = len(block_names)
    preds = [[] for _ in range(num_nodes)]
    for src, succ_list in cfg_succ.items():
        for tgt in succ_list:
            preds[index_of[tgt]].append(index_of[src])
    # Convert preds lists to flattened array + offsets
    pred_index_list = []
    pred_offsets = []
    for i in range(num_nodes):
        pred_offsets.append(len(pred_index_list))
        # sort preds for deterministic order (optional)
        for p in preds[i]:
            pred_index_list.append(p)
    num_preds = len(pred_index_list)
    pred_offsets = np.array(pred_offsets, dtype=np.int32)
    predecessors = np.array(pred_index_list, dtype=np.int32)

    # Determine mask count for bit-vector (each mask is 64 bits)
    mask_count = (num_nodes + BITS_PER_MASK - 1) // BITS_PER_MASK
    if mask_count > MAX_MASK_COUNT:
        raise ValueError(f"mask_count ({mask_count}) exceeds MAX_MASK_COUNT ({MAX_MASK_COUNT})")
    # Initialize dominator matrix (num_nodes x mask_count)
    dom = np.zeros((num_nodes, mask_count), dtype=np.uint64)
    # Initial dominator for entry (assuming index 0 is entry block)
    dom[0, 0] = 1 << 0  # entry node dominates itself (bit0 = 1)

    # Allocate device memory (use page-locked host memory for faster transfer)
    pinned_pred = cuda.pagelocked_empty_like(predecessors); pinned_pred[:] = predecessors
    pinned_offs = cuda.pagelocked_empty_like(pred_offsets); pinned_offs[:] = pred_offsets
    pinned_dom  = cuda.pagelocked_empty_like(dom);          pinned_dom[:]  = dom
    d_pred = cuda.to_device(pinned_pred)
    d_offs = cuda.to_device(pinned_offs)
    d_dom  = cuda.to_device(pinned_dom)
    d_changed = cuda.mem_alloc(np.int32(0).nbytes)  # device flag

    # JIT compile CUDA kernel (NVPTX code generation via NVCC/NVRTC through PyCUDA)
    module = SourceModule(f"""
    #define BITS_PER_MASK {BITS_PER_MASK}
    #define MAX_MASK_COUNT {MAX_MASK_COUNT}
    __global__ void compute_dominator_optimized(unsigned long long *dom,
                                               const int *preds, const int *pred_offs,
                                               int num_nodes, int num_preds, int mask_count,
                                               int *d_changed) {{
        int node = blockIdx.x * blockDim.x + threadIdx.x;
        if (node >= num_nodes) return;
        if (mask_count > MAX_MASK_COUNT) return;
        if (node == 0) {{
            // Entry node: only dominates itself
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
        int end   = (node + 1 < num_nodes) ? pred_offs[node + 1] : num_preds;
        for(int i = start; i < end; ++i) {{
            int pred = preds[i];
            for(int j=0; j<mask_count; ++j) {{
                unsigned long long dom_val = dom[pred * mask_count + j];
                my_intersection[j] &= dom_val;
            }}
        }}
        // A node always dominates itself
        int mask_index = node / BITS_PER_MASK;
        int bit_index  = node % BITS_PER_MASK;
        if(mask_index < mask_count) {{
            my_intersection[mask_index] |= (1ULL << bit_index);
        }}
        // Check if dominator set changed
        bool changed = false;
        for(int j=0; j<mask_count; ++j) {{
            unsigned long long newval = my_intersection[j];
            if(dom[node * mask_count + j] != newval) {{
                changed = true;
            }}
        }}
        // If changed, update dominator set and mark the flag
        if(changed) {{
            for(int j=0; j<mask_count; ++j) {{
                dom[node * mask_count + j] = my_intersection[j];
            }}
            atomicExch(d_changed, 1);
        }}
    }}
    """, no_extern_c=True)  # no_extern_c to avoid name mangling
    kernel = module.get_function("compute_dominator_optimized")

    # Configure kernel launch dimensions
    block_size = 256
    grid_size = (num_nodes + block_size - 1) // block_size

    # Iteratively launch kernel until convergence
    max_iterations = 1000
    for iteration in range(max_iterations):
        # reset change flag to 0
        host_changed = np.zeros(1, dtype=np.int32)
        cuda.memcpy_htod(d_changed, host_changed)
        # launch kernel
        kernel(d_dom, d_pred, d_offs,
               np.int32(num_nodes), np.int32(num_preds), np.int32(mask_count),
               d_changed,
               block=(block_size,1,1), grid=(grid_size,1,1))
        # copy back the flag
        cuda.memcpy_dtoh(host_changed, d_changed)
        if host_changed[0] == 0:
            # no changes in this iteration -> converged
            break

    # Retrieve final dominator matrix
    cuda.memcpy_dtoh(pinned_dom, d_dom)
    result_dom = np.array(pinned_dom, copy=True)  # copy out of pinned memory
    return result_dom, block_names

# Example usage with a simple LLVM IR string (if run as a script/test)
if __name__ == "__main__":
    llvm_ir_code = r\"\"\" 
    ; Example LLVM IR (if-else structure)
    target triple = "nvptx64-nvidia-cuda"
    define i32 @cond(i32 %x) {
    entry:
      %cmp = icmp sgt i32 %x, 0
      br i1 %cmp, label %then, label %else
    then:
      %t = mul nsw i32 %x, 2
      br label %end
    else:
      %e = sub nsw i32 0, %x
      br label %end
    end:
      %y = phi i32 [ %t, %then ], [ %e, %else ]
      ret i32 %y
    } 
    \"\"\".strip()
    dom_matrix, blocks = compute_dominators_from_ir(llvm_ir_code)
    # Print dominator sets in binary form for each block
    for i, name in enumerate(blocks):
        bits = []
        for j in range(dom_matrix.shape[1]):
            # print each 64-bit mask as binary string
            bits.append(f\"{dom_matrix[i,j]:064b}\")
        dom_bits_str = ''.join(bits)[-len(blocks):]  # consider only lower num_nodes bits
        print(f\"Block {name}: Dominators bitset = {dom_bits_str}\")
