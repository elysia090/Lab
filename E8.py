import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
from enum import Enum
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass
import logging
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpCode(Enum):
    """Defines the instruction set."""
    NAND = 0
    LOAD = 1
    STORE = 2
    JUMP = 3 # Not implemented
    TROPICAL_ADD = 4
    TROPICAL_MUL = 5
    SHIFT = 6
    GRAY_ENCODE = 7
    GRAY_DECODE = 8

@dataclass
class Instruction:
    """Defines the instruction format."""
    opcode: OpCode
    dest_reg: int # Destination register
    src1_reg: int # Source register 1
    src2_reg: int # Source register 2
    immediate: int = 0 # Immediate value (if applicable)

class MemoryManager:
    """Manages memory allocation and deallocation."""
    def __init__(self, total_size: int):
        """Initializes the memory manager."""
        self.total_size = total_size
        self.memory = np.zeros(total_size, dtype=np.int32) # Host memory
        self.gpu_memory = cuda.mem_alloc(self.memory.nbytes) # GPU memory allocation
        self.free_blocks: List[Tuple[int, int]] = [(0, total_size)] # List of free memory blocks (start, length)

    def allocate(self, size: int) -> int:
        """Allocates a block of memory of the given size."""
        for i, (start, length) in enumerate(self.free_blocks):
            if length >= size:
                if length > size:
                    self.free_blocks[i] = (start + size, length - size) # Split block
                else:
                    self.free_blocks.pop(i) # Remove block if exact fit
                return start
        raise MemoryError("No sufficient memory block available")

    def free(self, start: int, size: int):
        """Frees a block of memory at the given start address and size."""
        self.free_blocks.append((start, size))
        self._merge_free_blocks() # Merge adjacent free blocks

    def _merge_free_blocks(self):
        """Merges adjacent free memory blocks to reduce fragmentation."""
        self.free_blocks.sort() # Sort blocks by start address
        i = 0
        while i < len(self.free_blocks) - 1:
            current_start, current_size = self.free_blocks[i]
            next_start, next_size = self.free_blocks[i + 1]
            if current_start + current_size == next_start: # Check if blocks are adjacent
                self.free_blocks[i] = (current_start, current_size + next_size) # Merge blocks
                self.free_blocks.pop(i + 1) # Remove merged block
            else:
                i += 1

class ParallelVirtualCPU:
    """Simulates a parallel virtual CPU with CUDA acceleration."""
    def __init__(self, dimension: int, memory_size: int = 1024*1024):
        """Initializes the ParallelVirtualCPU."""
        self.dimension = dimension
        self.size = 2 ** dimension # Number of parallel nodes
        self.memory_manager = MemoryManager(memory_size)
        self.register_file = np.zeros((32, self.size), dtype=np.int32) # Register file (host memory), now vector registers for parallel ops
        self.program_counter = 0 # Program counter

        # Initialize CUDA kernels
        self._initialize_cuda_kernels()

        # Initialize Gray code table
        self.gray_code_table = self._initialize_gray_code()

    def _initialize_cuda_kernels(self):
        """Compiles and initializes CUDA kernels."""
        # NAND kernel code
        nand_kernel_code = """
        __global__ void nand_kernel(int *a, int *b, int *out, int N)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < N) {
                out[idx] = !(a[idx] && b[idx]);
            }
        }
        """

        # Tropical operations kernel code
        tropical_kernel_code = """
        __global__ void tropical_kernel(float *a, float *b, float *out,
                                         int op_type, int N)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < N) {
                if (op_type == 0) {  // Tropical addition (min)
                    out[idx] = min(a[idx], b[idx]);
                } else {  // Tropical multiplication (regular addition)
                    out[idx] = a[idx] + b[idx];
                }
            }
        }
        """

        # Gray code conversion kernels code
        gray_code_kernel_code = """
        __global__ void gray_encode_kernel(int *input, int *output, int N)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < N) {
                output[idx] = input[idx] ^ (input[idx] >> 1);
            }
        }

        __global__ void gray_decode_kernel(int *input, int *output, int N)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < N) {
                int temp = input[idx];
                for (int i = 1; i <= 31; i <<= 1) { // Corrected loop condition and increment for 32-bit int
                    temp ^= (temp >> i);
                }
                output[idx] = temp;
            }
        }
        """
        # Shift kernel code
        shift_kernel_code = """
        __global__ void shift_kernel(int *input, int *output, int shift_amount, int N)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < N) {
                output[idx] = input[idx] << shift_amount;
            }
        }
        """


        # Compile kernels and get functions
        module = SourceModule(nand_kernel_code + tropical_kernel_code + gray_code_kernel_code + shift_kernel_code)
        self.nand_kernel = module.get_function("nand_kernel")
        self.tropical_kernel = module.get_function("tropical_kernel")
        self.gray_encode_kernel = module.get_function("gray_encode_kernel")
        self.gray_decode_kernel = module.get_function("gray_decode_kernel")
        self.shift_kernel = module.get_function("shift_kernel")


    def _initialize_gray_code(self) -> np.ndarray:
        """Initializes the Gray code lookup table."""
        gray_codes = np.zeros(self.size, dtype=np.int32)
        for i in range(self.size):
            gray_codes[i] = i ^ (i >> 1) # Calculate Gray code
        return gray_codes

    def _execute_instruction(self, instruction: Instruction):
        """Executes a single instruction, now with CUDA acceleration for parallel ops."""
        opcode = instruction.opcode
        N = self.size # Number of parallel nodes

        if opcode == OpCode.NAND:
            # NAND gate operation (CUDA accelerated)
            src1_values = self.register_file[instruction.src1_reg]
            src2_values = self.register_file[instruction.src2_reg]
            dest_values = np.zeros_like(src1_values) # Allocate space for results

            # Allocate GPU memory
            src1_gpu = cuda.mem_alloc(src1_values.nbytes)
            src2_gpu = cuda.mem_alloc(src2_values.nbytes)
            dest_gpu = cuda.mem_alloc(dest_values.nbytes)

            # Copy data to GPU
            cuda.memcpy_htod(src1_gpu, src1_values)
            cuda.memcpy_htod(src2_gpu, src2_values)

            # Kernel execution parameters
            block_dim = 256 # Adjust block size as needed
            grid_dim = ( (N + block_dim - 1) // block_dim, 1, 1)

            # Launch kernel
            self.nand_kernel(src1_gpu, src2_gpu, dest_gpu, np.int32(N), block=(block_dim,1,1), grid=grid_dim)

            # Copy result back from GPU
            cuda.memcpy_dtoh(dest_values, dest_gpu)
            self.register_file[instruction.dest_reg] = dest_values

            # Free GPU memory
            src1_gpu.free()
            src2_gpu.free()
            dest_gpu.free()


        elif opcode == OpCode.TROPICAL_ADD:
            # Tropical addition (min) (CUDA accelerated)
            src1_values = self.register_file[instruction.src1_reg].astype(np.float32) # Convert to float for tropical ops
            src2_values = self.register_file[instruction.src2_reg].astype(np.float32)
            dest_values = np.zeros_like(src1_values)

            src1_gpu = cuda.mem_alloc(src1_values.nbytes)
            src2_gpu = cuda.mem_alloc(src2_values.nbytes)
            dest_gpu = cuda.mem_alloc(dest_values.nbytes)

            cuda.memcpy_htod(src1_gpu, src1_values)
            cuda.memcpy_htod(src2_gpu, src2_values)

            block_dim = 256
            grid_dim = ( (N + block_dim - 1) // block_dim, 1, 1)

            self.tropical_kernel(src1_gpu, src2_gpu, dest_gpu, np.int32(0), np.int32(N), block=(block_dim,1,1), grid=grid_dim) # op_type=0 for ADD

            cuda.memcpy_dtoh(dest_values, dest_gpu)
            self.register_file[instruction.dest_reg] = dest_values.astype(np.int32) # Store back as integer


            src1_gpu.free()
            src2_gpu.free()
            dest_gpu.free()


        elif opcode == OpCode.TROPICAL_MUL:
            # Tropical multiplication (regular addition) (CUDA accelerated)
            src1_values = self.register_file[instruction.src1_reg].astype(np.float32) # Convert to float for tropical ops
            src2_values = self.register_file[instruction.src2_reg].astype(np.float32)
            dest_values = np.zeros_like(src1_values)

            src1_gpu = cuda.mem_alloc(src1_values.nbytes)
            src2_gpu = cuda.mem_alloc(src2_values.nbytes)
            dest_gpu = cuda.mem_alloc(dest_values.nbytes)

            cuda.memcpy_htod(src1_gpu, src1_values)
            cuda.memcpy_htod(src2_gpu, src2_values)

            block_dim = 256
            grid_dim = ( (N + block_dim - 1) // block_dim, 1, 1)

            self.tropical_kernel(src1_gpu, src2_gpu, dest_gpu, np.int32(1), np.int32(N), block=(block_dim,1,1), grid=grid_dim) # op_type=1 for MUL

            cuda.memcpy_dtoh(dest_values, dest_gpu)
            self.register_file[instruction.dest_reg] = dest_values.astype(np.int32) # Store back as integer

            src1_gpu.free()
            src2_gpu.free()
            dest_gpu.free()


        elif opcode == OpCode.LOAD:
            # Load immediate value into register (CPU - no parallel load in this example, immediate is same for all nodes)
            self.register_file[instruction.dest_reg][:] = instruction.src1_reg # Broadcast immediate to all nodes' registers

        elif opcode == OpCode.STORE:
            # Store register value to memory (dummy store in this example - registers are directly accessed)
            pass # For now, STORE does nothing as registers are directly used. Memory operations to be implemented later.

        elif opcode == OpCode.SHIFT: # SHIFT Rdest, Rsrc1, immediate (CUDA accelerated)
            src_values = self.register_file[instruction.src1_reg]
            shift_amount = instruction.immediate
            dest_values = np.zeros_like(src_values)

            src_gpu = cuda.mem_alloc(src_values.nbytes)
            dest_gpu = cuda.mem_alloc(dest_values.nbytes)

            cuda.memcpy_htod(src_gpu, src_values)

            block_dim = 256
            grid_dim = ( (N + block_dim - 1) // block_dim, 1, 1)

            self.shift_kernel(src_gpu, dest_gpu, np.int32(shift_amount), np.int32(N), block=(block_dim,1,1), grid=grid_dim)

            cuda.memcpy_dtoh(dest_values, dest_gpu)
            self.register_file[instruction.dest_reg] = dest_values

            src_gpu.free()
            dest_gpu.free()


        elif opcode == OpCode.GRAY_ENCODE: # GRAY_ENCODE Rdest, Rsrc1 (CUDA accelerated)
            input_values = self.register_file[instruction.src1_reg]
            dest_values = np.zeros_like(input_values)

            input_gpu = cuda.mem_alloc(input_values.nbytes)
            dest_gpu = cuda.mem_alloc(dest_values.nbytes)

            cuda.memcpy_htod(input_gpu, input_values)

            block_dim = 256
            grid_dim = ( (N + block_dim - 1) // block_dim, 1, 1)

            self.gray_encode_kernel(input_gpu, dest_gpu, np.int32(N), block=(block_dim,1,1), grid=grid_dim)

            cuda.memcpy_dtoh(dest_values, dest_gpu)
            self.register_file[instruction.dest_reg] = dest_values


            input_gpu.free()
            dest_gpu.free()


        elif opcode == OpCode.GRAY_DECODE: # GRAY_DECODE Rdest, Rsrc1 (CUDA accelerated)
            input_values = self.register_file[instruction.src1_reg]
            dest_values = np.zeros_like(input_values)

            input_gpu = cuda.mem_alloc(input_values.nbytes)
            dest_gpu = cuda.mem_alloc(dest_values.nbytes)

            cuda.memcpy_htod(input_gpu, input_values)

            block_dim = 256
            grid_dim = ( (N + block_dim - 1) // block_dim, 1, 1)

            self.gray_decode_kernel(input_gpu, dest_gpu, np.int32(N), block=(block_dim,1,1), grid=grid_dim)

            cuda.memcpy_dtoh(dest_values, dest_gpu)
            self.register_file[instruction.dest_reg] = dest_values

            input_gpu.free()
            dest_gpu.free()


    def execute_program(self, instructions: List[Instruction]):
        """Executes a list of instructions (the program)."""
        self.program_counter = 0
        while self.program_counter < len(instructions):
            instruction = instructions[self.program_counter]
            try:
                self._execute_instruction(instruction) # Execute current instruction
                self.program_counter += 1 # Increment program counter
            except Exception as e:
                logger.error(f"Error executing instruction at PC={self.program_counter}: {e}")
                raise

class AssemblyCompiler:
    """Compiles assembly code into a list of Instructions."""
    def __init__(self):
        """Initializes the AssemblyCompiler."""
        self.symbol_table: Dict[str, int] = {} # Symbol table for labels (not used in this example, but can be extended)
        self.current_address = 0 # Current memory address (not used in this example)

    def compile(self, assembly_code: str) -> List[Instruction]:
        """Compiles assembly code string into a list of Instruction objects."""
        instructions = []
        lines = assembly_code.strip().split('\n') # Split code into lines

        for line in lines:
            line = line.strip() # Remove leading/trailing whitespace
            if not line or line.startswith(';'): # Skip empty lines and comments
                continue

            parts = line.split() # Split line into parts (opcode and operands)
            opcode_str = parts[0].upper() # Get opcode string and convert to uppercase
            opcode = OpCode[opcode_str] # Convert opcode string to OpCode enum

            operands_str = parts[1:] # Get operand strings
            operands = [op.strip() for ops in operands_str for op in ops.split(',')] # Handle comma-separated operands
            operands = [op for op in operands if op] # Filter out empty operands

            if opcode == OpCode.STORE: # STORE dest_symbol, Rsrc1
                dest_operand = operands[0] # Symbolic destination (not used in this simple example)
                src1_reg = self._resolve_operand(operands[1]) # Source register
                instructions.append(Instruction(opcode, 0, src1_reg, 0)) # dest_reg is dummy for STORE in this example
            elif opcode == OpCode.LOAD: # LOAD Rdest, src
                dest_reg = int(operands[0][1:]) # Destination register
                src = self._resolve_operand(operands[1]) # Source operand (immediate or register - in LOAD, it's immediate value)
                instructions.append(Instruction(opcode, dest_reg, src, 0)) # src is immediate value, stored in src1_reg
            elif opcode == OpCode.SHIFT: # SHIFT Rdest, Rsrc, immediate
                dest_reg = int(operands[0][1:])
                src1_reg = self._resolve_operand(operands[1])
                immediate = int(operands[2])
                instructions.append(Instruction(opcode, dest_reg, src1_reg, 0, immediate)) # Immediate for SHIFT
            elif opcode in [OpCode.GRAY_ENCODE, OpCode.GRAY_DECODE]: # GRAY_ENCODE/DECODE Rdest, Rsrc
                dest_reg = int(operands[0][1:])
                src1_reg = self._resolve_operand(operands[1])
                instructions.append(Instruction(opcode, dest_reg, src1_reg, 0))
            else: # OP Rdest, Rsrc1, Rsrc2 (NAND, TROPICAL_ADD, TROPICAL_MUL)
                dest_reg = int(operands[0][1:]) # Destination register
                src1_reg = self._resolve_operand(operands[1]) # Source register 1
                src2_reg = self._resolve_operand(operands[2]) # Source register 2
                instructions.append(Instruction(opcode, dest_reg, src1_reg, src2_reg))
        return instructions

    def _resolve_operand(self, operand: str) -> int:
        """Resolves an operand string to an integer value (register number or immediate)."""
        if operand.startswith('R'): # Register operand
            return int(operand[1:]) # Extract register number
        elif operand in self.symbol_table: # Symbolic operand (label - not used in this example)
            return self.symbol_table[operand]
        else: # Immediate value operand
            try:
                return int(operand) # Convert to integer
            except ValueError:
                raise ValueError(f"Invalid operand: {operand}")

class AssemblyOptimizer:
    """Optimizes assembly code instructions (currently a placeholder)."""
    def __init__(self, dimension: int):
        """Initializes the AssemblyOptimizer."""
        self.dimension = dimension
        self.dependency_graph = {} # Dependency graph for instruction scheduling (not implemented yet)

    def analyze_dependencies(self, instructions: List[Instruction]) -> Dict:
        """Analyzes data dependencies between instructions (not implemented yet)."""
        dependencies = {} # Placeholder for dependency analysis
        return dependencies

    def _find_register_dependencies(self, prev_instructions: List[Instruction], reg: int) -> Set[int]:
        """Finds instructions that the current instruction depends on due to register usage (not implemented yet)."""
        dependencies = set() # Placeholder for dependency finding
        return dependencies

    def schedule_instructions(self, instructions: List[Instruction]) -> List[List[Instruction]]:
        """Schedules instructions for optimized execution (currently a placeholder - no scheduling)."""
        scheduled_instructions = [instructions] # No scheduling implemented yet - execute instructions in order
        return scheduled_instructions

class HypercubeRouter:
    """Simulates routing in a hypercube topology (not implemented in this example)."""
    def __init__(self, dimension: int):
        """Initializes the HypercubeRouter."""
        self.dimension = dimension
        self.size = 2 ** dimension

    def calculate_route(self, source: int, destination: int) -> List[int]:
        """Calculates the shortest route between two nodes in a hypercube (not used in this example)."""
        route = [source] # Placeholder route calculation
        return route

    def generate_routing_table(self) -> Dict[Tuple[int, int], List[int]]:
        """Generates a routing table for the hypercube (not used in this example)."""
        routing_table = {} # Placeholder routing table generation
        return routing_table

class TropicalAlgebra:
    """Implements Tropical Algebra operations using CUDA kernels (matrix/vector operations)."""
    def __init__(self, cuda_context):
        """Initializes the TropicalAlgebra module."""
        self.cuda_context = cuda_context

    def matrix_multiply(self, matrix_a_gpu, matrix_b_gpu, result_gpu, matrix_size):
        """Performs tropical matrix multiplication using a CUDA kernel."""
        # Kernel execution parameters
        block_dim = (16, 16, 1)
        grid_dim = (
            (matrix_size + block_dim[0] - 1) // block_dim[0],
            (matrix_size + block_dim[1] - 1) // block_dim[1]
        )
        # Launch kernel
        # Note: Assuming tropical_kernel is modified for matrix multiplication or a dedicated kernel exists.
        # For now, using the vector kernel as a placeholder - matrix multiplication kernel needs to be implemented.
        # self.tropical_kernel (This is vector kernel, matrix kernel needs separate implementation and call)
        pass # Matrix multiplication kernel call to be implemented

class VirtualMachine:
    """Orchestrates the virtual CPU, compiler, optimizer, and execution."""
    def __init__(self, cpu: ParallelVirtualCPU, optimizer: AssemblyOptimizer):
        """Initializes the VirtualMachine."""
        self.cpu = cpu
        self.optimizer = optimizer
        self.memory = np.zeros(1024 * 1024, dtype=np.int32)  # 1MB memory (host) - not directly used in this example
        self.registers = np.zeros((32, self.cpu.size), dtype=np.int32) # Registers (host) - mirrored from CPU, now vector registers

    def load_program(self, assembly_code: str) -> List[Instruction]: # Modified to return instructions
        """Loads and executes an assembly program."""
        compiler = AssemblyCompiler()
        instructions = compiler.compile(assembly_code) # Compile assembly code
        scheduled_instructions = self.optimizer.schedule_instructions(instructions) # Optimize instructions (currently no-op)

        self.cpu.execute_program([inst for wave in scheduled_instructions for inst in wave]) # Execute program
        self.registers = self.cpu.register_file # Update VM's register view
        return instructions # Return the instructions

class DebugPrinter:
    """Manages debug output for the Virtual CPU."""
    def __init__(self, debug_level=logging.INFO):
        """Initializes the DebugPrinter."""
        self.logger = logging.getLogger("VirtualCPU")
        self.logger.setLevel(debug_level)

        # Configure console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(debug_level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def print_instruction(self, instruction: Instruction, pc: int):
        """Logs the execution of an instruction."""
        self.logger.info(f"PC={pc:04x}: {instruction.opcode.name} "
                        f"R{instruction.dest_reg}, R{instruction.src1_reg}, R{instruction.src2_reg} Immediate={instruction.immediate}")

    def print_registers(self, registers: np.ndarray):
        """Logs the state of the registers."""
        reg_str = "\nRegister State (per node):\n"
        for i in range(0, len(registers), 4):
            reg_str += f"R{i:2d}-R{i+3:2d}:\n"
            for node_id in range(registers.shape[1]): # Iterate through nodes
                reg_str += f"  Node {node_id}: "
                reg_str += " ".join(f"{registers[j][node_id]:08x}" for j in range(i, min(i+4, len(registers))))
                reg_str += "\n"
        ##self.logger.info(reg_str)

    def print_memory_block(self, memory: np.ndarray, start: int, size: int):
        """Logs a block of memory (not used in this example)."""
        mem_str = f"\nMemory Dump (0x{start:04x} - 0x{start+size-1:04x}):\n" # Placeholder for memory dump
        self.logger.info(mem_str) # Memory dump not implemented in detail

def main():
    """Main function to demonstrate the Virtual CPU."""
    # Initialization
    dimension = 4
    cpu = ParallelVirtualCPU(dimension)
    optimizer = AssemblyOptimizer(dimension)
    vm = VirtualMachine(cpu, optimizer)
    debug = DebugPrinter(debug_level=logging.DEBUG) # Enable DEBUG level for more detailed output

    print("仮想CPUシミュレーション開始")
    print(f"ハイパーキューブ次元: {dimension}")
    print(f"利用可能なノード数: {2**dimension}")

    # Sample data preparation (not used in this register-based example)
    print("\nサンプルデータの準備中...")
    matrix_a = np.random.rand(8, 8).astype(np.float32)
    matrix_b = np.random.rand(8, 8).astype(np.float32)

    print("\n入力行列A:")
    print(matrix_a)
    print("\n入力行列B:")
    print(matrix_b)

    # Sample assembly code
    assembly_code = """
    ; Tropical ALU and NAND example + SHIFT, GRAY_ENCODE, GRAY_DECODE
    LOAD R1, 2    ; Load immediate value 2 to R1
    LOAD R2, 3    ; Load immediate value 3 to R2
    TROPICAL_MUL R3, R1, R2  ; R3 = R1 + R2 (tropical mult is regular add)
    NAND R4, R1, R2 ; R4 = NAND(R1, R2)
    TROPICAL_ADD R5, R1, R2 ; R5 = min(R1, R2)
    SHIFT R6, R1, 2 ; R6 = R1 << 2 (shift left by 2)
    GRAY_ENCODE R7, R1 ; R7 = Gray code of R1
    GRAY_DECODE R8, R7 ; R8 = Binary code from Gray code R7
    STORE result, R3     ; Dummy store, result in R3 register (no memory store in this example)
    """

    print("\nアセンブリコード:")
    print(assembly_code)

    # Program loading and execution
    print("\nプログラムをロード中...")
    instructions = vm.load_program(assembly_code) # Load program and get instructions

    print("\nレジスタ状態 (初期状態):")
    debug.print_registers(vm.cpu.register_file)

    print("\nプログラム実行開始:")
    # Removed incorrect loop

    vm.cpu.execute_program(instructions) # Pass the instructions to execute_program

    print("\nレジスタ状態 (実行後):")
    debug.print_registers(vm.cpu.register_file)

    # Get and display results
    result_registers = vm.cpu.register_file
    print("\n計算結果 (レジスタ R3, R4, R5, R6, R7, R8, for Node 0):") # Show results for Node 0 for brevity
    node_zero_results = result_registers[:, 0] # Get values for node 0
    print(f"R3 (TROPICAL_MUL): {node_zero_results[3]}") # R3 = 2 + 3 = 5
    print(f"R4 (NAND): {node_zero_results[4]}")       # R4 = NAND(2, 3) = -1 (int representation of all bits 1s)
    print(f"R5 (TROPICAL_ADD): {node_zero_results[5]}")      # R5 = min(2, 3) = 2
    print(f"R6 (SHIFT): {node_zero_results[6]}")         # R6 = 2 << 2 = 8
    print(f"R7 (GRAY_ENCODE): {node_zero_results[7]}")    # R7 = GrayEncode(2) = 3
    print(f"R8 (GRAY_DECODE): {node_zero_results[8]}")    # R8 = GrayDecode(3) = 2


    # Performance information (not applicable for single register operations)
    print("\nパフォーマンス情報:")
    print(f"総実行ノード数: {2**dimension}")
    print(f"並列実行ブロック数: Configured in CUDA kernels")
    print(f"1ブロックあたりのスレッド数: Configured in CUDA kernels")

    # Memory usage (not applicable in this register-based example)
    print("\nメモリ使用状況:")
    print(f"入力行列サイズ: {matrix_a.nbytes / 1024:.2f}KB x 2 (not used in this example)")
    print(f"出力行列サイズ: {matrix_a.nbytes / 1024:.2f}KB (not used in this example)")

    # Resource cleanup
    print("\nリソースを解放中...")
    del cpu
    del optimizer
    del vm
    del debug

    print("\n実行完了")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        traceback.print_exc()
