!apt-get update
!apt-get install -y llvm

import llvmlite.ir as ir
import llvmlite.binding as llvm
import subprocess
import pycuda.driver as cuda
import pycuda.autoinit  # CUDA の初期化
import numpy as np
import tempfile
import os
import time  # ベンチマーク用

class VecAddKernel:
    # PTX コードはクラス変数としてキャッシュ（初回生成後は再生成しない）
    _ptx_cache = None

    def __init__(self, compute_capability="sm_35"):
        self.compute_capability = compute_capability
        # 非同期実行用のストリーム（再利用する）
        self.async_stream = cuda.Stream()
        # 非同期用のバッファは後で確保する（入力サイズに依存するため）
        self.async_host_a = None
        self.async_host_b = None
        self.async_host_c = None
        self.async_dev_a = None
        self.async_dev_b = None
        self.async_dev_c = None

        if VecAddKernel._ptx_cache is None:
            llvm_ir = self.generate_llvm_ir()
            optimized_ir = self.optimize_llvm_ir(llvm_ir)
            ptx_code = self.compile_to_ptx(optimized_ir)
            # NVPTX バックエンドでは関数宣言が .func として出力されるため、
            # cuModuleGetFunction で認識されるよう .entry に置換する
            ptx_code = ptx_code.replace(".func vecAdd(", ".entry vecAdd(")
            VecAddKernel._ptx_cache = ptx_code
        self.ptx_code = VecAddKernel._ptx_cache
        # PTX コードを直接バッファからロード（nvcc による再コンパイルを回避）
        self.cuda_module = cuda.module_from_buffer(self.ptx_code.encode("utf-8"))
        try:
            self.kernel_function = self.cuda_module.get_function("vecAdd")
        except Exception as e:
            print("生成された PTX code:\n", self.ptx_code)
            raise RuntimeError("カーネル関数 'vecAdd' が見つかりませんでした: " + str(e))

    def generate_llvm_ir(self) -> str:
        # （前回と同じ IR 生成コード）
        module = ir.Module(name="vecAddModule")
        module.triple = "nvptx64-nvidia-cuda"
        i32 = ir.IntType(32)
        float_ty = ir.FloatType()
        float_ptr = ir.PointerType(float_ty)
        func_ty = ir.FunctionType(ir.VoidType(), [float_ptr, float_ptr, float_ptr, i32])
        kernel_func = ir.Function(module, func_ty, name="vecAdd")
        a, b, c, n = kernel_func.args
        a.name = "a"
        b.name = "b"
        c.name = "c"
        n.name = "n"
        entry_block = kernel_func.append_basic_block("entry")
        builder = ir.IRBuilder(entry_block)
        tid_func    = ir.Function(module, ir.FunctionType(i32, []), name="llvm.nvvm.read.ptx.sreg.tid.x")
        ctaid_func  = ir.Function(module, ir.FunctionType(i32, []), name="llvm.nvvm.read.ptx.sreg.ctaid.x")
        ntid_func   = ir.Function(module, ir.FunctionType(i32, []), name="llvm.nvvm.read.ptx.sreg.ntid.x")
        nctaid_func = ir.Function(module, ir.FunctionType(i32, []), name="llvm.nvvm.read.ptx.sreg.nctaid.x")
        tid_val    = builder.call(tid_func, [])
        ctaid_val  = builder.call(ctaid_func, [])
        ntid_val   = builder.call(ntid_func, [])
        nctaid_val = builder.call(nctaid_func, [])
        start = builder.add(tid_val, builder.mul(ctaid_val, ntid_val))
        stride = builder.mul(ntid_val, nctaid_val)
        loop_block = kernel_func.append_basic_block("loop")
        body_block = kernel_func.append_basic_block("body")
        exit_block = kernel_func.append_basic_block("exit")
        builder.branch(loop_block)
        builder.position_at_start(loop_block)
        loop_var = builder.phi(i32, "i")
        loop_var.add_incoming(start, entry_block)
        cond = builder.icmp_signed("<", loop_var, n)
        builder.cbranch(cond, body_block, exit_block)
        builder.position_at_start(body_block)
        a_ptr = builder.gep(a, [loop_var], inbounds=True)
        b_ptr = builder.gep(b, [loop_var], inbounds=True)
        c_ptr = builder.gep(c, [loop_var], inbounds=True)
        a_val = builder.load(a_ptr)
        b_val = builder.load(b_ptr)
        sum_val = builder.fadd(a_val, b_val)
        builder.store(sum_val, c_ptr)
        next_val = builder.add(loop_var, stride)
        builder.branch(loop_block)
        loop_var.add_incoming(next_val, body_block)
        builder.position_at_start(exit_block)
        builder.ret_void()
        llvm_ir = str(module) + """
!nvvm.annotations = !{!0}
!0 = !{ i8* bitcast (void (float*, float*, float*, i32)* @vecAdd to i8*), !"kernel", i32 1 }
@llvm.used = appending global [1 x i8*] [i8* bitcast (void (float*, float*, float*, i32)* @vecAdd to i8*)]
"""
        return llvm_ir

    def optimize_llvm_ir(self, llvm_ir: str) -> str:
        llvm.initialize()
        llvm.initialize_native_target()
        llvm.initialize_native_asmprinter()
        mod_ref = llvm.parse_assembly(llvm_ir)
        mod_ref.verify()
        pass_manager = llvm.create_module_pass_manager()
        pass_manager.run(mod_ref)
        return str(mod_ref)

    def compile_to_ptx(self, optimized_ir: str) -> str:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ll", delete=False) as f:
            ir_filename = f.name
            f.write(optimized_ir)
        ptx_filename = ir_filename.replace(".ll", ".ptx")
        try:
            llc_cmd = [
                "llc", "-O3", "-march=nvptx64", f"-mcpu={self.compute_capability}",
                ir_filename, "-o", ptx_filename
            ]
            subprocess.run(llc_cmd, check=True)
            with open(ptx_filename, "r") as f:
                ptx_code = f.read()
        finally:
            if os.path.exists(ir_filename):
                os.remove(ir_filename)
            if os.path.exists(ptx_filename):
                os.remove(ptx_filename)
        return ptx_code

    def run(self, a_np: np.ndarray, b_np: np.ndarray) -> np.ndarray:
        N = a_np.size
        c_np = np.empty_like(a_np)
        a_gpu = cuda.mem_alloc(a_np.nbytes)
        b_gpu = cuda.mem_alloc(b_np.nbytes)
        c_gpu = cuda.mem_alloc(c_np.nbytes)
        cuda.memcpy_htod(a_gpu, a_np)
        cuda.memcpy_htod(b_gpu, b_np)
        block_size = 256
        grid_size = (N + block_size - 1) // block_size
        self.kernel_function(a_gpu, b_gpu, c_gpu, np.int32(N),
                             block=(block_size, 1, 1), grid=(grid_size, 1))
        cuda.memcpy_dtoh(c_np, c_gpu)
        return c_np

    def run_async_optimized(self, a_np: np.ndarray, b_np: np.ndarray, stream=None) -> np.ndarray:
        """
        非同期実行時のバッファとストリームを再利用することで、毎回のメモリ割り当てオーバーヘッドを削減する。
        もし以前に確保済みでサイズが十分なら、それらを再利用します。
        """
        N = a_np.size
        if stream is None:
            stream = self.async_stream

        # 既に非同期用バッファが確保済みかチェックし、サイズが不足していれば再確保
        if (self.async_host_a is None or self.async_host_a.size < N):
            self.async_host_a = cuda.pagelocked_empty(a_np.shape, a_np.dtype)
            self.async_host_b = cuda.pagelocked_empty(b_np.shape, b_np.dtype)
            self.async_host_c = cuda.pagelocked_empty(a_np.shape, a_np.dtype)
            self.async_dev_a = cuda.mem_alloc(a_np.nbytes)
            self.async_dev_b = cuda.mem_alloc(b_np.nbytes)
            self.async_dev_c = cuda.mem_alloc(a_np.nbytes)
        # 入力データをピンメモリバッファにコピー（高速なホスト側メモリ）
        self.async_host_a[:] = a_np
        self.async_host_b[:] = b_np

        # 非同期転送（ホスト→デバイス）
        cuda.memcpy_htod_async(self.async_dev_a, self.async_host_a, stream)
        cuda.memcpy_htod_async(self.async_dev_b, self.async_host_b, stream)

        block_size = 256
        grid_size = (N + block_size - 1) // block_size
        # 非同期カーネル起動
        self.kernel_function(self.async_dev_a, self.async_dev_b, self.async_dev_c, np.int32(N),
                             block=(block_size, 1, 1), grid=(grid_size, 1), stream=stream)
        # 非同期転送（デバイス→ホスト）
        cuda.memcpy_dtoh_async(self.async_host_c, self.async_dev_c, stream)
        stream.synchronize()
        return self.async_host_c

def benchmark_kernel_host(kernel: VecAddKernel, a_np: np.ndarray, b_np: np.ndarray, iterations: int = 10):
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = kernel.run(a_np, b_np)
        end = time.perf_counter()
        times.append(end - start)
    avg_time_ms = (sum(times) / len(times)) * 1000
    print(f"[Host Benchmark] 平均実行時間: {avg_time_ms:.2f} ms over {iterations} iterations")

def benchmark_kernel_gpu(kernel: VecAddKernel, a_np: np.ndarray, b_np: np.ndarray, iterations: int = 10):
    N = a_np.size
    block_size = 256
    grid_size = (N + block_size - 1) // block_size
    a_gpu = cuda.mem_alloc(a_np.nbytes)
    b_gpu = cuda.mem_alloc(b_np.nbytes)
    c_gpu = cuda.mem_alloc(a_np.nbytes)
    cuda.memcpy_htod(a_gpu, a_np)
    cuda.memcpy_htod(b_gpu, b_np)
    times = []
    for _ in range(iterations):
        start_event = cuda.Event()
        end_event = cuda.Event()
        start_event.record()
        kernel.kernel_function(a_gpu, b_gpu, c_gpu, np.int32(N),
                                 block=(block_size, 1, 1), grid=(grid_size, 1))
        end_event.record()
        end_event.synchronize()
        times.append(end_event.time_since(start_event))
    avg_gpu_time = sum(times) / len(times)
    print(f"[GPU Benchmark] 平均カーネル実行時間: {avg_gpu_time:.2f} ms over {iterations} iterations")

def benchmark_kernel_async_optimized(kernel: VecAddKernel, a_np: np.ndarray, b_np: np.ndarray, iterations: int = 10):
    stream = kernel.async_stream  # 既存のストリームを再利用
    times = []
    for _ in range(iterations):
        start_event = cuda.Event()
        end_event = cuda.Event()
        start_event.record(stream)
        kernel.run_async_optimized(a_np, b_np, stream)
        end_event.record(stream)
        end_event.synchronize()
        times.append(end_event.time_since(start_event))
    avg_async_time = sum(times) / len(times)
    print(f"[Async Optimized GPU Benchmark] 平均非同期実行時間: {avg_async_time:.2f} ms over {iterations} iterations")

def main():
    N = 1024 * 1024
    a_np = np.random.randn(N).astype(np.float32)
    b_np = np.random.randn(N).astype(np.float32)
    kernel = VecAddKernel(compute_capability="sm_35")
    c_np = kernel.run(a_np, b_np)
    if np.allclose(c_np, a_np + b_np):
        print("Kernel execution successful!")
    else:
        print("Kernel execution failed!")
    iterations = 10
    benchmark_kernel_host(kernel, a_np, b_np, iterations)
    benchmark_kernel_gpu(kernel, a_np, b_np, iterations)
    benchmark_kernel_async_optimized(kernel, a_np, b_np, iterations)

if __name__ == "__main__":
    main()
