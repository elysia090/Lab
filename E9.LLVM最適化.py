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
import time
import math

class VecAddKernel:
    # PTX コードはクラス変数としてキャッシュ（初回生成後は再生成しない）
    _ptx_cache = None

    def __init__(self, compute_capability="sm_35"):
        self.compute_capability = compute_capability
        # 非同期実行用のバッファセットキャッシュ（チャンクサイズが同じなら再利用）
        self.async_buffers = None  
        if VecAddKernel._ptx_cache is None:
            llvm_ir = self.generate_llvm_ir()
            optimized_ir = self.optimize_llvm_ir(llvm_ir)
            ptx_code = self.compile_to_ptx(optimized_ir)
            # NVPTX バックエンドでは関数宣言が .func として出力されるため、
            # cuModuleGetFunction で認識されるよう .entry に置換する
            ptx_code = ptx_code.replace(".func vecAdd(", ".entry vecAdd(")
            VecAddKernel._ptx_cache = ptx_code
        self.ptx_code = VecAddKernel._ptx_cache
        # PTX コードを直接バッファからロード（nvcc による再コンパイル回避）
        self.cuda_module = cuda.module_from_buffer(self.ptx_code.encode("utf-8"))
        try:
            self.kernel_function = self.cuda_module.get_function("vecAdd")
        except Exception as e:
            print("生成された PTX code:\n", self.ptx_code)
            raise RuntimeError("カーネル関数 'vecAdd' が見つかりませんでした: " + str(e))

    def generate_llvm_ir(self) -> str:
        module = ir.Module(name="vecAddModule")
        module.triple = "nvptx64-nvidia-cuda"

        # 型定義
        i32 = ir.IntType(32)
        float_ty = ir.FloatType()
        float_ptr = ir.PointerType(float_ty)

        # カーネル関数: void vecAdd(float* a, float* b, float* c, i32 n)
        func_ty = ir.FunctionType(ir.VoidType(), [float_ptr, float_ptr, float_ptr, i32])
        kernel_func = ir.Function(module, func_ty, name="vecAdd")
        a, b, c, n = kernel_func.args
        a.name = "a"
        b.name = "b"
        c.name = "c"
        n.name = "n"

        # エントリーブロック作成
        entry_block = kernel_func.append_basic_block("entry")
        builder = ir.IRBuilder(entry_block)

        # NVVM 内蔵関数宣言
        tid_func    = ir.Function(module, ir.FunctionType(i32, []), name="llvm.nvvm.read.ptx.sreg.tid.x")
        ctaid_func  = ir.Function(module, ir.FunctionType(i32, []), name="llvm.nvvm.read.ptx.sreg.ctaid.x")
        ntid_func   = ir.Function(module, ir.FunctionType(i32, []), name="llvm.nvvm.read.ptx.sreg.ntid.x")
        nctaid_func = ir.Function(module, ir.FunctionType(i32, []), name="llvm.nvvm.read.ptx.sreg.nctaid.x")

        tid_val    = builder.call(tid_func, [])
        ctaid_val  = builder.call(ctaid_func, [])
        ntid_val   = builder.call(ntid_func, [])
        nctaid_val = builder.call(nctaid_func, [])

        # start = tid + ctaid * ntid
        start = builder.add(tid_val, builder.mul(ctaid_val, ntid_val))
        # stride = ntid * nctaid (全グリッドのスレッド数)
        stride = builder.mul(ntid_val, nctaid_val)

        # グリッドストライドループのブロック作成
        loop_block = kernel_func.append_basic_block("loop")
        body_block = kernel_func.append_basic_block("body")
        exit_block = kernel_func.append_basic_block("exit")
        builder.branch(loop_block)

        # ループブロック：PHI ノードでループ変数 i を管理
        builder.position_at_start(loop_block)
        loop_var = builder.phi(i32, "i")
        loop_var.add_incoming(start, entry_block)

        # ループ条件: if (i < n) then body else exit
        cond = builder.icmp_signed("<", loop_var, n)
        builder.cbranch(cond, body_block, exit_block)

        # ループ本体
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

        # ループ終了ブロック
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
        """同期実行：ホスト⇔デバイス転送を含む全体処理"""
        N = a_np.size
        c_np = np.empty_like(a_np)
        a_gpu = cuda.mem_alloc(a_np.nbytes)
        b_gpu = cuda.mem_alloc(b_np.nbytes)
        c_gpu = cuda.mem_alloc(c_np.nbytes)
        cuda.memcpy_htod(a_gpu, a_np)
        cuda.memcpy_htod(b_gpu, b_np)
        block_size = 256
        grid_size = (N + block_size - 1) // block_size
        # 正しくは位置引数のみで呼び出す
        self.kernel_function(a_gpu, b_gpu, c_gpu, np.int32(N),
                             block=(block_size, 1, 1), grid=(grid_size, 1))
        cuda.memcpy_dtoh(c_np, c_gpu)
        return c_np


    def run_async_double_buffered(self, a_np: np.ndarray, b_np: np.ndarray, chunk_size: int = 256*1024) -> np.ndarray:
        """
        ダブルバッファリング＋複数ストリームを用いて、入力配列をチャンク単位で非同期実行する。
        バッファセットは事前にまとめてキャッシュし、各チャンク処理で再利用する。
        """
        N = a_np.size
        result = np.empty_like(a_np)
        num_chunks = math.ceil(N / chunk_size)
        # バッファセットが未確保、またはチャンクサイズが変更された場合は再確保
        if self.async_buffers is None or self.async_buffers.get("chunk_size", None) != chunk_size:
            buffers = []
            for i in range(2):
                pinned_a = cuda.pagelocked_empty(chunk_size, a_np.dtype)
                pinned_b = cuda.pagelocked_empty(chunk_size, b_np.dtype)
                pinned_c = cuda.pagelocked_empty(chunk_size, a_np.dtype)
                dev_a = cuda.mem_alloc(pinned_a.nbytes)
                dev_b = cuda.mem_alloc(pinned_b.nbytes)
                dev_c = cuda.mem_alloc(pinned_c.nbytes)
                stream = cuda.Stream()
                buffers.append({
                    'pinned_a': pinned_a, 'pinned_b': pinned_b, 'pinned_c': pinned_c,
                    'dev_a': dev_a, 'dev_b': dev_b, 'dev_c': dev_c,
                    'stream': stream
                })
            self.async_buffers = {"chunk_size": chunk_size, "buffers": buffers}
        else:
            buffers = self.async_buffers["buffers"]

        for chunk in range(num_chunks):
            start_idx = chunk * chunk_size
            current_size = min(chunk_size, N - start_idx)
            buf = buffers[chunk % 2]
            # コピー入力データ
            buf['pinned_a'][:current_size] = a_np[start_idx:start_idx+current_size]
            buf['pinned_b'][:current_size] = b_np[start_idx:start_idx+current_size]
            # 非同期転送：ホスト → デバイス
            cuda.memcpy_htod_async(buf['dev_a'], buf['pinned_a'], buf['stream'])
            cuda.memcpy_htod_async(buf['dev_b'], buf['pinned_b'], buf['stream'])
            blk = 256
            grd = (current_size + blk - 1) // blk
            # カーネル起動（非同期）
            self.kernel_function(buf['dev_a'], buf['dev_b'], buf['dev_c'], np.int32(current_size),
                                 block=(blk, 1, 1), grid=(grd, 1), stream=buf['stream'])
            # 非同期転送：デバイス → ホスト
            cuda.memcpy_dtoh_async(buf['pinned_c'], buf['dev_c'], buf['stream'])
            buf['stream'].synchronize()
            result[start_idx:start_idx+current_size] = buf['pinned_c'][:current_size]
        return result

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

def benchmark_kernel_async_double_buffered(kernel: VecAddKernel, a_np: np.ndarray, b_np: np.ndarray, iterations: int = 10):
    times = []
    for _ in range(iterations):
        start_event = cuda.Event()
        end_event = cuda.Event()
        start_event.record()
        _ = kernel.run_async_double_buffered(a_np, b_np)
        end_event.record()
        end_event.synchronize()
        times.append(end_event.time_since(start_event))
    avg_async_time = sum(times) / len(times)
    print(f"[Async Double-Buffered Benchmark] 平均実行時間: {avg_async_time:.2f} ms over {iterations} iterations")

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
    benchmark_kernel_async_double_buffered(kernel, a_np, b_np, iterations)

if __name__ == "__main__":
    main()
