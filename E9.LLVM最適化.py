!apt-get update
!apt-get install -y llvm

import llvmlite.ir as ir
import llvmlite.binding as llvm
import subprocess
import pycuda.driver as cuda
import pycuda.autoinit  # CUDA の初期化
#from pycuda.compiler import SourceModule  # 使用しません
import numpy as np
import tempfile
import os

def generate_llvm_ir() -> str:
    """
    LLVM IR を生成し、NVVM 用のメタデータと llvm.used を付与して文字列として返す。
    """
    # モジュール作成とターゲット設定
    module = ir.Module(name="vecAddModule")
    module.triple = "nvptx64-nvidia-cuda"

    # 型の定義
    i32 = ir.IntType(32)
    float_ty = ir.FloatType()
    float_ptr = ir.PointerType(float_ty)

    # 関数定義: void vecAdd(float*, float*, float*, i32)
    func_ty = ir.FunctionType(ir.VoidType(), [float_ptr, float_ptr, float_ptr, i32])
    kernel_func = ir.Function(module, func_ty, name="vecAdd")
    # 既定の外部リンク（external linkage）で十分
    a, b, c, n = kernel_func.args
    a.name = "a"
    b.name = "b"
    c.name = "c"
    n.name = "n"

    # エントリーブロックの作成
    entry_block = kernel_func.append_basic_block(name="entry")
    builder = ir.IRBuilder(entry_block)

    # NVVM 組み込み関数の宣言 (threadIdx.x, blockIdx.x, blockDim.x)
    nv_tid_ty = ir.FunctionType(i32, [])
    tid_func = ir.Function(module, nv_tid_ty, name="llvm.nvvm.read.ptx.sreg.tid.x")
    blockIdx_func = ir.Function(module, nv_tid_ty, name="llvm.nvvm.read.ptx.sreg.ctaid.x")
    blockDim_func = ir.Function(module, nv_tid_ty, name="llvm.nvvm.read.ptx.sreg.ntid.x")

    tid_val = builder.call(tid_func, [])
    blockIdx_val = builder.call(blockIdx_func, [])
    blockDim_val = builder.call(blockDim_func, [])

    # インデックス計算: i = blockIdx.x * blockDim.x + threadIdx.x
    i_val = builder.add(builder.mul(blockIdx_val, blockDim_val), tid_val)

    # if (i < n) 分岐
    cond = builder.icmp_signed('<', i_val, n)
    then_block = kernel_func.append_basic_block(name="then")
    end_block = kernel_func.append_basic_block(name="end")
    builder.cbranch(cond, then_block, end_block)

    # then ブロック: c[i] = a[i] + b[i]
    builder.position_at_start(then_block)
    a_ptr = builder.gep(a, [i_val], inbounds=True)
    b_ptr = builder.gep(b, [i_val], inbounds=True)
    c_ptr = builder.gep(c, [i_val], inbounds=True)
    a_val = builder.load(a_ptr)
    b_val = builder.load(b_ptr)
    sum_val = builder.fadd(a_val, b_val)
    builder.store(sum_val, c_ptr)
    builder.branch(end_block)

    # end ブロック
    builder.position_at_start(end_block)
    builder.ret_void()

    # NVVM 用メタデータの付与  
    # 定数式として "i8* bitcast (...)" の形で記述
    llvm_ir = str(module) + """
!nvvm.annotations = !{!0}
!0 = !{ i8* bitcast (void (float*, float*, float*, i32)* @vecAdd to i8*), !"kernel", i32 1 }
"""
    # llvm.used にカーネル関数のポインタを追加（最適化時の削除防止）
    llvm_ir += """
@llvm.used = appending global [1 x i8*] [i8* bitcast (void (float*, float*, float*, i32)* @vecAdd to i8*)]
"""
    return llvm_ir

def optimize_llvm_ir(llvm_ir: str) -> str:
    """
    llvmlite.binding を用いて LLVM IR に最適化パスを適用し、
    最適化済み IR を文字列で返す。
    """
    llvm.initialize()
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()

    module_ref = llvm.parse_assembly(llvm_ir)
    module_ref.verify()

    pass_manager = llvm.create_module_pass_manager()
    pass_manager.run(module_ref)
    return str(module_ref)

def compile_to_ptx(optimized_ir: str, compute_capability: str = "sm_35") -> str:
    """
    最適化済み LLVM IR から llc を使い PTX コードを生成する。
    一時ファイルを利用して llc を呼び出し、生成した PTX コードを文字列で返す。
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".ll", delete=False) as f:
        ir_filename = f.name
        f.write(optimized_ir)
    ptx_filename = ir_filename.replace(".ll", ".ptx")
    try:
        llc_cmd = [
            "llc", "-O3", "-march=nvptx64", f"-mcpu={compute_capability}",
            ir_filename, "-o", ptx_filename
        ]
        subprocess.run(llc_cmd, check=True)
        with open(ptx_filename, "r") as f:
            ptx_code = f.read()
    finally:
        # 一時ファイルのクリーンアップ
        if os.path.exists(ir_filename):
            os.remove(ir_filename)
        if os.path.exists(ptx_filename):
            os.remove(ptx_filename)
    # --- 修正ポイント ---
    # PTX コード中のカーネル関数が .func として出力される場合があるため、
    # cuModuleGetFunction に認識されるよう .entry に書き換える。
    ptx_code = ptx_code.replace(".func vecAdd(", ".entry vecAdd(")
    return ptx_code

def run_kernel(ptx_code: str, N: int = 1024):
    """
    生成済みの PTX コードを直接ロードし、pyCUDA によってカーネルを実行、
    計算結果を検証する。
    """
    # すでに PTX コードなので、バッファから直接ロードする
    mod = cuda.module_from_buffer(ptx_code.encode('utf-8'))
    try:
        vecAdd = mod.get_function("vecAdd")
    except Exception as e:
        print("PTX code:\n", ptx_code)
        raise RuntimeError("カーネル関数 'vecAdd' が見つかりませんでした: " + str(e))

    # サンプルデータの生成
    a_np = np.random.randn(N).astype(np.float32)
    b_np = np.random.randn(N).astype(np.float32)
    c_np = np.empty_like(a_np)

    # GPU メモリ確保とホスト→デバイスへの転送
    a_gpu = cuda.mem_alloc(a_np.nbytes)
    b_gpu = cuda.mem_alloc(b_np.nbytes)
    c_gpu = cuda.mem_alloc(c_np.nbytes)
    cuda.memcpy_htod(a_gpu, a_np)
    cuda.memcpy_htod(b_gpu, b_np)

    # カーネル呼び出し設定: スレッド数など
    block_size = 256
    grid_size = (N + block_size - 1) // block_size

    vecAdd(a_gpu, b_gpu, c_gpu, np.int32(N),
           block=(block_size, 1, 1), grid=(grid_size, 1))

    # 結果をホストに転送し、検証
    cuda.memcpy_dtoh(c_np, c_gpu)
    if np.allclose(c_np, a_np + b_np):
        print("Kernel execution successful!")
    else:
        print("Kernel execution failed!")

def main():
    try:
        print("Generating LLVM IR...")
        llvm_ir = generate_llvm_ir()

        print("Optimizing LLVM IR...")
        optimized_ir = optimize_llvm_ir(llvm_ir)

        print("Compiling to PTX...")
        ptx_code = compile_to_ptx(optimized_ir)

        print("Running kernel...")
        run_kernel(ptx_code)
    except Exception as e:
        print("Error occurred:", e)

if __name__ == "__main__":
    main()
