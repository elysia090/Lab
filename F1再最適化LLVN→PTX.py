import llvmlite.ir as ir
import llvmlite.binding as llvm
import subprocess, tempfile, os, time, math
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

# --- LLVM IR 生成／最適化／PTX へのコンパイル ---
def gen_llvm_ir():
    module = ir.Module(name="vecAdd")
    module.triple = "nvptx64-nvidia-cuda"
    i32 = ir.IntType(32)
    f32 = ir.FloatType()
    f32_ptr = ir.PointerType(f32)
    func = ir.Function(module, ir.FunctionType(ir.VoidType(), [f32_ptr, f32_ptr, f32_ptr, i32]), name="vecAdd")
    a, b, c, n = func.args
    a.name, b.name, c.name, n.name = "a", "b", "c", "n"
    
    entry = func.append_basic_block("entry")
    builder = ir.IRBuilder(entry)
    
    # NVVM 内蔵関数
    tid    = builder.call(ir.Function(module, ir.FunctionType(i32, []), name="llvm.nvvm.read.ptx.sreg.tid.x"), [])
    ctaid  = builder.call(ir.Function(module, ir.FunctionType(i32, []), name="llvm.nvvm.read.ptx.sreg.ctaid.x"), [])
    ntid   = builder.call(ir.Function(module, ir.FunctionType(i32, []), name="llvm.nvvm.read.ptx.sreg.ntid.x"), [])
    nctaid = builder.call(ir.Function(module, ir.FunctionType(i32, []), name="llvm.nvvm.read.ptx.sreg.nctaid.x"), [])
    start  = builder.add(tid, builder.mul(ctaid, ntid))
    stride = builder.mul(ntid, nctaid)
    
    loop   = func.append_basic_block("loop")
    body   = func.append_basic_block("body")
    exit_bb= func.append_basic_block("exit")
    
    builder.branch(loop)
    builder.position_at_start(loop)
    phi = builder.phi(i32, "i")
    phi.add_incoming(start, entry)
    builder.cbranch(builder.icmp_signed("<", phi, n), body, exit_bb)
    
    builder.position_at_start(body)
    a_val = builder.load(builder.gep(a, [phi]))
    b_val = builder.load(builder.gep(b, [phi]))
    builder.store(builder.fadd(a_val, b_val), builder.gep(c, [phi]))
    phi_next = builder.add(phi, stride)
    builder.branch(loop)
    phi.add_incoming(phi_next, body)
    
    builder.position_at_start(exit_bb)
    builder.ret_void()
    
    module_str = str(module) + """
!nvvm.annotations = !{!0}
!0 = !{ i8* bitcast (void (float*, float*, float*, i32)* @vecAdd to i8*), !"kernel", i32 1 }
@llvm.used = appending global [1 x i8*] [i8* bitcast (void (float*, float*, float*, i32)* @vecAdd to i8*)]
"""
    return module_str

def compile_ptx(llvm_ir, cc="sm_35"):
    llvm.initialize(); llvm.initialize_native_target(); llvm.initialize_native_asmprinter()
    mod = llvm.parse_assembly(llvm_ir)
    mod.verify()
    pm = llvm.create_module_pass_manager()
    pm.run(mod)
    optimized_ir = str(mod)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".ll", delete=False) as f:
        f.write(optimized_ir)
        ll_file = f.name
    ptx_file = ll_file.replace(".ll", ".ptx")
    subprocess.run(["llc", "-O3", "-march=nvptx64", f"-mcpu={cc}", ll_file, "-o", ptx_file], check=True)
    with open(ptx_file, "r") as f:
        ptx = f.read()
    os.remove(ll_file); os.remove(ptx_file)
    return ptx.replace(".func vecAdd(", ".entry vecAdd(")

# PTX コードのグローバルキャッシュ
PTX_CODE = compile_ptx(gen_llvm_ir())
mod = cuda.module_from_buffer(PTX_CODE.encode("utf-8"))
kernel = mod.get_function("vecAdd")

# --- 非同期バッファのキャッシュ（リングバッファ化） ---
async_buffers = None  # グローバルキャッシュ

def get_async_buffers(chunk_size, dtype, num_buffers=4):
    """
    チャンクサイズおよびバッファ数が変わらない限り、ピン止めホスト／デバイスメモリ、ストリームを再利用する。
    """
    global async_buffers
    if (async_buffers is None or 
        async_buffers.get('chunk_size') != chunk_size or 
        async_buffers.get('num_buffers') != num_buffers):
        buffers = []
        for _ in range(num_buffers):
            pinned_a = cuda.pagelocked_empty(chunk_size, dtype)
            pinned_b = cuda.pagelocked_empty(chunk_size, dtype)
            pinned_c = cuda.pagelocked_empty(chunk_size, dtype)
            dev_a = cuda.mem_alloc(pinned_a.nbytes)
            dev_b = cuda.mem_alloc(pinned_b.nbytes)
            dev_c = cuda.mem_alloc(pinned_c.nbytes)
            stream = cuda.Stream()
            buffers.append({
                'pinned_a': pinned_a,
                'pinned_b': pinned_b,
                'pinned_c': pinned_c,
                'dev_a': dev_a,
                'dev_b': dev_b,
                'dev_c': dev_c,
                'stream': stream
            })
        async_buffers = {'chunk_size': chunk_size, 'num_buffers': num_buffers, 'buffers': buffers}
    return async_buffers['buffers']

def vecAdd_async_optimized(a, b, chunk_size=256*1024, num_buffers=4):
    """
    より高いスループットを狙うため、リングバッファによる非同期転送・カーネル起動を実施する。
    ・バッファ再利用時にのみ synchronize() を呼び、各チャンクの転送・実行をパイプライン化する。
    """
    N = a.size
    result = np.empty_like(a)
    buffers = get_async_buffers(chunk_size, a.dtype, num_buffers)
    num_chunks = math.ceil(N / chunk_size)
    
    # 各チャンクの処理をスケジュール
    for i in range(num_chunks):
        start_idx = i * chunk_size
        current_size = min(chunk_size, N - start_idx)
        buf = buffers[i % num_buffers]
        
        # 同じバッファを再利用する場合、前回の処理完了を待ち、結果を回収する
        if i >= num_buffers:
            buf['stream'].synchronize()
            prev_idx = i - num_buffers
            prev_start = prev_idx * chunk_size
            prev_size = min(chunk_size, N - prev_start)
            result[prev_start:prev_start+prev_size] = buf['pinned_c'][:prev_size]
        
        # 入力データをピン止めホストメモリにコピー
        buf['pinned_a'][:current_size] = a[start_idx:start_idx+current_size]
        buf['pinned_b'][:current_size] = b[start_idx:start_idx+current_size]
        
        # 非同期転送＆カーネル起動
        cuda.memcpy_htod_async(buf['dev_a'], buf['pinned_a'], buf['stream'])
        cuda.memcpy_htod_async(buf['dev_b'], buf['pinned_b'], buf['stream'])
        grid = ((current_size + 255) // 256, 1)
        kernel(buf['dev_a'], buf['dev_b'], buf['dev_c'], np.int32(current_size),
               block=(256, 1, 1), grid=grid, stream=buf['stream'])
        cuda.memcpy_dtoh_async(buf['pinned_c'], buf['dev_c'], buf['stream'])
    
    # ループ終了後、残りのバッファの処理を待って結果を回収
    remaining = min(num_buffers, num_chunks)
    for j in range(remaining):
        buf = buffers[j]
        buf['stream'].synchronize()
        start_idx = (num_chunks - remaining + j) * chunk_size
        current_size = min(chunk_size, N - start_idx)
        result[start_idx:start_idx+current_size] = buf['pinned_c'][:current_size]
    
    return result

# --- テスト／ベンチマーク ---
if __name__ == "__main__":
    N = 1024 * 1024
    a = np.random.randn(N).astype(np.float32)
    b = np.random.randn(N).astype(np.float32)
    
    # 正当性確認
    c = vecAdd_async_optimized(a, b)
    assert np.allclose(c, a + b), "カーネル実行が失敗しました"
    
    # ベンチマーク
    iterations = 100
    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = vecAdd_async_optimized(a, b)
    avg_time = (time.perf_counter() - t0) / iterations * 1000
    print("Optimized Async average time: {:.2f} ms".format(avg_time))
