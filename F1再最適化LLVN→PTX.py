import llvmlite.ir as ir, llvmlite.binding as llvm, subprocess, tempfile, os, time, math
import pycuda.driver as cuda, pycuda.autoinit, numpy as np
from pycuda.tools import DeviceMemoryPool

# LLVM IR生成（ベクトル加算カーネル）
def gen_ir():
    mod = ir.Module(name="vecAdd"); mod.triple = "nvptx64-nvidia-cuda"
    i32, f32 = ir.IntType(32), ir.FloatType()
    f32p = ir.PointerType(f32)
    func = ir.Function(mod, ir.FunctionType(ir.VoidType(), [f32p, f32p, f32p, i32]), name="vecAdd")
    a, b, c, n = func.args; a.name, b.name, c.name, n.name = "a", "b", "c", "n"
    entry = func.append_basic_block("entry"); bld = ir.IRBuilder(entry)
    tid = bld.call(ir.Function(mod, ir.FunctionType(i32, []), name="llvm.nvvm.read.ptx.sreg.tid.x"), [])
    ctaid = bld.call(ir.Function(mod, ir.FunctionType(i32, []), name="llvm.nvvm.read.ptx.sreg.ctaid.x"), [])
    ntid = bld.call(ir.Function(mod, ir.FunctionType(i32, []), name="llvm.nvvm.read.ptx.sreg.ntid.x"), [])
    nctaid = bld.call(ir.Function(mod, ir.FunctionType(i32, []), name="llvm.nvvm.read.ptx.sreg.nctaid.x"), [])
    start = bld.add(tid, bld.mul(ctaid, ntid)); stride = bld.mul(ntid, nctaid)
    loop = func.append_basic_block("loop"); body = func.append_basic_block("body"); exit_bb = func.append_basic_block("exit")
    bld.branch(loop); bld.position_at_start(loop)
    phi = bld.phi(i32, "i"); phi.add_incoming(start, entry)
    bld.cbranch(bld.icmp_signed("<", phi, n), body, exit_bb)
    bld.position_at_start(body)
    a_val = bld.load(bld.gep(a, [phi])); b_val = bld.load(bld.gep(b, [phi]))
    bld.store(bld.fadd(a_val, b_val), bld.gep(c, [phi]))
    phi_next = bld.add(phi, stride)
    bld.branch(loop); phi.add_incoming(phi_next, body)
    bld.position_at_start(exit_bb); bld.ret_void()
    ann = """
!nvvm.annotations = !{!0}
!0 = !{ i8* bitcast (void (float*, float*, float*, i32)* @vecAdd to i8*), !"kernel", i32 1 }
@llvm.used = appending global [1 x i8*] [i8* bitcast (void (float*, float*, float*, i32)* @vecAdd to i8*)]
"""
    return str(mod) + ann

# IR→PTXコンパイル（-O3 と追加のアンローリング閾値を指定）
def comp_ptx(ir_str, cc="sm_35"):
    llvm.initialize(); llvm.initialize_native_target(); llvm.initialize_native_asmprinter()
    mod = llvm.parse_assembly(ir_str); mod.verify()
    pm = llvm.create_module_pass_manager(); pm.run(mod)
    opt_ir = str(mod)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".ll", delete=False) as f:
        f.write(opt_ir); ll_file = f.name
    ptx_file = ll_file.replace(".ll", ".ptx")
    subprocess.run(["llc", "-O3", "-march=nvptx64", f"-mcpu={cc}",
                    "-unroll-threshold=150", ll_file, "-o", ptx_file], check=True)
    with open(ptx_file, "r") as f: ptx = f.read()
    os.remove(ll_file); os.remove(ptx_file)
    return ptx.replace(".func vecAdd(", ".entry vecAdd(")

# グローバル初期化：PTXコード・カーネル・デバイスメモリプール
PTX_CODE = comp_ptx(gen_ir())
mod_cuda = cuda.module_from_buffer(PTX_CODE.encode("utf-8"))
kernel = mod_cuda.get_function("vecAdd")
mem_pool = DeviceMemoryPool()
async_bufs = None

# 非同期バッファ（リングバッファ＋イベント付き）の取得
def get_bufs(chk, dtype, nb=8):
    global async_bufs
    if async_bufs is None or async_bufs.get('chk') != chk or async_bufs.get('nb') != nb:
        bufs = []
        for _ in range(nb):
            pa = cuda.pagelocked_empty(chk, dtype)
            pb = cuda.pagelocked_empty(chk, dtype)
            pc = cuda.pagelocked_empty(chk, dtype)
            da = mem_pool.allocate(pa.nbytes)
            db = mem_pool.allocate(pb.nbytes)
            dc = mem_pool.allocate(pc.nbytes)
            s = cuda.Stream()
            bufs.append({'pa': pa, 'pb': pb, 'pc': pc, 'da': da, 'db': db, 'dc': dc, 's': s, 'ev': None})
        async_bufs = {'chk': chk, 'nb': nb, 'bufs': bufs}
    return async_bufs['bufs']

# 非同期ベクトル加算（リングバッファとイベントでパイプライン処理）
def vecAdd_async(a, b, chk=256*1024, nb=8):
    N = a.size; res = np.empty_like(a)
    bufs = get_bufs(chk, a.dtype, nb); nc = math.ceil(N / chk)
    for i in range(nc):
        idx, sz = i * chk, min(chk, N - i * chk)
        buf = bufs[i % nb]
        if i >= nb and buf['ev'] is not None:
            if not buf['ev'].query():
                buf['ev'].synchronize()
            # 結果回収：前回の該当チャンク
            prev = (i - nb) * chk; psz = min(chk, N - prev)
            res[prev:prev+psz] = buf['pc'][:psz]
            buf['ev'] = None
        buf['pa'][:sz] = a[idx:idx+sz]
        buf['pb'][:sz] = b[idx:idx+sz]
        cuda.memcpy_htod_async(buf['da'], buf['pa'], buf['s'])
        cuda.memcpy_htod_async(buf['db'], buf['pb'], buf['s'])
        grid = ((sz + 255) // 256, 1)
        kernel(buf['da'], buf['db'], buf['dc'], np.int32(sz),
               block=(256,1,1), grid=grid, stream=buf['s'])
        cuda.memcpy_dtoh_async(buf['pc'], buf['dc'], buf['s'])
        ev = cuda.Event(); ev.record(buf['s']); buf['ev'] = ev
    # 残りのチャンクの結果回収
    for i in range(max(nc - nb, 0), nc):
        buf = bufs[i % nb]
        if buf['ev'] is not None and not buf['ev'].query():
            buf['ev'].synchronize()
        idx, sz = i * chk, min(chk, N - i * chk)
        res[idx:idx+sz] = buf['pc'][:sz]
    return res

if __name__ == "__main__":
    N = 1024*1024
    a = np.random.randn(N).astype(np.float32)
    b = np.random.randn(N).astype(np.float32)
    c = vecAdd_async(a, b)
    assert np.allclose(c, a+b), "Kernel execution failed"
    iters = 100; t0 = time.perf_counter()
    for _ in range(iters):
        vecAdd_async(a, b)
    print("Optimized Async avg time: {:.2f} ms".format((time.perf_counter()-t0)/iters*1000))
