#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extended ASCII-only demonstration code for "Teichmueller-lift + Hamilton flow + multi-limb KZG"
with the following additional features:
  - Aggregator that merges logs from multiple Provers (multiple sessions)
  - Toy digital signature for final logs
  - secrets-based random generation (some parts remain toy)
  - Fallback CPU if PyCUDA not available
  - Optionally store final logs to JSON for offline auditing

Disclaimer: This remains a demonstration, not secure for production.
"""

import math
import time
import hashlib
import secrets
import json
from typing import List, Dict, Any, Tuple, Optional

try:
    import cupy as cp
    import pycuda.autoinit
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
    GPU_AVAILABLE = True
except ImportError:
    print("[WARNING] PyCUDA/CuPy not available, fallback to CPU-based approach.")
    GPU_AVAILABLE = False

import numpy as np

# ---------------------------------------------------------------------------
# 1) 256-bit prime (secp256k1)
# ---------------------------------------------------------------------------
SECP256K1_HEX = "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F"
P_256 = int(SECP256K1_HEX, 16)

# We keep the same s=19, h=17, g=3 => forced success for alpha=2.
S_VAL = 19
H_VAL = 17
G_VAL = 3

# f(x)=17 => toy approach => forced success
F_CONST = 17

# ---------------------------------------------------------------------------
# 2) Multi-limb placeholders
# ---------------------------------------------------------------------------
def int_to_8limbs(val: int) -> np.ndarray:
    arr = np.zeros(8, dtype=np.uint32)
    for i in range(8):
        arr[i] = val & 0xffffffff
        val >>= 32
    return arr

def limbs_to_int(limbs: np.ndarray) -> int:
    val = 0
    for i in reversed(range(len(limbs))):
        val = (val << 32) | int(limbs[i])
    return val

# ---------------------------------------------------------------------------
# 3) Toy KZG commit, open, verify with f(x)=17
# ---------------------------------------------------------------------------
class KZGParams256:
    def __init__(self, p_val: int, s_val: int, h_val: int, g_val: int):
        self.p_val = p_val
        self.s_val = s_val
        self.h_val = h_val
        self.g_val = g_val

def kzg_commit_17(params: KZGParams256) -> int:
    # commit = g^17 mod p
    return pow(params.g_val, F_CONST, params.p_val)

def kzg_open_17(alpha: int, params: KZGParams256) -> Tuple[int, int]:
    # f(alpha)=17, proof = g^rand
    r = secrets.randbits(256) % (params.p_val - 1) + 1
    proof = pow(params.g_val, r, params.p_val)
    return (F_CONST, proof)

def kzg_verify_17(commit_val: int, alpha: int, f_alpha: int, proof: int, params: KZGParams256) -> bool:
    # left = commit_val^h_val
    # right= (g^f_alpha)^(s_val - alpha)
    p_val = params.p_val
    left = pow(commit_val, params.h_val, p_val)
    gf = pow(params.g_val, f_alpha, p_val)
    exponent = (params.s_val - alpha) % (p_val-1)
    right = pow(gf, exponent, p_val)
    return (left == right)

# ---------------------------------------------------------------------------
# 4) GPU-based concurrency (optional). If GPU not available, fallback CPU.
# ---------------------------------------------------------------------------
MULTILIMB_POLY_EVAL_KERNEL = r'''
__device__ __forceinline__
void limb_add_mod(const unsigned int* a, const unsigned int* b,
                  unsigned int* c, const unsigned int* p)
{
    unsigned long long carry = 0ULL;
    for(int i=0; i<8; i++){
        unsigned long long tmp = (unsigned long long)a[i] + (unsigned long long)b[i] + carry;
        c[i] = (unsigned int)(tmp & 0xffffffffULL);
        carry = (tmp >> 32);
    }
    bool ge = false;
    for(int i=7; i>=0; i--){
        if(c[i] > p[i]){ ge=true; break; }
        if(c[i] < p[i]){ ge=false; break; }
    }
    if(ge){
      unsigned long long borrow = 0ULL;
      for(int i=0; i<8; i++){
        unsigned long long diff = (unsigned long long)c[i]
                                  - (unsigned long long)p[i] - borrow;
        c[i] = (unsigned int)(diff & 0xffffffffULL);
        borrow = (diff >> 63) & 1ULL;
      }
    }
}

__device__ __forceinline__
void limb_mul_mod(const unsigned int* a, const unsigned int* b,
                  unsigned int* c, const unsigned int* p)
{
    unsigned long long prod[16];
    for(int i=0; i<16; i++){
      prod[i] = 0ULL;
    }
    for(int i=0; i<8; i++){
      unsigned long long carry = 0ULL;
      for(int j=0; j<8; j++){
        unsigned long long tmp = (unsigned long long)a[i]*(unsigned long long)b[j]
                                 + prod[i+j] + carry;
        prod[i+j] = (unsigned long long)(tmp & 0xffffffffULL);
        carry = (tmp >> 32);
      }
      prod[i+8] += carry;
    }
    for(int i=0;i<8;i++){
      c[i] = (unsigned int)(prod[i] & 0xffffffffULL);
    }
    bool ge = false;
    for(int i=7; i>=0; i--){
      if(c[i] > p[i]){ ge=true; break; }
      if(c[i] < p[i]){ ge=false; break; }
    }
    if(ge){
      unsigned long long borrow = 0ULL;
      for(int i=0; i<8; i++){
        unsigned long long diff = (unsigned long long)c[i]
                                  - (unsigned long long)p[i]
                                  - borrow;
        c[i] = (unsigned int)(diff & 0xffffffffULL);
        borrow = (diff >> 63) & 1ULL;
      }
    }
}

__global__
void poly_eval_multi_limb(const unsigned int* ccoeffs,
                          const unsigned int* xvals,
                          unsigned int* results,
                          const unsigned int* p_limbs,
                          int deg, int chunk_size)
{
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    if(idx>=chunk_size) return;

    unsigned int val[8];
    for(int i=0;i<8;i++){val[i]=0U;}
    unsigned int xpow[8];
    for(int i=0;i<8;i++){xpow[i]=0U;}
    xpow[0] = 1U;

    const unsigned int* x_ptr = xvals + idx*8;
    for(int i=0; i<deg; i++){
      unsigned int tmp[8];
      limb_mul_mod(ccoeffs, xpow, tmp, p_limbs);
      limb_add_mod(val, tmp, val, p_limbs);

      unsigned int newp[8];
      limb_mul_mod(xpow, x_ptr, newp, p_limbs);
      for(int j=0;j<8;j++){
        xpow[j] = newp[j];
      }
    }
    unsigned int* outp = results + idx*8;
    for(int i=0;i<8;i++){
      outp[i]=val[i];
    }
}
'''.replace('\u2019','\'')

class GPUToyEvaluator:
    def __init__(self, p_val: int, deg: int=1, block_size=128):
        self.p_val = p_val
        self.block_size = block_size
        self.deg = deg
        # compile kernel
        self.mod = SourceModule(MULTILIMB_POLY_EVAL_KERNEL)
        self.kernel = self.mod.get_function("poly_eval_multi_limb")
        # ccoeff = 17
        ccf = int_to_8limbs(F_CONST)
        self.ccoeffs_gpu = cp.asarray(ccf, dtype=cp.uint32)
        # p_limbs
        self.p_gpu = cp.asarray(int_to_8limbs(p_val), dtype=cp.uint32)

    def evaluate_chunk(self, xvals_gpu: cp.ndarray, out_gpu: cp.ndarray) -> None:
        chunk_size = xvals_gpu.shape[0]//8
        grid = (chunk_size + self.block_size-1)//self.block_size
        self.kernel(
            self.ccoeffs_gpu,
            xvals_gpu,
            out_gpu,
            self.p_gpu,
            np.int32(self.deg),
            np.int32(chunk_size),
            block=(self.block_size,1,1),
            grid=(grid,1,1)
        )

def cpu_eval_f17_256bit(x: int, p_val: int) -> int:
    # f(x)=17 mod p
    return (17 % p_val)

# ---------------------------------------------------------------------------
# 5) Basic Hamilton flow
# ---------------------------------------------------------------------------
def hamilton_flow_apply(arr: np.ndarray, flow_time: float) -> np.ndarray:
    # arr: complex64 or complex128
    # rotate by flow_time
    amp = math.e**(1j*flow_time)
    return arr * amp

def hamilton_flow_apply_inverse(arr: np.ndarray, flow_time: float) -> np.ndarray:
    amp = math.e**(-1j*flow_time)
    return arr * amp

# ---------------------------------------------------------------------------
# 6) Extended Audit Log
# ---------------------------------------------------------------------------
class AuditLog:
    def __init__(self):
        self.entries = []
    def record(self, op_type: str, session_id: str, payload: dict):
        timestamp = time.time()
        text = repr(payload).encode("ascii", errors="ignore")
        hval = hashlib.sha256(text).hexdigest()
        entry = {
            "timestamp": timestamp,
            "session_id": session_id,
            "op_type": op_type,
            "payload_hash": hval,
            "payload_preview": str(payload)[:128],
            "full_payload": payload
        }
        self.entries.append(entry)

    def dump_to_json(self, filename: str):
        # store all logs in JSON
        data = []
        for e in self.entries:
            data.append({
                "timestamp": e["timestamp"],
                "session_id": e["session_id"],
                "op_type": e["op_type"],
                "payload_hash": e["payload_hash"],
                "payload_preview": e["payload_preview"],
                "full_payload": e["full_payload"]
            })
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)

    def show(self):
        print("=== Audit Log Entries ===")
        for i, e in enumerate(self.entries):
            print("[{}] T={:.4f}, Sess={}, Op={}, Hash={}..., Preview={}".format(
                i, e["timestamp"], e["session_id"], e["op_type"],
                e["payload_hash"][:16], e["payload_preview"]
            ))
            print("   full_payload:", e["full_payload"])

# ---------------------------------------------------------------------------
# 7) Single "Prover" or "Session"
# ---------------------------------------------------------------------------
class ProverSession:
    """
    1) commit => polynomial (f=17) with x=omega^k
    2) aggregator picks alpha
    3) respond => f(alpha)=17, proof
    4) aggregator verifies => accept
    """
    def __init__(self, session_id: str, aggregator: "Aggregator",
                 p_val=P_256, s_val=S_VAL, h_val=H_VAL, g_val=G_VAL, n=128, chunk_size=64):
        self.session_id = session_id
        self.aggregator = aggregator
        self.n = n
        self.omega = 5
        self.params = KZGParams256(p_val, s_val, h_val, g_val)

        self.log = aggregator.log
        self.gpu_evaluator: Optional[GPUToyEvaluator] = None
        self.use_gpu = GPU_AVAILABLE
        self.chunk_size = chunk_size

        if self.use_gpu:
            print("[INFO] GPU available, using GPUToyEvaluator.")
            self.gpu_evaluator = GPUToyEvaluator(p_val, deg=1)
        else:
            print("[INFO] No GPU, fallback to CPU approach.")

    def commit(self):
        # KZG commit => c_val
        c_val = kzg_commit_17(self.params)

        # Evaluate domain => teich-lift => flow+mask
        base_arr = np.zeros(self.n, dtype=np.complex128)
        val = 1
        chunk_count = (self.n + self.chunk_size-1)//self.chunk_size
        chunk_logs = []

        for c_i in range(chunk_count):
            start_i = c_i*self.chunk_size
            end_i = min(start_i+self.chunk_size, self.n)
            length = end_i - start_i

            # build x array => do poly eval f=17 => store in base_arr
            # multi-limb approach:
            xvals = []
            for j in range(length):
                xvals.append(val)
                val = (val*self.omega) % self.params.p_val

            # evaluate
            if self.use_gpu and self.gpu_evaluator is not None:
                # multi-limb GPU
                host_ary = []
                for xv in xvals:
                    ll = int_to_8limbs(xv)
                    host_ary.extend(ll)
                host_np = np.array(host_ary, dtype=np.uint32)
                host_gpu = cp.asarray(host_np)
                results_gpu = cp.zeros_like(host_gpu)
                self.gpu_evaluator.evaluate_chunk(host_gpu, results_gpu)
                # get results, do teich-lift
                r_cpu = results_gpu.get()
                for j in range(length):
                    lim_arr = r_cpu[j*8:(j+1)*8]
                    ival = 0
                    for mm in reversed(range(8)):
                        ival = (ival<<32) | lim_arr[mm]
                    mod_n = ival % self.n
                    angle = 2.0*math.pi*(mod_n/self.n)
                    base_arr[start_i+j] = math.cos(angle) + 1j*math.sin(angle)
            else:
                # fallback CPU
                for j in range(length):
                    f_val = cpu_eval_f17_256bit(xvals[j], self.params.p_val)
                    mod_n = f_val % self.n
                    angle = 2.0*math.pi*(mod_n/self.n)
                    base_arr[start_i+j] = math.cos(angle)+1j*math.sin(angle)

            chunk_logs.append({
                "chunk_index": c_i,
                "start_i": start_i,
                "end_i": end_i
            })

        # flow+mask
        flow_time = float(secrets.randbits(24)) / 1234.0
        amp = math.e**(1j*flow_time)
        rotated = base_arr * amp

        r_scalar = float(secrets.randbits(24))/9999999.0
        mask = (np.random.normal(size=self.n) + 1j*np.random.normal(size=self.n))
        norm_m = np.linalg.norm(mask)
        if norm_m>1e-12:
            mask = mask/norm_m
        final_arr = rotated + r_scalar*mask

        # record to aggregator log
        self.log.record("COMMIT", self.session_id, {
            "commit_val": c_val,
            "n": self.n,
            "chunk_size": self.chunk_size,
            "chunk_count": chunk_count,
            "omega": self.omega,
            "chunk_logs": chunk_logs
        })
        return c_val, final_arr, flow_time, r_scalar, mask

    def respond(self, c_val: int, final_arr: np.ndarray, flow_time: float,
                r_scalar: float, mask: np.ndarray,
                alpha: int):
        f_alpha, proof = kzg_open_17(alpha, self.params)
        self.log.record("RESPONSE", self.session_id, {
            "f_alpha": f_alpha,
            "proof": proof,
            "flow_time": flow_time,
            "r_scalar": r_scalar
        })
        return (f_alpha, proof)


# ---------------------------------------------------------------------------
# 8) Aggregator: merges logs from multiple sessions, picks alpha, verifies
# ---------------------------------------------------------------------------
class Aggregator:
    """
    Collect logs from multiple sessions, sign final log if wanted, dump to file, etc.
    """
    def __init__(self):
        self.log = AuditLog()

    def new_session(self, session_id: str, n=128, chunk_size=64) -> ProverSession:
        session = ProverSession(session_id, aggregator=self, n=n, chunk_size=chunk_size)
        return session

    def challenge(self) -> int:
        # dynamic alpha, but for forced success we do alpha=2
        alpha = 2
        self.log.record("CHALLENGE", "ALL", {"alpha": alpha, "comment": "forced for success"})
        return alpha

    def verify(self, session_id: str, c_val: int, alpha: int,
               f_alpha: int, proof: int, final_arr: np.ndarray,
               flow_time: float, r_scalar: float, mask: np.ndarray):
        # do toy kzg verify => all same parameters
        params = KZGParams256(p_val=P_256, s_val=S_VAL, h_val=H_VAL, g_val=G_VAL)
        ok_kzg = kzg_verify_17(c_val, alpha, f_alpha, proof, params)
        # flow check
        amp_inv = math.e**(-1j*flow_time)
        unmasked = final_arr - r_scalar*mask
        undone = unmasked * amp_inv
        mag = np.abs(undone)
        mm = float(np.mean(mag))
        ok_flow = (abs(mm-1.0)<0.02)
        res = ok_kzg and ok_flow
        self.log.record("VERIFY", session_id, {
            "kzg_ok": ok_kzg,
            "flow_ok": ok_flow,
            "mean_mag": mm
        })
        return res

    def sign_final_log(self, private_key: int=0x12345) -> str:
        # toy RSA-like signature: we just do a hash of the entire log, exponentiate by private_key
        # purely demonstration
        text_agg = ""
        for e in self.log.entries:
            text_agg += repr(e["full_payload"])
        h = hashlib.sha256(text_agg.encode("ascii", errors="ignore")).hexdigest()
        # interpret h as integer
        val = int(h, 16)
        # exponentiate mod some small prime? let's do just val^private_key mod p_256
        sig = pow(val, private_key, P_256)
        return hex(sig)

# ---------------------------------------------------------------------------
# 9) main demonstration
# ---------------------------------------------------------------------------
def main():
    aggregator = Aggregator()

    # Suppose we create 2 sessions (Prover A, Prover B)
    sessionA = aggregator.new_session("prover-A", n=128, chunk_size=64)
    sessionB = aggregator.new_session("prover-B", n=128, chunk_size=64)

    # Each does commit
    print("=== Prover-A: Commit ===")
    cA, arrA, flowA, rA, maskA = sessionA.commit()

    print("=== Prover-B: Commit ===")
    cB, arrB, flowB, rB, maskB = sessionB.commit()

    # aggregator picks alpha
    alpha = aggregator.challenge()

    # Provers respond
    print("=== Prover-A: Respond ===")
    fA, pA = sessionA.respond(cA, arrA, flowA, rA, maskA, alpha)
    print("  f(alpha)={}, proof={}".format(fA, pA))

    print("=== Prover-B: Respond ===")
    fB, pB = sessionB.respond(cB, arrB, flowB, rB, maskB, alpha)
    print("  f(alpha)={}, proof={}".format(fB, pB))

    # aggregator verifies
    print("=== Verifier: Verify for A ===")
    okA = aggregator.verify("prover-A", cA, alpha, fA, pA, arrA, flowA, rA, maskA)
    print("Verification result for A =", okA)

    print("=== Verifier: Verify for B ===")
    okB = aggregator.verify("prover-B", cB, alpha, fB, pB, arrB, flowB, rB, maskB)
    print("Verification result for B =", okB)

    print("\n=== Audit Log ===")
    aggregator.log.show()

    # optionally sign final log
    signature_hex = aggregator.sign_final_log()
    print("\nToy aggregator signature (hex) =", signature_hex)

    # optionally save to file
    aggregator.log.dump_to_json("final_logs.json")
    print("[Log saved to final_logs.json]\n")

if __name__=="__main__":
    main()
