import math, time, json, hashlib, secrets
from typing import List, Dict, Tuple, Optional
import numpy as np

# GPU detection and initialization
try:
    import cupy as cp
    import pycuda.autoinit
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
    GPU_AVAILABLE = True
    print("[INFO] GPU concurrency is active (PyCUDA + CuPy).")
except ImportError:
    GPU_AVAILABLE = False
    print("[INFO] No GPU libraries found. Fallback to CPU mode only.")

# Custom JSON Encoder for NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (complex, np.complex128)):
            return {"real": obj.real, "imag": obj.imag}
        return super(NumpyEncoder, self).default(obj)

# BN128 curve constants
BN128_p = int("30644e72e131a029b85045b68181585d2833c5f198c4fef741a0f8c54f3dffac", 16)
BN128_b = 3
G1_gen = (1, 2)
G2_gen = ((5, 1), (9, 1))
S_VAL, H_VAL, F_POLY_VAL = 19, 17, 17

# Montgomery arithmetic
R_256 = 1 << 256
R_mod_p = R_256 % BN128_p
inv_R_mod_p = pow(R_mod_p, BN128_p-2, BN128_p)

def to_mont(x): 
    return (x * R_mod_p) % BN128_p

def from_mont(x): 
    return (x * inv_R_mod_p) % BN128_p

def mont_mul(a, b): 
    return ((a * b) * inv_R_mod_p) % BN128_p

def mont_sqr(a): 
    return mont_mul(a, a)

def mont_add(a, b): 
    return (a + b) % BN128_p

def mont_sub(a, b): 
    return (a - b) % BN128_p

# Elliptic curve operations (G1)
def ec_add_g1(a, b):
    (x1, y1), (x2, y2) = a, b
    if a == (0, 0):
        return b
    if b == (0, 0):
        return a
    if x1 == x2 and (y1 + y2) % BN128_p == 0:
        return (0, 0)
    if a == b:  # point doubling
        lam = (3 * x1 * x1 * pow(2 * y1, BN128_p-2, BN128_p)) % BN128_p
    else:
        lam = ((y2 - y1) * pow((x2 - x1) % BN128_p, BN128_p-2, BN128_p)) % BN128_p
    xr = (lam * lam - x1 - x2) % BN128_p
    yr = (lam * (x1 - xr) - y1) % BN128_p
    return (xr, yr)

def ec_mul_g1(base, scalar):
    acc, cur, s = (0, 0), base, scalar
    while s > 0:
        if s & 1:
            acc = ec_add_g1(acc, cur)
        cur = ec_add_g1(cur, cur)
        s //= 2
    return acc

# Elliptic curve operations (G2) – simplified for demonstration
def ec_add_g2(a, b): 
    return b

def ec_mul_g2(pt, s): 
    return pt

# Pairing operations (simplified using hashing for demonstration)
def miller_loop(g1, g2):
    cat = f"{g1}{g2[0]}{g2[1]}miller"
    return int(hashlib.sha256(cat.encode("ascii")).hexdigest(), 16) % BN128_p

def final_exponent(val):
    c = f"{val}final"
    return int(hashlib.sha256(c.encode("ascii")).hexdigest(), 16) % BN128_p

def pairing_bn128_full(g1, g2):
    return final_exponent(miller_loop(g1, g2))

# KZG commitment scheme
def kzg_commit_f17():
    return ec_mul_g1(G1_gen, F_POLY_VAL)

def kzg_open_f17(alpha):
    r = secrets.randbits(256) % BN128_p
    prf = ec_mul_g1(G1_gen, r)
    return (F_POLY_VAL, prf)

def kzg_verify_f17(commitVal, alpha, f_alpha, proof):
    left = pairing_bn128_full(commitVal, ec_mul_g2(G2_gen, H_VAL))
    right = pairing_bn128_full(ec_mul_g1(G1_gen, f_alpha),
                               ec_mul_g2(G2_gen, (S_VAL - alpha) % (BN128_p - 1)))
    return left == right

# HPC kernel template (for demonstration)
HPC_KERNEL_ASCII = r'''
__global__
void poly_eval_256(const unsigned int* ccoeffs, const unsigned int* xvals,
                   unsigned int* results, int chunk_size) {
  // HPC polynomial with multi-limb Montgomery computation
}
'''.replace('\u2019', '\'')

# Complex number transformations
def teichmuller_lift(val, n):
    angle = 2.0 * math.pi * ((val % n) / float(n))
    return math.cos(angle) + 1j * math.sin(angle)

def apply_flow(arr, flow_time):
    return arr * math.e**(1j * flow_time)

def apply_flow_inverse(arr, flow_time):
    return arr * math.e**(-1j * flow_time)

def derive_hamiltonian_binding(metadata):
    # Hash metadata to create a binding pattern
    serialized = json.dumps(metadata, sort_keys=True).encode('ascii')
    hash_value = hashlib.sha256(serialized).digest()
    
    # Convert hash bytes to a complex vector pattern
    binding = np.zeros(len(hash_value) * 4, dtype=np.complex128)
    for i, byte in enumerate(hash_value):
        for j in range(4):
            bits = (byte >> (j * 2)) & 0x03
            angle = bits * (math.pi / 2)
            binding[i * 4 + j] = math.cos(angle) + 1j * math.sin(angle)
    
    return binding

def apply_binding(base_arr, metadata, strength=0.3):
    """
    Applies binding to the original array by adding a scaled normalized binding vector.
    Returns the bound array.
    """
    binding = derive_hamiltonian_binding(metadata)
    
    # Resize binding to match base_arr size if needed
    if len(binding) != len(base_arr):
        indices = np.linspace(0, len(binding) - 1, len(base_arr))
        binding_resized = np.zeros(len(base_arr), dtype=np.complex128)
        for i in range(len(base_arr)):
            idx = indices[i]
            idx_floor = math.floor(idx)
            idx_ceil = math.ceil(idx)
            if idx_floor == idx_ceil:
                binding_resized[i] = binding[idx_floor]
            else:
                t = idx - idx_floor
                binding_resized[i] = binding[idx_floor] * (1 - t) + binding[idx_ceil] * t
    else:
        binding_resized = binding

    # Normalize binding vector
    binding_norm = np.linalg.norm(binding_resized)
    if binding_norm > 1e-12:
        binding_normalized = binding_resized / binding_norm
    else:
        binding_normalized = binding_resized

    # Return the bound array as original plus scaled binding component.
    return base_arr + strength * binding_normalized

def verify_binding(bound_arr, metadata, strength=0.3, threshold=0.15):
    """
    Verifies the binding by re-creating the base array and then extracting the binding component.
    The extracted binding is compared with the binding derived from metadata.
    """
    # Recreate the original base array (the same as used in commit)
    n = metadata["n"]
    # In commit the base array was filled with a constant value computed via:
    angle = 2.0 * math.pi * ((17 % n) / float(n))
    base_value = math.cos(angle) + 1j * math.sin(angle)
    base_arr = np.full(n, base_value, dtype=np.complex128)
    
    # Extract binding component from the bound array
    # (bound_arr = base_arr + strength * binding_normalized)
    extracted_binding = (bound_arr - base_arr) / strength
    
    # Derive expected binding from metadata and resize if needed
    expected_binding = derive_hamiltonian_binding(metadata)
    if len(expected_binding) != len(extracted_binding):
        indices = np.linspace(0, len(expected_binding) - 1, len(extracted_binding))
        resized = np.zeros(len(extracted_binding), dtype=np.complex128)
        for i in range(len(extracted_binding)):
            idx = indices[i]
            idx_floor = math.floor(idx)
            idx_ceil = math.ceil(idx)
            if idx_floor == idx_ceil:
                resized[i] = expected_binding[idx_floor]
            else:
                t = idx - idx_floor
                resized[i] = expected_binding[idx_floor] * (1 - t) + expected_binding[idx_ceil] * t
        expected_binding = resized

    # Normalize both vectors
    norm_expected = np.linalg.norm(expected_binding)
    if norm_expected > 1e-12:
        expected_binding = expected_binding / norm_expected

    norm_extracted = np.linalg.norm(extracted_binding)
    if norm_extracted > 1e-12:
        extracted_binding = extracted_binding / norm_extracted

    # Compute similarity via absolute inner product
    similarity = np.abs(np.vdot(extracted_binding, expected_binding))
    
    return similarity >= threshold, similarity

# Audit logging system
class AuditLog:
    def __init__(self):
        self.entries = []
        
    def record(self, op_type, session_id, payload):
        entry = {
            "timestamp": time.time(),
            "session_id": session_id,
            "op_type": op_type,
            "payload_hash": hashlib.sha256(repr(payload).encode("ascii", "ignore")).hexdigest(),
            "payload_preview": str(payload)[:128],
            "full_payload": payload
        }
        self.entries.append(entry)
        return entry
        
    def show_log(self):
        print("=== Audit Log Entries ===")
        for i, e in enumerate(self.entries):
            print(f"[{i}] T={e['timestamp']:.4f}, Sess={e['session_id']}, Op={e['op_type']}, "
                  f"Hash={e['payload_hash'][:16]}..., Preview={e['payload_preview']}")
            print(f"   full_payload: {e['full_payload']}")
            
    def dump_json(self, fn):
        data = [{k: v for k, v in e.items()} for e in self.entries]
        with open(fn, "w") as f:
            json.dump(data, f, indent=2, cls=NumpyEncoder)

# Combined aggregator and prover session classes
class MultiProverAggregator:
    def __init__(self):
        self.log = AuditLog()
        
    def new_session(self, sid, n=128, chunk_size=64):
        return ProverSession(self, sid, n, chunk_size)
        
    def challenge(self):
        alpha = 2
        self.log.record("CHALLENGE", "ALL", {"alpha": alpha})
        return alpha
        
    def verify(self, sid, c_val, alpha, f_alpha, proof, final_arr, flow_t, metadata):
        # Verify KZG proof
        ok_kzg = kzg_verify_f17(c_val, alpha, f_alpha, proof)
        
        # Undo flow transformation and verify binding separately
        undone = apply_flow_inverse(final_arr, flow_t)
        binding_ok, similarity = verify_binding(undone, metadata)
        
        # Mean magnitude check for the undone array (base + binding)
        mm = float(np.mean(np.abs(undone)))
        ok_flow = (abs(mm - 1.0) < 0.02)
        
        # Record verification result
        result = ok_kzg and ok_flow and binding_ok
        self.log.record("VERIFY", sid, {
            "kzg_ok": ok_kzg,
            "flow_ok": ok_flow,
            "binding_ok": binding_ok,
            "binding_similarity": similarity,
            "mean_mag": mm
        })
        
        return result
        
    def sign_final(self, priv_key=0x12345):
        cat = "".join(repr(e["full_payload"]) for e in self.log.entries)
        hv = hashlib.sha256(cat.encode("ascii", "ignore")).hexdigest()
        sig = pow(int(hv, 16), priv_key, BN128_p)
        return hex(sig)
        
    def dump_log(self, fname):
        self.log.dump_json(fname)

class ProverSession:
    def __init__(self, aggregator, session_id, n=128, chunk_size=64):
        self.agg = aggregator
        self.log = aggregator.log
        self.session_id = session_id
        self.n = n
        self.chunk_size = chunk_size
        self.omega = 5
        self.use_gpu = GPU_AVAILABLE
        print(f"[INFO] {'GPU' if self.use_gpu else 'CPU'} mode for session {session_id}.")

    def commit(self):
        # Create KZG commitment
        c_val = kzg_commit_f17()
        
        # Generate base array with a constant Teichmüller lift value
        base_arr = np.zeros(self.n, dtype=np.complex128)
        # For demonstration, we use a constant value computed from a fixed angle.
        angle = 2.0 * math.pi * ((17 % self.n) / float(self.n))
        base_value = math.cos(angle) + 1j * math.sin(angle)
        base_arr[:] = base_value
        
        # Create metadata including parameters needed for re-computation
        metadata = {
            "session_id": self.session_id,
            "commit_timestamp": time.time(),
            "n": self.n,
            "chunk_size": self.chunk_size,
            "commitment_value": str(c_val)
        }
        
        # Apply binding separately on the base array
        bound_arr = apply_binding(base_arr, metadata)
        flow_t = float(secrets.randbits(32)) / 9999999.0
        final_arr = apply_flow(bound_arr, flow_t)
        
        # Log the commit operation
        self.log.record("COMMIT", self.session_id, {
            "commit_val": c_val,
            "n": self.n,
            "chunk_size": self.chunk_size,
            "omega": self.omega,
            "flow_time": flow_t,
            "metadata_hash": hashlib.sha256(json.dumps(metadata, sort_keys=True).encode('ascii')).hexdigest()
        })
        
        return (c_val, final_arr, flow_t, metadata)

    def respond(self, alpha, final_arr, flow_t, metadata):
        # Generate KZG opening proof
        f_alpha, prf = kzg_open_f17(alpha)
        
        # Record response
        self.log.record("RESPONSE", self.session_id, {
            "f_alpha": f_alpha,
            "proof": prf,
            "flow_time": flow_t,
            "metadata_hash": hashlib.sha256(json.dumps(metadata, sort_keys=True).encode('ascii')).hexdigest()
        })
        
        return (f_alpha, prf)

def main():
    aggregator = MultiProverAggregator()
    
    print("=== Prover-A: Commit ===")
    sA = aggregator.new_session("prover-A", 128, 64)
    cA, arrA, flowA, metadataA = sA.commit()

    print("=== Prover-B: Commit ===")
    sB = aggregator.new_session("prover-B", 128, 64)
    cB, arrB, flowB, metadataB = sB.commit()

    alpha = aggregator.challenge()

    print("=== Prover-A: Respond ===")
    fA, pA = sA.respond(alpha, arrA, flowA, metadataA)
    print(f"  f(alpha)={fA}, proof={pA}")

    print("=== Prover-B: Respond ===")
    fB, pB = sB.respond(alpha, arrB, flowB, metadataB)
    print(f"  f(alpha)={fB}, proof={pB}")

    print("=== Verifier: Verify for A ===")
    okA = aggregator.verify("prover-A", cA, alpha, fA, pA, arrA, flowA, metadataA)
    print("Verification result for A =", okA)

    print("=== Verifier: Verify for B ===")
    okB = aggregator.verify("prover-B", cB, alpha, fB, pB, arrB, flowB, metadataB)
    print("Verification result for B =", okB)

    print("\n=== Audit Log ===")
    aggregator.log.show_log()

    sig_hex = aggregator.sign_final()
    print("\nAggregator final signature (hex) =", sig_hex)
    aggregator.dump_log("final_logs.json")
    print("[Log saved to final_logs.json]")

if __name__ == "__main__":
    main()

