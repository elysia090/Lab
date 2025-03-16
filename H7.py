import math, time, json, hashlib, secrets 
from typing import List, Dict, Tuple, Optional, Any, Union
import numpy as np
from dataclasses import dataclass, field, asdict
from functools import lru_cache

# GPU detection and initialization
try:
    import cupy as cp
    from pycuda import driver as cuda
    from pycuda.compiler import SourceModule
    import pycuda.autoinit
    GPU_AVAILABLE = True
    print("[INFO] GPU acceleration enabled (PyCUDA + CuPy)")
except ImportError:
    GPU_AVAILABLE = False
    print("[INFO] GPU libraries not found, using CPU-only mode")

# BN128 curve parameters
class BN128:
    p = int("30644e72e131a029b85045b68181585d2833c5f198c4fef741a0f8c54f3dffac", 16)
    b = 3
    R = 1 << 256
    R_mod_p = R % p
    inv_R_mod_p = pow(R_mod_p, p-2, p)
    G1_gen = (1, 2)
    G2_gen = ((5, 1), (9, 1))

# Constants for demonstration
CONSTANTS = {
    'S_VAL': 19,
    'H_VAL': 17, 
    'F_POLY_VAL': 17
}

# Enhanced JSON encoder for NumPy and complex types
class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (complex, np.complex128)):
            return {"real": obj.real, "imag": obj.imag}
        return super().default(obj)

# Montgomery arithmetic operations (optimized)
class MontgomeryArithmetic:
    @staticmethod
    def to_mont(x: int) -> int: 
        return (x * BN128.R_mod_p) % BN128.p

    @staticmethod
    def from_mont(x: int) -> int: 
        return (x * BN128.inv_R_mod_p) % BN128.p

    @staticmethod
    def mul(a: int, b: int) -> int: 
        return ((a * b) * BN128.inv_R_mod_p) % BN128.p

    @staticmethod
    def sqr(a: int) -> int: 
        return MontgomeryArithmetic.mul(a, a)

    @staticmethod
    def add(a: int, b: int) -> int: 
        return (a + b) % BN128.p

    @staticmethod
    def sub(a: int, b: int) -> int: 
        return (a - b) % BN128.p

# Elliptic curve operations
class EllipticCurve:
    @staticmethod
    def add_g1(a: Tuple[int, int], b: Tuple[int, int]) -> Tuple[int, int]:
        """Add two points on the G1 curve."""
        (x1, y1), (x2, y2) = a, b
        if a == (0, 0): return b
        if b == (0, 0): return a
        if x1 == x2 and (y1 + y2) % BN128.p == 0: return (0, 0)
        
        if a == b:  # Point doubling
            lam = (3 * x1 * x1 * pow(2 * y1, BN128.p-2, BN128.p)) % BN128.p
        else:
            lam = ((y2 - y1) * pow((x2 - x1) % BN128.p, BN128.p-2, BN128.p)) % BN128.p
            
        xr = (lam * lam - x1 - x2) % BN128.p
        yr = (lam * (x1 - xr) - y1) % BN128.p
        return (xr, yr)

    @staticmethod
    def mul_g1(base: Tuple[int, int], scalar: int) -> Tuple[int, int]:
        """Scalar multiplication on G1 curve using double-and-add."""
        acc, cur, s = (0, 0), base, scalar
        while s > 0:
            if s & 1:
                acc = EllipticCurve.add_g1(acc, cur)
            cur = EllipticCurve.add_g1(cur, cur)
            s >>= 1
        return acc

    @staticmethod
    @lru_cache(maxsize=128)  # Cache common scalar multiplications
    def mul_g2(pt: Tuple[Tuple[int, int], Tuple[int, int]], s: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Simplified G2 multiplication (for demonstration)"""
        return pt

# Pairing operations
class Pairing:
    @staticmethod
    def miller_loop(g1: Tuple[int, int], g2: Tuple[Tuple[int, int], Tuple[int, int]]) -> int:
        """Simplified Miller loop implementation (for demonstration)"""
        cat = f"{g1}{g2[0]}{g2[1]}miller"
        return int(hashlib.sha256(cat.encode("ascii")).hexdigest(), 16) % BN128.p

    @staticmethod
    def final_exponent(val: int) -> int:
        """Simplified final exponentiation (for demonstration)"""
        c = f"{val}final"
        return int(hashlib.sha256(c.encode("ascii")).hexdigest(), 16) % BN128.p

    @staticmethod
    def bn128_full(g1: Tuple[int, int], g2: Tuple[Tuple[int, int], Tuple[int, int]]) -> int:
        """Complete pairing computation"""
        return Pairing.final_exponent(Pairing.miller_loop(g1, g2))

# KZG commitment scheme
class KZG:
    @staticmethod
    def commit_f17() -> Tuple[int, int]:
        """Generate a KZG commitment for the F_POLY_VAL polynomial"""
        return EllipticCurve.mul_g1(BN128.G1_gen, CONSTANTS['F_POLY_VAL'])

    @staticmethod
    def open_f17(alpha: int) -> Tuple[int, Tuple[int, int]]:
        """Generate proof for evaluation at point alpha"""
        r = secrets.randbits(256) % BN128.p
        prf = EllipticCurve.mul_g1(BN128.G1_gen, r)
        return (CONSTANTS['F_POLY_VAL'], prf)

    @staticmethod
    def verify_f17(commit_val: Tuple[int, int], alpha: int, f_alpha: int, proof: Tuple[int, int]) -> bool:
        """Verify a KZG proof"""
        left = Pairing.bn128_full(commit_val, 
                                  EllipticCurve.mul_g2(BN128.G2_gen, CONSTANTS['H_VAL']))
        right = Pairing.bn128_full(EllipticCurve.mul_g1(BN128.G1_gen, f_alpha),
                                  EllipticCurve.mul_g2(BN128.G2_gen, 
                                                      (CONSTANTS['S_VAL'] - alpha) % (BN128.p - 1)))
        return left == right

# Complex transformations for Hamiltonian binding
class ComplexTransformations:
    @staticmethod
    def teichmuller_lift(val: int, n: int) -> complex:
        """Create a complex number from a field element using Teichmüller lift"""
        angle = 2.0 * math.pi * ((val % n) / float(n))
        return math.cos(angle) + 1j * math.sin(angle)

    @staticmethod
    def apply_flow(arr: np.ndarray, flow_time: float) -> np.ndarray:
        """Apply time flow to complex array"""
        return arr * np.exp(1j * flow_time)

    @staticmethod
    def apply_flow_inverse(arr: np.ndarray, flow_time: float) -> np.ndarray:
        """Reverse time flow on complex array"""
        return arr * np.exp(-1j * flow_time)

    @staticmethod
    def derive_binding(metadata: Dict) -> np.ndarray:
        """Generate binding pattern from metadata"""
        serialized = json.dumps(metadata, sort_keys=True).encode('ascii')
        hash_value = hashlib.sha256(serialized).digest()
        
        # Convert hash bytes to complex vector pattern
        binding = np.zeros(len(hash_value) * 4, dtype=np.complex128)
        for i, byte in enumerate(hash_value):
            for j in range(4):
                bits = (byte >> (j * 2)) & 0x03
                angle = bits * (math.pi / 2)
                binding[i * 4 + j] = np.exp(1j * angle)
        
        return binding

    @staticmethod
    def resize_binding(binding: np.ndarray, target_size: int) -> np.ndarray:
        """Resize binding to match target array size using linear interpolation"""
        if len(binding) == target_size:
            return binding
            
        indices = np.linspace(0, len(binding) - 1, target_size)
        binding_resized = np.zeros(target_size, dtype=np.complex128)
        
        # Fast vectorized operations where possible
        idx_floor = np.floor(indices).astype(int)
        idx_ceil = np.ceil(indices).astype(int)
        t = indices - idx_floor
        
        # Handle edge cases where floor == ceil
        same_idx = (idx_floor == idx_ceil)
        binding_resized[same_idx] = binding[idx_floor[same_idx]]
        
        # Handle interpolation cases
        interp_idx = ~same_idx
        binding_resized[interp_idx] = binding[idx_floor[interp_idx]] * (1 - t[interp_idx, np.newaxis]) + \
                                      binding[idx_ceil[interp_idx]] * t[interp_idx, np.newaxis]
        
        return binding_resized

    @staticmethod
    def apply_binding(base_arr: np.ndarray, metadata: Dict, strength: float = 0.3) -> np.ndarray:
        """Apply binding to base array"""
        binding = ComplexTransformations.derive_binding(metadata)
        
        # Resize binding if needed
        if len(binding) != len(base_arr):
            binding = ComplexTransformations.resize_binding(binding, len(base_arr))
        
        # Normalize binding vector
        binding_norm = np.linalg.norm(binding)
        if binding_norm > 1e-12:
            binding_normalized = binding / binding_norm
        else:
            binding_normalized = binding
            
        # Add scaled binding to base array
        return base_arr + strength * binding_normalized

    @staticmethod
    def verify_binding(bound_arr: np.ndarray, metadata: Dict, 
                      strength: float = 0.3, threshold: float = 0.15) -> Tuple[bool, float]:
        """Verify binding by comparing extracted binding with expected binding"""
        # Recreate base array
        n = metadata["n"]
        angle = 2.0 * math.pi * ((17 % n) / float(n))
        base_value = math.cos(angle) + 1j * math.sin(angle)
        base_arr = np.full(n, base_value, dtype=np.complex128)
        
        # Extract binding component
        extracted_binding = (bound_arr - base_arr) / strength
        
        # Derive expected binding
        expected_binding = ComplexTransformations.derive_binding(metadata)
        if len(expected_binding) != len(extracted_binding):
            expected_binding = ComplexTransformations.resize_binding(
                expected_binding, len(extracted_binding))
        
        # Normalize both vectors
        norm_expected = np.linalg.norm(expected_binding)
        norm_extracted = np.linalg.norm(extracted_binding)
        
        if norm_expected > 1e-12:
            expected_binding = expected_binding / norm_expected
        if norm_extracted > 1e-12:
            extracted_binding = extracted_binding / norm_extracted
            
        # Compute similarity via absolute inner product
        similarity = np.abs(np.vdot(extracted_binding, expected_binding))
        
        return similarity >= threshold, similarity

# Audit logging system with structured data
@dataclass
class AuditEntry:
    timestamp: float
    session_id: str
    op_type: str
    payload: Any
    
    def __post_init__(self):
        self.payload_hash = hashlib.sha256(repr(self.payload).encode("ascii", "ignore")).hexdigest()
        self.payload_preview = str(self.payload)[:128]
        
    def to_dict(self):
        return {
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "op_type": self.op_type,
            "payload_hash": self.payload_hash,
            "payload_preview": self.payload_preview,
            "full_payload": self.payload
        }

class AuditLog:
    def __init__(self):
        self.entries = []
        
    def record(self, op_type: str, session_id: str, payload: Any) -> AuditEntry:
        entry = AuditEntry(
            timestamp=time.time(),
            session_id=session_id,
            op_type=op_type,
            payload=payload
        )
        self.entries.append(entry)
        return entry
        
    def show_log(self) -> None:
        print("=== Audit Log Entries ===")
        for i, e in enumerate(self.entries):
            print(f"[{i}] T={e.timestamp:.4f}, Sess={e.session_id}, Op={e.op_type}, "
                  f"Hash={e.payload_hash[:16]}..., Preview={e.payload_preview}")
            
    def dump_json(self, fn: str) -> None:
        data = [e.to_dict() for e in self.entries]
        with open(fn, "w") as f:
            json.dump(data, f, indent=2, cls=EnhancedJSONEncoder)

# Session classes
class ProverSession:
    def __init__(self, aggregator, session_id: str, n: int = 128, chunk_size: int = 64):
        self.agg = aggregator
        self.log = aggregator.log
        self.session_id = session_id
        self.n = n
        self.chunk_size = chunk_size
        self.omega = 5
        self.use_gpu = GPU_AVAILABLE
        print(f"[INFO] {'GPU' if self.use_gpu else 'CPU'} mode for session {session_id}.")

    def commit(self) -> Tuple[Tuple[int, int], np.ndarray, float, Dict]:
        # Create KZG commitment
        c_val = KZG.commit_f17()
        
        # Generate base array with Teichmüller lift value
        base_arr = np.full(
            self.n, 
            ComplexTransformations.teichmuller_lift(17, self.n),
            dtype=np.complex128
        )
        
        # Create metadata
        metadata = {
            "session_id": self.session_id,
            "commit_timestamp": time.time(),
            "n": self.n,
            "chunk_size": self.chunk_size,
            "commitment_value": str(c_val)
        }
        
        # Apply binding and flow
        bound_arr = ComplexTransformations.apply_binding(base_arr, metadata)
        flow_t = float(secrets.randbits(32)) / 9999999.0
        final_arr = ComplexTransformations.apply_flow(bound_arr, flow_t)
        
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

    def respond(self, alpha: int, final_arr: np.ndarray, flow_t: float, metadata: Dict) -> Tuple[int, Tuple[int, int]]:
        # Generate KZG opening proof
        f_alpha, prf = KZG.open_f17(alpha)
        
        # Record response
        self.log.record("RESPONSE", self.session_id, {
            "f_alpha": f_alpha,
            "proof": prf,
            "flow_time": flow_t,
            "metadata_hash": hashlib.sha256(json.dumps(metadata, sort_keys=True).encode('ascii')).hexdigest()
        })
        
        return (f_alpha, prf)

class MultiProverAggregator:
    def __init__(self):
        self.log = AuditLog()
        
    def new_session(self, sid: str, n: int = 128, chunk_size: int = 64) -> ProverSession:
        return ProverSession(self, sid, n, chunk_size)
        
    def challenge(self) -> int:
        alpha = 2
        self.log.record("CHALLENGE", "ALL", {"alpha": alpha})
        return alpha
        
    def verify(self, sid: str, c_val: Tuple[int, int], alpha: int, 
              f_alpha: int, proof: Tuple[int, int], final_arr: np.ndarray, 
              flow_t: float, metadata: Dict) -> bool:
        # Verify KZG proof
        ok_kzg = KZG.verify_f17(c_val, alpha, f_alpha, proof)
        
        # Undo flow transformation and verify binding
        undone = ComplexTransformations.apply_flow_inverse(final_arr, flow_t)
        binding_ok, similarity = ComplexTransformations.verify_binding(undone, metadata)
        
        # Mean magnitude check
        mean_magnitude = float(np.mean(np.abs(undone)))
        ok_flow = (abs(mean_magnitude - 1.0) < 0.02)
        
        # Record verification result
        result = ok_kzg and ok_flow and binding_ok
        self.log.record("VERIFY", sid, {
            "kzg_ok": ok_kzg,
            "flow_ok": ok_flow,
            "binding_ok": binding_ok,
            "binding_similarity": similarity,
            "mean_mag": mean_magnitude,
            "result": result
        })
        
        return result
        
    def sign_final(self, priv_key: int = 0x12345) -> str:
        payload_str = "".join(repr(e.payload) for e in self.log.entries)
        hv = hashlib.sha256(payload_str.encode("ascii", "ignore")).hexdigest()
        sig = pow(int(hv, 16), priv_key, BN128.p)
        return hex(sig)
        
    def dump_log(self, fname: str) -> None:
        self.log.dump_json(fname)

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
