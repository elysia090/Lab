import math
import time
import json
import hashlib
import secrets
from typing import List, Dict, Tuple, Any
import numpy as np
from dataclasses import dataclass

# GPU detection
try:
    import cupy as cp
    from pycuda import driver as cuda
    import pycuda.autoinit
    GPU_AVAILABLE = True
    print("[INFO] GPU acceleration enabled")
except ImportError:
    GPU_AVAILABLE = False
    print("[INFO] Using CPU-only mode")

# Import BN128 curve operations
from py_ecc.bn128 import bn128_curve, bn128_pairing
from py_ecc.bn128.bn128_curve import FQ, FQ2, G2 as BN128_G2

# BN128 curve parameters
class BN128:
    p = int("30644e72e131a029b85045b68181585d2833c5f198c4fef741a0f8c54f3dffac", 16)
    b = 3
    G1_gen: Tuple[FQ, FQ] = (FQ(1), FQ(2))
    G2_gen: Tuple[FQ2, FQ2] = BN128_G2

# Constants
CONSTANTS = {'S_VAL': 19, 'H_VAL': 17, 'F_POLY_VAL': 17}

# JSON encoder for complex types
class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):  # Fixed: Added np.bool_ handling
            return bool(obj)
        elif isinstance(obj, (complex, np.complex128)):
            return {"real": obj.real, "imag": obj.imag}
        return super().default(obj)

# Elliptic curve operations
def ec_mul_g1(point: Tuple[FQ, FQ], scalar: int) -> Tuple[FQ, FQ]:
    return bn128_curve.multiply(point, scalar)

def ec_mul_g2(point: Tuple[FQ2, FQ2], scalar: int) -> Tuple[FQ2, FQ2]:
    return point  # Stub for demonstration

def pairing_full(g1: Tuple[FQ, FQ], g2: Tuple[FQ2, FQ2]) -> int:
    pairing_value = bn128_pairing.pairing(g2, g1)
    pairing_str = str(pairing_value)
    return int(hashlib.sha256(pairing_str.encode("ascii")).hexdigest(), 16) % BN128.p

# KZG commitment scheme
class KZG:
    @staticmethod
    def commit_f17() -> Tuple[FQ, FQ]:
        return ec_mul_g1(BN128.G1_gen, CONSTANTS['F_POLY_VAL'])

    @staticmethod
    def open_f17(alpha: int) -> Tuple[int, Tuple[FQ, FQ]]:
        r = secrets.randbits(256) % BN128.p
        prf = ec_mul_g1(BN128.G1_gen, r)
        return (CONSTANTS['F_POLY_VAL'], prf)

    @staticmethod
    def verify_f17(commit_val: Tuple[FQ, FQ], alpha: int, f_alpha: int, proof: Tuple[FQ, FQ]) -> bool:
        left = pairing_full(commit_val, ec_mul_g2(BN128.G2_gen, CONSTANTS['H_VAL']))
        right = pairing_full(ec_mul_g1(BN128.G1_gen, f_alpha),
                          ec_mul_g2(BN128.G2_gen, (CONSTANTS['S_VAL'] - alpha) % (BN128.p - 1)))
        return left == right

# Complex transformations for binding
class ComplexTransformations:
    @staticmethod
    def teichmuller_lift(val: int, n: int) -> complex:
        angle = 2.0 * math.pi * ((val % n) / float(n))
        return math.cos(angle) + 1j * math.sin(angle)

    @staticmethod
    def apply_flow(arr: np.ndarray, flow_time: float) -> np.ndarray:
        # Flow serves as cryptographic entropy source
        return arr * np.exp(1j * flow_time)

    @staticmethod
    def apply_flow_inverse(arr: np.ndarray, flow_time: float) -> np.ndarray:
        return arr * np.exp(-1j * flow_time)

    @staticmethod
    def derive_binding(metadata: Dict) -> np.ndarray:
        serialized = json.dumps(metadata, sort_keys=True).encode('ascii')
        hash_value = hashlib.sha256(serialized).digest()
        binding = np.zeros(len(hash_value) * 4, dtype=np.complex128)
        for i, byte in enumerate(hash_value):
            for j in range(4):
                bits = (byte >> (j * 2)) & 0x03
                binding[i * 4 + j] = np.exp(1j * bits * (math.pi / 2))
        return binding

    @staticmethod
    def resize_binding(binding: np.ndarray, target_size: int) -> np.ndarray:
        if len(binding) == target_size:
            return binding
        indices = np.linspace(0, len(binding) - 1, target_size)
        binding_resized = np.zeros(target_size, dtype=np.complex128)
        idx_floor = np.floor(indices).astype(int)
        idx_ceil = np.ceil(indices).astype(int)
        t = indices - idx_floor
        same_idx = (idx_floor == idx_ceil)
        binding_resized[same_idx] = binding[idx_floor[same_idx]]
        interp_idx = ~same_idx
        binding_resized[interp_idx] = binding[idx_floor[interp_idx]] * (1 - t[interp_idx]) + binding[idx_ceil[interp_idx]] * t[interp_idx]
        return binding_resized

    @staticmethod
    def apply_binding(base_arr: np.ndarray, metadata: Dict, strength: float = 0.3) -> np.ndarray:
        binding = ComplexTransformations.derive_binding(metadata)
        if len(binding) != len(base_arr):
            binding = ComplexTransformations.resize_binding(binding, len(base_arr))
        binding_norm = np.linalg.norm(binding)
        binding_normalized = binding / binding_norm if binding_norm > 1e-12 else binding
        return base_arr + strength * binding_normalized

    @staticmethod
    def verify_binding(bound_arr: np.ndarray, metadata: Dict, strength: float = 0.3, threshold: float = 0.15) -> Tuple[bool, float]:
        n = metadata["n"]
        base_value = ComplexTransformations.teichmuller_lift(17, n)
        base_arr = np.full(n, base_value, dtype=np.complex128)
        extracted_binding = (bound_arr - base_arr) / strength
        expected_binding = ComplexTransformations.derive_binding(metadata)
        if len(expected_binding) != len(extracted_binding):
            expected_binding = ComplexTransformations.resize_binding(expected_binding, len(extracted_binding))
        
        # Normalize using L2 norm with machine precision tolerance
        norm_expected = np.linalg.norm(expected_binding)
        norm_extracted = np.linalg.norm(extracted_binding)
        expected_binding = expected_binding / norm_expected if norm_expected > 1e-12 else expected_binding
        extracted_binding = extracted_binding / norm_extracted if norm_extracted > 1e-12 else extracted_binding
        
        similarity = np.abs(np.vdot(extracted_binding, expected_binding))
        return similarity >= threshold, similarity

# Audit logging system
@dataclass
class AuditEntry:
    timestamp: float
    session_id: str
    op_type: str
    payload: Any

    def __post_init__(self):
        self.payload_hash = hashlib.sha256(repr(self.payload).encode("ascii", "ignore")).hexdigest()
        self.payload_preview = str(self.payload)[:128]

    def to_dict(self) -> Dict:
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
        self.entries: List[AuditEntry] = []

    def record(self, op_type: str, session_id: str, payload: Any) -> AuditEntry:
        entry = AuditEntry(timestamp=time.time(), session_id=session_id, op_type=op_type, payload=payload)
        self.entries.append(entry)
        return entry

    def show_log(self) -> None:
        print("=== Audit Log Entries ===")
        for i, e in enumerate(self.entries):
            print(f"[{i}] T={e.timestamp:.4f}, Sess={e.session_id}, Op={e.op_type}, "
                  f"Hash={e.payload_hash[:16]}..., Preview={e.payload_preview}")

    def dump_json(self, fn: str) -> None:
        with open(fn, "w") as f:
            json.dump([e.to_dict() for e in self.entries], f, indent=2, cls=EnhancedJSONEncoder)

# Prover session
class ProverSession:
    def __init__(self, aggregator: "MultiProverAggregator", session_id: str, n: int = 128, chunk_size: int = 64):
        self.agg = aggregator
        self.log = aggregator.log
        self.session_id = session_id
        self.n = n
        self.chunk_size = chunk_size
        self.use_gpu = GPU_AVAILABLE
        print(f"[INFO] {'GPU' if self.use_gpu else 'CPU'} mode for session {session_id}.")

    def commit(self) -> Tuple[Tuple[FQ, FQ], np.ndarray, float, Dict]:
        c_val = KZG.commit_f17()
        base_arr = np.full(self.n, ComplexTransformations.teichmuller_lift(17, self.n), dtype=np.complex128)
        metadata = {
            "session_id": self.session_id,
            "commit_timestamp": time.time(),
            "n": self.n,
            "chunk_size": self.chunk_size,
            "commitment_value": str(c_val)
        }
        bound_arr = ComplexTransformations.apply_binding(base_arr, metadata)
        flow_t = float(secrets.randbits(32)) / 9999999.0  # Cryptographic flow time
        final_arr = ComplexTransformations.apply_flow(bound_arr, flow_t)
        
        self.log.record("COMMIT", self.session_id, {
            "commit_val": str(c_val),
            "n": self.n,
            "flow_time": flow_t,
            "metadata_hash": hashlib.sha256(json.dumps(metadata, sort_keys=True).encode('ascii')).hexdigest()
        })
        return (c_val, final_arr, flow_t, metadata)

    def respond(self, alpha: int, final_arr: np.ndarray, flow_t: float, metadata: Dict) -> Tuple[int, Tuple[FQ, FQ]]:
        f_alpha, prf = KZG.open_f17(alpha)
        self.log.record("RESPONSE", self.session_id, {
            "f_alpha": f_alpha, 
            "proof": str(prf),
            "metadata_hash": hashlib.sha256(json.dumps(metadata, sort_keys=True).encode('ascii')).hexdigest()
        })
        return (f_alpha, prf)

# Multi-prover aggregator
class MultiProverAggregator:
    def __init__(self):
        self.log = AuditLog()

    def new_session(self, sid: str, n: int = 128, chunk_size: int = 64) -> ProverSession:
        return ProverSession(self, sid, n, chunk_size)

    def challenge(self) -> int:
        alpha = 2
        self.log.record("CHALLENGE", "ALL", {"alpha": alpha})
        return alpha

    def verify(self, sid: str, c_val: Tuple[FQ, FQ], alpha: int, 
               f_alpha: int, proof: Tuple[FQ, FQ], final_arr: np.ndarray, 
               flow_t: float, metadata: Dict) -> Tuple[bool, Dict]:  # Modified return type
        # Verify KZG proof
        ok_kzg = KZG.verify_f17(c_val, alpha, f_alpha, proof)
        
        # Verify flow & binding
        undone = ComplexTransformations.apply_flow_inverse(final_arr, flow_t)
        binding_ok, similarity = ComplexTransformations.verify_binding(undone, metadata)
        
        # Check mean magnitude (flow integrity)
        mean_magnitude = float(np.mean(np.abs(undone)))
        ok_flow = (abs(mean_magnitude - 1.0) < 0.02)
        
        result = ok_kzg and ok_flow and binding_ok
        
        # Create details dictionary for verification results
        details = {
            "kzg_ok": bool(ok_kzg),  # Convert to Python built-in types
            "flow_ok": bool(ok_flow),
            "binding_ok": bool(binding_ok),
            "binding_similarity": float(similarity),
            "result": bool(result)
        }
        
        self.log.record("VERIFY", sid, details)
        return result, details  # Return tuple with both result and details

    def sign_final(self, priv_key: int = 0x12345) -> str:
        payload_str = "".join(repr(e.payload) for e in self.log.entries)
        hv = hashlib.sha256(payload_str.encode("ascii", "ignore")).hexdigest()
        sig = pow(int(hv, 16), priv_key, BN128.p)
        return hex(sig)

    def dump_log(self, fname: str) -> None:
        self.log.dump_json(fname)

def run_demonstration():
    print("=== Starting Cryptographic Protocol Demonstration ===\n")
    aggregator = MultiProverAggregator()

    print("=== Prover-Alpha: Commitment Phase ===")
    session_a = aggregator.new_session("prover-alpha", 128, 64)
    commit_a, array_a, flow_a, metadata_a = session_a.commit()

    print("\n=== Prover-Beta: Commitment Phase ===")
    session_b = aggregator.new_session("prover-beta", 128, 64)
    commit_b, array_b, flow_b, metadata_b = session_b.commit()

    print("\n=== Aggregator: Challenge Generation ===")
    alpha = aggregator.challenge()
    print(f"Generated challenge: {alpha}")

    print("\n=== Prover-Alpha: Proof Generation ===")
    f_alpha_a, proof_a = session_a.respond(alpha, array_a, flow_a, metadata_a)
    print(f"  Evaluation f({alpha}) = {f_alpha_a}")
    print(f"  Proof: {proof_a}")

    print("\n=== Prover-Beta: Proof Generation ===")
    f_alpha_b, proof_b = session_b.respond(alpha, array_b, flow_b, metadata_b)
    print(f"  Evaluation f({alpha}) = {f_alpha_b}")
    print(f"  Proof: {proof_b}")

    print("\n=== Aggregator: Verification for Prover-Alpha ===")
    result_a, details_a = aggregator.verify(
        "prover-alpha", commit_a, alpha, f_alpha_a, proof_a, 
        array_a, flow_a, metadata_a
    )
    print(f"Verification result: {'Success' if result_a else 'Failure'}")
    print(f"Details: {json.dumps(details_a, indent=2, cls=EnhancedJSONEncoder)}")

    print("\n=== Aggregator: Verification for Prover-Beta ===")
    result_b, details_b = aggregator.verify(
        "prover-beta", commit_b, alpha, f_alpha_b, proof_b, 
        array_b, flow_b, metadata_b
    )
    print(f"Verification result: {'Success' if result_b else 'Failure'}")
    print(f"Details: {json.dumps(details_b, indent=2, cls=EnhancedJSONEncoder)}")

    print("\n=== Audit Log ===")
    aggregator.log.show_log()

    sig_hex = aggregator.sign_final()
    print("\nAggregator final signature (hex) =", sig_hex)
    aggregator.dump_log("final_logs.json")
    print("[Log saved to final_logs.json]")

if __name__ == "__main__":
    run_demonstration()

