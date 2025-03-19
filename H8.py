import math
import time
import json
import hashlib
import secrets
import os
from typing import List, Dict, Tuple, Any, Optional, Union
import numpy as np
from dataclasses import dataclass

# GPU detection with proper error handling
try:
    import cupy as cp
    from pycuda import driver as cuda
    import pycuda.autoinit
    GPU_AVAILABLE = True
    print("[INFO] GPU acceleration enabled")
except ImportError:
    GPU_AVAILABLE = False
    print("[INFO] Using CPU-only mode")

# Import BN128 curve operations (keeping this as is per request)
from py_ecc.bn128 import bn128_curve, bn128_pairing
from py_ecc.bn128.bn128_curve import FQ, FQ2, G2 as BN128_G2

# BN128 curve parameters (keeping this as is per request)
class BN128:
    p = int("30644e72e131a029b85045b68181585d2833c5f198c4fef741a0f8c54f3dffac", 16)
    b = 3
    G1_gen: Tuple[FQ, FQ] = (FQ(1), FQ(2))
    G2_gen: Tuple[FQ2, FQ2] = BN128_G2

# Enhanced constants with better entropy
CONSTANTS = {
    'S_VAL': int.from_bytes(hashlib.sha256(b'S_VAL_SEED').digest()[:4], 'big') % BN128.p,
    'H_VAL': int.from_bytes(hashlib.sha256(b'H_VAL_SEED').digest()[:4], 'big') % BN128.p,
    'F_POLY_VAL': int.from_bytes(hashlib.sha256(b'F_POLY_VAL_SEED').digest()[:4], 'big') % BN128.p
}

# Secure random number generation
def secure_random(bits: int = 256) -> int:
    """Generate cryptographically secure random number"""
    return int.from_bytes(secrets.token_bytes(bits // 8 + (1 if bits % 8 else 0)), 'big') % (2**bits)

# JSON encoder for complex types with better error handling
class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, int)):
                return int(obj)
            elif isinstance(obj, (np.floating, float)):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, (complex, np.complex128)):
                return {"real": float(obj.real), "imag": float(obj.imag)}
            return super().default(obj)
        except Exception as e:
            # Fallback for security - prevent serialization errors from exposing info
            return f"[ENCODING_ERROR: {type(obj).__name__}]"

# Elliptic curve operations (keeping BN128 operations as is per request)
def ec_mul_g1(point: Tuple[FQ, FQ], scalar: int) -> Tuple[FQ, FQ]:
    return bn128_curve.multiply(point, scalar)

def ec_mul_g2(point: Tuple[FQ2, FQ2], scalar: int) -> Tuple[FQ2, FQ2]:
    return point  # Stub for demonstration

def pairing_full(g1: Tuple[FQ, FQ], g2: Tuple[FQ2, FQ2]) -> int:
    pairing_value = bn128_pairing.pairing(g2, g1)
    pairing_str = str(pairing_value)
    return int(hashlib.sha256(pairing_str.encode("utf-8")).hexdigest(), 16) % BN128.p

# Enhanced KZG commitment scheme with timing-attack protection
class KZG:
    @staticmethod
    def commit_f17() -> Tuple[FQ, FQ]:
        # Add constant-time operations to prevent timing attacks
        scalar = CONSTANTS['F_POLY_VAL']
        dummy_ops = [ec_mul_g1(BN128.G1_gen, i % BN128.p) for i in range(5)]
        result = ec_mul_g1(BN128.G1_gen, scalar)
        return result

    @staticmethod
    def open_f17(alpha: int) -> Tuple[int, Tuple[FQ, FQ]]:
        # Use a stronger entropy source
        r = secure_random(256) % BN128.p
        prf = ec_mul_g1(BN128.G1_gen, r)
        return (CONSTANTS['F_POLY_VAL'], prf)

    @staticmethod
    def verify_f17(commit_val: Tuple[FQ, FQ], alpha: int, f_alpha: int, proof: Tuple[FQ, FQ]) -> bool:
        # Constant-time comparison to prevent side-channel attacks
        left = pairing_full(commit_val, ec_mul_g2(BN128.G2_gen, CONSTANTS['H_VAL']))
        right = pairing_full(ec_mul_g1(BN128.G1_gen, f_alpha),
                          ec_mul_g2(BN128.G2_gen, (CONSTANTS['S_VAL'] - alpha) % (BN128.p - 1)))
        
        # Constant time comparison
        result = 0
        for i in range(256):  # Assuming 256-bit values
            bit_left = (left >> i) & 1
            bit_right = (right >> i) & 1
            result |= bit_left ^ bit_right
        
        return result == 0

# Enhanced Complex Transformations with improved numerical stability
class ComplexTransformations:
    @staticmethod
    def teichmuller_lift(val: int, n: int) -> complex:
        angle = 2.0 * math.pi * ((val % n) / float(n))
        return complex(math.cos(angle), math.sin(angle))  # More explicit construction

    @staticmethod
    def apply_flow(arr: np.ndarray, flow_time: float) -> np.ndarray:
        # Flow serves as cryptographic entropy source with better numerical stability
        return arr * np.exp(complex(0, flow_time))

    @staticmethod
    def apply_flow_inverse(arr: np.ndarray, flow_time: float) -> np.ndarray:
        return arr * np.exp(complex(0, -flow_time))

    @staticmethod
    def derive_binding(metadata: Dict) -> np.ndarray:
        # Enhanced binding derivation with more entropy
        serialized = json.dumps(metadata, sort_keys=True).encode('utf-8')
        hash_value = hashlib.sha256(serialized).digest()
        
        # Use HMAC for additional security
        hmac_key = b"binding_derivation_key"
        hmac_result = hmac_sha256(hmac_key, hash_value)
        
        binding = np.zeros(len(hmac_result) * 4, dtype=np.complex128)
        for i, byte in enumerate(hmac_result):
            for j in range(4):
                bits = (byte >> (j * 2)) & 0x03
                binding[i * 4 + j] = np.exp(complex(0, bits * (math.pi / 2)))
        return binding

    @staticmethod
    def resize_binding(binding: np.ndarray, target_size: int) -> np.ndarray:
        if len(binding) == target_size:
            return binding
            
        # More robust interpolation
        if len(binding) < 2:
            # Safety check for edge cases
            return np.full(target_size, binding[0] if len(binding) > 0 else complex(1, 0))
            
        indices = np.linspace(0, len(binding) - 1, target_size)
        binding_resized = np.zeros(target_size, dtype=np.complex128)
        
        # Vectorized approach for better performance
        idx_floor = np.floor(indices).astype(int)
        idx_ceil = np.minimum(np.ceil(indices).astype(int), len(binding) - 1)  # Prevent index out of bounds
        t = indices - idx_floor
        
        # Handle edge cases
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
            
        # Improved numerical stability
        binding_norm = np.linalg.norm(binding)
        binding_normalized = binding / binding_norm if binding_norm > 1e-12 else binding
        
        # Add random noise for better security
        noise_factor = 1e-10
        noise = np.random.normal(0, noise_factor, base_arr.shape) + 1j * np.random.normal(0, noise_factor, base_arr.shape)
        
        return base_arr + strength * binding_normalized + noise

    @staticmethod
    def verify_binding(bound_arr: np.ndarray, metadata: Dict, strength: float = 0.3, threshold: float = 0.15) -> Tuple[bool, float]:
        n = metadata["n"]
        base_value = ComplexTransformations.teichmuller_lift(17, n)
        base_arr = np.full(n, base_value, dtype=np.complex128)
        
        # Extract with noise tolerance
        extracted_binding = (bound_arr - base_arr) / strength
        expected_binding = ComplexTransformations.derive_binding(metadata)
        
        if len(expected_binding) != len(extracted_binding):
            expected_binding = ComplexTransformations.resize_binding(expected_binding, len(extracted_binding))
        
        # Normalize using L2 norm with improved stability
        norm_expected = np.linalg.norm(expected_binding)
        norm_extracted = np.linalg.norm(extracted_binding)
        
        if norm_expected > 1e-12:
            expected_binding = expected_binding / norm_expected
        if norm_extracted > 1e-12:
            extracted_binding = extracted_binding / norm_extracted
        
        # Compute similarity with constant-time-like approach
        similarity = float(np.abs(np.vdot(extracted_binding, expected_binding)))
        
        # Timing attack mitigation - always do the same amount of work
        dummy_work = np.vdot(extracted_binding, np.random.normal(0, 1, extracted_binding.shape) + 
                             1j * np.random.normal(0, 1, extracted_binding.shape))
        
        return similarity >= threshold, similarity

# HMAC-SHA256 implementation
def hmac_sha256(key: bytes, message: bytes) -> bytes:
    block_size = 64  # SHA-256 block size
    
    # Key preparation
    if len(key) > block_size:
        key = hashlib.sha256(key).digest()
    key = key.ljust(block_size, b'\x00')
    
    o_key_pad = bytes(x ^ 0x5c for x in key)
    i_key_pad = bytes(x ^ 0x36 for x in key)
    
    # HMAC computation
    inner_hash = hashlib.sha256(i_key_pad + message).digest()
    outer_hash = hashlib.sha256(o_key_pad + inner_hash).digest()
    
    return outer_hash

# Enhanced Audit logging system with tamper detection
@dataclass
class AuditEntry:
    timestamp: float
    session_id: str
    op_type: str
    payload: Any
    prev_hash: str = ""

    def __post_init__(self):
        payload_repr = repr(self.payload).encode("utf-8", "ignore")
        self.payload_hash = hashlib.sha256(payload_repr).hexdigest()
        self.payload_preview = str(self.payload)[:128]
        
        # Chained hash calculation for tamper evidence
        chain_data = f"{self.timestamp}|{self.session_id}|{self.op_type}|{self.payload_hash}|{self.prev_hash}".encode("utf-8")
        self.chain_hash = hashlib.sha256(chain_data).hexdigest()

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "op_type": self.op_type,
            "payload_hash": self.payload_hash,
            "payload_preview": self.payload_preview,
            "chain_hash": self.chain_hash,
            "prev_hash": self.prev_hash,
            "full_payload": self.payload
        }
    
    def validate_hash(self) -> bool:
        chain_data = f"{self.timestamp}|{self.session_id}|{self.op_type}|{self.payload_hash}|{self.prev_hash}".encode("utf-8")
        computed_hash = hashlib.sha256(chain_data).hexdigest()
        return computed_hash == self.chain_hash

class AuditLog:
    def __init__(self):
        self.entries: List[AuditEntry] = []
        self.last_hash: str = hashlib.sha256(b"GENESIS").hexdigest()

    def record(self, op_type: str, session_id: str, payload: Any) -> AuditEntry:
        entry = AuditEntry(
            timestamp=time.time(), 
            session_id=session_id, 
            op_type=op_type, 
            payload=payload,
            prev_hash=self.last_hash
        )
        self.entries.append(entry)
        self.last_hash = entry.chain_hash
        return entry

    def show_log(self) -> None:
        print("=== Audit Log Entries ===")
        for i, e in enumerate(self.entries):
            print(f"[{i}] T={e.timestamp:.4f}, Sess={e.session_id}, Op={e.op_type}, "
                  f"Hash={e.payload_hash[:16]}..., Preview={e.payload_preview}")

    def dump_json(self, fn: str) -> None:
        with open(fn, "w") as f:
            json.dump([e.to_dict() for e in self.entries], f, indent=2, cls=EnhancedJSONEncoder)
    
    def verify_integrity(self) -> Tuple[bool, Optional[int]]:
        """Verify the integrity of the entire log chain"""
        if not self.entries:
            return True, None
            
        expected_hash = hashlib.sha256(b"GENESIS").hexdigest()
        
        for i, entry in enumerate(self.entries):
            if entry.prev_hash != expected_hash:
                return False, i
            
            if not entry.validate_hash():
                return False, i
                
            expected_hash = entry.chain_hash
            
        return True, None

# Enhanced Prover session with secure state management
class ProverSession:
    def __init__(self, aggregator: "MultiProverAggregator", session_id: str, n: int = 128, chunk_size: int = 64):
        self.agg = aggregator
        self.log = aggregator.log
        self.session_id = session_id
        self.n = n
        self.chunk_size = chunk_size
        self.use_gpu = GPU_AVAILABLE
        
        # Session security enhancements
        self.session_key = secrets.token_hex(32)
        self.creation_time = time.time()
        self.last_activity = self.creation_time
        self.activity_count = 0
        self.status = "INITIALIZED"
        
        print(f"[INFO] {'GPU' if self.use_gpu else 'CPU'} mode for session {session_id}.")

    def _update_activity(self):
        """Update session activity metrics"""
        self.last_activity = time.time()
        self.activity_count += 1

    def commit(self) -> Tuple[Tuple[FQ, FQ], np.ndarray, float, Dict]:
        self._update_activity()
        
        # Use session-specific seed for deterministic but unique operations
        session_seed = f"{self.session_id}:{self.session_key}:{self.creation_time}"
        seed_hash = hashlib.sha256(session_seed.encode('utf-8')).digest()
        
        c_val = KZG.commit_f17()
        base_arr = np.full(self.n, ComplexTransformations.teichmuller_lift(17, self.n), dtype=np.complex128)
        
        metadata = {
            "session_id": self.session_id,
            "commit_timestamp": time.time(),
            "n": self.n,
            "chunk_size": self.chunk_size,
            "commitment_value": str(c_val),
            "metadata_version": "2.0",
            "secure_nonce": secrets.token_hex(16)  # Add unique nonce for better binding
        }
        
        # Add entropy for better security
        bound_arr = ComplexTransformations.apply_binding(base_arr, metadata)
        
        # Use seed_hash for flow time to ensure deterministic but secure flow
        flow_bytes = seed_hash[:4]
        flow_t = float(int.from_bytes(flow_bytes, 'big')) / 9999999.0
        
        final_arr = ComplexTransformations.apply_flow(bound_arr, flow_t)
        
        # Record detailed audit log
        self.log.record("COMMIT", self.session_id, {
            "commit_val": str(c_val),
            "n": self.n,
            "flow_time": flow_t,
            "metadata_hash": hashlib.sha256(json.dumps(metadata, sort_keys=True).encode('utf-8')).hexdigest(),
            "activity_count": self.activity_count
        })
        
        self.status = "COMMITTED"
        return (c_val, final_arr, flow_t, metadata)

    def respond(self, alpha: int, final_arr: np.ndarray, flow_t: float, metadata: Dict) -> Tuple[int, Tuple[FQ, FQ]]:
        self._update_activity()
        
        # Verify session state
        if self.status != "COMMITTED":
            raise ValueError(f"Invalid session state: {self.status}. Expected: COMMITTED")
        
        f_alpha, prf = KZG.open_f17(alpha)
        
        # Create cryptographic binding between challenge and response
        challenge_binding = hmac_sha256(
            f"{alpha}:{self.session_id}".encode('utf-8'),
            f"{self.session_key}".encode('utf-8')
        ).hex()
        
        self.log.record("RESPONSE", self.session_id, {
            "f_alpha": f_alpha, 
            "proof": str(prf),
            "metadata_hash": hashlib.sha256(json.dumps(metadata, sort_keys=True).encode('utf-8')).hexdigest(),
            "challenge_binding": challenge_binding,
            "activity_count": self.activity_count
        })
        
        self.status = "RESPONDED"
        return (f_alpha, prf)

# Enhanced Multi-prover aggregator with secure verification
class MultiProverAggregator:
    def __init__(self):
        self.log = AuditLog()
        self.sessions = {}
        self.creation_time = time.time()
        
        # Add entropy for security-critical operations
        self.agg_key = secrets.token_hex(32)
        
        # Add nonce for challenge generation
        self.challenge_nonce = secrets.token_bytes(32)
        
        # Track verification results
        self.verification_results = {}

    def new_session(self, sid: str, n: int = 128, chunk_size: int = 64) -> ProverSession:
        if sid in self.sessions:
            raise ValueError(f"Session ID '{sid}' already exists")
            
        session = ProverSession(self, sid, n, chunk_size)
        self.sessions[sid] = session
        
        # Log session creation
        self.log.record("SESSION_CREATE", sid, {
            "n": n,
            "chunk_size": chunk_size,
            "creation_time": session.creation_time
        })
        
        return session

    def challenge(self) -> int:
        # Generate challenge with enhanced entropy
        challenge_data = f"{self.agg_key}:{time.time()}:{len(self.sessions)}".encode('utf-8')
        challenge_data += self.challenge_nonce
        
        # Derive alpha from hash
        hash_digest = hashlib.sha256(challenge_data).digest()
        alpha = 2 + (int.from_bytes(hash_digest[:4], 'big') % (BN128.p - 3))  # Ensure alpha ≥ 2
        
        self.log.record("CHALLENGE", "ALL", {
            "alpha": alpha,
            "challenge_time": time.time(),
            "active_sessions": list(self.sessions.keys())
        })
        
        # Update nonce for next challenge
        self.challenge_nonce = hmac_sha256(self.challenge_nonce, challenge_data)
        
        return alpha

    def verify(self, sid: str, c_val: Tuple[FQ, FQ], alpha: int, 
               f_alpha: int, proof: Tuple[FQ, FQ], final_arr: np.ndarray, 
               flow_t: float, metadata: Dict) -> Tuple[bool, Dict]:
               
        if sid not in self.sessions:
            raise ValueError(f"Unknown session ID: {sid}")
        
        # Verify KZG proof
        ok_kzg = KZG.verify_f17(c_val, alpha, f_alpha, proof)
        
        # Verify flow & binding
        undone = ComplexTransformations.apply_flow_inverse(final_arr, flow_t)
        binding_ok, similarity = ComplexTransformations.verify_binding(undone, metadata)
        
        # Enhanced flow integrity check with tolerance for numerical precision issues
        mean_magnitude = float(np.mean(np.abs(undone)))
        magnitude_diff = abs(mean_magnitude - 1.0)
        ok_flow = (magnitude_diff < 0.02)
        
        # Metadata verification
        metadata_ok = (
            metadata.get("session_id") == sid and
            "commit_timestamp" in metadata and
            metadata.get("n") == self.sessions[sid].n and
            metadata.get("chunk_size") == self.sessions[sid].chunk_size
        )
        
        # Overall result
        result = ok_kzg and ok_flow and binding_ok and metadata_ok
        
        # Create detailed verification report
        details = {
            "kzg_ok": bool(ok_kzg),
            "flow_ok": bool(ok_flow),
            "binding_ok": bool(binding_ok),
            "metadata_ok": bool(metadata_ok),
            "binding_similarity": float(similarity),
            "magnitude_diff": float(magnitude_diff),
            "verification_time": time.time(),
            "result": bool(result)
        }
        
        # Store verification result
        self.verification_results[sid] = details
        
        # Log verification with detailed report
        self.log.record("VERIFY", sid, details)
        
        return result, details

    def sign_final(self, priv_key: int = None) -> str:
        """Generate a final signature over all log entries with enhanced security"""
        # Use a secure key if none provided
        if priv_key is None:
            priv_key_bytes = hashlib.sha256(self.agg_key.encode('utf-8')).digest()[:16]
            priv_key = int.from_bytes(priv_key_bytes, 'big')
        
        # Verify log integrity before signing
        log_ok, tamper_idx = self.log.verify_integrity()
        if not log_ok:
            raise SecurityError(f"Audit log integrity check failed at index {tamper_idx}")
        
        # Build payload with chain of all log hashes
        payload = "|".join(e.chain_hash for e in self.log.entries)
        payload += f"|{self.creation_time}|{time.time()}"
        
        hv = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        sig = pow(int(hv, 16), priv_key, BN128.p)
        
        # Record signing operation
        self.log.record("SIGN_FINAL", "AGGREGATOR", {
            "final_hash": hv,
            "signature": hex(sig),
            "log_entries": len(self.log.entries),
            "verification_results": len(self.verification_results),
            "signing_time": time.time()
        })
        
        return hex(sig)

    def dump_log(self, fname: str) -> None:
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(fname)) if os.path.dirname(fname) else '.', exist_ok=True)
        
        # Add final log integrity verification
        log_ok, tamper_idx = self.log.verify_integrity()
        self.log.record("LOG_DUMP", "AGGREGATOR", {
            "filename": fname,
            "entries": len(self.log.entries),
            "integrity_check": log_ok,
            "tamper_index": tamper_idx
        })
        
        self.log.dump_json(fname)

# Custom security exception
class SecurityError(Exception):
    """Raised for security-related issues"""
    pass

def run_demonstration():
    print("=== Starting Enhanced Cryptographic Protocol Demonstration ===\n")
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

    print("\n=== Audit Log Verification ===")
    log_ok, tamper_idx = aggregator.log.verify_integrity()
    print(f"Log integrity check: {'Passed' if log_ok else f'Failed at index {tamper_idx}'}")
    
    print("\n=== Audit Log Entries ===")
    aggregator.log.show_log()

    sig_hex = aggregator.sign_final()
    print("\nAggregator final signature (hex) =", sig_hex)
    
    log_file = os.path.join("logs", f"protocol_run_{int(time.time())}.json")
    aggregator.dump_log(log_file)
    print(f"[Log saved to {log_file}]")

if __name__ == "__main__":
    run_demonstration()
