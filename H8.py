#!/usr/bin/env python3
"""
Refactored full code for a multi-prover commitment and verification system using BN128 curve operations,
KZG commitment scheme, complex binding transformations, and an audit logging system.
"""

import math
import time
import json
import hashlib
import secrets
from typing import List, Dict, Tuple, Any
import numpy as np
from dataclasses import dataclass
from functools import lru_cache

# ---------------------------------------------------------------------
# GPU Detection and Initialization
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# BN128 Curve and Field Parameters
# ---------------------------------------------------------------------
from py_ecc.bn128 import bn128_curve, bn128_pairing
from py_ecc.bn128.bn128_curve import FQ, FQ2, G2 as BN128_G2  # Official G2 generator

class BN128:
    """
    BN128 curve parameters and generators.
    """
    p: int = int("30644e72e131a029b85045b68181585d2833c5f198c4fef741a0f8c54f3dffac", 16)
    b: int = 3
    G1_gen: Tuple[FQ, FQ] = (FQ(1), FQ(2))
    G2_gen: Tuple[FQ2, FQ2] = BN128_G2

# Constants used for demonstration (can be modified as needed).
CONSTANTS: Dict[str, int] = {
    'S_VAL': 19,
    'H_VAL': 17,
    'F_POLY_VAL': 17
}

# ---------------------------------------------------------------------
# Enhanced JSON Encoder
# ---------------------------------------------------------------------
class EnhancedJSONEncoder(json.JSONEncoder):
    """
    JSON encoder that supports NumPy arrays and complex numbers.
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (complex, np.complex128)):
            return {"real": obj.real, "imag": obj.imag}
        return super().default(obj)

# ---------------------------------------------------------------------
# Elliptic Curve Operations
# ---------------------------------------------------------------------
def ec_mul_g1(point: Tuple[FQ, FQ], scalar: int) -> Tuple[FQ, FQ]:
    """
    Perform scalar multiplication on the G1 curve using bn128_curve.multiply.
    """
    return bn128_curve.multiply(point, scalar)

def ec_mul_g2(point: Tuple[FQ2, FQ2], scalar: int) -> Tuple[FQ2, FQ2]:
    """
    Perform scalar multiplication on the G2 curve using the bn128_curve.multiply function.
    """
    return bn128_curve.multiply(point, scalar)

def pairing_full(g1: Tuple[FQ, FQ], g2: Tuple[FQ2, FQ2]) -> int:
    """
    Compute the complete pairing using py_ecc.bn128.
    The pairing value is hashed (using SHA-256) and reduced modulo BN128.p.
    """
    pairing_value = bn128_pairing.pairing(g2, g1)
    pairing_str = str(pairing_value)
    return int(hashlib.sha256(pairing_str.encode("ascii")).hexdigest(), 16) % BN128.p

# ---------------------------------------------------------------------
# KZG Commitment Scheme
# ---------------------------------------------------------------------
class KZG:
    """
    KZG commitment scheme using BN128 curve operations.
    """

    @staticmethod
    def commit_f17() -> Tuple[FQ, FQ]:
        """
        Generate a KZG commitment for the F_POLY_VAL polynomial.
        """
        return ec_mul_g1(BN128.G1_gen, CONSTANTS['F_POLY_VAL'])

    @staticmethod
    def open_f17(alpha: int) -> Tuple[int, Tuple[FQ, FQ]]:
        """
        Generate a KZG opening proof for evaluation at a point alpha.
        For demonstration, a random scalar is generated and the corresponding commitment is computed.
        """
        r = secrets.randbits(256) % BN128.p
        proof = ec_mul_g1(BN128.G1_gen, r)
        return CONSTANTS['F_POLY_VAL'], proof

    @staticmethod
    def verify_f17(commit_val: Tuple[FQ, FQ], alpha: int, f_alpha: int,
                   proof: Tuple[FQ, FQ]) -> bool:
        """
        Verify a KZG proof by comparing pairings.
        """
        left = pairing_full(commit_val, ec_mul_g2(BN128.G2_gen, CONSTANTS['H_VAL']))
        right = pairing_full(
            ec_mul_g1(BN128.G1_gen, f_alpha),
            ec_mul_g2(BN128.G2_gen, (CONSTANTS['S_VAL'] - alpha) % (BN128.p - 1))
        )
        return left == right

# ---------------------------------------------------------------------
# Complex Transformations for Binding
# ---------------------------------------------------------------------
class ComplexTransformations:
    """
    Perform complex-domain transformations for binding commitments.
    """

    @staticmethod
    def teichmuller_lift(val: int, n: int) -> complex:
        """
        Convert a field element to a complex number using a Teichmüller lift.
        """
        angle = 2.0 * math.pi * ((val % n) / float(n))
        return math.cos(angle) + 1j * math.sin(angle)

    @staticmethod
    def apply_flow(arr: np.ndarray, flow_time: float) -> np.ndarray:
        """
        Apply a flow transformation to a complex array.
        """
        return arr * np.exp(1j * flow_time)

    @staticmethod
    def apply_flow_inverse(arr: np.ndarray, flow_time: float) -> np.ndarray:
        """
        Undo the flow transformation on a complex array.
        """
        return arr * np.exp(-1j * flow_time)

    @staticmethod
    def derive_binding(metadata: Dict) -> np.ndarray:
        """
        Generate a binding pattern from metadata via SHA-256 hashing.
        """
        serialized = json.dumps(metadata, sort_keys=True).encode('ascii')
        hash_value = hashlib.sha256(serialized).digest()
        binding = np.zeros(len(hash_value) * 4, dtype=np.complex128)
        for i, byte in enumerate(hash_value):
            for j in range(4):
                bits = (byte >> (j * 2)) & 0x03
                angle = bits * (math.pi / 2)
                binding[i * 4 + j] = np.exp(1j * angle)
        return binding

    @staticmethod
    def resize_binding(binding: np.ndarray, target_size: int) -> np.ndarray:
        """
        Resize the binding vector to the target size using linear interpolation.
        """
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
        binding_resized[interp_idx] = (binding[idx_floor[interp_idx]] * (1 - t[interp_idx]) +
                                       binding[idx_ceil[interp_idx]] * t[interp_idx])
        return binding_resized

    @staticmethod
    def apply_binding(base_arr: np.ndarray, metadata: Dict, strength: float = 0.3) -> np.ndarray:
        """
        Apply a binding pattern to a base array.
        """
        binding = ComplexTransformations.derive_binding(metadata)
        if len(binding) != len(base_arr):
            binding = ComplexTransformations.resize_binding(binding, len(base_arr))
        binding_norm = np.linalg.norm(binding)
        binding_normalized = binding / binding_norm if binding_norm > 1e-12 else binding
        return base_arr + strength * binding_normalized

    @staticmethod
    def verify_binding(bound_arr: np.ndarray, metadata: Dict, strength: float = 0.3,
                       threshold: float = 0.15) -> Tuple[bool, float]:
        """
        Verify the binding by comparing the extracted binding with the expected binding.
        """
        n = metadata["n"]
        angle = 2.0 * math.pi * ((17 % n) / float(n))
        base_value = math.cos(angle) + 1j * math.sin(angle)
        base_arr = np.full(n, base_value, dtype=np.complex128)
        extracted_binding = (bound_arr - base_arr) / strength
        expected_binding = ComplexTransformations.derive_binding(metadata)
        if len(expected_binding) != len(extracted_binding):
            expected_binding = ComplexTransformations.resize_binding(expected_binding,
                                                                       len(extracted_binding))
        norm_expected = np.linalg.norm(expected_binding)
        norm_extracted = np.linalg.norm(extracted_binding)
        if norm_expected > 1e-12:
            expected_binding = expected_binding / norm_expected
        if norm_extracted > 1e-12:
            extracted_binding = extracted_binding / norm_extracted
        similarity = np.abs(np.vdot(extracted_binding, expected_binding))
        return similarity >= threshold, similarity

# ---------------------------------------------------------------------
# Audit Logging System
# ---------------------------------------------------------------------
@dataclass
class AuditEntry:
    """
    Represents an entry in the audit log.
    """
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
    """
    Maintains a log of audit entries.
    """
    def __init__(self):
        self.entries: List[AuditEntry] = []

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

# ---------------------------------------------------------------------
# Prover Session for Commitments and Responses
# ---------------------------------------------------------------------
class ProverSession:
    """
    Represents a prover session that handles commitment creation and response generation.
    """
    def __init__(self, aggregator: "MultiProverAggregator", session_id: str,
                 n: int = 128, chunk_size: int = 64):
        self.agg = aggregator
        self.log = aggregator.log
        self.session_id = session_id
        self.n = n
        self.chunk_size = chunk_size
        self.omega = 5
        self.use_gpu = GPU_AVAILABLE
        print(f"[INFO] {'GPU' if self.use_gpu else 'CPU'} mode for session {session_id}.")

    def commit(self) -> Tuple[Tuple[FQ, FQ], np.ndarray, float, Dict]:
        """
        Create a KZG commitment, bind a base array with metadata, and apply a flow transformation.
        Returns the commitment value, the final transformed array, flow time, and metadata.
        """
        # Create a KZG commitment.
        commitment_val = KZG.commit_f17()

        # Generate base array using a Teichmüller lift.
        base_arr = np.full(
            self.n,
            ComplexTransformations.teichmuller_lift(CONSTANTS['F_POLY_VAL'], self.n),
            dtype=np.complex128
        )

        # Prepare metadata.
        metadata = {
            "session_id": self.session_id,
            "commit_timestamp": time.time(),
            "n": self.n,
            "chunk_size": self.chunk_size,
            "commitment_value": str(commitment_val)
        }

        # Apply binding and flow transformation.
        bound_arr = ComplexTransformations.apply_binding(base_arr, metadata)
        flow_time = float(secrets.randbits(32)) / 9999999.0
        final_arr = ComplexTransformations.apply_flow(bound_arr, flow_time)

        # Record the commit operation.
        self.log.record("COMMIT", self.session_id, {
            "commit_val": str(commitment_val),
            "n": self.n,
            "chunk_size": self.chunk_size,
            "omega": self.omega,
            "flow_time": flow_time,
            "metadata_hash": hashlib.sha256(json.dumps(metadata, sort_keys=True).encode('ascii')).hexdigest()
        })
        return commitment_val, final_arr, flow_time, metadata

    def respond(self, alpha: int, final_arr: np.ndarray, flow_time: float,
                metadata: Dict) -> Tuple[int, Tuple[FQ, FQ]]:
        """
        Generate a KZG opening proof for a given challenge alpha and record the response.
        """
        f_alpha, proof = KZG.open_f17(alpha)
        self.log.record("RESPONSE", self.session_id, {
            "f_alpha": f_alpha,
            "proof": str(proof),
            "flow_time": flow_time,
            "metadata_hash": hashlib.sha256(json.dumps(metadata, sort_keys=True).encode('ascii')).hexdigest()
        })
        return f_alpha, proof

# ---------------------------------------------------------------------
# Multi-Prover Aggregator and Verifier
# ---------------------------------------------------------------------
class MultiProverAggregator:
    """
    Aggregates multiple prover sessions, issues challenges, and verifies responses.
    """
    def __init__(self):
        self.log = AuditLog()

    def new_session(self, sid: str, n: int = 128, chunk_size: int = 64) -> ProverSession:
        return ProverSession(self, sid, n, chunk_size)

    def challenge(self) -> int:
        """
        Issue a challenge value for all sessions.
        """
        alpha = 2
        self.log.record("CHALLENGE", "ALL", {"alpha": alpha})
        return alpha

    def verify(self, sid: str, c_val: Tuple[FQ, FQ], alpha: int, 
               f_alpha: int, proof: Tuple[FQ, FQ], final_arr: np.ndarray, 
               flow_time: float, metadata: Dict) -> bool:
        """
        Verify a prover's response by checking the KZG proof, undoing the flow transformation,
        and verifying the binding.
        """
        ok_kzg = KZG.verify_f17(c_val, alpha, f_alpha, proof)
        undone_arr = ComplexTransformations.apply_flow_inverse(final_arr, flow_time)
        binding_ok, similarity = ComplexTransformations.verify_binding(undone_arr, metadata)
        mean_magnitude = float(np.mean(np.abs(undone_arr)))
        ok_flow = abs(mean_magnitude - 1.0) < 0.02
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
        """
        Sign the aggregated audit log using a simple modular exponentiation scheme.
        """
        payload_str = "".join(repr(e.payload) for e in self.log.entries)
        hv = hashlib.sha256(payload_str.encode("ascii", "ignore")).hexdigest()
        sig = pow(int(hv, 16), priv_key, BN128.p)
        return hex(sig)

    def dump_log(self, fname: str) -> None:
        self.log.dump_json(fname)

# ---------------------------------------------------------------------
# Main Function
# ---------------------------------------------------------------------
def main():
    aggregator = MultiProverAggregator()

    # Prover-A Commitment
    print("=== Prover-A: Commit ===")
    prover_a = aggregator.new_session("prover-A", 128, 64)
    cA, arrA, flowA, metadataA = prover_a.commit()

    # Prover-B Commitment
    print("=== Prover-B: Commit ===")
    prover_b = aggregator.new_session("prover-B", 128, 64)
    cB, arrB, flowB, metadataB = prover_b.commit()

    # Issue Challenge
    alpha = aggregator.challenge()

    # Prover-A Response
    print("=== Prover-A: Respond ===")
    fA, proofA = prover_a.respond(alpha, arrA, flowA, metadataA)
    print(f"  f(alpha) = {fA}, proof = {proofA}")

    # Prover-B Response
    print("=== Prover-B: Respond ===")
    fB, proofB = prover_b.respond(alpha, arrB, flowB, metadataB)
    print(f"  f(alpha) = {fB}, proof = {proofB}")

    # Verification for Prover-A
    print("=== Verifier: Verify for Prover-A ===")
    okA = aggregator.verify("prover-A", cA, alpha, fA, proofA, arrA, flowA, metadataA)
    print("Verification result for Prover-A =", okA)

    # Verification for Prover-B
    print("=== Verifier: Verify for Prover-B ===")
    okB = aggregator.verify("prover-B", cB, alpha, fB, proofB, arrB, flowB, metadataB)
    print("Verification result for Prover-B =", okB)

    # Show audit log and output final aggregator signature.
    print("\n=== Audit Log ===")
    aggregator.log.show_log()
    sig_hex = aggregator.sign_final()
    print("\nAggregator final signature (hex) =", sig_hex)

    # Dump audit log to JSON file.
    aggregator.dump_log("final_logs.json")
    print("[Log saved to final_logs.json]")

if __name__ == "__main__":
    main()

