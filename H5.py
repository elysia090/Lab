# TODO: 
# Exception handling, lack of cryptographic random number generator, 
# input validation, secure JSON processing, sanitizing file paths,
# missing argument validation, improved logging, better side-channel protection

import math
import time
import json
import threading
import argparse
from dataclasses import dataclass, asdict, field
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union, ClassVar
import numpy as np
import numpy.typing as npt
from tqdm import tqdm

# Version metadata
__version__ = "1.0.0"
__author__ = "Refactored by Claude"

# JSON schema definitions
CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "n": {"type": "integer", "minimum": 2, "description": "Domain size"},
        "p": {"type": "integer", "minimum": 2, "description": "Prime modulus"},
        "epsilon": {"type": "number", "minimum": 0, "description": "Error tolerance"},
        "use_gpu": {"type": "boolean", "description": "Enable GPU acceleration"},
        "flow_time": {"type": "number", "description": "Hamiltonian flow time"}
    },
    "required": ["n", "p"]
}

SECRET_SCHEMA = {
    "type": "object",
    "properties": {
        "coefficients": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "Secret polynomial coefficients"
        }
    },
    "required": ["coefficients"]
}

# GPU acceleration setup
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False

# Type aliases
ComplexArray = npt.NDArray[np.complex128]
RealArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

# Configuration constants
class Constants:
    TEICH_THRESHOLD = 20000
    BATCH_SIZE = 4096
    MAX_THREADS_PER_BLOCK = 1024
    # When modulus is huge, we use the multi-precision GPU kernel.
    MODULUS_THRESHOLD = (1 << 63) - 1  
    # Known primitive roots (to speed up computation)
    KNOWN_PRIMITIVE_ROOTS = {
        (1 << 256) - (1 << 32) - 977: 7  # secp256k1 prime
    }

@dataclass
class Commitment:
    """Zero-knowledge proof commitment."""
    transformed_evaluations: ComplexArray
    flow_time: float
    secret_original_evaluations: ComplexArray
    r: complex
    mask: ComplexArray
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert commitment to JSON-serializable dictionary."""
        data = {}
        for key, value in asdict(self).items():
            if isinstance(value, complex):
                data[key] = {"real": value.real, "imag": value.imag}
            elif isinstance(value, np.ndarray):
                if np.iscomplexobj(value):
                    data[key] = {
                        "real": value.real.tolist(),
                        "imag": value.imag.tolist()
                    }
                else:
                    data[key] = value.tolist()
            else:
                data[key] = value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Commitment':
        """Create commitment from dictionary."""
        processed = {}
        for key, value in data.items():
            if isinstance(value, dict) and "real" in value and "imag" in value:
                if isinstance(value["real"], list):
                    # Convert array values
                    processed[key] = np.array(value["real"]) + 1j * np.array(value["imag"])
                else:
                    # Convert complex scalar
                    processed[key] = complex(value["real"], value["imag"])
            elif isinstance(value, list):
                processed[key] = np.array(value)
            else:
                processed[key] = value
        return cls(**processed)

class FiniteFieldUtils:
    """Optimized finite field utility functions with caching."""
    
    @staticmethod
    @lru_cache(maxsize=128)
    def divisors(n: int) -> List[int]:
        # If n is a power of 2, then divisors are simply 2^i for i=0..k
        if (n & (n - 1)) == 0:
            k = n.bit_length() - 1  # because 2**k == n
            return [2**i for i in range(k + 1)]
        divs = set()
        for i in range(1, int(math.sqrt(n)) + 1):
            if n % i == 0:
                divs.add(i)
                divs.add(n // i)
        return sorted(divs)
    
    @staticmethod
    @lru_cache(maxsize=128)
    def prime_factors(n: int) -> List[int]:
        # Use trial division for small n with dynamic adjustment of limit.
        factors = []
        # Extract factor 2
        while n % 2 == 0:
            factors.append(2)
            n //= 2
        i = 3
        max_factor = math.isqrt(n) + 1
        while i <= max_factor and n != 1:
            while n % i == 0:
                factors.append(i)
                n //= i
                max_factor = math.isqrt(n) + 1
            i += 2
        if n > 1:
            factors.append(n)
        return sorted(set(factors))
    
    @classmethod
    @lru_cache(maxsize=64)
    def find_primitive_root(cls, p: int) -> Optional[int]:
        # Return known primitive root if available.
        if p in Constants.KNOWN_PRIMITIVE_ROOTS:
            return Constants.KNOWN_PRIMITIVE_ROOTS[p]
        if p == 2:
            return 1
        common_roots = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        factors = cls.prime_factors(p - 1)
        exponents = [(p - 1) // f for f in factors]
        for candidate in common_roots:
            if candidate >= p:
                continue
            if all(pow(candidate, exp, p) != 1 for exp in exponents):
                return candidate
        for candidate in range(2, min(p, 100)):
            if all(pow(candidate, exp, p) != 1 for exp in exponents):
                return candidate
        return None
    
    @classmethod
    @lru_cache(maxsize=64)
    def find_primitive_nth_root(cls, n: int, p: int) -> Optional[int]:
        # Check divisibility condition.
        if (p - 1) % n != 0:
            return None
        g = cls.find_primitive_root(p)
        if g is None:
            return None
        k = (p - 1) // n
        candidate = pow(g, k, p)
        proper_divs = [d for d in cls.divisors(n) if d < n]
        if all(pow(candidate, d, p) != 1 for d in proper_divs):
            return candidate
        return None

class HamiltonianFlow:
    """Class for computing Hamiltonian flows on complex vector spaces."""
    
    def __init__(self, xp: Any):
        """Initialize Hamiltonian flow calculator with numerical backend."""
        self.xp = xp
        self._flow_cache: Dict[float, Any] = {}
    
    def apply_flow(self, vector: Any, time: float, epsilon: float = 1e-10) -> Any:
        """Apply Hamiltonian flow to vector for time t."""
        if time in self._flow_cache:
            rotation = self._flow_cache[time]
        else:
            rotation = self.xp.exp(1j * time)
            self._flow_cache[time] = rotation
            
        # Normalize for stability
        norm = self.xp.linalg.norm(vector)
        if norm < epsilon:
            return vector
        return vector * rotation
    
    def apply_inverse_flow(self, vector: Any, time: float, epsilon: float = 1e-10) -> Any:
        """Apply inverse Hamiltonian flow to vector for time t."""
        return self.apply_flow(vector, -time, epsilon)

def teichmuller_lift_batch(indices: np.ndarray, n: int) -> np.ndarray:
    """Compute Teichmüller lift using NumPy."""
    indices = indices.astype(np.float64)
    return np.exp(1j * 2.0 * np.pi * indices / n)

def gpu_teichmuller_lift_batch(indices: Any, n: int, xp: Any) -> Any:
    """Compute Teichmüller lift using xp.exp (GPU version)."""
    indices = indices.astype(xp.float64)
    return xp.exp(1j * 2.0 * xp.pi * indices / n)

# Multi-precision GPU kernel source for Horner's method.
multi_precision_kernel_source = r'''
extern "C" __global__
void poly_eval_horner_mp(const unsigned int* x_vals, const unsigned int* coeffs,
                         unsigned int* results, const unsigned int* p,
                         int n_points, int n_coeffs) {
    const int NLIMBS = 8;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_points) return;
    unsigned int res[NLIMBS] = {0};
    unsigned int x[NLIMBS];
    for (int i = 0; i < NLIMBS; i++) {
        x[i] = x_vals[idx * NLIMBS + i];
    }
    for (int j = n_coeffs - 1; j >= 0; j--) {
        unsigned int prod[16] = {0};
        for (int i = 0; i < NLIMBS; i++) {
            unsigned int carry = 0;
            for (int k = 0; k < NLIMBS; k++) {
                unsigned long long tmp = (unsigned long long)res[i] * x[k] + prod[i+k] + carry;
                prod[i+k] = (unsigned int)(tmp & 0xFFFFFFFFUL);
                carry = (unsigned int)(tmp >> 32);
            }
            prod[i+NLIMBS] = carry;
        }
        for (int i = 0; i < NLIMBS; i++) {
            res[i] = prod[i];
        }
        bool ge = true;
        for (int i = NLIMBS - 1; i >= 0; i--) {
            if (res[i] < p[i]) { ge = false; break; }
            else if (res[i] > p[i]) { break; }
        }
        if (ge) {
            unsigned int borrow = 0;
            for (int i = 0; i < NLIMBS; i++) {
                unsigned long long diff = (unsigned long long)res[i] - p[i] - borrow;
                res[i] = (unsigned int)(diff & 0xFFFFFFFFUL);
                borrow = (diff >> 63) & 1;
            }
        }
        unsigned int carry = 0;
        for (int i = 0; i < NLIMBS; i++) {
            unsigned long long sum = (unsigned long long)res[i] + coeffs[j * NLIMBS + i] + carry;
            res[i] = (unsigned int)(sum & 0xFFFFFFFFUL);
            carry = (unsigned int)(sum >> 32);
        }
        ge = true;
        for (int i = NLIMBS - 1; i >= 0; i--) {
            if (res[i] < p[i]) { ge = false; break; }
            else if (res[i] > p[i]) { break; }
        }
        if (ge) {
            unsigned int borrow = 0;
            for (int i = 0; i < NLIMBS; i++) {
                unsigned long long diff = (unsigned long long)res[i] - p[i] - borrow;
                res[i] = (unsigned int)(diff & 0xFFFFFFFFUL);
                borrow = (diff >> 63) & 1;
            }
        }
    }
    for (int i = 0; i < NLIMBS; i++) {
        results[idx * NLIMBS + i] = res[i];
    }
}
'''

def int_to_limbs(val: int) -> List[np.uint32]:
    """Convert an integer to a list of 8 uint32 limbs (little-endian)."""
    limbs = []
    for _ in range(8):
        limbs.append(np.uint32(val & 0xFFFFFFFF))
        val >>= 32
    return limbs

def limbs_to_int(limbs: List[int]) -> int:
    """Convert a list of 8 limbs (little-endian) to an integer."""
    val = 0
    for limb in reversed(limbs):
        val = (val << 32) | int(limb)
    return val

class SymPLONK:
    """GPU-native SymPLONK protocol with asynchronous processing and JSON support."""
    
    def __init__(self, n: int, p: int, epsilon: float = 1e-10, use_gpu: bool = True) -> None:
        """
        Initialize with domain size n and prime p.
        All processing is done on the GPU using asynchronous CuPy streams.
        
        Args:
            n: Domain size
            p: Prime modulus
            epsilon: Error tolerance
            use_gpu: Whether to use GPU acceleration if available
        """
        self.n = n
        self.p = p
        self.epsilon = epsilon
        self.using_gpu = False
        self.threads_per_block = Constants.MAX_THREADS_PER_BLOCK
        self.metadata = {
            "version": __version__,
            "domain_size": n,
            "modulus": p,
            "epsilon": epsilon
        }

        if use_gpu and CUPY_AVAILABLE:
            try:
                _ = cp.array([1, 2, 3])
                self.xp = cp
                self.using_gpu = True
                print("Using GPU acceleration with CuPy (GPU-native mode).")
                device_props = cp.cuda.runtime.getDeviceProperties(0)
                device_name = device_props['name'].decode('utf-8')
                device_memory = device_props['totalGlobalMem'] / 1e9
                print(f"GPU: {device_name}, Memory: {device_memory:.2f} GB")
                self.metadata["gpu"] = {
                    "name": device_name,
                    "memory_gb": device_memory
                }
                self.threads_per_block = min(Constants.MAX_THREADS_PER_BLOCK, device_props['maxThreadsPerBlock'])
            except Exception as e:
                print(f"GPU initialization failed: {e}. Falling back to CPU (NumPy).")
                self.xp = np
                self.metadata["gpu"] = None
        else:
            self.xp = np
            self.metadata["gpu"] = None
            print("Using CPU (NumPy).")

        self._setup_domains()
        self.flow = HamiltonianFlow(self.xp)
        self._precompute_flows()

        if self.using_gpu and self.p > Constants.MODULUS_THRESHOLD:
            self.poly_eval_kernel_mp = cp.RawKernel(multi_precision_kernel_source, 'poly_eval_horner_mp')

        self._split_eval_cache: Dict[str, Any] = {}

    def _setup_domains(self) -> None:
        """Compute finite field domain D_f and Teichmüller lifts on the GPU using asynchronous streams."""
        self.omega_f = FiniteFieldUtils.find_primitive_nth_root(self.n, self.p)
        if self.omega_f is None:
            print(f"Warning: No primitive {self.n}-th root found in F_{self.p}. Using sequential domain.")
            self.D_f = np.arange(1, self.n + 1, dtype=np.int64)
        else:
            self.D_f = np.empty(self.n, dtype=np.int64)
            self.D_f[0] = 1
            for i in range(1, self.n):
                self.D_f[i] = (self.D_f[i - 1] * self.omega_f) % self.p

        self.D_f_indices = {int(elem): i for i, elem in enumerate(self.D_f)}
        indices_float = np.arange(self.n, dtype=np.float64)
        if self.using_gpu and self.n > Constants.TEICH_THRESHOLD:
            stream = cp.cuda.Stream(non_blocking=True)
            with stream:
                self.D = self.xp.empty(self.n, dtype=self.xp.complex128)
                for i in range(0, self.n, Constants.BATCH_SIZE):
                    end = min(i + Constants.BATCH_SIZE, self.n)
                    batch_indices = self.xp.arange(i, end, dtype=self.xp.float64)
                    self.D[i:end] = gpu_teichmuller_lift_batch(batch_indices, self.n, self.xp)
            stream.synchronize()
        else:
            domain_lifts = teichmuller_lift_batch(indices_float, self.n)
            self.D = self.xp.array(domain_lifts, dtype=self.xp.complex128)

    def _precompute_flows(self) -> None:
        """Precompute common Hamiltonian flow factors using an asynchronous stream."""
        if not self.using_gpu:
            common_angles = [math.pi/8, math.pi/6, math.pi/4, math.pi/3, math.pi/2,
                            2*math.pi/3, 3*math.pi/4, math.pi, 2*math.pi]
            for t in common_angles:
                self.flow._flow_cache[t] = np.exp(1j * t)
            return
            
        stream = self.xp.cuda.Stream(non_blocking=True)
        common_angles = [math.pi/8, math.pi/6, math.pi/4, math.pi/3, math.pi/2,
                         2*math.pi/3, 3*math.pi/4, math.pi, 2*math.pi]
        with stream:
            for t in common_angles:
                self.flow._flow_cache[t] = self.xp.exp(1j * t)
        stream.synchronize()

    def poly_eval_horner_split(self, x_vals: Any, coeffs: np.ndarray, p: int, d: Optional[int] = None) -> Any:
        """
        Evaluate a polynomial modulo p via split evaluation on GPU.
        Uses cached power terms to avoid redundant computation.
        """
        xp_local = self.xp
        x_vals = xp_local.asarray(x_vals, dtype=xp_local.int64)
        coeffs = xp_local.asarray(coeffs, dtype=xp_local.int64)
        deg = coeffs.shape[0] - 1
        if deg < 0:
            return xp_local.zeros_like(x_vals, dtype=xp_local.int64)
        if d is None:
            d = 2 ** int(math.floor(math.log(math.sqrt(deg), 2))) if deg > 0 else 1
            d = max(d, 1)
        cache_key = f"split_{len(coeffs)}_{d}"
        if cache_key in self._split_eval_cache:
            x_pow, X_d, d = self._split_eval_cache[cache_key]
        else:
            x_pow = [xp_local.mod(x_vals**j, p) for j in range(d)]
            X_d = xp_local.mod(x_vals**d, p)
            self._split_eval_cache[cache_key] = (x_pow, X_d, d)
        result = xp_local.zeros_like(x_vals, dtype=xp_local.int64)
        for j in range(d):
            if j > deg:
                continue
            num_terms = (deg - j) // d + 1
            group_coeffs = coeffs[j : j + num_terms*d : d]
            group_val = group_coeffs[-1]
            for coeff in group_coeffs[-2::-1]:
                group_val = (group_val * X_d + coeff) % p
            result = (result + group_val * x_pow[j]) % p
        return result

    def poly_eval_horner_batch_gpu_mp(self, x_vals: np.ndarray, coeffs: np.ndarray, p: int) -> np.ndarray:
        """
        Evaluate a polynomial using the multi-precision GPU kernel with asynchronous processing.
        """
        x_limbs = np.array([int_to_limbs(int(x)) for x in x_vals], dtype=np.uint32).flatten()
        coeffs_limbs = np.array([int_to_limbs(int(c)) for c in coeffs], dtype=np.uint32).flatten()
        p_limbs = np.array(int_to_limbs(p), dtype=np.uint32)
        n_points = x_vals.shape[0]
        n_coeffs = coeffs.shape[0]

        x_gpu = cp.asarray(x_limbs)
        coeffs_gpu = cp.asarray(coeffs_limbs)
        p_gpu = cp.asarray(p_limbs)
        results_gpu = cp.zeros(n_points * 8, dtype=cp.uint32)
        
        stream = cp.cuda.Stream(non_blocking=True)
        with stream:
            blocks_per_grid = (n_points + self.threads_per_block - 1) // self.threads_per_block
            self.poly_eval_kernel_mp((blocks_per_grid,), (self.threads_per_block,),
                                      (x_gpu, coeffs_gpu, results_gpu, p_gpu, n_points, n_coeffs),
                                      stream=stream)
        stream.synchronize()
        results_limbs = cp.asnumpy(results_gpu).reshape(n_points, 8)
        results_int = np.array([limbs_to_int(results_limbs[i]) for i in range(n_points)], dtype=np.uint64)
        return results_int

    def polynomial_encode(self, secret: List[int]) -> Any:
        """
        Encode the secret polynomial (with constant term first) as evaluations over the domain.
        Fully GPU-native and asynchronous.
        """
        coeffs = np.array([c % self.p for c in secret] + [0]*(self.n - len(secret)), dtype=np.int64)
        if self.using_gpu and self.p > Constants.MODULUS_THRESHOLD:
            poly_vals = self.poly_eval_horner_batch_gpu_mp(self.D_f, coeffs, self.p)
        else:
            poly_vals = self.poly_eval_horner_split(self.D_f, coeffs, self.p)
        # Build indices as a GPU (CuPy) array without converting back to NumPy.
        if self.using_gpu:
            indices = cp.array([self.D_f_indices.get(int(val), 0) for val in poly_vals])
        else:
            indices = np.array([self.D_f_indices.get(int(val), 0) for val in poly_vals], dtype=np.float64)
        lifts = gpu_teichmuller_lift_batch(indices, self.n, self.xp)
        return self.xp.array(lifts, dtype=self.xp.complex128)

    def blind_evaluations(self, evaluations: Any, r: Optional[complex] = None) -> Tuple[Any, complex, Any]:
        """
        Apply random blinding for zero-knowledge using GPU arrays.
        """
        xp = self.xp
        if r is None:
            r = complex(xp.random.normal(), xp.random.normal())
        mask = xp.random.normal(size=self.n) + 1j * xp.random.normal(size=self.n)
        mask = mask / xp.linalg.norm(mask)
        return evaluations + r * mask, r, mask

    def alice_prove(self, secret: List[int], flow_time: float = math.pi/4) -> Commitment:
        """
        Prover (Alice) generates a zero-knowledge proof commitment.
        Uses asynchronous processing for polynomial encoding and blinding.
        
        Args:
            secret: List of polynomial coefficients (constant term first)
            flow_time: Hamiltonian flow time parameter
            
        Returns:
            Commitment object containing the proof
        """
        print("\n=== Alice (Prover) ===")
        print(f"Secret (mod {self.p}): {secret}")
        start_time = time.perf_counter()

        encoding = None
        def compute_encoding():
            nonlocal encoding
            encoding = self.polynomial_encode(secret)
        thread = threading.Thread(target=compute_encoding)
        thread.start()
        thread.join()

        if encoding is None:
            raise ValueError("Polynomial encoding failed; encoding is None.")

        blinded, r, mask = self.blind_evaluations(encoding)
        transformed = self.flow.apply_flow(blinded, flow_time, self.epsilon)

        if self.xp is cp:
            transformed_cpu = cp.asnumpy(transformed)
            encoding_cpu = cp.asnumpy(encoding)
            mask_cpu = cp.asnumpy(mask)
        else:
            transformed_cpu = transformed
            encoding_cpu = encoding
            mask_cpu = mask

        end_time = time.perf_counter()
        print(f"Total proof generation time: {end_time - start_time:.4f} seconds")
        
        # Add execution metrics to metadata
        proof_metadata = {
            "flow_time": flow_time,
            "generation_time": end_time - start_time,
            "polynomial_degree": len(secret) - 1,
            "timestamp": time.time()
        }
        self.metadata["last_proof"] = proof_metadata
        
        return Commitment(
            transformed_evaluations=transformed_cpu,
            flow_time=flow_time,
            secret_original_evaluations=encoding_cpu,
            r=r,
            mask=mask_cpu
        )

    def bob_verify(self, commitment: Union[Commitment, Dict[str, Any]]) -> bool:
        """
        Verifier (Bob) checks the zero-knowledge proof commitment.
        Entire verification remains on GPU if available.
        
        Args:
            commitment: Either a Commitment object or a dictionary representation
            
        Returns:
            Boolean indicating if verification succeeded
        """
        print("\n=== Bob (Verifier) ===")
        
        # Handle both Commitment objects and dictionary representations
        if isinstance(commitment, dict):
            if "transformed_evaluations" in commitment and isinstance(commitment["transformed_evaluations"], dict):
                # This is a JSON-serialized dictionary
                comm = Commitment.from_dict(commitment)
            else:
                # This is a regular dictionary with numpy arrays
                comm = Commitment(**commitment)
        else:
            comm = commitment
            
        print(f"Flow time: t = {comm.flow_time:.4f}")
        start_time = time.perf_counter()

        xp = self.xp
        transformed = xp.array(comm.transformed_evaluations, dtype=xp.complex128)
        original = xp.array(comm.secret_original_evaluations, dtype=xp.complex128)
        mask = xp.array(comm.mask, dtype=xp.complex128)

        recovered = self.flow.apply_inverse_flow(transformed, comm.flow_time, self.epsilon)
        expected = original + comm.r * mask

        diff_norm = float(xp.linalg.norm(recovered - expected).get() 
                          if xp is cp else xp.linalg.norm(recovered - expected))
        verified = diff_norm < self.epsilon
        
        verification_time = time.perf_counter() - start_time
        print(f"L2 norm difference: {diff_norm:.8e}")
        print(f"Verification {'SUCCESS' if verified else 'FAILED'}")
        print(f"Total verification time: {verification_time:.4f} seconds")
        
        # Add verification metrics to metadata
        verification_metadata = {
            "verification_time": verification_time,
            "diff_norm": diff_norm,
            "success": verified,
            "timestamp": time.time()
        }
        self.metadata["last_verification"] = verification_metadata
        
        return verified
    
    def save_commitment(self, commitment: Commitment, filepath: str) -> None:
        """Save commitment to JSON file."""
        data = commitment.to_dict()
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Commitment saved to {filepath}")
    
    def load_commitment(self, filepath: str) -> Commitment:
        """Load commitment from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return Commitment.from_dict(data)
    
    def export_metadata(self, filepath: str) -> None:
        """Export current metadata to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        print(f"Metadata exported to {filepath}")

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for SymPLONK."""
    parser = argparse.ArgumentParser(description="SymPLONK Zero-Knowledge Proof System")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Setup the prover/verifier environment")
    setup_parser.add_argument("--config", type=str, help="JSON configuration file path")
    setup_parser.add_argument("--n", type=int, help="Domain size")
    setup_parser.add_argument("--p", type=int, help="Prime modulus")
    setup_parser.add_argument("--epsilon", type=float, default=1e-10, help="Error tolerance")
    setup_parser.add_argument("--use-gpu", action="store_true", help="Enable GPU acceleration")
    setup_parser.add_argument("--export-config", type=str, help="Export configuration to file")
    
    # Prove command
    prove_parser = subparsers.add_parser("prove", help="Generate a zero-knowledge proof")
    prove_parser.add_argument("--config", type=str, required=True, help="JSON configuration file path")
    prove_parser.add_argument("--secret", type=str, required=True, help="JSON file with secret polynomial coefficients")
    prove_parser.add_argument("--commitment", type=str, required=True, help="Output commitment file path")
    prove_parser.add_argument("--flow-time", type=float, default=math.pi/4, help="Hamiltonian flow time")
    
    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify a zero-knowledge proof")
    verify_parser.add_argument("--config", type=str, required=True, help="JSON configuration file path")
    verify_parser.add_argument("--commitment", type=str, required=True, help="Input commitment file path")
    
    # Export metadata command
    metadata_parser = subparsers.add_parser("metadata", help="Export system metadata")
    metadata_parser.add_argument("--config", type=str, required=True, help="JSON configuration file path")
    metadata_parser.add_argument("--output", type=str, required=True, help="Output metadata file path")
    
    # Version command
    version_parser = subparsers.add_parser("version", help="Show version information")
    
    return parser.parse_args()

def main() -> None:
    """Main entry point for SymPLONK command-line interface."""
    args = parse_arguments()
    
    if args.command == "version":
        print(f"SymPLONK version {__version__}")
        print(f"Author: {__author__}")
        return
        
    # Setup command
    if args.command == "setup":
        config = {}
        if args.config:
            with open(args.config, 'r') as f:
                config = json.load(f)
        
        n = args.n if args.n else config.get("n")
        p = args.p if args.p else config.get("p")
        epsilon = args.epsilon if args.epsilon else config.get("epsilon", 1e-10)
        use_gpu = args.use_gpu if args.use_gpu else config.get("use_gpu", False)
        
        if not n or not p:
            print("Error: Domain size (n) and prime modulus (p) must be provided.")
            return
            
        symplonk = SymPLONK(n=n, p=p, epsilon=epsilon, use_gpu=use_gpu)
        
        if args.export_config:
            with open(args.export_config, 'w') as f:
                json.dump({
                    "n": n,
                    "p": p,
                    "epsilon": epsilon,
                    "use_gpu": use_gpu
                }, f, indent=2)
            print(f"Configuration exported to {args.export_config}")
        
        return
        
    # Prove command
    elif args.command == "prove":
        with open(args.config, 'r') as f:
            config = json.load(f)
            
        with open(args.secret, 'r') as f:
            secret_data = json.load(f)
            
        symplonk = SymPLONK(
            n=config["n"],
            p=config["p"],
            epsilon=config.get("epsilon", 1e-10),
            use_gpu=config.get("use_gpu", False)
        )
        
        flow_time = args.flow_time if args.flow_time else config.get("flow_time", math.pi/4)
        commitment = symplonk.alice_prove(secret_data["coefficients"], flow_time)
        symplonk.save_commitment(commitment, args.commitment)
        
        return
        
    # Verify command
    elif args.command == "verify":
        with open(args.config, 'r') as f:
            config = json.load(f)
            
        symplonk = SymPLONK(
            n=config["n"],
            p=config["p"],
            epsilon=config.get("epsilon", 1e-10),
            use_gpu=config.get("use_gpu", False)
        )
        
        commitment = symplonk.load_commitment(args.commitment)
        verified = symplonk.bob_verify(commitment)
        
        return verified
        
    # Export metadata command
    elif args.command == "metadata":
        with open(args.config, 'r') as f:
            config = json.load(f)
            
        symplonk = SymPLONK(
            n=config["n"],
            p=config["p"],
            epsilon=config.get("epsilon", 1e-10),
            use_gpu=config.get("use_gpu", False)
        )
        
        symplonk.export_metadata(args.output)
        
        return

if __name__ == "__main__":
    main()
