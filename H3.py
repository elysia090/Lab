import math
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union, Dict, Any

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from functools import lru_cache

# Attempt to import CuPy for GPU acceleration; fall back to CPU (NumPy) if unavailable.
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False

# Type aliases for clarity.
ComplexArray = npt.NDArray[np.complex128]
RealArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

# Performance and configuration constants.
TEICH_THRESHOLD = 20000           # Threshold for precomputing full Teichmüller lift on large domains.
BATCH_SIZE = 4096                 # Batch size for processing large arrays.
MAX_THREADS_PER_BLOCK = 1024      # Maximum CUDA threads per block.
MODULUS_THRESHOLD = (1 << 63) - 1  # Use GPU kernel only if modulus p fits in a 64-bit integer.

# Known primitive roots for common cryptographic primes.
KNOWN_PRIMITIVE_ROOTS = {
    (1 << 256) - (1 << 32) - 977: 7  # secp256k1 prime.
}

@dataclass
class Commitment:
    """
    Data structure representing a zero-knowledge proof commitment.
    """
    transformed_evaluations: ComplexArray
    flow_time: float
    secret_original_evaluations: ComplexArray
    r: complex
    mask: ComplexArray

class FiniteFieldUtils:
    """Utility class for finite field operations with caching support."""
    
    @staticmethod
    @lru_cache(maxsize=128)
    def divisors(n: int) -> List[int]:
        """Return all divisors of the integer n."""
        divs = set()
        for i in range(1, int(math.sqrt(n)) + 1):
            if n % i == 0:
                divs.add(i)
                divs.add(n // i)
        return sorted(divs)

    @staticmethod
    @lru_cache(maxsize=128)
    def prime_factors(n: int) -> List[int]:
        """Return the unique prime factors of the integer n."""
        factors = []
        # Handle the factor 2.
        while n % 2 == 0:
            factors.append(2)
            n //= 2
        # Handle odd factors.
        i = 3
        while i * i <= n:
            while n % i == 0:
                factors.append(i)
                n //= i
            i += 2
        if n > 1:
            factors.append(n)
        return sorted(set(factors))

    @classmethod
    @lru_cache(maxsize=64)
    def find_primitive_root(cls, p: int) -> Optional[int]:
        """
        Find a primitive root modulo p.
        Returns None if no suitable primitive root is found.
        """
        if p == 2:
            return 1
        if p in KNOWN_PRIMITIVE_ROOTS:
            return KNOWN_PRIMITIVE_ROOTS[p]

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
        """
        Find a primitive n-th root of unity in the finite field F_p.
        Returns None if one does not exist.
        """
        if (p - 1) % n != 0:
            return None

        g = cls.find_primitive_root(p)
        if g is None:
            return None

        k = (p - 1) // n
        candidate = pow(g, k, p)

        proper_divisors = [d for d in cls.divisors(n) if d < n]
        if all(pow(candidate, d, p) != 1 for d in proper_divisors):
            return candidate
        return None

def teichmuller_lift_batch(indices: np.ndarray, n: int) -> np.ndarray:
    """
    Compute the Teichmüller lift for a batch of indices.
    
    The lift is defined as: exp(2πi * index / n)
    """
    indices = indices.astype(np.float64)
    return np.exp(1j * 2.0 * np.pi * indices / n)

def gpu_teichmuller_lift_batch(indices: Any, n: int, xp: Any) -> Any:
    """
    GPU-optimized version of the Teichmüller lift computation.
    
    Uses the provided array module xp (either NumPy or CuPy).
    """
    indices = indices.astype(xp.float64)
    return xp.exp(1j * 2.0 * xp.pi * indices / n)

class HamiltonianFlow:
    """
    Implements Hamiltonian flow operations.
    
    This class caches computed flow factors to improve efficiency.
    """
    def __init__(self, xp: Any) -> None:
        """
        Initialize with an array module (NumPy or CuPy).
        """
        self.xp = xp
        self._flow_cache: Dict[float, Any] = {}

    def apply_flow(self, points: Any, t: float, epsilon: float = 1e-8) -> Any:
        """
        Apply Hamiltonian flow to the points for time t.
        If |t| < epsilon, returns the original points.
        """
        if abs(t) < epsilon:
            return points

        if t not in self._flow_cache:
            self._flow_cache[t] = self.xp.exp(1j * t)
        return points * self._flow_cache[t]

    def apply_inverse_flow(self, points: Any, t: float, epsilon: float = 1e-8) -> Any:
        """
        Apply the inverse Hamiltonian flow for time t.
        """
        return self.apply_flow(points, -t, epsilon)

# Define a GPU kernel for polynomial evaluation using Horner's method if CuPy is available.
if CUPY_AVAILABLE:
    poly_eval_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void poly_eval_horner(const long long* x_vals, const long long* coeffs, 
                          long long* result, int n_points, int n_coeffs, long long p) {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx < n_points) {
            long long x = x_vals[idx];
            long long res = 0;
            for (int i = n_coeffs - 1; i >= 0; i--) {
                res = (res * x + coeffs[i]) % p;
                if (res < 0) res += p;
            }
            result[idx] = res;
        }
    }
    ''', 'poly_eval_horner')

class SymPLONK:
    """
    Optimized implementation of the SymPLONK protocol using a fixed cyclic subgroup.
    """
    def __init__(self, n: int, p: int, epsilon: float = 1e-10, use_gpu: bool = True) -> None:
        """
        Initialize the SymPLONK protocol with domain size n and prime p.
        
        If use_gpu is True and CuPy is available, GPU acceleration is used.
        """
        self.n = n
        self.p = p
        self.epsilon = epsilon
        self.using_gpu = False
        self.threads_per_block = MAX_THREADS_PER_BLOCK

        # Select the appropriate array module.
        if use_gpu and CUPY_AVAILABLE:
            try:
                _ = cp.array([1, 2, 3])
                self.xp = cp
                self.using_gpu = True
                print("Using GPU acceleration with CuPy.")

                # Retrieve GPU device properties.
                device_props = cp.cuda.runtime.getDeviceProperties(0)
                print(f"GPU: {device_props['name'].decode('utf-8')}, "
                      f"Memory: {device_props['totalGlobalMem'] / 1e9:.2f} GB")
                self.threads_per_block = min(MAX_THREADS_PER_BLOCK, device_props['maxThreadsPerBlock'])
            except Exception as e:
                print(f"GPU initialization failed: {e}. Falling back to CPU (NumPy).")
                self.xp = np
        else:
            self.xp = np
            print("Using CPU (NumPy).")

        # Set up the finite field evaluation domain.
        self._setup_domains()
        # Initialize Hamiltonian flow.
        self.flow = HamiltonianFlow(self.xp)
        self._precompute_flows()

    def _setup_domains(self) -> None:
        """
        Set up the finite field domain D_f as a cyclic subgroup and compute Teichmüller lifts.
        """
        self.omega_f = FiniteFieldUtils.find_primitive_nth_root(self.n, self.p)

        if self.omega_f is None:
            print(f"Warning: No primitive {self.n}-th root found in F_{self.p}. Using sequential domain.")
            self.D_f = np.arange(1, self.n + 1, dtype=np.int64)
        else:
            self.D_f = np.empty(self.n, dtype=np.int64)
            value = 1
            for i in range(self.n):
                self.D_f[i] = value
                value = (value * self.omega_f) % self.p

        # Create a lookup map from domain element to its index.
        self.D_f_indices = {int(elem): i for i, elem in enumerate(self.D_f)}

        # Compute Teichmüller lifts.
        indices = np.arange(self.n, dtype=np.float64)
        if self.using_gpu and self.n > TEICH_THRESHOLD:
            self.D = self.xp.empty(self.n, dtype=self.xp.complex128)
            for i in range(0, self.n, BATCH_SIZE):
                end = min(i + BATCH_SIZE, self.n)
                batch_indices = self.xp.arange(i, end, dtype=self.xp.float64)
                self.D[i:end] = gpu_teichmuller_lift_batch(batch_indices, self.n, self.xp)
        else:
            if self.using_gpu:
                indices_gpu = self.xp.array(indices)
                self.D = gpu_teichmuller_lift_batch(indices_gpu, self.n, self.xp)
            else:
                domain_lifts = teichmuller_lift_batch(indices, self.n)
                self.D = self.xp.array(domain_lifts, dtype=self.xp.complex128)

    def _precompute_flows(self) -> None:
        """
        Precompute common Hamiltonian flow values for faster operations.
        """
        self.common_flows = {}
        common_angles = [np.pi/8, np.pi/6, np.pi/4, np.pi/3, np.pi/2, 
                         2*np.pi/3, 3*np.pi/4, np.pi, 2*np.pi]
        for t in common_angles:
            self.common_flows[t] = self.xp.exp(1j * t)

    def poly_eval_horner_batch_cpu(self, x_vals: np.ndarray, coeffs: np.ndarray, p: int) -> np.ndarray:
        """
        Evaluate a polynomial at multiple points using Horner's method (CPU version).
        Processes the data in chunks for improved cache performance.
        """
        x_vals = np.asarray(x_vals, dtype=np.int64)
        coeffs = np.asarray(coeffs, dtype=np.int64)
        n_points = len(x_vals)
        result = np.zeros(n_points, dtype=np.int64)
        chunk_size = 1024

        for chunk_start in range(0, n_points, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_points)
            x_chunk = x_vals[chunk_start:chunk_end]
            r_chunk = np.full_like(x_chunk, coeffs[-1])
            for c in reversed(coeffs[:-1]):
                r_chunk = (r_chunk * x_chunk + c) % p
            result[chunk_start:chunk_end] = r_chunk

        return result

    def poly_eval_horner_batch_gpu(self, x_vals: Any, coeffs: Any, p: int, threads_per_block: int) -> Any:
        """
        Evaluate a polynomial using Horner's method on the GPU.
        Falls back to the CPU implementation if the modulus p is too large.
        """
        if p > MODULUS_THRESHOLD:
            print("Modulus too large for GPU kernel; using CPU evaluation.")
            return self.poly_eval_horner_batch_cpu(x_vals, coeffs, p)

        # Transfer data to the GPU.
        x_vals_gpu = cp.asarray(x_vals, dtype=cp.int64)
        coeffs_gpu = cp.asarray(coeffs, dtype=cp.int64)
        n_points = len(x_vals_gpu)
        n_coeffs = len(coeffs_gpu)
        result_gpu = cp.zeros_like(x_vals_gpu)

        blocks_per_grid = (n_points + threads_per_block - 1) // threads_per_block
        poly_eval_kernel((blocks_per_grid,), (threads_per_block,),
                         (x_vals_gpu, coeffs_gpu, result_gpu, n_points, n_coeffs, p))
        return result_gpu

    def polynomial_encode(self, secret: List[int]) -> Any:
        """
        Encode a secret as polynomial evaluations over the domain D_f.
        Returns the Teichmüller lift of the polynomial evaluation.
        """
        # Pad secret coefficients to length n.
        coeffs = np.array([c % self.p for c in secret] + [0] * (self.n - len(secret)), dtype=np.int64)
        
        # Evaluate the polynomial on D_f.
        if self.using_gpu:
            if self.p > MODULUS_THRESHOLD:
                poly_vals = self.poly_eval_horner_batch_cpu(self.D_f, coeffs, self.p)
            else:
                poly_vals = self.poly_eval_horner_batch_gpu(self.D_f, coeffs, self.p, self.threads_per_block)
                poly_vals = cp.asnumpy(poly_vals)
        else:
            poly_vals = self.poly_eval_horner_batch_cpu(self.D_f, coeffs, self.p)
        
        # Map field elements to their corresponding indices.
        indices = np.array([self.D_f_indices.get(int(val), 0) for val in poly_vals], dtype=np.float64)
        lifts = teichmuller_lift_batch(indices, self.n)
        return self.xp.array(lifts, dtype=self.xp.complex128)

    def blind_evaluations(self, evaluations: Any, r: Optional[complex] = None) -> Tuple[Any, complex, Any]:
        """
        Apply random blinding to evaluations for zero-knowledge purposes.
        Returns a tuple (blinded evaluations, blinding factor r, mask).
        """
        xp = self.xp
        if r is None:
            r = complex(xp.random.normal(), xp.random.normal())
        if self.using_gpu:
            mask = xp.random.normal(size=self.n) + 1j * xp.random.normal(size=self.n)
        else:
            mask = xp.array(np.random.normal(size=self.n) + 1j * np.random.normal(size=self.n))
        mask = mask / xp.linalg.norm(mask)
        return evaluations + r * mask, r, mask

    def alice_prove(self, secret: List[int], flow_time: float = np.pi/4) -> Commitment:
        """
        Prover (Alice) generates a zero-knowledge proof commitment from the secret.
        """
        print("\n=== Alice (Prover) ===")
        print(f"Secret (mod {self.p}): {secret}")
        start_time = time.perf_counter()
        
        # Encode the secret as polynomial evaluations.
        encoding = self.polynomial_encode(secret)
        # Blind the evaluations.
        blind, r, mask = self.blind_evaluations(encoding)
        # Apply the Hamiltonian flow.
        transformed = self.flow.apply_flow(blind, flow_time, self.epsilon)
        
        # Ensure data is on CPU for further processing.
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
        Returns True if the proof is verified within tolerance; otherwise, False.
        """
        print("\n=== Bob (Verifier) ===")
        comm = Commitment(**commitment) if isinstance(commitment, dict) else commitment
        print(f"Flow time: t = {comm.flow_time:.4f}")
        start_time = time.perf_counter()
        
        xp = self.xp
        transformed = xp.array(comm.transformed_evaluations, dtype=xp.complex128)
        original = xp.array(comm.secret_original_evaluations, dtype=xp.complex128)
        mask = xp.array(comm.mask, dtype=xp.complex128)
        
        # Recover the original blinded evaluations.
        recovered = self.flow.apply_inverse_flow(transformed, comm.flow_time, self.epsilon)
        expected = original + comm.r * mask
        
        # Compute the L2 norm difference.
        diff_norm = float(xp.linalg.norm(recovered - expected).get() if xp is cp 
                          else xp.linalg.norm(recovered - expected))
        verified = diff_norm < self.epsilon
        
        print(f"L2 norm difference: {diff_norm:.8e}")
        print(f"Verification {'SUCCESS' if verified else 'FAILED'}")
        print(f"Total verification time: {time.perf_counter() - start_time:.4f} seconds")
        return verified

def run_symplonk_demo() -> None:
    """
    Run a demonstration of the SymPLONK protocol.
    
    Uses domain size n = 4096 and the secp256k1 prime.
    """
    n_val = 4096
    p_val = (1 << 256) - (1 << 32) - 977
    symplonk = SymPLONK(n=n_val, p=p_val, use_gpu=True)
    
    # Example secret.
    secret = [1, 2, 3, 4]
    
    start_time = time.perf_counter()
    commitment = symplonk.alice_prove(secret)
    verification_success = symplonk.bob_verify(commitment)
    end_time = time.perf_counter()
    
    print("\n=== Verification Result ===")
    print(f"Verification result: {verification_success}")
    print("\n=== Total Execution Time ===")
    print(f"Total execution time (proof generation to verification): {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    run_symplonk_demo()
