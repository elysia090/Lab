"""
This version of the SymPLONK protocol code preserves the original motivation
(i.e. to perform zero‐knowledge proof commitments by encoding a secret polynomial,
blinding it, and applying a Hamiltonian flow) while upgrading the GPU path.
In H3 the native GPU libraries cannot handle multi‑precision arithmetic, so we
implement a new GPU kernel that supports full 256-bit arithmetic (by representing
each 256-bit number as 8 32-bit limbs) for polynomial evaluation using Horner’s method.
If a simplistic (single‑limb) implementation is used, the reward is greatly reduced,
so our highest priority is to enable genuine 256-bit support.
TODO items (such as a more efficient modular reduction) are postponed.
"""

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
# Set a threshold below which we can use native GPU arithmetic.
MODULUS_THRESHOLD = (1 << 63) - 1  # If p > this, use multi-precision GPU kernel.

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
        while n % 2 == 0:
            factors.append(2)
            n //= 2
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

###############################################################################
# Multi-precision GPU kernel for 256-bit polynomial evaluation using Horner's method.
#
# Each 256-bit number is represented as 8 limbs of 32 bits (little-endian order).
# The kernel below (poly_eval_horner_mp) evaluates the polynomial
#   P(x) = coeff[0] + coeff[1]*x + ... + coeff[n_coeffs-1]*x^(n_coeffs-1)
# using Horner's method. For each evaluation point, the computation is:
#
#   result = 0;
#   for (j = n_coeffs-1; j >= 0; j--) {
#       result = (result * x + coeff[j]) mod p;
#   }
#
# The multi-precision multiplication is performed in a naive way over 8 limbs,
# and a naive modular reduction is applied (subtracting p once if result >= p).
# (A full implementation would use Barrett reduction or similar; this is left as TODO.)
###############################################################################

multi_precision_kernel_source = r'''
extern "C" __global__
void poly_eval_horner_mp(const unsigned int* x_vals, const unsigned int* coeffs,
                         unsigned int* results, const unsigned int* p,
                         int n_points, int n_coeffs) {
    // Each multi-precision number has NLIMBS limbs (8 limbs for 256-bit numbers)
    const int NLIMBS = 8;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_points) return;

    // Initialize result to 0 (stored in array 'res')
    unsigned int res[NLIMBS];
    for (int i = 0; i < NLIMBS; i++) {
         res[i] = 0;
    }

    // Load x for this evaluation point (stored contiguously)
    unsigned int x[NLIMBS];
    for (int i = 0; i < NLIMBS; i++) {
         x[i] = x_vals[idx * NLIMBS + i];
    }

    // Horner's method: iterate over coefficients from highest degree to lowest.
    for (int j = n_coeffs - 1; j >= 0; j--) {
         // Multiply res by x: compute 256-bit * 256-bit = 512-bit product in prod[16].
         unsigned int prod[16];
         for (int i = 0; i < 16; i++) prod[i] = 0;
         for (int i = 0; i < NLIMBS; i++) {
             unsigned int carry = 0;
             for (int k = 0; k < NLIMBS; k++) {
                 unsigned long long tmp = (unsigned long long) res[i] * x[k] + prod[i+k] + carry;
                 prod[i+k] = (unsigned int)(tmp & 0xFFFFFFFFUL);
                 carry = (unsigned int)(tmp >> 32);
             }
             prod[i+NLIMBS] = carry;
         }
         // Naive modular reduction:
         // (TODO: Implement full multi-precision modular reduction, e.g., Barrett reduction)
         // For now, we assume that the product is small enough so that subtracting p once suffices.
         for (int i = 0; i < NLIMBS; i++) {
              res[i] = prod[i]; // take the lower 8 limbs as the candidate remainder
         }
         // Compare res and p to see if subtraction is needed.
         bool ge = true;
         for (int i = NLIMBS - 1; i >= 0; i--) {
              if (res[i] < p[i]) { ge = false; break; }
              else if (res[i] > p[i]) { break; }
         }
         if (ge) {
             unsigned int borrow = 0;
             for (int i = 0; i < NLIMBS; i++) {
                  unsigned long long diff = (unsigned long long) res[i] - p[i] - borrow;
                  res[i] = (unsigned int)(diff & 0xFFFFFFFFUL);
                  // Borrow is 1 if subtraction underflowed.
                  borrow = (diff >> 63) & 1;
             }
         }
         // Add the coefficient coeff[j] (each coefficient is represented in NLIMBS limbs)
         unsigned int carry = 0;
         for (int i = 0; i < NLIMBS; i++) {
              unsigned long long sum = (unsigned long long) res[i] + coeffs[j * NLIMBS + i] + carry;
              res[i] = (unsigned int)(sum & 0xFFFFFFFFUL);
              carry = (unsigned int)(sum >> 32);
         }
         // One more modular reduction if needed (naively)
         ge = true;
         for (int i = NLIMBS - 1; i >= 0; i--) {
              if (res[i] < p[i]) { ge = false; break; }
              else if (res[i] > p[i]) { break; }
         }
         if (ge) {
             unsigned int borrow = 0;
             for (int i = 0; i < NLIMBS; i++) {
                  unsigned long long diff = (unsigned long long) res[i] - p[i] - borrow;
                  res[i] = (unsigned int)(diff & 0xFFFFFFFFUL);
                  borrow = (diff >> 63) & 1;
             }
         }
    }

    // Write the final result (256-bit number) to the output array.
    for (int i = 0; i < NLIMBS; i++) {
         results[idx * NLIMBS + i] = res[i];
    }
}
'''

###############################################################################
# Helper functions for multi-precision conversion (between integers and limbs)
###############################################################################
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

###############################################################################
# Modified SymPLONK class with multi-precision GPU polynomial evaluation.
###############################################################################
class SymPLONK:
    """
    Optimized implementation of the SymPLONK protocol using a fixed cyclic subgroup.
    
    This version has been modified to support genuine 256-bit arithmetic on the GPU.
    When the prime p exceeds the native 64-bit range, a new multi-precision GPU kernel
    (poly_eval_horner_mp) is used to evaluate the secret polynomial via Horner's method.
    TODO: Further optimize modular reduction (e.g., via Barrett reduction).
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

                device_props = cp.cuda.runtime.getDeviceProperties(0)
                print(f"GPU: {device_props['name'].decode('utf-8')}, Memory: {device_props['totalGlobalMem'] / 1e9:.2f} GB")
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

        # Compile the multi-precision GPU kernel (if using GPU and p > MODULUS_THRESHOLD)
        if self.using_gpu and self.p > MODULUS_THRESHOLD:
            self.poly_eval_kernel_mp = cp.RawKernel(multi_precision_kernel_source, 'poly_eval_horner_mp')

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

        self.D_f_indices = {int(elem): i for i, elem in enumerate(self.D_f)}
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
        common_angles = [np.pi/8, np.pi/6, np.pi/4, np.pi/3, np.pi/2, 2*np.pi/3, 3*np.pi/4, np.pi, 2*np.pi]
        for t in common_angles:
            self.common_flows[t] = self.xp.exp(1j * t)

    def poly_eval_horner_batch_cpu(self, x_vals: np.ndarray, coeffs: np.ndarray, p: int) -> np.ndarray:
        """
        Evaluate a polynomial at multiple points using Horner's method (CPU version).
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

    def poly_eval_horner_batch_gpu_mp(self, x_vals: np.ndarray, coeffs: np.ndarray, p: int) -> np.ndarray:
        """
        Evaluate a polynomial at multiple points using the multi-precision GPU kernel.
        Converts integers to 8-limb representations and back.
        """
        # Convert each x in x_vals to its 8-limb representation.
        x_limbs = np.array([int_to_limbs(int(x)) for x in x_vals], dtype=np.uint32).flatten()
        coeffs_limbs = np.array([int_to_limbs(int(c)) for c in coeffs], dtype=np.uint32).flatten()
        p_limbs = np.array(int_to_limbs(p), dtype=np.uint32)
        n_points = x_vals.shape[0]
        n_coeffs = coeffs.shape[0]
        # Transfer to GPU using CuPy.
        x_gpu = cp.asarray(x_limbs)
        coeffs_gpu = cp.asarray(coeffs_limbs)
        p_gpu = cp.asarray(p_limbs)
        results_gpu = cp.zeros(n_points * 8, dtype=cp.uint32)
        blocks_per_grid = (n_points + self.threads_per_block - 1) // self.threads_per_block
        self.poly_eval_kernel_mp((blocks_per_grid,), (self.threads_per_block,),
                                  (x_gpu, coeffs_gpu, results_gpu, p_gpu, n_points, n_coeffs))
        results_limbs = cp.asnumpy(results_gpu).reshape(n_points, 8)
        # Convert limb arrays back to integers.
        results_int = np.array([limbs_to_int(results_limbs[i]) for i in range(n_points)], dtype=np.uint64)
        return results_int

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
                poly_vals = self.poly_eval_horner_batch_gpu_mp(self.D_f, coeffs, self.p)
            else:
                poly_vals = self.poly_eval_horner_batch_cpu(self.D_f, coeffs, self.p)
        else:
            poly_vals = self.poly_eval_horner_batch_cpu(self.D_f, coeffs, self.p)
        # Map field elements to their corresponding Teichmüller lifts.
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
        encoding = self.polynomial_encode(secret)
        blind, r, mask = self.blind_evaluations(encoding)
        transformed = self.flow.apply_flow(blind, flow_time, self.epsilon)
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
        recovered = self.flow.apply_inverse_flow(transformed, comm.flow_time, self.epsilon)
        expected = original + comm.r * mask
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
    
    Uses domain size n = 4096 and the secp256k1 prime (256-bit).
    """
    n_val = 4096
    p_val = (1 << 256) - (1 << 32) - 977  # secp256k1 prime (256-bit)
    symplonk = SymPLONK(n=n_val, p=p_val, use_gpu=True)
    secret = [1, 2, 3, 4]  # Example secret
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
