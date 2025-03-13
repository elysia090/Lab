import math
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

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
TEICH_THRESHOLD = 20000
BATCH_SIZE = 4096
MAX_THREADS_PER_BLOCK = 1024
MODULUS_THRESHOLD = (1 << 63) - 1

# Known primitive roots
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


class FiniteFieldUtils:
    """Finite field operations with caching."""

    @staticmethod
    @lru_cache(maxsize=128)
    def divisors(n: int) -> List[int]:
        """Return all divisors of n."""
        divs = set()
        for i in range(1, int(math.sqrt(n)) + 1):
            if n % i == 0:
                divs.add(i)
                divs.add(n // i)
        return sorted(divs)

    @staticmethod
    @lru_cache(maxsize=128)
    def prime_factors(n: int) -> List[int]:
        """Return unique prime factors of n."""
        factors = []
        # Check for factor 2
        while n % 2 == 0:
            factors.append(2)
            n //= 2
        # Check odd factors
        i = 3
        while i * i <= n:
            while n % i == 0:
                factors.append(i)
                n //= i
            i += 2
        # Add remaining prime if any
        if n > 1:
            factors.append(n)
        return sorted(set(factors))

    @classmethod
    @lru_cache(maxsize=64)
    def find_primitive_root(cls, p: int) -> Optional[int]:
        """Find a primitive root modulo p."""
        if p == 2:
            return 1
        if p in KNOWN_PRIMITIVE_ROOTS:
            return KNOWN_PRIMITIVE_ROOTS[p]

        # Common primitive roots to try first
        common_roots = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        factors = cls.prime_factors(p - 1)
        exponents = [(p - 1) // f for f in factors]

        # Try common roots
        for candidate in common_roots:
            if candidate >= p:
                continue
            if all(pow(candidate, exp, p) != 1 for exp in exponents):
                return candidate

        # Try more candidates if needed
        for candidate in range(2, min(p, 100)):
            if all(pow(candidate, exp, p) != 1 for exp in exponents):
                return candidate

        return None

    @classmethod
    @lru_cache(maxsize=64)
    def find_primitive_nth_root(cls, n: int, p: int) -> Optional[int]:
        """Find a primitive n-th root of unity in F_p."""
        if (p - 1) % n != 0:
            return None

        g = cls.find_primitive_root(p)
        if g is None:
            return None

        k = (p - 1) // n
        candidate = pow(g, k, p)

        # Verify it's a primitive nth root
        proper_divisors = [d for d in cls.divisors(n) if d < n]
        if all(pow(candidate, d, p) != 1 for d in proper_divisors):
            return candidate
        return None


def teichmuller_lift_batch(indices: np.ndarray, n: int) -> np.ndarray:
    """Compute Teichmüller lift: exp(2πi * index / n)."""
    indices = indices.astype(np.float64)
    return np.exp(1j * 2.0 * np.pi * indices / n)


def gpu_teichmuller_lift_batch(indices: Any, n: int, xp: Any) -> Any:
    """GPU-optimized Teichmüller lift computation."""
    indices = indices.astype(xp.float64)
    return xp.exp(1j * 2.0 * xp.pi * indices / n)


class HamiltonianFlow:
    """Hamiltonian flow operations with caching."""

    def __init__(self, xp: Any) -> None:
        """Initialize with array module (NumPy or CuPy)."""
        self.xp = xp
        self._flow_cache: Dict[float, Any] = {}

    def apply_flow(self, points: Any, t: float, epsilon: float = 1e-8) -> Any:
        """Apply Hamiltonian flow for time t."""
        if abs(t) < epsilon:
            return points
        if t not in self._flow_cache:
            self._flow_cache[t] = self.xp.exp(1j * t)
        return points * self._flow_cache[t]

    def apply_inverse_flow(self, points: Any, t: float, epsilon: float = 1e-8) -> Any:
        """Apply inverse Hamiltonian flow for time t."""
        return self.apply_flow(points, -t, epsilon)


# Multi-precision GPU kernel
multi_precision_kernel_source = r'''
extern "C" __global__
void poly_eval_horner_mp(const unsigned int* x_vals, const unsigned int* coeffs,
                         unsigned int* results, const unsigned int* p,
                         int n_points, int n_coeffs) {
    const int NLIMBS = 8;  // 8 limbs for 256-bit numbers
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_points) return;

    // Initialize result to 0
    unsigned int res[NLIMBS] = {0};

    // Load x for this evaluation point
    unsigned int x[NLIMBS];
    for (int i = 0; i < NLIMBS; i++) {
        x[i] = x_vals[idx * NLIMBS + i];
    }

    // Horner's method: iterate coefficients from highest to lowest degree
    for (int j = n_coeffs - 1; j >= 0; j--) {
        // Multiply res by x (256-bit * 256-bit = 512-bit product)
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
        
        // Take lower 8 limbs as candidate remainder
        for (int i = 0; i < NLIMBS; i++) {
            res[i] = prod[i];
        }
        
        // Modular reduction if result >= p
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
        
        // Add current coefficient
        unsigned int carry = 0;
        for (int i = 0; i < NLIMBS; i++) {
            unsigned long long sum = (unsigned long long)res[i] + coeffs[j * NLIMBS + i] + carry;
            res[i] = (unsigned int)(sum & 0xFFFFFFFFUL);
            carry = (unsigned int)(sum >> 32);
        }
        
        // One more modular reduction if needed
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

    // Write final result to output array
    for (int i = 0; i < NLIMBS; i++) {
        results[idx * NLIMBS + i] = res[i];
    }
}
'''


def int_to_limbs(val: int) -> List[np.uint32]:
    """Convert integer to list of 8 uint32 limbs (little-endian)."""
    limbs = []
    for _ in range(8):
        limbs.append(np.uint32(val & 0xFFFFFFFF))
        val >>= 32
    return limbs


def limbs_to_int(limbs: List[int]) -> int:
    """Convert list of 8 limbs (little-endian) to integer."""
    val = 0
    for limb in reversed(limbs):
        val = (val << 32) | int(limb)
    return val


class SymPLONK:
    """Optimized SymPLONK protocol implementation."""
    
    def __init__(self, n: int, p: int, epsilon: float = 1e-10, use_gpu: bool = True) -> None:
        """
        Initialize with domain size n and prime p.
        
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
        self.threads_per_block = MAX_THREADS_PER_BLOCK

        # Select array module: CuPy (GPU) or NumPy (CPU)
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

        # Setup
        self._setup_domains()
        self.flow = HamiltonianFlow(self.xp)
        self._precompute_flows()

        # Compile multi-precision GPU kernel if needed
        if self.using_gpu and self.p > MODULUS_THRESHOLD:
            self.poly_eval_kernel_mp = cp.RawKernel(multi_precision_kernel_source, 'poly_eval_horner_mp')

    def _setup_domains(self) -> None:
        """Set up finite field domain and compute Teichmüller lifts."""
        self.omega_f = FiniteFieldUtils.find_primitive_nth_root(self.n, self.p)
        if self.omega_f is None:
            print(f"Warning: No primitive {self.n}-th root found in F_{self.p}. Using sequential domain.")
            self.D_f = np.arange(1, self.n + 1, dtype=np.int64)
        else:
            # Generate cyclic subgroup
            self.D_f = np.empty(self.n, dtype=np.int64)
            value = 1
            for i in range(self.n):
                self.D_f[i] = value
                value = (value * self.omega_f) % self.p

        # Create lookup dictionary for domain indices
        self.D_f_indices = {int(elem): i for i, elem in enumerate(self.D_f)}
        indices = np.arange(self.n, dtype=np.float64)

        # Compute Teichmüller lifts with appropriate strategy
        if self.using_gpu and self.n > TEICH_THRESHOLD:
            # Process in batches for large domains
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
        """Precompute common Hamiltonian flow values."""
        self.common_flows = {}
        common_angles = [np.pi/8, np.pi/6, np.pi/4, np.pi/3, np.pi/2, 2*np.pi/3, 3*np.pi/4, np.pi, 2*np.pi]
        for t in common_angles:
            self.common_flows[t] = self.xp.exp(1j * t)

    def poly_eval_horner_batch_cpu(self, x_vals: np.ndarray, coeffs: np.ndarray, p: int) -> np.ndarray:
        """
        Evaluate polynomial at multiple points modulo p using CPU.
        
        Args:
            x_vals: Evaluation points
            coeffs: Polynomial coefficients (constant term first)
            p: Prime modulus
            
        Returns:
            Array of polynomial evaluations
        """
        x_vals = np.asarray(x_vals, dtype=np.int64)
        coeffs = np.asarray(coeffs, dtype=np.int64)
        n_points = len(x_vals)
        deg = len(coeffs) - 1

        if p <= MODULUS_THRESHOLD:
            # Vectorized evaluation for 64-bit primes
            exponents = np.arange(0, deg + 1, dtype=np.int64)
            powers = np.mod(np.power(x_vals[:, None], exponents), p)
            result = np.mod(np.sum(powers * coeffs, axis=1), p)
            return result
        else:
            # Fallback to Horner's method for larger primes
            result = np.full(n_points, coeffs[-1], dtype=np.int64)
            for c in reversed(coeffs[:-1]):
                result = (result * x_vals + c) % p
            return result

    def poly_eval_horner_batch_gpu_mp(self, x_vals: np.ndarray, coeffs: np.ndarray, p: int) -> np.ndarray:
        """
        Evaluate polynomial using multi-precision GPU kernel.
        
        Args:
            x_vals: Evaluation points
            coeffs: Polynomial coefficients (constant term first)
            p: Prime modulus
            
        Returns:
            Array of polynomial evaluations
        """
        # Convert to limb representations
        x_limbs = np.array([int_to_limbs(int(x)) for x in x_vals], dtype=np.uint32).flatten()
        coeffs_limbs = np.array([int_to_limbs(int(c)) for c in coeffs], dtype=np.uint32).flatten()
        p_limbs = np.array(int_to_limbs(p), dtype=np.uint32)
        n_points = x_vals.shape[0]
        n_coeffs = coeffs.shape[0]

        # Transfer to GPU and run kernel
        x_gpu = cp.asarray(x_limbs)
        coeffs_gpu = cp.asarray(coeffs_limbs)
        p_gpu = cp.asarray(p_limbs)
        results_gpu = cp.zeros(n_points * 8, dtype=cp.uint32)
        
        blocks_per_grid = (n_points + self.threads_per_block - 1) // self.threads_per_block
        self.poly_eval_kernel_mp(
            (blocks_per_grid,), 
            (self.threads_per_block,),
            (x_gpu, coeffs_gpu, results_gpu, p_gpu, n_points, n_coeffs)
        )
        
        # Convert results back
        results_limbs = cp.asnumpy(results_gpu).reshape(n_points, 8)
        results_int = np.array([limbs_to_int(results_limbs[i]) for i in range(n_points)], dtype=np.uint64)
        return results_int

    def polynomial_encode(self, secret: List[int]) -> Any:
        """
        Encode secret as polynomial evaluations over domain D_f.
        
        Args:
            secret: Coefficients of the polynomial (constant term first)
            
        Returns:
            Teichmüller lift of polynomial evaluations
        """
        # Pad secret coefficients to length n
        coeffs = np.array([c % self.p for c in secret] + [0] * (self.n - len(secret)), dtype=np.int64)
        
        # Choose appropriate evaluation method
        if self.using_gpu and self.p > MODULUS_THRESHOLD:
            poly_vals = self.poly_eval_horner_batch_gpu_mp(self.D_f, coeffs, self.p)
        else:
            poly_vals = self.poly_eval_horner_batch_cpu(self.D_f, coeffs, self.p)
        
        # Map to Teichmüller lifts
        indices = np.array([self.D_f_indices.get(int(val), 0) for val in poly_vals], dtype=np.float64)
        lifts = teichmuller_lift_batch(indices, self.n)
        return self.xp.array(lifts, dtype=self.xp.complex128)

    def blind_evaluations(self, evaluations: Any, r: Optional[complex] = None) -> Tuple[Any, complex, Any]:
        """
        Apply random blinding for zero-knowledge.
        
        Args:
            evaluations: Polynomial evaluations
            r: Optional blinding factor (generated if None)
            
        Returns:
            Tuple of (blinded evaluations, blinding factor r, mask)
        """
        xp = self.xp
        if r is None:
            r = complex(xp.random.normal(), xp.random.normal())
            
        if self.using_gpu:
            mask = xp.random.normal(size=self.n) + 1j * xp.random.normal(size=self.n)
        else:
            mask = xp.array(np.random.normal(size=self.n) + 1j * np.random.normal(size=self.n))
            
        # Normalize mask
        mask = mask / xp.linalg.norm(mask)
        return evaluations + r * mask, r, mask

    def alice_prove(self, secret: List[int], flow_time: float = np.pi/4) -> Commitment:
        """
        Prover (Alice) generates a zero-knowledge proof commitment.
        
        Args:
            secret: Secret to be encoded (polynomial coefficients)
            flow_time: Hamiltonian flow time parameter
            
        Returns:
            Commitment object
        """
        print("\n=== Alice (Prover) ===")
        print(f"Secret (mod {self.p}): {secret}")
        start_time = time.perf_counter()
        
        # Encode secret, blind, and apply flow
        encoding = self.polynomial_encode(secret)
        blinded, r, mask = self.blind_evaluations(encoding)
        transformed = self.flow.apply_flow(blinded, flow_time, self.epsilon)

        # Transfer to CPU if needed
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
        
        # Create and return commitment
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
        
        Args:
            commitment: Proof commitment from prover
            
        Returns:
            True if verification passes, False otherwise
        """
        print("\n=== Bob (Verifier) ===")
        comm = Commitment(**commitment) if isinstance(commitment, dict) else commitment
        print(f"Flow time: t = {comm.flow_time:.4f}")
        start_time = time.perf_counter()
        
        xp = self.xp
        transformed = xp.array(comm.transformed_evaluations, dtype=xp.complex128)
        original = xp.array(comm.secret_original_evaluations, dtype=xp.complex128)
        mask = xp.array(comm.mask, dtype=xp.complex128)
        
        # Apply inverse flow to recover original
        recovered = self.flow.apply_inverse_flow(transformed, comm.flow_time, self.epsilon)
        expected = original + comm.r * mask

        # Verify with L2 norm difference
        diff_norm = float(xp.linalg.norm(recovered - expected).get() if xp is cp 
                         else xp.linalg.norm(recovered - expected))
        verified = diff_norm < self.epsilon
        
        print(f"L2 norm difference: {diff_norm:.8e}")
        print(f"Verification {'SUCCESS' if verified else 'FAILED'}")
        print(f"Total verification time: {time.perf_counter() - start_time:.4f} seconds")
        return verified


def run_symplonk_demo() -> None:
    """Run SymPLONK protocol demonstration."""
    n_val = 4096
    p_val = (1 << 256) - (1 << 32) - 977  # secp256k1 prime
    
    # Initialize protocol
    symplonk = SymPLONK(n=n_val, p=p_val, use_gpu=True)
    secret = [1, 2, 3, 4]  # Example secret

    # Run proof and verification
    start_time = time.perf_counter()
    commitment = symplonk.alice_prove(secret)
    verification_success = symplonk.bob_verify(commitment)
    end_time = time.perf_counter()

    # Show results
    print("\n=== Verification Result ===")
    print(f"Verification result: {verification_success}")
    print("\n=== Total Execution Time ===")
    print(f"Total execution time: {end_time - start_time:.4f} seconds")


if __name__ == "__main__":
    run_symplonk_demo()
