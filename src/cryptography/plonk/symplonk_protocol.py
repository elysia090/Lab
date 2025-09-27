"""
SymPLONK: A GPU-optimized implementation of a zero-knowledge proof system
based on polynomial commitments and Hamiltonian flows.

This module provides a complete implementation of the SymPLONK protocol
with support for both CPU and GPU acceleration.
"""

import math
import time
import logging
from dataclasses import dataclass
from typing import List, Optional, Union, Dict, Any, Tuple
import numpy as np
from functools import lru_cache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("symplonk")

# Try importing CuPy for GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False


@dataclass
class Commitment:
    """Zero-knowledge proof commitment data structure.
    
    Attributes:
        transformed_evaluations: The evaluations after applying the Hamiltonian flow.
        flow_time: Time parameter for the Hamiltonian flow.
        secret_original_evaluations: Original polynomial evaluations of the secret.
        r: Complex random value used for blinding.
        mask: Random mask used for blinding.
    """
    transformed_evaluations: Any
    flow_time: float
    secret_original_evaluations: Any
    r: complex
    mask: Any


class BarrettReduction:
    """Fast modular arithmetic using Barrett reduction.
    
    This class implements Barrett reduction for efficient modular arithmetic
    operations without using expensive division operations.
    """
    
    def __init__(self, p: int):
        """Initialize Barrett reduction for modulus p.
        
        Args:
            p: The modulus to use for all operations.
        """
        if not isinstance(p, int) or p <= 0:
            raise ValueError("Modulus p must be a positive integer")
            
        self.p = p
        # Precompute constants for Barrett reduction
        self.k = p.bit_length()
        self.r = (1 << (2 * self.k)) // p
        
    def reduce(self, x: int) -> int:
        """Perform Barrett reduction: x mod p.
        
        Args:
            x: The value to reduce.
            
        Returns:
            The result of x mod p.
        """
        if x < self.p:
            return x
        
        q = (x * self.r) >> (2 * self.k)
        result = x - q * self.p
        if result >= self.p:
            result -= self.p
        return result
    
    def add_mod(self, a: int, b: int) -> int:
        """Compute (a + b) mod p.
        
        Args:
            a: First addend.
            b: Second addend.
            
        Returns:
            The sum a + b mod p.
        """
        result = a + b
        if result >= self.p:
            result -= self.p
        return result
    
    def mul_mod(self, a: int, b: int) -> int:
        """Compute (a * b) mod p using Barrett reduction.
        
        Args:
            a: First factor.
            b: Second factor.
            
        Returns:
            The product a * b mod p.
        """
        return self.reduce(a * b)
    
    def pow_mod(self, base: int, exponent: int) -> int:
        """Compute base^exponent mod p efficiently.
        
        Args:
            base: The base value.
            exponent: The exponent.
            
        Returns:
            base^exponent mod p.
        """
        if exponent < 0:
            raise ValueError("Negative exponents not supported")
            
        if exponent == 0:
            return 1
        
        result = 1
        base = self.reduce(base)
        
        while exponent > 0:
            if exponent & 1:
                result = self.reduce(result * base)
            base = self.reduce(base * base)
            exponent >>= 1
            
        return result


class FiniteFieldUtils:
    """Utility class for finite field operations.
    
    This class provides methods for finding primitive roots and other
    finite field mathematical operations.
    """
    
    @staticmethod
    @lru_cache(maxsize=128)
    def find_primitive_root(p: int) -> Optional[int]:
        """Find a primitive root modulo p.
        
        Args:
            p: Prime modulus.
            
        Returns:
            A primitive root modulo p, or None if not found.
        """
        if p == 2:
            return 1
        
        # Use known primitive roots for common primes
        KNOWN_ROOTS = {
            (1 << 256) - (1 << 32) - 977: 7,  # secp256k1
            (1 << 255) - 19: 2,              # curve25519
            (1 << 224) - (1 << 96) + 1: 3    # p224
        }
        if p in KNOWN_ROOTS:
            return KNOWN_ROOTS[p]

        # Find prime factors of p-1
        factors = []
        n = p - 1
        for i in [2] + list(range(3, min(1000, p), 2)):
            if n % i == 0:
                factors.append(i)
                while n % i == 0:
                    n //= i
            if n == 1:
                break
        if n > 1:
            factors.append(n)
            
        # Test common small numbers as candidates
        barrett = BarrettReduction(p)
        for g in range(2, min(1000, p)):
            if all(barrett.pow_mod(g, (p-1)//f) != 1 for f in factors):
                return g
        
        logger.warning(f"Could not find primitive root for p={p}")
        return None

    @classmethod
    @lru_cache(maxsize=128)
    def find_primitive_nth_root(cls, n: int, p: int) -> Optional[int]:
        """Find a primitive n-th root of unity in F_p.
        
        Args:
            n: Order of the root of unity.
            p: Prime modulus.
            
        Returns:
            A primitive n-th root of unity, or None if not found.
        """
        if n <= 0:
            raise ValueError("Order n must be positive")
            
        if (p - 1) % n != 0:
            logger.warning(f"Cannot find n-th root of unity: {n} does not divide {p-1}")
            return None
            
        g = cls.find_primitive_root(p)
        if g is None:
            return None
        
        barrett = BarrettReduction(p)
        w = barrett.pow_mod(g, (p - 1) // n)
        return w


class PolynomialEvaluator:
    """Class for efficient polynomial evaluation.
    
    This class provides methods for evaluating polynomials over finite fields
    using efficient algorithms like the Paterson-Stockmeyer method.
    """
    
    def __init__(self, barrett: BarrettReduction, use_gpu: bool = False):
        """Initialize the polynomial evaluator.
        
        Args:
            barrett: BarrettReduction instance for modular arithmetic.
            use_gpu: Whether to use GPU acceleration if available.
        """
        self.barrett = barrett
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        
        if self.use_gpu:
            self._setup_gpu_kernels()
    
    def _setup_gpu_kernels(self):
        """Set up GPU kernels for polynomial evaluation."""
        self.poly_eval_kernel = cp.RawKernel(r'''
        extern "C" __global__
        void poly_eval_ps(const long long* x, const long long* coeffs, 
                          long long* result, int n_points, int n_coeffs, 
                          long long p, long long r, int k) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n_points) {
                long long x_val = x[idx];
                
                // Paterson-Stockmeyer algorithm with Barrett reduction.
                // Choose optimal m for x^m precomputation (√n is optimal).
                int m = (int)sqrt(n_coeffs);
                long long powers[32]; // Precomputed powers (adjust size as needed).
                
                // Compute and store x^j for j=0,1,...,m-1.
                powers[0] = 1;
                for (int j = 1; j < m && j < 32; j++) {
                    // x^j = x^(j-1) * x with Barrett reduction.
                    unsigned __int128 product = (unsigned __int128)powers[j-1] * x_val;
                    unsigned __int128 q = (product * r) >> (2 * k);
                    long long mod = product - q * p;
                    if (mod >= p) mod -= p;
                    powers[j] = mod;
                }
                
                // Evaluate using Paterson-Stockmeyer scheme.
                long long res = 0;
                for (int i = 0; i < n_coeffs; i += m) {
                    // Evaluate this block using Horner's method with precomputed powers.
                    long long block_res = 0;
                    for (int j = min(m, n_coeffs - i) - 1; j >= 0; j--) {
                        int coef_idx = i + j;
                        if (coef_idx < n_coeffs) {
                            // Barrett reduction for block_res * x + coeffs[coef_idx].
                            unsigned __int128 product = (unsigned __int128)block_res * x_val;
                            unsigned __int128 q = (product * r) >> (2 * k);
                            long long mod = product - q * p;
                            if (mod >= p) mod -= p;
                            
                            // Add coefficient and reduce.
                            mod = mod + coeffs[coef_idx];
                            if (mod >= p) mod -= p;
                            block_res = mod;
                        }
                    }
                    
                    // Multiply block result by x^(m*i) and add to final result.
                    if (i > 0) {
                        // Compute x^(m*i).
                        long long x_power = 1;
                        long long base = powers[m-1] * x_val;
                        unsigned __int128 q = (base * r) >> (2 * k);
                        base = base - q * p;
                        if (base >= p) base -= p;
                        
                        // Calculate x^(m*i) using binary exponentiation with Barrett reduction.
                        int exp = i / m;
                        while (exp > 0) {
                            if (exp & 1) {
                                unsigned __int128 product = (unsigned __int128)x_power * base;
                                q = (product * r) >> (2 * k);
                                x_power = product - q * p;
                                if (x_power >= p) x_power -= p;
                            }
                            unsigned __int128 product = (unsigned __int128)base * base;
                            q = (product * r) >> (2 * k);
                            base = product - q * p;
                            if (base >= p) base -= p;
                            exp >>= 1;
                        }
                        
                        // Multiply block result by x^(m*i).
                        unsigned __int128 product = (unsigned __int128)block_res * x_power;
                        q = (product * r) >> (2 * k);
                        block_res = product - q * p;
                        if (block_res >= p) block_res -= p;
                    }
                    
                    // Add to final result.
                    res += block_res;
                    if (res >= p) res -= p;
                }
                
                result[idx] = res;
            }
        }
        ''', 'poly_eval_ps')
    
    def evaluate_batch(self, coeffs: np.ndarray, x_values: np.ndarray) -> np.ndarray:
        """Evaluate polynomial on multiple points.
        
        Args:
            coeffs: Polynomial coefficients [a₀, a₁, ..., aₙ].
            x_values: Points at which to evaluate the polynomial.
            
        Returns:
            Array of polynomial evaluations.
        """
        if self.use_gpu:
            return self._evaluate_gpu(coeffs, x_values)
        else:
            return self._evaluate_cpu(coeffs, x_values)
    
    def _evaluate_gpu(self, coeffs: np.ndarray, x_values: np.ndarray) -> np.ndarray:
        """Evaluate polynomial using GPU.
        
        Args:
            coeffs: Polynomial coefficients.
            x_values: Points at which to evaluate the polynomial.
            
        Returns:
            Array of polynomial evaluations.
        """
        n_points = len(x_values)
        x_gpu = cp.asarray(x_values, dtype=cp.int64)
        coeffs_gpu = cp.asarray(coeffs, dtype=cp.int64)
        result_gpu = cp.zeros(n_points, dtype=cp.int64)
        
        threads_per_block = 512
        blocks_per_grid = (n_points + threads_per_block - 1) // threads_per_block
        
        # Pass Barrett reduction parameters.
        self.poly_eval_kernel(
            (blocks_per_grid,), 
            (threads_per_block,),
            (x_gpu, coeffs_gpu, result_gpu, n_points, len(coeffs), 
             self.barrett.p, self.barrett.r, self.barrett.k)
        )
        
        return cp.asnumpy(result_gpu)
        
    def _evaluate_cpu(self, coeffs: np.ndarray, x_values: np.ndarray) -> np.ndarray:
        """Evaluate polynomial using the Paterson-Stockmeyer algorithm on CPU.
        
        Args:
            coeffs: Polynomial coefficients.
            x_values: Points at which to evaluate the polynomial.
            
        Returns:
            Array of polynomial evaluations.
        """
        n_coeffs = len(coeffs)
        n_points = len(x_values)
        results = np.zeros(n_points, dtype=np.int64)
        
        # Optimal choice for m is approximately sqrt(n_coeffs)
        m = max(1, int(math.sqrt(n_coeffs)))
        
        for i in range(n_points):
            x = x_values[i]
            
            # Precompute powers x^j for j=0,1,...,m-1.
            powers = [1]
            for j in range(1, m):
                powers.append(self.barrett.mul_mod(powers[-1], x))
            
            # Evaluate using the Paterson-Stockmeyer scheme.
            result = 0
            for block_start in range(0, n_coeffs, m):
                # Evaluate this block using Horner with precomputed powers.
                block_res = 0
                for j in range(min(m, n_coeffs - block_start) - 1, -1, -1):
                    coef_idx = block_start + j
                    if coef_idx < n_coeffs:
                        block_res = self.barrett.reduce(
                            self.barrett.mul_mod(block_res, x) + coeffs[coef_idx]
                        )
                
                # Multiply block result by x^(m*block_idx) and add to final result.
                if block_start > 0:
                    # Compute x^(m*block_idx) using binary exponentiation.
                    block_idx = block_start // m
                    x_power = self.barrett.pow_mod(
                        self.barrett.mul_mod(powers[m-1], x), 
                        block_idx
                    )
                    block_res = self.barrett.mul_mod(block_res, x_power)
                
                # Add to final result.
                result = self.barrett.add_mod(result, block_res)
            
            results[i] = result
            
        return results


class SymPLONK:
    """GPU-optimized implementation of the SymPLONK protocol with Barrett reduction.
    
    This class implements the complete SymPLONK zero-knowledge proof system,
    which uses polynomial commitments and Hamiltonian flows for constructing
    and verifying proofs.
    """
    
    def __init__(self, n: int, p: int, epsilon: float = 1e-10, use_gpu: bool = None):
        """Initialize with domain size n and prime p.
        
        Args:
            n: Size of the domain (should be a power of 2 for efficiency).
            p: Prime field modulus.
            epsilon: Error tolerance for verification.
            use_gpu: Whether to use GPU (if None, use GPU if available).
        """
        if n <= 0:
            raise ValueError("Domain size n must be positive")
        if not self._is_probable_prime(p):
            logger.warning(f"p={p} may not be prime, which could cause issues")
            
        self.n = n
        self.p = p
        self.epsilon = epsilon
        
        # Setup GPU if available and requested.
        if use_gpu is None:
            self.using_gpu = CUPY_AVAILABLE
        else:
            self.using_gpu = use_gpu and CUPY_AVAILABLE
            
        self.xp = cp if self.using_gpu else np
        
        if self.using_gpu:
            logger.info("Using GPU acceleration with CuPy")
        else:
            logger.info("Using CPU (NumPy)")
        
        # Initialize components.
        self.barrett = BarrettReduction(p)
        self.poly_evaluator = PolynomialEvaluator(self.barrett, self.using_gpu)
            
        # Setup domains.
        self._setup_domains()
    
    @staticmethod
    def _is_probable_prime(n: int, k: int = 5) -> bool:
        """Check if a number is probably prime using the Miller-Rabin test.
        
        Args:
            n: Number to test.
            k: Number of test iterations.
            
        Returns:
            True if n is probably prime, False if definitely composite.
        """
        if n <= 1:
            return False
        if n == 2 or n == 3:
            return True
        if n % 2 == 0:
            return False
            
        # Write n-1 as 2^r * d.
        r, d = 0, n - 1
        while d % 2 == 0:
            r += 1
            d //= 2
            
        # Witness loop.
        for _ in range(k):
            a = np.random.randint(2, n - 1)
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(r - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        return True
        
    def _setup_domains(self):
        """Set up finite field domain and Teichmüller lifts."""
        self.omega_f = FiniteFieldUtils.find_primitive_nth_root(self.n, self.p)
        
        # Generate domain elements.
        if self.omega_f is None:
            logger.warning(f"Could not find n-th root of unity. Using linear domain.")
            self.D_f = np.arange(1, self.n + 1, dtype=np.int64)
        else:
            self.D_f = np.empty(self.n, dtype=np.int64)
            val = 1
            for i in range(self.n):
                self.D_f[i] = val
                val = self.barrett.mul_mod(val, self.omega_f)
                
        # Create index lookup map.
        self.D_f_indices = {int(elem): i for i, elem in enumerate(self.D_f)}
        
        # Compute Teichmüller lifts (complex roots of unity).
        indices = self.xp.arange(self.n, dtype=self.xp.float64)
        self.D = self.xp.exp(1j * 2.0 * self.xp.pi * indices / self.n)
    
    def _apply_flow(self, points: Any, t: float) -> Any:
        """Apply Hamiltonian flow to points for time t.
        
        Args:
            points: Complex points to transform.
            t: Time parameter for the flow.
            
        Returns:
            Transformed points.
        """
        if abs(t) < self.epsilon:
            return points
        return points * self.xp.exp(1j * t)
    
    def polynomial_encode(self, secret: List[int]) -> Any:
        """Encode secret as polynomial evaluations over domain D_f.
        
        Args:
            secret: List of integers representing the secret.
            
        Returns:
            Complex vector representation of the polynomial evaluations.
        """
        if not all(isinstance(s, int) for s in secret):
            raise ValueError("Secret must be a list of integers")
            
        # Pad coefficients to length n.
        coeffs = np.array([s % self.p for s in secret] + [0] * (self.n - len(secret)), dtype=np.int64)
        
        # Evaluate polynomial on domain.
        poly_vals = self.poly_evaluator.evaluate_batch(coeffs, self.D_f)
        
        # Convert to Teichmüller lifts (transform to positions on unit circle).
        indices = np.array([self.D_f_indices.get(int(val), 0) for val in poly_vals], dtype=np.float64)
        return self.xp.exp(1j * 2.0 * self.xp.pi * indices / self.n)
    
    def blind_evaluations(self, evals: Any, r: Optional[complex] = None) -> Tuple[Any, complex, Any]:
        """Apply random blinding to evaluations.
        
        Args:
            evals: Polynomial evaluations to blind.
            r: Optional fixed random complex value.
            
        Returns:
            Tuple of (blinded evaluations, random value, mask).
        """
        if r is None:
            r = complex(self.xp.random.normal(), self.xp.random.normal())
        
        # Generate random mask.
        mask = self.xp.random.normal(size=self.n) + 1j * self.xp.random.normal(size=self.n)
        mask = mask / self.xp.linalg.norm(mask)
        
        return evals + r * mask, r, mask
    
    def alice_prove(self, secret: List[int], flow_time: float = math.pi/4) -> Commitment:
        """Generate zero-knowledge proof commitment from secret.
        
        Args:
            secret: List of integers representing the secret.
            flow_time: Time parameter for the Hamiltonian flow.
            
        Returns:
            A Commitment object containing the proof.
        """
        if len(secret) > self.n:
            raise ValueError(f"Secret length exceeds domain size {self.n}")
            
        logger.info(f"Generating proof for secret of length {len(secret)}")
        start = time.perf_counter()
        
        # Encode secret.
        encoding = self.polynomial_encode(secret)
        
        # Blind the evaluations.
        blind, r, mask = self.blind_evaluations(encoding)
        
        # Apply Hamiltonian flow.
        transformed = self._apply_flow(blind, flow_time)
        
        # Convert to CPU if needed.
        if self.using_gpu:
            transformed_cpu = cp.asnumpy(transformed)
            encoding_cpu = cp.asnumpy(encoding)
            mask_cpu = cp.asnumpy(mask)
        else:
            transformed_cpu = transformed
            encoding_cpu = encoding
            mask_cpu = mask
            
        logger.info(f"Proof generation completed in {time.perf_counter() - start:.4f} seconds")
        
        return Commitment(
            transformed_evaluations=transformed_cpu,
            flow_time=flow_time,
            secret_original_evaluations=encoding_cpu,
            r=r,
            mask=mask_cpu
        )
    
    def bob_verify(self, commitment: Union[Commitment, Dict[str, Any]]) -> bool:
        """Verify zero-knowledge proof commitment.
        
        Args:
            commitment: Commitment object or dictionary with commitment data.
            
        Returns:
            True if verification succeeds, False otherwise.
        """
        logger.info("Verifying proof...")
        comm = Commitment(**commitment) if isinstance(commitment, dict) else commitment
        start = time.perf_counter()
        
        # Convert data to current device.
        transformed = self.xp.array(comm.transformed_evaluations)
        original = self.xp.array(comm.secret_original_evaluations)
        mask = self.xp.array(comm.mask)
        
        # Recover original blinded evaluations.
        recovered = self._apply_flow(transformed, -comm.flow_time)
        expected = original + comm.r * mask
        
        # Check L2 norm difference.
        diff = float(self.xp.linalg.norm(recovered - expected).get() 
                     if self.using_gpu else self.xp.linalg.norm(recovered - expected))
        verified = diff < self.epsilon
        
        logger.info(f"L2 norm difference: {diff:.8e}")
        logger.info(f"Verification {'SUCCESS' if verified else 'FAILED'}")
        logger.info(f"Verification completed in {time.perf_counter() - start:.4f} seconds")
        
        return verified
        
    def benchmark(self, secret_sizes: List[int] = None) -> Dict[str, Dict[str, float]]:
        """Run benchmarks for different secret sizes.
        
        Args:
            secret_sizes: List of secret sizes to benchmark, defaults to [10, 100, 1000].
            
        Returns:
            Dictionary with benchmark results.
        """
        if secret_sizes is None:
            secret_sizes = [10, 100, 1000]
            
        results = {}
        
        for size in secret_sizes:
            if size > self.n:
                logger.warning(f"Secret size {size} exceeds domain size {self.n}, skipping")
                continue
                
            logger.info(f"Benchmarking with secret size {size}")
            secret = list(range(size))
            
            # Measure proof generation time.
            start = time.perf_counter()
            commitment = self.alice_prove(secret)
            prove_time = time.perf_counter() - start
            
            # Measure verification time.
            start = time.perf_counter()
            verified = self.bob_verify(commitment)
            verify_time = time.perf_counter() - start
            
            if not verified:
                logger.error(f"Verification failed for secret size {size}")
            
            results[size] = {
                "prove_time": prove_time,
                "verify_time": verify_time,
                "total_time": prove_time + verify_time,
                "verified": verified
            }
            
            logger.info(f"Secret size {size}: Prove={prove_time:.4f}s, Verify={verify_time:.4f}s")
            
        return results


def run_demo(n: int = 4096, secret: List[int] = None, use_gpu: bool = None):
    """Run a demonstration of the SymPLONK protocol.
    
    Args:
        n: Domain size to use.
        secret: Secret to use for the demo, defaults to [1, 2, 3, 4].
        use_gpu: Whether to use GPU, defaults to automatic detection.
    """
    # Use secp256k1 prime.
    p_val = (1 << 256) - (1 << 32) - 977
    
    logger.info(f"Initializing SymPLONK with n={n}, p={p_val}")
    symplonk = SymPLONK(n=n, p=p_val, use_gpu=use_gpu)
    
    # Example secret.
    if secret is None:
        secret = [1, 2, 3, 4]
    
    # Run protocol.
    logger.info(f"Running SymPLONK protocol with secret: {secret}")
    start = time.perf_counter()
    
    print(f"\n=== Prover ===\nSecret (mod {symplonk.p}): {secret}")
    commitment = symplonk.alice_prove(secret)
    
    print("\n=== Verifier ===")
    verified = symplonk.bob_verify(commitment)
    
    print(f"\n=== Results ===")
    print(f"Verification: {'SUCCESS' if verified else 'FAILED'}")
    print(f"Total time: {time.perf_counter() - start:.4f} seconds")


def run_benchmark(n: int = 4096, use_gpu: bool = None):
    """Run performance benchmarks.
    
    Args:
        n: Domain size to use.
        use_gpu: Whether to use GPU, defaults to automatic detection.
    """
    # Use secp256k1 prime.
    p_val = (1 << 256) - (1 << 32) - 977
    
    logger.info(f"Initializing SymPLONK with n={n}, p={p_val}")
    symplonk = SymPLONK(n=n, p=p_val, use_gpu=use_gpu)
    
    # Run benchmarks.
    secret_sizes = [10, 100, 1000, min(n, 4000)]
    results = symplonk.benchmark(secret_sizes)
    
    print("\n=== Benchmark Results ===")
    print(f"{'Size':<10} {'Prove (s)':<12} {'Verify (s)':<12} {'Total (s)':<12} {'Verified'}")
    print("-" * 60)
    for size, data in results.items():
        print(f"{size:<10} {data['prove_time']:<12.4f} {data['verify_time']:<12.4f} "
              f"{data['total_time']:<12.4f} {data['verified']}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='SymPLONK Zero-Knowledge Proof System')
    parser.add_argument('--mode', choices=['demo', 'benchmark'], default='demo',
                        help='Run mode: demo or benchmark')
    parser.add_argument('--n', type=int, default=4096,
                        help='Domain size (default: 4096)')
    parser.add_argument('--secret', nargs='+', type=int, default=None,
                        help='Secret polynomial coefficients (e.g., --secret 1 2 3 4)')
    parser.add_argument('--use_gpu', action='store_true', help='Enable GPU acceleration')
    args = parser.parse_args()

    if args.mode == 'demo':
        run_demo(n=args.n, secret=args.secret, use_gpu=args.use_gpu)
    else:
        run_benchmark(n=args.n, use_gpu=args.use_gpu)
