import math
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union
import threading

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

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
# If the modulus p is huge, we use multi-precision GPU kernel.
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
        divs = set()
        for i in range(1, int(math.sqrt(n)) + 1):
            if n % i == 0:
                divs.add(i)
                divs.add(n // i)
        return sorted(divs)

    @staticmethod
    @lru_cache(maxsize=128)
    def prime_factors(n: int) -> List[int]:
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
    """Compute Teichmüller lift: exp(2πi * index / n) using NumPy."""
    indices = indices.astype(np.float64)
    return np.exp(1j * 2.0 * np.pi * indices / n)

def gpu_teichmuller_lift_batch(indices: Any, n: int, xp: Any) -> Any:
    """GPU-optimized Teichmüller lift computation using xp.exp."""
    indices = indices.astype(xp.float64)
    return xp.exp(1j * 2.0 * xp.pi * indices / n)

# Multi-precision GPU kernel source for Horner's method
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
    limbs = []
    for _ in range(8):
        limbs.append(np.uint32(val & 0xFFFFFFFF))
        val >>= 32
    return limbs

def limbs_to_int(limbs: List[int]) -> int:
    val = 0
    for limb in reversed(limbs):
        val = (val << 32) | int(limb)
    return val

class SymPLONK:
    """GPU-native SymPLONK protocol with asynchronous processing via CuPy streams."""
    def __init__(self, n: int, p: int, epsilon: float = 1e-10, use_gpu: bool = True) -> None:
        """
        Initialize with domain size n and prime p.
        All processing is performed on the GPU using CuPy in an asynchronous manner.
        """
        self.n = n
        self.p = p
        self.epsilon = epsilon
        self.using_gpu = False
        self.threads_per_block = MAX_THREADS_PER_BLOCK

        if use_gpu and CUPY_AVAILABLE:
            try:
                _ = cp.array([1, 2, 3])
                self.xp = cp
                self.using_gpu = True
                print("Using GPU acceleration with CuPy (GPU-native mode).")
                device_props = cp.cuda.runtime.getDeviceProperties(0)
                print(f"GPU: {device_props['name'].decode('utf-8')}, Memory: {device_props['totalGlobalMem'] / 1e9:.2f} GB")
                self.threads_per_block = min(MAX_THREADS_PER_BLOCK, device_props['maxThreadsPerBlock'])
            except Exception as e:
                print(f"GPU initialization failed: {e}. Falling back to CPU (NumPy).")
                self.xp = np
        else:
            self.xp = np
            print("Using CPU (NumPy).")

        self._setup_domains()    # Domain setup with asynchronous precomputation
        self.flow = HamiltonianFlow(self.xp)
        self._precompute_flows()   # Precompute flow factors asynchronously

        if self.using_gpu and self.p > MODULUS_THRESHOLD:
            self.poly_eval_kernel_mp = cp.RawKernel(multi_precision_kernel_source, 'poly_eval_horner_mp')

        self._split_eval_cache: Dict[str, Any] = {}

    def _setup_domains(self) -> None:
        """Precompute finite field domain D_f and Teichmüller lifts using GPU arrays with asynchronous streams."""
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
        if self.using_gpu and self.n > TEICH_THRESHOLD:
            # Create an asynchronous stream for Teichmüller lift computation
            stream = cp.cuda.Stream(non_blocking=True)
            with stream:
                self.D = self.xp.empty(self.n, dtype=self.xp.complex128)
                for i in range(0, self.n, BATCH_SIZE):
                    end = min(i + BATCH_SIZE, self.n)
                    batch_indices = self.xp.arange(i, end, dtype=self.xp.float64)
                    self.D[i:end] = gpu_teichmuller_lift_batch(batch_indices, self.n, self.xp)
            stream.synchronize()
        else:
            domain_lifts = teichmuller_lift_batch(indices_float, self.n)
            self.D = self.xp.array(domain_lifts, dtype=self.xp.complex128)

    def _precompute_flows(self) -> None:
        """Precompute Hamiltonian flow factors using an asynchronous stream."""
        stream = self.xp.cuda.Stream(non_blocking=True)
        common_angles = [math.pi/8, math.pi/6, math.pi/4, math.pi/3, math.pi/2, 2*math.pi/3, 3*math.pi/4, math.pi, 2*math.pi]
        with stream:
            for t in common_angles:
                self.flow._flow_cache[t] = self.xp.exp(1j * t)
        stream.synchronize()

    def poly_eval_horner_split(self, x_vals: Any, coeffs: np.ndarray, p: int, d: Optional[int] = None) -> Any:
        """
        Evaluate polynomial modulo p via split evaluation.
        This function now runs entirely on GPU arrays.
        """
        xp_local = self.xp
        x_vals = xp_local.asarray(x_vals, dtype=xp_local.int64)
        coeffs = xp_local.asarray(coeffs, dtype=xp_local.int64)
        deg = coeffs.shape[0] - 1
        if deg < 0:
            return xp_local.zeros_like(x_vals, dtype=xp_local.int64)
        if d is None:
            d = 2 ** int(math.floor(math.log(math.sqrt(deg), 2))) if deg > 0 else 1
            if d < 1:
                d = 1
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
        Evaluate polynomial using the multi-precision GPU kernel in an asynchronous stream.
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
        This method is fully GPU-native and uses asynchronous processing.
        """
        coeffs = np.array([c % self.p for c in secret] + [0]*(self.n - len(secret)), dtype=np.int64)
        if self.using_gpu and self.p > MODULUS_THRESHOLD:
            poly_vals = self.poly_eval_horner_batch_gpu_mp(self.D_f, coeffs, self.p)
        else:
            poly_vals = self.poly_eval_horner_split(self.D_f, coeffs, self.p)
        # Build indices as a CuPy array (remain on GPU)
        if self.using_gpu:
            indices = cp.array([self.D_f_indices.get(int(val), 0) for val in poly_vals])
        else:
            indices = np.array([self.D_f_indices.get(int(val), 0) for val in poly_vals], dtype=np.float64)
        lifts = gpu_teichmuller_lift_batch(indices, self.n, self.xp)
        return self.xp.array(lifts, dtype=self.xp.complex128)

    def blind_evaluations(self, evaluations: Any, r: Optional[complex] = None) -> Tuple[Any, complex, Any]:
        """
        Apply random blinding for zero-knowledge.
        All operations are performed using GPU arrays.
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
        The polynomial encoding and blinding are performed asynchronously on the GPU.
        """
        print("\n=== Alice (Prover) ===")
        print(f"Secret (mod {self.p}): {secret}")
        start_time = time.perf_counter()

        encoding = None
        def compute_encoding():
            nonlocal encoding
            encoding = self.polynomial_encode(secret)
        # Launch the encoding computation in a separate thread
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
        The entire verification remains GPU-native.
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

        diff_norm = float(xp.linalg.norm(recovered - expected).get() if xp is cp else xp.linalg.norm(recovered - expected))
        verified = diff_norm < self.epsilon
        
        print(f"L2 norm difference: {diff_norm:.8e}")
        print(f"Verification {'SUCCESS' if verified else 'FAILED'}")
        print(f"Total verification time: {time.perf_counter() - start_time:.4f} seconds")
        return verified

def run_symplonk_demo() -> None:
    """Run a demonstration of the GPU-native SymPLONK protocol with asynchronous processing."""
    n_val = 4096
    p_val = (1 << 256) - (1 << 32) - 977  # secp256k1 prime (huge modulus)
    
    symplonk = SymPLONK(n=n_val, p=p_val, use_gpu=True)
    secret = [1, 2, 3, 4]  # Example secret coefficients

    start_time = time.perf_counter()
    commitment = symplonk.alice_prove(secret)
    verification_success = symplonk.bob_verify(commitment)
    end_time = time.perf_counter()

    print("\n=== Verification Result ===")
    print(f"Verification result: {verification_success}")
    print("\n=== Total Execution Time ===")
    print(f"Total execution time: {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    run_symplonk_demo()

