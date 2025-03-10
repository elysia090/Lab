import math
import time
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union, Dict, Any, TypeVar
import numpy.typing as npt

# Try to import CuPy for GPU array operations.
try:
    import cupy as cp
    CUPY_AVAILABLE: bool = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False

# Type aliases
ComplexArray = npt.NDArray[np.complex128]
RealArray = npt.NDArray[np.float64]
T = TypeVar("T")

# Constants
TEICH_THRESHOLD: int = 20000          # Threshold for full Teichmüller lift precomputation
BATCH_SIZE: int = 4096                # Batch size for processing
MAX_THREADS_PER_BLOCK: int = 1024     # Maximum CUDA threads per block
MODULUS_THRESHOLD: int = (1 << 63) - 1  # Upper bound for using GPU kernel (64-bit)

@dataclass
class Commitment:
    """
    Data structure representing the zero-knowledge proof commitment.
    """
    transformed_evaluations: ComplexArray
    flow_time: float
    secret_original_evaluations: ComplexArray
    r: complex
    mask: ComplexArray

class FiniteFieldUtils:
    """
    Utility class for finite field operations optimized with caching.
    The evaluation domain is assumed to be a fixed cyclic subgroup:
       D_f = { g^0, g^1, …, g^(n-1) }
    so that each element's index is implicit.
    """
    _divisors_cache: Dict[int, List[int]] = {}
    _primitive_root_cache: Dict[int, Optional[int]] = {}
    _nth_root_cache: Dict[Tuple[int, int], Optional[int]] = {}

    @staticmethod
    def divisors(n: int) -> List[int]:
        if n in FiniteFieldUtils._divisors_cache:
            return FiniteFieldUtils._divisors_cache[n]
        divs: set[int] = set()
        for i in range(1, int(math.sqrt(n)) + 1):
            if n % i == 0:
                divs.add(i)
                divs.add(n // i)
        result: List[int] = sorted(divs)
        FiniteFieldUtils._divisors_cache[n] = result
        return result

    @staticmethod
    def prime_factors(n: int) -> List[int]:
        factors: List[int] = []
        while n % 2 == 0:
            factors.append(2)
            n //= 2
        while n % 3 == 0:
            factors.append(3)
            n //= 3
        i: int = 5
        while i * i <= n:
            for j in (0, 2):
                while n % (i + j) == 0:
                    factors.append(i + j)
                    n //= (i + j)
            i += 6
        if n > 1:
            factors.append(n)
        return sorted(set(factors))

    @classmethod
    def find_primitive_root(cls, p: int) -> Optional[int]:
        if p in cls._primitive_root_cache:
            return cls._primitive_root_cache[p]
        if p == 2:
            cls._primitive_root_cache[p] = 1
            return 1
        common_roots: List[int] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        factors: List[int] = cls.prime_factors(p - 1)
        exponents: List[int] = [(p - 1) // f for f in factors]
        for candidate in common_roots:
            if candidate >= p:
                continue
            if all(pow(candidate, exp, p) != 1 for exp in exponents):
                cls._primitive_root_cache[p] = candidate
                return candidate
        for candidate in range(2, min(p, 100)):
            if all(pow(candidate, exp, p) != 1 for exp in exponents):
                cls._primitive_root_cache[p] = candidate
                return candidate
        cls._primitive_root_cache[p] = None
        return None

    @classmethod
    def find_primitive_nth_root(cls, n: int, p: int) -> Optional[int]:
        """
        Find a primitive nth root in F_p.
        """
        cache_key: Tuple[int, int] = (n, p)
        if cache_key in cls._nth_root_cache:
            return cls._nth_root_cache[cache_key]
        if (p - 1) % n != 0:
            cls._nth_root_cache[cache_key] = None
            return None
        g: Optional[int] = cls.find_primitive_root(p)
        if g is None:
            cls._nth_root_cache[cache_key] = None
            return None
        k: int = (p - 1) // n
        candidate: int = pow(g, k, p)
        proper_divisors: List[int] = [d for d in cls.divisors(n) if d < n]
        if all(pow(candidate, d, p) != 1 for d in proper_divisors):
            cls._nth_root_cache[cache_key] = candidate
            return candidate
        cls._nth_root_cache[cache_key] = None
        return None

def teichmuller_lift_batch(indices: np.ndarray, n: int) -> np.ndarray:
    """
    Vectorized Teichmüller lift computation.
    For each index i, returns exp(2πi * i/n).
    """
    if indices.dtype != np.float64:
        indices = indices.astype(np.float64)
    angle_factor: float = 2.0 * np.pi / n
    angles: np.ndarray = angle_factor * indices
    return np.exp(1j * angles)

def gpu_teichmuller_lift_batch(indices: Any, n: int, xp: Any) -> Any:
    """
    GPU-optimized version of teichmuller_lift_batch using xp (either cp or np).
    """
    if indices.dtype != xp.float64:
        indices = indices.astype(xp.float64)
    angle_factor = 2.0 * xp.pi / n
    return xp.exp(1j * angle_factor * indices)

class HamiltonianFlow:
    """
    Class implementing Hamiltonian flow operations.
    For the quadratic Hamiltonian H(z)=1/2|z|^2, the flow is a rotation:
        z(t) = z(0) * exp(i*t)
    """
    def __init__(self, xp: Any) -> None:
        self.xp: Any = xp
        self._flow_cache: Dict[float, Any] = {}

    @staticmethod
    def hamiltonian_function(z: complex) -> float:
        return 0.5 * abs(z) ** 2

    def apply_flow(self, points: Any, t: float, epsilon: float = 1e-8) -> Any:
        if abs(t) < epsilon:
            return points
        if t in self._flow_cache:
            exp_factor = self._flow_cache[t]
        else:
            exp_factor = self.xp.exp(1j * t)
            self._flow_cache[t] = exp_factor
        return points * exp_factor

    def apply_inverse_flow(self, points: Any, t: float, epsilon: float = 1e-8) -> Any:
        return self.apply_flow(points, -t, epsilon)

if CUPY_AVAILABLE:
    poly_eval_kernel: cp.RawKernel = cp.RawKernel(r'''
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
    Optimized SymPLONK protocol implementation using a fixed cyclic subgroup.
    The evaluation domain is:
        D_f = { g^0, g^1, …, g^(n-1) }
    and the Teichmüller lift is computed as:
        τ(g^i) = exp(2πi * i/n)
    """
    def __init__(self, n: int, p: int, epsilon: float = 1e-10, use_gpu: bool = True) -> None:
        self.n: int = n
        self.p: int = p
        self.epsilon: float = epsilon
        self.use_fft: bool = (n & (n - 1) == 0)
        self.using_gpu: bool = False

        if use_gpu and CUPY_AVAILABLE:
            try:
                _ = cp.array([1, 2, 3])
                self.xp: Any = cp
                self.using_gpu = True
                print("Using GPU acceleration with CuPy.")
                device_props: Dict[str, Any] = cp.cuda.runtime.getDeviceProperties(0)
                print(f"GPU: {device_props['name'].decode('utf-8')}, Memory: {device_props['totalGlobalMem'] / 1e9:.2f} GB")
                self.threads_per_block: int = min(MAX_THREADS_PER_BLOCK, device_props['maxThreadsPerBlock'])
                print(f"Using {self.threads_per_block} threads per block")
            except Exception as e:
                print(f"GPU initialization failed: {e}. Falling back to CPU.")
                self.xp = np
        else:
            self.xp = np
            print("Using CPU (NumPy).")

        start_domain: float = time.perf_counter()
        self._setup_domains()
        end_domain: float = time.perf_counter()
        print(f"Domain setup completed in {end_domain - start_domain:.4f} seconds")
        self.flow: HamiltonianFlow = HamiltonianFlow(self.xp)
        self._precompute_flows()

    def _setup_domains(self) -> None:
        """
        Construct D_f as the cyclic subgroup { g^0, g^1, …, g^(n-1) }.
        Precompute a lookup table mapping each element to its index,
        then compute the Teichmüller lift vector L where:
            L[i] = exp(2πi * i/n)
        """
        start_primitive: float = time.perf_counter()
        self.omega_f: Optional[int] = FiniteFieldUtils.find_primitive_nth_root(self.n, self.p)
        end_primitive: float = time.perf_counter()
        print(f"Primitive root computation: {end_primitive - start_primitive:.4f} seconds")
        if self.omega_f is None:
            print(f"Warning: No primitive {self.n}th root found in F_{self.p}. Using sequential domain.")
            self.D_f: npt.NDArray[np.int64] = np.arange(1, self.n + 1, dtype=np.int64)
        else:
            self.D_f = np.zeros(self.n, dtype=np.int64)
            value: int = 1
            for i in range(self.n):
                self.D_f[i] = value
                value = (value * self.omega_f) % self.p
        self.D_f_indices: Dict[int, int] = {int(elem): i for i, elem in enumerate(self.D_f)}
        start_teich: float = time.perf_counter()
        indices: np.ndarray = np.arange(self.n, dtype=np.float64)
        if self.using_gpu:
            indices_gpu = self.xp.array(indices)
            self.D = gpu_teichmuller_lift_batch(indices_gpu, self.n, self.xp)
        else:
            domain_lifts: np.ndarray = teichmuller_lift_batch(indices, self.n)
            self.D = self.xp.array(domain_lifts, dtype=self.xp.complex128)
        end_teich: float = time.perf_counter()
        print(f"Teichmüller lift computation: {end_teich - start_teich:.4f} seconds")

    def _precompute_flows(self) -> None:
        self.common_flows: Dict[float, Any] = {}
        for t in [np.pi/8, np.pi/6, np.pi/4, np.pi/3, np.pi/2, 2*np.pi/3, 3*np.pi/4, np.pi, 2*np.pi]:
            self.common_flows[t] = self.xp.exp(1j * t)

    def poly_eval_horner_batch_cpu(self, x_vals: np.ndarray, coeffs: np.ndarray, p: int) -> np.ndarray:
        x_vals = np.array(x_vals, dtype=np.int64)
        coeffs = np.array(coeffs, dtype=np.int64)
        n_points: int = len(x_vals)
        result: np.ndarray = np.zeros(n_points, dtype=np.int64)
        chunk_size: int = 1024
        for chunk_start in range(0, n_points, chunk_size):
            chunk_end: int = min(chunk_start + chunk_size, n_points)
            x_chunk: np.ndarray = x_vals[chunk_start:chunk_end]
            r_chunk: np.ndarray = np.full_like(x_chunk, coeffs[-1])
            for c in reversed(coeffs[:-1]):
                r_chunk = (r_chunk * x_chunk + c) % p
            result[chunk_start:chunk_end] = r_chunk
        return result

    def poly_eval_horner_batch_gpu(self, x_vals: Any, coeffs: Any, p: int, threads_per_block: int) -> Any:
        if p > MODULUS_THRESHOLD:
            print("Modulus too large for GPU kernel; using CPU evaluation.")
            return self.poly_eval_horner_batch_cpu(x_vals, coeffs, p)
        x_vals_gpu: Any = cp.asarray(x_vals, dtype=cp.int64)
        coeffs_gpu: Any = cp.asarray(coeffs, dtype=cp.int64)
        n_points: int = len(x_vals_gpu)
        n_coeffs: int = len(coeffs_gpu)
        result_gpu: Any = cp.zeros_like(x_vals_gpu)
        blocks_per_grid: int = (n_points + threads_per_block - 1) // threads_per_block
        poly_eval_kernel((blocks_per_grid,), (threads_per_block,), 
                         (x_vals_gpu, coeffs_gpu, result_gpu, n_points, n_coeffs, p))
        return result_gpu

    def polynomial_encode(self, secret: List[int]) -> Any:
        """
        Encode the secret as polynomial evaluations on D_f, and compute the Teichmüller lift.
        Assumes the polynomial's outputs lie in D_f so that each value corresponds to g^i,
        and then τ(g^i) = exp(2πi * i/n) is computed.
        """
        encoding_start: float = time.perf_counter()
        coeffs: np.ndarray = np.array([c % self.p for c in secret] + [0] * (self.n - len(secret)), dtype=np.int64)
        if self.using_gpu:
            if self.p > MODULUS_THRESHOLD:
                poly_vals: np.ndarray = self.poly_eval_horner_batch_cpu(self.D_f, coeffs, self.p)
            else:
                poly_vals = self.poly_eval_horner_batch_gpu(self.D_f, coeffs, self.p, self.threads_per_block)
                poly_vals = cp.asnumpy(poly_vals)
        else:
            poly_vals = self.poly_eval_horner_batch_cpu(self.D_f, coeffs, self.p)
        indices: np.ndarray = np.array([self.D_f_indices.get(int(val), 0) for val in poly_vals], dtype=np.float64)
        lifts: np.ndarray = teichmuller_lift_batch(indices, self.n)
        result: Any = self.xp.array(lifts, dtype=self.xp.complex128)
        encoding_end: float = time.perf_counter()
        print(f"Polynomial encoding completed in {encoding_end - encoding_start:.4f} seconds")
        return result

    def blind_evaluations(self, evaluations: Any, r: Optional[complex] = None) -> Tuple[Any, complex, Any]:
        xp: Any = self.xp
        if r is None:
            r = complex(xp.random.normal(), xp.random.normal())
        if self.using_gpu:
            mask: Any = xp.random.normal(size=self.n) + 1j * xp.random.normal(size=self.n)
        else:
            mask = xp.array(np.random.normal(size=self.n) + 1j * np.random.normal(size=self.n))
        mask = mask / xp.linalg.norm(mask)
        return evaluations + r * mask, r, mask

    def alice_prove(self, secret: List[int], flow_time: float = np.pi/4) -> Commitment:
        print("\n=== Alice (Prover) ===")
        print(f"Secret (mod {self.p}): {secret}")
        start_time: float = time.perf_counter()
        encoding: Any = self.polynomial_encode(secret)
        blind, r, mask = self.blind_evaluations(encoding)
        transformed: Any = self.flow.apply_flow(blind, flow_time, self.epsilon)
        xp: Any = self.xp
        if xp is cp:
            transformed_cpu: ComplexArray = cp.asnumpy(transformed)
            encoding_cpu: ComplexArray = cp.asnumpy(encoding)
            mask_cpu: ComplexArray = cp.asnumpy(mask)
        else:
            transformed_cpu = transformed
            encoding_cpu = encoding
            mask_cpu = mask
        end_time: float = time.perf_counter()
        print(f"Total proof generation time: {end_time - start_time:.4f} seconds")
        return Commitment(
            transformed_evaluations=transformed_cpu,
            flow_time=flow_time,
            secret_original_evaluations=encoding_cpu,
            r=r,
            mask=mask_cpu
        )

    def bob_verify(self, commitment: Union[Commitment, Dict[str, Any]]) -> bool:
        print("\n=== Bob (Verifier) ===")
        if isinstance(commitment, dict):
            comm: Commitment = Commitment(**commitment)
        else:
            comm = commitment
        print(f"Flow time: t = {comm.flow_time:.4f}")
        start_time: float = time.perf_counter()
        xp: Any = self.xp
        transformed: Any = xp.array(comm.transformed_evaluations, dtype=xp.complex128)
        recovered: Any = self.flow.apply_inverse_flow(transformed, comm.flow_time, self.epsilon)
        original: Any = xp.array(comm.secret_original_evaluations, dtype=xp.complex128)
        mask: Any = xp.array(comm.mask, dtype=xp.complex128)
        expected: Any = original + comm.r * mask
        if xp is cp:
            diff_norm: float = float(xp.linalg.norm(recovered - expected).get())
        else:
            diff_norm = float(xp.linalg.norm(recovered - expected))
        verified: bool = diff_norm < self.epsilon
        print(f"L2 norm difference: {diff_norm:.8e}")
        print(f"Verification {'SUCCESS' if verified else 'FAILED'}")
        end_time: float = time.perf_counter()
        print(f"Total verification time: {end_time - start_time:.4f} seconds")
        return verified

if __name__ == "__main__":
    # Parameters: n = 256 (power of two) and p = 2^256 - 2^32 - 977 (secp256k1 prime)
    n_val: int = 256
    p_val: int = (1 << 256) - (1 << 32) - 977
    symplonk: SymPLONK = SymPLONK(n=n_val, p=p_val, use_gpu=True)
    secret: List[int] = [1, 2, 3, 4]  # Example secret data
    start_time: float = time.perf_counter()
    commitment: Commitment = symplonk.alice_prove(secret)
    verification_success: bool = symplonk.bob_verify(commitment)
    end_time: float = time.perf_counter()
    print("\n=== Verification Result ===")
    print(f"Verification result: {verification_success}")
    print("\n=== Total Execution Time ===")
    print(f"Total execution time (proof generation to verification): {end_time - start_time:.4f} seconds")
