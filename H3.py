import math
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union, Dict, Any, Callable
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from functools import lru_cache

# Try to import CuPy for GPU array operations
try:
    import cupy as cp
    CUPY_AVAILABLE: bool = True
except ImportError:
    cp = None
    CUPY_AVAILABLE: bool = False

# Type aliases for improved readability
ComplexArray = npt.NDArray[np.complex128]
RealArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

# Performance-tuned constants
TEICH_THRESHOLD: int = 20000          # Threshold for full Teichmüller lift precomputation
BATCH_SIZE: int = 4096                # Batch size for processing large arrays
MAX_THREADS_PER_BLOCK: int = 1024     # Maximum CUDA threads per block
MODULUS_THRESHOLD: int = (1 << 63) - 1  # Use GPU kernel only if p fits in a 64-bit integer

# Known primitive roots for common cryptographic primes
KNOWN_PRIMITIVE_ROOTS: Dict[int, int] = {
    (1 << 256) - (1 << 32) - 977: 7  # secp256k1 prime
}

@dataclass
class Commitment:
    """
    Data structure representing a zero-knowledge proof commitment.
    
    Attributes:
        transformed_evaluations: Complex array after Hamiltonian flow application
        flow_time: Time parameter used in the Hamiltonian flow
        secret_original_evaluations: Original complex encoding of the secret
        r: Complex blinding factor
        mask: Random complex mask for blinding
    """
    transformed_evaluations: ComplexArray
    flow_time: float
    secret_original_evaluations: ComplexArray
    r: complex
    mask: ComplexArray

class FiniteFieldUtils:
    """
    Utility class for finite field operations with caching.
    
    Assumes the evaluation domain is a fixed cyclic subgroup:
    D_f = { g^0, g^1, …, g^(n-1) }
    where each element's index is implicit.
    """
    
    @staticmethod
    @lru_cache(maxsize=128)
    def divisors(n: int) -> List[int]:
        """
        Compute all divisors of an integer n.
        
        Args:
            n: The integer to find divisors for
            
        Returns:
            Sorted list of all divisors of n
        """
        divs: set[int] = set()
        for i in range(1, int(math.sqrt(n)) + 1):
            if n % i == 0:
                divs.add(i)
                divs.add(n // i)
        return sorted(divs)

    @staticmethod
    @lru_cache(maxsize=128)
    def prime_factors(n: int) -> List[int]:
        """
        Compute the prime factorization of an integer n.
        
        Args:
            n: The integer to factorize
            
        Returns:
            Sorted list of unique prime factors of n
        """
        factors: List[int] = []
        
        # Handle factors of 2
        while n % 2 == 0:
            factors.append(2)
            n //= 2
            
        # Handle factors of 3
        while n % 3 == 0:
            factors.append(3)
            n //= 3
            
        # Handle remaining factors using 6k±1 optimization
        i: int = 5
        while i * i <= n:
            for j in (0, 2):  # Check 5 and 7, then 11 and 13, etc.
                while n % (i + j) == 0:
                    factors.append(i + j)
                    n //= (i + j)
            i += 6
            
        # Add remaining prime factor if any
        if n > 1:
            factors.append(n)
            
        return sorted(set(factors))

    @classmethod
    @lru_cache(maxsize=64)
    def find_primitive_root(cls, p: int) -> Optional[int]:
        """
        Find a primitive root modulo p.
        
        Args:
            p: Prime modulus
            
        Returns:
            A primitive root modulo p, or None if not found
        """
        # Special cases
        if p == 2:
            return 1
            
        if p in KNOWN_PRIMITIVE_ROOTS:
            return KNOWN_PRIMITIVE_ROOTS[p]
            
        # Try common small primes first for efficiency
        common_roots: List[int] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        
        # Get prime factors of p-1 for testing primitive root property
        factors: List[int] = cls.prime_factors(p - 1)
        exponents: List[int] = [(p - 1) // f for f in factors]
        
        # Check common roots first
        for candidate in common_roots:
            if candidate >= p:
                continue
            if all(pow(candidate, exp, p) != 1 for exp in exponents):
                return candidate
                
        # Fall back to sequential search with a practical limit
        for candidate in range(2, min(p, 100)):
            if all(pow(candidate, exp, p) != 1 for exp in exponents):
                return candidate
                
        return None

    @classmethod
    @lru_cache(maxsize=64)
    def find_primitive_nth_root(cls, n: int, p: int) -> Optional[int]:
        """
        Find a primitive nth root of unity in F_p.
        
        Args:
            n: Order of the root
            p: Prime modulus
            
        Returns:
            A primitive nth root of unity, or None if not found
        """
        # A primitive nth root exists only if n divides p-1
        if (p - 1) % n != 0:
            return None
            
        # Find a primitive root
        g: Optional[int] = cls.find_primitive_root(p)
        if g is None:
            return None
            
        # Compute a candidate nth root
        k: int = (p - 1) // n
        candidate: int = pow(g, k, p)
        
        # Verify it's a primitive nth root by checking that it doesn't
        # have a smaller order (i.e., it doesn't satisfy x^d ≡ 1 for any d < n)
        proper_divisors: List[int] = [d for d in cls.divisors(n) if d < n]
        if all(pow(candidate, d, p) != 1 for d in proper_divisors):
            return candidate
            
        return None

def teichmuller_lift_batch(indices: np.ndarray, n: int) -> np.ndarray:
    """
    Vectorized Teichmüller lift computation.
    For each index i, returns exp(2πi * i/n).
    
    Args:
        indices: Array of indices to lift
        n: Order of the group
        
    Returns:
        Complex array of lifted values
    """
    if indices.dtype != np.float64:
        indices = indices.astype(np.float64)
    angle_factor: float = 2.0 * np.pi / n
    angles: np.ndarray = angle_factor * indices
    return np.exp(1j * angles)

def gpu_teichmuller_lift_batch(indices: Any, n: int, xp: Any) -> Any:
    """
    GPU-optimized version of teichmuller_lift_batch.
    
    Args:
        indices: CuPy array of indices to lift
        n: Order of the group
        xp: Array module (CuPy)
        
    Returns:
        CuPy complex array of lifted values
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
        """
        Initialize the Hamiltonian flow calculator.
        
        Args:
            xp: Array module (NumPy or CuPy)
        """
        self.xp: Any = xp
        self._flow_cache: Dict[float, Any] = {}

    @staticmethod
    def hamiltonian_function(z: complex) -> float:
        """
        Compute the Hamiltonian function value.
        
        Args:
            z: Complex input
            
        Returns:
            Real value of the Hamiltonian
        """
        return 0.5 * abs(z) ** 2

    def apply_flow(self, points: Any, t: float, epsilon: float = 1e-8) -> Any:
        """
        Apply Hamiltonian flow to a set of points for time t.
        
        Args:
            points: Complex array of points
            t: Time parameter
            epsilon: Small value to avoid unnecessary calculations
            
        Returns:
            Complex array after flow application
        """
        if abs(t) < epsilon:
            return points
            
        if t in self._flow_cache:
            exp_factor = self._flow_cache[t]
        else:
            exp_factor = self.xp.exp(1j * t)
            self._flow_cache[t] = exp_factor
            
        return points * exp_factor

    def apply_inverse_flow(self, points: Any, t: float, epsilon: float = 1e-8) -> Any:
        """
        Apply inverse Hamiltonian flow for time t.
        
        Args:
            points: Complex array of points
            t: Time parameter
            epsilon: Small value to avoid unnecessary calculations
            
        Returns:
            Complex array after inverse flow application
        """
        return self.apply_flow(points, -t, epsilon)

# Define GPU kernel for polynomial evaluation if CuPy is available
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
    Optimized implementation of the SymPLONK protocol using a fixed cyclic subgroup.
    
    The evaluation domain is defined as:
          D_f = { g^0, g^1, …, g^(n-1) }
    and the Teichmüller lift is computed by:
          τ(g^i) = exp(2πi * i/n)
          
    This avoids runtime discrete log computations.
    """
    def __init__(self, n: int, p: int, epsilon: float = 1e-10, use_gpu: bool = True) -> None:
        """
        Initialize the SymPLONK protocol.
        
        Args:
            n: Size of the evaluation domain
            p: Prime field modulus
            epsilon: Error tolerance for verification
            use_gpu: Whether to use GPU acceleration if available
        """
        self.n: int = n
        self.p: int = p
        self.epsilon: float = epsilon
        self.use_fft: bool = (n & (n - 1) == 0)  # Check if n is a power of 2
        self.using_gpu: bool = False
        self.threads_per_block: int = MAX_THREADS_PER_BLOCK

        # Initialize the array module (NumPy or CuPy)
        if use_gpu and CUPY_AVAILABLE:
            try:
                _ = cp.array([1, 2, 3])
                self.xp: Any = cp
                self.using_gpu = True
                print("Using GPU acceleration with CuPy.")
                
                # Get GPU device properties
                device_props: Dict[str, Any] = cp.cuda.runtime.getDeviceProperties(0)
                print(f"GPU: {device_props['name'].decode('utf-8')}, Memory: {device_props['totalGlobalMem'] / 1e9:.2f} GB")
                
                # Set thread configuration
                self.threads_per_block = min(MAX_THREADS_PER_BLOCK, device_props['maxThreadsPerBlock'])
                print(f"Using {self.threads_per_block} threads per block")
            except Exception as e:
                print(f"GPU initialization failed: {e}. Falling back to CPU.")
                self.xp = np
        else:
            self.xp = np
            print("Using CPU (NumPy).")

        # Set up evaluation domains
        start_domain: float = time.perf_counter()
        self._setup_domains()
        end_domain: float = time.perf_counter()
        print(f"Domain setup completed in {end_domain - start_domain:.4f} seconds")
        
        # Initialize Hamiltonian flow
        self.flow: HamiltonianFlow = HamiltonianFlow(self.xp)
        self._precompute_flows()

    def _setup_domains(self) -> None:
        """
        Set up the finite field domain D_f as a cyclic subgroup and compute Teichmüller lifts.
        
        The domain is: D_f = { g^0, g^1, …, g^(n-1) }
        The Teichmüller lift is: L[i] = exp(2πi * i/n)
        """
        # Find a primitive nth root of unity
        start_primitive: float = time.perf_counter()
        self.omega_f: Optional[int] = FiniteFieldUtils.find_primitive_nth_root(self.n, self.p)
        end_primitive: float = time.perf_counter()
        print(f"Primitive root computation: {end_primitive - start_primitive:.4f} seconds")
        
        # Generate the finite field domain
        if self.omega_f is None:
            print(f"Warning: No primitive {self.n}th root found in F_{self.p}. Using sequential domain.")
            self.D_f: IntArray = np.arange(1, self.n + 1, dtype=np.int64)
        else:
            self.D_f = np.zeros(self.n, dtype=np.int64)
            value: int = 1
            for i in range(self.n):
                self.D_f[i] = value
                value = (value * self.omega_f) % self.p
                
        # Create lookup map from domain elements to indices
        self.D_f_indices: Dict[int, int] = {int(elem): i for i, elem in enumerate(self.D_f)}
        
        # Compute Teichmüller lifts
        start_teich: float = time.perf_counter()
        indices: np.ndarray = np.arange(self.n, dtype=np.float64)
        
        # Use batch processing for large domains when on GPU
        if self.using_gpu and self.n > TEICH_THRESHOLD:
            self.D = self.xp.empty(self.n, dtype=self.xp.complex128)
            for i in range(0, self.n, BATCH_SIZE):
                end: int = min(i + BATCH_SIZE, self.n)
                batch_indices = self.xp.arange(i, end, dtype=self.xp.float64)
                self.D[i:end] = gpu_teichmuller_lift_batch(batch_indices, self.n, self.xp)
        else:
            if self.using_gpu:
                indices_gpu = self.xp.array(indices)
                self.D = gpu_teichmuller_lift_batch(indices_gpu, self.n, self.xp)
            else:
                domain_lifts: np.ndarray = teichmuller_lift_batch(indices, self.n)
                self.D = self.xp.array(domain_lifts, dtype=self.xp.complex128)
                
        end_teich: float = time.perf_counter()
        print(f"Teichmüller lift computation: {end_teich - start_teich:.4f} seconds")

    def _precompute_flows(self) -> None:
        """Precompute common Hamiltonian flow values for efficiency."""
        self.common_flows: Dict[float, Any] = {}
        common_angles = [np.pi/8, np.pi/6, np.pi/4, np.pi/3, np.pi/2, 
                         2*np.pi/3, 3*np.pi/4, np.pi, 2*np.pi]
                         
        for t in common_angles:
            self.common_flows[t] = self.xp.exp(1j * t)

    def poly_eval_horner_batch_cpu(self, x_vals: np.ndarray, coeffs: np.ndarray, p: int) -> np.ndarray:
        """
        Vectorized Horner's method for polynomial evaluation on CPU using chunking.
        
        Args:
            x_vals: Array of points to evaluate at
            coeffs: Polynomial coefficients
            p: Field modulus
            
        Returns:
            Array of polynomial evaluations
        """
        x_vals = np.asarray(x_vals, dtype=np.int64)
        coeffs = np.asarray(coeffs, dtype=np.int64)
        n_points: int = len(x_vals)
        result: np.ndarray = np.zeros(n_points, dtype=np.int64)
        
        # Process in chunks to improve cache efficiency
        chunk_size: int = 1024
        for chunk_start in range(0, n_points, chunk_size):
            chunk_end: int = min(chunk_start + chunk_size, n_points)
            x_chunk: np.ndarray = x_vals[chunk_start:chunk_end]
            
            # Apply Horner's method
            r_chunk: np.ndarray = np.full_like(x_chunk, coeffs[-1])
            for c in reversed(coeffs[:-1]):
                r_chunk = (r_chunk * x_chunk + c) % p
                
            result[chunk_start:chunk_end] = r_chunk
            
        return result

    def poly_eval_horner_batch_gpu(self, x_vals: Any, coeffs: Any, p: int, threads_per_block: int) -> Any:
        """
        GPU-accelerated polynomial evaluation using Horner's method.
        
        Args:
            x_vals: Array of points to evaluate at
            coeffs: Polynomial coefficients
            p: Field modulus
            threads_per_block: Number of CUDA threads per block
            
        Returns:
            Array of polynomial evaluations
        """
        # Fall back to CPU for large moduli
        if p > MODULUS_THRESHOLD:
            print("Modulus too large for GPU kernel; using CPU evaluation.")
            return self.poly_eval_horner_batch_cpu(x_vals, coeffs, p)
            
        # Copy data to GPU
        x_vals_gpu: Any = cp.asarray(x_vals, dtype=cp.int64)
        coeffs_gpu: Any = cp.asarray(coeffs, dtype=cp.int64)
        n_points: int = len(x_vals_gpu)
        n_coeffs: int = len(coeffs_gpu)
        
        # Prepare result array
        result_gpu: Any = cp.zeros_like(x_vals_gpu)
        
        # Launch kernel
        blocks_per_grid: int = (n_points + threads_per_block - 1) // threads_per_block
        poly_eval_kernel((blocks_per_grid,), (threads_per_block,),
                         (x_vals_gpu, coeffs_gpu, result_gpu, n_points, n_coeffs, p))
                         
        return result_gpu

    def polynomial_encode(self, secret: List[int]) -> Any:
        """
        Encode a secret as polynomial evaluations on D_f and compute the Teichmüller lift.
        
        Args:
            secret: List of integer coefficients representing the secret polynomial
            
        Returns:
            Complex array of Teichmüller lifts of polynomial evaluations
        """
        encoding_start: float = time.perf_counter()
        
        # Pad coefficients to length n with zeros
        coeffs: np.ndarray = np.array([c % self.p for c in secret] + 
                                      [0] * (self.n - len(secret)), dtype=np.int64)
        
        # Evaluate polynomial at all points in D_f
        if self.using_gpu:
            if self.p > MODULUS_THRESHOLD:
                poly_vals: np.ndarray = self.poly_eval_horner_batch_cpu(self.D_f, coeffs, self.p)
            else:
                poly_vals = self.poly_eval_horner_batch_gpu(self.D_f, coeffs, self.p, self.threads_per_block)
                poly_vals = cp.asnumpy(poly_vals)
        else:
            poly_vals = self.poly_eval_horner_batch_cpu(self.D_f, coeffs, self.p)
        
        # Convert field elements to indices in D_f
        indices: np.ndarray = np.array([self.D_f_indices.get(int(val), 0) 
                                       for val in poly_vals], dtype=np.float64)
        
        # Apply Teichmüller lift
        lifts: np.ndarray = teichmuller_lift_batch(indices, self.n)
        result: Any = self.xp.array(lifts, dtype=self.xp.complex128)
        
        encoding_end: float = time.perf_counter()
        print(f"Polynomial encoding completed in {encoding_end - encoding_start:.4f} seconds")
        
        return result

    def blind_evaluations(self, evaluations: Any, r: Optional[complex] = None) -> Tuple[Any, complex, Any]:
        """
        Apply random blinding to the evaluations for zero-knowledge.
        
        Args:
            evaluations: Complex array of evaluations
            r: Optional fixed blinding factor
            
        Returns:
            Tuple of (blinded evaluations, blinding factor, random mask)
        """
        xp: Any = self.xp
        
        # Generate random blinding factor if not provided
        if r is None:
            r = complex(xp.random.normal(), xp.random.normal())
            
        # Generate random mask
        if self.using_gpu:
            mask: Any = xp.random.normal(size=self.n) + 1j * xp.random.normal(size=self.n)
        else:
            mask = xp.array(np.random.normal(size=self.n) + 1j * np.random.normal(size=self.n))
            
        # Normalize mask
        mask = mask / xp.linalg.norm(mask)
        
        # Apply blinding
        return evaluations + r * mask, r, mask

    def alice_prove(self, secret: List[int], flow_time: float = np.pi/4) -> Commitment:
        """
        Generate a zero-knowledge proof from a secret.
        
        Args:
            secret: List of integers representing the secret
            flow_time: Time parameter for the Hamiltonian flow
            
        Returns:
            Commitment containing the proof data
        """
        print("\n=== Alice (Prover) ===")
        print(f"Secret (mod {self.p}): {secret}")
        
        start_time: float = time.perf_counter()
        
        # Step 1: Encode the secret as polynomial evaluations
        encoding: Any = self.polynomial_encode(secret)
        
        # Step 2: Blind the evaluations with a random mask
        blind, r, mask = self.blind_evaluations(encoding)
        
        # Step 3: Apply the Hamiltonian flow for time t
        transformed: Any = self.flow.apply_flow(blind, flow_time, self.epsilon)
        
        # Move data to CPU if needed
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
        
        # Create and return the commitment
        return Commitment(
            transformed_evaluations=transformed_cpu,
            flow_time=flow_time,
            secret_original_evaluations=encoding_cpu,
            r=r,
            mask=mask_cpu
        )

    def bob_verify(self, commitment: Union[Commitment, Dict[str, Any]]) -> bool:
        """
        Verify a zero-knowledge proof.
        
        Args:
            commitment: Commitment object or dict with proof data
            
        Returns:
            Boolean indicating verification success or failure
        """
        print("\n=== Bob (Verifier) ===")
        
        # Convert dict to Commitment if needed
        if isinstance(commitment, dict):
            comm: Commitment = Commitment(**commitment)
        else:
            comm = commitment
            
        print(f"Flow time: t = {comm.flow_time:.4f}")
        start_time: float = time.perf_counter()
        
        # Get array module
        xp: Any = self.xp
        
        # Move data to appropriate device
        transformed: Any = xp.array(comm.transformed_evaluations, dtype=xp.complex128)
        original: Any = xp.array(comm.secret_original_evaluations, dtype=xp.complex128)
        mask: Any = xp.array(comm.mask, dtype=xp.complex128)
        
        # Apply inverse flow to recover blinded evaluations
        recovered: Any = self.flow.apply_inverse_flow(transformed, comm.flow_time, self.epsilon)
        
        # Compute expected blinded evaluations
        expected: Any = original + comm.r * mask
        
        # Compute L2 distance between recovered and expected values
        if xp is cp:
            diff_norm: float = float(xp.linalg.norm(recovered - expected).get())
        else:
            diff_norm = float(xp.linalg.norm(recovered - expected))
            
        # Check if distance is within tolerance
        verified: bool = diff_norm < self.epsilon
        
        print(f"L2 norm difference: {diff_norm:.8e}")
        print(f"Verification {'SUCCESS' if verified else 'FAILED'}")
        
        end_time: float = time.perf_counter()
        print(f"Total verification time: {end_time - start_time:.4f} seconds")
        
        return verified

def run_symplonk_demo() -> None:
    """Run a demonstration of the SymPLONK protocol."""
    # Parameters: n = 256 (power of two) and p = 2^256 - 2^32 - 977 (secp256k1 prime)
    n_val: int = 256
    p_val: int = (1 << 256) - (1 << 32) - 977
    
    # Initialize SymPLONK
    symplonk: SymPLONK = SymPLONK(n=n_val, p=p_val, use_gpu=True)
    
    # Example secret data
    secret: List[int] = [1, 2, 3, 4]
    
    # Generate and verify proof
    start_time: float = time.perf_counter()
    commitment: Commitment = symplonk.alice_prove(secret)
    verification_success: bool = symplonk.bob_verify(commitment)
    end_time: float = time.perf_counter()
    
    # Report results
    print("\n=== Verification Result ===")
    print(f"Verification result: {verification_success}")
    
    print("\n=== Total Execution Time ===")
    print(f"Total execution time (proof generation to verification): {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    run_symplonk_demo()

##TODO： Barrett縮小法を利用し、CPUでのポリノミアル計算を回避する
