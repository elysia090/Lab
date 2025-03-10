import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import math
from typing import List, Tuple, Dict, Optional, Union, Any, TypeVar, Callable
import numpy.typing as npt
from dataclasses import dataclass
from functools import lru_cache

# Type aliases
ComplexArray = npt.NDArray[np.complex128]
RealArray = npt.NDArray[np.float64]
T = TypeVar('T')

@dataclass
class Commitment:
    """Data structure representing the zero-knowledge proof commitment."""
    transformed_evaluations: ComplexArray
    flow_time: float
    secret_original_evaluations: ComplexArray
    r: complex
    mask: ComplexArray

class FiniteFieldUtils:
    """Utility class for finite field operations."""
    
    @staticmethod
    @lru_cache(maxsize=128)
    def divisors(n: int) -> List[int]:
        """Compute all divisors of the integer n."""
        divs = set()
        for i in range(1, int(math.sqrt(n)) + 1):
            if n % i == 0:
                divs.add(i)
                divs.add(n // i)
        return sorted(list(divs))
    
    @staticmethod
    def prime_factors(n: int) -> List[int]:
        """Compute the prime factors of the integer n."""
        i, factors = 2, []
        while i * i <= n:
            if n % i:
                i += 1
            else:
                n //= i
                if i not in factors:
                    factors.append(i)
        if n > 1 and n not in factors:
            factors.append(n)
        return factors
    
    @classmethod
    def find_primitive_nth_root(cls, n: int, p: int) -> Optional[int]:
        """
        Find a primitive nth root of unity ω_f in F_p.
        Requires that n divides (p - 1).
        
        Args:
            n: Order of the root of unity
            p: Prime modulus for the finite field F_p
            
        Returns:
            A primitive nth root of unity in F_p, or None if none exists
        """
        if (p - 1) % n != 0:
            print(f"Warning: n={n} is not a divisor of p-1={p-1}")
            
        # Precompute proper divisors for efficiency
        divisors = cls.divisors(n)
        proper_divisors = [d for d in divisors if d < n]
        
        for candidate in range(2, p):
            if pow(candidate, n, p) != 1:
                continue
            if all(pow(candidate, d, p) != 1 for d in proper_divisors):
                return candidate
        return None
    
    @classmethod
    def find_primitive_root(cls, p: int) -> Optional[int]:
        """
        Find a primitive element (generator) of F_p (of order p-1).
        
        Args:
            p: Prime modulus for the finite field F_p
            
        Returns:
            A primitive element of F_p, or None if none exists
        """
        factors = cls.prime_factors(p - 1)
        coprime_exponents = [(p - 1) // f for f in factors]
        
        for candidate in range(2, p):
            if all(pow(candidate, exp, p) != 1 for exp in coprime_exponents):
                return candidate
        return None
    
    @staticmethod
    def discrete_log(a: int, g: int, p: int) -> int:
        """
        Compute the discrete logarithm: find k such that g^k ≡ a (mod p).
        For small p, brute-force search is used; for larger p, Baby-step Giant-step is applied.
        
        Args:
            a: Target element in F_p
            g: Generator element in F_p
            p: Prime modulus for the finite field F_p
            
        Returns:
            k such that g^k ≡ a (mod p)
            
        Raises:
            ValueError: If no discrete log exists
        """
        if p < 100:  # Brute-force for small p
            for k in range(p - 1):
                if pow(g, k, p) == a:
                    return k
        else:
            m = int(math.sqrt(p - 1)) + 1
            # Baby-step
            baby_steps = {pow(g, j, p): j for j in range(m)}
            g_inv = pow(g, p - 1 - m, p)  # g^(-m) mod p
            value = a
            for i in range(m):
                if value in baby_steps:
                    return (i * m + baby_steps[value]) % (p - 1)
                value = (value * g_inv) % p
        raise ValueError(f"No discrete log exists: a={a}, g={g}, p={p}")

class HamiltonianFlow:
    """Handles Hamiltonian flow operations for complex numbers."""
    
    @staticmethod
    def hamiltonian_function(z: complex) -> float:
        """
        Define Hamiltonian function H(z) = 1/2 |z|^2 (for potential energy, etc.)
        
        Args:
            z: Complex number
            
        Returns:
            Energy value H(z)
        """
        return 0.5 * np.abs(z)**2
    
    @staticmethod
    def apply_flow(points: ComplexArray, t: float, epsilon: float = 1e-8) -> ComplexArray:
        """
        Apply the Hamiltonian flow (rotation) to complex points.
        For circular flow, the analytical solution is: z(t) = z(0) * exp(i*t)
        
        Args:
            points: Array of complex numbers
            t: Flow time parameter
            epsilon: Numerical error tolerance
            
        Returns:
            Transformed array of complex numbers
        """
        if abs(t) < epsilon:
            return points
        return points * np.exp(1j * t)
    
    @classmethod
    def apply_inverse_flow(cls, points: ComplexArray, t: float, epsilon: float = 1e-8) -> ComplexArray:
        """
        Apply the inverse Hamiltonian flow.
        
        Args:
            points: Array of complex numbers
            t: Flow time parameter
            epsilon: Numerical error tolerance
            
        Returns:
            Inverse-transformed array of complex numbers
        """
        return cls.apply_flow(points, -t, epsilon)


class SymPLONK:
    """
    SymPLONK:
    Demonstrates a geometric zero-knowledge proof protocol by mapping a finite field
    interpolation domain to complex numbers via the Teichmüller lift, then applying
    a Hamiltonian flow (rotation) to hide and later verify secret information.
    """
    def __init__(self, n: int = 8, p: int = 17, epsilon: float = 1e-8):
        """
        Initialize the SymPLONK protocol.
        
        Args:
            n: Number of interpolation points
            p: Prime for finite field F_p
            epsilon: Numerical error tolerance
        
        Raises:
            ValueError: If primitive roots cannot be found
        """
        self.n = n                  # Number of interpolation points
        self.p = p                  # Prime for finite field F_p
        self.epsilon = epsilon      # Numerical error tolerance
        
        # Define the interpolation domain in F_p: D_f = { ω_f^0, ω_f^1, ..., ω_f^(n-1) }
        self.omega_f = FiniteFieldUtils.find_primitive_nth_root(n, p)
        if self.omega_f is None:
            raise ValueError(f"Could not find a primitive {n}th root in F_{p}")
        self.D_f = [pow(self.omega_f, i, p) for i in range(n)]
        
        # Find a primitive element g of F_p* (used for computing discrete logs)
        self.g = FiniteFieldUtils.find_primitive_root(p)
        if self.g is None:
            raise ValueError(f"Could not find a primitive root for F_{p}")
        
        # Apply the Teichmüller lift to each element in D_f to obtain complex points
        self.D = np.array([self._teichmuller_lift(a) for a in self.D_f], dtype=np.complex128)
        
        # Initialize the Hamiltonian flow handler
        self.flow = HamiltonianFlow()

        self._log_setup_info()
    
    def _log_setup_info(self) -> None:
        """Log initialization information."""
        print(f"Setup complete: n={self.n}, p={self.p}")
        print(f"Finite field domain D_f: {self.D_f}")
        print(f"Teichmüller-lifted domain D (in ℂ): {np.around(self.D, 4)}")

    def _teichmuller_lift(self, a: int) -> complex:
        """
        Teichmüller lift of an element a in F_p.
          - If a == 0, returns 0.
          - For nonzero a, find k such that a ≡ g^k (mod p), then return exp(2πi * k/(p-1)).
        
        Args:
            a: Element in F_p
        
        Returns:
            Complex number representing the Teichmüller lift of a
            
        Raises:
            ValueError: If the primitive root is not set
        """
        if a == 0:
            return complex(0.0, 0.0)
        if self.g is None:
            raise ValueError("Primitive root is not set")
        k = FiniteFieldUtils.discrete_log(a, self.g, self.p)
        return np.exp(2j * np.pi * k / (self.p - 1))
    
    def polynomial_encode(self, secret: List[int]) -> ComplexArray:
        """
        Encode the secret (a list of elements in F_p) as a polynomial f(x) and evaluate it at the domain points.
        Then, apply the Teichmüller lift to convert each evaluation to a complex number.
        Horner's method is used for efficient polynomial evaluation.
        
        Args:
            secret: List of integers representing the secret data (mod p)
            
        Returns:
            Array of complex numbers representing the encoded secret
        """
        coeffs = [0] * self.n
        for j in range(min(len(secret), self.n)):
            coeffs[j] = secret[j] % self.p
            
        evaluations = []
        for x in self.D_f:
            val = 0
            for coeff in reversed(coeffs):
                val = (val * x + coeff) % self.p
            evaluations.append(self._teichmuller_lift(val))
            
        return np.array(evaluations, dtype=np.complex128)
    
    def blind_evaluations(self, evaluations: ComplexArray, r: Optional[complex] = None) -> Tuple[ComplexArray, complex, ComplexArray]:
        """
        Apply random blinding to the polynomial evaluations (complex numbers).
        A normalized random mask is generated for numerical stability.
        
        Args:
            evaluations: Array of complex numbers to blind
            r: Optional blinding factor (random complex number)
            
        Returns:
            Tuple containing:
            - Blinded evaluations
            - Blinding factor r
            - Blinding mask
        """
        if r is None:
            r = complex(np.random.normal(), np.random.normal())
        mask = np.array([complex(np.random.normal(), np.random.normal()) for _ in range(self.n)], dtype=np.complex128)
        mask = mask / np.linalg.norm(mask)
        return evaluations + r * mask, r, mask
    
    def alice_prove(self, secret: List[int], flow_time: float = np.pi/4) -> Commitment:
        """
        Alice's protocol:
          1. Encode the secret as a polynomial and evaluate it on the domain.
          2. Apply random blinding.
          3. Apply Hamiltonian flow (rotation).
          4. Generate and return the commitment.
        
        Args:
            secret: List of integers representing the secret data (mod p)
            flow_time: Time parameter for the Hamiltonian flow
            
        Returns:
            Commitment object containing the proof data
        """
        print("\n=== Alice (Prover) ===")
        print(f"Secret data (finite field elements mod {self.p}): {secret}")
        
        evaluations = self.polynomial_encode(secret)
        print("Polynomial encoding (with Teichmüller lift) complete")
        
        blinded_evals, r, mask = self.blind_evaluations(evaluations)
        print(f"Random blinding applied with coefficient r = {r:.4f}")
        
        transformed_evals = self.flow.apply_flow(blinded_evals, flow_time, self.epsilon)
        print(f"Hamiltonian flow applied with time t = {flow_time:.4f}")
        
        commitment = Commitment(
            transformed_evaluations=transformed_evals,
            flow_time=flow_time,
            secret_original_evaluations=evaluations,
            r=r,
            mask=mask
        )
        
        print("Commitment generated and ready to send to verifier")
        return commitment
    
    def bob_verify(self, commitment: Union[Commitment, Dict[str, Any]]) -> bool:
        """
        Bob's protocol:
          1. Apply the inverse Hamiltonian flow to the commitment.
          2. Verify that the masked (blinded) evaluations are preserved.
        Verification is performed by computing the L2 norm of the difference.
        
        Args:
            commitment: Commitment object or dictionary containing the proof data
            
        Returns:
            Boolean indicating whether verification succeeded
        """
        print("\n=== Bob (Verifier) ===")
        print("Received commitment from Alice")
        
        if isinstance(commitment, dict):
            comm = Commitment(
                transformed_evaluations=commitment['transformed_evaluations'],
                flow_time=commitment['flow_time'],
                secret_original_evaluations=commitment['secret_original_evaluations'],
                r=commitment['r'],
                mask=commitment['mask']
            )
        else:
            comm = commitment
        
        flow_time = comm.flow_time
        print(f"Flow time parameter: t = {flow_time:.4f}")
        
        recovered_evals = self.flow.apply_inverse_flow(comm.transformed_evaluations, flow_time, self.epsilon)
        print("Inverse Hamiltonian flow applied")
        
        original = comm.secret_original_evaluations
        r = comm.r
        mask = comm.mask
        blinded = original + r * mask
        
        norm_diff = np.linalg.norm(recovered_evals - blinded)
        print(f"L2 norm of difference: {norm_diff:.8e}")
        
        is_verified = norm_diff < self.epsilon
        print(f"Verification {'SUCCESS' if is_verified else 'FAILED'}: " 
              f"Geometric invariants {'preserved' if is_verified else 'not preserved'}")
        return is_verified
    
    def visualize_flow_3d_comparison(self, secret: List[int], flow_time: float = np.pi, num_steps: int = 100) -> None:
        """
        3D visualization comparing the trajectories before and after branding.
        
        Left subplot: Plain domain (Teichmüller-lifted points without blinding)
        Right subplot: Masked evaluations (after applying polynomial encoding and random blinding)
        Both are rotated according to the analytical Hamiltonian flow.
        
        Args:
            secret: List of integers representing the secret data (mod p)
            flow_time: Maximum time parameter for the Hamiltonian flow
            num_steps: Number of time steps for visualization
        """
        times = np.linspace(0, flow_time, num_steps)
        
        # Compute plain trajectories from the precomputed Teichmüller-lifted domain (self.D)
        plain_start = self.D
        plain_trajs = self._compute_trajectories(plain_start, times)
        
        # Compute polynomial encoding and then apply random blinding (without flow)
        plain_evals = self.polynomial_encode(secret)
        blinded_evals, r, mask = self.blind_evaluations(plain_evals)
        blinded_start = blinded_evals
        blinded_trajs = self._compute_trajectories(blinded_start, times)
        
        self._plot_3d_comparison(plain_trajs, blinded_trajs, times, flow_time)
    
    def _compute_trajectories(self, start_points: ComplexArray, times: RealArray) -> List[ComplexArray]:
        """
        Compute trajectories for a set of starting points.
        
        Args:
            start_points: Array of complex numbers representing starting points
            times: Array of time values
            
        Returns:
            List of trajectories (arrays of complex numbers)
        """
        trajectories = []
        for z in start_points:
            traj = z * np.exp(1j * times)
            trajectories.append(traj)
        return trajectories
    
    def _plot_3d_comparison(self, plain_trajs: List[ComplexArray], blinded_trajs: List[ComplexArray], 
                           times: RealArray, flow_time: float) -> None:
        """
        Plot 3D comparison of trajectories.
        
        Args:
            plain_trajs: List of trajectories for plain points
            blinded_trajs: List of trajectories for blinded points
            times: Array of time values
            flow_time: Maximum time parameter
        """
        # Create 3D subplots for comparison
        fig = plt.figure(figsize=(16, 8))
        
        # Left subplot: Before branding
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.set_title("Before Branding")
        ax1.set_xlabel("Re(z)")
        ax1.set_ylabel("Im(z)")
        ax1.set_zlabel("Time t")
        ax1.set_xlim(-1.5, 1.5)
        ax1.set_ylim(-1.5, 1.5)
        ax1.set_zlim(0, flow_time)
        colors = cm.viridis(np.linspace(0, 1, self.n))
        
        self._plot_trajectories(ax1, plain_trajs, times, colors)
        
        # Right subplot: After branding
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.set_title("After Branding")
        ax2.set_xlabel("Re(z)")
        ax2.set_ylabel("Im(z)")
        ax2.set_zlabel("Time t")
        ax2.set_xlim(-1.5, 1.5)
        ax2.set_ylim(-1.5, 1.5)
        ax2.set_zlim(0, flow_time)
        
        self._plot_trajectories(ax2, blinded_trajs, times, colors)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_trajectories(self, ax: plt.Axes, trajectories: List[ComplexArray], 
                          times: RealArray, colors: np.ndarray) -> None:
        """
        Plot trajectories on a given axes.
        
        Args:
            ax: Matplotlib axes to plot on
            trajectories: List of trajectories (arrays of complex numbers)
            times: Array of time values
            colors: Array of colors for each trajectory
        """
        for i, traj in enumerate(trajectories):
            xs = traj.real
            ys = traj.imag
            ax.plot(xs, ys, times, color=colors[i], lw=2)
            ax.scatter(xs[0], ys[0], times[0], color=colors[i], s=50, edgecolor='black')
            ax.scatter(xs[-1], ys[-1], times[-1], color=colors[i], s=50, edgecolor='white')


def run_demo() -> None:
    """Run demonstration of the SymPLONK protocol with 3D comparison visualization."""
    print("="*50)
    print("SymPLONK Protocol Demonstration (Finite Field with Teichmüller Lift)")
    print("="*50)
    print("A geometric approach to zero-knowledge proofs using Kähler manifolds\n")
    
    symplonk = SymPLONK(n=8, p=17, epsilon=1e-15)
    secret = [1, 2, 3, 4]
    
    commitment = symplonk.alice_prove(secret)
    verification_result = symplonk.bob_verify(commitment)
    
    print("\n" + "="*50)
    print(f"Verification result: {'SUCCESS' if verification_result else 'FAILED'}")
    print("="*50)
    
    print("\nVisualizing 3D comparison (Before vs. After Branding)...")
    symplonk.visualize_flow_3d_comparison(secret, flow_time=np.pi, num_steps=100)
    
    print("\nDemonstration complete!")


if __name__ == "__main__":
    run_demo()
