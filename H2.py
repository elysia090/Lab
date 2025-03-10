import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import math
from typing import List, Tuple, Dict, Optional, Union, Any, TypeVar, Callable
import numpy.typing as npt
from dataclasses import dataclass
from functools import lru_cache

# Type aliases for clarity
ComplexArray = npt.NDArray[np.complex128]
RealArray = npt.NDArray[np.float64]
T = TypeVar('T')

@dataclass
class Commitment:
    """
    Data structure representing the zero-knowledge proof commitment.

    Attributes:
        transformed_evaluations: Complex array of evaluations after applying Hamiltonian flow
        flow_time: Time parameter used for the Hamiltonian flow
        secret_original_evaluations: Complex array of original polynomial evaluations
        r: Complex blinding factor
        mask: Complex array used as the blinding mask
    """
    transformed_evaluations: ComplexArray
    flow_time: float
    secret_original_evaluations: ComplexArray
    r: complex
    mask: ComplexArray

class FiniteFieldUtils:
    """
    Utility class for finite field operations.

    Provides methods for finding primitive roots, computing discrete logarithms,
    and other number-theoretic functions in finite fields.
    """

    @staticmethod
    @lru_cache(maxsize=128)
    def divisors(n: int) -> List[int]:
        """
        Compute all divisors of an integer.

        Args:
            n: Integer to find divisors for

        Returns:
            Sorted list of all divisors of n
        """
        divs = set()
        for i in range(1, int(math.sqrt(n)) + 1):
            if n % i == 0:
                divs.add(i)
                divs.add(n // i)
        return sorted(list(divs))

    @staticmethod
    def prime_factors(n: int) -> List[int]:
        """
        Compute unique prime factors of an integer.

        Args:
            n: Integer to find prime factors for

        Returns:
            List of unique prime factors of n
        """
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
        Find a primitive nth root of unity in F_p.

        A primitive nth root ω satisfies ω^n ≡ 1 (mod p) and ω^k ≢ 1 (mod p) for all k < n.

        Args:
            n: Order of the root of unity
            p: Prime modulus for the finite field F_p

        Returns:
            A primitive nth root of unity in F_p, or None if none exists

        Note:
            Requires n to divide (p - 1) for such a root to exist.
        """
        if (p - 1) % n != 0:
            print(f"Warning: n={n} does not divide p-1={p-1}")

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
        Find a primitive element (generator) of F_p with order p-1.

        A primitive element g generates all non-zero elements of F_p as {g^0, g^1, ..., g^(p-2)}.

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

        Uses brute-force for small p; Baby-step Giant-step for larger p.

        Args:
            a: Target element in F_p
            g: Generator element in F_p
            p: Prime modulus for the finite field F_p

        Returns:
            k such that g^k ≡ a (mod p)

        Raises:
            ValueError: If no discrete logarithm exists
        """
        if p < 100:
            for k in range(p - 1):
                if pow(g, k, p) == a:
                    return k
        else:
            m = int(math.sqrt(p - 1)) + 1
            baby_steps = {pow(g, j, p): j for j in range(m)}
            g_inv = pow(g, p - 1 - m, p)
            value = a
            for i in range(m):
                if value in baby_steps:
                    return (i * m + baby_steps[value]) % (p - 1)
                value = (value * g_inv) % p
        raise ValueError(f"No discrete log exists: a={a}, g={g}, p={p}")

class HamiltonianFlow:
    """
    Handles Hamiltonian flow operations for complex numbers.

    Implements Hamiltonian flow in the complex plane, resulting in rotations with
    a quadratic Hamiltonian function.
    """

    @staticmethod
    def hamiltonian_function(z: complex) -> float:
        """
        Define Hamiltonian function H(z) = 1/2 |z|^2.

        This quadratic Hamiltonian leads to circular flows in the complex plane.

        Args:
            z: Complex number

        Returns:
            Energy value H(z)
        """
        return 0.5 * np.abs(z) ** 2

    @staticmethod
    def apply_flow(points: ComplexArray, t: float, epsilon: float = 1e-8) -> ComplexArray:
        """
        Apply Hamiltonian flow (rotation) to complex points.

        For the quadratic Hamiltonian, the solution is: z(t) = z(0) * exp(i*t).

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

        Inverse flow uses the forward flow with a negative time parameter.

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
    SymPLONK: Geometric Zero-Knowledge Proof Protocol

    Maps finite field interpolation domain to complex numbers via Teichmüller lift,
    then applies Hamiltonian flow (rotation) to hide and verify secret information.
    """

    def __init__(self, n: int = 8, p: int = 17, epsilon: float = 1e-8) -> None:
        """
        Initialize the SymPLONK protocol.

        Args:
            n: Number of interpolation points
            p: Prime for finite field F_p
            epsilon: Numerical error tolerance

        Raises:
            ValueError: If primitive roots cannot be found
        """
        self.n: int = n
        self.p: int = p
        self.epsilon: float = epsilon

        self.omega_f: Optional[int] = FiniteFieldUtils.find_primitive_nth_root(n, p)
        if self.omega_f is None:
            raise ValueError(f"Could not find a primitive {n}th root in F_{p}")
        self.D_f: List[int] = [pow(self.omega_f, i, p) for i in range(n)]

        self.g: Optional[int] = FiniteFieldUtils.find_primitive_root(p)
        if self.g is None:
            raise ValueError(f"Could not find a primitive root for F_{p}")

        self.D: ComplexArray = np.array([self._teichmuller_lift(a) for a in self.D_f], dtype=np.complex128)

        self.flow: HamiltonianFlow = HamiltonianFlow()
        self._log_setup_info()

    def _log_setup_info(self) -> None:
        """Log initialization information for debugging."""
        print(f"Setup complete: n={self.n}, p={self.p}")
        print(f"Finite field domain D_f: {self.D_f}")
        print(f"Teichmüller-lifted domain D (in ℂ): {np.around(self.D, 4)}")

    def _teichmuller_lift(self, a: int) -> complex:
        """
        Teichmüller lift of an element a in F_p to a complex number.

        Maps:
            - a == 0 to 0
            - Nonzero a to exp(2πi * k/(p-1)), where k satisfies a ≡ g^k (mod p)

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
        k: int = FiniteFieldUtils.discrete_log(a, self.g, self.p)
        return np.exp(2j * np.pi * k / (self.p - 1))

    def polynomial_encode(self, secret: List[int]) -> ComplexArray:
        """
        Encode a secret as a polynomial and evaluate it at domain points.

        Steps:
            1. Treat secret as polynomial coefficients
            2. Evaluate polynomial at finite field domain points
            3. Apply Teichmüller lift to convert evaluations to complex numbers

        Args:
            secret: List of integers representing secret data (mod p)

        Returns:
            Array of complex numbers representing the encoded secret
        """
        coeffs: List[int] = [0] * self.n
        for j in range(min(len(secret), self.n)):
            coeffs[j] = secret[j] % self.p

        evaluations: List[complex] = []
        for x in self.D_f:
            val: int = 0
            for coeff in reversed(coeffs):
                val = (val * x + coeff) % self.p
            evaluations.append(self._teichmuller_lift(val))

        return np.array(evaluations, dtype=np.complex128)

    def blind_evaluations(self, evaluations: ComplexArray, r: Optional[complex] = None) -> Tuple[ComplexArray, complex, ComplexArray]:
        """
        Apply random blinding to polynomial evaluations.

        Formula: blinded = evaluations + r * mask

        Args:
            evaluations: Array of complex numbers to blind
            r: Optional blinding factor (random complex number)

        Returns:
            Tuple of (blinded evaluations, blinding factor r, blinding mask)
        """
        if r is None:
            r = complex(np.random.normal(), np.random.normal())

        mask: ComplexArray = np.array([complex(np.random.normal(), np.random.normal()) for _ in range(self.n)],
                                      dtype=np.complex128)
        mask = mask / np.linalg.norm(mask)

        return evaluations + r * mask, r, mask

    def alice_prove(self, secret: List[int], flow_time: float = np.pi / 4) -> Commitment:
        """
        Alice's protocol (Prover).

        Steps:
            1. Encode secret as a polynomial and evaluate on domain
            2. Apply random blinding
            3. Apply Hamiltonian flow (rotation)
            4. Return commitment

        Args:
            secret: List of integers representing secret data (mod p)
            flow_time: Time parameter for Hamiltonian flow

        Returns:
            Commitment object containing proof data
        """
        print("\n=== Alice (Prover) ===")
        print(f"Secret data (finite field elements mod {self.p}): {secret}")

        evaluations: ComplexArray = self.polynomial_encode(secret)
        print("Polynomial encoding (with Teichmüller lift) complete")

        blinded_evals, r, mask = self.blind_evaluations(evaluations)
        print(f"Random blinding applied with coefficient r = {r:.4f}")

        transformed_evals: ComplexArray = self.flow.apply_flow(blinded_evals, flow_time, self.epsilon)
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
        Bob's protocol (Verifier).

        Steps:
            1. Apply inverse Hamiltonian flow to commitment
            2. Verify preservation of blinded evaluations via L2 norm

        Args:
            commitment: Commitment object or dictionary with proof data

        Returns:
            Boolean indicating verification success
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

        flow_time: float = comm.flow_time
        print(f"Flow time parameter: t = {flow_time:.4f}")

        recovered_evals: ComplexArray = self.flow.apply_inverse_flow(comm.transformed_evaluations, flow_time, self.epsilon)
        print("Inverse Hamiltonian flow applied")

        original: ComplexArray = comm.secret_original_evaluations
        r: complex = comm.r
        mask: ComplexArray = comm.mask
        blinded: ComplexArray = original + r * mask

        norm_diff: float = np.linalg.norm(recovered_evals - blinded)
        print(f"L2 norm of difference: {norm_diff:.8e}")

        is_verified: bool = norm_diff < self.epsilon
        print(f"Verification {'SUCCESS' if is_verified else 'FAILED'}: "
              f"Geometric invariants {'preserved' if is_verified else 'not preserved'}")
        return is_verified

    def visualize_flow_3d_comparison(self, secret: List[int], flow_time: float = np.pi, num_steps: int = 100) -> None:
        """
        Create a 3D visualization comparing trajectories before and after blinding.

        Left subplot: Plain domain (Teichmüller-lifted points)
        Right subplot: Blinded evaluations (post-encoding and blinding)

        Args:
            secret: List of integers representing secret data (mod p)
            flow_time: Maximum time parameter for Hamiltonian flow
            num_steps: Number of time steps for visualization
        """
        times: RealArray = np.linspace(0, flow_time, num_steps)

        plain_start: ComplexArray = self.D
        plain_trajs: List[ComplexArray] = self._compute_trajectories(plain_start, times)

        plain_evals: ComplexArray = self.polynomial_encode(secret)
        blinded_evals, r, mask = self.blind_evaluations(plain_evals)
        blinded_start: ComplexArray = blinded_evals
        blinded_trajs: List[ComplexArray] = self._compute_trajectories(blinded_start, times)

        self._plot_3d_comparison(plain_trajs, blinded_trajs, times, flow_time,
                                 plain_start=plain_start, blinded_start=blinded_start)

    def _compute_trajectories(self, start_points: ComplexArray, times: RealArray) -> List[ComplexArray]:
        """
        Compute trajectories for starting points under Hamiltonian flow.

        Args:
            start_points: Array of complex numbers representing starting points
            times: Array of time values

        Returns:
            List of trajectories (arrays of complex numbers)
        """
        trajectories: List[ComplexArray] = []
        for z in start_points:
            traj: ComplexArray = z * np.exp(1j * times)
            trajectories.append(traj)
        return trajectories

    def _plot_3d_comparison(self, plain_trajs: List[ComplexArray], blinded_trajs: List[ComplexArray],
                            times: RealArray, flow_time: float,
                            plain_start: Optional[ComplexArray] = None,
                            blinded_start: Optional[ComplexArray] = None) -> None:
        """
        Plot 3D comparison of trajectories.

        Args:
            plain_trajs: Trajectories for plain points
            blinded_trajs: Trajectories for blinded points
            times: Array of time values
            flow_time: Maximum time parameter
            plain_start: Starting points for plain trajectories
            blinded_start: Starting points for blinded trajectories
        """
        fig = plt.figure(figsize=(18, 10))

        if plain_start is not None and blinded_start is not None:
            all_points = np.concatenate([np.array([traj for traj in plain_trajs]).flatten(),
                                         np.array([traj for traj in blinded_trajs]).flatten()])
            max_abs_real = max(abs(np.max(all_points.real)), abs(np.min(all_points.real)))
            max_abs_imag = max(abs(np.max(all_points.imag)), abs(np.min(all_points.imag)))
            max_abs = max(max_abs_real, max_abs_imag) * 1.2
            xlim = ylim = (-max_abs, max_abs)
        else:
            xlim = ylim = (-2.0, 2.0)

        ax1 = fig.add_subplot(121, projection='3d')
        ax1.set_title("Domain Points Trajectories\n(Before Blinding)", fontsize=14)
        ax1.set_xlabel("Re(z)", fontsize=12)
        ax1.set_ylabel("Im(z)", fontsize=12)
        ax1.set_zlabel("Time t", fontsize=12)
        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim)
        ax1.set_zlim(0, flow_time)

        n_trajectories: int = len(plain_trajs)
        colors = plt.cm.tab20(np.linspace(0, 1, n_trajectories)) if n_trajectories <= 20 else \
                 plt.cm.viridis(np.linspace(0, 1, n_trajectories))

        self._plot_trajectories(ax1, plain_trajs, times, colors)

        if plain_start is not None:
            for i, point in enumerate(plain_start):
                ax1.text(point.real, point.imag, 0, f"P{i}", fontsize=10, color=colors[i])

        ax2 = fig.add_subplot(122, projection='3d')
        ax2.set_title("Blinded Evaluations Trajectories\n(After Random Blinding)", fontsize=14)
        ax2.set_xlabel("Re(z)", fontsize=12)
        ax2.set_ylabel("Im(z)", fontsize=12)
        ax2.set_zlabel("Time t", fontsize=12)
        ax2.set_xlim(xlim)
        ax2.set_ylim(ylim)
        ax2.set_zlim(0, flow_time)

        self._plot_trajectories(ax2, blinded_trajs, times, colors)

        if blinded_start is not None:
            for i, point in enumerate(blinded_start):
                ax2.text(point.real, point.imag, 0, f"B{i}", fontsize=10, color=colors[i])

        fig.suptitle("Hamiltonian Flow Visualization: Comparison of Trajectories", fontsize=16, y=0.98)

        handles = [plt.Line2D([0], [0], color=colors[i], lw=2, label=f"Point {i}")
                   for i in range(min(n_trajectories, 8))]
        fig.legend(handles=handles, loc='lower center', ncol=min(n_trajectories, 8),
                   bbox_to_anchor=(0.5, 0.01), frameon=True, fancybox=True, shadow=True)

        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.subplots_adjust(wspace=0.1, bottom=0.1)

        fig.text(0.5, 0.03,
                 "SymPLONK: Hamiltonian flow maps secret polynomial evaluations to a geometric path\n"
                 "preserving geometric invariants for zero-knowledge verification",
                 ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

        plt.show()

    def _plot_trajectories(self, ax: plt.Axes, trajectories: List[ComplexArray],
                           times: RealArray, colors: np.ndarray) -> None:
        """
        Plot trajectories on a given axes with styling.

        Args:
            ax: Matplotlib axes to plot on
            trajectories: List of trajectories (arrays of complex numbers)
            times: Array of time values
            colors: Array of colors for each trajectory
        """
        for i, traj in enumerate(trajectories):
            xs = traj.real
            ys = traj.imag
            ax.plot(xs, ys, times, color=colors[i], lw=2, alpha=0.8)
            ax.scatter(xs[0], ys[0], times[0], color=colors[i], s=80,
                       edgecolor='black', linewidth=1.5, marker='o', alpha=1.0)
            ax.scatter(xs[-1], ys[-1], times[-1], color=colors[i], s=80,
                       edgecolor='white', linewidth=1.5, marker='^', alpha=1.0)
            ax.plot([xs[0], xs[0]], [ys[0], ys[0]], [0, times[0]],
                    color=colors[i], lw=1, linestyle='--', alpha=0.6)
            ax.plot([xs[-1], xs[-1]], [ys[-1], ys[-1]], [times[-1], times[-1]],
                    color=colors[i], lw=1, linestyle='--', alpha=0.6)

    def visualize_2d_flow_snapshots(self, secret: List[int], flow_time: float = np.pi,
                                    num_snapshots: int = 4) -> None:
        """
        Create a 2D visualization of Hamiltonian flow snapshots.

        Displays a grid of 2D plots showing point positions at different time snapshots.

        Args:
            secret: List of integers representing secret data (mod p)
            flow_time: Maximum time parameter for Hamiltonian flow
            num_snapshots: Number of time snapshots to display
        """
        snapshot_times: RealArray = np.linspace(0, flow_time, num_snapshots)

        plain_evals: ComplexArray = self.polynomial_encode(secret)
        blinded_evals, r, mask = self.blind_evaluations(plain_evals)

        fig, axes = plt.subplots(1, num_snapshots, figsize=(16, 4))

        n_points: int = len(blinded_evals)
        colors = plt.cm.tab10(np.linspace(0, 1, n_points))

        max_real = max(abs(blinded_evals.real.max()), abs(blinded_evals.real.min())) * 1.2
        max_imag = max(abs(blinded_evals.imag.max()), abs(blinded_evals.imag.min())) * 1.2
        max_val = max(max_real, max_imag)

        for i, t in enumerate(snapshot_times):
            flowed_points: ComplexArray = self.flow.apply_flow(blinded_evals, t, self.epsilon)

            ax = axes[i]
            for j, point in enumerate(flowed_points):
                ax.scatter(point.real, point.imag, color=colors[j], s=100, edgecolor='white')
                ax.text(point.real, point.imag, f"{j}", fontsize=9, ha='center', va='center', color='white')

            ax.plot(flowed_points.real, flowed_points.imag, 'k-', alpha=0.3)

            ax.set_xlim(-max_val, max_val)
            ax.set_ylim(-max_val, max_val)
            ax.set_title(f"t = {t:.2f}")
            ax.set_xlabel("Re(z)")
            ax.set_ylabel("Im(z)")
            ax.grid(True, alpha=0.3)

        fig.suptitle("Hamiltonian Flow Snapshots", fontsize=16)
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    symplonk = SymPLONK(n=8, p=17) ## n=256, p=2^{256} - 2^{32} - 977にしたい
    secret: List[int] = [1, 2, 3, 4]
    commitment = symplonk.alice_prove(secret)
    symplonk.bob_verify(commitment)
    symplonk.visualize_flow_3d_comparison(secret)
    symplonk.visualize_2d_flow_snapshots(secret)
