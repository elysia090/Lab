import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # required for 3D plotting
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Tuple, Optional, Union, Dict, Any, TypeVar
import numpy.typing as npt

# Type aliases for clarity
ComplexArray = npt.NDArray[np.complex128]
RealArray = npt.NDArray[np.float64]
T = TypeVar('T')


@dataclass
class Commitment:
    """
    Data structure representing the zero-knowledge proof commitment.

    Attributes:
        transformed_evaluations: Complex array of evaluations after applying Hamiltonian flow.
        flow_time: Time parameter used for the Hamiltonian flow.
        secret_original_evaluations: Complex array of original polynomial evaluations.
        r: Blinding factor (complex).
        mask: Blinding mask (complex array).
    """
    transformed_evaluations: ComplexArray
    flow_time: float
    secret_original_evaluations: ComplexArray
    r: complex
    mask: ComplexArray


class FiniteFieldUtils:
    """
    Utility class for finite field operations.
    Provides methods for computing divisors, prime factors, primitive roots, and discrete logarithms.
    """

    @staticmethod
    @lru_cache(maxsize=128)
    def divisors(n: int) -> List[int]:
        """Return sorted list of all divisors of n."""
        divs = set()
        for i in range(1, int(math.sqrt(n)) + 1):
            if n % i == 0:
                divs.add(i)
                divs.add(n // i)
        return sorted(divs)

    @staticmethod
    def prime_factors(n: int) -> List[int]:
        """Return unique prime factors of n."""
        factors = []
        i = 2
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
        Find a primitive n-th root of unity in F_p.
        Returns candidate ω satisfying ω^n ≡ 1 (mod p) and ω^k ≠ 1 for all proper divisors k < n.
        """
        if (p - 1) % n != 0:
            print(f"Warning: n={n} does not divide p-1={p-1}")
        proper_divisors = [d for d in cls.divisors(n) if d < n]

        for candidate in range(2, p):
            if pow(candidate, n, p) != 1:
                continue
            if all(pow(candidate, d, p) != 1 for d in proper_divisors):
                return candidate
        return None

    @classmethod
    def find_primitive_root(cls, p: int) -> Optional[int]:
        """
        Find a primitive root (generator) of F_p.
        Returns an element that generates F_p*.
        """
        factors = cls.prime_factors(p - 1)
        exponents = [(p - 1) // f for f in factors]
        for candidate in range(2, p):
            if all(pow(candidate, exp, p) != 1 for exp in exponents):
                return candidate
        return None

    @staticmethod
    def discrete_log(a: int, g: int, p: int) -> int:
        """
        Compute the discrete logarithm: find k such that g^k ≡ a (mod p).
        For small p, a brute-force search is used; for larger p, the baby-step giant-step method is applied.
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
                    return i * m + baby_steps[value]
                value = (value * g_inv) % p
        raise ValueError(f"No discrete log exists: a={a}, g={g}, p={p}")


class HamiltonianFlow:
    """
    Implements Hamiltonian flow operations for complex numbers.
    For the quadratic Hamiltonian H(z) = 1/2 |z|^2, the flow is given by a rotation:
    z(t) = z(0) * exp(i * t).
    """

    @staticmethod
    def hamiltonian_function(z: complex) -> float:
        """Compute H(z) = 1/2 * |z|^2."""
        return 0.5 * np.abs(z) ** 2

    @staticmethod
    def apply_flow(points: ComplexArray, t: float, epsilon: float = 1e-8) -> ComplexArray:
        """
        Apply Hamiltonian flow (rotation) with time parameter t.
        If |t| is negligible (below epsilon), returns points unchanged.
        """
        if abs(t) < epsilon:
            return points
        return points * np.exp(1j * t)

    @classmethod
    def apply_inverse_flow(cls, points: ComplexArray, t: float, epsilon: float = 1e-8) -> ComplexArray:
        """Apply the inverse Hamiltonian flow by using time -t."""
        return cls.apply_flow(points, -t, epsilon)


class SymPLONK:
    """
    SymPLONK: Geometric Zero-Knowledge Proof Protocol.
    Maps a finite field interpolation domain to ℂ via the Teichmüller lift,
    applies random blinding and a Hamiltonian flow (rotation) to hide secret information,
    and verifies the commitment using the inverse flow.
    """

    def __init__(self, n: int = 8, p: int = 17, epsilon: float = 1e-8) -> None:
        """
        Initialize protocol parameters.
        Args:
            n: Number of interpolation points.
            p: Prime modulus for F_p.
            epsilon: Numerical tolerance.
        """
        self.n = n
        self.p = p
        self.epsilon = epsilon

        self.omega_f = FiniteFieldUtils.find_primitive_nth_root(n, p)
        if self.omega_f is None:
            raise ValueError(f"Primitive {n}th root not found in F_{p}.")
        self.D_f: List[int] = [pow(self.omega_f, i, p) for i in range(n)]

        self.g = FiniteFieldUtils.find_primitive_root(p)
        if self.g is None:
            raise ValueError(f"Primitive root not found in F_{p}.")

        # Teichmüller lift of finite field domain to ℂ.
        self.D: ComplexArray = np.array([self._teichmuller_lift(a) for a in self.D_f],
                                         dtype=np.complex128)
        self.flow = HamiltonianFlow()
        self._log_setup_info()

    def _log_setup_info(self) -> None:
        """Output initialization details."""
        print(f"Setup: n={self.n}, p={self.p}")
        print(f"Finite field domain D_f: {self.D_f}")
        print(f"Teichmüller-lifted domain D: {np.around(self.D, 4)}")

    def _teichmuller_lift(self, a: int) -> complex:
        """
        Teichmüller lift:
          - Maps 0 to 0.
          - For nonzero a, finds k such that a ≡ g^k (mod p) and maps to exp(2πi * k/(p-1)).
        """
        if a == 0:
            return 0.0 + 0.0j
        k = FiniteFieldUtils.discrete_log(a, self.g, self.p)
        return np.exp(2j * np.pi * k / (self.p - 1))

    def polynomial_encode(self, secret: List[int]) -> ComplexArray:
        """
        Encode secret data as a polynomial and evaluate it over the finite field domain.
        Coefficients are taken modulo p and the resulting evaluations are Teichmüller lifted.
        This version uses a vectorized Horner’s method.
        """
        # Build coefficients array (pad with zeros if necessary) and ensure they are modulo p.
        coeffs = (np.array(secret + [0] * (self.n - len(secret)), dtype=int)) % self.p
        # Get the finite field domain as a NumPy array.
        x_vals = np.array(self.D_f, dtype=int)
        # Vectorized Horner's method over x_vals:
        # Start with an array of zeros (one for each x in the domain)
        vals = np.zeros_like(x_vals, dtype=int)
        # Iterate over coefficients in reversed order
        for c in coeffs[::-1]:
            vals = (vals * x_vals + c) % self.p
        # Vectorize the Teichmüller lift over the computed values.
        vectorized_lift = np.vectorize(self._teichmuller_lift)
        return vectorized_lift(vals).astype(np.complex128)

    def blind_evaluations(self, evaluations: ComplexArray, r: Optional[complex] = None
                          ) -> Tuple[ComplexArray, complex, ComplexArray]:
        """
        Apply random blinding to the evaluations:
            blinded = evaluations + r * mask.
        If r is not provided, a random complex number is generated.
        The mask is a normalized complex vector.
        """
        if r is None:
            r = complex(np.random.normal(), np.random.normal())
        mask = np.array([complex(np.random.normal(), np.random.normal()) for _ in range(self.n)],
                        dtype=np.complex128)
        mask = mask / np.linalg.norm(mask)
        return evaluations + r * mask, r, mask

    def alice_prove(self, secret: List[int], flow_time: float = np.pi / 4) -> Commitment:
        """
        Prover (Alice): Encode the secret, apply blinding, then apply the Hamiltonian flow,
        and generate a commitment.
        """
        print("\n=== Alice (Prover) ===")
        print(f"Secret (mod {self.p}): {secret}")
        evaluations = self.polynomial_encode(secret)
        print("Polynomial encoding complete.")

        blinded_evals, r, mask = self.blind_evaluations(evaluations)
        print(f"Random blinding applied with r = {r:.4f}")

        transformed_evals = self.flow.apply_flow(blinded_evals, flow_time, self.epsilon)
        print(f"Hamiltonian flow applied with t = {flow_time:.4f}")

        return Commitment(
            transformed_evaluations=transformed_evals,
            flow_time=flow_time,
            secret_original_evaluations=evaluations,
            r=r,
            mask=mask
        )

    def bob_verify(self, commitment: Union[Commitment, Dict[str, Any]]) -> bool:
        """
        Verifier (Bob): Apply the inverse Hamiltonian flow and check that
        the recovered evaluations match the original blinded evaluations.
        """
        print("\n=== Bob (Verifier) ===")
        if isinstance(commitment, dict):
            comm = Commitment(**commitment)
        else:
            comm = commitment

        print(f"Flow time: t = {comm.flow_time:.4f}")
        recovered = self.flow.apply_inverse_flow(comm.transformed_evaluations, comm.flow_time, self.epsilon)
        expected = comm.secret_original_evaluations + comm.r * comm.mask

        diff_norm = np.linalg.norm(recovered - expected)
        print(f"L2 norm difference: {diff_norm:.8e}")

        verified = diff_norm < self.epsilon
        print(f"Verification {'SUCCESS' if verified else 'FAILED'}")
        return verified

    def _compute_trajectories(self, start_points: ComplexArray, times: RealArray) -> np.ndarray:
        """
        Compute trajectories for each starting point under the Hamiltonian flow.
        This function uses broadcasting to vectorize the computation.
        Returns an array of shape (num_points, num_times).
        """
        return start_points[:, None] * np.exp(1j * times)

    def visualize_flow_3d_comparison(self, secret: List[int],
                                     flow_time: float = np.pi,
                                     num_steps: int = 100) -> None:
        """
        3D visualization: Compare trajectories on the Teichmüller-lifted domain (plain)
        and after blinding (transformed) side-by-side.
        """
        times: RealArray = np.linspace(0, flow_time, num_steps)
        plain_trajs = self._compute_trajectories(self.D, times)

        plain_evals = self.polynomial_encode(secret)
        blinded_evals, _, _ = self.blind_evaluations(plain_evals)
        blinded_trajs = self._compute_trajectories(blinded_evals, times)

        self._plot_3d_comparison(plain_trajs, blinded_trajs, times, flow_time, self.D, blinded_evals)

    def _plot_3d_comparison(self, plain_trajs: np.ndarray, blinded_trajs: np.ndarray,
                            times: RealArray, flow_time: float,
                            plain_start: ComplexArray, blinded_start: ComplexArray) -> None:
        """Plot 3D trajectories for plain and blinded points side by side."""
        fig = plt.figure(figsize=(18, 10))

        # Determine common axis limits using vectorized operations.
        all_points = np.concatenate((plain_trajs.flatten(), blinded_trajs.flatten()))
        max_val = 1.2 * max(np.abs(all_points.real).max(), np.abs(all_points.imag).max())
        xlim = ylim = (-max_val, max_val)

        # Plot plain trajectories.
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.set_title("Domain Points Trajectories (Before Blinding)", fontsize=14)
        ax1.set_xlabel("Re(z)")
        ax1.set_ylabel("Im(z)")
        ax1.set_zlabel("Time t")
        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim)
        ax1.set_zlim(0, flow_time)
        colors = plt.cm.tab20(np.linspace(0, 1, plain_trajs.shape[0]))
        self._plot_trajectories(ax1, plain_trajs, times, colors)
        for i, point in enumerate(plain_start):
            ax1.text(point.real, point.imag, 0, f"P{i}", color=colors[i], fontsize=10)

        # Plot blinded trajectories.
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.set_title("Blinded Evaluations Trajectories (After Blinding)", fontsize=14)
        ax2.set_xlabel("Re(z)")
        ax2.set_ylabel("Im(z)")
        ax2.set_zlabel("Time t")
        ax2.set_xlim(xlim)
        ax2.set_ylim(ylim)
        ax2.set_zlim(0, flow_time)
        self._plot_trajectories(ax2, blinded_trajs, times, colors)
        for i, point in enumerate(blinded_start):
            ax2.text(point.real, point.imag, 0, f"B{i}", color=colors[i], fontsize=10)

        fig.suptitle("Hamiltonian Flow Visualization: Trajectory Comparison", fontsize=16)
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.show()

    def _plot_trajectories(self, ax: plt.Axes, trajectories: np.ndarray,
                           times: RealArray, colors: np.ndarray) -> None:
        """Helper function to plot trajectories on the given axes."""
        num_points = trajectories.shape[0]
        for i in range(num_points):
            xs = trajectories[i].real
            ys = trajectories[i].imag
            ax.plot(xs, ys, times, color=colors[i], lw=2, alpha=0.8)
            ax.scatter(xs[0], ys[0], times[0], color=colors[i], s=80, edgecolor='black', marker='o')
            ax.scatter(xs[-1], ys[-1], times[-1], color=colors[i], s=80, edgecolor='white', marker='^')

    def visualize_2d_flow_snapshots(self, secret: List[int],
                                    flow_time: float = np.pi,
                                    num_snapshots: int = 4) -> None:
        """
        2D visualization: Display snapshots of the blinded evaluations at different times.
        """
        snapshot_times = np.linspace(0, flow_time, num_snapshots)
        plain_evals = self.polynomial_encode(secret)
        blinded_evals, _, _ = self.blind_evaluations(plain_evals)

        fig, axes = plt.subplots(1, num_snapshots, figsize=(16, 4))
        colors = plt.cm.tab10(np.linspace(0, 1, len(blinded_evals)))
        max_val = 1.2 * max(np.abs(blinded_evals.real).max(), np.abs(blinded_evals.imag).max())

        for i, t in enumerate(snapshot_times):
            flowed = self.flow.apply_flow(blinded_evals, t, self.epsilon)
            ax = axes[i]
            ax.scatter(flowed.real, flowed.imag, color=colors, s=100, edgecolor='white')
            for j, pt in enumerate(flowed):
                ax.text(pt.real, pt.imag, f"{j}", fontsize=9, ha='center', va='center', color='white')
            ax.plot(flowed.real, flowed.imag, 'k-', alpha=0.3)
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
    # For demonstration, we use small parameters.
    # In production, use larger values (e.g., n=256, p=2**256 - 2**32 - 977).
    symplonk = SymPLONK(n=8, p=17)
    secret: List[int] = [1, 2, 3, 4]
    commitment = symplonk.alice_prove(secret)
    symplonk.bob_verify(commitment)
    symplonk.visualize_flow_3d_comparison(secret)
    symplonk.visualize_2d_flow_snapshots(secret)
