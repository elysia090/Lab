import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import math
from typing import List, Tuple, Dict, Optional, Union, Any, TypeVar
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

class SymPLONK:
    """
    SymPLONK:
    Demonstrates a geometric zero-knowledge proof protocol by mapping a finite field
    interpolation domain to complex numbers via the Teichmüller lift, then applying
    a Hamiltonian flow (rotation) to hide and later verify secret information.
    """
    def __init__(self, n: int = 8, p: int = 17, epsilon: float = 1e-8):
        self.n = n                  # Number of interpolation points
        self.p = p                  # Prime for finite field F_p
        self.epsilon = epsilon      # Numerical error tolerance
        
        # Define the interpolation domain in F_p: D_f = { ω_f^0, ω_f^1, ..., ω_f^(n-1) }
        self.omega_f = self._find_primitive_nth_root(n, p)
        if self.omega_f is None:
            raise ValueError(f"Could not find a primitive {n}th root in F_{p}")
        self.D_f = [pow(self.omega_f, i, p) for i in range(n)]
        
        # Find a primitive element g of F_p* (used for computing discrete logs)
        self.g = self._find_primitive_root(p)
        if self.g is None:
            raise ValueError(f"Could not find a primitive root for F_{p}")
        
        # Apply the Teichmüller lift to each element in D_f to obtain complex points
        self.D = np.array([self.teichmuller_lift(a) for a in self.D_f], dtype=np.complex128)
        
        # Define Hamiltonian function H(z) = 1/2 |z|^2 (for potential energy, etc.)
        self.H_func = lambda z: 0.5 * np.abs(z)**2

        print(f"Setup complete: n={self.n}, p={self.p}")
        print(f"Finite field domain D_f: {self.D_f}")
        print(f"Teichmüller-lifted domain D (in ℂ): {np.around(self.D, 4)}")
    
    def _find_primitive_nth_root(self, n: int, p: int) -> Optional[int]:
        """
        Find a primitive nth root of unity ω_f in F_p.
        Requires that n divides (p - 1).
        """
        if (p - 1) % n != 0:
            print(f"Warning: n={n} is not a divisor of p-1={p-1}")
            
        # Precompute proper divisors for efficiency
        divisors = self._divisors(n)
        proper_divisors = [d for d in divisors if d < n]
        
        for candidate in range(2, p):
            if pow(candidate, n, p) != 1:
                continue
            if all(pow(candidate, d, p) != 1 for d in proper_divisors):
                return candidate
        return None

    @staticmethod
    @lru_cache(maxsize=128)
    def _divisors(n: int) -> List[int]:
        """Compute all divisors of the integer n."""
        divs = set()
        for i in range(1, int(math.sqrt(n)) + 1):
            if n % i == 0:
                divs.add(i)
                divs.add(n // i)
        return sorted(list(divs))
    
    def _find_primitive_root(self, p: int) -> Optional[int]:
        """Find a primitive element (generator) of F_p (of order p-1)."""
        factors = self._prime_factors(p - 1)
        coprime_exponents = [(p - 1) // f for f in factors]
        
        for candidate in range(2, p):
            if all(pow(candidate, exp, p) != 1 for exp in coprime_exponents):
                return candidate
        return None
    
    @staticmethod
    def _prime_factors(n: int) -> List[int]:
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

    def discrete_log(self, a: int, g: int, p: int) -> int:
        """
        Compute the discrete logarithm: find k such that g^k ≡ a (mod p).
        For small p, brute-force search is used; for larger p, Baby-step Giant-step can be applied.
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

    def teichmuller_lift(self, a: int) -> complex:
        """
        Teichmüller lift of an element a in F_p.
          - If a == 0, returns 0.
          - For nonzero a, find k such that a ≡ g^k (mod p), then return exp(2πi * k/(p-1)).
        """
        if a == 0:
            return complex(0.0, 0.0)
        if self.g is None:
            raise ValueError("Primitive root is not set")
        k = self.discrete_log(a, self.g, self.p)
        return np.exp(2j * np.pi * k / (self.p - 1))
    
    def apply_flow(self, points: ComplexArray, t: float) -> ComplexArray:
        """
        Apply the Hamiltonian flow (rotation) to complex points.
        For circular flow, the analytical solution is: z(t) = z(0) * exp(i*t)
        """
        if abs(t) < self.epsilon:
            return points
        return points * np.exp(1j * t)
    
    def apply_inverse_flow(self, points: ComplexArray, t: float) -> ComplexArray:
        """Apply the inverse Hamiltonian flow."""
        return self.apply_flow(points, -t)
    
    def polynomial_encode(self, secret: List[int]) -> ComplexArray:
        """
        Encode the secret (a list of elements in F_p) as a polynomial f(x) and evaluate it at the domain points.
        Then, apply the Teichmüller lift to convert each evaluation to a complex number.
        Horner's method is used for efficient polynomial evaluation.
        """
        coeffs = [0] * self.n
        for j in range(min(len(secret), self.n)):
            coeffs[j] = secret[j] % self.p
            
        evaluations = []
        for x in self.D_f:
            val = 0
            for coeff in reversed(coeffs):
                val = (val * x + coeff) % self.p
            evaluations.append(self.teichmuller_lift(val))
            
        return np.array(evaluations, dtype=np.complex128)
    
    def blind_evaluations(self, evaluations: ComplexArray, r: Optional[complex] = None) -> Tuple[ComplexArray, complex, ComplexArray]:
        """
        Apply random blinding to the polynomial evaluations (complex numbers).
        A normalized random mask is generated for numerical stability.
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
        """
        print("\n=== Alice (Prover) ===")
        print(f"Secret data (finite field elements mod {self.p}): {secret}")
        
        evaluations = self.polynomial_encode(secret)
        print("Polynomial encoding (with Teichmüller lift) complete")
        
        blinded_evals, r, mask = self.blind_evaluations(evaluations)
        print(f"Random blinding applied with coefficient r = {r:.4f}")
        
        transformed_evals = self.apply_flow(blinded_evals, flow_time)
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
        
        recovered_evals = self.apply_inverse_flow(comm.transformed_evaluations, flow_time)
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
    
    def visualize_domain_transformation(self, flow_time: float = np.pi/4, num_steps: int = 50, save_gif: bool = False) -> None:
        """
        Visualize the transformation of the interpolation domain under Hamiltonian flow.
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.grid(True)
        ax.set_aspect('equal')
        ax.set_xlabel('Re(z)')
        ax.set_ylabel('Im(z)')
        ax.set_title('SymPLONK: Hamiltonian Flow Transformation of Interpolation Domain')
        
        domain_points = self.D.copy()
        all_trajectories = [domain_points.copy()]
        
        # Draw the unit circle
        theta = np.linspace(0, 2*np.pi, 100)
        circle = np.exp(1j * theta)
        ax.plot(circle.real, circle.imag, 'k-', alpha=0.3)
        
        delta_t = flow_time / num_steps
        for step in range(1, num_steps + 1):
            t = step * delta_t
            rotated_points = domain_points * np.exp(1j * t)
            all_trajectories.append(rotated_points)
        
        colors = cm.viridis(np.linspace(0, 1, self.n))
        scatter = ax.scatter(self.D.real, self.D.imag, c=colors, s=100, label='Domain Points')
        for i in range(self.n):
            trajectory = np.array([traj[i] for traj in all_trajectories])
            ax.plot(trajectory.real, trajectory.imag, '-', color=colors[i], alpha=0.2)
        
        time_text = ax.text(0.02, 0.95, f't = 0.00', transform=ax.transAxes, fontsize=12)
        
        def update(frame):
            points = all_trajectories[frame]
            scatter.set_offsets(np.column_stack([points.real, points.imag]))
            current_time = delta_t * frame
            time_text.set_text(f't = {current_time:.2f}')
            ax.set_title(f'SymPLONK Hamiltonian Flow: t = {current_time:.2f}')
            return scatter, time_text
        
        ani = FuncAnimation(fig, update, frames=len(all_trajectories), interval=100, blit=True)
        if save_gif:
            ani.save('symplonk_flow.gif', writer='pillow', fps=10, dpi=100)
            print("Animation saved as 'symplonk_flow.gif'")
            
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def visualize_flow_3d_enhanced(self, flow_time: float = np.pi, num_points: int = 16, num_steps: int = 100) -> None:
        """
        Enhanced 3D visualization of the Hamiltonian flow on a complex grid.
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        grid_size = int(np.sqrt(num_points))
        x = np.linspace(-1, 1, grid_size)
        y = np.linspace(-1, 1, grid_size)
        X, Y = np.meshgrid(x, y)
        initial_points = X.flatten() + 1j * Y.flatten()
        
        colormap = cm.coolwarm
        colors = colormap(np.linspace(0, 1, len(initial_points)))
        times = np.linspace(0, flow_time, num_steps)
        
        # Plot grid planes at t=0 and t=flow_time
        for t_idx in [0, num_steps-1]:
            t_val = times[t_idx]
            tz_plane = np.ones((grid_size, grid_size)) * t_val
            ax.plot_surface(X, Y, tz_plane, alpha=0.1, color='gray')
        
        # Plot trajectories for each initial point
        for idx, z0 in enumerate(initial_points):
            zs = z0 * np.exp(1j * times)
            xs, ys = zs.real, zs.imag
            for i in range(len(times) - 1):
                t_ratio = times[i] / flow_time
                ax.plot([xs[i], xs[i+1]], [ys[i], ys[i+1]], [times[i], times[i+1]],
                        color=colormap(t_ratio), alpha=0.7, linewidth=1.5)
            ax.scatter(z0.real, z0.imag, 0, color=colors[idx], s=40, edgecolor='black')
            ax.scatter(xs[-1], ys[-1], flow_time, color=colors[idx], s=40, edgecolor='white')
        
        ax.set_xlabel('Re(z)', fontsize=12)
        ax.set_ylabel('Im(z)', fontsize=12)
        ax.set_zlabel('Time t', fontsize=12)
        ax.set_title('SymPLONK: 3D Visualization of Hamiltonian Flow', fontsize=14)
        ax.view_init(elev=30, azim=45)
        plt.tight_layout()
        
        # Optionally rotate the view for better visualization
        if hasattr(plt, 'ion'):
            plt.ion()
            print("Rotating view for better visualization...")
            for angle in range(0, 360, 5):
                ax.view_init(elev=30, azim=angle)
                plt.draw()
                plt.pause(0.05)
            plt.ioff()
            
        plt.show()
    
    def visualize_flow_3d_comparison(self, secret: List[int], flow_time: float = np.pi, num_points: int = 16, num_steps: int = 100) -> None:
        """
        3D visualization comparing the trajectories before and after branding.
        
        Left subplot: Plain domain (Teichmüller-lifted points without blinding)
        Right subplot: Masked evaluations (after applying polynomial encoding and random blinding)
        Both are rotated according to the analytical Hamiltonian flow.
        """
        times = np.linspace(0, flow_time, num_steps)
        
        # Compute plain trajectories from the precomputed Teichmüller-lifted domain (self.D)
        plain_start = self.D
        plain_trajs = []
        for z in plain_start:
            traj = z * np.exp(1j * times)
            plain_trajs.append(traj)
        
        # Compute polynomial encoding and then apply random blinding (without flow)
        plain_evals = self.polynomial_encode(secret)
        blinded_evals, r, mask = self.blind_evaluations(plain_evals)
        blinded_start = blinded_evals
        blinded_trajs = []
        for z in blinded_start:
            traj = z * np.exp(1j * times)
            blinded_trajs.append(traj)
        
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
        for i, traj in enumerate(plain_trajs):
            xs = traj.real
            ys = traj.imag
            ax1.plot(xs, ys, times, color=colors[i], lw=2)
            ax1.scatter(xs[0], ys[0], times[0], color=colors[i], s=50, edgecolor='black')
            ax1.scatter(xs[-1], ys[-1], times[-1], color=colors[i], s=50, edgecolor='white')
        
        # Right subplot: After branding
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.set_title("After Branding")
        ax2.set_xlabel("Re(z)")
        ax2.set_ylabel("Im(z)")
        ax2.set_zlabel("Time t")
        ax2.set_xlim(-1.5, 1.5)
        ax2.set_ylim(-1.5, 1.5)
        ax2.set_zlim(0, flow_time)
        for i, traj in enumerate(blinded_trajs):
            xs = traj.real
            ys = traj.imag
            ax2.plot(xs, ys, times, color=colors[i], lw=2)
            ax2.scatter(xs[0], ys[0], times[0], color=colors[i], s=50, edgecolor='black')
            ax2.scatter(xs[-1], ys[-1], times[-1], color=colors[i], s=50, edgecolor='white')
        
        plt.tight_layout()
        plt.show()

def run_demo() -> None:
    """Run demonstration of the SymPLONK protocol with 3D visualizations and comparison."""
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
    
    print("\nVisualizing the Hamiltonian flow transformation...")
    symplonk.visualize_domain_transformation(flow_time=np.pi/4, save_gif=True)
    
    print("\nVisualizing enhanced 3D Hamiltonian flow...")
    symplonk.visualize_flow_3d_enhanced(flow_time=np.pi)
    
    print("\nVisualizing 3D comparison (Before vs. After Branding)...")
    symplonk.visualize_flow_3d_comparison(secret, flow_time=np.pi, num_points=16, num_steps=100)
    
    print("\nDemonstration complete!")

if __name__ == "__main__":
    run_demo()
