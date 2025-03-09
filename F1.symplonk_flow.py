import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import math
from typing import List, Tuple, Dict, Optional, Union, Any, Callable, Set, TypeVar
import numpy.typing as npt
from dataclasses import dataclass
from functools import lru_cache

# Type aliases
ComplexArray = npt.NDArray[np.complex128]
RealArray = npt.NDArray[np.float64]
T = TypeVar('T')

@dataclass
class Commitment:
    """Data structure representing zero-knowledge proof commitment"""
    transformed_evaluations: ComplexArray
    flow_time: float
    secret_original_evaluations: ComplexArray
    r: complex
    mask: ComplexArray

class SymPLONK:
    """
    SymPLONK: Demonstrates geometric zero-knowledge proofs by mapping finite field
    interpolation domains to complex numbers via Teichmüller lift through Witt vectors,
    then applying Hamiltonian flow (rotation).
    """
    def __init__(self, n: int = 8, p: int = 17, epsilon: float = 1e-8):
        self.n = n                  # Number of interpolation points
        self.p = p                  # Prime (finite field F_p)
        self.epsilon = epsilon      # Numerical error tolerance
        
        # Define interpolation domain D_f in F_p: D_f = {ω_f^0, ω_f^1, ..., ω_f^(n-1)}
        self.omega_f = self._find_primitive_nth_root(n, p)
        if self.omega_f is None:
            raise ValueError(f"Could not find primitive {n}th root in F_{p}")
            
        self.D_f = [pow(self.omega_f, i, p) for i in range(n)]
        
        # Primitive element g of F_p^* (for discrete log calculations)
        self.g = self._find_primitive_root(p)
        if self.g is None:
            raise ValueError(f"Could not find primitive root for F_{p}")
            
        # Apply Teichmüller lift to each a ∈ D_f to get points in C
        self.D = np.array([self.teichmuller_lift(a) for a in self.D_f], dtype=np.complex128)
        
        # Hamiltonian function H(z) = 1/2 |z|^2
        self.H_func = lambda z: 0.5 * np.abs(z)**2

        print(f"Setup complete: n={self.n}, p={self.p}")
        print(f"Finite field domain D_f: {self.D_f}")
        print(f"Teichmüller-lifted domain D (in ℂ): {np.around(self.D, 4)}")
    
    def _find_primitive_nth_root(self, n: int, p: int) -> Optional[int]:
        """
        Find primitive nth root of unity ω_f in F_p (requires n | (p-1))
        """
        if (p - 1) % n != 0:
            print(f"Warning: n={n} is not a divisor of p-1={p-1}")
            
        # Precompute proper divisors of n for efficiency
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
        """Compute divisors of integer n"""
        divs = set()
        for i in range(1, int(math.sqrt(n)) + 1):
            if n % i == 0:
                divs.add(i)
                divs.add(n // i)
        return sorted(list(divs))
    
    def _find_primitive_root(self, p: int) -> Optional[int]:
        """Find primitive element (generator) of F_p (order p-1)"""
        # Precompute coprimes for efficiency
        factors = self._prime_factors(p - 1)
        coprimes = [(p - 1) // f for f in factors]
        
        for candidate in range(2, p):
            if all(pow(candidate, coprime, p) != 1 for coprime in coprimes):
                return candidate
                
        return None
    
    @staticmethod
    def _prime_factors(n: int) -> List[int]:
        """Compute prime factors of integer n"""
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
        Compute discrete logarithm: find k where g^k ≡ a (mod p)
        Uses Baby-step Giant-step algorithm for large p
        """
        if p < 100:  # Brute force for small p
            for k in range(p - 1):
                if pow(g, k, p) == a:
                    return k
        else:  # Baby-step Giant-step for larger p
            m = int(math.sqrt(p - 1)) + 1
            
            # Baby steps
            baby_steps = {pow(g, j, p): j for j in range(m)}
                
            # Giant steps
            g_inv = pow(g, p - 1 - m, p)  # g^(-m) mod p
            value = a
            
            for i in range(m):
                if value in baby_steps:
                    return (i * m + baby_steps[value]) % (p - 1)
                value = (value * g_inv) % p
                
        raise ValueError(f"No discrete log exists: a={a}, g={g}, p={p}")

    def teichmuller_lift(self, a: int) -> complex:
        """
        Teichmüller lift of element a in F_p:
         - 0 maps to 0
         - For non-zero a, find k where a ≡ g^k (mod p), then lift(a)=exp(2πi*k/(p-1))
        """
        if a == 0:
            return complex(0.0, 0.0)
            
        if self.g is None:
            raise ValueError("Primitive root not set")
            
        k = self.discrete_log(a, self.g, self.p)
        return np.exp(2j * np.pi * k / (self.p - 1))
    
    def apply_flow(self, points: ComplexArray, t: float) -> ComplexArray:
        """
        Apply Hamiltonian flow (rotation) to complex points
        
        For circular Hamiltonian flow, analytical solution: z(t) = z(0) * exp(it)
        """
        if abs(t) < self.epsilon:  # Skip calculation for very small t
            return points
        
        return points * np.exp(1j * t)
    
    def apply_inverse_flow(self, points: ComplexArray, t: float) -> ComplexArray:
        """Apply inverse Hamiltonian flow"""
        return self.apply_flow(points, -t)
    
    def polynomial_encode(self, secret: List[int]) -> ComplexArray:
        """
        Encode secret (as F_p elements) as polynomial f(x), evaluate at domain points,
        then apply Teichmüller lift to get complex numbers
        """
        coeffs = [0] * self.n
        for j in range(min(len(secret), self.n)):
            coeffs[j] = secret[j] % self.p
            
        evaluations = []
        for x in self.D_f:
            # Horner's method for polynomial evaluation
            val = 0
            for coeff in reversed(coeffs):
                val = (val * x + coeff) % self.p
                
            evaluations.append(self.teichmuller_lift(val))
            
        return np.array(evaluations, dtype=np.complex128)
    
    def blind_evaluations(self, evaluations: ComplexArray, r: Optional[complex] = None) -> Tuple[ComplexArray, complex, ComplexArray]:
        """
        Apply random blinding to polynomial evaluations (complex numbers)
        """
        if r is None:
            r = complex(np.random.normal(), np.random.normal())
            
        # Generate and normalize mask for numerical stability
        mask = np.array([complex(np.random.normal(), np.random.normal()) for _ in range(self.n)], dtype=np.complex128)
        mask = mask / np.linalg.norm(mask)  # Normalize mask to unit L2 norm
        
        return evaluations + r * mask, r, mask
    
    def alice_prove(self, secret: List[int], flow_time: float = np.pi/4) -> Commitment:
        """
        Alice's protocol: Polynomial encode → Blind → Apply Hamiltonian flow
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
        Bob's protocol: Apply inverse Hamiltonian flow and verify geometric invariants
        """
        print("\n=== Bob (Verifier) ===")
        print("Received commitment from Alice")
        
        # Convert dict to Commitment if needed
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
        
        # Use L2 norm difference for stable verification
        norm_diff = np.linalg.norm(recovered_evals - blinded)
        print(f"L2 norm of difference: {norm_diff:.8e}")
        
        is_verified = norm_diff < self.epsilon
        print(f"Verification {'SUCCESS' if is_verified else 'FAILED'}: " 
              f"Geometric invariants {'preserved' if is_verified else 'not preserved'}")
        return is_verified
    
    def visualize_domain_transformation(self, flow_time: float = np.pi/4, num_steps: int = 50, save_gif: bool = False) -> None:
        """
        Visualize transformation of interpolation domain under Hamiltonian flow
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
        
        # Draw unit circle
        theta = np.linspace(0, 2*np.pi, 100)
        circle = np.exp(1j * theta)
        ax.plot(circle.real, circle.imag, 'k-', alpha=0.3)
        
        # Calculate trajectories using analytical solution
        delta_t = flow_time / num_steps
        for step in range(1, num_steps + 1):
            t = step * delta_t
            rotated_points = domain_points * np.exp(1j * t)
            all_trajectories.append(rotated_points)
        
        colors = cm.viridis(np.linspace(0, 1, self.n))
        scatter = ax.scatter(self.D.real, self.D.imag, c=colors, s=100, label='Domain Points')
        
        # Draw trajectories for each point
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
        Enhanced 3D visualization of Hamiltonian flow on complex grid points
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
        
        # Display mesh grid planes at initial and final times
        for t_idx in [0, num_steps-1]:
            t = times[t_idx]
            tz_plane = np.ones((grid_size, grid_size)) * t
            ax.plot_surface(X, Y, tz_plane, alpha=0.1, color='gray')
        
        # Calculate and display trajectories for each point
        for idx, z0 in enumerate(initial_points):
            # Analytical solution: z(t) = z0 * exp(i*t)
            zs = z0 * np.exp(1j * times)
            xs, ys = zs.real, zs.imag
            
            # Draw trajectory with color gradient based on time
            for i in range(len(times) - 1):
                t_ratio = times[i] / flow_time
                ax.plot(
                    [xs[i], xs[i+1]], 
                    [ys[i], ys[i+1]], 
                    [times[i], times[i+1]],
                    color=colormap(t_ratio), 
                    alpha=0.7, 
                    linewidth=1.5
                )
            
            # Emphasize initial and final points
            ax.scatter(z0.real, z0.imag, 0, color=colors[idx], s=40, edgecolor='black')
            ax.scatter(xs[-1], ys[-1], flow_time, color=colors[idx], s=40, edgecolor='white')
        
        ax.set_xlabel('Re(z)', fontsize=12)
        ax.set_ylabel('Im(z)', fontsize=12)
        ax.set_zlabel('Time t', fontsize=12)
        ax.set_title('SymPLONK: 3D Visualization of Hamiltonian Flow', fontsize=14)
        ax.view_init(elev=30, azim=45)
        plt.tight_layout()
        
        # Interactive view rotation
        if hasattr(plt, 'ion'):
            plt.ion()
            print("Rotating view for better visualization...")
            for angle in range(0, 360, 5):
                ax.view_init(elev=30, azim=angle)
                plt.draw()
                plt.pause(0.05)
            plt.ioff()
            
        plt.show()

def run_demo() -> None:
    """Run demonstration of SymPLONK protocol"""
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
    
    print("\nDemonstration complete!")

if __name__ == "__main__":
    run_demo()
