import math
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Tuple, Optional, Union, Dict, Any, TypeVar
import numpy.typing as npt

try:
    import cupy as cp
except ImportError:
    cp = None

# Type aliases for clarity
ComplexArray = npt.NDArray[np.complex128]
RealArray = npt.NDArray[np.float64]
T = TypeVar('T')

# Threshold for precomputing full Teichmüller lift table.
TEICH_THRESHOLD: int = 1000

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
    Utility class for finite field operations.
    """
    @staticmethod
    @lru_cache(maxsize=128)
    def divisors(n: int) -> List[int]:
        divs: set[int] = set()
        for i in range(1, int(math.sqrt(n)) + 1):
            if n % i == 0:
                divs.add(i)
                divs.add(n // i)
        return sorted(divs)

    @staticmethod
    def prime_factors(n: int) -> List[int]:
        factors: List[int] = []
        i: int = 2
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
        if (p - 1) % n != 0:
            return None
        proper_divisors: List[int] = [d for d in cls.divisors(n) if d < n]
        for candidate in range(2, p):
            if pow(candidate, n, p) != 1:
                continue
            if all(pow(candidate, d, p) != 1 for d in proper_divisors):
                return candidate
        return None

    @classmethod
    def find_primitive_root(cls, p: int) -> Optional[int]:
        factors: List[int] = cls.prime_factors(p - 1)
        exponents: List[int] = [(p - 1) // f for f in factors]
        for candidate in range(2, p):
            if all(pow(candidate, exp, p) != 1 for exp in exponents):
                return candidate
        return None

    @staticmethod
    def discrete_log(a: int, g: int, p: int) -> int:
        if p < 100:
            for k in range(p - 1):
                if pow(g, k, p) == a:
                    return k
        else:
            m: int = int(math.sqrt(p - 1)) + 1
            baby_steps: Dict[int, int] = {pow(g, j, p): j for j in range(m)}
            g_inv: int = pow(g, p - 1 - m, p)
            value: int = a
            for i in range(m):
                if value in baby_steps:
                    return i * m + baby_steps[value]
                value = (value * g_inv) % p
        raise ValueError(f"No discrete log exists: a={a}, g={g}, p={p}")

@lru_cache(maxsize=None)
def teichmuller_lift_func(a: int, g: int, p: int) -> complex:
    if a == 0:
        return 0.0 + 0.0j
    k: int = FiniteFieldUtils.discrete_log(a, g, p)
    return np.exp(2j * np.pi * k / (p - 1))

class HamiltonianFlow:
    """
    Implements Hamiltonian flow operations.
    For the quadratic Hamiltonian H(z)=1/2|z|^2, the flow is a rotation:
      z(t)=z(0)*exp(i*t).
    """
    @staticmethod
    def hamiltonian_function(z: complex) -> float:
        return 0.5 * abs(z) ** 2

    @staticmethod
    def apply_flow(points: Any, t: float, epsilon: float = 1e-8) -> Any:
        if abs(t) < epsilon:
            return points
        xp = cp if (cp is not None and isinstance(points, cp.ndarray)) else np
        return points * xp.exp(1j * t)

    @classmethod
    def apply_inverse_flow(cls, points: Any, t: float, epsilon: float = 1e-8) -> Any:
        return cls.apply_flow(points, -t, epsilon)

class SymPLONK:
    """
    SymPLONK Protocol: Uses Teichmüller lift and Hamiltonian flow.
    """
    @staticmethod
    def poly_eval_horner(x_vals: List[int], coeffs: List[int], p: int) -> List[int]:
        results: List[int] = []
        for x in x_vals:
            res: int = 0
            for c in coeffs:
                res = (res * x + c) % p
            results.append(res)
        return results

    def __init__(self, n: int, p: int, epsilon: float = 1e-8, use_gpu: bool = True) -> None:
        self.n: int = n
        self.p: int = p
        self.epsilon: float = epsilon

        if use_gpu and cp is not None:
            self.xp: Any = cp
            print("Using GPU acceleration with CuPy.")
            try:
                device_props = cp.cuda.runtime.getDeviceProperties(0)
                print("GPU Device Info:")
                print(f"  Name: {device_props['name'].decode('utf-8')}")
                print(f"  Total Memory: {device_props['totalGlobalMem'] / 1e9:.2f} GB")
            except Exception as e:
                print("Unable to retrieve GPU properties:", e)
        else:
            self.xp = np
            print("Using CPU (NumPy).")

        # Use a smaller prime domain.
        self.omega_f: Optional[int] = FiniteFieldUtils.find_primitive_nth_root(n, p)
        if self.omega_f is None:
            print(f"Warning: No primitive {n}th root found in F_{p}. Using domain [1,2,...,{n}].")
            self.D_f: List[int] = list(range(1, n + 1))
        else:
            self.D_f = [pow(self.omega_f, i, p) for i in range(n)]

        self.g: Optional[int] = FiniteFieldUtils.find_primitive_root(p)
        if self.g is None:
            raise ValueError(f"Primitive root not found in F_{p}.")

        if self.p < TEICH_THRESHOLD:
            self.lift_table: Optional[np.ndarray] = np.array(
                [self._teichmuller_lift(a) for a in range(self.p)],
                dtype=np.complex128
            )
        else:
            self.lift_table = None

        if self.lift_table is not None:
            self.D: Any = self.xp.array([self.lift_table[a] for a in self.D_f],
                                         dtype=self.xp.complex128)
        else:
            domain_lifts: List[complex] = [teichmuller_lift_func(a, self.g, self.p) for a in self.D_f]
            self.D = self.xp.array(domain_lifts, dtype=self.xp.complex128)
        self.flow: HamiltonianFlow = HamiltonianFlow()
        self._log_setup_info()

    def _log_setup_info(self) -> None:
        print(f"Setup: n={self.n}, p={self.p}")
        print(f"Finite field domain D_f: {self.D_f}")
        if self.xp is cp:
            D_np: ComplexArray = cp.asnumpy(self.D)
        else:
            D_np = self.D
        print(f"Teichmüller-lifted domain D: {np.around(D_np, 4)}")

    def _teichmuller_lift(self, a: int) -> complex:
        return teichmuller_lift_func(a, self.g, self.p)  # type: ignore

    def polynomial_encode(self, secret: List[int]) -> Any:
        """
        Encode secret data as a polynomial and evaluate it over D_f.
        Uses vector-based Horner's method.
        """
        coeffs: List[int] = [c % self.p for c in (secret + [0]*(self.n-len(secret)))]
        poly_vals: List[int] = SymPLONK.poly_eval_horner(self.D_f, coeffs, self.p)
        lifts: List[complex] = [teichmuller_lift_func(val, self.g, self.p) for val in poly_vals]
        if self.xp is cp:
            return cp.array(lifts, dtype=cp.complex128)
        else:
            return np.array(lifts, dtype=np.complex128)

    def blind_evaluations(self, evaluations: Any, r: Optional[complex] = None) -> Tuple[Any, complex, Any]:
        if r is None:
            r = complex(self.xp.random.normal(), self.xp.random.normal())
        mask: Any = self.xp.random.normal(size=self.n) + 1j * self.xp.random.normal(size=self.n)
        mask = mask / self.xp.linalg.norm(mask)
        return evaluations + r * mask, r, mask

    def alice_prove(self, secret: List[int], flow_time: float = np.pi/4) -> Commitment:
        print("\n=== Alice (Prover) ===")
        print(f"Secret (mod {self.p}): {secret}")
        evaluations: Any = self.polynomial_encode(secret)
        print("Polynomial encoding complete.")
        blinded_evals, r, mask = self.blind_evaluations(evaluations)
        print(f"Random blinding applied with r = {r:.4f}")
        transformed_evals: Any = self.flow.apply_flow(blinded_evals, flow_time, self.epsilon)
        print(f"Hamiltonian flow applied with t = {flow_time:.4f}")
        xp = self.xp
        if xp is cp:
            transformed_evals_np: ComplexArray = cp.asnumpy(transformed_evals)
            evaluations_cpu: ComplexArray = cp.asnumpy(evaluations)
            mask_np: ComplexArray = cp.asnumpy(mask)
        else:
            transformed_evals_np = transformed_evals
            evaluations_cpu = evaluations
            mask_np = mask
        return Commitment(
            transformed_evaluations=transformed_evals_np,
            flow_time=flow_time,
            secret_original_evaluations=evaluations_cpu,
            r=r,
            mask=mask_np
        )

    def bob_verify(self, commitment: Union[Commitment, Dict[str, Any]]) -> bool:
        print("\n=== Bob (Verifier) ===")
        if isinstance(commitment, dict):
            comm: Commitment = Commitment(**commitment)
        else:
            comm = commitment
        print(f"Flow time: t = {comm.flow_time:.4f}")
        xp = self.xp
        transformed: Any = xp.array(comm.transformed_evaluations, dtype=xp.complex128)
        recovered: Any = self.flow.apply_inverse_flow(transformed, comm.flow_time, self.epsilon)
        original: Any = xp.array(comm.secret_original_evaluations, dtype=xp.complex128)
        mask: Any = xp.array(comm.mask, dtype=xp.complex128)
        expected: Any = original + comm.r * mask
        diff_norm: float = float(xp.linalg.norm(recovered - expected).get()) if xp is cp else float(xp.linalg.norm(recovered - expected))
        print(f"L2 norm difference: {diff_norm:.8e}")
        verified: bool = diff_norm < self.epsilon
        print(f"Verification {'SUCCESS' if verified else 'FAILED'}")
        return verified

    def _compute_trajectories(self, start_points: Any, times: RealArray) -> Any:
        xp = self.xp
        times_gpu: Any = xp.array(times) if xp is cp else times
        return start_points[:, None] * xp.exp(1j * times_gpu)

    def visualize_flow_3d_comparison(self, secret: List[int],
                                     flow_time: float = np.pi,
                                     num_steps: int = 100) -> None:
        times: RealArray = np.linspace(0, flow_time, num_steps)
        plain_trajs: Any = self._compute_trajectories(self.D, times)
        plain_trajs_np: ComplexArray = cp.asnumpy(plain_trajs) if self.xp is cp else plain_trajs
        plain_evals: Any = self.polynomial_encode(secret)
        blinded_evals, _, _ = self.blind_evaluations(plain_evals)
        blinded_trajs: Any = self._compute_trajectories(blinded_evals, times)
        blinded_trajs_np: ComplexArray = cp.asnumpy(blinded_trajs) if self.xp is cp else blinded_trajs
        D_np: ComplexArray = cp.asnumpy(self.D) if self.xp is cp else self.D
        blinded_evals_np: ComplexArray = cp.asnumpy(blinded_evals) if self.xp is cp else blinded_evals
        self._plot_3d_comparison(plain_trajs_np, blinded_trajs_np, times, flow_time, D_np, blinded_evals_np)

    def _plot_3d_comparison(self, plain_trajs: np.ndarray, blinded_trajs: np.ndarray,
                            times: RealArray, flow_time: float,
                            plain_start: ComplexArray, blinded_start: ComplexArray) -> None:
        fig = plt.figure(figsize=(18, 10))
        all_points: np.ndarray = np.concatenate((plain_trajs.flatten(), blinded_trajs.flatten()))
        max_val: float = 1.2 * max(np.abs(all_points.real).max(), np.abs(all_points.imag).max())
        xlim: Tuple[float, float] = (-max_val, max_val)
        ylim: Tuple[float, float] = (-max_val, max_val)
        ax: Any = fig.add_subplot(121, projection='3d')
        ax.set_title("Domain Points Trajectories (Before Blinding)", fontsize=14)
        ax.set_xlabel("Re(z)")
        ax.set_ylabel("Im(z)")
        ax.set_zlabel("Time t")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(0, flow_time)
        colors: np.ndarray = plt.cm.tab20(np.linspace(0, 1, plain_trajs.shape[0]))
        self._plot_trajectories(ax, plain_trajs, times, colors)
        for i, point in enumerate(plain_start):
            ax.text(point.real, point.imag, 0, f"P{i}", color=colors[i], fontsize=10)
        ax2: Any = fig.add_subplot(122, projection='3d')
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
        num_points: int = trajectories.shape[0]
        for i in range(num_points):
            xs: np.ndarray = trajectories[i].real
            ys: np.ndarray = trajectories[i].imag
            ax.plot(xs, ys, times, color=colors[i], lw=2, alpha=0.8)
            ax.scatter(xs[0], ys[0], times[0], color=colors[i], s=80, edgecolor='black', marker='o')
            ax.scatter(xs[-1], ys[-1], times[-1], color=colors[i], s=80, edgecolor='white', marker='^')

    def visualize_2d_flow_snapshots(self, secret: List[int],
                                    flow_time: float = np.pi,
                                    num_snapshots: int = 4) -> None:
        snapshot_times: RealArray = np.linspace(0, flow_time, num_snapshots)
        plain_evals: Any = self.polynomial_encode(secret)
        blinded_evals, _, _ = self.blind_evaluations(plain_evals)
        blinded_evals_np: ComplexArray = cp.asnumpy(blinded_evals) if self.xp is cp else blinded_evals
        fig, axes = plt.subplots(1, num_snapshots, figsize=(16, 4))
        colors: np.ndarray = plt.cm.tab10(np.linspace(0, 1, len(blinded_evals_np)))
        max_val: float = 1.2 * max(np.abs(blinded_evals_np.real).max(), np.abs(blinded_evals_np.imag).max())
        for i, t in enumerate(snapshot_times):
            flowed: Any = self.flow.apply_flow(blinded_evals, t, self.epsilon)
            flowed_np: ComplexArray = cp.asnumpy(flowed) if self.xp is cp else flowed
            ax: Any = axes[i]
            ax.scatter(flowed_np.real, flowed_np.imag, color=colors, s=100, edgecolor='white')
            for j, pt in enumerate(flowed_np):
                ax.text(pt.real, pt.imag, f"{j}", fontsize=9, ha='center', va='center', color='white')
            ax.plot(flowed_np.real, flowed_np.imag, 'k-', alpha=0.3)
            ax.set_xlim(-max_val, max_val)
            ax.set_ylim(-max_val, max_val)
            ax.set_title(f"t = {t:.2f}")
            ax.set_xlabel("Re(z)")
            ax.set_ylabel("Im(z)")
            ax.grid(True, alpha=0.3)
        fig.suptitle("Hamiltonian Flow Snapshots", fontsize=16)
        plt.tight_layout()
        plt.show()


# Example usage with timing measurement.
if __name__ == "__main__":
    # Change parameters to smaller values.
    n_val: int = 16
    p_val: int = 97
    symplonk: SymPLONK = SymPLONK(n=n_val, p=p_val, use_gpu=True)
    secret: List[int] = [1, 2, 3, 4]
    
    start_time: float = time.perf_counter()
    commitment: Commitment = symplonk.alice_prove(secret)
    verification_success: bool = symplonk.bob_verify(commitment)
    end_time: float = time.perf_counter()
    print("\n=== Verification Result ===")
    print(f"Verification result: {verification_success}")
    print("\n=== Average Test ===")
    print(f"Total execution time (generation to verification): {end_time - start_time:.4f} seconds")
    
    symplonk.visualize_flow_3d_comparison(secret)
    symplonk.visualize_2d_flow_snapshots(secret)
