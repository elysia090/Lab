Title: Constant-Time Per-Index Periodic Convolution
Subtitle: A Near-Optimal O(N) Full Convolution and Differentiation Scheme That Beats FFT in the Fixed-Bandwidth Regime

Abstract
We study circular (periodic) convolution of a length-N real signal x with a compact kernel w supported on offsets in [-K, K]. When K is a design constant independent of N, we give a simple algorithm with O(1) time per index and O(N) time for the full output. We prove correctness and tight complexity bounds, give lower bounds showing optimality up to constant factors, and derive forward- and reverse-mode derivatives and Hessian-vector products that inherit the same asymptotic costs. We characterize the precise crossover against FFT-based methods and provide engineering guidelines that drive the constant factor down to the practical limit. Extensions to 2D/3D, multi-channel, and learned kernels are included. The analysis is fully self-contained and uses only ASCII notation.
	1.	Problem Statement
Input:

	•	Signal x in R^N, indexed modulo N (circular boundary).
	•	Kernel w with support in indices k in {-K, …, K}. Let L := 2K+1.

Output:
	•	y[i] = sum_{k=-K..K} w[k] * x[(i - k) mod N] for i = 0,1,…,N-1.

Goal:
	•	Per-index evaluation in O(1) time when K is treated as a fixed constant.
	•	Full evaluation in O(N) time, which is information-theoretically optimal when all outputs are required.

	2.	Computational Model

	•	Word-RAM with unit-cost arithmetic and random access.
	•	Loop control, integer arithmetic, and modular addressing count in the constant factor.
	•	K is a design constant (small, fixed), independent of N.
	•	We consider both single-index queries and the full vector output.

	3.	Algorithm
PerIndexConv(x, w, K, i, N):

	1.	j := i mod N  (or j := i & (N-1) if N is a power of two)
	2.	acc := 0
	3.	for t := 0..(2K):
k := t - K
acc := acc + w[t] * x[(j - k) mod N]
	4.	return acc

FullConv(x, w, K, N):
	1.	for i := 0..N-1:
y[i] := PerIndexConv(x, w, K, i, N)
	2.	return y

Implementation notes:
	•	Replace mod by a ring-buffer pointer or two contiguous passes to avoid modulus in the inner loop.
	•	For small K, full unrolling eliminates loop overhead; compilers vectorize fixed-length loops easily.
	•	Process i in increasing order to exploit locality; each x element is reused L times across neighboring outputs.

	4.	Correctness
For each i, the loop accumulates exactly sum_{k=-K..K} w[k] * x[(i - k) mod N], which is the definition of circular convolution with a compact kernel. No approximation is introduced; only floating-point round-off applies. Periodicity is enforced by modular addressing or equivalent pointer wrapping. Therefore the algorithm computes the correct output by construction.
	5.	Complexity and Optimality
Operation counts:

	•	Per index: exactly L = 2K+1 multiply-accumulate pairs + O(1) index updates.
	•	Full output: N such computations.

Asymptotic bounds under fixed K:
	•	T_index = Theta(K) = O(1).
	•	T_full = N * Theta(K) = Theta(NK) = O(N).
	•	Working memory is O(1) beyond storing x and y.

Lower bounds:
	•	Output-size lower bound: any algorithm that produces all N outputs must be Omega(N) time in the RAM model, since writing N words is Omega(N).
	•	Per-index lower bound (algebraic decision-tree intuition): when w has L independent coefficients and the footprint is [-K, K], the value y[i] depends on L distinct entries of x. Any algorithm must inspect Omega(K) data in the worst case; thus Theta(K) is optimal up to constants.

Conclusion:
	•	For fixed K, O(1) per index and O(N) full time are asymptotically optimal up to constant factors.

	6.	Comparison With FFT
FFT-based circular convolution on length-N inputs costs Theta(N log N) arithmetic plus constant-factor overheads for complex packing and twiddle factors. The direct fixed-bandwidth method costs Theta(NK).

Crossover (ignoring constants):
	•	NK < c * N log_2 N  <=>  K < c * log_2 N.

Implications:
	•	If K is fixed (independent of N), there exists N_0 such that for all N > N_0 the direct method is strictly faster asymptotically than FFT.
	•	In practice on CPUs, the direct scheme usually dominates for K in [3..32] over a very wide range of N, due to small constants, good locality, and no transform setup.
	•	FFT wins when K grows with N (large blurs), or when many independent convolutions are batched to amortize transforms (especially on GPUs).

	7.	Differentiation: JVP, VJP, Gradients, HVP
Let L(y) be a scalar loss.

Forward-mode Jacobian-vector product w.r.t. x:
	•	For a tangent vector dx, JVP_x is exactly convolution with w: d(y)[i] = sum_{k} w[k] * dx[(i - k) mod N].
	•	Cost: O(K) per index, O(N) full.

Reverse-mode vector-Jacobian product w.r.t. x:
	•	For a cotangent vector g = dL/dy, the gradient w.r.t. x is convolution with the reversed kernel wr defined by wr[t] = w[-t].
	•	(dL/dx)[i] = sum_{k} wr[k] * g[(i - k) mod N].
	•	Cost: O(K) per index, O(N) full.

Gradient w.r.t. weights w:
	•	For each t in {0..2K} with k = t - K:
dL/dw[t] = sum_{i=0..N-1} g[i] * x[(i - k) mod N].
	•	Total cost: O(NK) = O(N) for fixed K. Can be fused with the backward sweep to reduce memory traffic.

Hessian-vector products:
	•	Because the operations are linear in x and w at fixed kernel position, HVPs inherit the same per-index O(K) structure. Full HVP costs O(N) for fixed K.

Conclusion:
	•	The entire differentiation stack (JVP, VJP, parameter gradients, HVP) matches the primal asymptotics: O(1) per index and O(N) for full outputs when K is fixed.

	8.	Multi-D and Multi-Channel Extensions
2D periodic grids (H x W) with a fixed S x T stencil:

	•	Per-pixel cost Theta(S*T) = O(1) when S and T are constants.
	•	Full cost Theta(HWST) = O(HW).
	•	Beats 2D FFT Theta(HWlog(HW)) whenever ST = o(log(H*W)).

Channels and multiple kernels:
	•	Depthwise convolution with fixed spatial footprint remains O(1) per pixel if the number of channels C is fixed.
	•	Pointwise (1x1) mixing is a CxC matrix; if C is fixed, per-pixel remains O(1).
	•	For Q kernels, direct cost is Theta(NKQ). FFT batching may win if K*Q becomes comparable to log N with large batched workloads.

	9.	Numerical Accuracy and Stability

	•	Floating-point error is dominated by accumulation: O(K * eps * max|x| * max|w|), eps = machine epsilon.
	•	For small K this is negligible. Pairwise or compensated summation can be used without changing asymptotics.
	•	Circular boundary avoids padding artifacts; for linear convolution, use overlap-save/add or enlarge the domain.

	10.	Engineering for the Constant Factor

	•	Remove modulus in the inner loop: use rolling pointers; for N being a power of two, use bitmasking (i & (N-1)).
	•	Unroll loops for fixed K; compilers can vectorize fully.
	•	SIMD: vector multiply-add across the L-window, then horizontal add.
	•	Cache locality: process indices in order; reuse sliding windows.
	•	Parallelism: independent slices of i; trivial to thread.
	•	Micro-kernel: keep weights in registers; prefetch upcoming x-blocks.
	•	2D/3D: tile into small blocks to fit L1/L2 caches; fuse depthwise and pointwise when beneficial.

	11.	When FFT Wins and When It Does Not

	•	FFT wins for very large effective bandwidth (K comparable to c * log N or larger), for many kernels that can be batched, or on GPUs with highly optimized vendor FFTs and large reuse of kernel transforms.
	•	The direct fixed-bandwidth scheme wins for small fixed K, latency-critical per-index queries, limited-batch throughput on CPUs, and scenarios where kernel changes often (transform reuse is limited).

	12.	Theorems (Sketches)
Theorem 1 (Output lower bound). Any algorithm that outputs all N entries of y must perform Omega(N) work in the RAM model.
Sketch. The algorithm must at least write N machine words; each write is Omega(1).

Theorem 2 (Per-index lower bound for fixed bandwidth). Computing y[i] for a bandwidth-K kernel requires Omega(K) primitive operations in the worst case.
Sketch. y[i] depends on L = 2K+1 independent entries of x. In the algebraic decision-tree model, at least a constant amount of work per independent dependency is required; thus Omega(K).

Corollary 3 (Optimality). For fixed K, the proposed algorithm is asymptotically optimal up to constant factors for both per-index (O(1)) and full-output (O(N)) computation.
	13.	Pseudocode (ASCII)
PerIndexConv(x, w, K, i, N):
j := i mod N
acc := 0
for t := 0..(2K):
k := t - K
acc := acc + w[t] * x[(j - k) mod N]
return acc

FullConv(x, w, K, N):
y := new array of length N
for i := 0..N-1:
y[i] := PerIndexConv(x, w, K, i, N)
return y

VJP_x(grad_y, w, K, N):
wr[t] := w[2K - t] for t in 0..2K
return FullConv(grad_y, wr, K, N)

Grad_w(grad_y, x, K, N):
g := zero array of length 2K+1
for t := 0..2K:
k := t - K
s := 0
for i := 0..N-1:
s := s + grad_y[i] * x[(i - k) mod N]
g[t] := s
return g

Note: Grad_w can be fused with VJP_x to write g in a single sweep. Asymptotics remain O(N) for fixed K.
	14.	Empirical Expectations

	•	Per-index time is essentially flat in N (O(1)) and typically a few microseconds for K in [3..16] on commodity CPUs.
	•	Full time scales linearly in N (O(N)).
	•	FFT is slower unless K grows toward c * log N or many kernels are batched with transform reuse.
	•	Differentiation timings (JVP, VJP, Grad_w) match the primal costs within constant factors.

	15.	Reproducibility Checklist

	•	Provide a reference implementation with explicit modulo or ring-buffer addressing.
	•	Unit tests that compare against a slow reference for random signals and kernels at multiple N and K.
	•	Microbenchmarks reporting per-index top-k mean and p95 latencies in milliseconds over N in a logarithmic grid.
	•	Backward implementation producing dL/dx and dL/dw; verify adjointness <Jv, w> == <v, J^T w> to within floating-point tolerance.
	•	Report crossover plots vs FFT baselines on the same hardware and precision.

	16.	Extensions and Integrations

	•	2D/3D periodic grids: same analysis with fixed spatial footprint.
	•	Multi-channel: depthwise (fixed footprint) and pointwise (fixed C) remain O(1) per site.
	•	Integration into constant-time differentiation templates: register the convolution as a linear operator; JVP/VJP/HVP plug in with the same O(1)/O(N) guarantees.

	17.	Limitations

	•	Non-periodic boundaries require overlap-save/add or domain enlargement; constants change.
	•	Very large kernels or many kernels may favor FFT batching.
	•	On GPUs with heavyweight batched workloads, vendor FFTs can dominate at very large N with moderate K.

	18.	Conclusion
For circular convolution with fixed bandwidth K, a direct scheme achieves O(1) per-index time and O(N) full time, which is optimal up to constant factors. The entire differentiation stack inherits these costs. The method beats FFT whenever K is fixed or o(log N), and is well-suited to low-latency and throughput-stable systems in 1D and higher dimensions.
