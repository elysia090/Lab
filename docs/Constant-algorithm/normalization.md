CCR normalization and O(1) methods, rigorously specified and ready to drop into LLM/RAG/MoE pipelines. ASCII only.

1. Exact CCR normalization
   Let scores s = {s_j}*{j in J}. Partition J into tiles T = {t_1,...,t_P} (arbitrary, possibly imbalanced).
   For a tile t, define
   m_t := max*{j in t} s_j
   Z_t := sum_{j in t} exp(s_j − m_t)
   Define the binary combine on pairs (m,Z)
   (m1,Z1) ⊕ (m2,Z2) := ( m, Z1 exp(m1−m) + Z2 exp(m2−m) ),  m := max(m1,m2)
   Identity e := (−inf, 0). Then:
   (i) ⊕ is commutative and associative; (R ∪ {−inf}) × R_{>=0} under ⊕ is a commutative monoid.
   (ii) For any partition T and any fold order on { (m_t,Z_t) }, the reduction ⊕*t (m_t,Z_t) yields (m**,Z_*), where
   m_* = max_{j in J} s_j
   Z_* = sum_{j in J} exp(s_j − m_*)
   Hence the global softmax is recovered exactly:
   alpha_j = exp(s_j − m_*) / Z_*
   This result is partition- and order-invariant.

2. Incremental O(1) update (streaming)
   Given current state (m,Z) for a set S and a new score s':
   m' = max(m, s')
   Z' = Z exp(m − m') + exp(s' − m')
   This is an O(1) update per arrival. It is numerically stable because exp arguments are nonpositive. For a batch B with local (m_B,Z_B), the same formula holds:
   (m,Z) ← (m,Z) ⊕ (m_B,Z_B)
   Deletion in a sliding window is not O(1) in the worst case unless one maintains a capped multiset for the max and a register bank for exp(s−m). For append-only streams, the above update is optimal.

3. Distributed O(1) payload protocol
   Each shard k computes (m_{t_k},Z_{t_k}) locally in O(|t_k|). The aggregator performs a one-round reduction of 2 scalars per shard via ⊕. The payload per shard is constant (2 values, typically f64), independent of |t_k| and |J|. Tree reductions with branching b give depth ceil(log_b P); the result is identical for any tree.

4. Certified approximations with composable bounds
   4.1) Safe-Z (per-tile pruning)
   Within each tile t, keep top k_t indices by s; let U_t be pruned indices. Define
   Zc_t := sum_{j in keep} exp(s_j − m_t)
   UB_t := max_{j in U_t} s_j  (if U_t nonempty)
   Rhat_t := |U_t| * exp(UB_t − m_t)  (per-tile tail upper bound)
   Combine with ⊕ on the adjusted tile masses:
   (m_hat, Z_hat) := ⊕*t ( m_t, Zc_t + Rhat_t )
   Let Rhat := (sum_t Rhat_t) / max(1e−12, sum_t Zc_t). Then the global L1 error of the resulting softmax is certified by
   L1(alpha_hat, alpha**) <= 2 Rhat / (1 + Rhat)
   The certificate is partition-invariant because Rhat_t enters only through ⊕.

4.2) Auto-k (global epsilon-stopping)
Prepare per tile t the sorted a_t = sort_{desc} exp(s_j − m_t) and compute the CCR reference (m_c, Z_c) = ⊕_t (m_t, sum a_t). Let scale_t := exp(m_t − m_c). In a global greedy, repeatedly add the next term from the tile whose current candidate maximizes scale_t * a_t[next], accumulating
head_sum := sum of selected (scale_t * a_t[•])
Stop when
eps_eff := 2 * (Z_c − head_sum) / Z_c <= eps
This rule is CCR-consistent (invariant to partition/order) and yields a global certificate eps_eff. A heap implementation makes the selection O(K log P), where K is the number of taken heads.

4.3) Safe-Z + Auto-k together
First prune per tile (Safe-Z) to obtain Z̃_c := sum_t (Zc_t) and the composite bound Rhat. Then run Auto-k on the kept masses. The overall deviation is bounded by
L1 <= 2 * (Rhat + (Z̃_c − head_sum)/Z̃_c) / (1 + Rhat + (Z̃_c − head_sum)/Z̃_c)
This follows from composability of tail surrogates inside ⊕ and the same two-line inequality used in 4.1/4.2.

5. Differentiation and backprop through CCR
   The forward outputs are alpha = softmax(s), identical to the standard operator. Thus the Jacobian is the usual
   ∂alpha_i/∂s_j = alpha_i (delta_{ij} − alpha_j)
   Two practical routes exist:
   (i) Differentiate w.r.t. s directly from alpha; the combine is a forward-only numerics device. VJP for a loss L with upstream g on alpha is
   ∂L/∂s_j = alpha_j ( g_j − sum_k alpha_k g_k )
   This costs O(|J|), same as a standard softmax backward.
   (ii) If one differentiates through the tree of ⊕ nodes (for custom kernels), the only non-smooth node is m:=max. Treat it like log-sum-exp: use subgradients on ties or smooth with logsumexp if needed. Z’s derivative is smooth given m.

6. Numerical stability
   Use f64 for (m,Z) at tile and combine; exp arguments are then >= −O(1e2) in typical deep-learning magnitudes. All other work can be f32. Optional: Kahan compensation on any running decayed sums or EW variance estimators around CCR if you also do streaming statistics nearby. The CCR combine itself does not need compensation.

7. Complexity summary
   Exact CCR:
   tile scans: O(|t_k|) each, perfectly parallel
   combine: O(P) time, O(1) payload per shard
   incremental append: O(1) per new score or per new tile
   Safe-Z:
   per tile: top-k via partial select (argpartition) O(|t_k|), bound formation O(1)
   global: same O(P) combine; certificate formation O(1)
   Auto-k:
   prepare: per tile scan + sort of a_t (or top runs); heap greedy: O(K log P)
   In decoding streams with fixed per-token fan-in, these yield O(1) per token work beyond QK^T.

8. Protocols for LLM/RAG/MoE
   Attention/retrieval normalization:
   compute local (m_t,Z_t) per head/tile on-device
   emit (m_t,Z_t) to the aggregator (2 scalars)
   reduce with ⊕ to (m_*,Z_*)
   form alpha on-device using m_* and Z_* (no N-size all-reduce)
   With approximations:
   replace Z_t by Zc_t+Rhat_t (Safe-Z), or
   select until eps via heap (Auto-k), both with explicit global certificates
   Gating/MoE:
   experts report (m_t,Z_t) for their pre-activations; the gate weights are exact softmax after a one-round ⊕.

9. Test and acceptance criteria
   Partition invariance: randomly partition and permute; require |m−m_*|<=1e−12 (f64), |Z−Z_*|/|Z_*|<=1e−12; L1(alpha_ccr,alpha_*)<=1e−12.
   Gauge invariance: alpha(s+c)=alpha(s).
   Refinement naturality: coarsen/refine partitions; invariants as above.
   Safe-Z: empirical L1 <= certified bound for a grid of prune fractions; monotone bound growth.
   Auto-k: eps_eff <= eps + 1e−2; L1 tracks eps within a small slack; sum of selected heads nondecreasing.
   Streaming: incremental O(1) update equals batch CCR on concatenated data.

10. Implementation notes
    Use a struct of two f64 scalars for (m,Z); ⊕ is a pure function. In GPU code, keep per-block m,Z in registers, reduce within block, then across blocks, then across devices with a 2-scalar all-reduce or a gather. Never transmit vectors for normalization. For Safe-Z, compute top-k with argpartition; for Auto-k, push each tile’s top element into a max-heap keyed by scale_t * a_t[k], pop/push until the eps rule fires. Record and export the certificate (L1 bound or eps_eff) with the output.
