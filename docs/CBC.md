TITLE
Counterfactual Boolean Circuits with Exact Compaction
A Semantics-Preserving Execution Model and an Optimization Stack

STATUS
Draft, self-contained, ASCII, English

ABSTRACT
We present Counterfactual Boolean Circuits (CFC), an exact evaluation model for Boolean circuits under a set of W tracked single-bit interventions. Each node carries a base value a and a W-bit counterfactual difference mask d that equals the PIVOT mask for that node. We then define COMPACT, a semantics-preserving transform that eliminates provably inactive masks and evaluates only the active subset using stable packing. We refine COMPACT into a strict three-phase pipeline ACTIVE, PROJECT, EVALUATE-PACKED and prove soundness requirements. Finally, we provide an optimization stack that reduces predicate overhead and improves packed throughput while preserving exactness (no false negatives). The resulting system approaches O(N*L*p) time in freezing regimes (where the active ratio p is small), while retaining O(N*L) worst-case behavior via adaptive fallback in melting regimes.

1. INTRODUCTION
   Counterfactual queries for Boolean systems often require evaluating f(x) and many perturbed values f(x xor e_i). A naive approach evaluates the circuit W times for W perturbations. CFC compresses these evaluations by propagating a single base value and a vector of counterfactual differences. The central observation is that, under Boolean operations, counterfactual differences obey exact local laws. This makes counterfactual evaluation an instance of exact mask calculus.

The second observation is that counterfactual masks tend to become sparse or vanish in many circuits and regimes. When a node mask is identically zero, the node does not contribute to any tracked counterfactual outcome. COMPACT leverages this to remove inactive computation without approximation. The remaining challenge is that deciding inactivity can be expensive if done naively. We therefore define strict soundness conditions and provide optimizations that reduce predicate work and improve locality, while preserving exactness.

2. MODEL
   Inputs
   x in {0,1}^n

Tracked interventions
S[0..W-1] where each S[k] in [0..n-1]
World k corresponds to flipping x at index S[k], i.e., x^(S[k]) = x xor e_{S[k]}

Circuit
A circuit is a DAG of nodes v.
Each internal node has opcode in {NOT, XOR, AND} and references its operands.
Leaves are IN(j) or CONST(c).

Node state
a(v) in {0,1} is the base value at x.
d(v) in {0,1}^W is the difference mask:
d(v)[k] = value(v, x) xor value(v, x^(S[k])).

Output semantics
For output node out computing f:
a(out) = f(x)
d(out) = PIVOT_S(f, x)
HOLD is:
HOLD_S(f,x) = not d(out)

3. EXACT GATE LAWS
   Masks use lane-wise Boolean operations. Let L = ceil(W/64) be the number of machine words.

Leaf initialization
CONST(c):
a = c
d = 0^W

```
IN(j):
  a = x[j]
  d[k] = 1 iff S[k] == j else 0
```

NOT
a = not a_u
d = d_u

XOR
a = a_u xor a_w
d = d_u xor d_w

AND (4-case exact delta law, conditioned on base values)
a = a_u and a_w
If (a_u,a_w) = (0,0): d = d_u and d_w
If (a_u,a_w) = (0,1): d = d_u and not d_w
If (a_u,a_w) = (1,0): d = d_w and not d_u
If (a_u,a_w) = (1,1): d = d_u or d_w

4. PRIMITIVES
   We define minimal operators needed for exact execution and compaction.

Mask predicates (exact)
ZERO(d): true iff d == 0^W
EQUAL(d,e): true iff d == e
NONEMPTY(d): true iff not ZERO(d)
INTERSECTS(d,e): true iff NONEMPTY(d and e)
SUBSET(d,e): true iff ZERO(d and not e)
DISJOINT(d,e): true iff ZERO(d and e)

Stable packing (PROJECT)
Input:
A[0..m-1] in {0,1} activity flags
X[0..m-1] items
Output:
Y is the stable packed list of X[i] where A[i]=1, in increasing i order
Implementation contract:
There exists pos[i] = rank(A,i)-1 for A[i]=1 such that Y[pos[i]] = X[i]

Mapping (dense to packed)
map[i] = pos[i] if A[i]=1 else -1

5. STRICT COMPACT
   COMPACT operates on a batch of independent gates (e.g., a circuit level).
   It preserves semantics exactly.

Inputs
A batch of m gates with operands referring to already-computed previous-level nodes.

Outputs
a values for all m gates (dense)
d masks for active gates only (packed), plus map[0..m-1]

Phase 1 ACTIVE (exact activity)
Determine A_gate[i] = 1 iff the output mask of gate i is nonzero under the exact gate law.

```
Exact activity conditions
  NOT(u):
    active iff u is active

  XOR(u,w):
    if both inactive -> inactive
    if exactly one active -> active
    if both active -> active iff not EQUAL(d_u, d_w)

  AND(u,w):
    Determine case by (a_u,a_w):

    (0,0): active iff INTERSECTS(d_u, d_w)
    (0,1): active iff NONEMPTY(d_u and not d_w)
    (1,0): active iff NONEMPTY(d_w and not d_u)
    (1,1): active iff NONEMPTY(d_u) or NONEMPTY(d_w)

Soundness requirement
  ACTIVE MUST be sound:
    A_gate[i]=0 implies d(out_i)=0^W.
  Completeness is optional but increases speedup:
    d(out_i)=0^W implies A_gate[i]=0.
```

Phase 2 PROJECT (stable)
Compute pos via rank/prefix-sum and allocate pack_out with length popcount(A_gate).
Construct map_out where map_out[i]=pos[i] if active else -1.

Phase 3 EVALUATE-PACKED
Evaluate only active gates and write their masks into pack_out at stable positions.
Inactive gates do not store masks.

Semantics invariant
For each gate i:
map_out[i] = -1  iff d(out_i) = 0^W
map_out[i] = r >= 0 implies pack_out[r] = d(out_i)

6. EXECUTION SCHEDULE
   Levelized execution (typical)
   For level t:
   Inputs are a_prev dense, pack_prev packed, map_prev dense->packed
   Compute a_t dense for all gates
   Compute A_gate_t using ACTIVE
   PROJECT to get pack_t and map_t
   EVALUATE-PACKED fills pack_t
   Repeat.

General DAG execution
COMPACT is defined for any batch of mutually independent gates whose operands are already known.
A scheduler MAY choose batches to maximize locality and compaction benefit.

7. COMPLEXITY
   Definitions
   N total gates, D levels, m_t gates at level t, sum m_t = N
   W lanes, L words, L = ceil(W/64)
   p_t = popcount(A_gate_t) / m_t (active ratio)
   q_t = fraction of gates in ACTIVE that require full-word predicates (both operands active and ambiguous)

Baseline (no compaction)
Time: O(N * L)
Space (two levels): O(max_t m_t * L)

Strict COMPACT
Time per level:
base a computation: O(m_t)
ACTIVE: O(m_t) + O(q_t * m_t * L)
PROJECT: O(m_t)
EVALUATE-PACKED: O(p_t * m_t * L)
Total:
O(N) + O(N*L*(p_bar + q_bar)) where p_bar, q_bar are level-weighted averages

```
Worst case:
  p_bar approx 1 and q_bar approx 1 gives O(N*L), same order as baseline

Freezing regime:
  p_bar and q_bar small yields near O(N) + O(N*L*small)

Space per level:
  a dense: O(m_t)
  map dense: O(m_t)
  packed masks: O(p_t * m_t * L)
  Total peak: O(max_t m_t * (1 + p_t * L))
```

8. OPTIMIZATION STACK (EXACTNESS PRESERVING)
All optimizations in this section preserve exactness.
The central constraint is no false negatives in ACTIVE.

8.1 Zero-rule first
Before any full-word predicate, apply structural rules that are always exact:
if an operand is inactive (map=-1) then its mask is 0^W
many activity outcomes become trivial under the exact gate laws
This reduces q_t in freezing regimes.

8.2 Fingerprint gatekeeper (sound with verification)
Maintain a per-mask fingerprint fp(d) (e.g., 64-bit hash) updated when a mask is produced.
Use it to avoid full-word EQUAL and other predicates:
If fp(d_u) != fp(d_w), then EQUAL is false, so XOR is active (sound).
If fp matches, equality is unknown, so verify with full-word EQUAL.
This never introduces false negatives because inactivity is declared only after full verification.
The expected number of full-word predicates becomes proportional to the collision rate rho_t.

```
Effect on complexity
  ACTIVE full-word predicate term becomes O(q_t * rho_t * m_t * L) in expectation.
```

8.3 Exact interning (optional) for O(1) equality
Maintain an exact canonicalization table over packed masks (interning):
masks are keyed by their exact word content, with collision resolution by full compare
Assign a stable mask_id to each distinct mask value.
Then EQUAL(d_u,d_w) can be answered by comparing mask_id in O(1).
This can drive the XOR contribution to q_t toward zero.
Correctness is preserved because interning is exact, not approximate.

```
Tradeoff
  Interning adds memory proportional to the number of distinct packed masks and CPU for table maintenance.
```

8.4 Witness-carrying masks (exact certificates)
Maintain with each packed mask an exact witness bit position widx such that d[widx]=1.
Update rules:
for OR, witness can be inherited from any nonzero operand
for AND/ANDNOT, witness may be found by scanning the first nonzero word produced
for XOR, witness can be inherited unless canceled; cancellation requires repair scan
Use witness to certify activity cheaply:
For INTERSECTS(d_u,d_w), if witness_u bit is also 1 in d_w then active.
For d_u and not d_w, if witness_u bit is 1 in d_u and 0 in d_w then active.
If certificate fails, fall back to full-word predicate.
This reduces q_t for AND cases.

8.5 Block summaries (exact or sound-above)
Maintain block-level OR summaries of each mask, e.g., summary[b] = OR over words in block b.
Then:
DISJOINT can be disproved quickly if any block has both summaries nonzero (sound to mark active)
SUBSET violations can be witnessed when u has a nonzero block where w is all-ones in that block is not sufficient; therefore summaries primarily help to find likely witnesses and reduce scans
Summaries are exact information (lossless OR), but coarse; they are best combined with witness and fallback.

8.6 Packing and allocation
PROJECT is implemented as:
pos = prefix_sum(A) - 1
map[A]=pos[A], map[not A]=-1
pack_out allocated from a pool with capacity m_t
Avoid INJECT:
Keep masks packed across levels; only a and map remain dense.

8.7 Locality and gate reordering within a level
Gates in a level are independent; their evaluation order may be permuted without changing semantics.
Group gates by operand packed indices (or blocks of indices) to reduce gather misses:
bucket_key = (block(map_u), block(map_w), opcode, basecase)
Use linear-time bucketization rather than comparison sorting to keep O(m_t) per level.
This reduces constant factors for packed gather and scattered writes.

8.8 Adaptive mode switching (dense fallback)
Packed evaluation can lose when the active ratio is high and predicate work dominates.
Define thresholds tau_low and tau_high.
If p_t <= tau_low, use strict packed COMPACT.
If p_t >= tau_high, use dense evaluation for that level (compute all masks).
Otherwise, use packed with predicate reduction (fingerprints, witness).
This does not affect correctness and ensures worst-case performance does not degrade.

9. PRACTICAL NOTES
   Mask width handling
   If W is not a multiple of 64, the last word must be masked after NOT.

Early exits
Full-word predicates should early-exit on the first discriminating word:
EQUAL exits on first mismatch
DISJOINT exits on first nonzero AND
SUBSET exits on first nonzero (u and not w)

Verification and testing
For development, verify:
final (a(out), d(out)) matches dense baseline for many random seeds
spot-check random gates: packed mask equals dense mask whenever map>=0

10. CONCLUSION
    CFC provides an exact counterfactual execution model based on local Boolean mask laws. COMPACT turns semantic inactivity into physical elimination of mask storage and computation, preserving correctness without approximation. The remaining costs shift to predicate work and packed gathers. The optimization stack presented here reduces full-word predicate frequency through sound gatekeeping (fingerprints), optional exact O(1) equality (interning), and witness certificates for set relations, while improving locality through stable packing, pooling, and level-local reordering. With adaptive dense fallback, the system approaches O(N*L*p) time when masks freeze, and retains O(N*L) worst-case behavior when they do not.

END OF PAPER
