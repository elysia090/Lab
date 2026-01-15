TITLE
Multi-seed Boolean Forward-mode AD over GF(2) with Exact Sparsity-aware Compaction
A Semantics-preserving Dual-value Execution Model with Stable Packing

STATUS
Draft, self-contained, ASCII, English

ABSTRACT
We formalize a semantics-preserving execution model for Boolean circuits seen as straight-line programs over GF(2). For a fixed set of W tracked single-bit flip directions, each node carries a dual value (a,d) where a is the base evaluation at x and d is the W-lane vector of directional discrete differences Delta_{S[k]}. We give exact propagation laws for NOT, XOR, AND; in particular AND uses the GF(2) discrete Leibniz identity a_u d_w xor a_w d_u xor (d_u and d_w), eliminating case splits. We then define an exact sparsity-aware compaction operator that removes and does not store provably zero d-vectors while preserving full semantics. Compaction is specified as a strict stable-packing contract driven by an exact nonzero predicate, producing packed dual differences plus a dense-to-packed map. We separate mathematical correctness (dual semantics) from representation contracts (bit-packed lanes, summary trees, optional exact interning), and we state soundness and completeness requirements for compaction. The resulting system performs multi-directional Boolean forward-mode AD in one pass and can exploit exact zero-sparsity without approximation.
	1.	ALGEBRAIC SETTING

1.1 Field and operations
Let F2 = {0,1} be the field with addition xor and multiplication and.
Negation is defined as not a := 1 xor a.
All vector operations on masks are lane-wise over F2 unless stated otherwise.

1.2 Inputs and directions
Let x in F2^n be an input assignment.
Fix W tracked directions S[0..W-1], where each S[k] is an integer in [0..n-1].
Direction k corresponds to flipping the input bit at index S[k], i.e. x xor e_{S[k]}.

1.3 Discrete difference operator
For any function f : F2^n -> F2 and index i in [0..n-1], define the discrete difference
Delta_i f(x) := f(x) xor f(x xor e_i).

For the tracked directions, define for each k:
Delta_{S[k]} f(x) := f(x) xor f(x xor e_{S[k]}).
	2.	CIRCUIT MODEL

2.1 Circuit
A circuit is a finite DAG of nodes. Each node v is one of:
CONST(c) where c in F2
IN(j) where j in [0..n-1]
NOT(u)
XOR(u,w)
AND(u,w)
Edges point from operands to the operator node. The circuit has one or more designated outputs.

2.2 Base semantics
For a node v, val(v,x) in F2 denotes the usual Boolean evaluation at input x.

2.3 Directional counterfactual semantics
For a node v and tracked direction k, define
diff(v,x,k) := val(v,x) xor val(v, x xor e_{S[k]}).
This is exactly the directional discrete difference of the node function at x.
	3.	MULTI-SEED FORWARD-MODE DUAL SEMANTICS

3.1 Dual value
Each node v carries a dual value
J(v) = (a(v), d(v))
where
a(v) in F2 is the base value
d(v) in F2^W is the W-lane difference vector

The intended meaning is:
a(v) = val(v,x)
d(v)[k] = diff(v,x,k) = Delta_{S[k]} val(v, . )(x)

3.2 Seed vectors
For each input index j, define the W-lane seed vector s_j in F2^W by:
s_j[k] = 1 iff S[k] = j else 0.

3.3 Leaf initialization rules
CONST(c):
a = c
d = 0^W

IN(j):
a = x[j]
d = s_j
	4.	EXACT FORWARD PROPAGATION LAWS

All laws below are exact and define how to compute J(out) from J(operands).
All operations on d are lane-wise.

4.1 NOT
Given J(u) = (a_u, d_u),
J(NOT(u)) = (1 xor a_u, d_u).

4.2 XOR
Given J(u) = (a_u, d_u), J(w) = (a_w, d_w),
J(XOR(u,w)) = (a_u xor a_w, d_u xor d_w).

4.3 AND (GF(2) discrete Leibniz rule)
Given J(u) = (a_u, d_u), J(w) = (a_w, d_w),
J(AND(u,w)) = (a_u and a_w, d_out),
where
d_out = (a_u * d_w) xor (a_w * d_u) xor (d_u and d_w).

Here scalar-by-vector multiplication uses:
(0 * d) = 0^W
(1 * d) = d

4.4 Equivalent per-lane form
For each lane k:
d_out[k] = (a_u and d_w[k]) xor (a_w and d_u[k]) xor (d_u[k] and d_w[k]).
	5.	CORRECTNESS OF DUAL SEMANTICS

Theorem 5.1 (Dual correctness)
For every node v in the circuit, if J(v) is computed using the initialization rules in 3.3 and the propagation laws in 4, then:
a(v) = val(v,x)
and for all k in [0..W-1]:
d(v)[k] = val(v,x) xor val(v, x xor e_{S[k]}).

Proof sketch
Proceed by induction over a topological order of the DAG.
Leaves hold by definition.
For NOT and XOR, use that negation does not change differences and xor is linear in F2.
For AND, use the identity for each lane:
(pq) xor (p’ q’) = p (q xor q’) xor q’ (p xor p’) xor (p xor p’) (q xor q’)
with p=a_u, p’=val(u, x xor e_{S[k]}), q=a_w, q’=val(w, x xor e_{S[k]}),
and note that (p xor p’) = d_u[k], (q xor q’) = d_w[k]. This yields the rule in 4.3.

Corollary 5.2 (Output semantics)
For any output node out computing f, a(out)=f(x) and d(out)[k]=Delta_{S[k]} f(x).
	6.	ZERO-SPARSITY AND EXACT COMPACTION

6.1 Zero predicate and activity
Define the exact zero predicate:
ZERO(d) is true iff d = 0^W.

Define node activity:
ACTIVE(v) := not ZERO(d(v)).

If ACTIVE(v) is false, then v contributes no directional differences to any downstream computation under the tracked directions, because its d-vector is identically zero.

6.2 Batch compaction problem
Let B be a batch of m mutually independent nodes v_0..v_{m-1} whose operands have already been computed.
The goal is to compute:
a(v_i) densely for all i,
and store d(v_i) only for ACTIVE(v_i)=true, using stable packed storage.

6.3 Stable packing contract
Input:
A[0..m-1] activity flags where A[i]=1 iff ACTIVE(v_i).
X[0..m-1] items (here, d(v_i) values, or references to them).

Output:
Y is the stable packed list of X[i] with A[i]=1, preserving increasing i order.
There exists pos[i] such that:
if A[i]=1 then pos[i] = rank(A,i) - 1 and Y[pos[i]] = X[i],
if A[i]=0 then pos[i] is undefined.

Define the dense-to-packed map:
map[i] = pos[i] if A[i]=1 else -1.

6.4 Compaction output contract
Compaction produces:
a_out[0..m-1] (dense)
map_out[0..m-1] (dense)
pack_out[0..p-1] (packed), where p = sum_i A[i]

with the semantic invariant:
map_out[i] = -1 iff d(v_i) = 0^W
map_out[i] = r >= 0 implies pack_out[r] = d(v_i)
and packing is stable with respect to i.

6.5 Soundness and completeness of compaction
Soundness requirement (mandatory):
If map_out[i] = -1 then d(v_i) = 0^W.

Completeness requirement (optional but performance-critical):
If d(v_i) = 0^W then map_out[i] = -1.

If both hold, compaction exactly corresponds to eliminating and not storing zero difference vectors.

6.6 Fused compute-and-compact schedule (normative)
For each batch B:
	1.	Compute all base values a(v_i) densely (scalar ops).
	2.	Compute each d(v_i) using the exact laws in section 4.
	3.	Compute A[i] by applying ZERO(d(v_i)) using an exact predicate.
	4.	Compute map_out and p via prefix-sum on A (stable packing indices).
	5.	Write pack_out[map_out[i]] = d(v_i) for active i.
Inactive i perform no packed write.

This fused specification forbids declaring inactivity without an exact ZERO proof.
	7.	REPRESENTATION CONTRACTS (IMPLEMENTATION-LEVEL, SEMANTICS-PRESERVING)

7.1 Bit-packing
Let L = ceil(W/64).
Represent d as L machine words, with lane bit k stored at word floor(k/64), bit (k mod 64).
If W is not a multiple of 64, the high bits of the last word are padding and MUST be masked to zero after bitwise NOT or any operation that could set them.

7.2 Exact nonzero predicate via summary trees (optional but recommended)
Implement ZERO(d) and NONZERO(d) using a maintained root OR summary:
root_or(d) = OR over all L words.
Then:
ZERO(d) iff root_or(d) == 0.
This is exact.

Maintain root_or for each produced d:
either compute it alongside word generation (streaming OR),
or maintain a k-ary segment-OR tree where internal nodes store OR of children, enabling polylog span updates and queries.

7.3 Exact equality via interning (optional)
Maintain an exact canonicalization table mapping the full word sequence of a mask to a unique id(d).
Then EQUAL(d1,d2) can be answered by id(d1) == id(d2), which is exact.
Interning MUST resolve collisions by full word comparison and MUST never merge unequal masks.

7.4 Witness certificates (optional)
Store with each nonzero mask a witness index widx such that d[widx]=1.
Witnesses may accelerate proofs of NONZERO for intersections or differences but MUST never be used to prove ZERO.
Any failure of witness-based short-circuit MUST fall back to exact predicates.
	8.	SCHEDULING AND GLOBAL EXECUTION

8.1 Topological execution
Evaluate the circuit in any topological order.
Compaction is defined on any batch of nodes whose operands are already available.
A scheduler MAY choose batches to optimize locality or active density, provided that data dependencies are respected.

8.2 Dense-to-packed operand access
Operands may be stored as:
base values a(u) densely indexed by node id,
difference vectors d(u) as packed values plus map(u) to locate the packed entry.

When an operand u has map(u) = -1, it MUST be treated as d(u)=0^W.
	9.	COMPLEXITY PARAMETERS

Let n_g be total gate count.
Let W be lane count and L=ceil(W/64).
Let p be the fraction of active nodes whose d is nonzero at a given batch or level.

Baseline (no compaction):
Work: O(n_g * L) word ops
Span: depends on depth and word-parallelization; without additional structure, per-node d cost is O(1) span if words are processed in parallel, but work remains O(L).

With exact compaction:
Work reduces by avoiding storage and downstream processing for nodes proven to have d=0^W.
The model does not change worst-case asymptotics; it changes the effective active set size.

If nonzero testing uses root summaries, ZERO checks are O(1).
Stable packing uses prefix-sum with span O(log m) per batch size m.
	10.	IDENTITY STATEMENT

This model is exactly:
multi-direction forward-mode automatic differentiation over GF(2), where the derivative is the discrete difference Delta_i,
augmented with semantics-preserving elimination and stable compaction of zero dual components.

END OF SPECIFICATION
