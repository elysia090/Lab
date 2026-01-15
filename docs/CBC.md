TITLE
R_W-Lifted Dual Execution with Handle-based Zero Elision
A Detailed, Semantics-preserving Specification for Multi-seed Boolean Differences

STATUS
Draft, self-contained, ASCII, English

ABSTRACT
We specify an exact execution model for Boolean circuits under W tracked single-bit flip directions. The mathematical core is a homomorphic lift of the original GF(2) circuit into an extension ring R_W = F2 x F2^W, yielding a uniform dual value J(v)=(a(v),d(v)) per node, where a is the base value and d is the W-lane discrete-difference vector. We then separate semantics from representation by introducing a representation congruence: the semantic zero vector 0^W is represented by a distinguished handle NONE and need not be stored. Nonzero vectors are stored in a mask store addressed by handles and may be managed with hole tolerance and delayed reclamation. The design defines normative contracts for mask loading, exact zero detection, handle lifetime, and permissible optimizations (exact interning, typed exact mask ADTs, one-sided nonzero certificates). All pruning is restricted to soundness-only rules: a value may be treated as NONE only when zero is proven. The specification is agnostic to scheduling policy and does not require stable packing or per-batch prefix sums for correctness.
	1.	DEFINITIONS AND NOTATION

1.1 Field
F2 = {0,1} with operations:
xor : F2 x F2 -> F2
and : F2 x F2 -> F2
not a := 1 xor a

1.2 Inputs and tracked directions
Input x in F2^n.
Tracked directions S[0..W-1], with each S[k] in [0..n-1].
Direction k refers to the perturbed input x^(k) := x xor e_{S[k]}.

1.3 Discrete difference
For f : F2^n -> F2 and i in [0..n-1]:
Delta_i f(x) := f(x) xor f(x xor e_i).
For tracked directions:
Df(x)[k] := Delta_{S[k]} f(x).

1.4 Circuit
A circuit is a DAG of nodes v with opcode in:
CONST(c), IN(j), NOT(u), XOR(u,w), AND(u,w),
where operands reference earlier nodes in a topological order.
A circuit may have multiple outputs.

1.5 Base evaluation
val(v,x) is the usual Boolean evaluation of node v at x.

1.6 Dual evaluation target
For each node v we intend:
a(v) := val(v,x)
d(v)[k] := val(v,x) xor val(v, x xor e_{S[k]})
	2.	THE R_W EXTENSION RING (SEMANTIC CORE)

2.1 Carrier
Define R_W := F2 x F2^W.
Write elements as pairs (a,d) with a in F2 and d in F2^W.

2.2 Addition (lifted XOR)
Define ⊕ on R_W:
(a,d) ⊕ (b,e) := (a xor b, d xor e).
Here d xor e is lane-wise xor of W bits.

2.3 Multiplication (lifted AND)
Define ⊗ on R_W:
(a,d) ⊗ (b,e) := (a and b,  (ae) xor (bd) xor (d and e)).
Scalar-by-vector multiplication is:
0e = 0^W
1e = e
Lane-wise AND is:
(d and e)[k] = d[k] and e[k].

2.4 Identities and NOT
Define:
0_R := (0,0^W)
1_R := (1,0^W)
Define NOT_R(x) := 1_R ⊕ x.

2.5 Algebraic facts (reference)
The structure (R_W, ⊕, ⊗, 0_R, 1_R) is a commutative ring.
Useful identities:
Idempotence: x ⊗ x = x
Complement annihilation: x ⊗ (1_R ⊕ x) = 0_R
These hold for all x in R_W.
	3.	LIFTED EVALUATION AND CORRECTNESS

3.1 Seed vectors
For each input index j define seed vector s_j in F2^W:
s_j[k] = 1 iff S[k] = j else 0.

3.2 Leaf embedding
Embed(CONST(c)) := (c, 0^W)
Embed(IN(j)) := (x[j], s_j)

3.3 Lifted evaluation definition
Define J(v) for each node v by evaluating the circuit in R_W:
	•	CONST and IN use 3.2
	•	NOT uses NOT_R
	•	XOR uses ⊕
	•	AND uses ⊗
All evaluation uses operand J values already computed.

3.4 Projection operators
Define projections:
pi0(a,d) = a
pi1(a,d) = d

3.5 Correctness theorem
Theorem 3.5 (R_W-lift correctness)
For every node v computed by 3.3:
pi0(J(v)) = val(v,x)
and for all k:
pi1(J(v))[k] = val(v,x) xor val(v, x xor e_{S[k]}).
Thus for an output node out computing f:
J(out) = (f(x), Df(x)).

Proof sketch
Induction on topological order.
Leaves hold by construction.
NOT and XOR follow from xor-linearity of Delta and not being affine (not is 1 xor a).
For AND, use the identity for each lane i:
(pq) xor (p’ q’) = p (q xor q’) xor q’ (p xor p’) xor (p xor p’) (q xor q’)
over F2, with p=val(u,x), p’=val(u,x^i), q=val(w,x), q’=val(w,x^i).
This yields the ⊗ definition.
	4.	REPRESENTATION CONGRUENCE AND HANDLE SEMANTICS

4.1 Semantic mask domain
Semantically, each node carries d(v) in F2^W.

4.2 Handles and the NONE element
Define a handle domain Handle, and a distinguished value NONE not in Handle.
A node stores h(v) in Handle union {NONE}.

4.3 Load function (normative)
The runtime MUST provide a total function:
load : (Handle union {NONE}) -> (F2^W),
satisfying:
load(NONE) = 0^W.

4.4 Store function (normative)
The runtime MUST provide a partial function to create handles for nonzero masks:
alloc() -> Handle
store(h, d) stores semantic d at handle h

After store, load(h) MUST equal d exactly.

4.5 Representation congruence
Any two representations are considered equivalent if their load images match.
In particular, treating a value as NONE is semantics-preserving iff its semantic mask is 0^W.

4.6 Soundness-only pruning rule (mandatory)
A computation may set h(v) = NONE only when it has proven that d(v) = 0^W.
Using approximations to conclude d(v)=0^W is forbidden.

4.7 Completeness (optional)
A runtime MAY choose to set h(v)=NONE whenever d(v)=0^W. This improves storage and work but is not required for correctness.
	5.	MASK REPRESENTATION AND EXACT ZERO TEST

5.1 Bit-packed representation
Let L = ceil(W/64).
A dense mask may be represented by L machine words w[0..L-1].
Lane bit k is stored at word floor(k/64), bit (k mod 64).
If W is not a multiple of 64, the high padding bits of the last word MUST be zero.

5.2 Exact nonzero summary (required for stored masks)
Each stored mask MUST carry an exact summary sufficient to decide zero.
Minimal requirement:
root_or = OR_{t=0..L-1} w[t].
Then:
ZERO(d) iff root_or == 0.
This is exact.

A runtime MAY carry additional exact summaries (block ORs, trees), but root_or is required.

5.3 Zero detection procedure (normative)
To decide whether a newly computed d is zero, the runtime MUST use an exact method.
If computing a dense representation, it SHOULD compute root_or while producing words, so no second pass is needed.

5.4 One-sided certificates (optional)
The runtime MAY store witnesses or coarse summaries to prove NONZERO quickly.
Such certificates MUST NOT be used to prove ZERO.
	6.	NODE EVALUATION USING HANDLES

6.1 Node storage
For each node v:
a(v) in F2 is stored densely.
h(v) in Handle union {NONE} is stored.

6.2 Operand retrieval
For operand u:
d_u := load(h(u)).
If h(u)=NONE, then d_u=0^W by 4.3.

6.3 Gate rules in handle form
The runtime computes each node v as:

CONST(c):
a(v) = c
h(v) = NONE

IN(j):
a(v) = x[j]
d(v) = s_j
If s_j == 0^W then h(v)=NONE else store and set handle

NOT(u):
a(v) = not a(u)
d(v) = d_u
h(v) is set to h(u) (aliasing is allowed) or a copy handle to identical content

XOR(u,w):
a(v) = a(u) xor a(w)
d(v) = d_u xor d_w
If d(v)=0^W then h(v)=NONE else store and set handle

AND(u,w):
a(v) = a(u) and a(w)
d(v) = (a(u)*d_w) xor (a(w)*d_u) xor (d_u and d_w)
If d(v)=0^W then h(v)=NONE else store and set handle

6.4 Aliasing and reference counting (implementation detail)
If the store supports aliasing, NOT and other pass-through operations MAY reuse handles.
If reuse is allowed, the runtime MUST ensure handle lifetime safety (e.g., refcounts or epoch GC).
	7.	MASK STORE: LIFETIME, HOLES, AND SAFETY

7.1 Lifetime rules (normative)
A handle h is live from the time it is assigned to any node output until all consumers of that node output have finished loading it.
A runtime MUST prevent reclamation of a handle while any consumer may still load it.

7.2 Delayed reclamation
free_later(h) schedules h for reclamation after it is safe.
Actual reclamation MAY be delayed arbitrarily and MAY be batched.

7.3 Hole tolerance
The store MAY allow holes: freed handles need not be reused immediately.
This does not affect semantics.

7.4 Defragmentation (optional)
The store MAY move stored masks to new handles to reduce fragmentation.
If so, it MUST update all referencing h(v) values consistently.
A moving collector MUST preserve load equivalence exactly.
	8.	OPTIONAL EXACT OPTIMIZATIONS

8.1 Exact interning
The store MAY canonicalize masks:
intern(d) returns a canonical handle id(d) such that equal masks share the same canonical id.
Interning MUST be exact: if id(d1)=id(d2) then d1=d2.

If interning is used, XOR may detect cancellation by id equality.

8.2 Typed exact mask ADT
A store MAY represent masks using an exact tagged union:
NONE/ZERO, SINGLE(i), SPARSE(K), DENSE.
Each variant MUST embed exactly into F2^W.
All operations MUST be exact and MUST preserve semantic equality.

8.3 Lane partitioning
Because operations on d are lane-wise, the runtime MAY partition lanes into chunks and compute them independently.
Chunking MUST preserve lane order in the final assembled vector.

8.4 Circuit rewriting under lift
Any rewriting based on GF(2) identities is valid under Eval_R.
In particular:
associativity and commutativity of xor and and,
idempotence x and x = x,
complement annihilation x and (not x) = 0,
and any derived-gate definition expressed in xor/and/not.
	9.	EXECUTION POLICIES (NON-NORMATIVE)

9.1 Scheduling
The specification is independent of scheduling.
A runtime MAY execute in level order, topological order, or dataflow style.

9.2 Separation of base and difference work
Because base values a(v) do not depend on d, a runtime MAY compute all a(v) first, then compute differences.
This does not change semantics and may enable specialized kernels.

9.3 Storage policy
The specification does not require stable packing.
A runtime MAY choose stable packed arrays, unstable arrays, handle pools, or any other scheme satisfying section 4.
	10.	ADDITIONAL USEFUL LEMMAS (REFERENCE)

Lemma 10.1 (Lane independence)
For fixed base scalars, each lane k of d is computed independently of other lanes.

Lemma 10.2 (No-new-lanes)
For any gate output:
supp(d_out) ⊆ supp(d_u) union supp(d_w).

Lemma 10.3 (Nilpotence)
Delta_i(Delta_i f)(x) = 0 for any i.

Lemma 10.4 (ANF identity)
In algebraic normal form, Delta_i removes variable x_i from monomials containing it and never increases degree.
	11.	CONFORMANCE

An implementation conforms to this specification iff:
	•	It evaluates circuits in R_W as in sections 2 and 3, producing base values and semantic differences exactly.
	•	It implements handle semantics with load(NONE)=0^W.
	•	It never treats a nonzero semantic mask as NONE (soundness-only pruning).
	•	Its zero test is exact.
All additional optimizations are optional and must preserve exactness.

END OF SPECIFICATION
