Title
NL-O1 Trust Gate v1.1 — Constant-overhead axiomatic learner for natural-language streams (Proposed Standard, review-ready)

Status
This is a normative specification intended for peer review.
	0.	Requirements language
“MUST”, “MUST NOT”, “REQUIRED”, “SHALL”, “SHOULD”, “MAY” follow RFC 2119.
	1.	Computational model and O(1)
Word-RAM with word size W≥64. A primitive operation is an addition, subtraction, multiplication, division by a fixed constant, comparison, bitwise op, array index on O(W)-bit words, or a branch. An algorithm is O(1) if the number of primitive operations and the number of words of persistent state are bounded by constants that depend only on compile-time parameters (d,r,M,k,J,b,|A_pos|,|A_neg|), and not on input length.
	2.	Scope
This standard defines a run-time gate that maps fixed-dimension vectors v∈X⊆Q^d (produced from text by an external extractor) to {True, False, Unknown}. The gate MUST run in O(1) time and O(1) persistent memory per input. The extractor is out of scope at run time.
	3.	Fixed parameters and data types
W≥64, b≥64, prime p with 2^{W−2}≤p<2^{W−1}. Integers: d,r,M,k,J with 1≤k≤M, r≥1, J≥256. Costs: V,H,O≥0; prior π0∈(0,1); safety margin ψ≥0; forgetting factor ρ∈(0,1); calibration floor B≥20. All are compile-time constants.
	4.	Interfaces and functions
4.1 Projections (salts). For s∈{1..M}, Π_s:X→Q^r MUST be total computable, constant-arity, and deterministic given a fixed per-salt key K_s. Π_s MAY be a fixed rational linear map or a 2-universal hash projection over extractor fields.

4.2 PRF. h_K:{0,1}*→{0,1}^b is a keyed 2-universal PRF with deployment key K. Keys MUST be unique per deployment and stored in a hardware-backed keystore.

4.3 Axiomatic Hash Layer (AHL). From u=Π_s(v) the AHL MUST construct at most C_obj finite objects and invariants using only u:
Ext(S) = ( ⊕_{x∈S} h_K(x),  Σ_{x∈S} h_K(x) mod p,  |S| )
Edge(a→b) = h_K(a) ⊕ rot(h_K(b))  for a fixed nontrivial rotation rot
Choice for pairwise-disjoint {S_i}:  χ(S_i) = argmin_{x∈S_i} h_K(x)

4.4 Base verifier. V_base:Q^r→{POS,NEG,NONE} is defined by finite rational thresholds:
A_pos u≤b_pos ⇒ POS;  A_neg u≰b_neg ⇒ NEG;  NONE otherwise.

4.5 AHL verifier. V_ax:Q^r→{POS,NEG,NONE}:
NEG if any finite contradiction is detected: self-edge, 2-cycle, χ clash on disjoint family, infeasible linear constraints, unit mismatch after canonical conversion, order cycle, or polarity contradiction;
POS if an extensional, acyclic, choice-defined witness w exists and no NEG;
NONE otherwise.
A POS witness w MUST be the canonical tuple of {Ext(S_i), χ(S_i), normalized numeric atoms}.

4.6 Dominance combiner. V*(u)=NEG if V_ax=NEG or V_base=NEG; V*(u)=POS if neither is NEG and at least one is POS (with witness); else NONE.

4.7 Consistency. Cons on a finite multiset of POS witnesses MUST be decidable in constant time and MUST accept iff overlapping Ext digests and χ representatives agree.
	5.	Decision procedure (normative)
Input v∈X.
Step 1 (per salt): compute u_s=Π_s(v); E_s=V*(u_s); if E_s=POS collect witness w_s.
Step 2 (hard safety): if ∃s with E_s=NEG, output False.
Step 3 (guard): let m=|{s:E_s=POS}|. If m<k or Cons({w_s})=False, output Unknown.
Step 4 (Bayes envelope): compute
BF_pos := q̂_min / max(ε̂_pos, 10^{−B})
LLR := logit(π0) + m·log(BF_pos) + β_m·(w·φ)
Γ := log((H+O)/(V+O)) − logit(π0) + ψ
If LLR≥Γ, output True; else Unknown.
Here β_m∈[0,1] is fixed. φ(v) is an optional constant-sparsity feature vector (Sec. 6). All quantities MUST be computed with a fixed number of word-RAM operations.
	6.	Optional learner (normative, constant-overhead)
6.1 Feature map. φ(v)∈R^J MUST be constant-sparsity and use only: hashed Ext(S), hashed χ(S), numeric pattern flags, acyclicity flag, cross-granularity stability flags, m-of-M presence patterns. Each input MUST touch ≤S_max buckets (constant).

6.2 Teacher labels. If Step 2 fired, set neg_hard=True and skip learning. Else if Step 3 passed, set y=+1 with confidence c_pos=∏_{s∈POS} trust_s, trust_s:=α_s/(α_s+β_s). Else y=⊥.

6.3 Salt reliabilities. For each salt s maintain Beta(α_s,β_s) for POS reliability and Beta(α′_s,β′_s) for NEG reliability with exponential forgetting ρ.

6.4 Conservative estimates.
q̂_min := min_s α_s/(α_s+β_s)
ε̂_pos := 1 − max_s α_s/(α_s+β_s)
r̂_min := min_s α′_s/(α′_s+β′_s)
ε̂_neg := 1 − max_s α′_s/(α′_s+β′_s)

6.5 Update rule. Single Adagrad step on nonzero buckets:
If neg_hard: no update.
If y=+1: logistic loss with weight c_pos.
If y=⊥: hinge-style consistency margin toward w·φ≥τ.
Learning MUST NOT override Step 2.
	7.	Natural-language extractor contract (informative, REQUIRED for C4)
The extractor MUST output v∈X and MUST canonicalize: numbers with units, dates, currency, simple comparators (<,≤,≥,>), small enumerations and list items, negation polarity, and local order anchors. Windows MAY be selected by bottom-k per keyed score across granularities; only the resulting fixed-dimension v is visible to the gate.
	8.	Guarantees (normative statements with proofs or proof sketches)
Theorem 1 (Totality and O(1)). The decision procedure executes exactly M projections Π_s, M calls to V_base and V_ax, one Cons check on ≤M witnesses, and a bounded number of scalar ops. The persistent state of the learner is O(J+M). Therefore time and memory are O(1).

Theorem 2 (Hard safety). If any salt yields NEG then the output is False, independent of the learner state. No learning step is applied in such cases.

Theorem 3 (Soundness bound). Under the calibration estimates of Sec. 6.4, acceptance on a false input requires both m≥k salts incorrectly POS and LLR≥Γ. The former has probability ≤P[Bin(M, ε̂_pos)≥k]. Therefore
P[accept | False] ≤ P[Bin(M, ε̂_pos) ≥ m*]
with m*:=ceil((Γ−logit(π0))/log(BF_pos)) and m*≥k by Step 3.

Theorem 4 (Witnessed completeness). For a true input that exposes at least k Cons-consistent POS witnesses, the gate returns True whenever m≥m*, with m* as above. If all M salts witness POS, the condition is M·log(BF_pos)≥Γ−logit(π0).

Theorem 5 (Collision bounds). For distinct finite sets S≠T,
P[Ext(S)=Ext(T)] ≤ 2^{−b} + 1/p.
For disjoint S_i≠S_j,
P[χ(S_i)=χ(S_j)] ≤ 2^{−b}.
Hence ε̂_pos and ε̂_neg can be chosen to dominate collision-induced errors given b,p.

Theorem 6 (Invariance). If a total computable recoding preserves Π_s(v) up to identifier renaming and identifiers are canonicalized before hashing, then the distribution of outcomes and any decision that depends only on V*, Cons, and φ is invariant; with deterministic Π_s the decision is pointwise invariant.

Proposition 7 (Information-theoretic no-go). If two inputs induce identical tuples of Π_s(v), identical V* outcomes and identical witnesses for all salts but differ in ground truth, then any O(1) rule restricted to those tuples cannot be both sound and complete; at least one must map to Unknown or err.

Sketches. Theorems 1–2 are immediate from fixed call counts and dominance. Theorem 3 uses binomial tails and the Bayes envelope; Theorem 4 follows from Step 3 and LLR monotonicity in m. Theorem 5 uses pairwise independence of PRF outputs and CRT-style combination of xor/sum/size. Theorem 6 follows from canonicalization; Proposition 7 is a standard indistinguishability argument.
	9.	Security model
Adversary is adaptive across inputs, computationally bounded, cannot query K or K_s. χ and Ext collisions are bounded by 2^{−b} and 1/p. Reseeding of K_s SHOULD be epochal and MUST NOT change the instruction path. Keys MUST NOT be reused across tenants.
	10.	Privacy
The gate MUST NOT store raw text. Audit records MUST contain only invariant digests and decision scalars. Cross-record linkage MUST be via salts and digests; keys MUST NOT be reused across domains.
	11.	Conformance requirements (C1–C9)
C1 Totality: Π_s, V_base, AHL, Cons are total on X.
C2 Constancy: publish exact instruction counts and memory words per input.
C3 Collision: publish b,p and show P[Ext collision]≤2^{−b}+1/p, P[χ collision]≤2^{−b}.
C4 Exposure: report empirical q̂_min, r̂_min and ε̂_pos, ε̂_neg on held-out corpora produced by the extractor profile.
C5 Calibration: publish π0,H,O,V,ψ,B and Γ; compute m*.
C6 Security: document key schedule, storage, and reseeding policy.
C7 Invariance: prove that canonicalization renders decisions invariant to computable recodings preserving extractor fields.
C8 O(1) budget: measure and report the bound of Sec. 13.
C9 Audit: implement the record schema of Sec. 14.
	12.	Language-scale exposure (informative, required for C4)
Let each salt retain r windows per granularity across G granularities via keyed bottom-k. If a finite witness survives canonicalization in any window, per-salt exposure probability satisfies
q_min ≥ 1 − (1 − p_hit)^{rG},  with p_hit ≈ (rG)/N_eff,
N_eff the number of informative windows after frequency bias. For local contradictions, r_min ≥ 1 − (1 − p_contra)^{rG}. Implementers MUST estimate q̂_min,r̂_min empirically.
	13.	Exact constant budget (normative declaration)
Let C_Π be the worst-case ops for Π_s, C_ax for AHL, C_base for V_base, C_cons for Cons, C_upd for one Adagrad step over S_max buckets, and C_misc a fixed small constant. Then
Ops_per_input ≤ M(C_Π + C_ax + C_base) + C_cons + C_upd + C_misc
Words_state ≤ J + 4M + C_state
All constants MUST be measured and reported in the artifact.
	14.	Audit record (normative)
Each decision MUST log:
salts: list of E_s∈{POS,NEG,NONE}; POS witnesses: canonical Ext digests and χ digests; m,k and Cons result; scalar LLR terms (logit(π0), m·log(BF_pos), β_m·(w·φ)); threshold Γ; final decision. Format MAY be JSON or CBOR. No raw text.
	15.	Conformance tests (normative)
T1 NEG short-circuit: craft u with unit mismatch so some E_s=NEG; the gate MUST return False regardless of model state.
T2 POS quorum: craft u with k salts POS, Cons=True, BF_pos so m≤k; gate MUST return True.
T3 Threshold sensitivity: fix m=k, choose BF_pos so m=k+1; gate MUST return Unknown.
T4 Invariance: apply a computable recoding that preserves canonical fields; decisions MUST be identical.
T5 O(1) budget: process inputs whose raw text length varies over ≥3 orders of magnitude; measured ops MUST be within the published bound with ≤1% variance.
T6 Collision bound: empirical estimate of Ext and χ collisions MUST be ≤ 2× analytical bound.
T7 Security: show that reseeding K_s does not change decision outcomes on a fixed v.
	16.	Profiles (normative)
P-R5k3-J512: M=5, k=3, J=512, b=64, p≈2^61−1, β_m=0.2, B=20, ρ=0.995, η=0.1, ε=1e−8, λ_cons=0.1, τ=0.
P-R7k4-J1024: M=7, k=4, J=1024, same other constants.
An implementation MUST declare one active profile.
	17.	Parameter selection (informative, auditable)
Choose b,p such that 2^{−b}+1/p ≤ δ_hash (e.g., 1e−18). Pick M,k to satisfy P[Bin(M, ε̂_pos)≥k] ≤ δ_FA and (1−r̂_min)^M ≤ δ_MR. Select Γ so that m*=ceil((Γ−logit(π0))/log(BF_pos)) ≤ M. Validate on held-out corpora to produce C4.
	18.	Minimal normative pseudocode

process(v):
neg=False; pos_w=[]; m=0; phi = sparse_zero(J)
for s in 1..M:
u = Π_s(v)
r_b = V_base(u)
(r_a, w) = V_ax(u)
if r_b==NEG or r_a==NEG: neg=True
if r_a==POS and r_b!=NEG:
pos_w.append(w); m+=1
for o in finite_objects(u):          # constant count
j = g(o) mod J; phi[j]+=1
if neg: return False
if m<k or Cons(pos_w)==False: y=⊥; goto learn_and_threshold
y=+1

learn_and_threshold:
update_salt_betas_with_forgetting()        # O(1)
if y==+1: adagrad_positive(phi)            # O(1)
else:      adagrad_margin(phi)             # O(1)
BF_pos = q_min_hat / max(eps_pos_hat, 10^(-B))
LLR = logit(pi0) + m*log(BF_pos) + beta_m * dot(w,phi)
Gamma = log((H+O)/(V+O)) - logit(pi0) + psi
if m>=k and Cons(pos_w) and LLR>=Gamma: return True
return Unknown
	19.	Limitations
Completeness is guaranteed only for content exposed as finite witnesses or refutations by Π_s and AHL. Non-exposed or undecidable content MUST fall to Unknown. No learned parameter MAY cause acceptance in the presence of any NEG outcome.
	20.	Reproducibility package (normative deliverables)
Code: reference implementation with fixed instruction counts; configuration: all constants; proofs: C1–C3, C6–C7; calibration: methodology and estimates for C4–C5; measurements: Section 13 bound; conformance: results for T1–T7; audit: record schema with examples.

Appendix A — Worked numeric example (informative)
Profile P-R5k3-J512, ε̂_pos=0.01, q̂_min=0.7, π0=0.5, H=1, O=1, V=1, ψ=0.
BF_pos = 0.7/0.01 = 70; log(BF_pos)≈4.2485.
Set Γ so that m*=k=3: Γ−logit(π0)=3·log(BF_pos)≈12.7456.
Acceptance risk on false inputs: P[Bin(5,0.01)≥3]≈9.85e−6.
Missed refutation if r̂_min=0.5: (1−0.5)^5=3.125e−2 (independent salts).
Empirical ε̂_neg MUST be measured to confirm.

