Title
NL-O1 Trust Gate v0.0.1 — Constant-overhead axiomatic gate for natural-language streams (Proposed Standard, review-ready)

Status
This is a normative draft intended for peer review.
	0.	Requirements language
“MUST”, “MUST NOT”, “REQUIRED”, “SHALL”, “SHOULD”, “MAY” follow RFC 2119.
	1.	Computational model and O(1)
Word-RAM with word size W ≥ 64. A primitive op is addition, subtraction, multiplication, division by a fixed constant, comparison, bitwise op, array index on O(W)-bit words, or a branch. An algorithm is O(1) if the number of primitive ops and the number of words of persistent state are bounded by constants that depend only on compile-time parameters (d, r, M, k, J, b, |A_pos|, |A_neg|), not on input length.
	2.	Scope
This standard defines a run-time gate mapping fixed-dimension vectors v ∈ X ⊆ Q^d (produced from text by an external extractor) to {True, False, Unknown}. The gate MUST run in O(1) time and O(1) persistent memory per input. The extractor is out of scope at run time.
	3.	Fixed parameters and data types
W ≥ 64; b ≥ 64; prime p with 2^{W−2} ≤ p < 2^{W−1}. Integers d, r, M, k, J with 1 ≤ k ≤ M, r ≥ 1, J ≥ 256. Costs V, H, O ≥ 0; prior π0 ∈ (0,1); safety margin ψ ≥ 0; forgetting factor ρ ∈ (0,1); calibration floor B ≥ 20. All are compile-time constants.
	4.	Interfaces and functions
4.1 Projections (salts). For s ∈ {1..M}, Π_s: X → Q^r MUST be total computable, constant-arity, and deterministic given a fixed per-salt key K_s. Π_s MAY be a fixed rational linear map or a 2-universal hash projection over extractor fields.

4.2 PRFs vs bucket hashes.
PRF h_K: {0,1}* → {0,1}^b is a keyed pseudorandom function with deployment key K. Separately, the feature-bucket maps g_i used in Sec. 6 MUST come from an independent 2-universal hash family. PRF and 2-universal hashing MUST NOT be conflated.

4.3 Axiomatic Hash Layer (AHL). From u = Π_s(v), the AHL MUST construct at most C_obj finite objects and invariants using only u:
Ext(S) = ( ⊕_{x∈S} h_K(x),  Σ_{x∈S} h_K(x) mod p,  |S| ), with set semantics (no multiplicity).
Edge(a→b) = h_K(a) ⊕ rot(h_K(b)) for a fixed nontrivial rotation rot.
Choice on pairwise-disjoint {S_i}: χ(S_i) = argmin_lex( h1(x), h2(x) ), where h1,h2 are independent keyed PRFs; this fixes tie-break deterministically.

4.4 Base verifier (linear/units/order). V_base: Q^r → {POS, NEG, NONE} is defined by finite rational thresholds:
A_pos u ≤ b_pos ⇒ POS;  A_neg u ≰ b_neg ⇒ NEG;  NONE otherwise.
Units and simple order contradictions (after canonical numeric conversion) MUST be expressed in these linear constraints (not in V_ax).

4.5 AHL verifier (finite-witness logic). V_ax: Q^r → {POS, NEG, NONE}:
NEG if any finite contradiction is detected: self-edge, 2-cycle, or χ clash on a declared disjoint family.
POS if an extensional, acyclic, choice-defined witness w exists and no NEG is present.
NONE otherwise.
A POS witness w MUST be the canonical tuple of {Ext(S_i), χ(S_i)} (constant size).

4.6 Dominance combiner. V*(u) = NEG if V_ax=NEG or V_base=NEG; V*(u) = POS if neither is NEG and at least one is POS (with witness); else NONE.

4.7 Consistency. Cons on a finite multiset of POS witnesses MUST be decidable in constant time and MUST accept iff overlapping Ext digests and χ representatives agree.
	5.	Decision procedure (normative)
Input v ∈ X.

Step 1 (per salt): compute u_s = Π_s(v); E_s = V*(u_s); if E_s = POS collect witness w_s.
Step 2 (hard safety): if ∃s with E_s = NEG, output False.
Step 3 (quorum and consistency): let m = |{s : E_s = POS}|. If m < k or Cons({w_s}) = False, output Unknown.
Step 4 (Bayes envelope, statistics-free v0.0.1): set β_m = 0 and compute
BF_pos := q̂_min / max(ε̂_pos, 10^{−B})
LLR := logit(π0) + m · log(BF_pos)
Γ := log((H+O)/(V+O)) − logit(π0) + ψ
Choose Γ so that m := ceil((Γ − logit(π0))/log(BF_pos)) ≥ k* (REQUIRED).
If LLR ≥ Γ, output True; else Unknown.
All quantities MUST be computed with a fixed number of word-RAM ops.
	6.	Optional learner (normative, constant-overhead; does not alter Steps 2–3)
6.1 Feature map. φ(v) ∈ R^J MUST be constant-sparsity and use only: hashed Ext(S), hashed χ(S), acyclicity flag, and fixed m-of-M patterns. Each input MUST touch ≤ S_max buckets (constant). Hashing uses 2-universal g_i independent of PRFs in 4.3.

6.2 Teacher labels. If Step 2 fired, set neg_hard=True and skip learning. Else if Step 3 passed, set y=+1 with confidence c_pos=∏_{s∈POS} trust_s, trust_s:=α_s/(α_s+β_s). Else y=⊥.

6.3 Salt reliabilities. For each salt s maintain Beta(α_s,β_s) for POS reliability and Beta(α′_s,β′_s) for NEG reliability with exponential forgetting ρ.

6.4 Conservative estimates.
q̂_min := min_s α_s/(α_s+β_s)
ε̂_pos := 1 − max_s α_s/(α_s+β_s)
r̂_min := min_s α′_s/(α′_s+β′_s)
ε̂_neg := 1 − max_s α′_s/(α′_s+β′_s)

6.5 Update rule (O(1)). Single Adagrad step on nonzero buckets:
If neg_hard: no update.
If y=+1: logistic loss with weight c_pos.
If y=⊥: small hinge-style margin toward w·φ ≥ τ.
Learning MUST NOT override Step 2 or Step 3.
	7.	Extractor contract (informative, REQUIRED for C4)
The extractor MUST output v ∈ X and MUST canonicalize numbers with units, dates, currencies, simple comparators, short enumerations, negation polarity, and local order anchors. Only the resulting fixed-dimension v is visible to the gate.
	8.	Guarantees (normative statements with proof sketches)
Theorem 1 (Totality and O(1)). The procedure executes exactly M projections Π_s, M calls to V_base and V_ax, one Cons check on ≤ M witnesses, and a bounded number of scalar ops. The persistent state of the learner is O(J+M). Therefore time and memory are O(1).
Sketch: fixed call counts; constant-arity checks.

Theorem 2 (Hard safety). If any salt yields NEG then the output is False, independent of learner state. No learning step is applied.
Sketch: dominance + learning guard.

Theorem 3 (Soundness bound, β_m=0). Acceptance on a false input requires both m ≥ k incorrect POS with Cons and LLR ≥ Γ. The first event has probability ≤ P[Bin(M, ε̂_pos) ≥ k]. With β_m=0, LLR depends only on m; defining m* := ceil((Γ−logit(π0))/log(BF_pos)) and enforcing m* ≥ k gives
P[accept | False] ≤ P[Bin(M, ε̂_pos) ≥ m*].
Sketch: binomial tail + monotonicity.

Theorem 4 (Witnessed completeness). For a true input with at least k Cons-consistent POS witnesses, the gate returns True whenever m ≥ m*. If all M salts are POS, this is M·log(BF_pos) ≥ Γ − logit(π0).
Sketch: LLR increases linearly in m.

Theorem 5 (Collision bounds). For distinct finite sets S ≠ T,
P[Ext(S)=Ext(T)] ≤ 2^{−b} + 1/p.
For disjoint S_i ≠ S_j, P[χ(S_i)=χ(S_j)] ≤ 2^{−b}.
Hence ε̂_pos and ε̂_neg can be chosen to dominate collision-induced errors with chosen (b,p).
Sketch: pairwise independence + CRT-style combination of xor/sum/size; lexicographic χ.

Theorem 6 (Invariance). If a total computable recoding preserves Π_s(v) up to identifier renaming and identifiers are canonicalized before hashing, then any decision that depends only on V*, Cons, and (optional) φ is distribution-invariant; with deterministic Π_s it is pointwise invariant.
Sketch: decisions compute on invariants.

Proposition 7 (Information-theoretic no-go). If two inputs induce identical tuples of Π_s(v), identical V* outcomes and identical witnesses for all salts but differ in ground truth, then any O(1) rule restricted to those tuples cannot be both sound and complete; at least one must map to Unknown or err.
Sketch: indistinguishability.
	9.	Security model
Adversary is adaptive across inputs, computationally bounded, cannot query K or K_s. χ and Ext collisions are bounded by 2^{−b} and 1/p. Reseeding of K_s SHOULD be epochal and MUST NOT change the instruction path; decisions across epochs MAY differ for the same v but MUST remain within published risk bounds. Keys MUST NOT be reused across tenants.
	10.	Privacy
The gate MUST NOT store raw text. Audit records MUST contain only invariant digests and decision scalars. Cross-record linkage MUST be via salts and digests; keys MUST NOT be reused across domains.
	11.	Conformance requirements (C1–C9)
C1 Totality: Π_s, V_base, AHL, Cons are total on X.
C2 Constancy: publish exact instruction counts and memory words per input (Sec. 13).
C3 Collision: publish (b,p) and show P[Ext collision] ≤ 2^{−b}+1/p, P[χ collision] ≤ 2^{−b}.
C4 Exposure: report empirical q̂_min, r̂_min, ε̂_pos, ε̂_neg on held-out corpora produced by the extractor profile.
C5 Calibration: publish π0, H, O, V, ψ, B and Γ; compute m* and verify m ≥ k*.
C6 Security: document key schedule, storage, reseeding policy.
C7 Invariance: prove canonicalization renders decisions invariant to computable recodings preserving extractor fields.
C8 O(1) budget: measure and report the bound of Sec. 13.
C9 Audit: implement the record schema of Sec. 14.
	12.	Language-scale exposure (informative, for C4)
Let each salt retain r windows per granularity across G granularities via keyed bottom-k. If a finite witness survives canonicalization in any window, per-salt exposure satisfies
q_min ≥ 1 − (1 − p_hit)^{rG}, with p_hit ≈ (rG)/N_eff (without-replacement correction optional), N_eff = effective informative windows. Implementers MUST estimate q̂_min, r̂_min empirically.
	13.	Exact constant budget (normative declaration)
Let C_Π be worst-case ops for Π_s, C_ax for AHL, C_base for V_base, C_cons for Cons, C_upd for one Adagrad step over S_max buckets, and C_misc a fixed constant. Then
Ops_per_input ≤ M (C_Π + C_ax + C_base) + C_cons + C_upd + C_misc
Words_state ≤ J + 4M + C_state
All constants MUST be measured and reported in the artifact.
	14.	Audit record (normative)
Each decision MUST log: salts with E_s ∈ {POS,NEG,NONE}; POS witnesses (canonical Ext and χ digests); m, k and Cons result; scalar LLR terms (logit(π0), m·log(BF_pos)); threshold Γ; final decision. JSON or CBOR. No raw text.
	15.	Conformance tests (normative)
T1 NEG short-circuit: craft u with a linear contradiction so some E_s=NEG; the gate MUST return False regardless of learner state.
T2 POS quorum: craft u with m=k, Cons=True, BF_pos set so LLR ≥ Γ; the gate MUST return True.
T3 Threshold sensitivity: with m=k and Cons=True, set Γ so m* = k+1; the gate MUST return Unknown.
T4 Invariance: apply a computable recoding that preserves canonical fields; decisions MUST be identical.
T5 O(1) budget: process inputs whose raw text length varies over ≥ 3 orders of magnitude; measured ops MUST remain within the published bound with ≤ 1% variance.
T6 Collision bound: empirical estimate of Ext and χ collisions MUST be ≤ 2× the analytical bound.
T7 Reseeding: within an epoch (fixed K_s, K) decisions MUST be deterministic; across epochs, instruction path MUST be unchanged and measured risks MUST remain within published bounds.
	16.	Profiles (normative)
P-R5k3-J512: M=5, k=3, J=512, b=64, p≈2^61−1, B=20, ρ=0.995.
P-R7k4-J1024: M=7, k=4, J=1024, b=64, p≈2^61−1, B=20, ρ=0.995.
An implementation MUST declare one active profile.
	17.	Parameter selection (informative, auditable)
Choose b,p so 2^{−b}+1/p ≤ δ_hash (e.g., 1e−18). Pick M,k so P[Bin(M, ε̂_pos) ≥ k] ≤ δ_FA and (1−r̂_min)^M ≤ δ_MR. Select Γ so m* = ceil((Γ−logit(π0))/log(BF_pos)) ≤ M and ≥ k. Validate on held-out corpora to produce C4.
	18.	Minimal normative pseudocode

function judge(v):
neg=False; pos_w=[]; m=0
for s in 1..M:
u = Π_s(v)
r_b = V_base(u)                 # POS/NEG/NONE
(r_a, w) = V_ax(u)              # POS with witness / NEG / NONE
if r_b==NEG or r_a==NEG: neg=True
if r_a==POS and r_b!=NEG:
pos_w.append(w); m+=1
if neg: return False
if m<k or not Cons(pos_w): return Unknown
BF_pos = q_min_hat / max(eps_pos_hat, 10**(-B))
LLR = logit(pi0) + m*log(BF_pos)
Gamma = log((H+O)/(V+O)) - logit(pi0) + psi
if LLR >= Gamma: return True
return Unknown
	19.	Limitations
Completeness is guaranteed only for content exposed as finite witnesses or finite refutations by Π_s and AHL. Non-exposed or undecidable content MUST fall to Unknown. No learned parameter MAY cause acceptance in the presence of any NEG outcome.
	20.	Reproducibility package (normative deliverables)
Code: reference implementation with fixed instruction counts.
Configuration: all compile-time constants.
Proofs: C1–C3, C6–C7.
Calibration: methodology and estimates for C4–C5.
Measurements: Section 13 bound.
Conformance: results for T1–T7.
Audit: record schema with examples.

Appendix A — Worked numeric example (informative)
Profile P-R5k3-J512; ε̂_pos=0.01; q̂_min=0.7; π0=0.5; H=1, O=1, V=1; ψ=0.
BF_pos = 0.7/0.01 = 70; log(BF_pos) ≈ 4.2485.
Choose Γ so that m* = k = 3: Γ − logit(π0) = 3·log(BF_pos) ≈ 12.7456.
False-accept bound: P[Bin(5, 0.01) ≥ 3] ≈ 9.85e−6.
Missed refutation if r̂_min=0.5: (1−0.5)^5 = 3.125e−2 (independent salts).
Empirical ε̂_neg MUST be measured to confirm C4.

Notes for future versions
– v0.0.1 sets β_m=0. A future revision MAY introduce a bounded statistical factor BF_stat(m) ∈ [1, C_stat] with updated m* := ceil((Γ−logit(π0)−log C_stat)/log BF_pos).
– A multiset-aware Ext′ may be standardized if multiplicities become required.
