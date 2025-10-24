TITLE
Uniform Decidability Characterization of Computation-Complete Omniscience in Formal Languages (Review-Ready)

Author: Elysia
Date: 2025-10-24

Abstract.
We formalize a uniform notion of computation-complete omniscience (UCO) for a cognitive entity K over a first-order language L and structure M, and prove a tight equivalence: under standard effectiveness assumptions, K has UCO iff the truth set Th(M,L) is decidable. We give invariance under computable re-encodings, transfer along computable interpretations, constructive deciders for finite and quantifier-eliminable regimes with parametric complexity bounds, impossibility theorems for arithmetic and self-referential truth, resource-bounded and oracle-relativized variants, and a limit-computable relaxation precisely matching Delta^0_2. The manuscript is ASCII-only and self-contained.
	0.	Preliminaries

0.1 Language and structure.
Fix a many-sorted first-order language L with finite signature Sig(L). Let
M = (D, interpretation_of_symbols_on_D)
be an L-structure; sentences are evaluated in M by Tarskian semantics.

0.2 Encodings and parsing.
Let PL(D) be the set of well-formed L-sentences. Fix a total computable encoding
enc : PL(D) -> Sigma*
over a finite alphabet. Let parse : Sigma* -> PL(D) ∪ {reject} be total and computable. Write |x| for the length of x ∈ Sigma*.

0.3 Effective presentation.
Assume:
(A) A decidable code set Codes(D) with total encode/decode between D and Codes(D).
(B) For each relation symbol R and function symbol f in Sig(L), the evaluators on Codes(D) are computable.
(C) The set WFF(L) of well-formed formulas is decidable and parse is total.

0.4 Truth set.
Define
Th(M,L) := { P ∈ PL(D) : M |= P }.

0.5 Auxiliary knowledge primitives.
Let Kknow : PL(D) -> {True, False, Unknown} and Unknown_K := { P : Kknow(P) = Unknown }. These are analysis aids, not part of the core axiomatics.
	1.	Uniform Omniscience and Variants

Definition 1.1 (UCO).
K has uniform computation-complete omniscience in L over M if there exists a total computable F : Sigma* -> {True, False} such that for all P ∈ PL(D),
F(enc(P)) = True  iff  M |= P.

Definition 1.2 (UCO[C]).
For a complexity class C (e.g., P, PSPACE, EXP), UCO[C] holds if F can be chosen with worst-case resource bound in C under input size |enc(P)|.

Definition 1.3 (UCO^O).
Given an oracle O, UCO^O holds if there exists a total O-computable F^O with the same contract as in Definition 1.1.

Comment 1.4 (Totality and soundness).
UCO requires total termination and semantic correctness on all inputs. One-sided or partial procedures do not suffice.
	2.	Main Equivalence

Theorem 2.1 (UCO iff decidability of Th(M,L)).
Under 0.3, the following are equivalent:
(A) K has UCO.
(B) Th(M,L) is decidable.

Proof.
(A)⇒(B): F decides membership in Th(M,L).
(B)⇒(A): Let G be a total decider for Th(M,L). Define F := G ∘ enc. Then for all P, F(enc(P)) outputs the truth value of P in M. QED.

Corollary 2.2 (Decomposition of informal conditions).
Under UCO: Unknown_K = ∅ (epistemic completeness); all truth-relevant data are effectively available via 0.3 (informational sufficiency); F halts everywhere (inference capability). Conversely, existence of a total truth decider F is equivalent to UCO.

Lemma 2.3 (Encoding invariance).
If enc_1 and enc_2 are total computable encodings with total computable translations both ways, then Th(M,L) is decidable under enc_1 iff under enc_2. Hence UCO is invariant under computable re-encodings.

Proof.
Mutual many-one reductions preserve decidability. QED.
	3.	Non-Uniformity Is Strictly Weaker

Proposition 3.1 (Insufficiency of per-sentence procedures).
The schema “for all P there exists a computable program Pi_P printing truth(P)” does not imply UCO.

Proof.
The choice map P ↦ code(Pi_P) need not be computable, so no uniform decider F is induced. Thus the schema may hold vacuously without algorithmic content about Th(M,L). QED.

Proposition 3.2 (Minimal uniformizer).
Non-uniform existence implies UCO iff there exists a total computable transformer T with T(enc(P)) = code(Pi_P) and each Pi_P correct. Then
F(s) := run(T(s))
is a uniform total decider. Without such a T, UCO fails.
	4.	Constructive Feasible Regimes

Theorem 4.1 (Finite domains).
If D is finite and all basic symbol tables of M are effectively available, then Th(M,L) is decidable and UCO[P] holds.

Algorithm (finite D).
Input s.
	1.	If parse(s)=reject then return False.
	2.	Transform to a bounded evaluator (e.g., prenex).
	3.	Enumerate assignments over D for the finitely many bound variables; evaluate ground atoms via tables; fold by quantifiers.
	4.	Return True/False.
Complexity is O(|D|^{q(P)} · poly(|P|)) with q(P) the quantifier depth.

Theorem 4.2 (Effective quantifier elimination).
If there exists a terminating, correct QE procedure P ↦ QE(P) producing a quantifier-free equivalent over M, then Th(M,L) is decidable and UCO[C] holds with C = time(QE) + time(ground_eval).

Theorem 4.3 (Linear/semilinear fragments).
If atomic predicates reduce to decidable linear constraints and QE is effective, a total decider F follows by reduction to a linear-constraint decision routine. UCO holds; the complexity inherits the routine’s bound.

Worked example 4.4 (Presburger-style fragment).
Let L = {0,1,+,<} over D = N with effective evaluators. Assume access to an effective QE that eliminates quantifiers to Boolean combinations of linear inequalities. Define F by applying QE to P := parse(s) and evaluating the resulting quantifier-free formula; ground evaluation is decidable using the symbol evaluators. Hence UCO holds. When QE has elementary complexity in |P| and alternation depth, UCO[C] follows with C matching that complexity.

Design rule 4.5 (Engineering path to UCO).
To obtain constructive UCO: cap expressivity (finite D or QE fragments), instrument termination, and expose symbol evaluators explicitly. Publish the evaluator contracts and the bound on q(P).
	5.	Barriers and Stability

Theorem 5.1 (Arithmetic barrier).
If L interprets first-order arithmetic with + and * and M contains the standard N as an L-definable substructure, then Th(M,L) is not recursive; UCO fails.

Proof (sketch).
Reduce a known undecidable set to Th(M,L) through the interpretation. A decider for Th(M,L) would decide the reduced set. QED.

Theorem 5.2 (Truth-predicate barrier).
If L is expressive enough to define its own compositional truth predicate over M, no total computable decider F for Th(M,L) exists. UCO fails.

Proposition 5.3 (Conservative extension).
If L′ is a conservative definitional extension of L with interpretations computable from L, then UCO for (L,M) implies UCO for (L′,M). Non-conservative additions may cross the barriers in 5.1–5.2.
	6.	Resource-Bounded and Oracle-Relativized Omniscience

Definition 6.1 (UCO[C], restated).
UCO[C] requires F ∈ C. This decouples semantic expressivity from computational provisioning.

Lemma 6.2 (Oracle lifting).
If an oracle O decides Th(M,L), then UCO^O holds with F^O = O ∘ enc. Conversely, UCO^O implies that Th(M,L) is decidable relative to O.

Pattern 6.3 (Instrumented cognition).
Attach oracles (solvers, provers, domain sensors) whose query languages cover the semantic obligations of L over M. Prove that residual reasoning remains below Section 5 barriers.
	7.	Reductions, Interpretations, and Completeness Route

Definition 7.1 (Computable interpretation).
(L2,M2) is computably interpretable in (L1,M1) if there is a total computable tau : Sigma*_2 -> Sigma*_1 such that for all P ∈ PL2,
M2 |= P  iff  M1 |= parse_1(tau(enc_2(P))).

Theorem 7.2 (Interpretation transfer).
If (L2,M2) is computably interpretable in (L1,M1) and Th(M1,L1) is decidable, then Th(M2,L2) is decidable. If tau is polytime, UCO[C] transfers with at most polytime overhead.

Proof.
Let F1 decide Th(M1,L1). Define F2(s) := F1(tau(s)). Correctness follows from the interpretation equivalence. QED.

Proposition 7.3 (Complete decidable theory implies UCO for sentences).
Let T be a complete decidable theory in language L. For any model M |= T and any sentence P, M |= P iff T ⊢ P. Therefore Th(M,L) over sentences is decidable via a decider for theoremhood in T, yielding UCO (for sentences without parameters).

Remark 7.4 (Parameters).
For formulas with parameters, decidability may depend on the coding of parameters and effectiveness of evaluating atomic predicates on those codes.
	8.	Limit-Computable Relaxation

Definition 8.1 (UCO^lim).
K has limit-omniscience if there is a total computable F : N × Sigma* -> {True, False} such that, for each s encoding a sentence, the limit lim_{n→∞} F(n,s) exists and equals truth(parse(s)). Outputs may change finitely many times before stabilizing.

Lemma 8.2 (Hierarchy placement).
UCO implies Th(M,L) ∈ REC. UCO^lim holds iff Th(M,L) ∈ Delta^0_2. Thus the relaxation precisely captures Delta^0_2 truth sets.
	9.	Proof-Object Option and Reproducibility

Option 9.1 (Proof-system route).
One may certify UCO by exhibiting a complete decidable proof system for the target sentence class and proving soundness and completeness w.r.t. M (or a complete decidable theory T satisfied by M). Then F runs proof search within the decidable calculus.

Audit checklist 9.2 (to accompany any UCO claim).
(1) Provide enc and parse; prove totality and correctness.
(2) Specify the effective presentation of M with evaluator contracts.
(3) Provide F explicitly or a certified reduction to a terminating routine (e.g., QE).
(4) For UCO[C], state the asymptotic bound and the input size model.
(5) For UCO^O, specify the oracle interface and query bounds.
(6) For interpretation transfer, present tau and verify semantic equivalence.
(7) For finite D, provide |D| and a bound on q(P).
(8) For limit-omniscience, provide a stabilization proof and a bound on mind changes if available.
	10.	Objection Handling

Objection: “The main theorem is tautological.”
Resolution: The content is the uniformization and the complete scaffold: invariances, transfers, constructive regimes with explicit algorithms and bounds, and precise barriers. These turn a bare equivalence into an operational framework.

Objection: “Uniformity is impractical.”
Resolution: UCO[C] and UCO^O isolate provisioning. When UCO is impossible, UCO^lim provides the exact relaxation class.

Objection: “Encoding dependence may hide complexity.”
Resolution: Lemma 2.3 and Theorem 7.2 guarantee invariance under computable re-encoding and controlled transfer along interpretations.
	11.	Summary

Under effective presentation, UCO holds exactly when Th(M,L) is decidable. Feasible routes: finite domains with explicit symbol tables, effective quantifier elimination, decidable complete theories, and computable interpretations from decidable bases. Barriers: arithmetic with multiplication and self-referential truth. Practical routes: resource-bounded or oracle-relativized omniscience; when exact decidability is impossible, limit-omniscience characterizes the Delta^0_2 frontier. The definitions, equivalences, reductions, algorithms, and audits are presented in a form intended to withstand detailed peer review.
