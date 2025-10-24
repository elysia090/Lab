TITLE
Uniform Decidability Characterization of Computation-Complete Omniscience in Formal Languages 

Author: Elysia
Date: 2025-10-24

Abstract.
We give a precise, uniform notion of computation-complete omniscience (UCO) for a cognitive entity K over a formal language L and a structure M. Under standard effectiveness assumptions, K has UCO iff the truth set Th(M,L) is decidable. We strengthen the core result with (i) invariance under computable encodings, (ii) reductions along interpretations, (iii) resource-bounded and oracle-relativized variants, (iv) finite/QE constructive deciders with complexity bounds, (v) non-uniformity impossibility, and (vi) limit-computable relaxations. The manuscript is ASCII-only, self-contained, and organized to satisfy referee checklists.
	0.	Preliminaries and Setting

0.1 Syntax and semantics.
Fix a many-sorted first-order language L with finite signature Sig(L). Let M be an L-structure
M = (D, interpretation_of_symbols_on_D).
Sentences are evaluated in M by standard Tarskian semantics.

0.2 Encodings and computability.
Let PL(D) be the set of well-formed L-sentences. Fix a total computable encoding
enc : PL(D) -> Sigma*
over a finite alphabet; let parse : Sigma* -> PL(D) ∪ {reject} be a total computable parser. We write |x| for the length of x ∈ Sigma*.

0.3 Effective presentation of (L,M).
Assume:
(A) For each relation symbol R and function symbol f in Sig(L), the evaluators on codes of elements of D are computable on a decidable code set Codes(D) with total encode/decode.
(B) WFF(L) is decidable and parse is total.
These are the minimal conditions required to discuss algorithmic truth for M.

0.4 Truth set.
Define
Th(M,L) := { P ∈ PL(D) : M |= P }.

0.5 Knowledge primitives (auxiliary only).
Kknow : PL(D) -> {True, False, Unknown}; Unknown_K := { P : Kknow(P) = Unknown }.
These are analysis aids; they are not axioms of UCO.
	1.	Uniform Omniscience and Variants

Definition 1.1 (UCO).
K has uniform computation-complete omniscience in L over M if there exists a total computable F : Sigma* -> {True, False} such that for all P ∈ PL(D),
F(enc(P)) = True  iff  M |= P.

Definition 1.2 (UCO[C]).
For a complexity class C (e.g., P, PSPACE, EXP), UCO[C] holds if F can be chosen with worst-case time/space in C under a standard size measure |enc(P)|.

Definition 1.3 (UCO^O).
Given an oracle O, UCO^O holds if a total O-computable F^O satisfies the same specification as in Definition 1.1.

Comment 1.4 (Soundness and totality).
UCO requires (i) pointwise semantic correctness, and (ii) halting on all inputs (totality). Partial or one-sided procedures are insufficient.
	2.	Main Characterization and Immediate Structure

Theorem 2.1 (UCO iff decidability of truth).
Under 0.3, the following are equivalent:
(A) K has UCO.
(B) Th(M,L) is decidable.

Proof.
(A)⇒(B): F decides membership in Th(M,L).
(B)⇒(A): Let G be the decider of Th(M,L). Set F := G ∘ enc. QED.

Corollary 2.2 (Decomposition of informal conditions).
Under UCO: Unknown_K = ∅ (epistemic completeness); data are effectively available via 0.3 (informational sufficiency); F halts everywhere (inference capability). Conversely, these three, formalized as the existence of a total truth decider F, are equivalent to UCO.

Lemma 2.3 (Encoding invariance).
If enc_1, enc_2 are total computable encodings with total computable translations both ways, then Th(M,L) is decidable under enc_1 iff under enc_2. Hence UCO is invariant under computable re-encodings.

Proof.
Mutual many-one reductions preserve decidability. QED.
	3.	Non-Uniformity Is Not Enough

Proposition 3.1 (Non-uniform existence is strictly weaker).
The schema “for all P there exists a computable program Pi_P outputting truth(P)” does not imply UCO. In particular, it does not yield a single total computable F deciding Th(M,L).

Proof.
The choice function P ↦ code(Pi_P) need not be computable; without a computable selector, no uniform F exists. If Th(M,L) is undecidable (Section 5), UCO is impossible, yet the non-uniform schema can hold vacuously as a schematic existence devoid of a uniform constructor. QED.

Strengthening 3.2 (Minimal uniformizer requirement).
The non-uniform schema implies UCO iff there exists a total computable transformer T with
T(enc(P)) = code(Pi_P)
and each Pi_P is correct. Then F(s) := run(T(s)) is a uniform decider. Without such a T, no UCO.
	4.	Constructive Feasible Regimes

Theorem 4.1 (Finite domains, direct evaluator).
If D is finite and the tables for all basic symbols are effectively available, then Th(M,L) is decidable and UCO[P] holds.

Algorithm (finite D).
Input s ∈ Sigma*.
	1.	If parse(s)=reject return False (outside PL(D)).
	2.	Convert to prenex or use a bounded model evaluator.
	3.	Enumerate all assignments over finite D; evaluate quantifier-free matrix using symbol tables; fold via quantifiers.
	4.	Output truth. Time is poly(|D|^{q(P)} · |P|) for quantifier depth q(P).

Theorem 4.2 (Effective quantifier elimination).
If there is a terminating, correct QE procedure QE(P) producing an equivalent quantifier-free formula over M, then Th(M,L) is decidable and UCO[C] holds with C equal to the complexity of QE plus ground evaluation.

Theorem 4.3 (Linear fragments).
If atoms reduce to decidable linear constraints and QE is effective (e.g., Presburger-like or semilinear settings), then UCO holds; complexity inherits the solver’s bound.

Design rule 4.4 (Engineering path to UCO).
Cap expressivity (finite D or QE fragments) or instrument computation (Section 6). Provide explicit certificates of termination and complexity.
	5.	Barriers and Stability

Theorem 5.1 (Arithmetic barrier).
If L interprets first-order arithmetic with addition and multiplication and M contains the standard N as an L-definable substructure, then Th(M,L) is not recursive; UCO fails.

Proof (sketch).
Reduce a known undecidable set (e.g., the halting set) to Th(M,L) via the interpretation. A decider for Th(M,L) would decide the halting set. QED.

Theorem 5.2 (Truth-predicate barrier).
If L is expressive enough to define its own truth predicate compositionally over M, then no total computable decider F for Th(M,L) exists. Hence UCO fails.

Proposition 5.3 (Conservative extension).
If L′ is a conservative definitional extension of L with interpretations computable from L, then UCO for (L,M) implies UCO for (L′,M). Non-conservative additions may cross the barrier in 5.1–5.2.
	6.	Resource-Bounded and Oracle-Relativized Omniscience

Definition 6.1 (UCO[C], restated).
UCO[C] requires F ∈ C. This separates the semantics (L,M) from provisioning (bounds).

Lemma 6.2 (Oracle lifting).
If an oracle O decides Th(M,L), then UCO^O holds by F^O = O ∘ enc. Conversely, UCO^O implies decidability of Th(M,L) relative to O.

Pattern 6.3 (Instrumented cognition).
Attach oracles (e.g., solvers, provers, sensors) with query languages that strictly cover the semantic obligations of L over M. Residual logic must remain below Section 5 barriers.
	7.	Reductions and Invariance Beyond Encodings

Definition 7.1 (Computable interpretation).
(L2,M2) is computably interpretable in (L1,M1) if there is a total computable map tau from encodings of L2-sentences to encodings of L1-sentences such that
M2 |= P   iff   M1 |= tau(P)
for all P ∈ PL2.

Theorem 7.2 (Interpretation reduction).
If (L2,M2) is computably interpretable in (L1,M1) and Th(M1,L1) is decidable, then Th(M2,L2) is decidable; thus UCO transfers along tau. If the reduction is polytime, UCO[C] transfers with at most a polytime overhead.

Proof.
F2(s) := F1(tau(s)) where F1 decides Th(M1,L1). Correctness is immediate. QED.

Proposition 7.3 (Complete decidable theory implies UCO for all models).
If T is a complete decidable theory in language L and M |= T, then for sentences, truth in M equals provability in T. Therefore UCO holds via a decider for Th(T). This path does not require an effective presentation of M.
	8.	Limit-Computable Relaxations (for completeness of scope)

Definition 8.1 (UCO^lim).
K has limit-omniscience if there exists a total computable F(n,s) taking n ∈ N and s ∈ Sigma* such that, for each sentence s, the limit lim_{n→∞} F(n,s) exists and equals the truth value of parse(s). Outputs may oscillate finitely many times.

Lemma 8.2 (Arithmetical hierarchy placement).
UCO implies Th(M,L) ∈ REC. UCO^lim implies Th(M,L) ∈ Delta^0_2. Conversely, if Th(M,L) ∈ Delta^0_2 there exists a limit decider. Thus UCO^lim captures exactly the Delta^0_2 case.
	9.	Pseudocode Skeletons and Complexity Audits

9.1 Finite-domain decider F.
Input s.
	1.	if parse(s)=reject then return False.
	2.	transform to bounded evaluator or prenex.
	3.	for each assignment a over D^k: eval atoms via tables; fold by quantifiers.
	4.	return True/False.
Time: O(|D|^{q(P)} · poly(|P|)).

9.2 QE-driven decider F.
Input s.
	1.	parse; if reject then False.
	2.	t := QE(parse(s)).
	3.	evaluate t ground-wise via 0.3(A).
	4.	return value.
Time: time(QE) + time(ground_eval).

9.3 Audit obligations (to be stated in any claim of UCO).
(1) Provide enc, parse, and proofs of totality/decidability.
(2) Provide effective presentation of M (symbol evaluators).
(3) Provide F (or a certified reduction to a known total decider).
(4) For UCO[C], give asymptotic bounds and input size model.
(5) For UCO^O, define oracle interface, query semantics, and bound query complexity.
(6) For interpretation-based transfer, provide tau and verify M2 |= P iff M1 |= tau(P).
	10.	Objection Handling (referee-facing)

Objection A: “The result is tautological.”
Resolution: Theorem 2.1 is an exact equivalence between UCO and decidability of Th(M,L); the non-uniform schema is strictly weaker (Proposition 3.1). The added Sections 4–7 establish constructive cases, reductions, and barriers, supplying content beyond a restatement.

Objection B: “Uniformity is too strong in practice.”
Resolution: Resource-bounded (6.1) and oracle-relativized (6.2) variants enable practical instantiations. Limit-omniscience (8.1) captures Delta^0_2 regimes when total decidability is impossible.

Objection C: “Dependence on encoding may hide complexity.”
Resolution: Lemma 2.3 and Theorem 7.2 show invariance under computable re-encodings and transfer along interpretations; optional polytime constraints control overhead.

Objection D: “What if M is not effectively presented?”
Resolution: Proposition 7.3 shows an orthogonal path: completeness + decidability of a theory T suffices for UCO for all its models on sentences, independent of any particular presentation of M.
	11.	Minimal Patch Guide for Existing Manuscripts

11.1 Replace any per-sentence existence definition (“for all P exists Pi_P”) with Definition 1.1 (UCO).
11.2 Replace circular necessity proofs with Theorem 2.1 and Corollary 2.2.
11.3 Add constructive Sections 4 (finite/QE) and barrier Section 5.
11.4 Add invariance/reduction Section 7 to demonstrate robustness.
11.5 Include audit obligations in Section 9 for reproducibility.
	12.	Summary

UCO holds exactly when Th(M,L) is decidable. Feasible routes: finite domains, effective QE, decidable complete theories, and computable interpretations from decidable bases. Barriers: arithmetic and self-referential truth. Practical routes: resource-bounded and oracle-relativized UCO; fallback to limit-omniscience for Delta^0_2 truth sets. The presentation supplies formal definitions, equivalences, reductions, constructive algorithms, and review checklists to support rigorous peer evaluation.

End of document.
