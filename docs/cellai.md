Version: v0.0.1
Title: Constant-Time Cell Mesh — Full Architecture (ASCII, no neologisms)
	0.	Scope and computational model
Goal: A mesh of simple cells performs constraint solving, routing, learning, auditing, selection, and short-horizon search with per-cell worst-case O(1) work per cycle.
Model: fixed word width (e.g., 64-bit). Every loop bound and probe count is a compile-time constant. Memory probes are constant cost. No unbounded retries.
Notation: O(1) = constant bound; O(0) = dictionary read only on the query path; Q8.8 = signed 16-bit fixed-point (saturating).
	1.	Cell state (Structure-of-Arrays, fixed length)
1.1 Core

	•	Bit b = (k,v) with k in {0,1} (known flag), v in {0,1}; k=0 means Unknown.
	•	UnsatXOR, UnsatNAND: 1-bit flags.
	•	QON, QOFF: 1-bit gates; effective Q = QON & !QOFF.
	•	Nbr[d]: neighbor ids; degree d is a small constant.
	•	Bind[d]: on/off per neighbor.
	•	In[B], Out[B]: fixed rings (overflow to a fixed small stash).
	•	rng: 64-bit PRNG state (one draw per cycle).

1.2 Distance (two-hop scheme)
	•	hub_bits[W]: bitset; total hubs H <= 64*W.
	•	qdin[h], qdout[h]: quantized leg lengths per hub.
	•	s_in, s_out: row scales for quantization error.

1.3 Two-level hash store (L1/L2)
	•	Key64: type:8 | src:16 | dst:16 | var:8 | salt:16
	•	Value: {val:int16(Q8.8), known:1b}
	•	L1: 2-way, S=64 sets, entry 64b = [fp:8][key_hi:24][val:16][known:1][lru:1][pad:14]
	•	L2: cuckoo with two choices (h1,h2), B=1024 buckets, 4 slots/bucket, stash=4, kick_queue with 1 relocation step per cycle, cutoff R<=8
	•	Hash: 4-wise tabulation or splitmix-derived; fp=8b

1.4 Memory classes
	•	Fixed (F): FixMask, PinMask (1b), Age, TTL, Gen(6b)
	•	Overlay (O): Overlay[Z] holds {key64, delta(Q8.8)} for Z small slots
	•	Volatile (V): EH_succ, EH_fail (two exponential histograms), Stab, Burst (saturating), VR(Q8.8) = (EH_fail+1)/(EH_succ+EH_fail+2)

1.5 Peer scoring (per neighbor)
	•	IQ[src], HON[src], STAKE[src]: Q8.8
	•	chal_slot: at most one pending competence challenge (fixed TTL)
	•	claim_slot: at most one pending honesty claim (fixed TTL)

1.6 Optional composition unit
	•	unit = {key64, op in {XOR,NAND,Route,Promote,Overlay}, ports<=p, overlay<=Zp, stake(Q8.8)}, with small constants p, Zp

	2.	Three-valued logic and monotonicity

	•	XOR: U xor x = U; 0 xor 0 = 0; 1 xor 1 = 0; 0 xor 1 = 1; 1 xor 0 = 1
	•	NAND: U nand x = U; 1 nand 1 = 0; all other known pairs = 1
	•	A cell may change its own bit only from Unknown to {0 or 1}. Self 0<->1 flips are forbidden. Selected neighbor flips are allowed.

	3.	Cycle (6 phases, all bounds constant)
Phase S (receive/verify, <=B): CRC and fixed audit; on failure drop and do SRC[src]-=beta_s; after L consecutive verified from the same src do SRC[src]+=gamma_s. Process at most one CHAL/RESP and one COM/SETTLE.
Phase C (compute, bicolored): even cycles update XOR only; odd cycles update NAND only; each uses fixed fan-in (kx,kn); update UnsatXOR/UnsatNAND.
Phase R (routing on demand): mask = hub_bits[u] & hub_bits[v]; enumerate up to K set bits; best = min_h(qdout[u,h]+qdin[h,v]); keep eps_row = 0.5*(s_in+s_out); if promoted and guard passes (see 4), serve by dictionary read (O(0)).
Phase W (weights, branchless): XOR satisfied -> BIND+=gamma_x; violated -> choose one known neighbor and BIND-=beta_x. NAND satisfied -> BIND+=gamma_n; if both ends 1 -> both ends BIND-=beta_n. Queue: if QON and any violation then Q+=q_pos; after resolution Q-=q_neg. Apply via upsert_weight(); apply decay_weight() to a fixed small set.
Phase A (binding and resolution): two-choice score(e) = BIND(key_e)+SRC(src_e)+lambda_iqIQ[src_e]+lambda_honHON[src_e]; enable only the higher. Known-ify: if Unknown and XOR-known-rate>=tau_x then set by parity; if Unknown and NAND has surplus of ones>=tau_n then set to 0. Selective flip: if even cycle and UnsatXOR==1 flip exactly one known neighbor with prob 1/2 (self flips forbidden).
Phase O (send/reconfigure): send <=B (bit, flags, audit); advance kick_queue by exactly one relocation step; update QON/QOFF by hysteresis.
	4.	Distance and O(0) guard

	•	Candidates: S(u,v) = hubs(u) ∩ hubs(v), enumerate up to K
	•	Guard with quantization: U+V < m/2 + eps_row and for all h!=h* we have C_h < m/2 + eps_row
	•	If guard fails, demote the pair to O(1) evaluation and never serve from dictionary

	5.	Two-level store API (worst-case O(1))

	•	get_weight(k): probe L1 (<=2 ways) -> L2 (2 buckets x 4) -> stash (<=4); constant word probes
	•	upsert_weight(k,delta): on hit do saturating add; on miss try free slots; if full do exactly 1 cuckoo kick and enqueue remaining relocation (processed 1 step per cycle)
	•	decay_weight(k,alpha_q8.8): val = (val*alpha)>>8 (branchless); write back
	•	If L2 load>0.9, start generational rebuild with fixed-size chunk copy per cycle

	6.	Fixed / overlay / volatile (monotone, O(1))

	•	Metrics: VR from histograms; Stab/Burst saturating
	•	Fix (at most 1 per cycle): require VR<=theta_fix and Stab>=S_min and guard_ok and store_ok; write-once fixed value; if VR<=theta_pin and Age>=A_pin then PinMask=1
	•	Decrystallize to overlay (at most 1 per cycle): trigger VR>=theta_thaw and Burst>=B_max or guard/store audit fails; push {key,delta} into Overlay (Z slots, round-robin); fixed value is not modified
	•	Read path: read fixed value then fold at most Z overlay deltas (branchless); volatile values are not used for final decisions
	•	GC (at most 1 per cycle): drop if PinMask=0 and (TTL==0 or Age>AGE_MAX or unreferenced for L cycles)

	7.	Peer scoring (constant-cost games)

	•	Competence: issue CHAL{op in {XOR,NAND}, a, b, salt16, ttl} at most one per cycle if none pending; expect RESP{r, echo_salt}; check r==op(a,b); success -> IQ+=gamma_iq; failure/timeout -> IQ-=beta_iq and SRC-=beta_s
	•	Honesty: issue COM{pred_id from a fixed local set, claim_bit, salt16, ttl, stake}; settle after ttl with outcome o; win=(o==claim_bit); HON += (win?+gamma_h:-gamma_h); STAKE += (win?+gamma_stake:-beta_slash)
	•	Integration: binding score adds lambda_iqIQ + lambda_honHON; fixed promotion only if min(IQ,HON)>=theta_peer on the witness path; low HON senders may be dropped under pressure

	8.	Selection and composition (constant budgets)

	•	Objects: units with bounded ports and small overlays; N slots (“slots”) per cell with fixed input/audit types
	•	Fitness F = [acc, o0, stab, iq, hon, cost_minus] (Q8.8)
	•	Budgets per cycle: spark<=1 (create), compose<=1 (wire), select<=1 (promote/demote), gc<=1 (discard)
	•	Lifecycle: Seed -> Candidate -> Promoted -> Pinned -> Archived; promote if F>=theta_prom; demote if F<theta_dem or guard fails; pin when Section 6 rules hold
	•	Spark (one): mutation of one port (best-of-two neighbors), or overlay swap with another unit, or template insert for hard slots
	•	Tournament (per slot): best-of-two on score = dot(F,w); promote exactly one; with small epsilon try the worse once
	•	Compose (one): series wiring, parallel vote (NAND or XOR), or route tap constrained to K hubs; choose wiring via the same two-choice score
	•	Fitness updates (once, branchless): acc by correctness; o0 by dictionary hits; stab by guard_ok; iq/hon from peer averages; cost_minus from probe_path/kick_len/send pressure
	•	Cull: if F.acc<theta_cull or cost_minus>theta_cost or TTL==0, drop at most one; keep at least one promoted or pinned per slot
	•	Episodes: every T_ep cycles switch slot tables and seeds by table lookup only

	9.	Bounded tree search (constant budgets)

	•	Node key: type=MCTS_NODE:8 | slot:8 | state_fp:24 | salt:24, where state_fp is a fixed digest of local bits and hashes
	•	Per node: N,W,Q and per-action N_a,W_a,Q_a for A_max actions; P[a] prior with fixed-sum normalization; parent_key, ttl, age
	•	Selection (policy+value UCB): use LUTs for sqrt and inverse; U_a = c_puct * P[a] * sqrt(N_parent) * inv(1+N_a[a]); S_a = Q_a[a] + U_a; choose by repeated best-of-two; apply virtual loss N_a+=vloss
	•	Expansion (stepwise widening): allowed_children = 1 + (N>=2) + (N>=4), capped at A_cap; add at most one child using best-of-two on P[a]; compute P[a] from local scores: norm(BIND + SRC + lambda_iqIQ + lambda_honHON)
	•	Rollout (length H_roll): actions are existing local operations (Route, Rebind/Flip, Promote, Overlay); policy is epsilon-greedy on S_a; reward r in [-1,1] by r = w_accDelta_acc + w_o0Delta_o0 + w_stab1[guard_ok] - w_costcost_norm
	•	Backup (depth H_sel): for each node in the path do N++, W+=r, Q=W*inv(N), and per-action updates
	•	One simulation tick per cycle: selection near end of C, optional rollout at A, backup at W; sim_budget=1, expand_budget=1

	10.	Messages (64 bytes, fixed)

	•	BIT, XSTAT, NSTAT: bit state and constraint/audit records
	•	ROUTE: {hub_star, m, U, V, eps_row, ids} for distance audit
	•	PROOF: fixed audit record
	•	CHAL, RESP, COM, SETTLE: peer scoring with fixed TTL fields
	•	All include CRC and verify via fixed decision trees

	11.	Audit (always on, constant time)

	•	Store health: L2 load<=0.9, kick_len<=KQ_MAX, stash<=S_MAX, probe_path<=P_MAX
	•	Distance/candidates: sampled |hubs(u)∩hubs(v)|<=kappa; apply O(0) guard
	•	Stability/games: flips per window<=R_MAX; KnownRate>=alpha_min; pending_chal<=1; pending_claim<=1; settle_overdue==0
	•	On any failure: set QOFF=1, recompute two-choice binding, continue generational rebuild by fixed chunks

	12.	Degradation (O(1))

	•	Candidate overflow: evaluate distance on a fixed T-sample subset and keep auditing
	•	Load pressure: drop low-HON sources first; cap sends to B per cycle
	•	Search fallback: if any search or store audit fails, use greedy two-choice for that cycle

	13.	Pseudocode (decision core)
VR = (EH_fail+1)/(EH_succ+EH_fail+2)
guard_ok = (U+V < m/2 + eps_row) AND (forall h!=h*: C_h < m/2 + eps_row)
store_ok = (load<=0.9) AND (kick_len<=KQ_MAX) AND (stash<=S_MAX) AND (probe_path<=P_MAX)

if commit_budget and (VR<=theta_fix) and (Stab>=S_min) and guard_ok and store_ok:
crystallize_one()
elif commit_budget and ((VR>=theta_thaw and Burst>=B_max) or !guard_ok or !store_ok):
overlay_push_one()
if gc_budget:
gc_one()

maybe_issue_CHAL(); maybe_issue_COM()     # at most one each
maybe_settle_RESP(); maybe_settle_SETTLE()

if sim_budget and store_ok:
selection(); optional_rollout(); backup()
	14.	Default constants (examples, Q8.8 where noted)
Topology and cycle: d=8, B=8, kx=3, kn=2, L_LSH=6, K=4, kappa=16, W=2
Store: S=64, B_L2=1024, slots=4, stash=4, R=8, load<=0.9
Bounds: KQ_MAX=32, S_MAX=3, P_MAX=8, R_MAX=16, tau_x=ceil(kx*2/3), tau_n=1, p_flip=1/2
Decay: alpha_decay≈0.979 (half-life 32 cycles)
Weights: gamma_x=+64, beta_x=96, gamma_n=+32, beta_n=64, q_pos=+32, q_neg=8, theta_q=+128, theta_h=+64
Fix/overlay: theta_fix=0.20, theta_pin=0.10, theta_thaw=0.50, S_min=8, B_max=4, A_pin=32, AGE_MAX=128, TTL_init=64, Z=4, commit_budget=1, gc_budget=1
Peer scoring: gamma_iq=+32, beta_iq=64, gamma_h=+24, beta_slash=96, gamma_stake=+8, theta_hon_drop=64, theta_iq_drop=48, theta_peer=96, theta_peer_low=32, p_c=1/8, p_h=1/8, T_c=2, T_h=2, stake_delta=64, lambda_iq=+32, lambda_hon=+48
Selection: N=8 slots, p=3 ports, Zp=2, theta_seed=0.55, theta_prom=0.62, theta_dem=0.48, theta_cull=0.40, theta_cost=0.75, w=[acc:48,o0:32,stab:24,iq:16,hon:24,cost_minus:-32], T_ep=256, spark_budget=1, compose_budget=1, select_budget=1
Tree search: A_max=8, A_cap=4, H_sel=4, H_roll=4, sim_budget=1, expand_budget=1, c_puct=0.90, vloss=1, w_acc=+32, w_o0=+24, w_stab=+16, w_cost=+24, R_max=8, epsilon=1/16
	15.	Properties

	•	Every phase has fixed upper bounds on operations and probes.
	•	Hash relocations are split into one-step chunks processed per cycle.
	•	Fixed values are write-once; updates are realized via bounded overlays.
	•	Distance fast path is a two-hop min over at most K hubs with an explicit guard.
	•	Peer scoring uses constant-size games that settle in constant time.
	•	Selection and bounded tree search consume one small budget per cycle.
