Title
Sera v0.0.2 – Deep Kernel Streaming Core with Decision and Value Heads

Status
Draft, implementation-oriented, self-contained, constant-time per token and per decision (sequence-length independent)

Language
ASCII, English only
	0.	Scope, notation, configuration

0.1 Scope

This document defines Sera v0.0.2 as a self-contained streaming model core with the following properties:
	1.	Fixed-depth, fixed-width nonlinear token trunk mapping input embeddings to hidden states.
	2.	Two-stage kernel feature layer (large feature psi, compressed feature phi) feeding a low-rank kernel memory.
	3.	Kernel memory with exponential decay, Delta-style gated updates, and multi-scale retention in a single unified interface.
	4.	Finite-dimensional rational memory (stable linear state-space model) with explicit stable parameterization.
	5.	Base and residual heads for language modeling (vocabulary logits).
	6.	Decision head and scalar value head over external discrete actions, suitable as a policy prior and value estimator for external planning or search modules.
	7.	Complexity per token and per decision bounded by configuration constants that do not depend on sequence length or episode length.

The specification fixes state layout, interfaces, and asymptotic complexity. It specifies the update equations and the allowed parameterizations. Optimizer choice, data pipelines, and exact training schedules are outside the scope; section 11 defines a set of normative training objectives compatible with the architecture.

0.2 Notation

R           : real numbers
N           : nonnegative integers {0,1,2,…}
N_pos       : positive integers {1,2,…}

Dimensions:

d_in        : input embedding dimension
d_h         : trunk hidden dimension
L_trunk     : trunk layer count in N_pos
d_mid       : trunk MLP hidden dimension

R_big       : large kernel feature dimension
r_phi       : compressed kernel feature dimension
d_val       : value vector dimension
r_v         : value low-rank dimension

d_diag      : diagnostic feature dimension
d_mem_in    : rational memory input dimension
d_mem       : rational memory state dimension
d_mem_out   : rational memory output dimension

d_base      : base feature dimension
d_rep       : representation dimension
d_tpl_feat  : template feature dimension
M_tpl       : number of templates in N_pos
d_res       : residual head feature dimension

V_size      : vocabulary size (LM output logits dimension)

d_site      : site feature dimension for decision head
d_act       : action feature dimension for decision head
d_dec       : decision head hidden dimension

A_max       : maximum number of candidate external actions per decision

For a vector x in R^d, ||x|| denotes the Euclidean norm. For a matrix M, ||M||_op denotes operator norm. E[·] denotes expectation. For a finite set C, |C| is its cardinality.

For a distribution pi over a finite set C, pi[a] denotes the probability assigned to action a in C.

0.3 Configuration constants

A configuration C_Sera for Sera v0.0.2 consists of:
	1.	Dimensions
d_in, d_h, L_trunk, d_mid,
R_big, r_phi, d_val, r_v,
d_diag, d_mem_in, d_mem, d_mem_out,
d_base, d_rep, d_tpl_feat, M_tpl, d_res,
V_size,
d_site, d_act, d_dec,
A_max.
A_max must be a finite positive integer and all dimensions must be positive integers.
	2.	Kernel memory parameters
gamma_mem          in (0,1]     (exponential decay factor)
lambda_mem         > 0          (denominator floor)
mu_ridge           >= 0         (ridge regularization for low-rank projection)
K_mem              in N_pos     (number of multi-scale kernel memory components)
gamma_mem_k[k]     in (0,1]     for k = 1..K_mem
alpha_mem_k[k]     in R         for k = 1..K_mem (linear combination weights)
	3.	Rational memory parameters
eta_mem            in (0,1)     (stability margin)
d_eig              = d_mem      (number of eigenvalues)
diag_eig[i]        in (-1 + eta_mem, 1 - eta_mem) for i = 1..d_eig
P_mem              in R^{d_mem x d_mem} (invertible)
G_mem              in R^{d_mem x d_mem_in}
H_mem              in R^{d_mem_out x d_mem}
F_mem is defined from P_mem and diag_eig in section 6.
	4.	Nonlinearities
sigma_trunk: R -> R     (e.g. GELU, ReLU) applied elementwise.
sigmoid: R -> (0,1)     applied elementwise or to scalars.
	5.	Decision mode
decision_mode in {“external_actions”}.
In v0.0.2 the decision head is always present for external discrete actions and must support the interfaces in section 10. If a deployment does not require external decisions, it can feed empty action sets and ignore the outputs; the core still implements the decision equations.
	6.	Numerical constants
eps_ln       > 0         (layernorm epsilon)
epsilon_prob in (0,1)    (minimum probability scale)
eps_log      in (0,1)    (log floor for probabilities)
temperature parameters and loss weights used in training (defined in section 11).

All constants in C_Sera are fixed for the lifetime of a run. Changing them defines a new configuration.

0.4 Per-token language modeling interface

Input at time t:

x_t in R^{d_in}              token embedding at step t

Outputs at time t:

z_tok_t in R^{V_size}        unnormalized vocabulary logits
p_tok_t in R^{V_size}        softmax(z_tok_t) distribution

Persistent state:

A_t[k] in R^{r_phi x r_v}    kernel memory matrices for k = 1..K_mem
s_t[k] in R^{r_phi}          kernel memory vectors for k = 1..K_mem
m_t in R^{d_mem}             rational memory state

Initial state:

A_0[k] = 0, s_0[k] = 0 for all k
m_0 = 0

No other state is persisted across tokens.

0.5 Decision-level interface for external actions

Sera is used as a decision prior and value estimator for external discrete actions.

At a decision step aligned with token t, the environment integration provides:

s_dec           : opaque environment state (not interpreted by Sera)
d_dec_site      : abstract decision site identifier for s_dec
C_dec(s_dec)    : finite list of external actions a with |C_dec(s_dec)| in {1,…,A_max}

External feature maps (deterministic, not learned inside Sera):

SiteFeat(s_dec, d_dec_site) -> e_site in R^{d_site}
ActFeat(a)                  -> e_act(a) in R^{d_act}

At that step, Sera receives:

phi_t = h_rep_t in R^{d_rep}   representation from section 7
e_site = SiteFeat(s_dec, d_dec_site)
e_act(a) = ActFeat(a) for each a in C_dec(s_dec)

Sera outputs:

z_dec_t[a] in R for each a in C_dec(s_dec)   unnormalized decision logits
pi_dec_t[a] in R for each a                 decision distribution over actions
V_dec_t in R                                scalar value estimate for state s_dec

The mapping from (phi_t, e_site, e_act(a)) to (z_dec_t[a], pi_dec_t[a], V_dec_t) is fully specified in section 10.
	1.	High-level computational map

Given x_t and persistent states (A_{t-1}[k], s_{t-1}[k], m_t), Sera computes:
	1.	h_t         = Trunk(x_t)
	2.	psi_t, phi_t = kernel feature mapping of h_t
	3.	v_t, r_hat_t = value projection and low-rank coefficient
	4.	A_t[k], s_t[k] updated; y_att_t from kernel memory
	5.	m_{t+1}, y_mem_t from rational memory
	6.	h_base_t, h_rep_t from base and representation projection
	7.	x_tpl_t, q_tpl_t from template classifier
	8.	g_res_t, z_tok_t, p_tok_t from LM head
	9.	For a decision step, given (s_dec, d_dec_site, C_dec), compute z_dec_t[a], pi_dec_t[a], V_dec_t from h_rep_t and external descriptors.
	10.	Trunk specification

2.1 Parameters

P_in               in R^{d_h x d_in}
For each layer ell in {0,…,L_trunk-1}:
W1_trunk[ell]    in R^{d_mid x d_h}
W2_trunk[ell]    in R^{d_h x d_mid}
a_gate[ell]      in R^{d_h}
b_gate[ell]      in R
gamma_ln[ell]    in R^{d_h}
beta_ln[ell]     in R^{d_h}

eps_ln > 0 is a fixed scalar constant.

2.2 Layer normalization

For h in R^{d_h} at layer ell:

mu      = (1 / d_h) * sum_{i=1..d_h} h[i]
var     = (1 / d_h) * sum_{i=1..d_h} (h[i] - mu)^2
std     = sqrt(var + eps_ln)

norm[i] = (h[i] - mu) / std for i in {1..d_h}

LN(h; ell)[i] = gamma_ln[ell][i] * norm[i] + beta_ln[ell][i]

2.3 Per-layer recurrence

Initial:

h_t^(0) = P_in x_t

For each ell in {0,…,L_trunk-1}:

u_t^(ell)   = LN(h_t^(ell); ell)
z1_t^(ell)  = W1_trunk[ell] u_t^(ell)
a1_t^(ell)  = sigma_trunk(z1_t^(ell))                 elementwise
f_t^(ell)   = W2_trunk[ell] a1_t^(ell)

gate_pre    = a_gate[ell]^T u_t^(ell) + b_gate[ell]
g_t^(ell)   = sigmoid(gate_pre)                       scalar in (0,1)

h_t^(ell+1) = h_t^(ell) + g_t^(ell) * f_t^(ell)

Final trunk output:

h_t = h_t^(L_trunk)

2.4 Complexity

Per token:

Cost_trunk = O(L_trunk * d_h * d_mid)

Since L_trunk, d_h, d_mid are configuration constants, Cost_trunk is O(1) with respect to sequence length.
	3.	Kernel feature layer

3.1 psi_fn modes

psi_fn: R^{d_h} -> R^{R_big} maps h_t to psi_t.

Mode psi_RFF (random Fourier features):

Parameters:
W_psi in R^{R_big x d_h}
b_psi in R^{R_big}

Definition:
psi_t[i] = sqrt(2 / R_big) * cos( W_psi[i,:] dot h_t + b_psi[i] )

Mode psi_MLP (learned MLP):

Parameters:
W1_psi in R^{d_mid x d_h}
b1_psi in R^{d_mid}
W2_psi in R^{R_big x d_mid}
b2_psi in R^{R_big}

Definition:
u_psi_t = W1_psi h_t + b1_psi
a_psi_t = sigma_trunk(u_psi_t)
psi_t   = W2_psi a_psi_t + b2_psi

Mode psi_POS (positive orthogonal features):

Parameters:
W_psi in R^{(R_big/2) x d_h}
scale_psi > 0

Computation:
u_psi_t = W_psi h_t
For j in {1..R_big/2}:
psi_t[2j-1] = exp( scale_psi * u_psi_t[j] )
psi_t[2j]   = exp( - scale_psi * u_psi_t[j] )

The configuration selects exactly one psi_mode in {“psi_RFF”, “psi_MLP”, “psi_POS”} and fixes the associated parameters.

3.2 Compression matrix and phi_t

Compression matrix:

C_phi in R^{r_phi x R_big}

Compressed feature:

phi_t = C_phi psi_t in R^{r_phi}

3.3 Kernel approximations

For any h, h_prime in R^{d_h}:

k_big(h, h_prime)  = psi_fn(h)^T psi_fn(h_prime)
k_phi(h, h_prime)  = (C_phi psi_fn(h))^T (C_phi psi_fn(h_prime))

Training losses may encourage k_phi to approximate some target kernel k_target, but that is handled in section 11.
	4.	Value projection and low-rank basis

4.1 Value projection

Parameters:

W_val in R^{d_val x d_h}
b_val in R^{d_val}

Value vector:

v_t = W_val h_t + b_val in R^{d_val}

4.2 Low-rank basis

Parameters:

U_val in R^{d_val x r_v}

Assume v_t is approximated by:

v_t = U_val r_t + e_t

for some r_t in R^{r_v} and residual e_t in R^{d_val}.

4.3 Ridge coefficient r_hat_t

Define:

G_val = U_val^T U_val + mu_ridge * I_{r_v}

Factorize:

G_val = L_val L_val^T

with L_val lower triangular. This factorization is computed once per configuration and reused.

Per token:

b_r_t = U_val^T v_t
Solve L_val y_t = b_r_t by forward substitution.
Solve L_val^T r_hat_t = y_t by backward substitution.

Result:

r_hat_t = G_val^{-1} U_val^T v_t in R^{r_v}
	5.	Kernel memory with multi-scale retention and Delta-style updates

5.1 State and initialization

For k in {1..K_mem}:

A_t[k] in R^{r_phi x r_v}
s_t[k] in R^{r_phi}

Initial:

A_0[k] = 0
s_0[k] = 0

5.2 Update rule

Given phi_t in R^{r_phi}, r_hat_t in R^{r_v}, and previous A_{t-1}[k], s_{t-1}[k]:

For each k in {1..K_mem}:
	1.	Delta term:
delta_A_t[k] = phi_t r_hat_t^T
This is the outer product.
	2.	Gated combination (Delta-style):
Optionally introduce a scalar gate g_mem_t in (0,1):
g_mem_t = sigmoid( w_mem_gate^T h_t + b_mem_gate )
where w_mem_gate in R^{d_h}, b_mem_gate in R are parameters shared across all k.
Define:
A_t[k] = gamma_mem_k[k] * A_{t-1}[k] + g_mem_t * delta_A_t[k]
s_t[k] = gamma_mem_k[k] * s_{t-1}[k] + g_mem_t * phi_t

If g_mem_t is not desired, it can be fixed to 1; the specification requires that the implementation support the gated form, and g_mem_t can be forced to a constant by appropriate parameter choices.

5.3 Query and readout

Given query representation h_q (usually h_t):
	1.	Compute psi_q = psi_fn(h_q) and phi_q = C_phi psi_q.
	2.	For each k in {1..K_mem}:
num_t[k] = phi_q^T A_t[k] in R^{r_v}
den_t[k] = phi_q^T s_t[k] in R
y_att_t[k] = U_val ( num_t[k] / (den_t[k] + lambda_mem) ) in R^{d_val}
Division is elementwise on num_t[k] by the scalar den_t[k] + lambda_mem.
	3.	Combine across scales:
y_att_t = sum_{k=1..K_mem} alpha_mem_k[k] * y_att_t[k] in R^{d_val}
	4.	Rational memory (stable state-space model)

6.1 Parameterization of F_mem

Parameters:

diag_eig in R^{d_mem} with diag_eig[i] in (-1 + eta_mem, 1 - eta_mem)
P_mem in R^{d_mem x d_mem} invertible

Define diagonal matrix:

D_mem[i,i] = diag_eig[i]

Define:

F_mem = P_mem D_mem P_mem^{-1} in R^{d_mem x d_mem}

By construction, all eigenvalues of F_mem have magnitude strictly less than 1 - eta_mem, so the system is stable.

6.2 State and parameters

State:

m_t in R^{d_mem}

Parameters:

F_mem in R^{d_mem x d_mem} as defined above
G_mem in R^{d_mem x d_mem_in}
H_mem in R^{d_mem_out x d_mem}
W_u in R^{d_mem_in x d_h}
B_u in R^{d_mem_in x d_val}
C_u in R^{d_mem_in x d_diag}

6.3 Diagnostic features and input

Diagnostic features:

diag_t = F_diag(h_t, y_att_t) in R^{d_diag}

F_diag is a fixed deterministic mapping with bounded cost. It can be chosen to include norms, clipped values, or low-dimensional summaries of h_t and y_att_t.

Input to rational memory:

u_t = W_u h_t + B_u y_att_t + C_u diag_t in R^{d_mem_in}

6.4 Update and readout

Update:

m_{t+1} = F_mem m_t + G_mem u_t

Readout:

y_mem_t = H_mem m_t in R^{d_mem_out}

6.5 Boundedness

Because all eigenvalues of F_mem are strictly inside the unit circle by at least eta_mem and inputs u_t are bounded in norm by construction, the sequence m_t remains bounded for all t.

6.6 Complexity

Per token:

Cost_mem = O(d_mem^2 + d_mem * d_mem_in + d_mem_out * d_mem)

All dimensions are configuration constants; Cost_mem is O(1) with respect to sequence length.
	7.	Base feature and representation

7.1 Base feature

Define:

concat_base_t = concat(h_t, y_att_t, y_mem_t, diag_t) in R^{dim_concat}

where

dim_concat = d_h + d_val + d_mem_out + d_diag

Parameters:

W_base_proj in R^{d_base x dim_concat}
b_base_proj in R^{d_base}

Base feature:

h_base_t = W_base_proj concat_base_t + b_base_proj in R^{d_base}

7.2 Representation feature

Parameters:

W_rep in R^{d_rep x d_base}
b_rep in R^{d_rep}

Representation:

h_rep_t = W_rep h_base_t + b_rep in R^{d_rep}

We also write phi_t_rep = h_rep_t when used as a generic representation for decision and value heads.
	8.	Template classifier

8.1 Template features

Parameters:

W_tpl_feat in R^{d_tpl_feat x d_base}
b_tpl_feat in R^{d_tpl_feat}

Template features:

x_tpl_t = W_tpl_feat h_base_t + b_tpl_feat in R^{d_tpl_feat}

8.2 Template scores and probabilities

Parameters:

W_tpl in R^{M_tpl x d_tpl_feat}
b_tpl in R^{M_tpl}

Scores for templates:

s_tpl_t[h] = W_tpl[h,:] dot x_tpl_t + b_tpl[h] for h in {1,…,M_tpl}

Template probabilities:

q_tpl_t[h] = exp(s_tpl_t[h] - s_tpl_max) / sum_{k=1..M_tpl} exp(s_tpl_t[k] - s_tpl_max)

where s_tpl_max = max_{k} s_tpl_t[k]. The vector q_tpl_t is used as a feature in the residual LM head and may be used in template classification losses.
	9.	Residual LM head

9.1 Residual input and hidden

Define:

concat_res_t = concat(h_base_t, h_rep_t, q_tpl_t, diag_t) in R^{dim_res_in}

where

dim_res_in = d_base + d_rep + M_tpl + d_diag

Parameters:

W_res1 in R^{d_res_mid x dim_res_in}
b_res1 in R^{d_res_mid}
W_res2 in R^{d_res x d_res_mid}
b_res2 in R^{d_res}

Computations:

a_res1_t = sigma_trunk(W_res1 concat_res_t + b_res1) in R^{d_res_mid}
g_res_t  = W_res2 a_res1_t + b_res2 in R^{d_res}

9.2 Base LM logits

Parameters:

W_out_base in R^{V_size x d_base}
b_out_base in R^{V_size}

Base logits:

z_base_t = W_out_base h_base_t + b_out_base in R^{V_size}

9.3 Residual LM logits

Parameters:

W_out_res in R^{V_size x d_res}
b_out_res in R^{V_size}

Residual LM logits:

r_tok_t = W_out_res g_res_t + b_out_res in R^{V_size}

9.4 Final LM logits and distribution

Final LM logits:

z_tok_t = z_base_t + r_tok_t in R^{V_size}

Token distribution:

p_tok_t[y] = exp(z_tok_t[y] - z_max_t) / sum_{y_prime=1..V_size} exp(z_tok_t[y_prime] - z_max_t)

where z_max_t = max_{y} z_tok_t[y].
	10.	Decision head and scalar value for external actions

The decision head and value head are always present in v0.0.2 and must implement this section. An integration can choose to ignore the outputs but the equations are normative.

10.1 Site and action descriptors

For a decision state (s_dec, d_dec_site) with candidate actions C_dec(s_dec):

e_site = SiteFeat(s_dec, d_dec_site) in R^{d_site}
e_act(a) = ActFeat(a) in R^{d_act} for each a in C_dec(s_dec)

SiteFeat and ActFeat are deterministic functions supplied by the caller.

10.2 Site-conditioned representation

Parameters:

W_site in R^{d_rep x d_site}
b_site in R^{d_rep}

Define:

phi_dec_t = h_rep_t + W_site e_site + b_site in R^{d_rep}

10.3 Action-conditioned hidden and logits

Parameters:

W_dec_cat in R^{d_dec x (d_rep + d_site + d_act)}
b_dec_cat in R^{d_dec}
w_dec_out in R^{d_dec}
b_dec_out in R

For each a in C_dec(s_dec):

concat_dec_t(a) = concat(phi_dec_t, e_site, e_act(a)) in R^{d_rep + d_site + d_act}
h_dec_t(a)      = sigma_trunk(W_dec_cat concat_dec_t(a) + b_dec_cat) in R^{d_dec}
z_dec_t[a]      = w_dec_out^T h_dec_t(a) + b_dec_out in R

Decision distribution:

z_dec_max = max_{b in C_dec(s_dec)} z_dec_t[b]

pi_dec_t[a] = exp(z_dec_t[a] - z_dec_max) /
sum_{b in C_dec(s_dec)} exp(z_dec_t[b] - z_dec_max)

10.4 Scalar value head

Parameters:

w_val_dec in R^{d_rep}
b_val_dec in R

Value estimate for s_dec:

V_dec_t = w_val_dec^T phi_dec_t + b_val_dec in R
	11.	Training objectives

This section defines a set of losses that can be combined to train Sera v0.0.2. All objectives are formally defined; an implementation may set weights to zero to disable specific terms. Loss weights and temperatures are part of the configuration.

11.1 Language modeling cross-entropy

Given ground-truth next token y_{t+1}:

L_ce = - E_t[ log p_tok_t[y_{t+1}] ]

11.2 Trunk and kernel distillation

If a teacher model provides trunk states h_teacher_t in R^{d_teacher} and contextual output y_att_teacher_t in R^{d_val_teacher}, define projections:

h_teacher_proj_t = W_teacher h_teacher_t in R^{d_h}
y_att_teacher_proj_t = W_att_teacher y_att_teacher_t in R^{d_val}

with W_teacher in R^{d_h x d_teacher}, W_att_teacher in R^{d_val x d_val_teacher}.

Define:

L_trunk = E_t[ || h_t - h_teacher_proj_t ||^2 ]
L_att   = E_t[ || y_att_t - y_att_teacher_proj_t ||^2 ]

11.3 Decision policy objectives from advantage signals

Assume for some decision states (s_dec, d_dec_site) we have action set C_dec(s_dec) and an estimated advantage A_hat(d_dec_site, s_dec, a) for each a in C_dec(s_dec).

11.3.1 AWR-style target policy

Let mu_dec(a | d_dec_site, s_dec) be a behavior distribution over C_dec(s_dec). For example:

mu_dec(a | d_dec_site, s_dec)
= beta_ref   * pi_ref_dec(a | d_dec_site, s_dec)
	•	beta_model * pi_dec_t[a]

where pi_ref_dec is any reference policy and beta_ref, beta_model >= 0 with beta_ref + beta_model > 0, normalized appropriately.

Define unnormalized target:

pi_AWR_unnorm[a] = mu_dec(a | d_dec_site, s_dec) * exp(A_hat(d_dec_site, s_dec, a) / beta_AWR)

with beta_AWR > 0.

Normalize:

pi_AWR[a] = pi_AWR_unnorm[a] / sum_{b in C_dec(s_dec)} pi_AWR_unnorm[b]

Define AWR policy loss:

L_dec_AWR = - E_{(d_dec_site,s_dec)} [ sum_{a in C_dec(s_dec)} pi_AWR[a] * log pi_dec_t[a] ]

11.3.2 CRR-style weighted behavioral cloning

Define weights:

w_CRR(d_dec_site, s_dec, a) = f_CRR(A_hat(d_dec_site, s_dec, a))

where f_CRR is a nondecreasing function, for example:

f_CRR_soft(A)   = min( exp(A / beta_CRR), w_CRR_max )
f_CRR_binary(A) = 1 if A > tau_CRR, else 0

with beta_CRR > 0, tau_CRR in R, w_CRR_max > 0.

If (d_dec_site, s_dec, a) are sampled from a dataset with empirical distribution D_dec, define:

L_dec_CRR = - E_{(d_dec_site,s_dec,a) ~ D_dec} [ w_CRR(d_dec_site, s_dec, a) * log pi_dec_t[a] ]

11.3.3 Gumbel-style completed Q target at root decisions

For root decision states, suppose completed values Q_comp(d_dec_site, s_dec, a) are available, combining visited Q estimates with a bootstrap value. Define a monotone transform sigma_Q, for example:

sigma_Q(q) = clip( q / beta_Q, -sigma_max, sigma_max )

with beta_Q > 0 and sigma_max > 0.

Define unnormalized target:

pi_Gumbel_unnorm[a] = pi_dec_t[a] * exp( sigma_Q(Q_comp(d_dec_site, s_dec, a)) )

Normalize:

pi_Gumbel[a] = pi_Gumbel_unnorm[a] / sum_{b in C_dec(s_dec)} pi_Gumbel_unnorm[b]

Gumbel-style policy loss:

L_dec_Gumbel = - E_{(d_dec_site,s_dec) in root decisions} [ sum_{a in C_dec(s_dec)} pi_Gumbel[a] * log pi_dec_t[a] ]

11.4 Decision value regression

If target scalar values V_tar(s_dec) are provided (for example Monte Carlo returns, search-improved value estimates, or Q-based state values), define:

L_val_dec = E_{s_dec} [ ( V_dec_t - V_tar(s_dec) )^2 ]

11.5 LM residual regularization

Define:

L_res_reg = E_t[ || r_tok_t ||^2 ]

11.6 Decision head regularization

Define:

L_dec_reg_h = E_{(d_dec_site,s_dec)} [ (1 / |C_dec(s_dec)|) * sum_{a in C_dec(s_dec)} || h_dec_t(a) ||^2 ]
L_dec_reg_z = E_{(d_dec_site,s_dec)} [ mean_a (z_dec_t[a]^2) ]

These terms penalize excessive magnitude in decision features and logits.

11.7 Total loss

A typical total loss is:

L_total = alpha_ce       * L_ce
+ alpha_trunk    * L_trunk
+ alpha_att      * L_att
+ alpha_dec_AWR  * L_dec_AWR
+ alpha_dec_CRR  * L_dec_CRR
+ alpha_dec_G    * L_dec_Gumbel
+ alpha_val_dec  * L_val_dec
+ alpha_res      * L_res_reg
+ alpha_dec_regH * L_dec_reg_h
+ alpha_dec_regZ * L_dec_reg_z

All alpha_* are nonnegative scalars fixed by configuration or training schedule. Gradients from L_total flow through trunk, kernel features, kernel memory, rational memory, base and residual LM heads, and decision and value heads.
	12.	State and complexity summary

12.1 Persistent state

Persistent state across tokens:

For k in {1..K_mem}:
A_t[k] in R^{r_phi x r_v}
s_t[k] in R^{r_phi}

m_t in R^{d_mem}

Total persistent state size:

K_mem * (r_phi * r_v + r_phi) + d_mem

This size is independent of sequence length.

The decision head does not maintain additional persistent state.

12.2 Per-token computation

Per token, Sera v0.0.2 executes:
	1.	Trunk forward: O(L_trunk * d_h * d_mid)
	2.	psi_fn and phi: O(R_big * d_h + r_phi * R_big)
	3.	Value and r_hat_t: O(d_val * d_h + d_val * r_v + r_v^2)
	4.	Kernel memory update and query across K_mem: O(K_mem * (r_phi * r_v + r_phi)) plus shared feature cost
	5.	Rational memory update and readout: O(d_mem^2 + d_mem * d_mem_in + d_mem_out * d_mem)
	6.	Base and representation: O(d_base * dim_concat + d_rep * d_base)
	7.	Template classifier: O(d_tpl_feat * d_base + M_tpl * d_tpl_feat)
	8.	Residual LM head and logits: O(d_res_mid * dim_res_in + d_res * d_res_mid + V_size * d_base + V_size * d_res)

All dimensions and K_mem are configuration constants. Therefore the per-token work is O(1) with respect to sequence length.

12.3 Per-decision computation

For each decision state (s_dec, d_dec_site):
	1.	External feature extraction (SiteFeat and ActFeat): cost defined outside this spec.
	2.	Site-conditioned representation phi_dec_t: O(d_rep * d_site)
	3.	For each a in C_dec(s_dec):
concat_dec_t(a), h_dec_t(a), z_dec_t[a]: O(d_dec * (d_rep + d_site + d_act))
	4.	Softmax over C_dec(s_dec): O(A_max)
	5.	Value head: O(d_rep)

With |C_dec(s_dec)| <= A_max and all dimensions constant, per-decision cost is O(1) with respect to episode length and sequence length.
	13.	Determinism and numerical stability

13.1 Determinism

Given a fixed configuration C_Sera, fixed parameters, and a fixed input token sequence x_{1:T}, all outputs z_tok_t, p_tok_t, h_rep_t, z_dec_t, pi_dec_t, and V_dec_t for t in {1..T} are deterministic functions of x_{1:T} and the initial state.

If stochastic regularization (such as dropout or additive noise) is used during training, it must be disabled or controlled by explicit pseudorandom seeds for evaluation. No internal randomness is used during inference in this specification.

13.2 Numerical stability guidelines

Implementations must observe the following:
	1.	Use at least 32-bit floating point for h_t, psi_t, phi_t, v_t, r_hat_t, A_t[k], s_t[k], m_t, y_att_t, y_mem_t, h_base_t, h_rep_t, x_tpl_t, q_tpl_t, g_res_t, z_tok_t, z_dec_t, V_dec_t.
	2.	Parameters can be stored in lower precision and cast to 32-bit before computation.
	3.	Enforce stable F_mem by constructing it from P_mem and diag_eig as in section 6.
	4.	In all softmax computations (LM and decision), subtract the maximum logit and use epsilon_prob as a floor for denominators when needed.
	5.	In kernel memory division, always add lambda_mem to denominators and treat very small denominators as lambda_mem for safety.
	6.	Constrain norms of U_val columns and monitor condition number of G_val = U_val^T U_val + mu_ridge I_{r_v}. Increase mu_ridge if conditioning degrades.

These rules ensure bounded internal states and stable numerical behavior.
	14.	Summary

Sera v0.0.2 defines a fully specified, constant-time streaming core with:
	1.	A gated residual trunk with fixed depth and width.
	2.	A two-stage kernel feature layer feeding a multi-scale, Delta-style low-rank kernel memory.
	3.	A rational memory implemented as a stable linear state-space model with explicit spectral control.
	4.	Base and residual heads for language modeling.
	5.	A decision head and scalar value head over external discrete actions, using the same internal representation as the LM head.
	6.	Explicit training objectives for language modeling, representation distillation, and decision-making with advantage- and critic-based policy updates.

