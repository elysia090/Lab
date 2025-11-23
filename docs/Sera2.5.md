Title
Sera v0.0.1+ – Deep Kernel Streaming Core Specification

Status
Draft, implementation-oriented, self-contained, constant-time per token (sequence-length independent)

Language
ASCII, English only
	0.	Scope, notation, configuration

0.1 Scope

This document defines Sera v0.0.1+ as a self-contained streaming model core with the following properties:
	1.	Fixed-depth, fixed-width nonlinear token trunk mapping input embeddings to hidden states.
	2.	Two-stage kernel feature layer (large feature psi, compressed feature phi).
	3.	Low-rank kernel memory with exponential decay and ratio-stable readout.
	4.	Finite-dimensional rational memory (linear state-space model).
	5.	Context-dependent base and residual heads producing logits for next-token prediction.
	6.	Complexity per token bounded by a constant that does not depend on sequence length.

The specification fixes state layout, interfaces, and asymptotic complexity. It does not fix optimizer choice or data pipelines.

0.2 Notation

R           : real numbers
N           : nonnegative integers {0,1,2,…}
N_pos       : positive integers {1,2,…}
d_in        : input embedding dimension
d_h         : trunk hidden dimension
L_trunk     : trunk layer count (in N_pos)
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
M_tpl       : number of templates (in N_pos)
d_res       : residual head feature dimension
V_size      : vocabulary size (output logits dimension)

For a vector x in R^d, ||x|| denotes the Euclidean norm. For matrices, ||·||_op denotes operator norm. E[·] denotes expectation.

0.3 Configuration constants

A Sera configuration fixes:
	1.	All dimensions listed above: d_in, d_h, L_trunk, d_mid, R_big, r_phi, d_val, r_v, d_diag, d_mem_in, d_mem, d_mem_out, d_base, d_rep, d_tpl_feat, M_tpl, d_res, V_size.
	2.	Kernel memory parameters:
gamma_mem in (0,1]            (exponential decay factor)
lambda_mem > 0                (denominator floor)
mu_ridge  >= 0                (ridge regularization)
	3.	Rational memory matrices:
F_mem in R^{d_mem x d_mem}
G_mem in R^{d_mem x d_mem_in}
H_mem in R^{d_mem_out x d_mem}
such that spectral radius rho(F_mem) < 1.
	4.	Nonlinearities:
sigma_trunk: R -> R (e.g. GELU or ReLU) applied elementwise.
sigmoid: R -> (0,1) applied elementwise or to scalars.
	5.	Training-time scalars:
learning rate schedules, weight decay, loss weights; not needed for inference.

Once chosen, configuration constants remain fixed during a run.

0.4 Per-token interface

Input at time t:
x_t in R^{d_in}    (token embedding)

Output at time t:
z_t in R^{V_size}  (unnormalized logits for next token)
optionally p_t in R^{V_size}, p_t[y] = softmax(z_t)[y]

Internal persistent state at time t:
A_t in R^{r_phi x r_v}    (kernel memory matrix)
s_t in R^{r_phi}          (kernel memory vector)
m_t in R^{d_mem}          (rational memory state)

No other per-token state is persisted across time unless explicitly added by an implementation.
	1.	High-level architecture

For each token step t:

1.1 Trunk
h_t = Trunk(x_t) in R^{d_h}

1.2 Kernel feature layer
psi_t = psi_fn(h_t) in R^{R_big}
phi_t = C_phi psi_t in R^{r_phi}

1.3 Value projection and low-rank coefficient
v_t      = W_val h_t + b_val in R^{d_val}
r_hat_t  = RidgeCoeff(v_t) in R^{r_v}

1.4 Kernel memory update
A_t = gamma_mem A_{t-1} + phi_t r_hat_t^T
s_t = gamma_mem s_{t-1} + phi_t

1.5 Kernel memory query
(with query h_q_t, usually h_t)
psi_q_t = psi_fn(h_q_t)
phi_q_t = C_phi psi_q_t
num_t   = phi_q_t^T A_t in R^{r_v}
den_t   = phi_q_t^T s_t in R
y_att_t = U_val ( num_t / (den_t + lambda_mem) ) in R^{d_val}

1.6 Rational memory
diag_t  = F_diag(h_t, y_att_t) in R^{d_diag}
u_t     = W_u h_t + B_u y_att_t + C_u diag_t in R^{d_mem_in}
m_{t+1} = F_mem m_t + G_mem u_t
y_mem_t = H_mem m_t in R^{d_mem_out}

1.7 Base feature and representation
concat_base_t = concat(h_t, y_att_t, y_mem_t, diag_t)
h_base_t      = W_base_proj concat_base_t + b_base_proj in R^{d_base}
h_rep_t       = W_rep h_base_t + b_rep in R^{d_rep}

1.8 Template classifier
x_tpl_t   = W_tpl_feat h_base_t + b_tpl_feat in R^{d_tpl_feat}
s_tpl_t[h] = W_tpl[h] · x_tpl_t + b_tpl[h] for h in {1,…,M_tpl}
q_tpl_t[h] = exp(s_tpl_t[h]) / sum_{k=1..M_tpl} exp(s_tpl_t[k])

1.9 Residual features and logits
g_res_t   = F_res(h_base_t, h_rep_t, q_tpl_t, diag_t) in R^{d_res}
z_base_t  = W_out_base h_base_t + b_out_base in R^{V_size}
r_t       = W_out_res g_res_t + b_out_res in R^{V_size}
z_t       = z_base_t + r_t
	2.	Trunk specification

2.1 Parameters

Trunk parameters:

P_in               in R^{d_h x d_in}
For each layer ell = 0,…,L_trunk-1:
W1_trunk[ell]    in R^{d_mid x d_h}
W2_trunk[ell]    in R^{d_h x d_mid}
a_gate[ell]      in R^{d_h}
b_gate[ell]      in R (scalar)
gamma_ln[ell]    in R^{d_h}
beta_ln[ell]     in R^{d_h}

eps_ln > 0 is a small scalar constant for numerical stability.

2.2 Layer normalization

For a given vector h in R^{d_h} and layer ell:

mu      = (1 / d_h) * sum_{i=1..d_h} h[i]
var     = (1 / d_h) * sum_{i=1..d_h} (h[i] - mu)^2
std     = sqrt(var + eps_ln)
norm[i] = (h[i] - mu) / std for i = 1..d_h

LN(h; ell)[i] = gamma_ln[ell][i] * norm[i] + beta_ln[ell][i]

2.3 Per-layer recurrence

Initial:

h_t^(0) = P_in x_t

For each ell = 0,…,L_trunk-1:

u_t^(ell)   = LN(h_t^(ell); ell)
z1_t^(ell)  = W1_trunk[ell] u_t^(ell)
a1_t^(ell)  = sigma_trunk(z1_t^(ell))            (elementwise)
f_t^(ell)   = W2_trunk[ell] a1_t^(ell)

gate_pre    = a_gate[ell]^T u_t^(ell) + b_gate[ell]
g_t^(ell)   = sigmoid(gate_pre)                  (scalar)

h_t^(ell+1) = h_t^(ell) + g_t^(ell) * f_t^(ell)

Final trunk output:

h_t = h_t^(L_trunk)

2.4 Complexity

Per token:

Cost_trunk = O(L_trunk * (d_h * d_mid))

Since L_trunk, d_h, d_mid are fixed constants, Cost_trunk is O(1) with respect to sequence length.
	3.	Big kernel feature layer

3.1 psi_fn definition

psi_fn: R^{d_h} -> R^{R_big} is a fixed architecture mapping trunk output to a high-dimensional feature vector psi_t.

Two canonical choices:

Option A: Random Fourier features
For i = 1..R_big:
psi_t[i] = sqrt(2 / R_big) * cos(w_i^T h_t + b_i)
where w_i in R^{d_h}, b_i in R are fixed after initialization.

Option B: Learned MLP
psi_t = W2_psi sigma_trunk(W1_psi h_t + b1_psi) + b2_psi
with W1_psi, W2_psi, b1_psi, b2_psi of appropriate dimensions.

This specification treats psi_fn as a black box with bounded cost per token:

Cost_psi = O(R_big * d_h)

3.2 Compression matrix and phi_t

Compression matrix:

C_phi in R^{r_phi x R_big}

Compressed feature:

phi_t = C_phi psi_t in R^{r_phi}

Cost_phi = O(r_phi * R_big)

3.3 Kernel approximation

A target kernel k_target: R^{d_h} x R^{d_h} -> R can be approximated by:

k_big(h, h’)  = psi_fn(h)^T psi_fn(h’)
k_phi(h, h’)  = phi(h)^T phi(h’)

Training may use regularizers to align k_big and k_phi with k_target; this is handled in the objective, not in the core specification.
	4.	Value projection and low-rank representation

4.1 Value projection

Parameters:

W_val in R^{d_val x d_h}
b_val in R^{d_val}

Value vector:

v_t = W_val h_t + b_val in R^{d_val}

4.2 Low-rank basis

Parameters:

U_val in R^{d_val x r_v}

Assume v_t is approximately in the column space of U_val:

v_t ≈ U_val r_t for some r_t in R^{r_v}, with residual error e_t:

v_t = U_val r_t + e_t

4.3 Ridge coefficient computation

Define:

G = U_val^T U_val + mu_ridge * I_{r_v}

G in R^{r_v x r_v} is positive definite for mu_ridge >= 0 and full column rank U_val. Factorize once:

G = L_G L_G^T

with L_G lower triangular. Precompute and store L_G.

Per token, compute:

b_t       = U_val^T v_t
Solve L_G y_t = b_t        by forward substitution
Solve L_G^T r_hat_t = y_t  by backward substitution

Result:

r_hat_t = G^{-1} U_val^T v_t

Complexity per token:

Cost_ridge = O(d_val * r_v + r_v^2)

d_val, r_v are fixed; cost is O(1) with respect to sequence length.
	5.	Kernel memory

5.1 State and initialization

Persistent state:

A_t in R^{r_phi x r_v}
s_t in R^{r_phi}

Initialize:

A_0 = 0
s_0 = 0

5.2 Update rule

Given phi_t in R^{r_phi} and r_hat_t in R^{r_v}:

A_t = gamma_mem A_{t-1} + phi_t r_hat_t^T
s_t = gamma_mem s_{t-1} + phi_t

phi_t r_hat_t^T is the outer product with shape r_phi x r_v.

5.3 Query and readout

Given query representation h_q (often equal to h_t):

psi_q = psi_fn(h_q)
phi_q = C_phi psi_q in R^{r_phi}

Compute:

num_t = phi_q^T A_t in R^{r_v}
den_t = phi_q^T s_t in R

Kernel memory output:

y_att_t = U_val ( num_t / (den_t + lambda_mem) ) in R^{d_val}

Division num_t / (den_t + lambda_mem) is elementwise division of each component of num_t by the scalar (den_t + lambda_mem).

5.4 Ratio stability

Define:

den_eff_t = den_t + lambda_mem

Assume that for queries of interest, with high probability:

den_t >= -lambda_mem/2

Then:

den_eff_t >= lambda_mem/2 =: beta_mem > 0

Thus:

|| num_t / den_eff_t || <= (2 / lambda_mem) || num_t ||

and the mapping (num_t, den_t) -> num_t / (den_t + lambda_mem) is Lipschitz with constant bounded by O(1 / lambda_mem^2) when restricted to a bounded set of (num_t, den_t).

5.5 Complexity

Per token:

Cost_update_A = O(r_phi * r_v)
Cost_update_s = O(r_phi)
Cost_query    = O(R_big * d_h + r_phi * R_big + r_phi * r_v + r_phi)

Total kernel memory cost per token is O(1) with respect to sequence length.
	6.	Rational memory

6.1 State and parameters

State:

m_t in R^{d_mem}

Parameters:

F_mem in R^{d_mem x d_mem}
G_mem in R^{d_mem x d_mem_in}
H_mem in R^{d_mem_out x d_mem}

Spectral radius rho(F_mem) must satisfy:

rho(F_mem) < 1 - eta_mem

for some margin eta_mem in (0,1), fixed at configuration. This ensures stability.

6.2 Diagnostic features and input

Diagnostic features:

diag_t = F_diag(h_t, y_att_t) in R^{d_diag}

F_diag is a fixed deterministic mapping with bounded cost; it may include norms, elementwise clips, or low-dimensional summaries.

Parameters:

W_u in R^{d_mem_in x d_h}
B_u in R^{d_mem_in x d_val}
C_u in R^{d_mem_in x d_diag}

Input:

u_t = W_u h_t + B_u y_att_t + C_u diag_t in R^{d_mem_in}

6.3 Update and readout

Update:

m_{t+1} = F_mem m_t + G_mem u_t

Readout:

y_mem_t = H_mem m_t in R^{d_mem_out}

6.4 Boundedness

Given bounded inputs u_t over t and rho(F_mem) < 1, m_t remains bounded in norm. This can be used as a long-range but finite-dimensional memory.

6.5 Complexity

Per token:

Cost_mem = O(d_mem^2 + d_mem * d_mem_in + d_mem_out * d_mem)

All dimensions are fixed; cost is O(1) with respect to sequence length.
	7.	Base feature and representation

7.1 Base feature h_base_t

Define concatenation:

concat_base_t = concat(h_t, y_att_t, y_mem_t, diag_t)

Let dim_concat = d_h + d_val + d_mem_out + d_diag.

Parameters:

W_base_proj in R^{d_base x dim_concat}
b_base_proj in R^{d_base}

Base feature:

h_base_t = W_base_proj concat_base_t + b_base_proj in R^{d_base}

7.2 Representation projection h_rep_t

Parameters:

W_rep in R^{d_rep x d_base}
b_rep in R^{d_rep}

Representation:

h_rep_t = W_rep h_base_t + b_rep in R^{d_rep}
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

Scores:

s_tpl_t[h] = W_tpl[h] · x_tpl_t + b_tpl[h] for h in {1,…,M_tpl}

Template probabilities:

q_tpl_t[h] = exp(s_tpl_t[h]) / sum_{k=1..M_tpl} exp(s_tpl_t[k])

q_tpl_t is used as:
	1.	A feature in the residual head.
	2.	A target in template-classification pretraining if labels are available.
	3.	Residual head and final logits

9.1 Residual head features g_res_t

Define residual input concatenation:

concat_res_t = concat(h_base_t, h_rep_t, q_tpl_t, diag_t)

Let dim_res_in = d_base + d_rep + M_tpl + d_diag.

Parameters for a two-layer MLP realization:

W_res1 in R^{d_res_mid x dim_res_in}
b_res1 in R^{d_res_mid}
W_res2 in R^{d_res x d_res_mid}
b_res2 in R^{d_res}

Residual features:

a_res1_t = sigma_trunk(W_res1 concat_res_t + b_res1)
g_res_t  = W_res2 a_res1_t + b_res2 in R^{d_res}

d_res_mid is an intermediate dimension fixed by configuration.

9.2 Base logits

Parameters:

W_out_base in R^{V_size x d_base}
b_out_base in R^{V_size}

Base logits:

z_base_t = W_out_base h_base_t + b_out_base in R^{V_size}

9.3 Residual logits

Parameters:

W_out_res in R^{V_size x d_res}
b_out_res in R^{V_size}

Residual logits:

r_t = W_out_res g_res_t + b_out_res in R^{V_size}

9.4 Final logits and distribution

Final logits:

z_t = z_base_t + r_t in R^{V_size}

Probability distribution:

p_t[y] = exp(z_t[y]) / sum_{y’=1..V_size} exp(z_t[y’])
	10.	State and complexity summary

10.1 Persistent state

Persistent across tokens:

A_t in R^{r_phi x r_v}
s_t in R^{r_phi}
m_t in R^{d_mem}

Total persistent state size:

r_phi * r_v + r_phi + d_mem

This is independent of t and sequence length.

10.2 Per-token operations

Per token, Sera performs:
	1.	Trunk forward:
O(L_trunk * d_h * d_mid)
	2.	Kernel feature:
O(R_big * d_h + r_phi * R_big)
	3.	Value and low-rank coefficient:
O(d_val * d_h + d_val * r_v + r_v^2)
	4.	Kernel memory update and query:
O(r_phi * r_v + r_phi)
	5.	Rational memory:
O(d_mem^2 + d_mem * d_mem_in + d_mem_out * d_mem)
	6.	Base feature and representation:
O(d_base * dim_concat + d_rep * d_base)
	7.	Template classifier:
O(d_tpl_feat * d_base + M_tpl * d_tpl_feat)
	8.	Residual head and logits:
O(d_res_mid * dim_res_in + d_res * d_res_mid + V_size * d_base + V_size * d_res)

All involved dimensions are configuration constants. Therefore, the total work per token is bounded by a constant with respect to sequence length and history length.
	11.	Training objectives (non-normative but consistent)

11.1 Next-token cross entropy

Given ground-truth next token y_{t+1} in {1,…,V_size}, define:

L_ce = - E_t[ log p_t[y_{t+1}] ]

This is the primary language modeling loss.

11.2 Trunk distillation (optional)

If a teacher model provides hidden states h_teacher_t in R^{d_teacher}, define projection:

h_teacher_proj_t = W_teacher h_teacher_t in R^{d_h}

with W_teacher in R^{d_h x d_teacher}.

Trunk distillation loss:

L_trunk = E_t[ || h_t - h_teacher_proj_t ||^2 ]

11.3 Kernel memory distillation (optional)

If a teacher provides a contextual vector y_att_teacher_t in R^{d_val}:

L_att = E_t[ || y_att_t - y_att_teacher_t ||^2 ]

11.4 Template supervision (optional)

If template labels h_target_t in {1,…,M_tpl} are available or derived:

L_tpl = - E_t[ log q_tpl_t[h_target_t] ]

11.5 Residual regularization (optional)

Residual magnitude regularization:

L_res_reg = E_t[ || r_t ||^2 ]

11.6 Combined loss

A typical combined loss:

L_total = alpha_ce    * L_ce
+ alpha_trunk * L_trunk
+ alpha_att   * L_att
+ alpha_tpl   * L_tpl
+ alpha_res   * L_res_reg

with nonnegative coefficients alpha_· fixed by schedule. Gradients flow through all components: trunk, psi_fn, C_phi, U_val, memory couplings, rational memory couplings, base and residual heads.
	12.	Implementation notes (numerical stability)

12.1 Data types

Inference and training should use at least 32-bit floating point for:

h_t, psi_t, phi_t, v_t, r_hat_t, A_t, s_t, m_t, y_att_t, y_mem_t, h_base_t, h_rep_t,
x_tpl_t, q_tpl_t, g_res_t, z_base_t, r_t, z_t

Parameters may be stored in 16-bit formats with appropriate scaling and cast to 32-bit for computation.

12.2 Ridge system conditioning

To keep G well-conditioned:
	1.	Constrain column norms of U_val by clipping or explicit normalization.
	2.	Set mu_ridge to at least a small positive value if smallest eigenvalue of U_val^T U_val is small.
	3.	Monitor condition number of G during training and adjust mu_ridge as needed.

12.3 Kernel memory scaling

To avoid overflow and underflow in A_t, s_t:
	1.	Restrict phi_t by clipping or normalization:
For example, scale psi_t and C_phi so that ||phi_t|| is bounded by a known constant.
	2.	Choose gamma_mem close to but less than 1 when long context is needed; exact gamma_mem is a configuration trade-off.

12.4 Rational memory stability

Ensure rho(F_mem) < 1 - eta_mem for a nontrivial margin eta_mem and verify numerically. Optionally reparameterize F_mem as:

F_mem = P diag(d) P^{-1}

with |d_i| <= 1 - eta_mem.
	13.	Limitations
	14.	Expressivity depends on depth and width of the trunk and heads. Increasing capacity increases cost but does not change sequence-length dependence.
	15.	Kernel approximation quality depends on R_big, r_phi, and psi_fn, C_phi. If r_phi is too small, the kernel memory cannot emulate high-rank attention.
	16.	Rational memory is linear; nonlinear long-term structure must be captured elsewhere (trunk, kernel memory, heads).
	17.	The specification does not guarantee convergence; training behavior depends on optimizer, schedules, and data.
	18.	Summary

Sera v0.0.1+ defines a deep kernel streaming architecture with:
	1.	Fixed-depth nonlinear trunk over token embeddings.
	2.	Two-stage kernel feature mapping (psi, phi) feeding a low-rank exponential-decay memory.
	3.	Finite-dimensional rational memory capturing linear long-range dynamics.
	4.	Base and residual heads modulated by a template classifier and representation projection.
	5.	Per-token time and state bounded by constants independent of sequence length.

