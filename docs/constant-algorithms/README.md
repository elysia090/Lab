# Constant-Time Algorithm Notes

This directory collects the Cartan–Čech–Robin (CCR) constant-time algorithm
material that used to be scattered across `docs/`. The goal is to offer a
single landing page that points to the mathematical template, the pragmatic
implementation guide, and specialized kernels that reuse the same offline / online
split.

## Document Map

| File | Scope |
| --- | --- |
| [`template.md`](./template.md) | Master reference for the CCR template: algebraic setup, perturbation lemmas, certification, and complexity theorems. |
| [`differentiation.md`](./differentiation.md) | Implementation-oriented walkthrough of the CCR differentiation pipeline with explicit operator recipes. |
| [`rank-k.md`](./rank-k.md) | Constant-time rank-$k$ update kernel that plugs into the CCR template as a reusable block. |

All notes share the same hypotheses: a finite good cover, a sign-stable total
differential, a Robin perturbation with bounded norms, and a convergent
homological contraction. The documents differ only in emphasis—`template.md`
supplies the full theory, `differentiation.md` translates it into runtime steps,
and `rank-k.md` shows how to encapsulate specialized kernels inside the same
structure.

## Related Code

* `src/utilities/constant_time_differentiation.py` — end-to-end runtime
  operators derived from the CCR template.
* `src/utilities/constantdiff.py` — minimal NumPy implementation mirroring the
  documentation's certification and curvature checks.

Refer to these modules when looking for executable counterparts to the
documentation.

## Linking Guidance

When referencing this material from elsewhere in the repository, link to this
directory rather than to individual files. The structure here is stable and is
the canonical entry point for any additional constant-time kernels that may be
added in the future (for example, constant-time attention or constant-memory
market primitives).
