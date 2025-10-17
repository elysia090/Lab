# Constant-Time Algorithm Notes

This directory consolidates the constant-time algorithm material that was
previously scattered across `docs/`.  The goal is to make it easy to locate the
theoretical template and the pragmatic differentiation walkthrough without
chasing ad-hoc filenames.

## Contents

| File | Purpose |
| --- | --- |
| [`template.md`](./template.md) | The master reference describing the Cartan–Čech–Robin transfer template, its algebraic invariants, and the offline/online split that delivers constant-time evaluation. |
| [`cartan-glue.md`](./cartan-glue.md) | Formalizes the Cartan–Čech gluing calculus and the O(1) patching operators used in the constant-time template. |
| [`differentiation.md`](./differentiation.md) | A practical restatement of the template focused on differentiation workflows and implementation notes for constant-size kernels. |
| [`constant-fft.md`](./constant-fft.md) | Derives the fixed-bandwidth convolution scheme with constant per-index cost and its forward/reverse derivatives. |

Both notes share the same structural assumptions (finite good cover, sign-stable
total differential, Robin perturbation with bounded norms, and a convergent
homological contraction).  `template.md` carries the full formal development;
`differentiation.md` distills the pieces that matter when implementing the
runtime operators in `src/utilities/constant_time_differentiation.py`.

For navigation from the wider documentation tree, link to this directory rather
than to the individual files directly.  Doing so keeps future additions (for
example, constant-time attention or constant-memory market primitives) grouped
under a single entry point.
