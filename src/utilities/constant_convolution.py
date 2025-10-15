"""Utility helpers for constant-time periodic convolutions.

This module implements a small helper that evaluates a circular (periodic)
convolution in constant time whenever the bandwidth of the convolution kernel
is fixed.  A kernel with bandwidth :math:`K` only touches values within
``[-K, K]`` around the current position.  When ``K`` is considered a constant,
each evaluation of the convolution requires :math:`O(1)` arithmetic
operations.  The implementation is intentionally lightweight so that it can be
used in analytical experiments where the number of learnable parameters must
stay fixed regardless of the size of the input signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum
from typing import Iterable, List, Sequence


@dataclass(frozen=True)
class PeriodicKernel:
    """A fixed-bandwidth kernel for periodic convolutions.

    Parameters
    ----------
    weights:
        Sequence of length ``2 * bandwidth + 1`` describing the kernel
        coefficients.  The entry at index ``bandwidth`` corresponds to the
        zero-offset, i.e. the coefficient multiplied with the current sample.
    bandwidth:
        The bandwidth :math:`K` of the kernel.  Only samples within ``K``
        positions are consulted when the kernel is applied, making each
        evaluation :math:`O(K)`.  With ``K`` fixed, the evaluation time is
        constant.
    """

    weights: Sequence[float]
    bandwidth: int

    def __post_init__(self) -> None:  # pragma: no cover - dataclass hook
        expected_length = 2 * self.bandwidth + 1
        if len(self.weights) != expected_length:
            raise ValueError(
                "Kernel weight count must match 2 * bandwidth + 1 "
                f"(expected {expected_length}, got {len(self.weights)})."
            )

    def iter_offsets(self) -> Iterable[tuple[int, float]]:
        """Iterate over ``(offset, weight)`` pairs used by the kernel."""

        start = -self.bandwidth
        for relative_offset, weight in enumerate(self.weights):
            yield start + relative_offset, float(weight)


class ConstantTimePeriodicConvolution:
    """Periodic convolution with a constant number of parameters.

    The class keeps a reference to a periodic signal and a :class:`PeriodicKernel`
    instance.  Each call to :meth:`convolve_at` evaluates the circular
    convolution at a single index.  Because the kernel bandwidth is fixed, the
    operation uses a constant amount of work, independent of the length of the
    signal.
    """

    def __init__(self, signal: Sequence[float], kernel: PeriodicKernel):
        if not isinstance(kernel, PeriodicKernel):
            raise TypeError("kernel must be an instance of PeriodicKernel")

        self._kernel = kernel
        self._signal = tuple(float(value) for value in signal)
        if len(self._signal) == 0:
            raise ValueError("signal must not be empty")

        self._length = len(self._signal)

    @property
    def length(self) -> int:
        """Length of the periodic signal."""

        return self._length

    @property
    def kernel(self) -> PeriodicKernel:
        """Return the kernel used by the convolution."""

        return self._kernel

    def update_signal(self, new_signal: Sequence[float]) -> None:
        """Replace the underlying periodic signal.

        The kernel (and therefore the number of parameters) stays fixed.  This
        makes it possible to reuse the same convolution setup with different
        periodic signals without having to reallocate intermediate buffers.
        """

        new_signal = tuple(float(value) for value in new_signal)
        if len(new_signal) != self._length:
            raise ValueError(
                "new signal must have the same length as the original signal"
            )
        self._signal = new_signal

    def convolve_at(self, index: int) -> float:
        r"""Evaluate the periodic convolution at ``index``.

        ``index`` can be any integer (positive or negative).  The result is
        equivalent to

        .. math::

            \sum_{k=-K}^{K} w_k x_{(i - k) \bmod N},

        where ``K`` is the kernel bandwidth and ``N`` is the length of the
        periodic signal.
        """

        wrapped_index = index % self._length
        products: List[float] = []
        for offset, weight in self._kernel.iter_offsets():
            products.append(
                weight * self._signal[(wrapped_index - offset) % self._length]
            )
        return fsum(products)

    def full_convolution(self) -> List[float]:
        """Return the convolution evaluated at every index."""

        return [self.convolve_at(i) for i in range(self._length)]


__all__ = [
    "PeriodicKernel",
    "ConstantTimePeriodicConvolution",
]

