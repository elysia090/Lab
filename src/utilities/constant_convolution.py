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

from dataclasses import dataclass, field
from math import fsum
from typing import Iterable, List, Sequence, Tuple


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

    weights: Tuple[float, ...]
    bandwidth: int
    _offset_weight_pairs: Tuple[tuple[int, float], ...] = field(
        init=False, repr=False
    )
    _offsets: Tuple[int, ...] = field(init=False, repr=False)
    _reversed_weights: Tuple[float, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:  # pragma: no cover - dataclass hook
        expected_length = 2 * self.bandwidth + 1
        weights = tuple(float(value) for value in self.weights)
        object.__setattr__(self, "weights", weights)

        if len(weights) != expected_length:
            raise ValueError(
                "Kernel weight count must match 2 * bandwidth + 1 "
                f"(expected {expected_length}, got {len(weights)})."
            )

        offsets = tuple(index - self.bandwidth for index in range(expected_length))
        object.__setattr__(
            self,
            "_offset_weight_pairs",
            tuple(zip(offsets, weights, strict=True)),
        )
        object.__setattr__(self, "_offsets", offsets)
        object.__setattr__(self, "_reversed_weights", tuple(reversed(weights)))

    @property
    def length(self) -> int:
        """Number of weights carried by the kernel."""

        return len(self.weights)

    @property
    def offsets(self) -> Tuple[int, ...]:
        """Return the offsets covered by the kernel footprint."""

        return self._offsets

    def iter_offsets(self) -> Iterable[tuple[int, float]]:
        """Iterate over ``(offset, weight)`` pairs used by the kernel."""

        yield from self._offset_weight_pairs

    @property
    def reversed_weights(self) -> Tuple[float, ...]:
        """Return the kernel weights reversed around the zero offset."""

        return self._reversed_weights


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
        self._signal = self._coerce_vector(signal, "signal")
        if len(self._signal) == 0:
            raise ValueError("signal must not be empty")

        self._length = len(self._signal)
        self._offsets = kernel.offsets

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

        self._signal = self._coerce_vector(new_signal, "new signal", require_length=True)

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
        return self._weighted_sum(self._signal, self._kernel.weights, wrapped_index)

    def full_convolution(self) -> List[float]:
        """Return the convolution evaluated at every index."""

        return [self.convolve_at(i) for i in range(self._length)]

    def jvp_full(self, tangent: Sequence[float]) -> List[float]:
        """Forward-mode Jacobian–vector product with respect to the signal."""

        tangent_vector = self._coerce_vector(tangent, "tangent", require_length=True)
        return [
            self._weighted_sum(tangent_vector, self._kernel.weights, i)
            for i in range(self._length)
        ]

    def vjp_full(self, cotangent: Sequence[float]) -> List[float]:
        """Reverse-mode vector–Jacobian product with respect to the signal."""

        cotangent_vector = self._coerce_vector(
            cotangent, "cotangent", require_length=True
        )
        reversed_weights = self._kernel.reversed_weights
        return [
            self._weighted_sum(cotangent_vector, reversed_weights, i)
            for i in range(self._length)
        ]

    def grad_weights(self, cotangent: Sequence[float]) -> List[float]:
        """Gradient of a scalar cotangent with respect to the kernel weights."""

        cotangent_vector = self._coerce_vector(
            cotangent, "cotangent", require_length=True
        )
        length = self._length
        gradient: List[float] = []
        for offset in self._offsets:
            products: List[float] = []
            for index in range(length):
                products.append(
                    cotangent_vector[index]
                    * self._signal[(index - offset) % length]
                )
            gradient.append(fsum(products))
        return gradient

    def _coerce_vector(
        self,
        values: Sequence[float],
        label: str,
        *,
        require_length: bool = False,
    ) -> Tuple[float, ...]:
        vector = tuple(float(value) for value in values)
        if require_length and len(vector) != self._length:
            raise ValueError(
                f"{label} must have length {self._length} (got {len(vector)})"
            )
        return vector

    def _weighted_sum(
        self,
        values: Sequence[float],
        weights: Sequence[float],
        index: int,
    ) -> float:
        length = self._length
        products = (
            weight * values[(index - offset) % length]
            for offset, weight in zip(self._offsets, weights, strict=True)
        )
        return fsum(products)


__all__ = [
    "PeriodicKernel",
    "ConstantTimePeriodicConvolution",
]

