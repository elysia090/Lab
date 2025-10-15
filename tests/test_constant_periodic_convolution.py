import math
import random

import pytest

from pathlib import Path
import sys

from math import fsum

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utilities.constant_convolution import (
    ConstantTimePeriodicConvolution,
    PeriodicKernel,
)


def naive_periodic_convolution(signal, kernel):
    n = len(signal)
    k = (len(kernel) - 1) // 2
    result = []
    for i in range(n):
        acc = 0.0
        for offset in range(-k, k + 1):
            acc += kernel[offset + k] * signal[(i - offset) % n]
        result.append(acc)
    return result


def naive_vjp(signal, kernel):
    """Reverse-mode product using the reversed kernel."""

    reversed_kernel = list(reversed(kernel))
    return naive_periodic_convolution(signal, reversed_kernel)


def naive_grad_w(signal, grad_output, kernel):
    """Gradient with respect to kernel weights using a naive sweep."""

    n = len(signal)
    k = (len(kernel) - 1) // 2
    gradients = []
    for offset in range(-k, k + 1):
        acc = 0.0
        for i in range(n):
            acc += grad_output[i] * signal[(i - offset) % n]
        gradients.append(acc)
    return gradients


def test_convolution_matches_naive():
    random.seed(1234)
    length = 17
    bandwidth = 2
    signal = [random.uniform(-1.0, 1.0) for _ in range(length)]
    kernel_weights = [random.uniform(-1.0, 1.0) for _ in range(2 * bandwidth + 1)]

    kernel = PeriodicKernel(weights=kernel_weights, bandwidth=bandwidth)
    conv = ConstantTimePeriodicConvolution(signal=signal, kernel=kernel)

    expected = naive_periodic_convolution(signal, kernel_weights)
    actual = conv.full_convolution()

    for lhs, rhs in zip(actual, expected):
        assert math.isclose(lhs, rhs, rel_tol=1e-12, abs_tol=1e-12)


def test_convolve_at_handles_wrapped_indices():
    signal = [float(value) for value in range(5)]
    kernel = PeriodicKernel(weights=[0.25, 0.5, 0.25], bandwidth=1)
    conv = ConstantTimePeriodicConvolution(signal, kernel)

    idx = -3
    result = conv.convolve_at(idx)
    expected = kernel.weights[0] * signal[(idx + 1) % 5]
    expected += kernel.weights[1] * signal[idx % 5]
    expected += kernel.weights[2] * signal[(idx - 1) % 5]

    assert math.isclose(result, expected, rel_tol=1e-12, abs_tol=1e-12)


def test_update_signal_requires_same_length():
    signal = [0.0, 0.0, 0.0, 0.0]
    kernel = PeriodicKernel(weights=[1.0, 2.0, 1.0], bandwidth=1)
    conv = ConstantTimePeriodicConvolution(signal, kernel)

    with pytest.raises(ValueError):
        conv.update_signal([0.0, 0.0, 0.0, 0.0, 0.0])


def test_kernel_length_validation():
    with pytest.raises(ValueError):
        PeriodicKernel(weights=[1.0, 0.5], bandwidth=1)


def test_jvp_matches_naive():
    random.seed(2024)
    length = 13
    bandwidth = 3
    signal = [random.uniform(-2.0, 2.0) for _ in range(length)]
    tangent = [random.uniform(-1.0, 1.0) for _ in range(length)]
    weights = [random.uniform(-0.5, 0.5) for _ in range(2 * bandwidth + 1)]

    kernel = PeriodicKernel(weights=weights, bandwidth=bandwidth)
    conv = ConstantTimePeriodicConvolution(signal, kernel)

    expected = naive_periodic_convolution(tangent, weights)
    actual = conv.jvp_full(tangent)

    for lhs, rhs in zip(actual, expected):
        assert math.isclose(lhs, rhs, rel_tol=1e-12, abs_tol=1e-12)


def test_vjp_matches_naive():
    random.seed(4321)
    length = 11
    bandwidth = 2
    signal = [random.uniform(-3.0, 3.0) for _ in range(length)]
    grad_output = [random.uniform(-1.5, 1.5) for _ in range(length)]
    weights = [random.uniform(-2.0, 2.0) for _ in range(2 * bandwidth + 1)]

    kernel = PeriodicKernel(weights=weights, bandwidth=bandwidth)
    conv = ConstantTimePeriodicConvolution(signal, kernel)

    expected = naive_vjp(grad_output, weights)
    actual = conv.vjp_full(grad_output)

    for lhs, rhs in zip(actual, expected):
        assert math.isclose(lhs, rhs, rel_tol=1e-12, abs_tol=1e-12)


def test_grad_weights_matches_naive():
    random.seed(99)
    length = 9
    bandwidth = 1
    signal = [random.uniform(-2.0, 2.0) for _ in range(length)]
    grad_output = [random.uniform(-1.0, 1.0) for _ in range(length)]
    weights = [random.uniform(-0.75, 0.75) for _ in range(2 * bandwidth + 1)]

    kernel = PeriodicKernel(weights=weights, bandwidth=bandwidth)
    conv = ConstantTimePeriodicConvolution(signal, kernel)

    expected = naive_grad_w(signal, grad_output, weights)
    actual = conv.grad_weights(grad_output)

    for lhs, rhs in zip(actual, expected):
        assert math.isclose(lhs, rhs, rel_tol=1e-12, abs_tol=1e-12)


def test_adjoint_consistency():
    random.seed(7)
    length = 8
    bandwidth = 2
    signal = [random.uniform(-1.0, 1.0) for _ in range(length)]
    tangent = [random.uniform(-0.5, 0.5) for _ in range(length)]
    cotangent = [random.uniform(-0.5, 0.5) for _ in range(length)]
    weights = [random.uniform(-1.0, 1.0) for _ in range(2 * bandwidth + 1)]

    kernel = PeriodicKernel(weights=weights, bandwidth=bandwidth)
    conv = ConstantTimePeriodicConvolution(signal, kernel)

    jvp = conv.jvp_full(tangent)
    vjp = conv.vjp_full(cotangent)

    lhs = fsum(lhs_val * rhs_val for lhs_val, rhs_val in zip(jvp, cotangent))
    rhs = fsum(lhs_val * rhs_val for lhs_val, rhs_val in zip(tangent, vjp))

    assert math.isclose(lhs, rhs, rel_tol=1e-12, abs_tol=1e-12)

