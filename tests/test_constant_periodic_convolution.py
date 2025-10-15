import math
import random

import pytest

from pathlib import Path
import sys

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

