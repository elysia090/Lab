from math import isclose

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import importlib.util

MODULE_PATH = SRC_ROOT / "utilities" / "constant_time_differentiation.py"
spec = importlib.util.spec_from_file_location("constant_time_differentiation", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = module
spec.loader.exec_module(module)  # type: ignore[union-attr]

ConstantTimeAlgorithm = module.ConstantTimeAlgorithm
ConstantTimeTemplate = module.ConstantTimeTemplate
ExpressionStep = module.ExpressionStep
PointwiseNonlinearity = module.PointwiseNonlinearity


def build_template() -> ConstantTimeTemplate:
    D = (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    )
    R = (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    )

    linear_ops = {
        "lift": (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        "D_R": (
            (0.0, 0.2, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        ),
        "J_edge": (
            (1.0, 0.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, 0.5),
        ),
        "L_X": (
            (0.0, 0.5, 0.0),
            (0.0, 0.0, -0.4),
            (0.3, 0.0, 0.0),
        ),
        "readout": (
            (1.2, -0.7, 0.3),
        ),
    }

    nonlinearities = {
        "tanh": PointwiseNonlinearity.tanh(),
    }

    return ConstantTimeTemplate(
        D=D,
        robin=R,
        linear_operators=linear_ops,
        nonlinearities=nonlinearities,
        homotopy=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        iota=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        pi=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        tolerance=1e-9,
    )


def build_expression() -> tuple[ExpressionStep, ...]:
    return (
        ExpressionStep.linear("lift"),
        ExpressionStep.linear("D_R"),
        ExpressionStep.nonlinear("tanh"),
        ExpressionStep.linear("J_edge"),
        ExpressionStep.linear("L_X"),
        ExpressionStep.linear("readout"),
    )


def vector_add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def vector_scale(v, scalar):
    return tuple(scalar * x for x in v)


def vector_close(a, b, tol=1e-6):
    return all(isclose(x, y, rel_tol=tol, abs_tol=tol) for x, y in zip(a, b))


def finite_difference_jvp(algorithm, expr, x, v, eps=1e-6):
    plus = algorithm.evaluate(expr, vector_add(x, vector_scale(v, eps)))
    minus = algorithm.evaluate(expr, vector_add(x, vector_scale(v, -eps)))
    return tuple((p - m) / (2 * eps) for p, m in zip(plus, minus))


def gradient_via_vjp(algorithm, expr, x):
    return algorithm.vjp(expr, x, (1.0,))


def finite_difference_hvp(algorithm, expr, x, v, eps=1e-6):
    grad_plus = gradient_via_vjp(algorithm, expr, vector_add(x, vector_scale(v, eps)))
    grad_minus = gradient_via_vjp(algorithm, expr, vector_add(x, vector_scale(v, -eps)))
    return tuple((p - m) / (2 * eps) for p, m in zip(grad_plus, grad_minus))


def test_constant_time_template_builds() -> None:
    template = build_template()
    assert template.gamma_bound == 0.0
    assert template.neumann_terms >= 1
    assert template.neumann_tail == 0.0


def test_jvp_matches_finite_difference() -> None:
    template = build_template()
    algorithm = ConstantTimeAlgorithm(template)
    expr = build_expression()

    x = (0.2, -0.3, 0.4)
    v = (0.7, -0.1, 0.5)

    autodiff = algorithm.jacobian_vector_product(expr, x, v)
    numeric = finite_difference_jvp(algorithm, expr, x, v)

    assert vector_close(autodiff, numeric, tol=1e-5)


def test_vjp_is_adjoint_of_jvp() -> None:
    template = build_template()
    algorithm = ConstantTimeAlgorithm(template)
    expr = build_expression()

    x = (0.05, 0.1, -0.2)
    v = (-0.3, 0.4, 0.2)
    cotangent = (1.5,)

    jvp = algorithm.jacobian_vector_product(expr, x, v)[0]
    vjp = algorithm.vjp(expr, x, cotangent)

    left = jvp * cotangent[0]
    right = sum(v_i * g_i for v_i, g_i in zip(v, vjp))

    assert isclose(left, right, rel_tol=1e-6, abs_tol=1e-6)


def test_hvp_matches_finite_difference() -> None:
    template = build_template()
    algorithm = ConstantTimeAlgorithm(template)
    expr = build_expression()

    x = (0.12, -0.34, 0.56)
    v = (0.6, -0.2, 0.3)
    cotangent = (1.0,)

    autodiff_hvp = algorithm.hessian_vector_product(expr, x, v, cotangent)
    numeric_hvp = finite_difference_hvp(algorithm, expr, x, v)

    assert vector_close(autodiff_hvp, numeric_hvp, tol=1e-5)

