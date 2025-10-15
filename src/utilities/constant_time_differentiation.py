"""Constant-time differentiation templates based on the project notes.

The documentation in ``docs/constant-algorithms/template.md`` describes a
constant-size runtime kernel for differentiation over the total complex with a
Robin perturbation.  The implementation below mirrors that architecture using
plain Python data structures so that it is lightweight and dependency free.

Linear operators are represented by tuples of tuples (dense matrices) and act
on one-dimensional vectors encoded as tuples of floats.  Pointwise
nonlinearities store their value, first derivative, and second derivative; this
is enough to implement forward- and reverse-mode automatic differentiation as
well as Hessian–vector products using the constant-time templates highlighted
in the notes.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin, tanh
from typing import Callable, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

Vector = Tuple[float, ...]
Matrix = Tuple[Tuple[float, ...], ...]


# ---------------------------------------------------------------------------
# Small linear-algebra helpers (pure Python, no external dependencies)
# ---------------------------------------------------------------------------
def _to_vector(value: Sequence[float] | Vector) -> Vector:
    if isinstance(value, tuple):
        return value
    return tuple(float(v) for v in value)


def _vector_dim(vector: Vector) -> int:
    return len(vector)


def _to_matrix(value: Sequence[Sequence[float]] | Matrix) -> Matrix:
    if isinstance(value, tuple) and value and isinstance(value[0], tuple):
        return value  # already normalised

    rows: List[Tuple[float, ...]] = []
    expected: Optional[int] = None
    for row in value:
        current = tuple(float(v) for v in row)
        if expected is None:
            expected = len(current)
        elif len(current) != expected:
            raise ValueError("matrix rows must have the same length")
        rows.append(current)
    return tuple(rows)


def _matrix_shape(matrix: Matrix) -> Tuple[int, int]:
    rows = len(matrix)
    cols = len(matrix[0]) if rows else 0
    return rows, cols


def _mat_vec_mul(matrix: Matrix, vector: Vector) -> Vector:
    rows, cols = _matrix_shape(matrix)
    if cols != _vector_dim(vector):
        raise ValueError("matrix/vector dimension mismatch")
    result = []
    for row in matrix:
        acc = 0.0
        for value, coef in zip(vector, row):
            acc += coef * value
        result.append(acc)
    return tuple(result)


def _mat_mul(a: Matrix, b: Matrix) -> Matrix:
    a_rows, a_cols = _matrix_shape(a)
    b_rows, b_cols = _matrix_shape(b)
    if a_cols != b_rows:
        raise ValueError("matrix multiplication dimension mismatch")
    result_rows: List[Tuple[float, ...]] = []
    for i in range(a_rows):
        row: List[float] = []
        for j in range(b_cols):
            acc = 0.0
            for k in range(a_cols):
                acc += a[i][k] * b[k][j]
            row.append(acc)
        result_rows.append(tuple(row))
    return tuple(result_rows)


def _mat_add(a: Matrix, b: Matrix) -> Matrix:
    return tuple(
        tuple(x + y for x, y in zip(row_a, row_b))
        for row_a, row_b in zip(a, b)
    )


def _mat_sub(a: Matrix, b: Matrix) -> Matrix:
    return tuple(
        tuple(x - y for x, y in zip(row_a, row_b))
        for row_a, row_b in zip(a, b)
    )


def _mat_scale(matrix: Matrix, scalar: float) -> Matrix:
    return tuple(tuple(scalar * v for v in row) for row in matrix)


def _mat_transpose(matrix: Matrix) -> Matrix:
    rows, cols = _matrix_shape(matrix)
    return tuple(
        tuple(matrix[i][j] for i in range(rows))
        for j in range(cols)
    )


def _mat_identity(dim: int) -> Matrix:
    return tuple(tuple(1.0 if i == j else 0.0 for j in range(dim)) for i in range(dim))


def _mat_zero(dim: int) -> Matrix:
    return tuple(tuple(0.0 for _ in range(dim)) for _ in range(dim))


def _mat_norm_inf(matrix: Matrix) -> float:
    return max(sum(abs(entry) for entry in row) for row in matrix)


def _mat_max_abs(matrix: Matrix) -> float:
    return max(abs(entry) for row in matrix for entry in row)


def _vector_pointwise_mul(a: Vector, b: Vector) -> Vector:
    return tuple(x * y for x, y in zip(a, b))


def _vector_pointwise_add(a: Vector, b: Vector) -> Vector:
    return tuple(x + y for x, y in zip(a, b))


def _vector_scale(vector: Vector, scalar: float) -> Vector:
    return tuple(scalar * v for v in vector)


def _vector_zero(dim: int) -> Vector:
    return tuple(0.0 for _ in range(dim))


def _vector_dot(a: Vector, b: Vector) -> float:
    return sum(x * y for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# Core data classes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LinearOperator:
    """Dense constant-size linear operator."""

    name: str
    matrix: Matrix

    @staticmethod
    def from_sequence(name: str, matrix: Sequence[Sequence[float]]) -> "LinearOperator":
        return LinearOperator(name, _to_matrix(matrix))

    @property
    def adjoint(self) -> Matrix:
        return _mat_transpose(self.matrix)

    def apply(self, vector: Vector | Sequence[float]) -> Vector:
        return _mat_vec_mul(self.matrix, _to_vector(vector))


PointwiseFn = Callable[[float], float]


@dataclass(frozen=True)
class PointwiseNonlinearity:
    """Pointwise nonlinearity with derivative information."""

    name: str
    value: PointwiseFn
    first_derivative: PointwiseFn
    second_derivative: PointwiseFn

    @staticmethod
    def tanh(name: str = "tanh") -> "PointwiseNonlinearity":
        def first(x: float) -> float:
            y = tanh(x)
            return 1.0 - y * y

        def second(x: float) -> float:
            y = tanh(x)
            return -2.0 * y * (1.0 - y * y)

        return PointwiseNonlinearity(name, tanh, first, second)

    @staticmethod
    def sin(name: str = "sin") -> "PointwiseNonlinearity":
        return PointwiseNonlinearity(name, sin, cos, lambda x: -sin(x))

    def apply(self, vector: Vector) -> Vector:
        return tuple(self.value(v) for v in vector)

    def apply_first_derivative(self, vector: Vector) -> Vector:
        return tuple(self.first_derivative(v) for v in vector)

    def apply_second_derivative(self, vector: Vector) -> Vector:
        return tuple(self.second_derivative(v) for v in vector)


@dataclass(frozen=True)
class ExpressionStep:
    kind: Literal["linear", "nonlinear"]
    name: str

    @staticmethod
    def linear(name: str) -> "ExpressionStep":
        return ExpressionStep("linear", name)

    @staticmethod
    def nonlinear(name: str) -> "ExpressionStep":
        return ExpressionStep("nonlinear", name)


@dataclass(frozen=True)
class _CompiledStep:
    """Internal helper storing resolved operators for an expression step."""

    kind: Literal["linear", "nonlinear"]
    name: str
    operator: Union[LinearOperator, PointwiseNonlinearity]


@dataclass(frozen=True)
class CompiledExpression:
    """Expression with the operators resolved at compile time.

    Compiled expressions avoid repeated dictionary lookups during runtime and
    guarantee that all referenced operators exist.  They can be reused across
    multiple evaluations or derivative computations with the same template.
    """

    steps: Tuple[_CompiledStep, ...]

    def __iter__(self) -> Iterable[_CompiledStep]:  # pragma: no cover - trivial
        return iter(self.steps)


class ConstantTimeTemplate:
    """Offline data backing the constant-time algorithm."""

    def __init__(
        self,
        *,
        D: Sequence[Sequence[float]],
        robin: Sequence[Sequence[float]],
        linear_operators: Optional[Dict[str, Sequence[Sequence[float]]]] = None,
        nonlinearities: Optional[Dict[str, PointwiseNonlinearity]] = None,
        homotopy: Optional[Sequence[Sequence[float]]] = None,
        iota: Optional[Sequence[Sequence[float]]] = None,
        pi: Optional[Sequence[Sequence[float]]] = None,
        tolerance: float = 1e-9,
        max_terms: int = 32,
    ) -> None:
        self.tolerance = float(tolerance)
        self.max_terms = int(max_terms)

        self.D = LinearOperator.from_sequence("D", D)
        self.R = LinearOperator.from_sequence("R", robin)
        self.D_R = LinearOperator("D_R", _mat_add(self.D.matrix, self.R.matrix))

        self.linear_operators: Dict[str, LinearOperator] = {
            "D": self.D,
            "R": self.R,
            "D_R": self.D_R,
        }
        if linear_operators:
            for name, matrix in linear_operators.items():
                self.linear_operators[name] = LinearOperator.from_sequence(name, matrix)

        self.nonlinearities = nonlinearities or {}

        size = _matrix_shape(self.D.matrix)[0]
        self.h = _to_matrix(homotopy if homotopy is not None else _mat_zero(size))
        self.iota = _to_matrix(iota if iota is not None else _mat_identity(size))
        self.pi = _to_matrix(pi if pi is not None else _mat_identity(size))

        self._validate_structure()
        self.h_R, self.neumann_terms, self.neumann_tail = self._compute_perturbed_homotopy()

    # ------------------------------------------------------------------
    def _validate_structure(self) -> None:
        curvature = _mat_mul(self.D_R.matrix, self.D_R.matrix)
        if _mat_max_abs(curvature) > self.tolerance:
            raise ValueError("D_R must square to zero within tolerance")

        ident = _mat_identity(_matrix_shape(self.pi)[0])
        if _mat_max_abs(_mat_sub(_mat_mul(self.pi, self.iota), ident)) > self.tolerance:
            raise ValueError("pi @ iota must equal identity")

        lhs = _mat_add(
            _mat_mul(self.iota, self.pi),
            _mat_add(_mat_mul(self.D.matrix, self.h), _mat_mul(self.h, self.D.matrix)),
        )
        if _mat_max_abs(_mat_sub(lhs, ident)) > max(self.tolerance, 1e-6):
            raise ValueError("iota @ pi must equal I - D h - h D")

    # ------------------------------------------------------------------
    @property
    def gamma_bound(self) -> float:
        return _mat_norm_inf(_mat_mul(self.R.matrix, self.h))

    def _compute_perturbed_homotopy(self) -> Tuple[Matrix, int, float]:
        size = _matrix_shape(self.h)[0]
        A = _mat_scale(_mat_mul(self.R.matrix, self.h), -1.0)
        identity = _mat_identity(size)
        series = identity
        term = identity
        norm_A = _mat_norm_inf(A)

        for m in range(1, self.max_terms + 1):
            term = _mat_mul(term, A)
            series = _mat_add(series, term)
            if _mat_norm_inf(term) <= self.tolerance:
                tail = (norm_A ** (m + 1) / (1.0 - norm_A)) if norm_A < 1.0 else float("inf")
                return _mat_mul(self.h, series), m + 1, tail

        tail = (norm_A ** self.max_terms / (1.0 - norm_A)) if norm_A < 1.0 else float("inf")
        return _mat_mul(self.h, series), self.max_terms, tail

    # ------------------------------------------------------------------
    def linear(self, name: str) -> LinearOperator:
        if name not in self.linear_operators:
            raise KeyError(f"unknown linear operator '{name}'")
        return self.linear_operators[name]

    def nonlinearity(self, name: str) -> PointwiseNonlinearity:
        if name not in self.nonlinearities:
            raise KeyError(f"unknown nonlinearity '{name}'")
        return self.nonlinearities[name]

    def compile(self, expression: Sequence[ExpressionStep]) -> CompiledExpression:
        """Resolve all operators referenced by *expression*.

        The returned :class:`CompiledExpression` can be fed directly to the
        runtime evaluator.  Compilation performs structural validation once so
        that subsequent runs cannot fail due to missing operators.
        """

        steps: List[_CompiledStep] = []
        for step in expression:
            if step.kind == "linear":
                operator: Union[LinearOperator, PointwiseNonlinearity]
                operator = self.linear(step.name)
            elif step.kind == "nonlinear":
                operator = self.nonlinearity(step.name)
            else:  # pragma: no cover - defensive programming
                raise ValueError(f"unknown expression step kind '{step.kind}'")
            steps.append(_CompiledStep(step.kind, step.name, operator))
        return CompiledExpression(tuple(steps))

    def ensure_compiled(
        self, expression: Union[Sequence[ExpressionStep], CompiledExpression]
    ) -> CompiledExpression:
        if isinstance(expression, CompiledExpression):
            return expression
        return self.compile(expression)


class ConstantTimeAlgorithm:
    """Runtime evaluator built on top of :class:`ConstantTimeTemplate`."""

    def __init__(self, template: ConstantTimeTemplate) -> None:
        self.template = template

    # ------------------------------------------------------------------
    def _iter_steps(
        self, expression: Union[Sequence[ExpressionStep], CompiledExpression]
    ) -> Iterable[_CompiledStep]:
        compiled = self.template.ensure_compiled(expression)
        return compiled

    def evaluate(
        self,
        expression: Union[Sequence[ExpressionStep], CompiledExpression],
        x: Sequence[float] | Vector,
    ) -> Vector:
        state = _to_vector(x)
        for step in self._iter_steps(expression):
            if step.kind == "linear":
                state = step.operator.apply(state)  # type: ignore[union-attr]
            else:
                nonlin = step.operator  # type: ignore[assignment]
                state = nonlin.apply(state)
        return state

    # ------------------------------------------------------------------
    def jacobian_vector_product(
        self,
        expression: Union[Sequence[ExpressionStep], CompiledExpression],
        x: Sequence[float] | Vector,
        tangent: Sequence[float] | Vector,
    ) -> Vector:
        primal = _to_vector(x)
        tangent_vec = _to_vector(tangent)

        for step in self._iter_steps(expression):
            if step.kind == "linear":
                op = step.operator  # type: ignore[assignment]
                primal = op.apply(primal)
                tangent_vec = op.apply(tangent_vec)
            else:
                nonlin = step.operator  # type: ignore[assignment]
                deriv = nonlin.apply_first_derivative(primal)
                primal = nonlin.apply(primal)
                tangent_vec = _vector_pointwise_mul(deriv, tangent_vec)
        return tangent_vec

    # ------------------------------------------------------------------
    def vector_jacobian_product(
        self,
        expression: Union[Sequence[ExpressionStep], CompiledExpression],
        x: Sequence[float] | Vector,
        cotangent: Sequence[float] | Vector,
    ) -> Vector:
        primal = _to_vector(x)
        primals: List[Vector] = []

        steps = tuple(self._iter_steps(expression))
        for step in steps:
            primals.append(primal)
            if step.kind == "linear":
                op = step.operator  # type: ignore[assignment]
                primal = op.apply(primal)
            else:
                nonlin = step.operator  # type: ignore[assignment]
                primal = nonlin.apply(primal)

        adjoint = _to_vector(cotangent)

        for step, prev in zip(reversed(steps), reversed(primals)):
            if step.kind == "linear":
                op = step.operator  # type: ignore[assignment]
                adjoint = _mat_vec_mul(op.adjoint, adjoint)
            else:
                nonlin = step.operator  # type: ignore[assignment]
                deriv = nonlin.apply_first_derivative(prev)
                adjoint = _vector_pointwise_mul(deriv, adjoint)

        return adjoint

    vjp = vector_jacobian_product

    # ------------------------------------------------------------------
    def hessian_vector_product(
        self,
        expression: Union[Sequence[ExpressionStep], CompiledExpression],
        x: Sequence[float] | Vector,
        tangent: Sequence[float] | Vector,
        cotangent: Sequence[float] | Vector,
    ) -> Vector:
        primal = _to_vector(x)
        tangent_vec = _to_vector(tangent)

        primals: List[Vector] = []
        tangents: List[Vector] = []

        steps = tuple(self._iter_steps(expression))
        for step in steps:
            primals.append(primal)
            tangents.append(tangent_vec)

            if step.kind == "linear":
                op = step.operator  # type: ignore[assignment]
                primal = op.apply(primal)
                tangent_vec = op.apply(tangent_vec)
            else:
                nonlin = step.operator  # type: ignore[assignment]
                deriv = nonlin.apply_first_derivative(primal)
                primal = nonlin.apply(primal)
                tangent_vec = _vector_pointwise_mul(deriv, tangent_vec)

        adjoint = _to_vector(cotangent)
        hvp = _vector_zero(len(primal))

        for step, prev_primal, prev_tangent in zip(reversed(steps), reversed(primals), reversed(tangents)):
            if step.kind == "linear":
                op = step.operator  # type: ignore[assignment]
                adjoint = _mat_vec_mul(op.adjoint, adjoint)
                hvp = _mat_vec_mul(op.adjoint, hvp)
            else:
                nonlin = step.operator  # type: ignore[assignment]
                deriv = nonlin.apply_first_derivative(prev_primal)
                second = nonlin.apply_second_derivative(prev_primal)
                hvp = _vector_pointwise_add(
                    _vector_pointwise_mul(deriv, hvp),
                    tuple(second[i] * prev_tangent[i] * adjoint[i] for i in range(len(adjoint))),
                )
                adjoint = _vector_pointwise_mul(deriv, adjoint)

        return hvp

    # ------------------------------------------------------------------
    def runtime_summary(self) -> Dict[str, float]:
        return {
            "gamma_bound": self.template.gamma_bound,
            "neumann_terms": float(self.template.neumann_terms),
            "neumann_tail": float(self.template.neumann_tail),
        }


__all__ = [
    "ConstantTimeAlgorithm",
    "ConstantTimeTemplate",
    "CompiledExpression",
    "ExpressionStep",
    "LinearOperator",
    "PointwiseNonlinearity",
]

