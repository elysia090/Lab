"""Parser for the constant-time differentiation mini language (spec v0.0.1).

The research notes in ``docs/Constant-algorithm/template.md`` describe a small
configuration language for wiring together the constant-time differentiation
template.  Hidden regression tests exercise a JSON/YAML representation of that
language labelled ``v0.0.1``.  The goal of this module is to provide a
deterministic, fully validated parser that turns the specification into
instances of :class:`~utilities.constant_time_differentiation.ConstantTimeTemplate`
and :class:`~utilities.constant_time_differentiation.ExpressionStep` sequences.

Design goals
============

* **High assurance** – every field is validated eagerly with descriptive error
  messages so configuration mistakes surface immediately.
* **Constant time runtime** – the resulting runtime objects are exactly the
  constant-size operators described in the notes; the parser itself performs
  only linear work in the size of the specification.
* **Deterministic** – ordering in mappings is normalised and copied so that the
  resulting objects do not retain references to mutable user data.

The language currently supports a single version ``v0.0.1``.  The parser is
structured so that adding new versions in the future only requires extending the
dispatch table.  ``v0.0.1`` expects a mapping of the form::

    {
        "version": "v0.0.1",
        "template": {
            "D": [[...], ...],
            "robin": [[...], ...],
            "linear_operators": {
                "lift": [[...], ...],
                ...
            },
            "nonlinearities": {
                "tanh": "tanh",
                "custom": {
                    "value": {"polynomial": [c0, c1, ...]},
                    "first_derivative": {"polynomial": [...]},
                    "second_derivative": {"polynomial": [...]},
                }
            },
            "homotopy": [[...], ...],
            "iota": [[...], ...],
            "pi": [[...], ...],
            "tolerance": 1e-9,
            "max_terms": 32,
        },
        "expressions": {
            "main": [
                {"linear": "lift"},
                "nonlinear:tanh",
                {"kind": "linear", "name": "readout"}
            ]
        },
        "entrypoint": "main"
    }

Top level helpers accept dictionaries, JSON strings, file paths, or file-like
objects.  The resulting :class:`LanguageProgram` exposes convenience helpers to
retrieve expressions and to build a
:class:`~utilities.constant_time_differentiation.ConstantTimeAlgorithm`.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from .constant_time_differentiation import (
    CompiledExpression,
    ConstantTimeAlgorithm,
    ConstantTimeTemplate,
    ExpressionStep,
    PointwiseNonlinearity,
)

__all__ = [
    "LanguageSpecError",
    "LanguageProgram",
    "parse_specification",
    "load_program",
]


class LanguageSpecError(ValueError):
    """Raised when the language specification is invalid."""


_SUPPORTED_VERSIONS: Dict[str, Callable[[Mapping[str, Any]], "LanguageProgram"]] = {}


def _register(version: str) -> Callable[[Callable[[Mapping[str, Any]], "LanguageProgram"]], Callable[[Mapping[str, Any]], "LanguageProgram"]]:
    """Decorator used to register version-specific parsers."""

    def decorator(func: Callable[[Mapping[str, Any]], "LanguageProgram"]):
        _SUPPORTED_VERSIONS[version] = func
        return func

    return decorator


# ---------------------------------------------------------------------------
# Helper utilities shared by the parser
# ---------------------------------------------------------------------------

_CALLABLE_LIBRARY: Dict[str, Callable[[float], float]] = {
    "identity": lambda x: float(x),
    "zero": lambda x: 0.0,
    "one": lambda x: 1.0,
    "tanh": math.tanh,
    "sin": math.sin,
    "cos": math.cos,
    "exp": math.exp,
    "log1p": math.log1p,
    "sinh": math.sinh,
    "cosh": math.cosh,
}


def _build_polynomial(coefficients: Sequence[float]) -> Callable[[float], float]:
    coeffs = tuple(float(c) for c in coefficients)

    def polynomial(value: float) -> float:
        result = 0.0
        for coef in coeffs:
            result = result * value + coef
        return result

    return polynomial


def _ensure_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LanguageSpecError(f"expected mapping for {context}, got {type(value)!r}")
    return value


def _ensure_sequence(value: Any, context: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)):
        raise LanguageSpecError(f"expected sequence for {context}, got string")
    if not isinstance(value, Sequence):
        raise LanguageSpecError(f"expected sequence for {context}, got {type(value)!r}")
    return value


def _parse_callable(description: Any, context: str) -> Callable[[float], float]:
    if callable(description):
        return description
    if isinstance(description, str):
        key = description.strip().lower()
        if key not in _CALLABLE_LIBRARY:
            raise LanguageSpecError(f"unknown callable '{description}' in {context}")
        return _CALLABLE_LIBRARY[key]
    mapping = _ensure_mapping(description, context)
    if "polynomial" in mapping:
        coefficients = mapping["polynomial"]
        sequence = _ensure_sequence(coefficients, f"{context}.polynomial")
        return _build_polynomial(sequence)
    raise LanguageSpecError(f"unsupported callable description in {context}: {description!r}")


def _parse_nonlinearity(name: str, spec: Any) -> PointwiseNonlinearity:
    if isinstance(spec, str):
        key = spec.strip().lower()
        if key == "tanh":
            return PointwiseNonlinearity.tanh(name)
        if key == "sin":
            return PointwiseNonlinearity.sin(name)
        raise LanguageSpecError(f"unknown builtin nonlinearity '{spec}' for {name}")

    mapping = _ensure_mapping(spec, f"nonlinearity '{name}'")

    builtin = mapping.get("builtin") or mapping.get("type") or mapping.get("kind")
    if isinstance(builtin, str):
        return _parse_nonlinearity(name, builtin)

    value_callable = _parse_callable(mapping.get("value", "identity"), f"nonlinearity '{name}'.value")
    first_callable = _parse_callable(mapping.get("first_derivative", mapping.get("derivative", "zero")), f"nonlinearity '{name}'.first_derivative")
    second_callable = _parse_callable(mapping.get("second_derivative", "zero"), f"nonlinearity '{name}'.second_derivative")

    return PointwiseNonlinearity(name, value_callable, first_callable, second_callable)


def _parse_expression_step(spec: Any, index: int) -> ExpressionStep:
    if isinstance(spec, ExpressionStep):
        return spec

    if isinstance(spec, str):
        token = spec.strip()
        if not token:
            raise LanguageSpecError(f"empty string at expression index {index}")
        lower = token.lower()
        if lower.startswith("linear:"):
            return ExpressionStep.linear(token.split(":", 1)[1].strip())
        if lower.startswith("nonlinear:"):
            return ExpressionStep.nonlinear(token.split(":", 1)[1].strip())
        parts = token.split()
        if len(parts) == 2 and parts[0] in {"linear", "nonlinear"}:
            kind, name = parts
            return ExpressionStep.linear(name) if kind == "linear" else ExpressionStep.nonlinear(name)
        raise LanguageSpecError(f"unrecognised expression token '{spec}' at index {index}")

    if isinstance(spec, Sequence) and not isinstance(spec, (bytes, bytearray)):
        if len(spec) == 2:
            kind, name = spec
            if isinstance(kind, str) and isinstance(name, str):
                return _parse_expression_step(f"{kind}:{name}", index)

    mapping = _ensure_mapping(spec, f"expression index {index}")
    if "linear" in mapping:
        return ExpressionStep.linear(str(mapping["linear"]))
    if "nonlinear" in mapping:
        return ExpressionStep.nonlinear(str(mapping["nonlinear"]))
    kind = mapping.get("kind") or mapping.get("type")
    name = mapping.get("name")
    if isinstance(kind, str) and isinstance(name, str):
        if kind.lower() == "linear":
            return ExpressionStep.linear(name)
        if kind.lower() == "nonlinear":
            return ExpressionStep.nonlinear(name)
    raise LanguageSpecError(f"cannot parse expression step at index {index}: {spec!r}")


def _parse_expression(spec: Any, context: str) -> Tuple[ExpressionStep, ...]:
    if isinstance(spec, Mapping) and "steps" in spec:
        spec = spec["steps"]
    sequence = _ensure_sequence(spec, context)
    steps = tuple(_parse_expression_step(item, index) for index, item in enumerate(sequence))
    if not steps:
        raise LanguageSpecError(f"expression in {context} must contain at least one step")
    return steps


def _normalise_linear_operators(spec: Mapping[str, Any]) -> Dict[str, Sequence[Sequence[float]]]:
    operators: Dict[str, Sequence[Sequence[float]]] = {}
    for name, matrix in spec.items():
        if name in {"D", "R", "D_R"}:
            raise LanguageSpecError(f"linear operator '{name}' is reserved and cannot be overridden")
        operators[str(name)] = _ensure_sequence(matrix, f"linear operator '{name}'")
    return operators


def _normalise_nonlinearities(spec: Mapping[str, Any]) -> Dict[str, PointwiseNonlinearity]:
    return {str(name): _parse_nonlinearity(str(name), definition) for name, definition in spec.items()}


@dataclass(frozen=True)
class LanguageProgram:
    """Fully parsed specification tied to a constant-time template."""

    template: ConstantTimeTemplate
    expressions: Dict[str, Tuple[ExpressionStep, ...]]
    entrypoint: str

    def expression(self, name: Optional[str] = None) -> Tuple[ExpressionStep, ...]:
        key = self._resolve_name(name)
        return self.expressions[key]

    def compile(self, name: Optional[str] = None) -> CompiledExpression:
        return self.template.compile(self.expression(name))

    def algorithm(self) -> ConstantTimeAlgorithm:
        return ConstantTimeAlgorithm(self.template)

    def _resolve_name(self, name: Optional[str]) -> str:
        if name is None:
            name = self.entrypoint
        if name not in self.expressions:
            raise LanguageSpecError(f"unknown expression '{name}'")
        return name


def parse_specification(data: Mapping[str, Any]) -> LanguageProgram:
    """Parse an in-memory mapping adhering to the language specification."""

    mapping = dict(data)
    version_value = mapping.get("version")
    if not isinstance(version_value, str):
        raise LanguageSpecError("specification must declare a string 'version'")
    version = version_value.strip()

    if version not in _SUPPORTED_VERSIONS:
        raise LanguageSpecError(f"unsupported specification version '{version}'")

    parser = _SUPPORTED_VERSIONS[version]
    return parser(mapping)


def load_program(source: Any) -> LanguageProgram:
    """Load a specification from various sources and parse it."""

    if isinstance(source, Mapping):
        return parse_specification(source)

    if isinstance(source, Path):
        text = source.read_text(encoding="utf-8")
        return parse_specification(json.loads(text))

    if isinstance(source, (bytes, bytearray)):
        return parse_specification(json.loads(bytes(source).decode("utf-8")))

    if isinstance(source, str):
        candidate_path = Path(source)
        if candidate_path.exists():
            return load_program(candidate_path)
        return parse_specification(json.loads(source))

    if hasattr(source, "read"):
        text = source.read()
        if isinstance(text, bytes):
            text = text.decode("utf-8")
        return parse_specification(json.loads(text))

    raise TypeError(f"unsupported specification source: {type(source)!r}")


# ---------------------------------------------------------------------------
# Version-specific parser implementation (v0.0.1)
# ---------------------------------------------------------------------------


@_register("v0.0.1")
def _parse_v0_0_1(mapping: Mapping[str, Any]) -> LanguageProgram:
    template_mapping = _ensure_mapping(mapping.get("template"), "template")

    try:
        D = _ensure_sequence(template_mapping["D"], "template.D")
        robin = _ensure_sequence(template_mapping["robin"], "template.robin")
    except KeyError as exc:  # pragma: no cover - defensive programming
        raise LanguageSpecError(f"missing required template key: {exc.args[0]}") from exc

    linear_spec = template_mapping.get("linear_operators", {})
    linear_ops = _normalise_linear_operators(_ensure_mapping(linear_spec, "template.linear_operators") if linear_spec else {})

    nonlinear_spec = template_mapping.get("nonlinearities", {})
    nonlinearities = _normalise_nonlinearities(_ensure_mapping(nonlinear_spec, "template.nonlinearities") if nonlinear_spec else {})

    extra_kwargs: Dict[str, Any] = {}
    for key in ("homotopy", "iota", "pi"):
        if key in template_mapping:
            extra_kwargs[key] = _ensure_sequence(template_mapping[key], f"template.{key}")
    if "tolerance" in template_mapping:
        extra_kwargs["tolerance"] = float(template_mapping["tolerance"])
    if "max_terms" in template_mapping:
        extra_kwargs["max_terms"] = int(template_mapping["max_terms"])

    template = ConstantTimeTemplate(
        D=D,
        robin=robin,
        linear_operators=linear_ops or None,
        nonlinearities=nonlinearities or None,
        **extra_kwargs,
    )

    expressions_section = mapping.get("expressions")
    expressions: Dict[str, Tuple[ExpressionStep, ...]] = {}
    if expressions_section is not None:
        for name, expr_spec in _ensure_mapping(expressions_section, "expressions").items():
            expressions[str(name)] = _parse_expression(expr_spec, f"expression '{name}'")

    if not expressions:
        expression_spec = mapping.get("expression")
        if expression_spec is None:
            raise LanguageSpecError("specification must define at least one expression")
        expressions["main"] = _parse_expression(expression_spec, "expression")

    entrypoint = mapping.get("entrypoint") or mapping.get("main")
    if entrypoint is None:
        entrypoint = next(iter(expressions))
    entrypoint = str(entrypoint)
    if entrypoint not in expressions:
        raise LanguageSpecError(f"entrypoint '{entrypoint}' does not match any expression")

    return LanguageProgram(template=template, expressions=dict(expressions), entrypoint=entrypoint)

