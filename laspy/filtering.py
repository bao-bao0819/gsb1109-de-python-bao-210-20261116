"""Small, dependency free filtering language used by coverage analysis.

The CLI has its own parser in :mod:`laspy.cli.core`.  The coverage tools
must also be usable when the optional CLI dependencies (``rich``/``typer``)
are not installed, so this module implements the same, deliberately small,
expression language without importing the CLI.

Supported expressions are combinations of comparisons::

    classification == 2
    intensity >= 50 and classification in [2, 3]
    not withheld == 1
    (classification == 2 or classification == 3) and z > 10

The supported operators are ``==``, ``!=``, ``<``, ``<=``, ``>``, ``>=``
and ``in``.  ``and``/``or``/``not`` may also be written as ``&&``/``||``/``!``.
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np


class FilterError(ValueError):
    """Raised when a filter expression cannot be parsed or applied."""


class Condition(enum.Enum):
    And = "and"
    Or = "or"


class Comparator(enum.Enum):
    Equality = "=="
    Difference = "!="
    GreaterOrEqual = ">="
    LessOrEqual = "<="
    LessThan = "<"
    GreaterThan = ">"
    In = "in"


_COMPARATOR_TO_FUNC: Dict[Comparator, Callable[[Any, Any], np.ndarray]] = {
    Comparator.Equality: np.equal,
    Comparator.Difference: np.not_equal,
    Comparator.GreaterThan: np.greater,
    Comparator.GreaterOrEqual: np.greater_equal,
    Comparator.LessThan: np.less,
    Comparator.LessOrEqual: np.less_equal,
    Comparator.In: np.isin,
}

# Same aliases as the CLI filter language.
_PDAL_NAMES = {
    "Classification": "classification",
    "Intensity": "intensity",
    "PointSourceId": "point_source_id",
    "ReturnNumber": "return_number",
    "NumberOfReturns": "number_of_returns",
    "ScanDirectionFlag": "scan_direction_flag",
    "EdgeOfFlightLine": "edge_of_flight_line",
    "ScanAngleRank": "scan_angle_rank",
    "UserData": "user_data",
    "Red": "red",
    "Green": "green",
    "Blue": "blue",
    "GpsTime": "gps_time",
    "Infrared": "nir",
    "ClassFlags": "classification_flags",
}


def _parse_number(value: str) -> Union[int, float]:
    value = value.strip()
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        raise FilterError(f"'{value}' is neither an int nor a float") from None


def _parse_list(value: str) -> List[Union[int, float]]:
    value = value.strip()
    if len(value) < 2 or (value[0], value[-1]) not in (("[", "]"), ("(", ")")):
        raise FilterError(f"'{value}' does not represent a list")
    body = value[1:-1].strip()
    if not body:
        return []
    return [_parse_number(item) for item in body.split(",")]


@dataclass(frozen=True)
class Token:
    kind: str
    value: str


_TOKEN_RE = re.compile(
    r"""
    \s*(?:
        (?P<list>\[[^\]]*\])
      | (?P<parenlist>\(\s*[-+0-9.eE,\s]*\))
      | (?P<op>==|!=|<=|>=|&&|\|\||[<>()!])
      | (?P<word>[^\s<>!=&|()\[\]]+)
    )
    """,
    re.VERBOSE,
)


def tokenize(expression: str) -> List[Token]:
    tokens: List[Token] = []
    position = 0
    while position < len(expression):
        match = _TOKEN_RE.match(expression, position)
        if match is None:
            raise FilterError(f"could not tokenize filter at {position!r}")
        position = match.end()
        if match.group("list") is not None:
            tokens.append(Token("literal", match.group("list")))
        elif match.group("parenlist") is not None:
            tokens.append(Token("literal", match.group("parenlist")))
        elif match.group("op") is not None:
            tokens.append(Token("op", match.group("op")))
        elif match.group("word") is not None:
            word = match.group("word")
            if word.lower() in ("and", "or", "not", "in"):
                tokens.append(Token("op", word.lower()))
            else:
                tokens.append(Token("literal", word))
        else:  # pragma: no cover - regex has no other branch
            raise FilterError(f"could not tokenize filter at {position!r}")
    return tokens


class _Parser:
    def __init__(self, tokens: Sequence[Token]):
        self.tokens = list(tokens)
        self.index = 0

    def peek(self) -> Optional[Token]:
        if self.index >= len(self.tokens):
            return None
        return self.tokens[self.index]

    def pop(self) -> Token:
        token = self.peek()
        if token is None:
            raise FilterError("unexpected end of filter expression")
        self.index += 1
        return token

    def parse(self) -> "FilterExpression":
        expression = self.parse_or()
        if self.peek() is not None:
            raise FilterError(f"unexpected token '{self.peek().value}'")
        return expression

    def parse_or(self) -> "FilterExpression":
        expression = self.parse_and()
        while self.peek() is not None and self.peek().value in ("or", "||"):
            self.pop()
            expression = FilterExpression(
                FilterExpressionKind.Binary,
                BinaryFilterExpression(
                    Condition.Or, expression, self.parse_and()
                ),
            )
        return expression

    def parse_and(self) -> "FilterExpression":
        expression = self.parse_unary()
        while self.peek() is not None and self.peek().value in ("and", "&&"):
            self.pop()
            expression = FilterExpression(
                FilterExpressionKind.Binary,
                BinaryFilterExpression(
                    Condition.And, expression, self.parse_unary()
                ),
            )
        return expression

    def parse_unary(self) -> "FilterExpression":
        token = self.peek()
        if token is not None and token.value in ("not", "!"):
            self.pop()
            return FilterExpression(
                FilterExpressionKind.Negated,
                NegatedFilterExpression(self.parse_unary()),
            )
        if token is not None and token.value == "(":
            self.pop()
            expression = self.parse_or()
            closing = self.pop()
            if closing.value != ")":
                raise FilterError("unmatched '(' in filter expression")
            return expression
        return self.parse_action()

    def parse_action(self) -> "FilterExpression":
        field = self.pop()
        if field.kind != "literal":
            raise FilterError(f"expected a field name, found '{field.value}'")
        comparator = self.pop()
        if comparator.kind != "op":
            raise FilterError(f"expected a comparator, found '{comparator.value}'")
        value = self.pop()
        if value.kind != "literal":
            raise FilterError(f"expected a value, found '{value.value}'")

        try:
            cmp = {
                "==": Comparator.Equality,
                "!=": Comparator.Difference,
                ">=": Comparator.GreaterOrEqual,
                "<=": Comparator.LessOrEqual,
                ">": Comparator.GreaterThan,
                "<": Comparator.LessThan,
                "in": Comparator.In,
            }[comparator.value]
        except KeyError:
            raise FilterError(
                f"'{comparator.value}' is not a comparator"
            ) from None

        return FilterExpression(
            FilterExpressionKind.Action,
            FilteringAction(field.value, cmp, value.value),
        )


@dataclass(frozen=True)
class FilteringAction:
    field_name: str
    comparator: Comparator
    value: str

    def apply(self, points: Any) -> np.ndarray:
        field_name = _PDAL_NAMES.get(self.field_name, self.field_name)
        try:
            field_values = points[field_name]
        except (KeyError, ValueError, IndexError):
            try:
                field_values = getattr(points, field_name)
            except AttributeError:
                raise FilterError(
                    f"No field named '{field_name}' is available for filtering"
                ) from None

        values = np.asarray(field_values)
        if self.comparator is Comparator.In:
            cmp_value = _parse_list(self.value)
        else:
            cmp_value = _parse_number(self.value)
        return np.asarray(_COMPARATOR_TO_FUNC[self.comparator](values, cmp_value))


@dataclass(frozen=True)
class NegatedFilterExpression:
    expr: "FilterExpression"


@dataclass(frozen=True)
class BinaryFilterExpression:
    condition: Condition
    lhs: "FilterExpression"
    rhs: "FilterExpression"


class FilterExpressionKind(enum.IntEnum):
    Action = enum.auto()
    Negated = enum.auto()
    Binary = enum.auto()


@dataclass(frozen=True)
class FilterExpression:
    kind: FilterExpressionKind
    data: Union[
        FilteringAction, NegatedFilterExpression, BinaryFilterExpression
    ]

    @classmethod
    def parse_string(cls, expression: str) -> "FilterExpression":
        return _Parser(tokenize(expression)).parse()

    def apply(self, points: Any) -> np.ndarray:
        if self.kind is FilterExpressionKind.Action:
            return self.data.apply(points)  # type: ignore[union-attr]
        if self.kind is FilterExpressionKind.Negated:
            return np.logical_not(self.data.expr.apply(points))  # type: ignore[union-attr]
        if self.kind is FilterExpressionKind.Binary:
            data = self.data  # type: ignore[assignment]
            lhs = data.lhs.apply(points)
            rhs = data.rhs.apply(points)
            if data.condition is Condition.And:
                return np.logical_and(lhs, rhs)
            return np.logical_or(lhs, rhs)
        raise FilterError(f"invalid filter expression kind {self.kind!r}")

    def __call__(self, points: Any) -> np.ndarray:
        return self.apply(points)


FilteringExpression = FilterExpression


def compile_filter(
    condition: Optional[
        Union[str, FilterExpression, Callable[[Any], np.ndarray], Iterable[str]]
    ],
    points: Any = None,
) -> Callable[[Any], np.ndarray]:
    """Return a callable that computes a boolean mask for *points*.

    ``condition`` may be an expression string, an already compiled
    :class:`FilterExpression`, a callable, or a sequence of strings which are
    combined with ``and``.
    """
    if condition is None:
        return lambda points: np.ones(len(points), dtype=bool)

    if isinstance(condition, str):
        expression = FilterExpression.parse_string(condition)
        return expression.apply

    if isinstance(condition, FilterExpression):
        return condition.apply

    if isinstance(condition, (list, tuple)) and all(
        isinstance(item, str) for item in condition
    ):
        expressions = [FilterExpression.parse_string(item) for item in condition]
        if not expressions:
            return lambda points: np.ones(len(points), dtype=bool)

        def apply_all(points: Any) -> np.ndarray:
            mask = np.ones(len(points), dtype=bool)
            for expression in expressions:
                mask &= expression.apply(points)
            return mask

        return apply_all

    if callable(condition):
        return condition

    apply_method = getattr(condition, "apply", None)
    if callable(apply_method):
        return apply_method

    raise TypeError(
        "filter_condition must be an expression string, a callable, "
        "a FilterExpression, or a sequence of expression strings"
    )


__all__ = [
    "BinaryFilterExpression",
    "Comparator",
    "Condition",
    "FilterError",
    "FilterExpression",
    "FilterExpressionKind",
    "FilteringAction",
    "FilteringExpression",
    "NegatedFilterExpression",
    "Token",
    "compile_filter",
    "tokenize",
]
