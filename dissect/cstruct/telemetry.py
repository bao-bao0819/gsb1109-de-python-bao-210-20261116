"""A lossless, re-importable text view over the string-like fields of a structure.

Telemetry frequently stores text in three different ways:

* a fixed width byte string, e.g. ``char magic[4]``;
* a zero terminated string, e.g. ``char name[]`` or ``wchar name[]``;
* a wide character array padded with zero code units, e.g. ``wchar label[8]``.

A consumer may only accept one unified text representation, with a sidecar that stores the original bytes for
whatever the text cannot reproduce. This module turns such fields into a :class:`TelemetryView`: a list of
:class:`TelemetryField` records that each carry their declaration kind, their exact byte boundary, the decoded text
and whether that text can be encoded back to the exact original bytes.

Fields that cannot be reproduced from text alone are *lossy* and need their original bytes in the sidecar. Values
that only become equal after transcoding and removing the terminator or padding are reported as collisions, and are
kept separate from genuine empty strings, missing terminators and truncated code units.

Use :meth:`TelemetryView.export` together with :meth:`TelemetryExport.rebuild` to round-trip a structure without
changing the offset or size of any field: string regions are patched in place, never inserted or removed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from dataclasses import field as dataclass_field
from enum import Enum
from typing import TYPE_CHECKING, Any

from dissect.cstruct.exception import TelemetryError
from dissect.cstruct.types.base import BaseArray
from dissect.cstruct.types.char import Char
from dissect.cstruct.types.structure import Structure, Union, UnionProxy
from dissect.cstruct.types.wchar import Wchar
from dissect.cstruct.util import normalize_endianness

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from typing_extensions import Self

    from dissect.cstruct.cstruct import Endianness

REPLACEMENT = "�"


class FieldKind(str, Enum):
    """How a telemetry field declares its text."""

    FIXED_BYTES = "fixed_bytes"
    """A fixed width byte string, e.g. ``char magic[4]``. Trailing zero bytes are padding."""

    ZERO_TERMINATED = "zero_terminated"
    """A zero terminated text field, e.g. ``char name[]`` or ``wchar name[]``."""

    WIDE_PADDED = "wide_padded"
    """A wide character array padded with zero code units, e.g. ``wchar label[8]``."""


class FieldState(str, Enum):
    """The boundary state of a telemetry field.

    The exceptional states are deliberately distinct from one another and from non-empty text.
    """

    TEXT = "text"
    """A non-empty text value was decoded."""

    EMPTY = "empty"
    """A genuine empty string: the field is zero length or terminated immediately."""

    UNTERMINATED = "unterminated"
    """A zero terminated declaration whose boundary ended without a terminator."""

    TRUNCATED = "truncated"
    """A field that ends in a partial code unit (an incomplete UTF-16 unit)."""


class InsufficientBudgetError(TelemetryError):
    """Raised when the sidecar budget cannot hold every original that must be kept."""

    def __init__(self, plan: SidecarPlan):
        self.plan = plan
        super().__init__(
            f"Sidecar budget of {plan.budget} bytes is insufficient: "
            f"{plan.total_size} bytes are required for {len(plan.required)} originals "
            f"({plan.deficit} bytes short)."
        )


class MissingOriginalError(TelemetryError):
    """Raised when rebuilding a lossy field that has no sidecar original."""


class OffsetInvarianceError(TelemetryError):
    """Raised when a rebuilt value would change the length (and thus the offset) of a field."""


def _unit_of(kind: FieldKind, unit: int | None) -> int:
    if unit is None:
        return 2 if kind is FieldKind.WIDE_PADDED else 1
    if unit not in (1, 2):
        raise ValueError(f"Unsupported character unit: {unit!r} (expected 1 or 2)")
    return unit


def _strip_padding(raw: bytes, unit: int) -> bytes:
    """Strip trailing zero code units, without eating into the previous code unit."""
    end = len(raw)
    while end >= unit and raw[end - unit : end] == b"\x00" * unit:
        end -= unit
    return raw[:end]


def _terminator_index(raw: bytes, unit: int) -> int | None:
    """Return the offset of the first terminator on a code unit boundary, relative to the field start."""
    terminator = b"\x00" * unit
    for index in range(0, len(raw) - unit + 1, unit):
        if raw[index : index + unit] == terminator:
            return index
    return None


def _decode_narrow(payload: bytes) -> tuple[str, bool]:
    """Decode a byte string as UTF-8, returning the text and whether the bytes were valid UTF-8."""
    try:
        return payload.decode("utf-8"), True
    except UnicodeDecodeError:
        return payload.decode("utf-8", "replace"), False


def _decode_wide(payload: bytes, endian: Endianness) -> tuple[str, bool]:
    """Decode UTF-16 with surrogatepass, then unify lone surrogates so the text is valid UTF-8.

    Returns the text and whether it can be encoded back to exactly ``payload``.
    """
    encoding = Wchar.__encoding_map__[endian]
    text = payload.decode(encoding, "surrogatepass")
    text = _unify_surrogates(text)

    try:
        return text, text.encode(encoding, "strict") == payload
    except UnicodeError:
        return text, False


def _unify_surrogates(text: str) -> str:
    """Replace lone surrogates with the same replacement character as invalid UTF-8, so both collide."""
    return "".join(REPLACEMENT if _is_lone_surrogate(char) else char for char in text)


def _is_lone_surrogate(char: str) -> bool:
    return 0xD800 <= ord(char) <= 0xDFFF


@dataclass(frozen=True)
class FieldSpec:
    """An explicit declaration and boundary for one telemetry field.

    Args:
        name: The name of the field.
        kind: How the text is declared.
        offset: The byte offset of the field in the data.
        length: The byte length of the field boundary.
        unit: The width of a single character in bytes. Defaults to 2 for padded wide characters, 1 otherwise.
        endian: The endianness for wide characters. Defaults to the endianness of the view.
    """

    name: str
    kind: FieldKind
    offset: int
    length: int
    unit: int | None = None
    endian: str | None = None

    def __post_init__(self) -> None:
        if self.offset < 0:
            raise ValueError("Field offset cannot be negative")
        if self.length < 0:
            raise ValueError("Field length cannot be negative")
        object.__setattr__(self, "unit", _unit_of(self.kind, self.unit))


@dataclass(frozen=True)
class TelemetryField:
    """A single telemetry field, with its declaration, boundary, text and round-trip properties."""

    name: str
    kind: FieldKind
    offset: int
    length: int
    unit: int
    endian: Endianness
    raw: bytes
    payload: bytes
    text: str
    state: FieldState
    terminated: bool
    truncated: bytes
    lossless: bool
    reason: str | None = None

    @property
    def end(self) -> int:
        """The exclusive end offset of this field."""
        return self.offset + self.length

    @property
    def needs_original(self) -> bool:
        """Whether this field needs its original bytes in the sidecar to be reproduced exactly."""
        return not self.lossless

    @property
    def is_empty(self) -> bool:
        """Whether this field is a genuine empty string."""
        return self.state is FieldState.EMPTY

    @property
    def is_unterminated(self) -> bool:
        """Whether this zero terminated field ended without a terminator."""
        return self.state is FieldState.UNTERMINATED

    @property
    def is_truncated(self) -> bool:
        """Whether this field ends in a partial code unit."""
        return self.state is FieldState.TRUNCATED

    def encode(self, text: str | None = None) -> bytes:
        """Encode a text value for this field's declaration, without terminator or padding."""
        value = self.text if text is None else text
        return _encode_text(self.unit, self.endian, value)

    def rebuild(self, text: str | None = None) -> bytes:
        """Rebuild the exact field bytes from text alone.

        Raises:
            OffsetInvarianceError: If the encoded text does not fit the declared boundary.
            MissingOriginalError: If this field is lossy and its text cannot reproduce the original bytes.
        """
        if not self.lossless and text is None:
            raise MissingOriginalError(f"Field {self.name!r} needs its sidecar original to be rebuilt")

        raw = _encode_value(
            FieldSpec(self.name, self.kind, self.offset, self.length, self.unit, self.endian),
            self.text if text is None else text,
        )

        if text is None and raw != self.raw:
            raise MissingOriginalError(f"Field {self.name!r} needs its sidecar original to be rebuilt")

        return raw


@dataclass(frozen=True)
class Collision:
    """A set of fields that share one text value while having different original bytes."""

    text: str
    fields: tuple[TelemetryField, ...]

    @property
    def names(self) -> tuple[str, ...]:
        """The names of the colliding fields."""
        return tuple(field.name for field in self.fields)

    @property
    def payloads(self) -> tuple[bytes, ...]:
        """The distinct de-terminated payloads that collapsed into one text value."""
        seen: list[bytes] = []
        for field in self.fields:
            if field.payload not in seen:
                seen.append(field.payload)
        return tuple(seen)

    @property
    def raws(self) -> tuple[bytes, ...]:
        """The distinct original byte strings that collapsed into one text value."""
        seen: list[bytes] = []
        for field in self.fields:
            if field.raw not in seen:
                seen.append(field.raw)
        return tuple(seen)


@dataclass(frozen=True)
class SidecarPlan:
    """The outcome of selecting the originals that must be kept within a sidecar budget."""

    required: tuple[TelemetryField, ...]
    total_size: int
    budget: int | None
    overhead: int = 0

    @property
    def feasible(self) -> bool:
        """Whether the budget can hold every original that must be kept."""
        return self.budget is None or self.total_size <= self.budget

    @property
    def deficit(self) -> int:
        """The number of bytes the budget is short, or 0 when the plan fits."""
        if self.budget is None:
            return 0
        return max(0, self.total_size - self.budget)

    @property
    def selection(self) -> tuple[TelemetryField, ...]:
        """The originals that must be kept. Every lossy original is mandatory."""
        return self.required

    def raise_if_infeasible(self) -> None:
        """Raise :class:`InsufficientBudgetError` if the plan does not fit the budget."""
        if not self.feasible:
            raise InsufficientBudgetError(self)


@dataclass(frozen=True)
class TelemetryExport:
    """A unified text view plus the sidecar originals, able to rebuild the original bytes in place."""

    template: bytes
    layout: tuple[FieldSpec, ...]
    texts: dict[str, str]
    sidecar: dict[str, bytes]
    plan: SidecarPlan = dataclass_field(compare=False)

    @property
    def size(self) -> int:
        """The size of the original data."""
        return len(self.template)

    def rebuild(self) -> bytes:
        """Rebuild the original bytes, patching every field at its declared offset and length."""
        data = bytearray(self.template)
        required = {field.name for field in self.plan.required}

        for spec in self.layout:
            if spec.name in self.sidecar:
                raw = self.sidecar[spec.name]
            else:
                if spec.name in required:
                    raise MissingOriginalError(f"Field {spec.name!r} needs its sidecar original to be rebuilt")
                try:
                    text = self.texts[spec.name]
                except KeyError:
                    raise MissingOriginalError(f"Field {spec.name!r} has neither text nor an original") from None
                raw = _encode_value(spec, text)

            if len(raw) != spec.length:
                raise OffsetInvarianceError(
                    f"Field {spec.name!r} rebuilds to {len(raw)} bytes, but its boundary is {spec.length} bytes"
                )

            data[spec.offset : spec.offset + spec.length] = raw

        if len(data) != len(self.template):
            raise OffsetInvarianceError("Rebuilding changed the size of the data")

        return bytes(data)

    def to_dict(self) -> dict[str, Any]:
        """Serialize this export to a JSON-compatible dictionary."""
        return {
            "version": 1,
            "size": self.size,
            "template": self.template.hex(),
            "layout": [
                {
                    "name": spec.name,
                    "kind": spec.kind.value,
                    "offset": spec.offset,
                    "length": spec.length,
                    "unit": spec.unit,
                    "endian": spec.endian,
                }
                for spec in self.layout
            ],
            "texts": dict(self.texts),
            "sidecar": {name: raw.hex() for name, raw in self.sidecar.items()},
            "plan": {
                "feasible": self.plan.feasible,
                "total_size": self.plan.total_size,
                "budget": self.plan.budget,
                "overhead": self.plan.overhead,
                "required": [field.name for field in self.plan.required],
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Load an export from a dictionary produced by :meth:`to_dict`."""
        if data.get("version") != 1:
            raise ValueError(f"Unsupported telemetry export version: {data.get('version')!r}")

        template = bytes.fromhex(data["template"])
        layout = tuple(
            FieldSpec(
                name=item["name"],
                kind=FieldKind(item["kind"]),
                offset=item["offset"],
                length=item["length"],
                unit=item.get("unit"),
                endian=item.get("endian"),
            )
            for item in data["layout"]
        )
        sidecar = {name: bytes.fromhex(raw) for name, raw in data["sidecar"].items()}
        texts = dict(data["texts"])

        patched = bytearray(template)
        for spec in layout:
            if spec.name in sidecar:
                patched[spec.offset : spec.offset + spec.length] = sidecar[spec.name]
        fields = tuple(_make_field(spec, bytes(patched)) for spec in layout)

        plan_info = data.get("plan", {})
        required_names = set(plan_info.get("required") or sidecar)
        required = tuple(field for field in fields if field.name in required_names)
        overhead = plan_info.get("overhead", 0)
        plan = SidecarPlan(
            required=required,
            total_size=sum(len(field.raw) + overhead for field in required),
            budget=plan_info.get("budget"),
            overhead=overhead,
        )

        return cls(template=template, layout=layout, texts=texts, sidecar=sidecar, plan=plan)


class TelemetryView:
    """A re-importable text view over the string-like fields of a structure or of explicit boundaries."""

    def __init__(
        self,
        data: bytes,
        fields: Sequence[TelemetryField],
        specs: Sequence[FieldSpec],
        *,
        endian: Endianness = "<",
    ):
        self.data = bytes(data)
        self.fields = tuple(fields)
        self.specs = tuple(specs)
        self.endian = endian

    @classmethod
    def from_structure(
        cls,
        obj: Structure | type[Structure],
        data: bytes | bytearray | memoryview | None = None,
        *,
        endian: str | None = None,
        include_unions: bool = False,
    ) -> Self:
        """Build a view from a parsed structure, or from a structure type and its data.

        Field boundaries are taken from the declaration and from the parsed sizes of dynamic fields. String fields
        whose offset or size cannot be determined are rejected rather than guessed.

        Args:
            obj: A parsed :class:`~dissect.cstruct.types.structure.Structure`, or a structure type.
            data: The data to parse when ``obj`` is a structure type.
            endian: The endianness for wide characters. Defaults to the endianness of the cstruct instance.
            include_unions: Whether to include string fields inside unions. Union members overlap, so their
                boundaries must be identical for a view to be re-importable.
        """
        if isinstance(obj, Structure):
            structure = obj
            if data is None:
                data = structure.dumps()
        elif isinstance(obj, type) and issubclass(obj, Structure) and data is not None:
            structure = obj(data)
        else:
            raise TypeError("Expected a Structure instance, or a Structure type with data")

        data = bytes(data)
        normalized = normalize_endianness(endian) if endian is not None else structure.__class__.__cs__.endian

        specs: list[FieldSpec] = []
        _collect_specs(structure, 0, "", specs, normalized, include_unions=include_unions)

        if not specs:
            raise TelemetryError(f"Structure {structure.__class__.__name__!r} has no telemetry string fields")

        return cls.from_parts(specs, data, endian=normalized)

    @classmethod
    def from_parts(
        cls,
        specs: Iterable[FieldSpec],
        data: bytes | bytearray | memoryview,
        *,
        endian: str = "<",
    ) -> Self:
        """Build a view from explicit field declarations and boundaries.

        This is the low-level entry point. It does not parse the data, so it is also the way to represent boundaries
        that a strict parser would reject, such as a missing terminator or a truncated wide code unit.
        """
        data = bytes(data)
        normalized = normalize_endianness(endian)

        normalized_specs = []
        names = set()
        for spec in specs:
            if spec.name in names:
                raise ValueError(f"Duplicate telemetry field name: {spec.name!r}")
            names.add(spec.name)
            if spec.offset + spec.length > len(data):
                raise ValueError(
                    f"Field {spec.name!r} boundary [{spec.offset}, {spec.offset + spec.length}) "
                    f"exceeds the data size of {len(data)} bytes"
                )
            if spec.unit not in (1, 2):
                raise ValueError(f"Unsupported character unit for field {spec.name!r}: {spec.unit!r}")
            normalized_specs.append(replace(spec, endian=normalize_endianness(spec.endian or normalized)))

        _validate_non_overlapping(normalized_specs)

        fields = [_make_field(spec, data) for spec in normalized_specs]
        return cls(data, fields, normalized_specs, endian=normalized)

    @property
    def records(self) -> tuple[TelemetryField, ...]:
        """The telemetry fields, in declaration order."""
        return self.fields

    @property
    def texts(self) -> dict[str, str]:
        """The unified text value of every field."""
        return {field.name: field.text for field in self.fields}

    @property
    def originals(self) -> dict[str, bytes]:
        """The original bytes of every field that cannot be reproduced from text alone."""
        return {field.name: field.raw for field in self.fields if field.needs_original}

    @property
    def collisions(self) -> tuple[Collision, ...]:
        """Groups of fields that map to the same text value from different original bytes.

        This catches byte strings and wide characters that were decoded from different bytes, a terminated value
        that equals an unterminated one once the terminator is removed, and values that only differ in padding.
        The fields keep their own state, so a missing terminator stays distinct from a terminated value.
        """
        groups = _group_by_text(self.fields)
        result = []
        for text, fields in groups.items():
            if len({field.raw for field in fields}) > 1:
                result.append(Collision(text, tuple(fields)))
        return tuple(result)

    @property
    def duplicates(self) -> tuple[Collision, ...]:
        """Groups of fields that share both their text and their original bytes."""
        groups = _group_by_text(self.fields)
        result = []
        for text, fields in groups.items():
            if len(fields) > 1 and len({field.raw for field in fields}) == 1:
                result.append(Collision(text, tuple(fields)))
        return tuple(result)

    @property
    def problems(self) -> tuple[TelemetryField, ...]:
        """Fields with a missing terminator or a truncated code unit, kept apart from empty or invalid text."""
        return tuple(field for field in self.fields if field.state in (FieldState.UNTERMINATED, FieldState.TRUNCATED))

    @property
    def lossy(self) -> tuple[TelemetryField, ...]:
        """Fields that cannot be reproduced from text alone and need their original in the sidecar."""
        return tuple(field for field in self.fields if field.needs_original)

    def plan(self, budget: int | None = None, *, overhead: int = 0) -> SidecarPlan:
        """Select the originals that must be kept to reproduce every field, within ``budget`` bytes.

        Args:
            budget: The maximum number of sidecar bytes, or ``None`` for no limit.
            overhead: An optional fixed cost per stored original, e.g. for its key and metadata.
        """
        if budget is not None and budget < 0:
            raise ValueError("Budget cannot be negative")
        if overhead < 0:
            raise ValueError("Overhead cannot be negative")

        required = tuple(sorted((field for field in self.fields if field.needs_original), key=lambda f: f.offset))
        total_size = sum(len(field.raw) + overhead for field in required)
        return SidecarPlan(required=required, total_size=total_size, budget=budget, overhead=overhead)

    def export(self, budget: int | None = None, *, overhead: int = 0) -> TelemetryExport:
        """Create a unified text export with the sidecar originals needed for an exact rebuild.

        Raises:
            InsufficientBudgetError: If the budget cannot hold every required original.
        """
        plan = self.plan(budget, overhead=overhead)
        plan.raise_if_infeasible()

        template = bytearray(self.data)
        for spec in self.specs:
            template[spec.offset : spec.offset + spec.length] = b"\x00" * spec.length

        return TelemetryExport(
            template=bytes(template),
            layout=self.specs,
            texts=self.texts,
            sidecar={field.name: field.raw for field in plan.selection},
            plan=plan,
        )

    def reimport(self, export: TelemetryExport | dict[str, Any]) -> bytes:
        """Rebuild the original bytes from an export (or its dictionary form)."""
        if isinstance(export, dict):
            export = TelemetryExport.from_dict(export)
        return export.rebuild()

    def __len__(self) -> int:
        return len(self.fields)

    def __getitem__(self, index: int | slice) -> TelemetryField | tuple[TelemetryField, ...]:
        return self.fields[index]

    def __iter__(self):
        return iter(self.fields)

    def __repr__(self) -> str:
        return f"<TelemetryView size={len(self.data)} fields={len(self.fields)}>"


def build_view(
    obj: Structure | type[Structure],
    data: bytes | bytearray | memoryview | None = None,
    *,
    endian: str | None = None,
    include_unions: bool = False,
) -> TelemetryView:
    """Build a :class:`TelemetryView` from a parsed structure or a structure type and data."""
    return TelemetryView.from_structure(obj, data, endian=endian, include_unions=include_unions)


def plan_sidecar(view: TelemetryView, budget: int | None = None, *, overhead: int = 0) -> SidecarPlan:
    """Select the originals that must be kept within ``budget`` bytes."""
    return view.plan(budget, overhead=overhead)


def export_telemetry(view: TelemetryView, budget: int | None = None, *, overhead: int = 0) -> TelemetryExport:
    """Export a view as unified text plus a sidecar of the originals that must be kept."""
    return view.export(budget, overhead=overhead)


def reimport_telemetry(export: TelemetryExport | dict[str, Any]) -> bytes:
    """Rebuild the original bytes from an exported telemetry view."""
    if isinstance(export, dict):
        export = TelemetryExport.from_dict(export)
    return export.rebuild()


def _encode_text(unit: int, endian: Endianness, value: str) -> bytes:
    """Encode text for a field, without terminator or padding."""
    try:
        if unit == 2:
            return value.encode(Wchar.__encoding_map__[endian], "strict")
        return value.encode("utf-8", "strict")
    except UnicodeError as e:
        raise TelemetryError(f"Text cannot be encoded: {e}") from e


def _encode_value(spec: FieldSpec, text: str) -> bytes:
    """Encode text into the exact declared boundary, adding the terminator or padding as declared.

    The length is always exactly ``spec.length``; a value that does not fit raises :class:`OffsetInvarianceError`.
    """
    encoded = _encode_text(spec.unit or 1, normalize_endianness(spec.endian or "<"), text)

    if spec.kind is FieldKind.ZERO_TERMINATED:
        if len(encoded) > spec.length - (spec.unit or 1):
            raise OffsetInvarianceError(
                f"Field {spec.name!r} needs a terminator inside its {spec.length} byte boundary, "
                f"but its text encodes to {len(encoded)} bytes"
            )
        raw = encoded + b"\x00" * (spec.length - len(encoded))
    else:
        if len(encoded) > spec.length:
            raise OffsetInvarianceError(
                f"Field {spec.name!r} encodes to {len(encoded)} bytes, but its boundary is {spec.length} bytes"
            )
        raw = encoded + b"\x00" * (spec.length - len(encoded))

    if len(raw) != spec.length:
        raise OffsetInvarianceError(
            f"Field {spec.name!r} rebuilds to {len(raw)} bytes, but its boundary is {spec.length} bytes"
        )
    return raw


def _group_by_text(fields: Sequence[TelemetryField]) -> dict[str, list[TelemetryField]]:
    groups: dict[str, list[TelemetryField]] = {}
    for field in fields:
        groups.setdefault(field.text, []).append(field)
    return groups


def _validate_non_overlapping(specs: Sequence[FieldSpec]) -> None:
    # Identical boundaries are allowed, e.g. for the members of a union. Partial overlaps are not, because the
    # fields would then share only some of their bytes and could not be patched independently.
    ordered = sorted((spec for spec in specs if spec.length), key=lambda spec: (spec.offset, spec.length))
    previous: FieldSpec | None = None
    for spec in ordered:
        if previous is not None and spec.offset < previous.offset + previous.length:
            identical = spec.offset == previous.offset and spec.length == previous.length
            if not identical:
                raise TelemetryError(
                    f"Field {spec.name!r} overlaps field {previous.name!r}; overlapping string boundaries "
                    "cannot be re-imported without changing offsets"
                )
        if previous is None or spec.offset + spec.length > previous.offset + previous.length:
            previous = spec


def _make_field(spec: FieldSpec, data: bytes) -> TelemetryField:
    endian = normalize_endianness(spec.endian or "<")
    raw = data[spec.offset : spec.offset + spec.length]
    unit = spec.unit or 1

    if len(raw) != spec.length:
        raise ValueError(f"Field {spec.name!r} extends beyond the data")

    if spec.kind is FieldKind.FIXED_BYTES:
        if unit != 1:
            raise ValueError(f"Fixed byte field {spec.name!r} must use 1-byte units")
        payload = _strip_padding(raw, 1)
        text, valid = _decode_narrow(payload)
        state = FieldState.EMPTY if not payload else FieldState.TEXT
        terminated = True
        truncated = b""
        reason = None if valid else "invalid UTF-8"
    elif spec.kind is FieldKind.WIDE_PADDED:
        if unit != 2:
            raise ValueError(f"Wide padded field {spec.name!r} must use 2-byte code units")
        if len(raw) % 2:
            truncated = raw[-1:]
            payload = _strip_padding(raw[:-1], 2)
            state = FieldState.TRUNCATED
            terminated = False
            text, valid = _decode_wide(payload, endian)
            reason = "truncated code unit"
        else:
            truncated = b""
            payload = _strip_padding(raw, 2)
            state = FieldState.EMPTY if not payload else FieldState.TEXT
            terminated = True
            text, valid = _decode_wide(payload, endian)
            reason = None if valid else "lone surrogate"
    elif spec.kind is FieldKind.ZERO_TERMINATED:
        index = _terminator_index(raw, unit)
        if index is not None:
            payload = raw[:index]
            trailing = raw[index + unit :]
            state = FieldState.EMPTY if not payload else FieldState.TEXT
            terminated = True
            truncated = b""
            reason = "bytes after terminator" if trailing else None
        elif len(raw) % unit:
            truncated = raw[len(raw) - (len(raw) % unit) :]
            payload = raw[: len(raw) - len(truncated)]
            state = FieldState.TRUNCATED
            terminated = False
            reason = "truncated code unit"
        else:
            payload = raw
            state = FieldState.UNTERMINATED
            terminated = False
            truncated = b""
            reason = "missing terminator"

        if unit == 2:
            text, valid = _decode_wide(payload, endian)
            if not valid and reason is None:
                reason = "lone surrogate"
        else:
            text, valid = _decode_narrow(payload)
            if not valid and reason is None:
                reason = "invalid UTF-8"
    else:
        raise ValueError(f"Unsupported telemetry kind: {spec.kind!r}")

    field = TelemetryField(
        name=spec.name,
        kind=spec.kind,
        offset=spec.offset,
        length=spec.length,
        unit=unit,
        endian=endian,
        raw=raw,
        payload=payload,
        text=text,
        state=state,
        terminated=terminated,
        truncated=truncated,
        lossless=False,
        reason=reason,
    )

    lossless, loss_reason = _is_lossless(field)
    if not lossless and field.reason is None:
        field = replace(field, reason=loss_reason)

    return replace(field, lossless=lossless)


def _is_lossless(field: TelemetryField) -> tuple[bool, str | None]:
    if field.state in (FieldState.UNTERMINATED, FieldState.TRUNCATED):
        return False, field.reason or field.state.value
    if field.truncated:
        return False, "truncated code unit"
    if field.kind is FieldKind.ZERO_TERMINATED and field.payload + b"\x00" * field.unit != field.raw:
        return False, field.reason or "bytes after terminator"

    try:
        rebuilt = field.rebuild(field.text)
    except TelemetryError:
        return False, field.reason or "text cannot be encoded"
    if rebuilt != field.raw:
        return False, field.reason or "text does not reproduce the original bytes"
    return True, None


def _collect_specs(
    structure: Structure,
    start: int,
    prefix: str,
    out: list[FieldSpec],
    endian: Endianness,
    *,
    include_unions: bool,
) -> None:
    cls = structure.__class__
    is_union = isinstance(structure, Union)
    offset = start

    for field in cls.__fields__:
        if field.offset is not None:
            offset = start + field.offset
        elif is_union:
            offset = start
        elif cls.__align__ and offset is not None:
            offset += -offset & (field.alignment - 1)

        if field.bits:
            continue

        # Of fields with a repeated name ("_"), a structure only keeps the value and size of the last one
        if cls.__lookup__[field._name] is not field:
            if offset is not None:
                offset = None if field.type.__size__ is None else offset + field.type.__size__
            continue

        value = getattr(structure, field._name)
        if isinstance(value, UnionProxy):
            # Structures nested inside a union are proxied; unwrap to the real Structure
            value = value.__target__
        size = structure.__sizes__.get(field._name)
        name = f"{prefix}.{field.name}" if prefix and field.name else (field.name or prefix)

        kind_spec = _string_kind(field.type)
        if kind_spec is not None:
            if is_union and not include_unions:
                if offset is not None:
                    offset = None if size is None else offset + size
                continue
            if offset is None or size is None:
                raise TelemetryError(
                    f"Cannot determine the boundary of telemetry field {name!r}; parse the structure first"
                )
            kind, unit = kind_spec
            out.append(FieldSpec(name=name, kind=kind, offset=offset, length=size, unit=unit, endian=endian))
        elif isinstance(value, Structure):
            if offset is None or size is None:
                raise TelemetryError(
                    f"Cannot determine the boundary of nested structure {name!r}; parse the structure first"
                )
            if is_union and not include_unions:
                offset = None if size is None else offset + size
                continue
            _collect_specs(value, offset, name, out, endian, include_unions=include_unions)
        elif isinstance(value, list) and value and isinstance(value[0], Structure):
            if offset is None:
                raise TelemetryError(
                    f"Cannot determine the boundary of nested structure array {name!r}; parse the structure first"
                )
            element_start = offset
            for index, element in enumerate(value):
                element = element.__target__ if isinstance(element, UnionProxy) else element
                _collect_specs(
                    element,
                    element_start,
                    f"{name}[{index}]",
                    out,
                    endian,
                    include_unions=include_unions,
                )
                element_size = _structure_boundary(element)
                if element_size is None:
                    raise TelemetryError(f"Cannot determine the boundary of nested structure {name}[{index}]")
                element_start += element_size

        if offset is not None:
            offset = None if size is None else offset + size


def _structure_boundary(structure: Structure) -> int | None:
    """Return the parsed byte size of a structure, including bit fields and alignment padding."""
    cls = structure.__class__
    if cls.__size__ is not None:
        return cls.__size__

    sizes = structure.__sizes__
    total = 0
    bits_remaining = 0
    bits_type = None

    for field in cls.__fields__:
        size = sizes.get(field._name)
        if size is None:
            return None

        if field.bits:
            field_type = field.type.__type__ if hasattr(field.type, "__type__") else field.type
            if bits_remaining == 0 or field_type is not bits_type:
                bits_type = field_type
                bits_remaining = (field_type.__size__ or 0) * 8
                total += field_type.__size__ or 0
            bits_remaining -= field.bits
            continue

        bits_type = None
        bits_remaining = 0
        total += size

    return total


def _string_kind(type_: Any) -> tuple[FieldKind, int] | None:
    dimensions = 0
    base = type_
    while isinstance(base, type) and issubclass(base, BaseArray):
        dimensions += 1
        base = base.__type__

    if dimensions != 1:
        return None

    if issubclass(base, Wchar):
        if type_.__null_terminated__:
            return FieldKind.ZERO_TERMINATED, 2
        return FieldKind.WIDE_PADDED, 2
    if issubclass(base, Char):
        if type_.__null_terminated__:
            return FieldKind.ZERO_TERMINATED, 1
        return FieldKind.FIXED_BYTES, 1

    return None
