"""
Preview character-field encoding conversions.

Character fields in SAS Transport files are fixed-width byte fields.  A
value may fit when encoded with one codec and overflow when encoded with
another, even when the number of characters is unchanged.  The helpers
in this module report the effect of an encoding conversion before any
bytes are written.

The report keeps the original bytes, the source and target encodings,
whether each value is representable, how truncation changes it, and
which distinct values collapse to the same target bytes.
"""

# Standard Library
import codecs
import dataclasses
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from typing import Any, List, Optional, Tuple

# Community Packages
import pandas as pd

__all__ = [
    'Collision',
    'DatasetEncodingReport',
    'EncodingReport',
    'LibraryEncodingReport',
    'ValueEncodingReport',
    'analyze_encoding',
    'audit_encoding',
    'check_encoding',
    'dataset_encoding_report',
    'encoding_report',
    'library_encoding_report',
    'preflight_encoding',
    'preview_encoding',
    'report_encoding',
    'transcode_report',
    'truncate_bytes',
]

DEFAULT_SOURCE_ENCODING = 'ISO-8859-1'
DEFAULT_TARGET_ENCODING = DEFAULT_SOURCE_ENCODING


def _is_missing(value):
    """
    Return whether a value should be treated as a missing character value.
    """
    if value is None:
        return True
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(result, bool):
        return result
    try:
        return bool(result)
    except (TypeError, ValueError):
        return False


def truncate_bytes(bytestring, width, encoding):
    """
    Truncate bytes to ``width`` without splitting a multibyte character.

    The returned bytes always decode successfully with ``encoding``.  If
    no complete character fits in ``width`` bytes, an empty byte string
    is returned.
    """
    bytestring = bytes(bytestring)
    if width is None or len(bytestring) <= width:
        return bytestring

    try:
        text = bytestring.decode(encoding)
    except (UnicodeDecodeError, LookupError):
        prefix = bytestring[:width]
        while prefix:
            try:
                prefix.decode(encoding)
            except (UnicodeDecodeError, LookupError):
                prefix = prefix[:-1]
            else:
                break
        return prefix

    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        try:
            encoded = text[:middle].encode(encoding)
        except UnicodeEncodeError:
            high = middle - 1
            continue
        if len(encoded) <= width:
            low = middle
        else:
            high = middle - 1
    return text[:low].encode(encoding)


def encode_character(value, width, encoding, errors='strict'):
    """
    Encode a character value to a fixed-width byte field.

    The result is exactly ``width`` bytes when ``width`` is not ``None``.
    Overlong values are truncated only at a character boundary.
    """
    if _is_missing(value):
        encoded = b''
    elif isinstance(value, (bytes, bytearray, memoryview)):
        encoded = bytes(value)
    else:
        encoded = str(value).encode(encoding, errors=errors)
    if width is None:
        return encoded
    return truncate_bytes(encoded, width, encoding).ljust(width)


@dataclasses.dataclass
class ValueEncodingReport:
    """
    The encoding result for one character value.
    """

    value: str
    original_bytes: bytes
    source_encoding: str
    encoding: str
    width: Optional[int]
    representable: bool
    encoded_bytes: Optional[bytes] = None
    truncated: bool = False
    truncated_bytes: bytes = b''
    truncated_value: Optional[str] = None
    error: Optional[str] = None
    errors: str = 'strict'
    index: Optional[Any] = None
    position: Optional[int] = None
    missing: bool = False
    source_representable: bool = True
    original_bytes_complete: bool = True
    original_error: Optional[str] = None
    collision: Optional[int] = None
    source_value: Any = dataclasses.field(default=None, repr=False)

    @property
    def original(self):
        """The original value."""
        return self.value

    @property
    def original_encoding(self):
        """The encoding used to recover the original bytes."""
        return self.source_encoding

    @property
    def used_encoding(self):
        """The encoding adopted for the output."""
        return self.encoding

    @property
    def target_encoding(self):
        """The target encoding."""
        return self.encoding

    @property
    def target_bytes(self):
        """The encoded bytes, before fixed-width padding."""
        return self.truncated_bytes

    @property
    def target_value(self):
        """The value after encoding and truncation."""
        return self.truncated_value

    @property
    def output_bytes(self):
        """The bytes that would be written, before fixed-width padding."""
        return self.truncated_bytes

    @property
    def output_value(self):
        """The value after encoding and truncation."""
        return self.truncated_value

    @property
    def final_bytes(self):
        """Alias for ``output_bytes``."""
        return self.truncated_bytes

    @property
    def final_value(self):
        """Alias for ``output_value``."""
        return self.truncated_value

    @property
    def original_length(self):
        """The number of original bytes."""
        return len(self.original_bytes)

    @property
    def byte_length(self):
        """The number of output bytes."""
        return len(self.truncated_bytes)

    @property
    def encoded_length(self):
        """The number of bytes before truncation."""
        if self.encoded_bytes is None:
            return 0
        return len(self.encoded_bytes)

    @property
    def output_length(self):
        """The number of output bytes before padding."""
        return len(self.truncated_bytes)

    @property
    def stored_bytes(self):
        """The bytes as stored in the fixed-width field, including padding."""
        if self.width is None:
            return self.truncated_bytes
        return self.truncated_bytes.ljust(self.width)

    @property
    def stored_value(self):
        """The value an XPT reader recovers from the fixed-width field."""
        return _decoded_field_value(self.stored_bytes, self.encoding)

    @property
    def stored_length(self):
        """The number of bytes as stored, including padding."""
        return len(self.stored_bytes)

    @property
    def char_count(self):
        """The number of characters in the original value."""
        return len(self.value)

    @property
    def char_length(self):
        """The number of characters in the original value."""
        return len(self.value)

    @property
    def byte_count(self):
        """The number of bytes in the target encoding, before truncation."""
        return self.encoded_length

    @property
    def overflows(self):
        """Whether the encoded value exceeds the target width."""
        return self.width is not None and self.encoded_length > self.width

    @property
    def overflow(self):
        """Alias for ``overflows``."""
        return self.overflows

    @property
    def overflow_bytes(self):
        """The number of bytes by which the encoded value exceeds the width."""
        if not self.overflows:
            return 0
        return self.encoded_length - self.width

    @property
    def fits_by_characters(self):
        """Whether the character count alone would fit the byte width."""
        return self.width is None or self.char_count <= self.width

    @property
    def fits_by_bytes(self):
        """Whether the encoded byte count fits the width."""
        return self.width is None or self.encoded_length <= self.width

    @property
    def character_only_overflow(self):
        """Whether characters fit but encoded bytes do not."""
        return self.fits_by_characters and not self.fits_by_bytes

    @property
    def output_char_count(self):
        """The number of characters after truncation."""
        if self.truncated_value is None:
            return 0
        return len(self.truncated_value)

    @property
    def removed_bytes(self):
        """The number of bytes removed by truncation."""
        if self.encoded_bytes is None:
            return 0
        return len(self.encoded_bytes) - len(self.truncated_bytes)

    @property
    def dropped_bytes(self):
        """The full encoded bytes removed by truncation."""
        if self.encoded_bytes is None:
            return b''
        return self.encoded_bytes[len(self.truncated_bytes):]

    @property
    def removed_characters(self):
        """The number of characters removed by truncation."""
        if self.truncated_value is None:
            return 0
        return len(self.value) - len(self.truncated_value)

    @property
    def truncated_char_count(self):
        """The number of characters removed by truncation."""
        return self.removed_characters

    @property
    def is_representable(self):
        """Whether the original value is representable in the target encoding."""
        return self.representable

    @property
    def is_source_representable(self):
        """Whether the original text is representable in the source encoding."""
        return self.source_representable

    @property
    def is_truncated(self):
        """Whether the encoded value was truncated."""
        return self.truncated

    @property
    def lossy(self):
        """Whether encoding, truncation, or substitution changed the value."""
        return (not self.representable) or (not self.source_representable) or self.truncated

    @property
    def ok(self):
        """Whether the value survives conversion unchanged."""
        return self.representable and self.source_representable and not self.truncated

    def as_dict(self):
        """Return a dictionary, retaining bytes objects."""
        return {
            'index': self.index,
            'position': self.position,
            'value': self.value,
            'original': self.value,
            'original_value': self.value,
            'original_bytes': self.original_bytes,
            'original_hex': self.original_bytes.hex(),
            'original_bytes_complete': self.original_bytes_complete,
            'original_error': self.original_error,
            'source_representable': self.source_representable,
            'source_encoding': self.source_encoding,
            'encoding': self.encoding,
            'target_encoding': self.encoding,
            'used_encoding': self.encoding,
            'width': self.width,
            'missing': self.missing,
            'representable': self.representable,
            'encoded_bytes': self.encoded_bytes,
            'encoded_hex': None if self.encoded_bytes is None else self.encoded_bytes.hex(),
            'encoded_length': self.encoded_length,
            'truncated': self.truncated,
            'truncated_bytes': self.truncated_bytes,
            'truncated_hex': self.truncated_bytes.hex(),
            'truncated_value': self.truncated_value,
            'stored_bytes': self.stored_bytes,
            'stored_hex': self.stored_bytes.hex(),
            'stored_value': self.stored_value,
            'char_count': self.char_count,
            'byte_count': self.byte_count,
            'overflow': self.overflows,
            'overflow_bytes': self.overflow_bytes,
            'character_only_overflow': self.character_only_overflow,
            'output_bytes': self.truncated_bytes,
            'output_value': self.truncated_value,
            'removed_bytes': self.removed_bytes,
            'removed_characters': self.removed_characters,
            'error': self.error,
            'errors': self.errors,
            'collision': self.collision,
        }

    def as_json(self):
        """Return a JSON-compatible dictionary."""
        data = self.as_dict()
        data['original_bytes'] = data['original_hex']
        data['encoded_bytes'] = data['encoded_hex']
        data['truncated_bytes'] = data['truncated_hex']
        data['stored_bytes'] = data['stored_hex']
        data['output_bytes'] = data['truncated_hex']
        return data

    def __getitem__(self, key):
        """Provide dictionary-style access to report fields."""
        if not isinstance(key, str):
            raise TypeError(f'Report keys must be strings, not {type(key).__name__}')
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)

    def __repr__(self):
        """REPL-format string."""
        fmt = (
            '{cls}(value={value!r}, encoding={encoding!r}, width={width!r}, '
            'representable={representable!r}, truncated={truncated!r})'
        )
        return fmt.format(cls=type(self).__name__, **self.as_dict())


@dataclasses.dataclass(eq=False)
class Collision:
    """
    Distinct values that converge to the same target value.
    """

    encoding: str
    target_bytes: bytes
    target_value: Optional[str]
    values: Tuple[str, ...]
    reports: Tuple[ValueEncodingReport, ...]
    stored_bytes: bytes = b''

    @property
    def originals(self):
        """The distinct original values."""
        return self.values

    @property
    def value(self):
        """The target value shared by the group."""
        return self.target_value

    @property
    def members(self):
        """The per-value reports in this collision group."""
        return self.reports

    @property
    def count(self):
        """The number of distinct original values."""
        return len(self.values)

    def __getitem__(self, key):
        """Provide dictionary-style access to collision fields."""
        return self.as_dict()[key]

    def __eq__(self, other):
        """Compare collisions by target bytes and original values."""
        if isinstance(other, Collision):
            return (self.target_bytes, self.values) == (other.target_bytes, other.values)
        return NotImplemented

    def __hash__(self):
        """Hash by target bytes and original values."""
        return hash((self.target_bytes, self.values))

    def as_dict(self):
        """Return a dictionary, retaining bytes objects."""
        return {
            'encoding': self.encoding,
            'target_bytes': self.target_bytes,
            'target_hex': self.target_bytes.hex(),
            'stored_bytes': self.stored_bytes,
            'stored_hex': self.stored_bytes.hex(),
            'target_value': self.target_value,
            'values': list(self.values),
            'count': self.count,
            'positions': [r.position for r in self.reports],
        }

    def as_json(self):
        """Return a JSON-compatible dictionary."""
        data = self.as_dict()
        data['target_bytes'] = data['target_hex']
        data['stored_bytes'] = data['stored_hex']
        return data

    def __iter__(self):
        """Iterate over the distinct original values."""
        return iter(self.values)

    def __len__(self):
        """The number of distinct original values."""
        return len(self.values)

    def __repr__(self):
        """REPL-format string."""
        return (
            f'{type(self).__name__}(target_value={self.target_value!r}, '
            f'values={self.values!r})'
        )


@dataclasses.dataclass
class EncodingReport:
    """
    The encoding results for one character field.
    """

    encoding: str
    width: Optional[int]
    source_encoding: str
    values: List[ValueEncodingReport] = dataclasses.field(default_factory=list)
    collisions: List[Collision] = dataclasses.field(default_factory=list)
    errors: str = 'strict'
    name: Optional[str] = None

    @property
    def entries(self):
        """The per-value reports."""
        return self.values

    @property
    def rows(self):
        """The per-value reports."""
        return self.values

    @property
    def per_value(self):
        """The per-value reports."""
        return self.values

    @property
    def target_encoding(self):
        """The target encoding."""
        return self.encoding

    @property
    def source(self):
        """The source encoding."""
        return self.source_encoding

    @property
    def width_bytes(self):
        """The field width in bytes."""
        return self.width

    @property
    def representable(self):
        """Whether every value is representable in the target encoding."""
        return all(v.representable for v in self.values)

    @property
    def source_representable(self):
        """Whether every value is representable in the source encoding."""
        return all(v.source_representable for v in self.values)

    @property
    def has_unrepresentable(self):
        """Whether any value is not representable."""
        return any(not v.representable for v in self.values)

    @property
    def has_source_unrepresentable(self):
        """Whether any original value is not representable in the source encoding."""
        return any(not v.source_representable for v in self.values)

    @property
    def has_truncation(self):
        """Whether any value was truncated."""
        return any(v.truncated for v in self.values)

    @property
    def has_collisions(self):
        """Whether any distinct values collide."""
        return bool(self.collisions)

    @property
    def truncated(self):
        """Whether any value was truncated."""
        return self.has_truncation

    @property
    def lossless(self):
        """Whether every value survives conversion unchanged."""
        return (
            self.representable
            and self.source_representable
            and not self.has_truncation
            and not self.has_collisions
        )

    @property
    def ok(self):
        """Alias for ``lossless``."""
        return self.lossless

    @property
    def unrepresentable(self):
        """The values that cannot be represented in the target encoding."""
        return [v for v in self.values if not v.representable]

    @property
    def not_representable(self):
        """Alias for ``unrepresentable``."""
        return self.unrepresentable

    @property
    def truncation(self):
        """The values affected by truncation."""
        return self.truncations

    @property
    def collision_groups(self):
        """The collision groups."""
        return self.collisions

    @property
    def truncations(self):
        """The values affected by truncation."""
        return [v for v in self.values if v.truncated]

    @property
    def representability(self):
        """Map each original value to whether it is representable."""
        return OrderedDict((v.value, v.representable) for v in self.values)

    def as_dict(self):
        """Return a dictionary, retaining bytes objects."""
        return {
            'name': self.name,
            'encoding': self.encoding,
            'target_encoding': self.encoding,
            'source_encoding': self.source_encoding,
            'width': self.width,
            'errors': self.errors,
            'values': [v.as_dict() for v in self.values],
            'collisions': [c.as_dict() for c in self.collisions],
            'summary': {
                'n_values': len(self.values),
                'n_representable': sum(v.representable for v in self.values),
                'n_unrepresentable': len(self.unrepresentable),
                'n_source_unrepresentable': sum(
                    not v.source_representable for v in self.values
                ),
                'n_truncated': len(self.truncations),
                'n_collisions': len(self.collisions),
                'lossless': self.lossless,
            },
        }

    def as_json(self):
        """Return a JSON-compatible dictionary."""
        return {
            'name': self.name,
            'encoding': self.encoding,
            'target_encoding': self.encoding,
            'source_encoding': self.source_encoding,
            'width': self.width,
            'errors': self.errors,
            'values': [v.as_json() for v in self.values],
            'collisions': [c.as_json() for c in self.collisions],
            'summary': self.as_dict()['summary'],
        }

    def format(self, show_bytes=True):
        """Render a human-readable preview of the conversion."""
        width = 'unbounded' if self.width is None else str(self.width)
        lines = [
            f'Character encoding report: {self.source_encoding} -> {self.encoding}, '
            f'width={width} bytes'
        ]
        for v in self.values:
            original = v.original_bytes.hex() if show_bytes else ''
            if v.missing:
                status = 'missing'
            elif not v.source_representable:
                status = f'SOURCE NOT REPRESENTABLE ({v.original_error})'
            elif not v.representable:
                status = f'NOT REPRESENTABLE ({v.error})'
            elif v.truncated:
                status = (
                    f'truncated -> {v.truncated_value!r} '
                    f'({len(v.truncated_bytes)} bytes, removed {v.removed_bytes})'
                )
            else:
                status = 'ok'
            lines.append(
                f'[{v.position}] {v.value!r} original_bytes={original} '
                f'original_encoding={v.source_encoding} bytes={v.original_length} '
                f'-> encoding={v.encoding} status={status}'
            )
        if self.collisions:
            lines.append('Collisions:')
            for c in self.collisions:
                lines.append(
                    f'  {c.target_value!r} ({c.target_bytes.hex()}): '
                    f'{list(c.values)!r}'
                )
        return '\n'.join(lines)

    def __str__(self):
        """Human-readable conversion preview."""
        return self.format()

    def __iter__(self):
        """Iterate over per-value reports."""
        return iter(self.values)

    def __len__(self):
        """The number of values."""
        return len(self.values)

    def __getitem__(self, key):
        """Index a value or read a report field by name."""
        if isinstance(key, str):
            try:
                return getattr(self, key)
            except AttributeError:
                raise KeyError(key)
        return self.values[key]

    def __repr__(self):
        """REPL-format string."""
        fmt = (
            '{cls}(encoding={encoding!r}, width={width!r}, n_values={n}, '
            'n_collisions={n_collisions})'
        )
        return fmt.format(
            cls=type(self).__name__,
            encoding=self.encoding,
            width=self.width,
            n=len(self.values),
            n_collisions=len(self.collisions),
        )

    def require(
        self,
        allow_truncation=False,
        allow_unrepresentable=False,
        allow_collisions=False,
    ):
        """
        Return self if the conversion is acceptable, else raise ``ValueError``.

        This is a convenience for the common pattern of refusing to write
        character data that would be silently altered.
        """
        problems = []
        if self.collisions and not allow_collisions:
            problems.append(f'{len(self.collisions)} collision(s)')
        if self.has_truncation and not allow_truncation:
            problems.append(f'{len(self.truncations)} truncated value(s)')
        if self.has_unrepresentable and not allow_unrepresentable:
            problems.append(f'{len(self.unrepresentable)} unrepresentable value(s)')
        if problems:
            raise ValueError(
                f'Character encoding {self.source_encoding} -> {self.encoding} '
                f'has ' + ', '.join(problems)
            )
        return self


class DatasetEncodingReport(Mapping):
    """
    Encoding reports for each character field in a dataset.
    """

    def __init__(self, reports, encoding, source_encoding):
        self.reports = OrderedDict(reports)
        self.encoding = encoding
        self.source_encoding = source_encoding

    @property
    def target_encoding(self):
        """The target encoding."""
        return self.encoding

    @property
    def collisions(self):
        """Map each field name to its collision groups."""
        return OrderedDict(
            (name, report.collisions)
            for name, report in self.reports.items()
            if report.collisions
        )

    @property
    def all_collisions(self):
        """All collision groups, with the field name attached."""
        return [
            (name, collision)
            for name, report in self.reports.items()
            for collision in report.collisions
        ]

    @property
    def has_collisions(self):
        """Whether any field has collisions."""
        return any(report.has_collisions for report in self.reports.values())

    @property
    def has_truncation(self):
        """Whether any field has truncated values."""
        return any(report.has_truncation for report in self.reports.values())

    @property
    def has_unrepresentable(self):
        """Whether any field has unrepresentable values."""
        return any(report.has_unrepresentable for report in self.reports.values())

    @property
    def values(self):
        """The per-value reports across all fields."""
        return [
            (name, entry)
            for name, report in self.reports.items()
            for entry in report.values
        ]

    @property
    def truncated(self):
        """Whether any field has truncated values."""
        return any(report.has_truncation for report in self.reports.values())

    @property
    def unrepresentable(self):
        """The per-value reports that cannot be represented."""
        return [
            (name, entry)
            for name, report in self.reports.items()
            for entry in report.unrepresentable
        ]

    @property
    def lossless(self):
        """Whether every field survives conversion unchanged."""
        return all(report.lossless for report in self.reports.values())

    @property
    def ok(self):
        """Alias for ``lossless``."""
        return self.lossless

    def as_dict(self):
        """Return a dictionary, retaining bytes objects."""
        return {
            'encoding': self.encoding,
            'target_encoding': self.encoding,
            'source_encoding': self.source_encoding,
            'reports': OrderedDict((k, v.as_dict()) for k, v in self.reports.items()),
            'summary': {
                'n_fields': len(self.reports),
                'n_values': sum(len(v) for v in self.reports.values()),
                'n_reports': sum(len(v) for v in self.reports.values()),
                'n_truncated': sum(len(v.truncations) for v in self.reports.values()),
                'n_unrepresentable': sum(len(v.unrepresentable) for v in self.reports.values()),
                'n_collisions': len(self.all_collisions),
                'lossless': self.lossless,
            },
        }

    def as_json(self):
        """Return a JSON-compatible dictionary."""
        return {
            'encoding': self.encoding,
            'target_encoding': self.encoding,
            'source_encoding': self.source_encoding,
            'reports': OrderedDict((k, v.as_json()) for k, v in self.reports.items()),
            'summary': self.as_dict()['summary'],
        }

    def format(self, show_bytes=True):
        """Render a human-readable preview of all fields."""
        lines = [
            f'Dataset character encoding report: '
            f'{self.source_encoding} -> {self.encoding}'
        ]
        for name, report in self.reports.items():
            lines.append(f'Field {name}:')
            lines.extend('  ' + line for line in report.format(show_bytes).splitlines())
        return '\n'.join(lines)

    def __str__(self):
        """Human-readable conversion preview."""
        return self.format()

    def __getitem__(self, name):
        """Get a field report."""
        return self.reports[name]

    def __iter__(self):
        """Iterate over field names."""
        return iter(self.reports)

    def __len__(self):
        """The number of fields."""
        return len(self.reports)

    def require(
        self,
        allow_truncation=False,
        allow_unrepresentable=False,
        allow_collisions=False,
    ):
        """
        Return self if every field is acceptable, else raise ``ValueError``.
        """
        for name, report in self.reports.items():
            try:
                report.require(
                    allow_truncation=allow_truncation,
                    allow_unrepresentable=allow_unrepresentable,
                    allow_collisions=allow_collisions,
                )
            except ValueError as exc:
                raise ValueError(f'Field {name!r}: {exc}') from None
        return self

    def __repr__(self):
        """REPL-format string."""
        return (
            f'{type(self).__name__}(encoding={self.encoding!r}, '
            f'fields={list(self.reports)!r})'
        )


class LibraryEncodingReport(Mapping):
    """
    Encoding reports for each dataset in a library.
    """

    def __init__(self, reports, encoding, source_encoding):
        self.reports = OrderedDict(reports)
        self.encoding = encoding
        self.source_encoding = source_encoding

    @property
    def target_encoding(self):
        """The target encoding."""
        return self.encoding

    @property
    def all_collisions(self):
        """All collision groups, with dataset and field names attached."""
        return [
            (dataset, field, collision)
            for dataset, report in self.reports.items()
            for field, collision in report.all_collisions
        ]

    @property
    def lossless(self):
        """Whether every dataset survives conversion unchanged."""
        return all(report.lossless for report in self.reports.values())

    @property
    def ok(self):
        """Alias for ``lossless``."""
        return self.lossless

    def as_dict(self):
        """Return a dictionary, retaining bytes objects."""
        return {
            'encoding': self.encoding,
            'target_encoding': self.encoding,
            'source_encoding': self.source_encoding,
            'reports': OrderedDict((k, v.as_dict()) for k, v in self.reports.items()),
            'summary': {
                'n_datasets': len(self.reports),
                'n_fields': sum(len(v) for v in self.reports.values()),
                'n_collisions': len(self.all_collisions),
                'lossless': self.lossless,
            },
        }

    def as_json(self):
        """Return a JSON-compatible dictionary."""
        return {
            'encoding': self.encoding,
            'target_encoding': self.encoding,
            'source_encoding': self.source_encoding,
            'reports': OrderedDict((k, v.as_json()) for k, v in self.reports.items()),
            'summary': self.as_dict()['summary'],
        }

    def format(self, show_bytes=True):
        """Render a human-readable preview of all datasets."""
        lines = [
            f'Library character encoding report: '
            f'{self.source_encoding} -> {self.encoding}'
        ]
        for name, report in self.reports.items():
            lines.append(f'Dataset {name}:')
            lines.extend('  ' + line for line in report.format(show_bytes).splitlines())
        return '\n'.join(lines)

    def __str__(self):
        """Human-readable conversion preview."""
        return self.format()

    def __getitem__(self, name):
        """Get a dataset report."""
        return self.reports[name]

    def __iter__(self):
        """Iterate over dataset names."""
        return iter(self.reports)

    def __len__(self):
        """The number of datasets."""
        return len(self.reports)

    def require(
        self,
        allow_truncation=False,
        allow_unrepresentable=False,
        allow_collisions=False,
    ):
        """
        Return self if every dataset is acceptable, else raise ``ValueError``.
        """
        for name, report in self.reports.items():
            try:
                report.require(
                    allow_truncation=allow_truncation,
                    allow_unrepresentable=allow_unrepresentable,
                    allow_collisions=allow_collisions,
                )
            except ValueError as exc:
                raise ValueError(f'Dataset {name!r}: {exc}') from None
        return self

    def __eq__(self, other):
        """Compare equality."""
        if isinstance(other, LibraryEncodingReport):
            return self.reports == other.reports
        return NotImplemented

    def __repr__(self):
        """REPL-format string."""
        return (
            f'{type(self).__name__}(encoding={self.encoding!r}, '
            f'datasets={list(self.reports)!r})'
        )


def _original_bytes(value, source_encoding):
    """
    Return the original text and bytes for a value.
    """
    missing = _is_missing(value)
    if missing:
        return '', b'', True, None, True
    if isinstance(value, (bytes, bytearray, memoryview)):
        original = bytes(value)
        try:
            text = original.decode(source_encoding)
        except UnicodeDecodeError as exc:
            text = original.decode(source_encoding, errors='surrogateescape')
            return text, original, False, str(exc), False
        return text, original, True, None, True
    text = str(value)
    try:
        original = text.encode(source_encoding)
    except UnicodeEncodeError as exc:
        return text, b'', False, str(exc), False
    return text, original, True, None, True


def _analyze_value(value, position, index, source_encoding, encoding, width, errors):
    """
    Analyze one value's encoding and truncation.
    """
    text, original, source_representable, original_error, complete = _original_bytes(
        value, source_encoding
    )
    missing = _is_missing(value)

    encoded = None
    error = None
    representable = True
    if not missing:
        try:
            encoded = text.encode(encoding, errors='strict')
        except UnicodeEncodeError as exc:
            error = str(exc)
            representable = False
            if errors != 'strict':
                encoded = text.encode(encoding, errors=errors)

    if encoded is None:
        output = b''
        truncated_value = None
        truncated = False
    else:
        output = encoded
        if width is not None and len(output) > width:
            output = truncate_bytes(output, width, encoding)
        truncated = len(output) < len(encoded)
        try:
            truncated_value = output.decode(encoding)
        except UnicodeDecodeError:
            truncated_value = output.decode(encoding, errors='replace')

    return ValueEncodingReport(
        value=text,
        original_bytes=original,
        source_encoding=source_encoding,
        encoding=encoding,
        width=width,
        representable=representable,
        encoded_bytes=encoded,
        truncated=truncated,
        truncated_bytes=output,
        truncated_value=truncated_value,
        error=error,
        errors=errors,
        index=index,
        position=position,
        missing=missing,
        source_representable=source_representable,
        original_bytes_complete=complete,
        original_error=original_error,
        source_value=value,
    )


def _decoded_field_value(bytestring, encoding):
    """
    Decode stored field bytes the way an XPT reader would.
    """
    try:
        return bytestring.decode(encoding).rstrip()
    except (UnicodeDecodeError, LookupError):
        return None


def _collision_key(value):
    """
    Return a hashable identity for a source value.

    Missing values are canonicalized so two NaNs count as one value, while
    ``None`` and ``''`` remain distinct source values.
    """
    if _is_missing(value):
        return ('missing',)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return ('bytes', bytes(value))
    return ('text', str(value))


def _find_collisions(entries, encoding):
    """
    Find distinct values that map to the same stored target value.

    Grouping uses the fixed-width stored bytes, so trailing blanks that an
    XPT reader discards still count as convergence.
    """
    groups = OrderedDict()
    for entry in entries:
        if entry.encoded_bytes is None and not entry.missing:
            continue
        groups.setdefault(entry.stored_bytes, []).append(entry)

    collisions = []
    for stored_bytes, members in groups.items():
        first = OrderedDict()
        for member in members:
            first.setdefault(_collision_key(member.source_value), member)
        if len(first) > 1:
            collisions.append(
                Collision(
                    encoding=encoding,
                    target_bytes=members[0].truncated_bytes,
                    target_value=_decoded_field_value(stored_bytes, encoding),
                    values=tuple(m.source_value for m in first.values()),
                    reports=tuple(first.values()),
                    stored_bytes=stored_bytes,
                )
            )
    for i, collision in enumerate(collisions):
        for member in collision.reports:
            member.collision = i
    return collisions


def _as_list(values):
    """
    Materialize an iterable of values, treating scalars as one value.
    """
    if isinstance(values, (str, bytes, bytearray, memoryview)) or _is_missing(values):
        return [values]
    if isinstance(values, pd.Series):
        return values.tolist()
    if isinstance(values, Iterable):
        return list(values)
    return [values]


def encoding_report(
    values,
    encoding=None,
    width=None,
    source_encoding=None,
    errors='strict',
    target_encoding=None,
    target_width=None,
    index=None,
):
    """
    Report the effect of encoding a character field to ``encoding``.

    ``values`` may be a sequence, a Pandas Series, a Dataset, or a single
    value.  ``width`` is the target width in bytes.  Each value's original
    bytes are retained, truncation is performed only at character
    boundaries, and distinct values that converge on the same target
    bytes are listed as collisions.
    """
    if isinstance(values, pd.DataFrame):
        return dataset_encoding_report(
            values,
            encoding=encoding,
            widths=width if width is not None else target_width,
            source_encoding=source_encoding,
            errors=errors,
        )
    source_encoding = source_encoding or DEFAULT_SOURCE_ENCODING
    encoding = encoding or target_encoding or source_encoding
    width = width if width is not None else target_width
    if width is not None:
        width = int(width)
        if width < 0:
            raise ValueError(f'Width must be non-negative, got {width}')
    codecs.lookup(source_encoding)
    codecs.lookup(encoding)
    codecs.lookup_error(errors)

    series_index = None
    if isinstance(values, pd.Series):
        if index is None:
            series_index = list(values.index)
        raw_values = values.tolist()
    else:
        raw_values = _as_list(values)

    if index is not None:
        if isinstance(index, (str, bytes)):
            index_values = [index] * len(raw_values)
        else:
            index_values = list(index)
        if len(index_values) != len(raw_values):
            raise ValueError('Index length must match values length')
    elif series_index is not None:
        index_values = series_index
    else:
        index_values = list(range(len(raw_values)))

    entries = [
        _analyze_value(
            value,
            position,
            index_values[position],
            source_encoding,
            encoding,
            width,
            errors,
        )
        for position, value in enumerate(raw_values)
    ]
    collisions = _find_collisions(entries, encoding)
    return EncodingReport(
        encoding=encoding,
        width=width,
        source_encoding=source_encoding,
        values=entries,
        collisions=collisions,
        errors=errors,
    )


def _is_character_variable(series):
    """
    Return whether a column holds character values.
    """
    vtype = getattr(series, 'vtype', None)
    if vtype is not None:
        return vtype == 2
    dtype = getattr(series, 'dtype', None)
    kind = getattr(dtype, 'kind', None)
    return kind in {'O', 'U', 'S'}


def dataset_encoding_report(
    dataset,
    encoding=None,
    widths=None,
    source_encoding=None,
    errors='strict',
    include_numeric=False,
):
    """
    Report the effect of encoding every character column in a dataset.
    """
    source_encoding = source_encoding or DEFAULT_SOURCE_ENCODING
    encoding = encoding or source_encoding
    reports = OrderedDict()
    for name, series in dataset.items():
        if not include_numeric and not _is_character_variable(series):
            continue
        if isinstance(widths, Mapping):
            width = widths.get(name)
        elif widths is not None:
            width = widths
        else:
            width = getattr(series, 'width', None)
        report = encoding_report(
            series,
            encoding=encoding,
            width=width,
            source_encoding=source_encoding,
            errors=errors,
        )
        report.name = name
        reports[name] = report
    return DatasetEncodingReport(reports, encoding=encoding, source_encoding=source_encoding)


def library_encoding_report(
    library,
    encoding=None,
    widths=None,
    source_encoding=None,
    errors='strict',
    include_numeric=False,
):
    """
    Report the effect of encoding every character field in a library.
    """
    source_encoding = source_encoding or DEFAULT_SOURCE_ENCODING
    encoding = encoding or source_encoding
    reports = OrderedDict()
    for name, dataset in library.items():
        if isinstance(widths, Mapping):
            try:
                dataset_widths = widths[name]
            except KeyError:
                dataset_widths = None
            if isinstance(dataset_widths, Mapping):
                dataset_width = dataset_widths
            else:
                dataset_width = widths.get(None)
        else:
            dataset_width = widths
        reports[name] = dataset_encoding_report(
            dataset,
            encoding=encoding,
            widths=dataset_width,
            source_encoding=source_encoding,
            errors=errors,
            include_numeric=include_numeric,
        )
    return LibraryEncodingReport(reports, encoding=encoding, source_encoding=source_encoding)


analyze_encoding = encoding_report
audit_encoding = encoding_report
check_encoding = encoding_report
preflight_encoding = encoding_report
preview_encoding = encoding_report
report_encoding = encoding_report
transcode_report = encoding_report
