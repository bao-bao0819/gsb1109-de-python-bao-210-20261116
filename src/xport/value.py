"""
SAS custom formats that map numeric intervals to labels.

A SAS ``VALUE`` format assigns a label to a range of a numeric variable.  Each
range has two endpoints and each endpoint can be inclusive or exclusive, so the
rules form a set of intervals rather than a simple binning.  This module parses
such a format, applies it to a numeric variable, and records which rule produced
each result.  It also reports rules that overlap and values that no rule covers.

The parser is independent of any file format: a format can come from a ``.sas``
file, a catalog export, or a table in a data dictionary.

The grouped view keeps one row per input value.  It never merges rules that
share a display label, because two different intervals can print the same label.
Each row cites its rule number, source line, rule text, and interval in
canonical form.  That boundary evidence survives even when labels repeat.
"""

# Standard Library
import re
from collections import namedtuple
from decimal import Decimal, InvalidOperation

# Community Packages
import pandas as pd

__all__ = [
    'FormatError',
    'GroupRow',
    'Grouped',
    'IntervalRule',
    'Overlap',
    'ValueFormat',
]

# SAS special missing values are written as .A through .Z and ._.
MISSING_TAGS = '_' + 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

_NEGATIVE_INFINITY = Decimal('-Infinity')
_POSITIVE_INFINITY = Decimal('Infinity')

_NUMBER = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
_SPECIAL = r'\.[A-Za-z_]?'
_KEYWORD = r'(?:LOW|MIN|HIGH|MAX)'
_ENDPOINT = rf'(?:{_NUMBER}|{_SPECIAL}|{_KEYWORD})'
_SEPARATOR = r'(?:<-<|<-|-<|-)'
_RANGE = rf'(?:{_ENDPOINT}(?:\s*{_SEPARATOR}\s*{_ENDPOINT})?|OTHER)'
_SINGLE_QUOTED = r"'(?:[^']|'')*'"
_DOUBLE_QUOTED = r'"(?:[^"]|"")*"'
_LABEL = rf'(?:{_SINGLE_QUOTED}|{_DOUBLE_QUOTED}|[^\s;]+)'

_RULE = re.compile(rf'(?P<range>{_RANGE})\s*=\s*(?P<label>{_LABEL})', re.IGNORECASE)
_HEADER = re.compile(
    r'^\s*VALUE\s+(?P<name>\$?[A-Za-z_]\w*)(?:\s*\([^)]*\))?',
    re.IGNORECASE,
)
_RANGE_PARTS = re.compile(
    rf'^(?P<low>{_ENDPOINT})\s*(?P<sep>{_SEPARATOR})\s*(?P<high>{_ENDPOINT})$',
    re.IGNORECASE,
)
_SPECIAL_TOKEN = re.compile(rf'^{_SPECIAL}$')
_SINGLE_TOKEN = re.compile(rf'^(?:{_NUMBER}|{_SPECIAL})$', re.IGNORECASE)


class FormatError(ValueError):
    """A SAS value-format definition could not be parsed or is inconsistent."""


def _normalize_tag(token):
    """
    Normalize a special-missing token: uppercase the tag, keep the leading dot.
    """
    if token == '.':
        return '.'
    return '.' + token[1:].upper()


def _as_decimal(text):
    """
    Convert a numeric token to ``Decimal`` without binary rounding.
    """
    try:
        value = Decimal(text)
    except InvalidOperation:
        raise FormatError(f'Not a numeric endpoint: {text!r}') from None
    if not value.is_finite():
        raise FormatError(f'Not a finite numeric endpoint: {text!r}')
    return value


def _render_number(value):
    """
    Render a ``Decimal`` without exponent notation or trailing zeros.
    """
    if value == 0:
        return '0'
    text = format(value.normalize(), 'f')
    if '.' in text:
        text = text.rstrip('0').rstrip('.')
    return text


class IntervalRule:
    """
    One line of a SAS ``VALUE`` format.

    A numeric rule is an interval.  ``kind`` is ``'numeric'``, ``'missing'``
    for a special missing value such as ``.A``, or ``'other'`` for the SAS
    ``OTHER`` fallback.  ``number`` is the one-based position within the format
    and is stable when a definition is edited in place; ``line`` is the source
    line, kept for citation.  ``source`` is the original rule text.
    """

    __slots__ = (
        'number',
        'line',
        'kind',
        'low',
        'low_inclusive',
        'high',
        'high_inclusive',
        'tag',
        'label',
        'source',
    )

    def __init__(
        self,
        number,
        label,
        source,
        kind='numeric',
        low=None,
        low_inclusive=True,
        high=None,
        high_inclusive=True,
        tag=None,
        line=None,
    ):
        """
        Initialize one rule.
        """
        self.number = number
        self.line = number if line is None else line
        self.kind = kind
        self.low = low
        self.low_inclusive = low_inclusive if low is not None else True
        self.high = high
        self.high_inclusive = high_inclusive if high is not None else True
        self.tag = tag
        self.label = label
        self.source = source
        if kind == 'numeric':
            if low is None or high is None:
                raise FormatError(f'Numeric rule needs two endpoints: {source!r}')
            if low > high:
                raise FormatError(f'Low endpoint exceeds high endpoint: {source!r}')
            if low == high and not (low_inclusive and high_inclusive):
                raise FormatError(f'Empty numeric interval: {source!r}')

    def __repr__(self):
        """REPL-format."""
        return f'IntervalRule({self.number}, {self.label!r}, {self.evidence})'

    def __eq__(self, other):
        """Equality includes every attribute."""
        if not isinstance(other, IntervalRule):
            return NotImplemented
        return all(getattr(self, name) == getattr(other, name) for name in self.__slots__)

    def __hash__(self):
        """Hash by all attributes."""
        return hash(tuple(getattr(self, name) for name in self.__slots__))

    def covers(self, value):
        """
        Test whether a numeric ``Decimal`` falls in this rule.
        """
        if self.kind != 'numeric' or value is None:
            return False
        if value < self.low or value > self.high:
            return False
        if value == self.low and not self.low_inclusive:
            return False
        if value == self.high and not self.high_inclusive:
            return False
        return True

    def same_interval(self, other):
        """
        Compare only the cover, ignoring label, source, and citation.
        """
        if self.kind != other.kind:
            return False
        if self.kind == 'numeric':
            return (
                self.low == other.low and self.low_inclusive == other.low_inclusive
                and self.high == other.high and self.high_inclusive == other.high_inclusive
            )
        if self.kind == 'missing':
            return self.tag == other.tag
        return True

    @property
    def evidence(self):
        """
        Canonical interval text, independent of the display label.

        For example ``[0, 18)`` says the low endpoint is included and the high
        endpoint is excluded.  Missing rules cite the missing token and the
        ``OTHER`` rule cites ``OTHER``.
        """
        if self.kind == 'missing':
            return self.tag
        if self.kind == 'other':
            return 'OTHER'
        low = 'LOW' if self.low == _NEGATIVE_INFINITY else _render_number(self.low)
        high = 'HIGH' if self.high == _POSITIVE_INFINITY else _render_number(self.high)
        left = '[' if self.low_inclusive else '('
        right = ']' if self.high_inclusive else ')'
        return f'{left}{low}, {high}{right}'


Overlap = namedtuple('Overlap', 'left right low low_inclusive high high_inclusive')
Overlap.__doc__ = """
Two rules that share at least one value.

The intersection is described by ``low``, ``high``, and the two inclusive
flags.  The pair is kept so callers can inspect both rules and decide how to
resolve the ambiguity that first-match-wins hides.
"""

Diagnostics = namedtuple('Diagnostics', 'overlaps uncovered gaps')
Diagnostics.__doc__ = """
Result of :meth:`ValueFormat.diagnose`.

``overlaps`` holds :class:`Overlap` pairs.  ``uncovered`` holds grouped rows
with no rule.  ``gaps`` holds numeric ranges without a rule.
"""

GroupRow = namedtuple(
    'GroupRow',
    [
        'position',
        'value',
        'token',
        'label',
        'rule_number',
        'line',
        'rule_text',
        'interval',
        'matches',
        'missing',
        'numeric',
        'source',
    ],
)
GroupRow.__doc__ = """
One value in a grouped view, with the rule that produced its label.

``matches`` lists every explicit rule that contains the value, so a row with
more than one entry exposes an overlap instead of hiding it behind the
first-match-wins assignment.  ``numeric`` is the exact ``Decimal`` used for
interval tests, and ``token`` is the missing-value token when ``missing`` is
true.  ``rule_number`` is the one-based rule position and ``line`` is the
source line of that rule, so a result can be traced back to the definition.
"""


class ValueFormat:
    """
    A SAS ``VALUE`` format mapping numeric ranges to labels.

    Build one by parsing the body of a ``VALUE`` statement::

        fmt = ValueFormat.parse('''
            0 -< 18 = 'Under 18'
            18 -< 65 = '18 to 64'
            65 - HIGH = '65 and over'
        ''')

    Rules are kept in source order.  A value is assigned to the first rule that
    contains it, but every matching rule is still reported, so overlaps are not
    silently hidden by that choice.  The ``OTHER`` rule is a fallback for values
    no explicit rule matches, following SAS semantics.
    """

    def __init__(self, rules, name=None):
        """
        Initialize from an iterable of :class:`IntervalRule` objects.
        """
        self.name = name
        self.rules = tuple(rules)

    @classmethod
    def parse(cls, text, name=None):
        """
        Parse a SAS ``VALUE`` format definition.

        The input may include a ``VALUE`` statement, trailing semicolons, and
        ``/* ... */`` comments.  Rules may share a line or span several lines.
        Labels may be single-quoted, double-quoted, or bare.
        """
        statements = _statements(text)
        if not statements:
            raise FormatError('Value format contains no rules')

        rules = []
        names = []
        for statement, offset in statements:
            body = statement
            base = 0
            header = _HEADER.match(body)
            if header is not None:
                names.append(header['name'])
                body = body[header.end():]
                base = header.end()
            matches = list(_RULE.finditer(body))
            if not matches:
                if body.strip():
                    raise FormatError(f'No rule found in: {statement.strip()!r}')
                continue
            residual = _RULE.sub('', body)
            if residual.strip():
                raise FormatError(f'Unrecognized text in value format: {residual.strip()!r}')
            for match in matches:
                position = offset + base + match.start()
                line = text.count('\n', 0, position) + 1
                rules.append(_make_rule(match, len(rules) + 1, line))
        if not rules:
            raise FormatError('Value format contains no rules')
        if len(set(names)) > 1:
            raise FormatError(
                f'Expected one VALUE definition, found {sorted(set(names))}; parse each separately'
            )
        if name is None and names:
            name = names[0]
        return cls(rules, name=name)

    @classmethod
    def from_table(
        cls,
        rows,
        name=None,
        low='low',
        high='high',
        label='label',
        low_inclusive='low_inclusive',
        high_inclusive='high_inclusive',
        source=None,
        missing='missing',
    ):
        """
        Build from a table of intervals, such as a data dictionary export.

        ``rows`` may be an iterable of mappings or a Pandas ``DataFrame``.  Each
        row supplies a low and high endpoint and a label.  The inclusivity
        columns may instead be written as ``low_op``/``high_op`` holding
        ``'<'`` or ``'<='``.  A row whose endpoints are both absent, or whose
        ``missing`` column names a token, becomes a special-missing rule;
        ``OTHER`` in either endpoint becomes the fallback rule.  Rows are
        numbered in input order.
        """
        if isinstance(rows, pd.DataFrame):
            records = rows.to_dict('records')
        else:
            records = [dict(row) for row in rows]

        rules = []
        for number, record in enumerate(records, 1):
            text = _table_source(record, low, high, label, source)
            label_text = record.get(label)
            if label_text is None:
                raise FormatError(f'Row {number} has no label: {record!r}')
            low_text = _cell(record, low)
            high_text = _cell(record, high)
            missing_text = _cell(record, missing)
            if missing_text is not None:
                token = str(missing_text).strip()
                if token in {'', '.', 'missing', 'MISSING'}:
                    tag = '.'
                elif _SPECIAL_TOKEN.match(token):
                    tag = _normalize_tag(token)
                elif len(token) == 1 and token.upper() in MISSING_TAGS:
                    tag = '.' + token.upper()
                else:
                    raise FormatError(
                        f'Row {number} has an invalid missing token: {missing_text!r}'
                    )
                rules.append(IntervalRule(
                    number,
                    str(label_text),
                    text,
                    kind='missing',
                    tag=tag,
                    line=number,
                ))
                continue
            if low_text is None and high_text is None:
                rules.append(IntervalRule(
                    number,
                    str(label_text),
                    text,
                    kind='missing',
                    tag='.',
                    line=number,
                ))
                continue
            if str(low_text).upper() == 'OTHER' or str(high_text).upper() == 'OTHER':
                rules.append(
                    IntervalRule(number, str(label_text), text, kind='other', line=number)
                )
                continue
            if _SPECIAL_TOKEN.match(str(low_text)) and (
                high_text is None or str(high_text) == str(low_text)
            ):
                rules.append(IntervalRule(
                    number,
                    str(label_text),
                    text,
                    kind='missing',
                    tag=_normalize_tag(str(low_text)),
                    line=number,
                ))
                continue
            if high_text is None:
                raise FormatError(f'Row {number} has a low endpoint but no high endpoint')
            low_value = _endpoint(str(low_text), lower=True, line=number)
            high_value = _endpoint(str(high_text), lower=False, line=number)
            rules.append(IntervalRule(
                number,
                str(label_text),
                text,
                low=low_value,
                low_inclusive=_inclusive(record, low_inclusive, default=True),
                high=high_value,
                high_inclusive=_inclusive(record, high_inclusive, default=True),
                line=number,
            ))
        if not rules:
            raise FormatError('Table contains no rules')
        return cls(rules, name=name)

    def __repr__(self):
        """REPL-format."""
        return f'ValueFormat({self.name!r}, rules={len(self.rules)})'

    def __len__(self):
        """Number of rules."""
        return len(self.rules)

    def __iter__(self):
        """Iterate rules in source order."""
        return iter(self.rules)

    def match_rules(self, value):
        """
        Return every explicit rule that contains ``value``, in source order.

        ``OTHER`` is not included; it is a fallback rather than a real match.
        """
        token, numeric, missing = _coerce(value)
        matches = []
        for rule in self.rules:
            if rule.kind == 'numeric' and not missing and rule.covers(numeric):
                matches.append(rule)
            elif rule.kind == 'missing' and missing and rule.tag == token:
                matches.append(rule)
        return tuple(matches)

    def rule_for(self, value):
        """
        Return the rule SAS would use for ``value``, or ``None``.

        The first explicit match wins.  If nothing matches, the first ``OTHER``
        rule is used.  Special missing values use their own rule before
        ``OTHER``.
        """
        matches = self.match_rules(value)
        if matches:
            return matches[0]
        for rule in self.rules:
            if rule.kind == 'other':
                return rule
        return None

    def assign(self, value):
        """
        Return the label for one value, or ``None`` if it is uncovered.
        """
        rule = self.rule_for(value)
        return rule.label if rule is not None else None

    def group(self, values, source=None, index=None):
        """
        Build a grouped view of ``values`` with rule provenance.

        Each input value produces one :class:`GroupRow`.  Rows with the same
        label but different rules remain distinct; their ``rule_number`` and
        ``interval`` preserve the boundary evidence.
        """
        rows = [
            self._attribute(value, position, source)
            for position, value in enumerate(values)
        ]
        return Grouped(self, rows, source=source, index=index)

    def group_series(self, series, source=None):
        """
        Group a Pandas ``Series`` or plain sequence, keeping its index.
        """
        if source is None:
            source = getattr(series, 'name', None)
        index = getattr(series, 'index', None)
        return self.group(list(series), source=source, index=index)

    def group_variable(self, dataset, column, source=None):
        """
        Group one column of a Pandas data frame or ``xport.Dataset``.

        The returned frame keeps the dataset index and adds one row per value
        with the assigned label and the rule that produced it.
        """
        series = dataset[column]
        return self.group_series(series, source=source or column)

    def group_dataframe(self, dataset, column, source=None, prefix=''):
        """
        Attach a grouped view to the original column.

        Returns a new data frame with every original column plus the label and
        the rule provenance for ``column``.  The prefix keeps the added names
        distinct from the originals.
        """
        grouped = self.group_variable(dataset, column, source=source)
        frame = dataset.copy()
        provenance = grouped.to_frame()
        frame[prefix + 'label'] = provenance['label'].to_numpy()
        frame[prefix + 'rule_number'] = provenance['rule_number'].to_numpy()
        frame[prefix + 'interval'] = provenance['interval'].to_numpy()
        frame[prefix + 'rule_text'] = provenance['rule_text'].to_numpy()
        return frame

    def _attribute(self, value, position, source):
        """
        Attribute one value to a rule and build its provenance row.
        """
        token, numeric, missing = _coerce(value)
        matches = self.match_rules(value)
        rule = matches[0] if matches else None
        if rule is None:
            rule = next((r for r in self.rules if r.kind == 'other'), None)
        return GroupRow(
            position=position,
            value=value,
            token=token,
            label=rule.label if rule is not None else None,
            rule_number=rule.number if rule is not None else None,
            line=rule.line if rule is not None else None,
            rule_text=rule.source if rule is not None else None,
            interval=rule.evidence if rule is not None else None,
            matches=tuple(r.number for r in matches),
            missing=missing,
            numeric=numeric,
            source=source,
        )

    def overlaps(self):
        """
        Return every pair of explicit rules whose intervals intersect.

        Missing-value rules overlap only an identical missing token.  ``OTHER``
        is a fallback and is excluded, since it intentionally catches everything
        that no explicit rule covers.
        """
        pairs = []
        explicit = [r for r in self.rules if r.kind != 'other']
        for i, left in enumerate(explicit):
            for right in explicit[i + 1:]:
                overlap = _intersection(left, right)
                if overlap is not None:
                    pairs.append(overlap)
        return tuple(pairs)

    def gaps(self, bounds=None):
        """
        Return numeric ranges that no explicit rule covers.

        ``bounds`` is an optional ``(low, high)`` pair of numbers describing the
        domain of interest, for example the observed minimum and maximum.  The
        returned ranges are tuples ``(low, low_inclusive, high, high_inclusive)``
        using ``Decimal`` values.
        """
        numeric = sorted(
            (r for r in self.rules if r.kind == 'numeric'),
            key=lambda rule: (rule.low, not rule.low_inclusive, rule.high),
        )
        if not numeric:
            if bounds is None:
                return ()
            gap = _gap_tuple(_as_decimal(str(bounds[0])), True, _as_decimal(str(bounds[1])), True)
            return (gap,) if gap is not None else ()

        merged = []
        for rule in numeric:
            if not merged:
                merged.append(_merge_entry(rule))
                continue
            last = merged[-1]
            if rule.low < last[2]:
                _extend(last, rule)
            elif rule.low == last[2]:
                if rule.low_inclusive or last[3]:
                    _extend(last, rule)
                else:
                    merged.append(_merge_entry(rule))
            else:
                merged.append(_merge_entry(rule))

        gaps = []
        for left, right in zip(merged, merged[1:]):
            gap = _gap_tuple(left[2], not left[3], right[0], not right[1])
            if gap is not None:
                gaps.append(gap)
        if bounds is not None:
            low, high = (_as_decimal(str(bound)) for bound in bounds)
            first = merged[0]
            if low < first[0] or (low == first[0] and not first[1]):
                gap = _gap_tuple(low, True, first[0], not first[1])
                if gap is not None:
                    gaps.insert(0, gap)
            last = merged[-1]
            if high > last[2] or (high == last[2] and not last[3]):
                gap = _gap_tuple(last[2], not last[3], high, True)
                if gap is not None:
                    gaps.append(gap)
        return tuple(gaps)

    def describe_gaps(self, bounds=None):
        """
        Render :meth:`gaps` as readable interval text.
        """
        described = []
        for low, low_inclusive, high, high_inclusive in self.gaps(bounds):
            left = '[' if low_inclusive else '('
            right = ']' if high_inclusive else ')'
            described.append(f'{left}{_render_number(low)}, {_render_number(high)}{right}')
        return tuple(described)

    def diagnose(self, values=None):
        """
        Summarize rule overlap and uncovered values.

        Returns a :class:`Diagnostics` with ``overlaps``, ``uncovered`` rows
        (when ``values`` is given), and structural ``gaps`` bounded by the
        observed numeric minimum and maximum.
        """
        grouped = self.group(values) if values is not None else None
        bounds = None
        if grouped is not None:
            numeric = [row.numeric for row in grouped if row.numeric is not None]
            if numeric:
                bounds = (min(numeric), max(numeric))
        return Diagnostics(
            overlaps=self.overlaps(),
            uncovered=grouped.uncovered() if grouped is not None else (),
            gaps=self.gaps(bounds),
        )


class Grouped:
    """
    A per-value grouping with rule provenance.

    The object is an immutable sequence of :class:`GroupRow`.  It can be
    exported to a Pandas ``DataFrame``, summarized per rule, or reclassified
    against a modified format.
    """

    __slots__ = ('format', 'rows', 'source', 'index', 'reclassified')

    def __init__(self, fmt, rows, source=None, index=None, reclassified=()):
        """
        Initialize a grouped view.
        """
        self.format = fmt
        self.rows = tuple(rows)
        self.source = source
        self.index = pd.RangeIndex(len(self.rows)) if index is None else index
        self.reclassified = tuple(reclassified)

    def __len__(self):
        """Number of rows."""
        return len(self.rows)

    def __iter__(self):
        """Iterate rows."""
        return iter(self.rows)

    def __getitem__(self, index):
        """Index or slice rows."""
        return self.rows[index]

    def __repr__(self):
        """REPL-format."""
        return f'Grouped({len(self.rows)} rows, source={self.source!r})'

    def labels(self):
        """
        Labels in input order, with ``None`` for uncovered values.
        """
        return tuple(row.label for row in self.rows)

    def uncovered(self):
        """
        Rows that no explicit rule and no ``OTHER`` fallback covered.
        """
        return tuple(row for row in self.rows if row.label is None)

    def conflicts(self):
        """
        Rows contained in more than one explicit rule.
        """
        return tuple(row for row in self.rows if len(row.matches) > 1)

    def to_frame(self):
        """
        Export one row per value, preserving rule and interval columns.
        """
        columns = [
            'value',
            'label',
            'rule_number',
            'line',
            'interval',
            'rule_text',
            'matches',
            'missing',
            'source',
        ]
        data = {name: [getattr(row, name) for row in self.rows] for name in columns}
        frame = pd.DataFrame(data, columns=columns, index=self.index)
        for column in ('rule_number', 'line'):
            frame[column] = frame[column].astype(pd.Int64Dtype())
        return frame

    def summary(self):
        """
        Count rows per rule, one table row per rule.

        Rules that share a label are separate rows with separate intervals.
        Counting by label alone would discard exactly the boundary evidence
        this view exists to preserve.
        """
        counts = self.counts()
        columns = ['rule_number', 'line', 'label', 'interval', 'kind', 'source', 'count']
        data = [
            {
                'rule_number': rule.number,
                'line': rule.line,
                'label': rule.label,
                'interval': rule.evidence,
                'kind': rule.kind,
                'source': rule.source,
                'count': counts.get(rule.number, 0),
            } for rule in self.format.rules
        ]
        return pd.DataFrame(data, columns=columns)

    def by_label(self):
        """
        Map each label to every distinct rule that uses it.

        The value is a tuple with one entry per rule, in source order.  Repeated
        labels therefore keep all of their intervals instead of collapsing into
        one.
        """
        counts = self.counts()
        grouped = {}
        for rule in self.format.rules:
            grouped.setdefault(rule.label, []).append({
                'rule_number': rule.number,
                'line': rule.line,
                'interval': rule.evidence,
                'source': rule.source,
                'count': counts.get(rule.number, 0),
            })
        return {label: tuple(entries) for label, entries in grouped.items()}

    def counts(self):
        """
        Map rule number to the number of values assigned to it.
        """
        counts = {}
        for row in self.rows:
            if row.rule_number is not None:
                counts[row.rule_number] = counts.get(row.rule_number, 0) + 1
        return counts

    def reclassify(self, new_format):
        """
        Apply a modified format, re-evaluating only affected rows.

        Unaffected rows are reused as the same objects.  Rows whose rule only
        changed its display label are updated in place without re-evaluation.
        ``reclassified`` lists the positions that were re-evaluated, which makes
        the incremental behavior observable.

        A change that adds or removes rules, or introduces or removes an
        ``OTHER`` rule, falls back to a full reclassification because rule
        identity can no longer be matched one-to-one.
        """
        old_rules = {rule.number: rule for rule in self.format.rules}
        new_rules = {rule.number: rule for rule in new_format.rules}
        if set(old_rules) != set(new_rules):
            return self._reclassify_all(new_format)
        # ``OTHER`` has no interval to compare, so a change in which position
        # holds it cannot be localized.  Its rows may also be non-numeric.
        if any(
            (old.kind == 'other') != (new_rules[number].kind == 'other')
            for number, old in old_rules.items()
        ):
            return self._reclassify_all(new_format)

        dirty_regions = []
        dirty_tokens = set()
        stale_rules = set()
        for number, old in old_rules.items():
            new = new_rules[number]
            if old == new:
                continue
            stale_rules.add(number)
            if old.kind != new.kind:
                _collect_cover(old, dirty_regions, dirty_tokens)
                _collect_cover(new, dirty_regions, dirty_tokens)
            elif old.kind == 'numeric':
                dirty_regions.extend(_symmetric_difference(old, new))
            elif old.kind == 'missing' and old.tag != new.tag:
                dirty_tokens.update((old.tag, new.tag))

        rows = []
        reclassified = []
        for row in self.rows:
            affected = False
            if row.missing:
                affected = row.token in dirty_tokens
            elif row.numeric is not None:
                affected = any(
                    _region_covers(region, row.numeric) for region in dirty_regions
                )
            if affected:
                rows.append(new_format._attribute(row.value, row.position, self.source))
                reclassified.append(row.position)
            elif row.rule_number in stale_rules:
                rule = new_rules[row.rule_number]
                rows.append(
                    row._replace(
                        label=rule.label,
                        line=rule.line,
                        rule_text=rule.source,
                        interval=rule.evidence,
                    )
                )
            else:
                rows.append(row)
        return Grouped(
            new_format,
            rows,
            source=self.source,
            index=self.index,
            reclassified=reclassified,
        )

    def _reclassify_all(self, new_format):
        """
        Re-evaluate every row, used when rule identity cannot be matched.
        """
        rows = [
            new_format._attribute(row.value, row.position, self.source) for row in self.rows
        ]
        return Grouped(
            new_format,
            rows,
            source=self.source,
            index=self.index,
            reclassified=tuple(row.position for row in rows),
        )


def _collect_cover(rule, regions, tokens):
    """
    Record the part of the value space a rule covers.
    """
    if rule.kind == 'numeric':
        regions.append((rule.low, rule.low_inclusive, rule.high, rule.high_inclusive))
    elif rule.kind == 'missing':
        tokens.add(rule.tag)


def _symmetric_difference(old, new):
    """
    Return the parts of the value space where two numeric intervals differ.

    Only values in these regions can change classification when one interval is
    edited.  Restricting work to the difference keeps a widened or narrowed end
    from re-evaluating every value in the untouched part of the interval.
    """
    parts = []
    old_low, old_low_inc, old_high, old_high_inc = (
        old.low, old.low_inclusive, old.high, old.high_inclusive
    )
    new_low, new_low_inc, new_high, new_high_inc = (
        new.low, new.low_inclusive, new.high, new.high_inclusive
    )
    if new_low > old_low:
        parts.append((old_low, old_low_inc, new_low, not new_low_inc))
    elif new_low < old_low:
        parts.append((new_low, new_low_inc, old_low, not old_low_inc))
    elif old_low_inc != new_low_inc:
        parts.append((old_low, True, old_low, True))
    if new_high < old_high:
        parts.append((new_high, not new_high_inc, old_high, old_high_inc))
    elif new_high > old_high:
        parts.append((old_high, not old_high_inc, new_high, new_high_inc))
    elif old_high_inc != new_high_inc:
        parts.append((old_high, True, old_high, True))
    return [part for part in parts if _region_nonempty(part)]


def _region_nonempty(region):
    """
    Test whether a region tuple contains at least one value.
    """
    low, low_inclusive, high, high_inclusive = region
    if low > high:
        return False
    if low == high and not (low_inclusive and high_inclusive):
        return False
    return True


def _region_covers(region, value):
    """
    Test whether a numeric value falls in a region tuple.
    """
    low, low_inclusive, high, high_inclusive = region
    if value < low or value > high:
        return False
    if value == low and not low_inclusive:
        return False
    if value == high and not high_inclusive:
        return False
    return True


def _cell(record, key):
    """
    Read a table cell, treating blank and null values as absent.
    """
    value = record.get(key) if key is not None else None
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return value


def _inclusive(record, key, default=True):
    """
    Read an endpoint-inclusivity cell, accepting booleans or ``<``/``<=``.
    """
    value = _cell(record, key)
    if value is None:
        operator = _cell(record, key.replace('inclusive', 'op') if key else None)
        if operator is None:
            return default
        value = operator
    if isinstance(value, str):
        text = value.strip()
        if text in {'<=', '≤', 'inclusive', 'closed', '['}:
            return True
        if text in {'<', 'exclusive', 'open', '('}:
            return False
        raise FormatError(f'Unrecognized inclusivity {value!r}')
    return bool(value)


def _table_source(record, low, high, label, source):
    """
    Build citable rule text for a table row.
    """
    if source is not None and record.get(source):
        return str(record[source])
    low_text = record.get(low)
    high_text = record.get(high)
    label_text = record.get(label)
    if low_text is None and high_text is None:
        return f'{label_text}'
    if high_text is None:
        return f'{low_text} = {label_text!r}'
    return f'{low_text} .. {high_text} = {label_text!r}'


def _statements(text):
    """
    Split a SAS format definition into semicolon-terminated statements.

    Returns ``(statement, offset)`` pairs.  Comments are blanked rather than
    removed so offsets and line numbers stay accurate.
    """
    statements = []
    buffer = []
    start = 0
    quote = None
    i = 0
    while i < len(text):
        char = text[i]
        following = text[i + 1] if i + 1 < len(text) else ''
        if quote is not None:
            if char == quote:
                if following == quote:
                    buffer.extend((char, following))
                    i += 2
                    continue
                quote = None
            buffer.append(char)
        elif char in {'\'', '"'}:
            quote = char
            buffer.append(char)
        elif char == '/' and following == '*':
            end = text.find('*/', i + 2)
            if end < 0:
                raise FormatError('Unterminated comment in value format')
            buffer.extend(' ' * (end + 2 - i))
            i = end + 2
            continue
        elif char == ';':
            statements.append((''.join(buffer), start))
            buffer = []
            start = i + 1
        else:
            buffer.append(char)
        i += 1
    if quote is not None:
        raise FormatError('Unterminated quoted label in value format')
    if ''.join(buffer).strip():
        statements.append((''.join(buffer), start))
    return statements


def _make_rule(match, number, line):
    """
    Create a rule from a regular-expression match.
    """
    range_text = match['range'].strip()
    label = _unquote(match['label'])
    source = ' '.join(f'{range_text} = {match["label"]}'.split())
    upper = range_text.upper()
    if upper == 'OTHER':
        return IntervalRule(number, label, source, kind='other', line=line)
    if _SPECIAL_TOKEN.match(range_text):
        return IntervalRule(
            number,
            label,
            source,
            kind='missing',
            tag=_normalize_tag(range_text),
            line=line,
        )
    parts = _RANGE_PARTS.match(range_text)
    if parts is None:
        if _SINGLE_TOKEN.match(range_text):
            value = _as_decimal(range_text)
            return IntervalRule(number, label, source, low=value, high=value, line=line)
        raise FormatError(f'Cannot parse range on line {line}: {range_text!r}')

    low_text = parts['low'].strip()
    high_text = parts['high'].strip()
    separator = parts['sep']
    low = _endpoint(low_text, lower=True, line=line)
    high = _endpoint(high_text, lower=False, line=line)
    return IntervalRule(
        number,
        label,
        source,
        low=low,
        low_inclusive=not separator.startswith('<'),
        high=high,
        high_inclusive=not separator.endswith('<'),
        line=line,
    )


def _endpoint(text, lower, line):
    """
    Parse one endpoint of a range.
    """
    upper = text.upper()
    if lower and upper in {'LOW', 'MIN'}:
        return _NEGATIVE_INFINITY
    if not lower and upper in {'HIGH', 'MAX'}:
        return _POSITIVE_INFINITY
    if upper in {'LOW', 'MIN'}:
        raise FormatError(f'LOW is not a valid high endpoint on line {line}')
    if upper in {'HIGH', 'MAX'}:
        raise FormatError(f'HIGH is not a valid low endpoint on line {line}')
    return _as_decimal(text)


def _unquote(text):
    """
    Remove SAS quoting and unescape doubled quote marks.
    """
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'\'', '"'}:
        quote = text[0]
        return text[1:-1].replace(quote * 2, quote)
    return text


def _coerce(value):
    """
    Convert a raw value to ``(token, Decimal or None, is_missing)``.
    """
    if value is None or value is pd.NA or value is pd.NaT:
        return '.', None, True
    tag = getattr(value, 'name', None)
    if isinstance(tag, str) and tag in MISSING_TAGS:
        return f'.{tag}', None, True
    if isinstance(value, str):
        text = value.strip()
        if text == '.':
            return '.', None, True
        if _SPECIAL_TOKEN.match(text):
            return _normalize_tag(text), None, True
        try:
            return text, Decimal(text), False
        except InvalidOperation:
            return text, None, False
    try:
        if value != value:
            return '.', None, True
    except (TypeError, ValueError):
        pass
    try:
        return str(value), Decimal(str(value)), False
    except (InvalidOperation, ValueError):
        return str(value), None, False


def _intersection(left, right):
    """
    Return an :class:`Overlap` if two rules share a value, else ``None``.
    """
    if left.kind != right.kind:
        return None
    if left.kind == 'missing':
        if left.tag == right.tag:
            return Overlap(left, right, None, True, None, True)
        return None
    if left.high < right.low or right.high < left.low:
        return None
    if left.low > right.low:
        low, low_inclusive = left.low, left.low_inclusive
    elif right.low > left.low:
        low, low_inclusive = right.low, right.low_inclusive
    else:
        low = left.low
        low_inclusive = left.low_inclusive and right.low_inclusive
    if left.high < right.high:
        high, high_inclusive = left.high, left.high_inclusive
    elif right.high < left.high:
        high, high_inclusive = right.high, right.high_inclusive
    else:
        high = left.high
        high_inclusive = left.high_inclusive and right.high_inclusive
    if low > high:
        return None
    if low == high and not (low_inclusive and high_inclusive):
        return None
    return Overlap(left, right, low, low_inclusive, high, high_inclusive)


def _merge_entry(rule):
    """
    Build a mutable merged-interval entry from a rule.
    """
    return [rule.low, rule.low_inclusive, rule.high, rule.high_inclusive]


def _extend(target, rule):
    """
    Extend a merged interval in place with another rule.
    """
    if rule.high > target[2]:
        target[2] = rule.high
        target[3] = rule.high_inclusive
    elif rule.high == target[2]:
        target[3] = target[3] or rule.high_inclusive


def _gap_tuple(low, low_inclusive, high, high_inclusive):
    """
    Create a gap tuple, normalizing an empty interval away.
    """
    if low > high or (low == high and not (low_inclusive and high_inclusive)):
        return None
    return (low, low_inclusive, high, high_inclusive)
