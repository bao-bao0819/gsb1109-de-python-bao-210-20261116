# =============================================================================
# Three-valued row filtering for ARFF datasets.
#
# A filter expression is built from field conditions combined with ``and``,
# ``or`` and ``not``.  Every condition is evaluated against a decoded ARFF
# row using Kleene three-valued logic so that missing values (the explicit
# ``?`` marker) are never silently treated as ``False``:
#
# * a definite result partitions the row into ``keep`` (expression is True)
#   or ``exclude`` (expression is False);
# * a result that depends on one or more missing values is ``unknown`` and
#   the exact set of fields it depends on is reported.
#
# Sparse ARFF semantics follow the attribute declarations and the conventions
# implemented by :mod:`arff`:
#
# * an omitted index is *not* missing: numeric attributes default to zero,
#   nominal attributes default to their first declared category and string
#   attributes default to ``'0'`` (the value liac-arff produces when a sparse
#   string cell is expanded into a dense row);
# * an explicit ``?`` (including ``{0 ?}`` in sparse syntax) is missing and
#   therefore unknown;
# * a quoted ``"?"`` is an ordinary value.  It is only a valid nominal value
#   when the attribute declaration quotes it as well, e.g. ``{yes, no, "?"}``.
# =============================================================================

import argparse
import json
import re
import sys

import arff


# RESULT LABELS ===============================================================
KEEP = 'keep'
EXCLUDE = 'exclude'
UNKNOWN = 'unknown'

# COMPARISON OPERATORS ========================================================
EQ = '='
NE = '!='
LT = '<'
LE = '<='
GT = '>'
GE = '>='
IN = 'in'
NOT_IN = 'not in'
IS_MISSING = 'is missing'
IS_NOT_MISSING = 'is not missing'

_COMPARATORS = frozenset((EQ, NE, LT, LE, GT, GE))
_MEMBERSHIP = frozenset((IN, NOT_IN))
_MISSING_OPS = frozenset((IS_MISSING, IS_NOT_MISSING))
_SUPPORTED_OPS = _COMPARATORS | _MEMBERSHIP | _MISSING_OPS


class FilterError(ValueError):
    '''Raised when an expression cannot be built or evaluated.'''


class BadExpression(FilterError):
    '''Raised when a filter expression string is syntactically invalid.'''


# THREE-VALUED LOGIC ==========================================================
def _not(value):
    if value is None:
        return None
    return not value


def _and(left, right):
    if left is False or right is False:
        return False
    if left is None or right is None:
        return None
    return True


def _or(left, right):
    if left is True or right is True:
        return True
    if left is None or right is None:
        return None
    return False


# EXPRESSION TREE =============================================================
class Node:
    '''Base class for expression tree nodes.

    Every node evaluates to a ``(tribool, dependencies)`` pair where
    ``tribool`` is ``True``, ``False`` or ``None`` (unknown) and
    ``dependencies`` is the set of missing field names the *outcome* actually
    depends on.  A branch that determines the result on its own contributes
    no dependencies, keeping the reported set minimal.
    '''

    def eval(self, record, defaults):
        raise NotImplementedError

    def fields(self):
        raise NotImplementedError


class Condition(Node):
    '''A single field condition.

    :param attr_index: position of the attribute in the ARFF declaration.
    :param field: attribute name.
    :param op: one of the operator constants defined in this module.
    :param value: constant value for comparisons, or a list for membership.
    '''

    def __init__(self, attr_index, field, op, value=None):
        if op not in _SUPPORTED_OPS:
            raise FilterError('Unsupported operator %r' % (op,))
        self.attr_index = attr_index
        self.field = field
        self.op = op
        self.value = value

    def _cell(self, record, defaults):
        if isinstance(record, dict):
            # LOD sparse row: an explicit key carries the decoded value
            # (None for an explicit ?), an omitted key gets the value declared
            # by the attribute rather than being treated as missing.
            if self.attr_index in record:
                return record[self.attr_index]
            return defaults[self.attr_index]
        return record[self.attr_index]

    def eval(self, record, defaults):
        cell = self._cell(record, defaults)

        if self.op in _MISSING_OPS:
            result = cell is None
            if self.op == IS_NOT_MISSING:
                result = not result
            return result, set()

        if cell is None:
            # Explicit missing value: the condition is genuinely unknown.
            return None, {self.field}

        try:
            if self.op in _COMPARATORS:
                result = _COMPARE[self.op](cell, self.value)
            elif self.op == IN:
                result = cell in self.value
            else:  # NOT_IN
                result = cell not in self.value
        except TypeError as exc:
            raise FilterError(
                'Cannot compare field %r (value %r) against %r using %r: %s'
                % (self.field, cell, self.value, self.op, exc))

        return result, set()

    def fields(self):
        return {self.field}

    def __str__(self):
        if self.op in _MISSING_OPS:
            return '%s %s' % (self.field, self.op)
        if self.op in _MEMBERSHIP:
            return '%s %s %s' % (self.field, self.op,
                                 _format_value_list(self.value))
        return '%s %s %s' % (self.field, self.op, _format_value(self.value))


class _BinOp(Node):
    symbol = ''

    def __init__(self, left, right):
        self.left = left
        self.right = right

    def fields(self):
        return self.left.fields() | self.right.fields()

    def __str__(self):
        return '(%s %s %s)' % (self.left, self.symbol, self.right)


class AndNode(_BinOp):
    symbol = 'and'

    def eval(self, record, defaults):
        left_tri, left_deps = self.left.eval(record, defaults)
        right_tri, right_deps = self.right.eval(record, defaults)
        result = _and(left_tri, right_tri)
        if result is False:
            # False dominates: missing values on either side cannot matter.
            return False, set()
        if result is None:
            deps = set()
            if left_tri is None:
                deps |= left_deps
            if right_tri is None:
                deps |= right_deps
            return None, deps
        return True, set()


class OrNode(_BinOp):
    symbol = 'or'

    def eval(self, record, defaults):
        left_tri, left_deps = self.left.eval(record, defaults)
        right_tri, right_deps = self.right.eval(record, defaults)
        result = _or(left_tri, right_tri)
        if result is True:
            # True dominates: missing values on either side cannot matter.
            return True, set()
        if result is None:
            deps = set()
            if left_tri is None:
                deps |= left_deps
            if right_tri is None:
                deps |= right_deps
            return None, deps
        return False, set()


class NotNode(Node):
    def __init__(self, operand):
        self.operand = operand

    def eval(self, record, defaults):
        tri, deps = self.operand.eval(record, defaults)
        return _not(tri), set(deps)

    def fields(self):
        return self.operand.fields()

    def __str__(self):
        return 'not %s' % (self.operand,)


_COMPARE = {
    EQ: lambda a, b: a == b,
    NE: lambda a, b: a != b,
    LT: lambda a, b: a < b,
    LE: lambda a, b: a <= b,
    GT: lambda a, b: a > b,
    GE: lambda a, b: a >= b,
}


def _format_value(value):
    if isinstance(value, str):
        if value in ('?', '') or re.search(r'[\s,(){}\'"=!<>]', value):
            return '"%s"' % value.replace('"', '\\"')
        return value
    return str(value)


def _format_value_list(values):
    return '(' + ', '.join(_format_value(v) for v in values) + ')'


# SPARSE DEFAULTS =============================================================
def build_sparse_defaults(attributes):
    '''Return the implicit value of each attribute in a sparse row.

    Mirrors liac-arff's dense expansion of omitted sparse indices: numeric
    attributes become zero, nominal attributes become their first declared
    category and STRING attributes become ``'0'``.
    '''
    defaults = []
    for _, attr_type in attributes:
        if isinstance(attr_type, (list, tuple)):
            defaults.append(attr_type[0])
        elif attr_type == 'INTEGER':
            defaults.append(0)
        elif attr_type in ('REAL', 'NUMERIC'):
            defaults.append(0.0)
        else:  # STRING
            defaults.append('0')
    return defaults


def coo_to_lod(coo_data, n_rows=None):
    '''Convert a decoded ``arff.COO`` triple into LOD-style dictionary rows.

    COO triples hold no entry for completely empty sparse rows, so their total
    number cannot always be recovered: pass ``n_rows`` to pad trailing empty
    rows.  Without it the result ends at the last row that contains a value.
    Explicit missing values (``None`` entries) are preserved.
    '''
    data, rows, cols = coo_data
    if not rows:
        return [{} for _ in range(n_rows or 0)]
    count = n_rows if n_rows is not None else max(rows) + 1
    lod = [{} for _ in range(count)]
    for row, col, value in zip(rows, cols, data):
        lod[row][col] = value
    return lod


# PARSING =====================================================================
_TOKEN_RE = re.compile(r'''
    \s*(?:
        (?P<lparen>\() |
        (?P<rparen>\)) |
        (?P<comma>,) |
        (?P<and>\band\b) |
        (?P<or>\bor\b) |
        (?P<not>\bnot\b) |
        (?P<is>\bis\b) |
        (?P<in>\bin\b) |
        (?P<missing>\bmissing\b) |
        (?P<op>>=|<=|!=|<>|=|>|<) |
        (?P<quoted>"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*') |
        (?P<word>[^\s(),=!<>]+)
    )
''', re.VERBOSE | re.IGNORECASE)


def _tokenize(text):
    tokens = []
    pos = 0
    while pos < len(text):
        match = _TOKEN_RE.match(text, pos)
        if not match or match.end() == pos:
            raise BadExpression(
                'Unexpected text %r at position %d' % (text[pos:], pos))
        kind = match.lastgroup
        tokens.append((kind, match.group(kind)))
        pos = match.end()
    return tokens


class _Parser:
    def __init__(self, text, attributes):
        self.tokens = _tokenize(text)
        self.pos = 0
        self.attributes = attributes
        self._index = {name: (i, attr_type)
                       for i, (name, attr_type) in enumerate(attributes)}

    # token helpers ----------------------------------------------------------
    def _peek(self):
        if self.pos >= len(self.tokens):
            return None, None
        return self.tokens[self.pos]

    def _peek_kind(self):
        return self._peek()[0]

    def _next(self):
        token = self.tokens[self.pos]
        self.pos += 1
        return token

    def _expect(self, *kinds):
        kind, value = self._peek()
        if kind not in kinds:
            raise BadExpression(
                'Expected one of %s but found %r' % (', '.join(kinds), value))
        return self._next()

    # grammar ----------------------------------------------------------------
    def parse(self):
        if not self.tokens:
            raise BadExpression('Empty filter expression')
        root = self.parse_or()
        if self.pos != len(self.tokens):
            kind, value = self.tokens[self.pos]
            raise BadExpression('Unexpected token %r' % value)
        return root

    def parse_or(self):
        node = self.parse_and()
        while self._peek_kind() == 'or':
            self._next()
            node = OrNode(node, self.parse_and())
        return node

    def parse_and(self):
        node = self.parse_not()
        while self._peek_kind() == 'and':
            self._next()
            node = AndNode(node, self.parse_not())
        return node

    def parse_not(self):
        if self._peek_kind() == 'not':
            self._next()
            return NotNode(self.parse_not())
        return self.parse_atom()

    def parse_atom(self):
        if self._peek_kind() == 'lparen':
            self._next()
            node = self.parse_or()
            self._expect('rparen')
            return node
        return self.parse_condition()

    def parse_condition(self):
        kind, raw_field = self._expect('word', 'quoted')
        field = self._field_name(kind, raw_field)
        if field not in self._index:
            raise FilterError('Unknown field %r' % field)
        attr_index, attr_type = self._index[field]

        lead = self._peek_kind()
        if lead == 'is':
            self._next()
            negated = self._peek_kind() == 'not'
            if negated:
                self._next()
            self._expect('missing')
            op = IS_NOT_MISSING if negated else IS_MISSING
            return Condition(attr_index, field, op)

        if lead == 'not':
            self._next()
            self._expect('in')
            values = self.parse_value_list(attr_type, field)
            return Condition(attr_index, field, NOT_IN, values)

        if lead == 'in':
            self._next()
            values = self.parse_value_list(attr_type, field)
            return Condition(attr_index, field, IN, values)

        if lead == 'op':
            _, raw_op = self._next()
            op = '!=' if raw_op == '<>' else raw_op
            token_kind, raw_value = self._expect('word', 'quoted')
            value = self._coerce(attr_type, token_kind, raw_value, field)
            return Condition(attr_index, field, op, value)

        raise BadExpression(
            'Expected an operator after field %r' % field)

    def parse_value_list(self, attr_type, field):
        opened = False
        if self._peek_kind() == 'lparen':
            self._next()
            opened = True
        values = []
        while True:
            token_kind, raw_value = self._expect('word', 'quoted')
            values.append(self._coerce(attr_type, token_kind, raw_value,
                                       field))
            if self._peek_kind() == 'comma':
                self._next()
                continue
            break
        if opened:
            self._expect('rparen')
        return values

    # value handling ---------------------------------------------------------
    @staticmethod
    def _field_name(kind, raw):
        if kind == 'quoted':
            name = arff._unquote(raw)
        else:
            name = raw
        if name is None:
            raise BadExpression('Missing field name')
        return name

    @staticmethod
    def _coerce(attr_type, kind, raw, field):
        if kind != 'quoted' and raw in ('?', ''):
            # A bare question mark is the missing marker, not a comparable
            # value.  A literal '?' category must be quoted.
            raise FilterError(
                'Cannot compare against the missing marker %r; '
                'use "is missing" or quote it ("?") to name a category'
                % raw)
        value = arff._unquote(raw) if kind == 'quoted' else raw
        if value is None:
            raise FilterError('Missing value in filter expression')

        if isinstance(attr_type, (list, tuple)):
            if value not in attr_type:
                raise FilterError(
                    'Value %r is not declared for nominal field %r; '
                    'expected one of %s'
                    % (value, field, ', '.join(map(str, attr_type))))
            return value

        if attr_type == 'STRING':
            return value

        try:
            if attr_type == 'INTEGER':
                return int(float(value))
            return float(value)
        except ValueError:
            raise FilterError(
                'Value %r is not numeric for field %r' % (value, field))


# PUBLIC EXPRESSION API =======================================================
class Expression:
    '''A compiled filter expression bound to ARFF attributes.'''

    def __init__(self, root, attributes):
        self.root = root
        self.attributes = attributes
        self._defaults = build_sparse_defaults(attributes)

    @property
    def fields(self):
        return self.root.fields()

    def evaluate(self, record):
        '''Return ``True``, ``False`` or ``None`` (unknown) for one row.'''
        return self.root.eval(record, self._defaults)[0]

    def classify(self, record):
        '''Return ``(label, dependencies)`` for one row.

        ``dependencies`` is an empty tuple for definite results and a sorted
        tuple of field names for unknown results.
        '''
        tri, deps = self.root.eval(record, self._defaults)
        if tri is True:
            return KEEP, ()
        if tri is False:
            return EXCLUDE, ()
        ordered = sorted(deps, key=lambda name: self._field_index(name))
        return UNKNOWN, tuple(ordered)

    def _field_index(self, name):
        for i, (attr_name, _) in enumerate(self.attributes):
            if attr_name == name:
                return i
        raise FilterError('Unknown field %r' % name)

    def __str__(self):
        return str(self.root)


def parse_expression(text, attributes):
    '''Compile a filter expression string against an ARFF attribute list.

    Keywords ``and``, ``or``, ``not``, ``is``, ``in`` and ``missing`` are
    case-insensitive; parentheses group sub-expressions.  Field names and
    values follow ARFF quoting rules.
    '''
    return Expression(_Parser(text, attributes).parse(), attributes)


def condition(attributes, field, op, value=None):
    '''Build a :class:`Condition` with constants coerced from Python values.

    For membership tests pass a list as ``value``.
    '''
    index = {name: (i, attr_type)
             for i, (name, attr_type) in enumerate(attributes)}
    if field not in index:
        raise FilterError('Unknown field %r' % field)
    attr_index, attr_type = index[field]

    if op in _MISSING_OPS:
        return Condition(attr_index, field, op)

    if op in _MEMBERSHIP:
        if not isinstance(value, (list, tuple)):
            raise FilterError('Operator %r requires a list of values' % op)
        coerced = [_coerce_python(attr_type, item, field) for item in value]
        return Condition(attr_index, field, op, coerced)

    if op in _COMPARATORS:
        return Condition(
            attr_index, field, op,
            _coerce_python(attr_type, value, field))

    raise FilterError('Unsupported operator %r' % (op,))


def _coerce_python(attr_type, value, field):
    if value is None:
        raise FilterError(
            'Cannot compare against None; use the %r operator' % IS_MISSING)
    if isinstance(attr_type, (list, tuple)):
        if value not in attr_type:
            raise FilterError(
                'Value %r is not declared for nominal field %r'
                % (value, field))
        return value
    if attr_type == 'STRING':
        return str(value)
    if attr_type == 'INTEGER':
        return int(value)
    return float(value)


def and_(left, right):
    return AndNode(left, right)


def or_(left, right):
    return OrNode(left, right)


def not_(operand):
    return NotNode(operand)


# FILTERING RESULT AND STATE ==================================================
class FilterResult:
    '''Partition of the evaluated rows.'''

    def __init__(self, results, records):
        # results: {row_index: (label, dependencies_tuple)}
        self._results = results
        self._records = records
        self.keep_indices = []
        self.exclude_indices = []
        self.unknown_indices = []
        self.dependencies = {}
        for index in sorted(results):
            label, deps = results[index]
            if label == KEEP:
                self.keep_indices.append(index)
            elif label == EXCLUDE:
                self.exclude_indices.append(index)
            else:
                self.unknown_indices.append(index)
                self.dependencies[index] = list(deps)

    @property
    def keep(self):
        return [self._records[i] for i in self.keep_indices]

    @property
    def exclude(self):
        return [self._records[i] for i in self.exclude_indices]

    @property
    def unknown(self):
        return [self._records[i] for i in self.unknown_indices]

    @property
    def labels(self):
        return {index: label for index, (label, _) in self._results.items()}

    def to_dict(self):
        return {
            KEEP: self.keep_indices,
            EXCLUDE: self.exclude_indices,
            UNKNOWN: [
                {'index': index, 'depends_on': self.dependencies[index]}
                for index in self.unknown_indices
            ],
        }


class FilterState:
    '''Stateful filter supporting incremental re-judgement.

    A definite verdict produced by three-valued logic cannot change when
    missing fields are later filled: if an expression is already True or
    False with some inputs unknown, no concrete assignment of those inputs
    can flip it.  After a column is filled only the unknown rows whose
    reported dependencies touch the updated fields therefore need to be
    re-evaluated.
    '''

    def __init__(self, expression, records, attributes=None):
        if isinstance(expression, str):
            if attributes is None:
                raise FilterError(
                    'attributes are required when expression is a string')
            expression = parse_expression(expression, attributes)
        self.expression = expression
        self.records = records
        self.results = {}

    def run(self):
        '''Evaluate every row and (re)build the partition.'''
        self.results = {
            index: self.expression.classify(record)
            for index, record in enumerate(self.records)
        }
        return FilterResult(self.results, self.records)

    def reclassify(self, updated_fields):
        '''Re-judge only unknown rows depending on ``updated_fields``.

        :param updated_fields: an iterable of field names that were filled in.
            Rows already labelled keep/exclude are left untouched (their
            verdicts cannot change when only missing cells get values) and
            rows whose unknown verdict depends on other fields stay unknown.
            To correct an already populated value, re-judge that row
            explicitly with :meth:`update_record`.
        '''
        updated = set(updated_fields)
        for index, (label, deps) in list(self.results.items()):
            if label == UNKNOWN and updated.intersection(deps):
                self.results[index] = self.expression.classify(
                    self.records[index])
        return FilterResult(self.results, self.records)

    def refill(self, *fields):
        '''Convenience alias for :meth:`reclassify`.'''
        return self.reclassify(fields)

    def update_record(self, index, record):
        '''Replace one row and re-judge just that row.'''
        self.records[index] = record
        self.results[index] = self.expression.classify(record)
        return self.results[index]


def partition(expression, records, attributes=None):
    '''Partition decoded ARFF rows into keep/exclude/unknown.'''
    return FilterState(expression, records, attributes).run()


# COMMAND LINE INTERFACE ======================================================
def _write_subset(obj, records, path):
    subset = dict(obj)
    if records:
        subset['data'] = records
    else:
        subset.pop('data', None)
    with open(path, 'w') as fp:
        arff.dump(subset, fp)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog='arff_filter',
        description='Filter ARFF rows with a three-valued boolean expression.')
    parser.add_argument('source', help='path to the input .arff file')
    parser.add_argument('-e', '--expr', action='append', default=[],
                        help='filter expression; repeatable, expressions are '
                             'combined with and')
    parser.add_argument('--expr-file',
                        help='file holding the filter expression')
    parser.add_argument('--sparse', action='store_true',
                        help='decode sparse data as LOD dictionaries instead '
                             'of dense rows')
    parser.add_argument('--keep', help='output path for kept rows')
    parser.add_argument('--exclude', help='output path for excluded rows')
    parser.add_argument('--unknown', help='output path for unknown rows')
    parser.add_argument('--report', help='write the JSON report here instead '
                                         'of stdout')
    args = parser.parse_args(argv)

    expression_text = '\n and '.join('(%s)' % expr for expr in args.expr)
    if args.expr_file:
        with open(args.expr_file) as fp:
            file_text = fp.read().strip()
        expression_text = ('%s and (%s)' % (expression_text, file_text)
                           if expression_text else file_text)
    if not expression_text:
        parser.error('provide at least one --expr or --expr-file')

    return_type = arff.LOD if args.sparse else arff.DENSE
    with open(args.source) as fp:
        obj = arff.load(fp, return_type=return_type)

    expression = parse_expression(expression_text, obj['attributes'])
    result = partition(expression, obj['data'], obj['attributes'])

    if args.keep:
        _write_subset(obj, result.keep, args.keep)
    if args.exclude:
        _write_subset(obj, result.exclude, args.exclude)
    if args.unknown:
        _write_subset(obj, result.unknown, args.unknown)

    report = result.to_dict()
    report['relation'] = obj['relation']
    report['expression'] = str(expression)
    report['counts'] = {
        KEEP: len(result.keep_indices),
        EXCLUDE: len(result.exclude_indices),
        UNKNOWN: len(result.unknown_indices),
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.report:
        with open(args.report, 'w') as fp:
            fp.write(rendered)
    else:
        print(rendered)
    return 0


if __name__ == '__main__':
    sys.exit(main())
