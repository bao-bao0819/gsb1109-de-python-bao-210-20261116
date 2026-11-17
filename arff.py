# =============================================================================
# Federal University of Rio Grande do Sul (UFRGS)
# Connectionist Artificial Intelligence Laboratory (LIAC)
# Renato de Pontes Pereira - rppereira@inf.ufrgs.br
# =============================================================================
# Copyright (c) 2011 Renato de Pontes Pereira, renato.ppontes at gmail dot com
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# =============================================================================

'''
The liac-arff module implements functions to read and write ARFF files in
Python. It was created in the Connectionist Artificial Intelligence Laboratory
(LIAC), which takes place at the Federal University of Rio Grande do Sul
(UFRGS), in Brazil.

ARFF (Attribute-Relation File Format) is an file format specially created for
describe datasets which are commonly used for machine learning experiments and
softwares. This file format was created to be used in Weka, the best
representative software for machine learning automated experiments.

An ARFF file can be divided into two sections: header and data. The Header
describes the metadata of the dataset, including a general description of the
dataset, its name and its attributes. The source below is an example of a
header section in a XOR dataset::

    %
    % XOR Dataset
    %
    % Created by Renato Pereira
    %            rppereira@inf.ufrgs.br
    %            http://inf.ufrgs.br/~rppereira
    %
    %
    @RELATION XOR

    @ATTRIBUTE input1 REAL
    @ATTRIBUTE input2 REAL
    @ATTRIBUTE y REAL

The Data section of an ARFF file describes the observations of the dataset, in
the case of XOR dataset::

    @DATA
    0.0,0.0,0.0
    0.0,1.0,1.0
    1.0,0.0,1.0
    1.0,1.0,0.0
    %
    %
    %

Notice that several lines are starting with an ``%`` symbol, denoting a
comment, thus, lines with ``%`` at the beginning will be ignored, except by the
description part at the beginning of the file. The declarations ``@RELATION``,
``@ATTRIBUTE``, and ``@DATA`` are all case insensitive and obligatory.

For more information and details about the ARFF file description, consult
http://www.cs.waikato.ac.nz/~ml/weka/arff.html


ARFF Files in Python
~~~~~~~~~~~~~~~~~~~~

This module uses built-ins python objects to represent a deserialized ARFF
file. A dictionary is used as the container of the data and metadata of ARFF,
and have the following keys:

- **description**: (OPTIONAL) a string with the description of the dataset.
- **relation**: (OBLIGATORY) a string with the name of the dataset.
- **attributes**: (OBLIGATORY) a list of attributes with the following
  template::

    (attribute_name, attribute_type)

  the attribute_name is a string, and attribute_type must be an string
  or a list of strings.
- **data**: (OBLIGATORY) a list of data instances. Each data instance must be
  a list with values, depending on the attributes.

The above keys must follow the case which were described, i.e., the keys are
case sensitive. The attribute type ``attribute_type`` must be one of these
strings (they are not case sensitive): ``NUMERIC``, ``INTEGER``, ``REAL`` or
``STRING``. For nominal attributes, the ``atribute_type`` must be a list of
strings. For date attributes, the ``attribute_type`` must be the string
``DATE``, optionally followed by a format string, e.g. ``DATE yyyy-MM-dd``.

Date attributes
~~~~~~~~~~~~~~~

This module does not implement any date parsing library. Instead, the values
of ``DATE`` attributes are converted through a caller-provided adapter
object, which must implement::

    adapter.parse(value, fmt)   # str (or number) -> date value
    adapter.format(value, fmt)  # date value -> str

where ``fmt`` is the format declared in the attribute specification (or
``None`` for a bare ``DATE``). The adapter is passed to ``load``/``loads``
and ``dump``/``dumps`` through the ``date_adapter`` argument.

Explicit dates, missing values and omitted sparse entries keep their
origins: missing values (``?``) are decoded as ``None`` and never reach the
adapter, while the numeric zero generated by omitted entries of sparse data
is passed to ``parse`` as a number and must be interpreted by the adapter
as the epoch of the date type -- it is never treated as a missing value.
Encoded date values are always produced by ``adapter.format``; they are
never degraded to plain strings.

The ``tz`` and ``version`` attributes of the adapter, when present, are
frozen when a reading (or writing) task starts: changing them in the middle
of a task raises ``BadDateAdapter``. Each task binds its own conversors, so
adapters are never shared between different tasks.

In this format, the XOR dataset presented above can be represented as a python
object as::

    xor_dataset = {
        'description': 'XOR Dataset',
        'relation': 'XOR',
        'attributes': [
            ('input1', 'REAL'),
            ('input2', 'REAL'),
            ('y', 'REAL'),
        ],
        'data': [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0]
        ]
    }


Features
~~~~~~~~

This module provides several features, including:

- Read and write ARFF files using python built-in structures, such dictionaries
  and lists;
- Supports `scipy.sparse.coo <http://docs.scipy
  .org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html#scipy.sparse.coo_matrix>`_
  and lists of dictionaries as used by SVMLight
- Supports the following attribute types: NUMERIC, REAL, INTEGER, STRING,
  DATE (through caller-provided adapters), and NOMINAL;
- Has an interface similar to other built-in modules such as ``json``, or
  ``zipfile``;
- Supports read and write the descriptions of files;
- Supports missing values and names with spaces;
- Supports unicode values and names;
- Fully compatible with Python 2.7+, Python 3.5+, pypy and pypy3;
- Under `MIT License <http://opensource.org/licenses/MIT>`_

'''
__author__ = 'Renato de Pontes Pereira, Matthias Feurer, Joel Nothman'
__author_email__ = ('renato.ppontes@gmail.com, '
                    'feurerm@informatik.uni-freiburg.de, '
                    'joel.nothman@gmail.com')
__version__ = '2.4.0'

import re
import sys
import csv
from typing import Optional

# CONSTANTS ===================================================================
_SIMPLE_TYPES = ['NUMERIC', 'REAL', 'INTEGER', 'STRING']

_TK_DESCRIPTION = '%'
_TK_COMMENT     = '%'
_TK_RELATION    = '@RELATION'
_TK_ATTRIBUTE   = '@ATTRIBUTE'
_TK_DATA        = '@DATA'

_RE_RELATION     = re.compile(r'^([^\{\}%,\s]*|\".*\"|\'.*\')$', re.UNICODE)
_RE_ATTRIBUTE    = re.compile(r'^(\".*\"|\'.*\'|[^\{\}%,\s]*)\s+(.+)$', re.UNICODE)
_RE_QUOTE_CHARS = re.compile(r'["\'\\\s%,\000-\031]', re.UNICODE)
_RE_ESCAPE_CHARS = re.compile(r'(?=["\'\\%])|[\n\r\t\000-\031]')
_RE_SPARSE_LINE = re.compile(r'^\s*\{.*\}\s*$', re.UNICODE)
_RE_NONTRIVIAL_DATA = re.compile('["\'{}\\s]', re.UNICODE)


def _build_re_values():
    quoted_re = r'''
                    "      # open quote followed by zero or more of:
                    (?:
                        (?<!\\)    # no additional backslash
                        (?:\\\\)*  # maybe escaped backslashes
                        \\"        # escaped quote
                    |
                        \\[^"]     # escaping a non-quote
                    |
                        [^"\\]     # non-quote char
                    )*
                    "      # close quote
                    '''
    # a value is surrounded by " or by ' or contains no quotables
    value_re = r'''(?:
        %s|          # a value may be surrounded by "
        %s|          # or by '
        [^,\s"'{}]+  # or may contain no characters requiring quoting
        )''' % (quoted_re,
                quoted_re.replace('"', "'"))

    # This captures (value, error) groups. Because empty values are allowed,
    # we cannot just look for empty values to handle syntax errors.
    # We presume the line has had ',' prepended...
    dense = re.compile(r'''(?x)
        ,                # may follow ','
        \s*
        ((?=,)|$|{value_re})  # empty or value
        |
        (\S.*)           # error
        '''.format(value_re=value_re))

    # This captures (key, value) groups and will have an empty key/value
    # in case of syntax errors.
    # It does not ensure that the line starts with '{' or ends with '}'.
    sparse = re.compile(r'''(?x)
        (?:^\s*\{|,)   # may follow ',', or '{' at line start
        \s*
        (\d+)          # attribute key
        \s+
        (%(value_re)s) # value
        |
        (?!}\s*$)      # not an error if it's }$
        (?!^\s*{\s*}\s*$)  # not an error if it's ^{}$
        \S.*           # error
        ''' % {'value_re': value_re})
    return dense, sparse



_RE_DENSE_VALUES, _RE_SPARSE_KEY_VALUES = _build_re_values()


_ESCAPE_SUB_MAP = {
    '\\\\': '\\',
    '\\"': '"',
    "\\'": "'",
    '\\t': '\t',
    '\\n': '\n',
    '\\r': '\r',
    '\\b': '\b',
    '\\f': '\f',
    '\\%': '%',
}
_UNESCAPE_SUB_MAP = {chr(i): '\\%03o' % i for i in range(32)}
_UNESCAPE_SUB_MAP.update({v: k for k, v in _ESCAPE_SUB_MAP.items()})
_UNESCAPE_SUB_MAP[''] = '\\'
_ESCAPE_SUB_MAP.update({'\\%d' % i: chr(i) for i in range(10)})


def _escape_sub_callback(match):
    s = match.group()
    if len(s) == 2:
        try:
            return _ESCAPE_SUB_MAP[s]
        except KeyError:
            raise ValueError('Unsupported escape sequence: %s' % s)
    if s[1] == 'u':
        return chr(int(s[2:], 16))
    else:
        return chr(int(s[1:], 8))


def _unquote(v):
    if v[:1] in ('"', "'"):
        return re.sub(r'\\([0-9]{1,3}|u[0-9a-f]{4}|.)', _escape_sub_callback,
                      v[1:-1])
    elif v in ('?', ''):
        return None
    else:
        return v


def _parse_values(s):
    '''(INTERNAL) Split a line into a list of values'''
    if not _RE_NONTRIVIAL_DATA.search(s):
        # Fast path for trivial cases (unfortunately we have to handle missing
        # values because of the empty string case :(.)
        return [None if s in ('?', '') else s
                for s in next(csv.reader([s]))]

    # _RE_DENSE_VALUES tokenizes despite quoting, whitespace, etc.
    values, errors = zip(*_RE_DENSE_VALUES.findall(',' + s))
    if not any(errors):
        return [_unquote(v) for v in values]
    if _RE_SPARSE_LINE.match(s):
        try:
            return {int(k): _unquote(v)
                    for k, v in _RE_SPARSE_KEY_VALUES.findall(s)}
        except ValueError as exc:
            # an ARFF syntax error in sparse data
            for match in _RE_SPARSE_KEY_VALUES.finditer(s):
                if not match.group(1):
                    raise BadLayout('Error parsing %r' % match.group())
            raise BadLayout('Unknown parsing error')
    else:
        # an ARFF syntax error
        for match in _RE_DENSE_VALUES.finditer(s):
            if match.group(2):
                raise BadLayout('Error parsing %r' % match.group())
        raise BadLayout('Unknown parsing error')


DENSE = 0     # Constant value representing a dense matrix
COO = 1       # Constant value representing a sparse matrix in coordinate format
LOD = 2       # Constant value representing a sparse matrix in list of
              # dictionaries format
DENSE_GEN = 3 # Generator of dictionaries
LOD_GEN = 4   # Generator of dictionaries
_SUPPORTED_DATA_STRUCTURES = [DENSE, COO, LOD, DENSE_GEN, LOD_GEN]


# EXCEPTIONS ==================================================================
class ArffException(Exception):
    message: Optional[str] = None

    def __init__(self):
        self.line = -1

    def __str__(self):
        return self.message%self.line

class BadRelationFormat(ArffException):
    '''Error raised when the relation declaration is in an invalid format.'''
    message = 'Bad @RELATION format, at line %d.'

class BadAttributeFormat(ArffException):
    '''Error raised when some attribute declaration is in an invalid format.'''
    message = 'Bad @ATTRIBUTE format, at line %d.'

class BadDataFormat(ArffException):
    '''Error raised when some data instance is in an invalid format.'''
    def __init__(self, value):
        super().__init__()
        self.message = (
            'Bad @DATA instance format in line %d: ' +
            ('%s' % value)
        )

class BadAttributeType(ArffException):
    '''Error raised when some invalid type is provided into the attribute
    declaration.'''
    message = 'Bad @ATTRIBUTE type, at line %d.'

class BadAttributeName(ArffException):
    '''Error raised when an attribute name is provided twice the attribute
    declaration.'''

    def __init__(self, value, value2):
        super().__init__()
        self.message = (
            ('Bad @ATTRIBUTE name %s at line' % value) +
            ' %d, this name is already in use in line' +
            (' %d.' % value2)
        )

class BadNominalValue(ArffException):
    '''Error raised when a value in used in some data instance but is not
    declared into it respective attribute declaration.'''

    def __init__(self, value):
        super().__init__()
        self.message = (
            ('Data value %s not found in nominal declaration, ' % value)
            + 'at line %d.'
        )

class BadNominalFormatting(ArffException):
    '''Error raised when a nominal value with space is not properly quoted.'''
    def __init__(self, value):
        super().__init__()
        self.message = (
            ('Nominal data value "%s" not properly quoted in line ' % value) +
            '%d.'
        )

class BadNumericalValue(ArffException):
    '''Error raised when and invalid numerical value is used in some data
    instance.'''
    message = 'Invalid numerical value, at line %d.'

class BadStringValue(ArffException):
    '''Error raise when a string contains space but is not quoted.'''
    message = 'Invalid string value at line %d.'

class BadDateAdapter(ArffException):
    '''Error raised when a DATE attribute is found but no date adapter was
    provided, or when the configuration of the adapter changes in the
    middle of a single reading or writing task.'''
    message = 'Bad date adapter, at line %d.'

    def __init__(self, msg=''):
        super().__init__()
        if msg:
            self.message = BadDateAdapter.message + ' ' + \
                msg.replace('%', '%%')

class BadDateValue(ArffException):
    '''Error raised when the date adapter fails to convert a date value.'''

    def __init__(self, value):
        super().__init__()
        self.message = (
            ('Bad date value %s, ' % value) +
            'at line %d.'
        )

class BadLayout(ArffException):
    '''Error raised when the layout of the ARFF file has something wrong.'''
    message = 'Invalid layout of the ARFF file, at line %d.'

    def __init__(self, msg=''):
        super().__init__()
        if msg:
            self.message = BadLayout.message + ' ' + msg.replace('%', '%%')


class BadObject(ArffException):
    '''Error raised when the object representing the ARFF file has something
    wrong.'''
    def __init__(self, msg='Invalid object.'):
        self.msg = msg

    def __str__(self):
        return '%s' % self.msg

# =============================================================================

# INTERNAL ====================================================================
def _unescape_sub_callback(match):
    return _UNESCAPE_SUB_MAP[match.group()]


def encode_string(s):
    if _RE_QUOTE_CHARS.search(s):
        return "'%s'" % _RE_ESCAPE_CHARS.sub(_unescape_sub_callback, s)
    return s


class EncodedNominalConversor:
    def __init__(self, values):
        self.values = {v: i for i, v in enumerate(values)}
        self.values[0] = 0

    def __call__(self, value):
        try:
            return self.values[value]
        except KeyError:
            raise BadNominalValue(value)


class NominalConversor:
    def __init__(self, values):
        self.values = set(values)
        self.zero_value = values[0]

    def __call__(self, value):
        if value not in self.values:
            if value == 0:
                # Sparse decode
                # See issue #52: nominals should take their first value when
                # unspecified in a sparse matrix. Naturally, this is consistent
                # with EncodedNominalConversor.
                return self.zero_value
            raise BadNominalValue(value)
        return str(value)


def _is_date_type(type_):
    '''(INTERNAL) Checks whether an attribute type string denotes a DATE
    attribute, with or without an explicit format.'''
    return isinstance(type_, str) and (
        type_ == 'DATE' or type_.startswith('DATE '))


class DateConversor:
    '''Conversor for DATE attributes.

    This conversor does not parse dates by itself; it delegates all parsing
    and formatting to a caller-provided adapter, which must implement::

        adapter.parse(value, fmt)   # str or number -> date value
        adapter.format(value, fmt)  # date value -> str

    ``fmt`` is the format declared in the attribute specification (e.g.
    ``@ATTRIBUTE when DATE "yyyy-MM-dd"``), or None for a bare ``DATE``.

    Explicit dates, missing values and omitted sparse entries keep their
    origins: missing values (``?``) are decoded as None before reaching
    this conversor, while the numeric zero generated by omitted entries of
    sparse data is passed to ``parse`` as a number and must be interpreted
    by the adapter as the epoch of the date type -- it is never treated as
    a missing value.

    The ``tz`` and ``version`` attributes of the adapter, when present, are
    frozen when the conversor is created: if they change in the middle of a
    reading (or writing) task, ``BadDateAdapter`` is raised. A new
    conversor is bound to the adapter on every task, so adapters are never
    shared between different tasks.
    '''

    _FROZEN_ATTRIBUTES = ('tz', 'version')

    def __init__(self, adapter, fmt=None):
        if adapter is None:
            raise BadDateAdapter(
                'a DATE attribute requires a date adapter.')
        self.adapter = adapter
        self.fmt = fmt
        self._frozen = {
            name: getattr(adapter, name)
            for name in self._FROZEN_ATTRIBUTES
            if hasattr(adapter, name)
        }

    def _check_frozen(self):
        for name, expected in self._frozen.items():
            if getattr(self.adapter, name, None) != expected:
                raise BadDateAdapter(
                    'date adapter attribute %r changed in the middle of a '
                    'single task.' % name)

    def __call__(self, value):
        self._check_frozen()
        try:
            return self.adapter.parse(value, self.fmt)
        except ArffException:
            raise
        except Exception as exc:
            raise BadDateValue(value) from exc

    def format(self, value):
        self._check_frozen()
        try:
            return self.adapter.format(value, self.fmt)
        except ArffException:
            raise
        except Exception as exc:
            raise BadDateValue(value) from exc


def _get_encode_conversors(attributes, date_adapter):
    '''(INTERNAL) Builds one encoding conversor per attribute. Attributes
    that do not need any special conversion get None.'''
    conversors = []
    for name, type_ in attributes:
        if _is_date_type(type_):
            if date_adapter is None:
                raise BadObject(
                    'DATE attribute "%s" requires a date adapter to be '
                    'encoded.' % name)
            fmt = type_[4:].strip() or None
            conversors.append(DateConversor(date_adapter, fmt))
        else:
            conversors.append(None)
    return conversors


class DenseGeneratorData:
    '''Internal helper class to allow for different matrix types without
    making the code a huge collection of if statements.'''

    def decode_rows(self, stream, conversors):
        for row in stream:
            values = _parse_values(row)

            if isinstance(values, dict):
                if values and max(values) >= len(conversors):
                    raise BadDataFormat(row)
                # XXX: int 0 is used for implicit values, not '0'
                values = [values[i] if i in values else 0 for i in
                          range(len(conversors))]
            else:
                if len(values) != len(conversors):
                    raise BadDataFormat(row)

            yield self._decode_values(values, conversors)

    @staticmethod
    def _decode_values(values, conversors):
        try:
            values = [None if value is None else conversor(value)
                      for conversor, value
                      in zip(conversors, values)]
        except ValueError as exc:
            if 'float: ' in str(exc):
                raise BadNumericalValue()
        return values

    def encode_data(self, data, attributes, date_adapter=None):
        '''(INTERNAL) Encodes a line of data.

        Data instances follow the csv format, i.e, attribute values are
        delimited by commas. After converted from csv.

        :param data: a list of values.
        :param attributes: a list of attributes. Used to check if data is valid.
        :param date_adapter: (OPTIONAL) an adapter used to format the values
            of DATE attributes.
        :return: a string with the encoded data line.
        '''
        current_row = 0
        conversors = _get_encode_conversors(attributes, date_adapter)

        for inst in data:
            if len(inst) != len(attributes):
                raise BadObject(
                    'Instance %d has %d attributes, expected %d' %
                     (current_row, len(inst), len(attributes))
                )

            new_data = []
            for value, conversor in zip(inst, conversors):
                if value is None or value == '' or value != value:
                    s = '?'
                elif conversor is not None:
                    s = encode_string(conversor.format(value))
                else:
                    s = encode_string(str(value))
                new_data.append(s)

            current_row += 1
            yield ','.join(new_data)


class _DataListMixin:
    """Mixin to return a list from decode_rows instead of a generator"""
    def decode_rows(self, stream, conversors):
        return list(super().decode_rows(stream, conversors))


class Data(_DataListMixin, DenseGeneratorData):
    pass


class COOData:
    def decode_rows(self, stream, conversors):
        data, rows, cols = [], [], []
        for i, row in enumerate(stream):
            values = _parse_values(row)
            if not isinstance(values, dict):
                raise BadLayout()
            if not values:
                continue
            row_cols, values = zip(*sorted(values.items()))
            try:
                values = [value if value is None else conversors[key](value)
                          for key, value in zip(row_cols, values)]
            except ValueError as exc:
                if 'float: ' in str(exc):
                    raise BadNumericalValue()
                raise
            except IndexError:
                # conversor out of range
                raise BadDataFormat(row)

            data.extend(values)
            rows.extend([i] * len(values))
            cols.extend(row_cols)

        return data, rows, cols

    def encode_data(self, data, attributes, date_adapter=None):
        num_attributes = len(attributes)
        new_data = []
        current_row = 0
        conversors = _get_encode_conversors(attributes, date_adapter)

        row = data.row
        col = data.col
        data = data.data

        # Check if the rows are sorted
        if not all(row[i] <= row[i + 1] for i in range(len(row) - 1)):
            raise ValueError("liac-arff can only output COO matrices with "
                             "sorted rows.")

        for v, col, row in zip(data, col, row):
            if row > current_row:
                # Add empty rows if necessary
                while current_row < row:
                    yield " ".join(["{", ','.join(new_data), "}"])
                    new_data = []
                    current_row += 1

            if col >= num_attributes:
                raise BadObject(
                    'Instance %d has at least %d attributes, expected %d' %
                    (current_row, col + 1, num_attributes)
                )

            if v is None or v == '' or v != v:
                s = '?'
            elif conversors[col] is not None:
                s = encode_string(conversors[col].format(v))
            else:
                s = encode_string(str(v))
            new_data.append("%d %s" % (col, s))

        yield " ".join(["{", ','.join(new_data), "}"])

class LODGeneratorData:
    def decode_rows(self, stream, conversors):
        for row in stream:
            values = _parse_values(row)

            if not isinstance(values, dict):
                raise BadLayout()
            try:
                yield {key: None if value is None else conversors[key](value)
                       for key, value in values.items()}
            except ValueError as exc:
                if 'float: ' in str(exc):
                    raise BadNumericalValue()
                raise
            except IndexError:
                # conversor out of range
                raise BadDataFormat(row)

    def encode_data(self, data, attributes, date_adapter=None):
        current_row = 0

        num_attributes = len(attributes)
        conversors = _get_encode_conversors(attributes, date_adapter)
        for row in data:
            new_data = []

            if len(row) > 0 and max(row) >= num_attributes:
                raise BadObject(
                    'Instance %d has %d attributes, expected %d' %
                    (current_row, max(row) + 1, num_attributes)
                )

            for col in sorted(row):
                v = row[col]
                if v is None or v == '' or v != v:
                    s = '?'
                elif conversors[col] is not None:
                    s = encode_string(conversors[col].format(v))
                else:
                    s = encode_string(str(v))
                new_data.append("%d %s" % (col, s))

            current_row += 1
            yield " ".join(["{", ','.join(new_data), "}"])

class LODData(_DataListMixin, LODGeneratorData):
    pass


def _get_data_object_for_decoding(matrix_type):
    if matrix_type == DENSE:
        return Data()
    elif matrix_type == COO:
        return COOData()
    elif matrix_type == LOD:
        return LODData()
    elif matrix_type == DENSE_GEN:
        return DenseGeneratorData()
    elif matrix_type == LOD_GEN:
        return LODGeneratorData()
    else:
        raise ValueError("Matrix type %s not supported." % str(matrix_type))

def _get_data_object_for_encoding(matrix):
    # Probably a scipy.sparse
    if hasattr(matrix, 'format'):
        if matrix.format == 'coo':
            return COOData()
        else:
            raise ValueError('Cannot guess matrix format!')
    elif isinstance(matrix[0], dict):
        return LODData()
    else:
        return Data()

# =============================================================================

# ADVANCED INTERFACE ==========================================================
class ArffDecoder:
    '''An ARFF decoder.'''

    def __init__(self):
        '''Constructor.'''
        self._conversors = []
        self._current_line = 0

    def _decode_comment(self, s):
        '''(INTERNAL) Decodes a comment line.

        Comments are single line strings starting, obligatorily, with the ``%``
        character, and can have any symbol, including whitespaces or special
        characters.

        This method must receive a normalized string, i.e., a string without
        padding, including the "\r\n" characters.

        :param s: a normalized string.
        :return: a string with the decoded comment.
        '''
        res = re.sub(r'^\%( )?', '', s)
        return res

    def _decode_relation(self, s):
        '''(INTERNAL) Decodes a relation line.

        The relation declaration is a line with the format ``@RELATION
        <relation-name>``, where ``relation-name`` is a string. The string must
        start with alphabetic character and must be quoted if the name includes
        spaces, otherwise this method will raise a `BadRelationFormat` exception.

        This method must receive a normalized string, i.e., a string without
        padding, including the "\r\n" characters.

        :param s: a normalized string.
        :return: a string with the decoded relation name.
        '''
        _, v = s.split(' ', 1)
        v = v.strip()

        if not _RE_RELATION.match(v):
            raise BadRelationFormat()

        res = str(v.strip('"\''))
        return res

    def _decode_attribute(self, s):
        '''(INTERNAL) Decodes an attribute line.

        The attribute is the most complex declaration in an arff file. All
        attributes must follow the template::

             @attribute <attribute-name> <datatype>

        where ``attribute-name`` is a string, quoted if the name contains any
        whitespace, and ``datatype`` can be:

        - Numerical attributes as ``NUMERIC``, ``INTEGER`` or ``REAL``.
        - Strings as ``STRING``.
        - Dates as ``DATE`` or ``DATE <format>``, where ``format`` is an
          arbitrary, optionally quoted, string interpreted by the date
          adapter provided to ``decode``.
        - Nominal attributes with format:

            {<nominal-name1>, <nominal-name2>, <nominal-name3>, ...}

        The nominal names follow the rules for the attribute names, i.e., they
        must be quoted if the name contains whitespaces.

        This method must receive a normalized string, i.e., a string without
        padding, including the "\r\n" characters.

        :param s: a normalized string.
        :return: a tuple (ATTRIBUTE_NAME, TYPE_OR_VALUES).
        '''
        _, v = s.split(' ', 1)
        v = v.strip()

        # Verify the general structure of declaration
        m = _RE_ATTRIBUTE.match(v)
        if not m:
            raise BadAttributeFormat()

        # Extracts the raw name and type
        name, type_ = m.groups()

        # Extracts the final name
        name = str(name.strip('"\''))

        # Extracts the final type
        if type_[:1] == "{" and type_[-1:] == "}":
            try:
                type_ = _parse_values(type_.strip('{} '))
            except Exception:
                raise BadAttributeType()
            if isinstance(type_, dict):
                raise BadAttributeType()

        else:
            # If not nominal, verify the type name
            type_ = str(type_)
            date_match = re.match(r'DATE(?:\s+(.*))?$', type_, re.IGNORECASE)
            if date_match:
                # Dates keep their (case sensitive) format string, which is
                # interpreted by the caller-provided date adapter
                fmt = (date_match.group(1) or '').strip()
                if fmt:
                    if fmt[:1] in ('"', "'"):
                        if len(fmt) < 2 or fmt[-1:] != fmt[:1]:
                            raise BadAttributeType()
                        fmt = _unquote(fmt)
                    type_ = 'DATE ' + fmt
                else:
                    type_ = 'DATE'
            else:
                type_ = type_.upper()
                if type_ not in ['NUMERIC', 'REAL', 'INTEGER', 'STRING']:
                    raise BadAttributeType()

        return (name, type_)

    def _decode(self, s, encode_nominal=False, matrix_type=DENSE,
                date_adapter=None):
        '''Do the job the ``encode``.'''

        # Make sure this method is idempotent
        self._current_line = 0
        # Never reuse conversors (in particular date adapters) from a
        # previous decoding task
        self._conversors = []

        # If string, convert to a list of lines
        if isinstance(s, str):
            s = s.strip('\r\n ').replace('\r\n', '\n').split('\n')

        # Create the return object
        obj = {
            'description': '',
            'relation': '',
            'attributes': [],
            'data': []
        }
        attribute_names = {}

        # Create the data helper object
        data = _get_data_object_for_decoding(matrix_type)

        # Read all lines
        STATE = _TK_DESCRIPTION
        s = iter(s)
        for row in s:
            self._current_line += 1
            # Ignore empty lines
            row = row.strip(' \r\n')
            if not row: continue

            u_row = row.upper()

            # DESCRIPTION -----------------------------------------------------
            if u_row.startswith(_TK_DESCRIPTION) and STATE == _TK_DESCRIPTION:
                obj['description'] += self._decode_comment(row) + '\n'
            # -----------------------------------------------------------------

            # RELATION --------------------------------------------------------
            elif u_row.startswith(_TK_RELATION):
                if STATE != _TK_DESCRIPTION:
                    raise BadLayout()

                STATE = _TK_RELATION
                obj['relation'] = self._decode_relation(row)
            # -----------------------------------------------------------------

            # ATTRIBUTE -------------------------------------------------------
            elif u_row.startswith(_TK_ATTRIBUTE):
                if STATE != _TK_RELATION and STATE != _TK_ATTRIBUTE:
                    raise BadLayout()

                STATE = _TK_ATTRIBUTE

                attr = self._decode_attribute(row)
                if attr[0] in attribute_names:
                    raise BadAttributeName(attr[0], attribute_names[attr[0]])
                else:
                    attribute_names[attr[0]] = self._current_line
                obj['attributes'].append(attr)

                if isinstance(attr[1], (list, tuple)):
                    if encode_nominal:
                        conversor = EncodedNominalConversor(attr[1])
                    else:
                        conversor = NominalConversor(attr[1])
                elif _is_date_type(attr[1]):
                    fmt = attr[1][4:].strip() or None
                    conversor = DateConversor(date_adapter, fmt)
                else:
                    CONVERSOR_MAP = {'STRING': str,
                                     'INTEGER': lambda x: int(float(x)),
                                     'NUMERIC': float,
                                     'REAL': float}
                    conversor = CONVERSOR_MAP[attr[1]]

                self._conversors.append(conversor)
            # -----------------------------------------------------------------

            # DATA ------------------------------------------------------------
            elif u_row.startswith(_TK_DATA):
                if STATE != _TK_ATTRIBUTE:
                    raise BadLayout()

                break
            # -----------------------------------------------------------------

            # COMMENT ---------------------------------------------------------
            elif u_row.startswith(_TK_COMMENT):
                pass
            # -----------------------------------------------------------------
        else:
            # Never found @DATA
            raise BadLayout()

        def stream():
            for row in s:
                self._current_line += 1
                row = row.strip()
                # Ignore empty lines and comment lines.
                if row and not row.startswith(_TK_COMMENT):
                    yield row

        # Alter the data object
        obj['data'] = data.decode_rows(stream(), self._conversors)
        if obj['description'].endswith('\n'):
            obj['description'] = obj['description'][:-1]

        return obj

    def decode(self, s, encode_nominal=False, return_type=DENSE,
               date_adapter=None):
        '''Returns the Python representation of a given ARFF file.

        When a file object is passed as an argument, this method reads lines
        iteratively, avoiding to load unnecessary information to the memory.

        :param s: a string or file object with the ARFF file.
        :param encode_nominal: boolean, if True perform a label encoding
            while reading the .arff file.
        :param return_type: determines the data structure used to store the
            dataset. Can be one of `arff.DENSE`, `arff.COO`, `arff.LOD`,
            `arff.DENSE_GEN` or `arff.LOD_GEN`.
            Consult the sections on `working with sparse data`_ and `loading
            progressively`_.
        :param date_adapter: (OPTIONAL) an object used to parse the values
            of DATE attributes. It must provide a ``parse(value, fmt)``
            method, where ``value`` is the string found in the ARFF file
            (or the number 0 for entries omitted in sparse data, which must
            be interpreted as the epoch of the date type) and ``fmt`` is
            the format declared in the attribute specification (or None).
        '''
        try:
            return self._decode(s, encode_nominal=encode_nominal,
                                matrix_type=return_type,
                                date_adapter=date_adapter)
        except ArffException as e:
            e.line = self._current_line
            raise e


class ArffEncoder:
    '''An ARFF encoder.'''

    def _encode_comment(self, s=''):
        '''(INTERNAL) Encodes a comment line.

        Comments are single line strings starting, obligatorily, with the ``%``
        character, and can have any symbol, including whitespaces or special
        characters.

        If ``s`` is None, this method will simply return an empty comment.

        :param s: (OPTIONAL) string.
        :return: a string with the encoded comment line.
        '''
        if s:
            return '%s %s'%(_TK_COMMENT, s)
        else:
            return '%s' % _TK_COMMENT

    def _encode_relation(self, name):
        '''(INTERNAL) Decodes a relation line.

        The relation declaration is a line with the format ``@RELATION
        <relation-name>``, where ``relation-name`` is a string.

        :param name: a string.
        :return: a string with the encoded relation declaration.
        '''
        for char in ' %{},':
            if char in name:
                name = '"%s"'%name
                break

        return '%s %s'%(_TK_RELATION, name)

    def _encode_attribute(self, name, type_):
        '''(INTERNAL) Encodes an attribute line.

        The attribute follow the template::

             @attribute <attribute-name> <datatype>

        where ``attribute-name`` is a string, and ``datatype`` can be:

        - Numerical attributes as ``NUMERIC``, ``INTEGER`` or ``REAL``.
        - Strings as ``STRING``.
        - Dates as ``DATE`` or ``DATE <format>``.
        - Nominal attributes with format:

            {<nominal-name1>, <nominal-name2>, <nominal-name3>, ...}

        This method must receive a the name of the attribute and its type, if
        the attribute type is nominal, ``type`` must be a list of values.

        :param name: a string.
        :param type_: a string or a list of string.
        :return: a string with the encoded attribute declaration.
        '''
        for char in ' %{},':
            if char in name:
                name = '"%s"'%name
                break

        if isinstance(type_, (tuple, list)):
            type_tmp = ['%s' % encode_string(type_k) for type_k in type_]
            type_ = '{%s}'%(', '.join(type_tmp))
        elif _is_date_type(type_):
            fmt = type_[4:].strip()
            type_ = 'DATE %s'%encode_string(fmt) if fmt else 'DATE'

        return '%s %s %s'%(_TK_ATTRIBUTE, name, type_)

    def encode(self, obj, date_adapter=None):
        '''Encodes a given object to an ARFF file.

        :param obj: the object containing the ARFF information.
        :param date_adapter: (OPTIONAL) an object used to format the values
            of DATE attributes. It must provide a ``format(value, fmt)``
            method, where ``fmt`` is the format declared in the attribute
            specification (or None).
        :return: the ARFF file as an string.
        '''
        data = [row for row in self.iter_encode(obj, date_adapter=date_adapter)]

        return '\n'.join(data)

    def iter_encode(self, obj, date_adapter=None):
        '''The iterative version of `arff.ArffEncoder.encode`.

        This encodes iteratively a given object and return, one-by-one, the
        lines of the ARFF file.

        :param obj: the object containing the ARFF information.
        :param date_adapter: (OPTIONAL) an object used to format the values
            of DATE attributes.
        :return: (yields) the ARFF file as strings.
        '''
        # DESCRIPTION
        if obj.get('description', None):
            for row in obj['description'].split('\n'):
                yield self._encode_comment(row)

        # RELATION
        if not obj.get('relation'):
            raise BadObject('Relation name not found or with invalid value.')

        yield self._encode_relation(obj['relation'])
        yield ''

        # ATTRIBUTES
        if not obj.get('attributes'):
            raise BadObject('Attributes not found.')

        attribute_names = set()
        for attr in obj['attributes']:
            # Verify for bad object format
            if not isinstance(attr, (tuple, list)) or \
               len(attr) != 2 or \
               not isinstance(attr[0], str):
                raise BadObject('Invalid attribute declaration "%s"'%str(attr))

            if isinstance(attr[1], str):
                # Verify for invalid types
                if attr[1] not in _SIMPLE_TYPES and not _is_date_type(attr[1]):
                    raise BadObject('Invalid attribute type "%s"'%str(attr))

            # Verify for bad object format
            elif not isinstance(attr[1], (tuple, list)):
                raise BadObject('Invalid attribute type "%s"'%str(attr))

            # Verify attribute name is not used twice
            if attr[0] in attribute_names:
                raise BadObject('Trying to use attribute name "%s" for the '
                                'second time.' % str(attr[0]))
            else:
                attribute_names.add(attr[0])

            yield self._encode_attribute(attr[0], attr[1])
        yield ''
        attributes = obj['attributes']

        # DATA
        yield _TK_DATA
        if 'data' in obj:
            data = _get_data_object_for_encoding(obj.get('data'))
            yield from data.encode_data(obj.get('data'), attributes,
                                        date_adapter=date_adapter)

        yield ''

# =============================================================================

# BASIC INTERFACE =============================================================
def load(fp, encode_nominal=False, return_type=DENSE, date_adapter=None):
    '''Load a file-like object containing the ARFF document and convert it into
    a Python object.

    :param fp: a file-like object.
    :param encode_nominal: boolean, if True perform a label encoding
        while reading the .arff file.
    :param return_type: determines the data structure used to store the
        dataset. Can be one of `arff.DENSE`, `arff.COO`, `arff.LOD`,
        `arff.DENSE_GEN` or `arff.LOD_GEN`.
        Consult the sections on `working with sparse data`_ and `loading
        progressively`_.
    :param date_adapter: (OPTIONAL) an object used to parse the values of
        DATE attributes; see `arff.ArffDecoder.decode`.
    :return: a dictionary.
     '''
    decoder = ArffDecoder()
    return decoder.decode(fp, encode_nominal=encode_nominal,
                          return_type=return_type, date_adapter=date_adapter)

def loads(s, encode_nominal=False, return_type=DENSE, date_adapter=None):
    '''Convert a string instance containing the ARFF document into a Python
    object.

    :param s: a string object.
    :param encode_nominal: boolean, if True perform a label encoding
        while reading the .arff file.
    :param return_type: determines the data structure used to store the
        dataset. Can be one of `arff.DENSE`, `arff.COO`, `arff.LOD`,
        `arff.DENSE_GEN` or `arff.LOD_GEN`.
        Consult the sections on `working with sparse data`_ and `loading
        progressively`_.
    :param date_adapter: (OPTIONAL) an object used to parse the values of
        DATE attributes; see `arff.ArffDecoder.decode`.
    :return: a dictionary.
    '''
    decoder = ArffDecoder()
    return decoder.decode(s, encode_nominal=encode_nominal,
                          return_type=return_type, date_adapter=date_adapter)

def dump(obj, fp, date_adapter=None):
    '''Serialize an object representing the ARFF document to a given file-like
    object.

    :param obj: a dictionary.
    :param fp: a file-like object.
    :param date_adapter: (OPTIONAL) an object used to format the values of
        DATE attributes; see `arff.ArffEncoder.encode`.
    '''
    encoder = ArffEncoder()
    generator = encoder.iter_encode(obj, date_adapter=date_adapter)

    last_row = next(generator)
    for row in generator:
        fp.write(last_row + '\n')
        last_row = row
    fp.write(last_row)

    return fp

def dumps(obj, date_adapter=None):
    '''Serialize an object representing the ARFF document, returning a string.

    :param obj: a dictionary.
    :param date_adapter: (OPTIONAL) an object used to format the values of
        DATE attributes; see `arff.ArffEncoder.encode`.
    :return: a string with the ARFF document.
    '''
    encoder = ArffEncoder()
    return encoder.encode(obj, date_adapter=date_adapter)
# =============================================================================
