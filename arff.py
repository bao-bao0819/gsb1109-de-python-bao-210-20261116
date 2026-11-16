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
strings.

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
- Supports the following attribute types: NUMERIC, REAL, INTEGER, STRING, and
  NOMINAL;
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
import os
import json
import copy
import hashlib
import tempfile
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


class IncompleteShards(ArffException):
    '''Error raised when a sharded dataset is published before every shard has
    been accepted.'''
    def __init__(self, accepted, expected):
        super().__init__()
        self.accepted = accepted
        self.expected = expected
        if expected is None:
            self.message = (
                'Cannot publish an incomplete dataset: the shard sequence is '
                'not sealed (accepted %d shards), at line %%d.' % accepted
            )
        else:
            self.message = (
                'Cannot publish an incomplete dataset: accepted %d of %d '
                'shards, at line %%d.' % (accepted, expected)
            )


class IncompleteNominalDomain(ArffException):
    '''Error raised when accepted data uses categories that no declaration has
    covered yet.'''
    def __init__(self, pending):
        super().__init__()
        self.pending = {name: list(values)
                        for name, values in pending.items()}
        details = ', '.join('%s=%r' % (name, values)
                            for name, values in sorted(self.pending.items()))
        self.message = (
            'Cannot publish an incomplete dataset: nominal domain is missing '
            '%s, at line %%d.' % details.replace('%', '%%')
        )


class ShardConflict(ArffException):
    '''Error raised when a shard disagrees with the declared schema or with a
    previously accepted shard.'''
    def __init__(self, msg):
        super().__init__()
        self.message = msg.replace('%', '%%') + ' at line %d.'

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

    def encode_data(self, data, attributes):
        '''(INTERNAL) Encodes a line of data.

        Data instances follow the csv format, i.e, attribute values are
        delimited by commas. After converted from csv.

        :param data: a list of values.
        :param attributes: a list of attributes. Used to check if data is valid.
        :return: a string with the encoded data line.
        '''
        current_row = 0

        for inst in data:
            if len(inst) != len(attributes):
                raise BadObject(
                    'Instance %d has %d attributes, expected %d' %
                     (current_row, len(inst), len(attributes))
                )

            new_data = []
            for value in inst:
                if value is None or value == '' or value != value:
                    s = '?'
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

    def encode_data(self, data, attributes):
        num_attributes = len(attributes)
        new_data = []
        current_row = 0

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

    def encode_data(self, data, attributes):
        current_row = 0

        num_attributes = len(attributes)
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
        - Dates (NOT IMPLEMENTED).
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
            type_ = str(type_).upper()
            if type_ not in ['NUMERIC', 'REAL', 'INTEGER', 'STRING']:
                raise BadAttributeType()

        return (name, type_)

    def _decode(self, s, encode_nominal=False, matrix_type=DENSE):
        '''Do the job the ``encode``.'''

        # Make sure this method is idempotent
        self._current_line = 0

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

    def decode(self, s, encode_nominal=False, return_type=DENSE):
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
        '''
        try:
            return self._decode(s, encode_nominal=encode_nominal,
                                matrix_type=return_type)
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
        - Dates (NOT IMPLEMENTED).
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

        return '%s %s %s'%(_TK_ATTRIBUTE, name, type_)

    def encode(self, obj):
        '''Encodes a given object to an ARFF file.

        :param obj: the object containing the ARFF information.
        :return: the ARFF file as an string.
        '''
        data = [row for row in self.iter_encode(obj)]

        return '\n'.join(data)

    def iter_encode(self, obj):
        '''The iterative version of `arff.ArffEncoder.encode`.

        This encodes iteratively a given object and return, one-by-one, the
        lines of the ARFF file.

        :param obj: the object containing the ARFF information.
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
                if attr[1] not in _SIMPLE_TYPES:
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
            yield from data.encode_data(obj.get('data'), attributes)

        yield ''

# =============================================================================

# SHARDED INGESTION ==========================================================
_SHARD_STATE_FORMAT = 'liac-arff-shard-state'
_SHARD_STATE_VERSION = 1


class _TrackingNominalConversor:
    '''A nominal conversor that accepts values outside the current declaration
    and records them instead of raising `BadNominalValue`.

    It mirrors `NominalConversor` for all declared values and for the implicit
    first value used by sparse rows, so validating a shard through the regular
    decoding machinery produces the same errors for everything else.
    '''

    def __init__(self, values, unknown):
        self.values = set(values)
        self.zero_value = values[0]
        self.unknown = unknown

    def __call__(self, value):
        if value == 0:
            return self.zero_value
        value = str(value)
        if value not in self.values:
            self.unknown.append(value)
        return value


def _conversor_for_type(type_, encode_nominal=False):
    if isinstance(type_, list):
        if encode_nominal:
            return EncodedNominalConversor(type_)
        return NominalConversor(type_)
    return {
        'STRING': str,
        'INTEGER': lambda x: int(float(x)),
        'NUMERIC': float,
        'REAL': float,
    }[type_]


def _normalize_attributes(attributes):
    '''Validate and normalize an attribute declaration.

    Returns a list of ``(name, type)`` tuples where simple types are upper
    cased and nominal types are lists.
    '''
    if not attributes:
        raise BadObject('Attributes not found.')

    if isinstance(attributes, dict):
        attributes = list(attributes.items())

    result = []
    names = set()
    for attr in attributes:
        if not isinstance(attr, (tuple, list)) or len(attr) != 2:
            raise BadObject('Invalid attribute declaration "%s"' % str(attr))
        name, type_ = attr

        if not isinstance(name, str):
            raise BadObject('Invalid attribute declaration "%s"' % str(attr))
        if name in names:
            raise BadObject('Trying to use attribute name "%s" for the '
                            'second time.' % name)
        names.add(name)

        if isinstance(type_, str):
            type_ = type_.upper()
            if type_ not in _SIMPLE_TYPES:
                raise BadObject('Invalid attribute type "%s"' % str(attr))
        elif isinstance(type_, (tuple, list)):
            values = [str(value) for value in type_]
            if not values:
                raise BadObject('Invalid attribute type "%s"' % str(attr))
            type_ = values
        else:
            raise BadObject('Invalid attribute type "%s"' % str(attr))

        result.append((name, type_))

    return result


def _merge_attribute_declarations(current, declared):
    '''Merge a shard's attribute declaration into the accepted declaration.

    Attribute names, order and non-nominal types are fixed. A nominal domain
    may only grow; when it does, the shard's order is adopted so that the final
    interpretation matches a full import using that declaration.
    '''
    if len(current) != len(declared):
        raise ShardConflict('Shard declares %d attributes, expected %d' %
                            (len(declared), len(current)))

    merged = []
    for (cur_name, cur_type), (new_name, new_type) in zip(current, declared):
        if cur_name != new_name:
            raise ShardConflict('Shard declares attribute %r, expected %r' %
                                (new_name, cur_name))

        if isinstance(cur_type, list):
            if not isinstance(new_type, list):
                raise ShardConflict('Attribute %r changed from nominal to %s' %
                                    (cur_name, new_type))
            missing = [value for value in cur_type if value not in new_type]
            if missing:
                raise ShardConflict(
                    'Declaration for attribute %r no longer contains %r' %
                    (cur_name, missing[0]))
            merged.append((cur_name, list(new_type)))
        else:
            if new_type != cur_type:
                raise ShardConflict('Attribute %r changed type from %s to %s' %
                                    (cur_name, cur_type, new_type))
            merged.append((cur_name, cur_type))

    return merged


class ArffShardAccumulator:
    '''Incrementally accept ARFF shards and publish one coherent dataset.

    Shards are accepted atomically: a shard that fails to parse or validate
    leaves the accumulator unchanged. Nominal domains may grow as later shards
    arrive; a category present in accepted data but not present in any
    declaration is kept in ``pending_nominals``. Pending categories are data
    that has been received, not part of the published declaration. Data rows
    are retained verbatim and are decoded only against the final declaration,
    so their nominal interpretation is identical to importing the concatenated
    document in one pass.

    Publication requires both a known end and a complete category domain:
    either ``expected_shards`` is reached or an open-ended accumulator is
    explicitly sealed, and no category remains pending. A pending category can
    be resolved by a later shard header or by :meth:`declare_attributes`. An
    incomplete accumulator can be checkpointed and resumed, but :meth:`publish`
    refuses to write a partial dataset or a half declaration.

    :param relation: optional relation name; a first shard may also supply it.
    :param attributes: optional attribute declaration, allowing header-less
        shards from the start.
    :param description: optional dataset description.
    :param expected_shards: number of shards that make the dataset complete,
        or ``None`` for an open-ended accumulator that must be sealed.
    '''

    def __init__(self, relation=None, attributes=None, description=None,
                 expected_shards=None):
        if expected_shards is not None:
            if (not isinstance(expected_shards, int) or
                    isinstance(expected_shards, bool) or expected_shards < 1):
                raise ValueError('expected_shards must be a positive integer '
                                 'or None.')

        self._relation = None if relation is None else str(relation)
        self._attributes = (None if attributes is None
                            else _normalize_attributes(attributes))
        self._description = '' if description is None else str(description)
        self._description_locked = description is not None

        self._data_lines = []
        self._pending_nominals = {}
        self._accepted_shards = []
        self._expected_shards = expected_shards
        self._sealed = False

    @property
    def relation(self):
        return self._relation

    @property
    def attributes(self):
        '''The accepted (fixed) declaration, without pending categories.'''
        if self._attributes is None:
            return None
        return copy.deepcopy(self._attributes)

    @property
    def pending_nominals(self):
        '''Categories seen in accepted data but not yet declared.'''
        return copy.deepcopy(self._pending_nominals)

    @property
    def accepted_shards(self):
        return copy.deepcopy(self._accepted_shards)

    @property
    def accepted_rows(self):
        return len(self._data_lines)

    @property
    def sealed(self):
        return self._sealed

    @property
    def complete(self):
        '''Whether the dataset may be published.

        This requires every expected shard to be accepted (or an open-ended
        accumulator to be sealed) *and* no category to remain pending.
        '''
        if self._expected_shards is not None:
            if len(self._accepted_shards) != self._expected_shards:
                return False
        elif not (self._sealed and self._accepted_shards):
            return False
        return not self._pending_nominals

    def seal(self):
        '''Declare that no further shards will be supplied.

        For an accumulator with ``expected_shards`` this only marks the end of
        input; it still remains incomplete if fewer shards were accepted.
        '''
        self._sealed = True
        return self.complete

    def declare_attributes(self, attributes):
        '''Complete the fixed declaration with categories still pending.

        This is the explicit fill-in step for an open-ended stream whose last
        shard introduced categories but had no header. The supplied declaration
        must keep every existing attribute and may only grow nominal domains.
        It is applied atomically.
        '''
        if self._attributes is None:
            raise BadLayout('No attribute declaration found in shard')

        declared = _normalize_attributes(attributes)
        merged = _merge_attribute_declarations(self._attributes, declared)

        pending = copy.deepcopy(self._pending_nominals)
        for name, type_ in merged:
            if isinstance(type_, list):
                remaining = [value for value in pending.get(name, [])
                             if value not in type_]
                if remaining:
                    pending[name] = remaining
                else:
                    pending.pop(name, None)

        if pending:
            raise IncompleteNominalDomain(pending)

        self._attributes = merged
        self._pending_nominals = pending
        return self.attributes

    def state(self):
        '''Return a JSON-serializable snapshot of the accumulator.

        ``data`` contains only rows that were fully accepted. Categories that
        are not part of ``attributes`` yet are kept separately in
        ``pending_nominals``; nothing in this snapshot is a publishable ARFF
        partition while ``complete`` is false.
        '''
        next_row = 0
        shards = []
        for shard in self._accepted_shards:
            record = copy.deepcopy(shard)
            record['first_row'] = next_row
            next_row += record['rows']
            shards.append(record)

        return {
            'format': _SHARD_STATE_FORMAT,
            'version': _SHARD_STATE_VERSION,
            'status': 'complete' if self.complete else 'partial',
            'relation': self._relation,
            'description': self._description,
            'attributes': copy.deepcopy(self._attributes),
            'pending_nominals': copy.deepcopy(self._pending_nominals),
            'data': list(self._data_lines),
            'accepted_shards': shards,
            'expected_shards': self._expected_shards,
            'sealed': self._sealed,
            'complete': self.complete,
        }

    def checkpoint(self, fp):
        '''Write a resumable JSON checkpoint to a path or file-like object.

        A checkpoint is deliberately not an ARFF document. Writing it to a
        path is atomic, so an interrupted write cannot replace a previous
        valid checkpoint.
        '''
        text = json.dumps(self.state(), ensure_ascii=False, indent=2) + '\n'
        if hasattr(fp, 'write'):
            fp.write(text)
        else:
            _atomic_write(fp, text)

    @classmethod
    def from_state(cls, state):
        '''Restore an accumulator from :meth:`state` output.'''
        if not isinstance(state, dict):
            raise ValueError('Shard state must be a mapping.')
        if state.get('format') != _SHARD_STATE_FORMAT:
            raise ValueError('Not a liac-arff shard state document.')
        if state.get('version') != _SHARD_STATE_VERSION:
            raise ValueError('Unsupported shard state version: %r' %
                             (state.get('version'),))

        expected_shards = state.get('expected_shards')
        if expected_shards is not None and (
                not isinstance(expected_shards, int) or
                isinstance(expected_shards, bool) or expected_shards < 1):
            raise ValueError('Invalid expected_shards in shard state.')

        relation = state.get('relation')
        if relation is not None and not isinstance(relation, str):
            raise ValueError('Invalid relation in shard state.')
        attributes = state.get('attributes')
        if attributes is not None and not isinstance(attributes, list):
            raise ValueError('Invalid attributes in shard state.')

        accumulator = cls(
            relation=relation,
            attributes=attributes,
            description=state.get('description', ''),
            expected_shards=expected_shards,
        )
        pending = state.get('pending_nominals', {})
        if not isinstance(pending, dict):
            raise ValueError('pending_nominals must be a mapping.')
        accumulator._pending_nominals = {
            str(name): [str(value) for value in values]
            for name, values in pending.items()
        }

        data = state.get('data', [])
        if not isinstance(data, list):
            raise ValueError('data must be a list.')
        accumulator._data_lines = [str(line) for line in data]

        shards = state.get('accepted_shards', [])
        if not isinstance(shards, list):
            raise ValueError('accepted_shards must be a list.')
        accumulator._accepted_shards = []
        for shard in shards:
            if (not isinstance(shard, dict) or 'id' not in shard or
                    not isinstance(shard.get('rows'), int) or
                    shard['rows'] < 0):
                raise ValueError('Invalid accepted shard record.')
            record = {
                'id': shard['id'],
                'digest': shard.get('digest', ''),
                'rows': shard['rows'],
            }
            accumulator._accepted_shards.append(record)

        if expected_shards is not None and (
                len(accumulator._accepted_shards) > expected_shards):
            raise ValueError('Shard state has more accepted shards than '
                             'expected.')
        if sum(record['rows'] for record in accumulator._accepted_shards) != \
                len(accumulator._data_lines):
            raise ValueError('Shard state row counts do not match its data.')

        accumulator._sealed = bool(state.get('sealed', False))
        return accumulator

    @classmethod
    def from_checkpoint(cls, fp):
        '''Read a checkpoint written by :meth:`checkpoint`.'''
        if hasattr(fp, 'read'):
            state = json.load(fp)
        else:
            with open(fp, 'r', encoding='utf-8') as stream:
                state = json.load(stream)
        return cls.from_state(state)

    def add_shard(self, shard, shard_id=None):
        '''Parse, validate and accept one shard.

        The operation is atomic: on any error the accumulator is unchanged. A
        shard header may repeat the fixed declaration and extend nominal
        domains. A shard without a header is interpreted with the accepted
        declaration.

        Re-supplying an already accepted ``shard_id`` with identical content is
        a no-op, which makes replay after a crash safe. The same id with
        different content is rejected.

        :return: ``True`` if the shard was newly accepted, ``False`` for an
            idempotent replay.
        '''
        text = _read_shard_text(shard)
        digest = _shard_digest(text)

        if shard_id is not None:
            for record in self._accepted_shards:
                if record['id'] == shard_id:
                    if record.get('digest') == digest:
                        return False
                    raise ShardConflict(
                        'Shard %r was already accepted with different content'
                        % (shard_id,))
        else:
            shard_id = self._next_shard_id()

        if (self._expected_shards is not None and
                len(self._accepted_shards) >= self._expected_shards):
            raise ShardConflict('All %d expected shards were already accepted'
                                % self._expected_shards)
        if self._sealed:
            raise ShardConflict('Shard sequence is already sealed')

        parsed = _parse_shard(text)

        relation = self._relation
        if parsed['relation'] is not None:
            if relation is not None and relation != parsed['relation']:
                raise ShardConflict('Shard relation %r differs from %r' %
                                    (parsed['relation'], relation))
            relation = parsed['relation']
        if relation is None:
            raise BadLayout('No relation declaration found in shard')

        if parsed['attributes'] is not None:
            declared = _normalize_attributes(parsed['attributes'])
            if self._attributes is None:
                attributes = declared
            else:
                attributes = _merge_attribute_declarations(self._attributes,
                                                           declared)
        else:
            if self._attributes is None:
                raise BadLayout('No attribute declaration found in shard')
            attributes = copy.deepcopy(self._attributes)

        pending = copy.deepcopy(self._pending_nominals)
        for name, type_ in attributes:
            if isinstance(type_, list):
                remaining = [value for value in pending.get(name, [])
                             if value not in type_]
                if remaining:
                    pending[name] = remaining
                else:
                    pending.pop(name, None)

        unknown = {}
        conversors = []
        for name, type_ in attributes:
            if isinstance(type_, list):
                bucket = unknown.setdefault(name, [])
                conversors.append(_TrackingNominalConversor(type_, bucket))
            else:
                conversors.append(_conversor_for_type(type_))

        decoder = DenseGeneratorData()
        for line, offset in zip(parsed['data'], parsed['data_offsets']):
            try:
                list(decoder.decode_rows([line], conversors))
            except ArffException as exc:
                if exc.line in (-1, None):
                    exc.line = offset
                raise

        for name, values in unknown.items():
            if not values:
                continue
            bucket = pending.setdefault(name, [])
            seen = set(bucket)
            for value in values:
                if value not in seen:
                    bucket.append(value)
                    seen.add(value)

        if parsed['description'] is not None and not self._description_locked:
            self._description = parsed['description']
            self._description_locked = True
        self._relation = relation
        self._attributes = attributes
        self._pending_nominals = pending
        self._data_lines.extend(parsed['data'])
        self._accepted_shards.append({
            'id': shard_id,
            'digest': digest,
            'rows': len(parsed['data']),
        })
        return True

    def _next_shard_id(self):
        used = {record['id'] for record in self._accepted_shards}
        candidate = len(self._accepted_shards)
        while candidate in used:
            candidate += 1
        return candidate

    def final_attributes(self):
        '''Return the fully declared attributes.

        Raises `IncompleteShards` until every shard is present and
        `IncompleteNominalDomain` while categories remain pending, because a
        declaration is not final before both conditions hold.
        '''
        self._require_complete()
        return copy.deepcopy(self._attributes)

    def finalize(self, encode_nominal=False, return_type=DENSE):
        '''Return the dataset object once the dataset is complete.

        The result is produced with the same decoder path and the same final
        declaration as a single full import of the concatenated document.
        '''
        self._require_complete()
        attributes = copy.deepcopy(self._attributes)

        conversors = []
        for _, type_ in attributes:
            conversors.append(_conversor_for_type(
                type_, encode_nominal=encode_nominal))

        data = _get_data_object_for_decoding(return_type)
        return {
            'description': self._description,
            'relation': self._relation,
            'attributes': attributes,
            'data': data.decode_rows(iter(self._data_lines), conversors),
        }

    def publish(self, fp=None):
        '''Finalize and optionally write the complete ARFF document.

        An incomplete accumulator raises `IncompleteShards` before anything is
        written. A path is replaced atomically; a file-like object receives the
        whole document only after it has been fully encoded.
        '''
        obj = self.finalize(encode_nominal=False, return_type=DENSE)
        text = dumps(obj)

        if fp is None:
            return obj

        if hasattr(fp, 'write'):
            fp.write(text)
        else:
            _atomic_write(fp, text)
        return obj

    def _require_complete(self):
        if self._expected_shards is not None:
            if len(self._accepted_shards) != self._expected_shards:
                raise IncompleteShards(len(self._accepted_shards),
                                       self._expected_shards)
        elif not (self._sealed and self._accepted_shards):
            raise IncompleteShards(len(self._accepted_shards), None)
        if self._pending_nominals:
            raise IncompleteNominalDomain(self._pending_nominals)


def _read_shard_text(shard):
    if hasattr(shard, 'read'):
        shard = shard.read()
    if isinstance(shard, bytes):
        shard = shard.decode('utf-8')
    if not isinstance(shard, str):
        raise TypeError('A shard must be a string, bytes or file-like object.')
    return shard


def _shard_digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _atomic_write(path, text):
    path = os.fspath(path)
    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp_path = tempfile.mkstemp(prefix='.arff-', dir=directory)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _header_present(lines):
    for row in lines:
        u_row = row.strip(' \r\n').upper()
        if (u_row.startswith(_TK_RELATION) or
                u_row.startswith(_TK_ATTRIBUTE) or
                u_row.startswith(_TK_DATA)):
            return True
    return False


def _parse_shard(text):
    '''Parse a shard into its header pieces and raw data lines.

    A shard may be a full ARFF document or a header-less data fragment. It is
    parsed independently rather than through `ArffDecoder` so that unknown
    nominal values can be recorded rather than rejected; the data lines
    themselves are retained verbatim for final decoding.
    '''
    text = text.strip('\r\n ')
    lines = text.replace('\r\n', '\n').split('\n')
    header = _header_present(lines)

    description = None
    relation = None
    attributes = []
    data = []
    data_offsets = []
    in_data = False
    attribute_names = {}

    for offset, raw_row in enumerate(lines, 1):
        row = raw_row.strip(' \r\n')
        if not row:
            continue
        u_row = row.upper()

        if in_data:
            if not row.startswith(_TK_COMMENT):
                data.append(row)
                data_offsets.append(offset)
            continue

        if header:
            if u_row.startswith(_TK_DESCRIPTION):
                if not attributes and relation is None:
                    if description is None:
                        description = ''
                    description += '%s\n' % ArffDecoder()._decode_comment(row)
            elif u_row.startswith(_TK_RELATION):
                if relation is not None or attributes:
                    raise BadLayout()
                decoder = ArffDecoder()
                try:
                    relation = decoder._decode_relation(row)
                except ArffException as exc:
                    if exc.line in (-1, None):
                        exc.line = offset
                    raise
            elif u_row.startswith(_TK_ATTRIBUTE):
                if relation is None:
                    raise BadLayout()
                decoder = ArffDecoder()
                try:
                    attr = decoder._decode_attribute(row)
                except ArffException as exc:
                    if exc.line in (-1, None):
                        exc.line = offset
                    raise
                if attr[0] in attribute_names:
                    raise BadAttributeName(attr[0], attribute_names[attr[0]])
                attribute_names[attr[0]] = offset
                attributes.append(attr)
            elif u_row.startswith(_TK_DATA):
                if relation is None:
                    raise BadLayout()
                if not attributes:
                    attributes = None
                in_data = True
            elif u_row.startswith(_TK_COMMENT):
                continue
            else:
                raise BadLayout('Unrecognized shard header %r' % row)
        else:
            if u_row.startswith(_TK_DATA):
                in_data = True
            elif row.startswith(_TK_COMMENT):
                continue
            else:
                data.append(row)
                data_offsets.append(offset)

    if header and not in_data:
        raise BadLayout()
    if not header:
        # A header-less shard reuses the accepted declaration; its rows and
        # their offsets were collected above.
        attributes = None
    elif attributes == []:
        if relation is not None:
            raise BadLayout('No attribute declaration found in shard')
        # A bare '@DATA' marker reuses the accepted declaration.
        attributes = None
    if description is not None and description.endswith('\n'):
        description = description[:-1]

    return {
        'description': description,
        'relation': relation,
        'attributes': attributes,
        'data': data,
        'data_offsets': data_offsets,
    }


def load_shards(shards, expected_shards=None, encode_nominal=False,
                return_type=DENSE):
    '''Convenience wrapper around :class:`ArffShardAccumulator`.

    :param shards: an iterable of shard strings or file-like objects.
    '''
    accumulator = ArffShardAccumulator(expected_shards=expected_shards)
    for index, shard in enumerate(shards):
        accumulator.add_shard(shard, shard_id=index)
    accumulator.seal()
    return accumulator.finalize(encode_nominal=encode_nominal,
                                return_type=return_type)

# =============================================================================

# BASIC INTERFACE =============================================================
def load(fp, encode_nominal=False, return_type=DENSE):
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
    :return: a dictionary.
     '''
    decoder = ArffDecoder()
    return decoder.decode(fp, encode_nominal=encode_nominal,
                          return_type=return_type)

def loads(s, encode_nominal=False, return_type=DENSE):
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
    :return: a dictionary.
    '''
    decoder = ArffDecoder()
    return decoder.decode(s, encode_nominal=encode_nominal,
                          return_type=return_type)

def dump(obj, fp):
    '''Serialize an object representing the ARFF document to a given file-like
    object.

    :param obj: a dictionary.
    :param fp: a file-like object.
    '''
    encoder = ArffEncoder()
    generator = encoder.iter_encode(obj)

    last_row = next(generator)
    for row in generator:
        fp.write(last_row + '\n')
        last_row = row
    fp.write(last_row)

    return fp

def dumps(obj):
    '''Serialize an object representing the ARFF document, returning a string.

    :param obj: a dictionary.
    :return: a string with the ARFF document.
    '''
    encoder = ArffEncoder()
    return encoder.encode(obj)
# =============================================================================
