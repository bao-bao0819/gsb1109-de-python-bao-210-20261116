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
import math
import numbers
import hashlib
import json
from collections import Counter
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

# IMPORT COMPARISON ===========================================================
# An import result is the dictionary returned by `load`/`loads` (or an
# equivalent object).  Incremental importers need two independent answers:
#
# * Is the observed data the same?
# * Which metadata version describes it?
#
# The conventions below are intentionally explicit:
#
# * Nominal *label order* is metadata.  It belongs to `metadata_version`, but
#   not to data equivalence when values are symbolic.  When values were decoded
#   with `encode_nominal=True`, integers are positions in that declaration, so
#   reordering the labels changes what an unchanged integer means and is
#   therefore a data change.  Replacing labels is always a data change, even
#   when the integer positions are unchanged.
# * Missing values are one value.  ``None``, ``?``, the empty string and NaN
#   all compare equal.
# * Row order is ignored by default and rows are compared as a multiset, so
#   duplicate rows still count.  Pass ``row_order='ordered'`` for a sequence
#   comparison.
# * Relation names, descriptions/comments and attribute names are metadata, not
#   data.  A changed or moved comment produces a new `metadata_version` but
#   never a re-import.  Attribute order is structural: columns are compared
#   positionally.
# * Columns are compared positionally.

ROW_ORDER_UNORDERED = 'unordered'
ROW_ORDER_ORDERED = 'ordered'

_ROW_ORDER_ALIASES = {
    None: ROW_ORDER_UNORDERED,
    False: ROW_ORDER_UNORDERED,
    True: ROW_ORDER_ORDERED,
    'unordered': ROW_ORDER_UNORDERED,
    'ignore': ROW_ORDER_UNORDERED,
    'ignored': ROW_ORDER_UNORDERED,
    'multiset': ROW_ORDER_UNORDERED,
    'set': ROW_ORDER_UNORDERED,
    'ordered': ROW_ORDER_ORDERED,
    'strict': ROW_ORDER_ORDERED,
    'sequence': ROW_ORDER_ORDERED,
}

_MISSING = ('missing',)


def _normalise_row_order(row_order, ordered=None):
    if ordered is not None:
        row_order = ordered
    try:
        return _ROW_ORDER_ALIASES[row_order]
    except (KeyError, TypeError):
        raise ValueError(
            "row_order must be 'unordered'/'ordered' (or a boolean), got %r"
            % (row_order,)
        )


def _attribute_semantics(type_):
    '''Return a hashable description of an attribute's value domain.'''
    if isinstance(type_, str):
        type_name = type_.strip().upper()
        if type_name in ('NUMERIC', 'REAL', 'INTEGER'):
            return ('numeric',)
        if type_name == 'STRING':
            return ('string',)
        return ('type', type_name)

    try:
        labels = tuple(str(label) for label in type_)
    except TypeError:
        return ('type', str(type_))
    return ('nominal', labels)


def _data_kind(semantics):
    '''The part of a declaration that affects how data values are compared.'''
    if semantics[0] == 'nominal':
        # Order belongs to metadata; the labels themselves belong to data.
        return ('nominal', tuple(sorted(set(semantics[1]))))
    if semantics[0] in ('numeric', 'string'):
        return (semantics[0],)
    return ('type', semantics[1])


def _attributes(obj):
    attributes = obj.get('attributes', []) if isinstance(obj, dict) else []
    result = []
    for attribute in attributes:
        if not isinstance(attribute, (tuple, list)) or len(attribute) != 2:
            raise ValueError('Invalid attribute declaration %r' % (attribute,))
        result.append((attribute[0], attribute[1]))
    return result


def _is_missing(value, semantics):
    if value is None:
        return True
    try:
        if bool(value != value):
            return True
    except (TypeError, ValueError):
        pass

    if isinstance(value, str):
        kind = semantics[0]
        if kind == 'nominal':
            # A declared label, including a quoted '?' or empty label, is a
            # value; an undeclared ARFF missing token is missing.
            if value in semantics[1]:
                return False
            return value == '' or value == '?'
        if kind == 'string':
            # Quoted empty strings and quoted '?' are legitimate STRING
            # values.  Unquoted forms have already been decoded to None.
            return False
        # Numerical attributes cannot hold these tokens as values.
        return value == '' or value == '?'
    return False


def _is_encoded_nominal_value(value):
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _canonical_value(value, semantics):
    if _is_missing(value, semantics):
        return _MISSING

    kind = semantics[0]
    if kind == 'numeric':
        try:
            number = float(value)
        except (TypeError, ValueError):
            return ('raw', str(value))
        if math.isnan(number):
            return _MISSING
        if number == 0:
            number = 0.0
        return ('number', number)

    if kind == 'string':
        return ('string', str(value))

    if kind == 'nominal':
        labels = semantics[1]
        if _is_encoded_nominal_value(value):
            index = int(value)
            if 0 <= index < len(labels):
                return ('nominal', labels[index])
            return ('nominal-index', index)
        return ('nominal', str(value))

    return ('raw', str(value))


def _default_canonical(semantics):
    '''The value a sparse representation implies for an omitted column.

    This mirrors the decoders: numeric slots default to 0, nominal slots to
    their first label and STRING slots to the literal ``'0'``.
    '''
    if semantics[0] == 'numeric':
        return _canonical_value(0.0, semantics)
    if semantics[0] == 'nominal' and semantics[1]:
        return _canonical_value(semantics[1][0], semantics)
    if semantics[0] == 'string':
        return _canonical_value('0', semantics)
    return _MISSING


def _looks_like_coo(data):
    # `load(return_type=COO)` returns a (data, rows, cols) tuple; dense data is
    # always a list.  scipy COO matrices expose a `format` attribute.
    if hasattr(data, 'row') and hasattr(data, 'col') and hasattr(data, 'data'):
        return True
    if not isinstance(data, tuple) or len(data) != 3:
        return False
    values, row_indices, col_indices = data
    if not all(isinstance(part, list) for part in data):
        return False
    if not (len(values) == len(row_indices) == len(col_indices)):
        return False
    return (all(isinstance(index, numbers.Integral) for index in row_indices)
            and all(isinstance(index, numbers.Integral) for index in col_indices))


def _row_from_mapping(row, semantics):
    result = []
    for index, attribute_semantics in enumerate(semantics):
        if index in row:
            result.append(_canonical_value(row[index], attribute_semantics))
        else:
            result.append(_default_canonical(attribute_semantics))

    for key in sorted(row, key=repr):
        if not isinstance(key, numbers.Integral) or key < 0 or key >= len(semantics):
            result.append(('extra-index', str(key),
                           _canonical_value(row[key], ('type', 'raw'))))
    return tuple(result)


def _row_from_dense(row, semantics):
    if isinstance(row, dict):
        return _row_from_mapping(row, semantics)

    values = list(row)
    result = [
        _canonical_value(values[index], attribute_semantics)
        if index < len(values) else _MISSING
        for index, attribute_semantics in enumerate(semantics)
    ]
    for value in values[len(semantics):]:
        result.append(('extra', _canonical_value(value, ('type', 'raw'))))
    if len(values) != len(semantics):
        result.append(('length', len(values)))
    return tuple(result)


def _rows_from_coo(data, semantics):
    if hasattr(data, 'format'):
        values = data.data
        row_indices = data.row
        col_indices = data.col
        row_count = int(data.shape[0])
    else:
        values, row_indices, col_indices = data
        row_count = max((int(index) for index in row_indices), default=-1) + 1

    rows = [dict() for _ in range(row_count)]
    for value, row_index, col_index in zip(values, row_indices, col_indices):
        rows[int(row_index)][int(col_index)] = value
    return [_row_from_mapping(row, semantics) for row in rows]


def _materialise_rows(obj, semantics):
    data = obj.get('data', []) if isinstance(obj, dict) else []
    if data is None:
        return []
    if isinstance(data, dict):
        return [_row_from_mapping(data, semantics)]
    if _looks_like_coo(data):
        return _rows_from_coo(data, semantics)
    if hasattr(data, '__next__'):
        raise ValueError(
            'Progressive data (return_type=DENSE_GEN/LOD_GEN) must be '
            'materialised before comparison; a consumed iterator cannot be '
            'compared reliably.'
        )

    iterator = iter(data)
    try:
        first = next(iterator)
    except StopIteration:
        return []

    if isinstance(first, dict):
        rows = [_row_from_mapping(first, semantics)]
        rows.extend(_row_from_mapping(row, semantics) for row in iterator)
    else:
        rows = [_row_from_dense(first, semantics)]
        rows.extend(_row_from_dense(row, semantics) for row in iterator)
    return rows


def _canonical_data(obj):
    attributes = _attributes(obj)
    semantics = [_attribute_semantics(type_) for _, type_ in attributes]
    rows = _materialise_rows(obj, semantics)
    return semantics, rows


def _canonical_dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'))


def _digest(payload):
    return hashlib.sha256(_canonical_dumps(payload).encode('utf-8')).hexdigest()


def data_equivalent(left, right, row_order=ROW_ORDER_UNORDERED, ordered=None):
    '''Return whether two import results contain equivalent data.

    Metadata such as relation names, comments and attribute names is ignored.
    Nominal label sets are part of the data (a renamed label is a data change),
    while nominal label order is metadata for symbolic values.  A nominal value
    decoded with ``encode_nominal=True`` is resolved through its declaration,
    so its label - not merely its integer position - is compared.

    ``None``, ``?``, ``''`` and NaN are all the same missing value.  Row order
    is ignored unless ``row_order='ordered'`` (or ``ordered=True``) is given.
    Duplicate rows are counted in both modes.
    '''
    mode = _normalise_row_order(row_order, ordered)

    left_semantics, left_rows = _canonical_data(left)
    right_semantics, right_rows = _canonical_data(right)

    if len(left_semantics) != len(right_semantics):
        return False
    if any(_data_kind(left_semantics[index]) !=
           _data_kind(right_semantics[index])
           for index in range(len(left_semantics))):
        return False

    if mode == ROW_ORDER_ORDERED:
        return left_rows == right_rows
    return Counter(left_rows) == Counter(right_rows)


def data_version(obj, row_order=ROW_ORDER_UNORDERED, ordered=None):
    '''Return a stable digest of the data in an import result.

    The digest follows the same comparison contract as `data_equivalent`.
    Equivalent import results have the same digest.
    '''
    mode = _normalise_row_order(row_order, ordered)
    semantics, rows = _canonical_data(obj)

    serialised_rows = [_canonical_dumps(list(row)) for row in rows]
    if mode == ROW_ORDER_UNORDERED:
        serialised_rows.sort()

    return _digest({
        'columns': [_data_kind(attribute_semantics)
                    for attribute_semantics in semantics],
        'rows': serialised_rows,
    })


def _metadata_payload(obj, include_comments=True):
    attributes = []
    for name, type_ in _attributes(obj):
        semantics = _attribute_semantics(type_)
        if semantics[0] == 'nominal':
            attributes.append({
                'name': str(name),
                'type': 'NOMINAL',
                'values': [str(label) for label in semantics[1]],
            })
        elif semantics[0] in ('numeric', 'string'):
            attributes.append({
                'name': str(name),
                'type': str(type_).upper(),
            })
        else:
            attributes.append({
                'name': str(name),
                'type': semantics[1],
            })

    payload = {
        'relation': str(obj.get('relation', '')),
        'attributes': attributes,
    }
    if include_comments:
        payload['description'] = str(obj.get('description', '') or '')
    return payload


def metadata_version(obj, include_comments=True):
    '''Return a stable digest of an import result's metadata.

    The version covers the relation name, attribute declarations and the
    description/comment block.  It includes attribute names, attribute order,
    numeric/string types and the *ordered* nominal labels.  Comments never
    affect data equivalence or `needs_reimport`; they only produce a new
    metadata version.  Pass ``include_comments=False`` when the importer treats
    comments as non-versioned annotations.
    '''
    return _digest(_metadata_payload(obj, include_comments))


def import_signature(obj, row_order=ROW_ORDER_UNORDERED, ordered=None):
    '''Return the data and metadata versions of one import result.'''
    return ImportSignature({
        'data_version': data_version(obj, row_order=row_order, ordered=ordered),
        'metadata_version': metadata_version(obj),
    })


class _AttributeDict(dict):
    '''A dict whose keys are also readable as attributes.'''

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


class ImportComparison(_AttributeDict):
    '''Result of `compare_imports`, usable as a mapping or with attributes.'''


class ImportSignature(_AttributeDict):
    '''Data and metadata versions for one import result.'''


def compare_imports(previous, current, row_order=ROW_ORDER_UNORDERED,
                    ordered=None):
    '''Compare two import results without conflating data and metadata.

    Returns a mapping with ``data_equivalent``, ``data_changed``,
    ``metadata_changed``, ``needs_reimport`` and the two metadata versions.
    ``needs_reimport`` is true only when the data changed: relation renames and
    comment changes are metadata updates, not reasons to re-import samples.
    '''
    mode = _normalise_row_order(row_order, ordered)
    same_data = data_equivalent(previous, current, row_order=mode)
    previous_metadata = _metadata_payload(previous)
    current_metadata = _metadata_payload(current)
    previous_version = metadata_version(previous)
    current_version = metadata_version(current)

    return ImportComparison({
        'data_equivalent': same_data,
        'data_changed': not same_data,
        'metadata_changed': previous_version != current_version,
        'metadata_equivalent': previous_version == current_version,
        'previous_metadata_version': previous_version,
        'metadata_version': current_version,
        'relation_changed': (
            previous_metadata['relation'] != current_metadata['relation']),
        'attributes_changed': (
            previous_metadata['attributes'] != current_metadata['attributes']),
        'description_changed': (
            str(previous.get('description', '') or '') !=
            str(current.get('description', '') or '')),
        'needs_reimport': not same_data,
    })


def needs_reimport(previous, current, row_order=ROW_ORDER_UNORDERED,
                   ordered=None):
    '''Return whether the data changed enough to require a re-import.'''
    return not data_equivalent(previous, current, row_order=row_order,
                               ordered=ordered)
# =============================================================================
