# -*- coding: utf-8 -*-
#
#   pyhwp : hwp file format parser in python
#   Copyright (C) 2010-2023 mete0r <https://github.com/mete0r>
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Affero General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Affero General Public License for more details.
#
#   You should have received a copy of the GNU Affero General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

from collections import namedtuple
from datetime import datetime
from datetime import timedelta
from uuid import UUID
import binascii
import codecs
import logging
import struct

from hwp5.dataio import Struct
from hwp5.dataio import Flags
from hwp5.dataio import N_ARRAY
from hwp5.dataio import ARRAY
from hwp5.dataio import BYTE
from hwp5.dataio import UINT16
from hwp5.dataio import UINT32
from hwp5.dataio import INT32
from hwp5.bintype import read_type


logger = logging.getLogger(__name__)


# Windows code page identifiers that do not map to a ``cp<NNN>`` codec.
CODEPAGE_ENCODINGS = {
    1200: 'utf-16-le',
    1201: 'utf-16-be',
    65000: 'utf-7',
    65001: 'utf-8',
}


def codepage_encoding(codepage):
    ''' Return a Python codec name for an OLE property-set code page.

        Returns ``None`` when the code page is not recognized, so callers can
        keep the original bytes instead of guessing an encoding.
    '''
    if codepage is None:
        return None
    encoding = CODEPAGE_ENCODINGS.get(codepage)
    if encoding is None:
        candidate = 'cp{}'.format(codepage)
        try:
            codecs.lookup(candidate)
        except LookupError:
            return None
        encoding = candidate
    return encoding


def _hex(data):
    return binascii.hexlify(data).decode('ascii')


class RawTextValue(object):
    ''' A text value whose raw bytes could not be decoded.

        The original bytes are retained together with the declared code page,
        so unknown code pages or invalid byte sequences never lead to data
        loss. Readable properties of other types remain available alongside
        these values.
    '''

    __slots__ = ('raw', 'codepage', 'encoding', 'reason')

    def __init__(self, raw, codepage=None, encoding=None,
                 reason='undecodable'):
        self.raw = raw
        self.codepage = codepage
        self.encoding = encoding
        self.reason = reason

    @property
    def raw_hex(self):
        return _hex(self.raw)

    def __str__(self):
        return self.raw_hex

    def __repr__(self):
        return (
            'RawTextValue(0x{}, codepage={}, encoding={}, reason={})'.format(
                self.raw_hex, self.codepage, self.encoding, self.reason,
            )
        )


class UnreadableValue(object):
    ''' A property whose value could not be parsed at all.

        The failure is recorded instead of being raised, so the remaining
        readable properties of the set are still exported.
    '''

    __slots__ = ('error', 'raw')

    def __init__(self, error, raw=b''):
        self.error = error
        self.raw = raw

    @property
    def raw_hex(self):
        return _hex(self.raw)

    def __str__(self):
        return 'unreadable({})=0x{}'.format(self.error, self.raw_hex)


def decode_codepage_text(raw, codepage):
    ''' Decode code-page-backed text, preserving raw bytes on failure. '''
    encoding = codepage_encoding(codepage)
    if encoding is None:
        return RawTextValue(raw, codepage=codepage, encoding=None,
                            reason='unknown-codepage')
    try:
        return raw.decode(encoding)
    except UnicodeDecodeError:
        return RawTextValue(raw, codepage=codepage, encoding=encoding,
                            reason='undecodable')


vt_types = dict()


def PropertyType(code):

    def decorator(cls):
        cls.code = code
        vt_types[code] = cls
        return cls

    return decorator


@PropertyType(code=0x0003)
class VT_I4(object):

    @classmethod
    def read_value(cls, context, f):
        return read_type(INT32, context, f)


@PropertyType(code=0x001E)
class VT_LPSTR(object):
    ''' A code-page-encoded, null-terminated single-byte string.

        The applicable code page is the PID_CODEPAGE of the enclosing
        property set, which is passed through the read context. When the code
        page is unknown or the bytes are not decodable, a
        :class:`RawTextValue` holding the original bytes is returned.
    '''

    @classmethod
    def read_value(cls, context, f):
        bytecount = read_type(UINT32, context, f)
        raw = f.read(bytecount)
        if raw.endswith(b'\x00'):
            raw = raw[:-1]
        return decode_codepage_text(raw, context.get('codepage'))


@PropertyType(code=0x001F)
class VT_LPWSTR(object):

    @classmethod
    def read_value(cls, context, f):
        length = read_type(UINT32, context, f)
        data = f.read(length * 2)
        text = data.decode('utf-16le')
        if text.endswith('\x00'):  # remove null character
            text = text[:-1]
        return text


@PropertyType(code=0x0040)
class VT_FILETIME(object):

    @classmethod
    def read_value(cls, context, f):
        lword = read_type(UINT32, context, f)
        hword = read_type(UINT32, context, f)
        value = hword << 32 | lword
        value = FILETIME(value)
        return value


# Human-readable names of the [MS-OLEPS] variant type tags we understand.
VT_TYPE_LABELS = {
    0x0000: 'VT_EMPTY',
    0x0001: 'VT_NULL',
    0x0002: 'VT_I2',
    0x0003: 'VT_I4',
    0x0006: 'VT_CY',
    0x0007: 'VT_DATE',
    0x000B: 'VT_BOOL',
    0x0010: 'VT_I1',
    0x0011: 'VT_UI1',
    0x0012: 'VT_UI2',
    0x0013: 'VT_UI4',
    0x0014: 'VT_I8',
    0x0015: 'VT_UI8',
    0x001E: 'VT_LPSTR',
    0x001F: 'VT_LPWSTR',
    0x0040: 'VT_FILETIME',
    0x0047: 'VT_CLSID',
}


def vt_type_label(code):
    return VT_TYPE_LABELS.get(code, 'VT_UNKNOWN_0x{:04x}'.format(code))


class FILETIME(object):
    __slots__ = ('value', )

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return str(self.datetime)

    @property
    def datetime(self):
        return (
            datetime(1601, 1, 1, 0, 0, 0) +
            timedelta(microseconds=self.value / 10)
        )


PropertyIdentifier = namedtuple('PropertyIdentifier', [
    'id',
    'label',
])


PID_DICTIONARY = PropertyIdentifier(
    id=0x00000000,
    label='PID_DICTIONARY',
)
PID_CODEPAGE = PropertyIdentifier(
    id=0x00000001,
    label='PID_CODEPAGE',
)
PID_LOCALE = PropertyIdentifier(
    id=0x80000000,
    label='PID_LOCALE',
)
PID_BEHAVIOR = PropertyIdentifier(
    id=0x80000003,
    label='PID_BEHAVIOR',
)
PIDSI_TITLE = PropertyIdentifier(
    id=0x02,
    label='PIDSI_TITLE'
)
PIDSI_SUBJECT = PropertyIdentifier(
    id=0x03,
    label='PIDSI_SUBJECT'
)
PIDSI_AUTHOR = PropertyIdentifier(
    id=0x04,
    label='PIDSI_AUTHOR'
)
PIDSI_KEYWORDS = PropertyIdentifier(
    id=0x05,
    label='PIDSI_KEYWORDS'
)
PIDSI_COMMENTS = PropertyIdentifier(
    id=0x06,
    label='PIDSI_COMMENTS'
)
PIDSI_TEMPLATE = PropertyIdentifier(
    id=0x07,
    label='PIDSI_TEMPLATE'
)
PIDSI_LASTAUTHOR = PropertyIdentifier(
    id=0x08,
    label='PIDSI_LASTAUTHOR'
)
PIDSI_REVNUMBER = PropertyIdentifier(
    id=0x09,
    label='PIDSI_REVNUMBER'
)
PIDSI_EDITTIME = PropertyIdentifier(
    id=0x0a,
    label='PIDSI_EDITTIME'
)
PIDSI_LASTPRINTED = PropertyIdentifier(
    id=0x0b,
    label='PIDSI_LASTPRINTED'
)
PIDSI_CREATE_DTM = PropertyIdentifier(
    id=0x0c,
    label='PIDSI_CREATE_DTM'
)
PIDSI_LASTSAVE_DTM = PropertyIdentifier(
    id=0x0d,
    label='PIDSI_LASTSAVE_DTM'
)
PIDSI_PAGECOUNT = PropertyIdentifier(
    id=0x0e,
    label='PIDSI_PAGECOUNT'
)
PIDSI_WORDCOUNT = PropertyIdentifier(
    id=0x0f,
    label='PIDSI_WORDCOUNT'
)
PIDSI_CHARCOUNT = PropertyIdentifier(
    id=0x10,
    label='PIDSI_CHARCOUNT'
)
PIDSI_THUMBNAIL = PropertyIdentifier(
    id=0x11,
    label='PIDSI_THUMBNAIL'
)
PIDSI_APPNAME = PropertyIdentifier(
    id=0x12,
    label='PIDSI_APPNAME'
)
PIDSI_SECURITY = PropertyIdentifier(
    id=0x13,
    label='PIDSI_SECURITY'
)


RESERVED_PROPERTIES = (
    PID_DICTIONARY,
    PID_CODEPAGE,
    PID_LOCALE,
    PID_BEHAVIOR,
)


SUMMARY_INFORMATION_PROPERTIES = (
    PIDSI_TITLE,
    PIDSI_SUBJECT,
    PIDSI_AUTHOR,
    PIDSI_KEYWORDS,
    PIDSI_COMMENTS,
    PIDSI_TEMPLATE,
    PIDSI_LASTAUTHOR,
    PIDSI_REVNUMBER,
    PIDSI_EDITTIME,
    PIDSI_LASTPRINTED,
    PIDSI_CREATE_DTM,
    PIDSI_LASTSAVE_DTM,
    PIDSI_PAGECOUNT,
    PIDSI_WORDCOUNT,
    PIDSI_CHARCOUNT,
    PIDSI_THUMBNAIL,
    PIDSI_APPNAME,
    PIDSI_SECURITY,
)


class Property(object):

    def __init__(self, desc, idLabel, type, value, propsetDesc=None,
                 codepage=None, byteRange=None, displayName=None):
        self.desc = desc
        self.idLabel = idLabel
        self.type = type
        self.value = value
        self.propsetDesc = propsetDesc
        self.codepage = codepage
        self.byteRange = byteRange
        self.displayName = displayName

    @property
    def id(self):
        return self.desc.id

    @property
    def typeLabel(self):
        if self.type is None:
            return None
        return vt_type_label(self.type.code)

    @property
    def typeCode(self):
        if self.type is None:
            return None
        return '0x{:04x}'.format(self.type.code)

    @property
    def streamOffset(self):
        ''' Absolute byte offset of the value inside the property stream. '''
        if self.propsetDesc is None or self.desc is None:
            return None
        return self.propsetDesc.offset + self.desc.offset


class PropertyDesc(Struct):

    def __init__(self, id, offset):
        self.id = id
        self.offset = offset

    @classmethod
    def fromDict(cls, d):
        return cls(id=d['id'], offset=d['offset'])

    def attributes():
        yield UINT32, 'id'
        yield UINT32, 'offset'  # offset from section start
    attributes = staticmethod(attributes)


class PropertyReader(object):

    def __init__(self, propsetDesc, propDesc, idLabel, codepage,
                 displayName=None):
        self.propsetDesc = propsetDesc
        self.propDesc = propDesc
        self.idLabel = idLabel
        self.codepage = codepage
        self.displayName = displayName

    def read(self, f):
        start = self.propsetDesc.offset + self.propDesc.offset
        f.seek(start)

        context = {'codepage': self.codepage}
        propType = read_type(TypedPropertyValue, context, f)
        propType = TypedPropertyValue.fromDict(propType)
        try:
            vt_type = vt_types[propType.code]
        except KeyError:
            logger.warning(
                'unknown OLE property type 0x%04x at offset %d; '
                'keeping the property unreadable instead of failing',
                propType.code, start,
            )
            propValue = UnreadableValue(
                'unknown type 0x{:04x}'.format(propType.code),
            )
        else:
            try:
                propValue = vt_type.read_value(context, f)
            except Exception as e:
                logger.warning(
                    'failed to read OLE property 0x%08x at offset %d: %s',
                    self.propDesc.id, start, e,
                )
                propValue = UnreadableValue(
                    '{}: {}'.format(type(e).__name__, e),
                )
        end = f.tell()

        return Property(
            desc=self.propDesc,
            idLabel=self.idLabel,
            type=propType,
            value=propValue,
            propsetDesc=self.propsetDesc,
            codepage=self.codepage,
            byteRange=(start, end),
            displayName=self.displayName,
        )


class TypedPropertyValue(Struct):
    '''
    [MS-OLEPS] 2.15 TypedPropertyValue
    '''

    def __init__(self, code):
        self.code = code

    @classmethod
    def fromDict(cls, d):
        return cls(code=d['type'].code)

    TypeFlags = Flags(UINT32,
                      0, 16, 'code')

    def attributes(cls):
        yield cls.TypeFlags, 'type'
    attributes = classmethod(attributes)

    @property
    def vt_type(self):
        try:
            return vt_types[self.code]
        except KeyError:
            return None


class DictionaryEntry(Struct):
    '''
    [MS-OLEPS] 2.16 DictionaryEntry
    '''

    def __init__(self, id, name):
        self.id = id
        self.name = name

    @classmethod
    def fromDict(cls, d):
        return cls(
            id=d['id'],
            name=nullterminated_string(d['name']),
        )

    def attributes():
        from hwp5.dataio import N_ARRAY
        from hwp5.dataio import BYTE
        yield UINT32, 'id'
        yield N_ARRAY(UINT32, BYTE), 'name'
    attributes = staticmethod(attributes)


class Dictionary(Struct):
    '''
    [MS-OLEPS] 2.17 Dictionary
    '''

    def __init__(self, entries):
        self.entries = entries

    @classmethod
    def fromDict(cls, d):
        entries = tuple(
            DictionaryEntry.fromDict(entry)
            for entry in d['entries']
        )
        return cls(entries=entries)

    def attributes():
        from hwp5.dataio import N_ARRAY
        yield N_ARRAY(UINT32, DictionaryEntry), 'entries'
    attributes = staticmethod(attributes)

    def get(self, id, defvalue=None):
        for entry in self.entries:
            if id == entry.id:
                return entry.name
        return defvalue


class DictionaryReader(object):

    def __init__(self, propsetDesc, propDesc, idLabel, codepage):
        self.propsetDesc = propsetDesc
        self.propDesc = propDesc
        self.idLabel = idLabel
        self.codepage = codepage

    def read(self, f):
        propsetDesc = self.propsetDesc
        propDesc = self.propDesc
        idLabel = self.idLabel

        start = propsetDesc.offset + propDesc.offset
        f.seek(start)
        context = {}
        propType = None
        propValue = read_type(Dictionary, context, f)
        propValue = Dictionary.fromDict(propValue)
        end = f.tell()
        return Property(
            desc=propDesc,
            idLabel=idLabel,
            type=propType,
            value=propValue,
            propsetDesc=propsetDesc,
            codepage=self.codepage,
            byteRange=(start, end),
        )


class PropertySet(object):
    '''
    [MS-OLEPS] 2.20 PropertySet
    '''

    def __init__(self, desc, header, properties):
        self.desc = desc
        self.header = header
        # A plain list on purpose: property identifiers or display names may
        # legitimately repeat, and every occurrence has to be retained.
        self.properties = properties

    @property
    def fmtid(self):
        return self.desc.fmtid

    def findall(self, propertyIdentifier):
        ''' Return every property value matching the identifier.

            Unlike :meth:`__getitem__` this does not collapse repeated
            identifiers, so conflicting duplicate fields are all exposed.
        '''
        return [
            property.value
            for property in self.properties
            if property.id == propertyIdentifier.id
        ]

    def __getitem__(self, propertyIdentifier):
        for property in self.properties:
            if property.id == propertyIdentifier.id:
                return property.value
        raise KeyError(propertyIdentifier)


class PropertySetHeader(Struct):

    def __init__(self, bytesize, propDescList):
        self.bytesize = bytesize,
        self.propDescList = propDescList

    @classmethod
    def fromDict(cls, d):
        return cls(
            bytesize=d['bytesize'],
            propDescList=tuple(
                PropertyDesc.fromDict(
                    propDesc
                )
                for propDesc in d['propDescList']
            ),
        )

    def attributes():
        from hwp5.dataio import N_ARRAY
        yield UINT32, 'bytesize'
        yield N_ARRAY(UINT32, PropertyDesc), 'propDescList'
    attributes = staticmethod(attributes)


class PropertySetDesc(Struct):

    def __init__(self, fmtid, offset):
        self.fmtid = fmtid
        self.offset = offset

    def attributes():
        yield ARRAY(BYTE, 16), 'fmtid'
        yield UINT32, 'offset'
    attributes = staticmethod(attributes)

    @classmethod
    def fromDict(cls, d):
        return cls(
            fmtid=uuid_from_bytes_tuple(d['fmtid']),
            offset=d['offset'],
        )


class PropertySetStreamHeader(Struct):

    def __init__(self, byteOrder, version, systemIdentifier, clsid,
                 propsetDescList):
        self.byteOrder = byteOrder
        self.version = version
        self.systemIdentifier = systemIdentifier
        self.clsid = clsid
        self.propsetDescList = propsetDescList

    @classmethod
    def fromDict(cls, d):
        return cls(
            byteOrder=d['byteOrder'],
            version=d['version'],
            systemIdentifier=d['systemIdentifier'],
            clsid=uuid_from_bytes_tuple(d['clsid']),
            propsetDescList=tuple(
                PropertySetDesc.fromDict(
                    propsetDesc
                )
                for propsetDesc in d['propsetDescList']
            )
        )

    def attributes():
        yield UINT16, 'byteOrder'
        yield UINT16, 'version'
        yield UINT32, 'systemIdentifier'
        yield ARRAY(BYTE, 16), 'clsid'
        yield N_ARRAY(UINT32, PropertySetDesc), 'propsetDescList'
    attributes = staticmethod(attributes)


class PropertySetStream(object):
    '''
    [MS-OLEPS] 2.21 PropertySetStream
    '''

    def __init__(self, header, propertysets):
        self.header = header
        self.propertysets = propertysets

    @property
    def byteOrder(self):
        return self.header.byteOrder

    @property
    def version(self):
        return self.header.version

    @property
    def systemIdentifier(self):
        return self.header.systemIdentifier

    @property
    def clsid(self):
        return self.header.clsid


class PropertySetFormat(object):

    def __init__(self, fmtid, propertyIdentifiers):
        self.fmtid = fmtid
        self.propertyIdentifiers = propertyIdentifiers

    @property
    def idLabels(self):
        return {
            p.id: p.label
            for p in self.propertyIdentifiers
        }


class PropertySetStreamReader(object):

    def __init__(self, propertySetFormats):
        self.propertySetFormats = {
            propsetFormat.fmtid: propsetFormat
            for propsetFormat in propertySetFormats
        }

    def read(self, f):
        context = {}
        streamHeader = read_type(PropertySetStreamHeader, context, f)
        streamHeader = PropertySetStreamHeader.fromDict(streamHeader)
        propertysetList = list()
        for propsetDesc in streamHeader.propsetDescList:
            f.seek(propsetDesc.offset)
            propsetHeader = read_type(PropertySetHeader, context, f)
            propsetHeader = PropertySetHeader.fromDict(
                propsetHeader,
            )
            try:
                propsetFormat = self.propertySetFormats[propsetDesc.fmtid]
            except KeyError:
                idLabels = {}
            else:
                idLabels = propsetFormat.idLabels

            propDescs = list(propsetHeader.propDescList)
            consumed = set()
            properties = []

            # PID_CODEPAGE has to be known before any code-page text is read.
            codepage = None
            codepageDesc = self._find_first_desc(propDescs, PID_CODEPAGE.id)
            if codepageDesc is not None:
                prop = self._read_property(
                    f, propsetDesc, codepageDesc, idLabels,
                    codepage=None,
                )
                properties.append(prop)
                consumed.add(codepageDesc)
                codepage = prop.value

            # PID_DICTIONARY maps property identifiers to display names.
            dictionary = None
            dictDesc = self._find_first_desc(propDescs, PID_DICTIONARY.id)
            if dictDesc is not None:
                prop = self._read_dictionary(
                    f, propsetDesc, dictDesc, idLabels, codepage,
                )
                properties.append(prop)
                consumed.add(dictDesc)
                dictionary = prop.value

            # Every remaining descriptor is read independently; repeated
            # identifiers are kept as separate properties instead of being
            # collapsed into one.
            for propDesc in propDescs:
                if propDesc in consumed:
                    continue
                displayName = None
                if dictionary is not None:
                    displayName = dictionary.get(propDesc.id, None)
                prop = self._read_property(
                    f, propsetDesc, propDesc, idLabels,
                    codepage=codepage,
                    displayName=displayName,
                )
                properties.append(prop)

            properties = sorted(
                properties,
                key=lambda property: property.desc.offset,
            )

            propertyset = PropertySet(
                desc=propsetDesc,
                header=propsetHeader,
                properties=properties,
            )
            propertysetList.append(propertyset)

        return PropertySetStream(
            header=streamHeader,
            propertysets=propertysetList,
        )

    @staticmethod
    def _find_first_desc(propDescs, identifier):
        for propDesc in propDescs:
            if propDesc.id == identifier:
                return propDesc
        return None

    def _read_property(self, f, propsetDesc, propDesc, idLabels,
                       codepage, displayName=None):
        idLabel = idLabels.get(propDesc.id)
        propReader = PropertyReader(
            propsetDesc=propsetDesc,
            propDesc=propDesc,
            idLabel=idLabel,
            codepage=codepage,
            displayName=displayName,
        )
        return propReader.read(f)

    def _read_dictionary(self, f, propsetDesc, propDesc, idLabels, codepage):
        idLabel = idLabels.get(propDesc.id)
        propReader = DictionaryReader(
            propsetDesc,
            propDesc,
            idLabel,
            codepage,
        )
        return propReader.read(f)


def render_property_value(value):
    ''' Render a property value as text without raising on raw bytes. '''
    if isinstance(value, RawTextValue):
        return 'raw-bytes(codepage={}, {})=0x{}'.format(
            value.codepage, value.reason, value.raw_hex,
        )
    if isinstance(value, UnreadableValue):
        return str(value)
    if isinstance(value, FILETIME):
        return str(value.datetime)
    if isinstance(value, Dictionary):
        entries = ', '.join(
            '0x{:08x}: {!r}'.format(entry.id, entry.name)
            for entry in value.entries
        )
        return '{' + entries + '}'
    return str(value)


class PropertySetStreamTextFormatter(object):

    def formatTextLines(self, stream):
        yield '- ByteOrder: 0x%x' % stream.byteOrder
        yield '- Version: %d' % stream.version
        yield '- SystemIdentifier: 0x%08x' % stream.systemIdentifier
        yield '- CLSID: %s' % stream.clsid
        yield ''

        for propertyset in stream.propertysets:
            title = 'Property Set {}'.format(
                propertyset.fmtid,
            )
            yield '- {:08x}: {}'.format(
                propertyset.desc.offset,
                title,
            )
            yield '            {}'.format(
                '-' * len(title)
            )

            properties = sorted(
                propertyset.properties,
                key=lambda property: property.desc.offset,
            )
            for property in properties:
                start, end = property.byteRange or (None, None)
                label = property.idLabel if property.idLabel is not None \
                    else ''
                type_label = property.typeLabel or ''
                type_segment = type_label + ' ' if type_label else ''
                codepage = property.codepage
                codepage_text = str(codepage) if codepage is not None else '-'
                if property.id == PID_DICTIONARY.id:
                    yield (
                        '- {:08x}-{:08x}: {}(=0x{:08x}) {}{}:'
                    ).format(
                        start, end, label, property.id,
                        type_segment,
                        'codepage={}'.format(codepage_text),
                    )
                    for entry in property.value.entries:
                        yield '            - {}: {}'.format(
                            entry.id,
                            entry.name,
                        )
                else:
                    yield (
                        '- {:08x}-{:08x}: {}(=0x{:08x}) {}{}: {}'
                    ).format(
                        start, end, label, property.id,
                        type_segment,
                        'codepage={}'.format(codepage_text),
                        render_property_value(property.value),
                    )


def uuid_from_bytes_tuple(t):
    fmt = 'B' * len(t)
    fmt = '<' + fmt
    bytes_le = struct.pack(fmt, *t)
    return UUID(bytes_le=bytes_le)


def nullterminated_string(bs):
    return ''.join(chr(x) for x in bs)[:-1]
