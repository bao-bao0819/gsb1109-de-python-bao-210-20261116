#!/bin/python
#
# This implements the composite value types used in settings.dat registry hive
# used to store AppContainer settings in Windows Apps aka UWP.
#
# ApplicationDataCompositeValue class is documented here:
# https://docs.microsoft.com/en-us/uwp/api/windows.storage.applicationdatacompositevalue
#
# The internals of types, values and structures had to be reverse engineered.
#
# Copyright (c) 2019 Yogesh Khatri <yogesh@swiftforensics.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from __future__ import print_function
from __future__ import unicode_literals

from datetime import datetime, timedelta
import binascii
import struct

REG_COMPOSITE_TYPE = 0x100
# When used in registry to denote value type, the REG_COMPOSITE_TYPE is or'd with one of the
# values below. Example: RegUInt8 will be (REG_COMPOSITE_TYPE | RegUint8)
# In the serialized ApplicationDataCompositeValue stream, the REG_COMPOSITE_TYPE is not present.
RegUint8 = 0x001
RegInt16 = 0x002
RegUint16 = 0x003
RegInt32 = 0x004
RegUint32 = 0x005
RegInt64 = 0x006
RegUint64 = 0x007
RegFloat = 0x008 # aka Single
RegDouble = 0x009
RegUnicodeChar = 0x00A
RegBoolean = 0x00B
RegUnicodeString = 0x00C
RegCompositeValue = 0x00D # Application Data Composite Value (Dictionary Object)
RegDateTimeOffset = 0x00E # Date as FILETIME
RegTimeSpan = 0x00F #  Span in 100ns ticks
RegGUID = 0x010
RegUnk111 = 0x011
RegUnk112 = 0x012
RegUnk113 = 0x013
RegBytesArray = 0x014
RegInt16Array = 0x015
RegUint16Array = 0x016
RegInt32Array = 0x017
RegUInt32Array = 0x018
RegInt64Array = 0x019
RegUInt64Array = 0x01A
RegFloatArray = 0x01B
RegDoubleArray = 0x01C
RegUnicodeCharArray = 0x01D
RegBooleanArray = 0x01E
RegUnicodeStringArray = 0x01F


class CompositeValueParseError(ValueError):
    """Raised when one composite item cannot be decoded.

    The stream parser catches this and keeps the item as an
    :class:`UnknownCompositeValue`, so one bad item never removes its
    siblings from the result.
    """


class CompositeValue(dict):
    """A parsed ApplicationDataCompositeValue stream.

    This is a normal ``dict`` (name -> value) with bookkeeping that lets the
    caller check the serialized layout independently of the decoded items:

    - ``offset``: byte offset of the stream in the buffer passed by its parent
    - ``size``: number of stream bytes accounted for
    - ``buffer_size``: number of bytes supplied to the parser
    - ``count``: number of items stored in this dictionary (including
      unknown/truncated placeholders)
    - ``truncated``: True when a declared item length could not be honored
      without reading past the end of the parent buffer
    - ``trailing``: an :class:`UnknownCompositeValue` for bytes that could not
      be interpreted as another item, when present
    - ``padding_size``: number of trailing zero bytes that are stream padding

    A nested composite value also carries ``item_byte_len``/``item_offset``
    from its enclosing item header, so the outer item length and the inner
    item count can be checked separately.
    """

    def __init__(self, *args, **kwargs):
        super(CompositeValue, self).__init__(*args, **kwargs)
        self.offset = 0
        self.size = 0
        self.buffer_size = 0
        self.count = 0
        self.truncated = False
        self.trailing = None
        self.padding_size = 0
        self.outer_size = None
        self.item_byte_len = None
        self.item_header_size = None
        self.item_offset = None

    @property
    def length(self):
        """Alias for :attr:`size`."""
        return self.size

    @property
    def byte_length(self):
        """Alias for :attr:`size`."""
        return self.size

    @property
    def item_count(self):
        """Alias for :attr:`count`."""
        return self.count

    @property
    def inner_count(self):
        """Number of items at this nesting level (alias for :attr:`count`)."""
        return self.count

    @property
    def outer_length(self):
        """Declared length of the enclosing item, or this stream's size."""
        if self.item_byte_len is None:
            return self.size
        return self.item_byte_len

    @property
    def verified(self):
        """True when the declared lengths agree with the parsed extents."""
        if self.truncated:
            return False
        payload_size = None
        if self.item_byte_len is not None and self.item_header_size is not None:
            payload_size = self.item_byte_len - self.item_header_size
        if self.outer_size is not None and self.outer_size != self.size:
            return False
        if payload_size is not None and payload_size != self.size:
            return False
        return True

    @property
    def items_count(self):
        """Alias for :attr:`count`."""
        return self.count

    @property
    def raw_length(self):
        """Alias for :attr:`size`."""
        return self.size

    @property
    def start(self):
        """Alias for :attr:`offset`."""
        return self.offset

    @property
    def raw_range(self):
        """``(offset, offset + size)`` for this stream."""
        return (self.offset, self.offset + self.size)

    @property
    def range(self):
        """Alias for :attr:`raw_range`."""
        return self.raw_range

    def unknown_values(self):
        """Return ``(name, UnknownCompositeValue)`` pairs at this level."""
        return [(name, value) for name, value in self.items()
                if isinstance(value, UnknownCompositeValue)]


class UnknownCompositeValue(object):
    """An item whose type is not understood, preserved byte-for-byte.

    ``data`` is exactly the item payload that lies inside the parent stream;
    it is never extended by guessing a type-specific length.  The byte range
    is available through ``offset``/``size`` and ``raw_range`` so the item can
    be reported, sliced, or re-decoded later.

    Once a parser for ``item_type`` is added to
    :data:`COMPOSITE_VALUE_PARSERS` (or supplied to :meth:`reparse`), only
    these bytes are reinterpreted; the surrounding values are left alone.
    """

    def __init__(self, item_type, name, data, offset=0, item_offset=None,
                 item_byte_len=None, data_size=None, truncated=False,
                 error=None, item_header_size=None, item_payload_size=None):
        self.item_type = item_type
        self.name = name
        self.data = data
        self.offset = offset
        self.size = len(data)
        self.item_offset = item_offset
        self.item_byte_len = item_byte_len
        self.item_header_size = item_header_size
        self.item_payload_size = item_payload_size
        self.data_size = self.size if data_size is None else data_size
        self.truncated = truncated
        self.error = error

    @property
    def type_code(self):
        """Alias for :attr:`item_type`."""
        return self.item_type

    @property
    def raw_data(self):
        """Alias for :attr:`data`."""
        return self.data

    @property
    def length(self):
        """Alias for :attr:`size`."""
        return self.size

    @property
    def raw_length(self):
        """Alias for :attr:`size`."""
        return self.size

    @property
    def declared_length(self):
        """The item length declared in the serialized header, if any."""
        return self.item_byte_len

    @property
    def end(self):
        """First byte after this item's payload."""
        return self.offset + self.size

    @property
    def raw_range(self):
        """``(offset, offset + size)`` of the preserved payload."""
        return (self.offset, self.offset + self.size)

    @property
    def range(self):
        """Alias for :attr:`raw_range`."""
        return self.raw_range

    def __bytes__(self):
        return self.data

    def __str__(self):
        type_text = "unknown" if self.item_type is None else "0x%X" % self.item_type
        name_text = "" if self.name is None else " name=%r" % (self.name,)
        return ("<UnknownCompositeValue type=%s%s range=[0x%X,0x%X) size=%d%s>" %
                (type_text, name_text, self.offset, self.end, self.size,
                 " truncated" if self.truncated else ""))

    __repr__ = __str__

    def reparse(self, parser=None):
        """Decode only this item's preserved payload.

        If ``parser`` is omitted, the parser registered for ``item_type`` in
        :data:`COMPOSITE_VALUE_PARSERS` is used.  If no parser is available,
        the original object is returned unchanged.
        """
        if parser is None:
            parser = COMPOSITE_VALUE_PARSERS.get(self.item_type)
        if parser is None:
            return self
        return parser(self.data, self.size, self.offset)


# Compatibility alias for callers that use a shorter name.
UnknownValue = UnknownCompositeValue


def _ParseFixedStruct(fmt, width):
    def parse(data, data_size, offset=0):
        if data_size < width:
            raise CompositeValueParseError(
                "item needs %d bytes, only %d available" % (width, data_size))
        return struct.unpack_from(str(fmt), data, 0)[0]
    return parse


def _ParseUnicodeChar(data, data_size, offset=0):
    if data_size < 2:
        raise CompositeValueParseError("UnicodeChar needs 2 bytes")
    return data[0:2].decode('utf-16')


def _ParseBoolean(data, data_size, offset=0):
    return data[0:1] != b'\x00'


def _ParseUnicodeString(data, data_size, offset=0):
    return data[:data_size].decode('utf-16')


def _ParseNestedComposite(data, data_size, offset=0):
    return ParseAppDataCompositeStream(data[:data_size], offset=offset,
                                       outer_size=data_size)


def _ParseDateTimeOffset(data, data_size, offset=0):
    if data_size < 8:
        raise CompositeValueParseError("DateTimeOffset needs 8 bytes")
    return parse_windows_timestamp(struct.unpack_from(str("<Q"), data, 0)[0])


def _ParseTimeSpan(data, data_size, offset=0):
    if data_size < 8:
        raise CompositeValueParseError("TimeSpan needs 8 bytes")
    return timedelta(seconds=10e-8 * struct.unpack_from(str("<Q"), data, 0)[0])


def _ParseGuid(data, data_size, offset=0):
    if data_size < 16:
        raise CompositeValueParseError("GUID needs 16 bytes")
    return ReadGuid(data)


def _ParseBytesArray(data, data_size, offset=0):
    return struct.unpack_from(str("<{}B").format(data_size), data, 0)


def _ParseInt16Array(data, data_size, offset=0):
    return struct.unpack_from(str("<{}h").format(data_size // 2), data, 0)


def _ParseUint16Array(data, data_size, offset=0):
    return struct.unpack_from(str("<{}H").format(data_size // 2), data, 0)


def _ParseInt32Array(data, data_size, offset=0):
    return struct.unpack_from(str("<{}i").format(data_size // 4), data, 0)


def _ParseUInt32Array(data, data_size, offset=0):
    return struct.unpack_from(str("<{}I").format(data_size // 4), data, 0)


def _ParseInt64Array(data, data_size, offset=0):
    return struct.unpack_from(str("<{}q").format(data_size // 8), data, 0)


def _ParseUInt64Array(data, data_size, offset=0):
    return struct.unpack_from(str("<{}Q").format(data_size // 8), data, 0)


def _ParseFloatArray(data, data_size, offset=0):
    return struct.unpack_from(str("<{}f").format(data_size // 4), data, 0)


def _ParseDoubleArray(data, data_size, offset=0):
    return struct.unpack_from(str("<{}d").format(data_size // 8), data, 0)


def _ParseUnicodeCharArray(data, data_size, offset=0):
    return data[:data_size].decode('utf-16')


def _ParseBooleanArray(data, data_size, offset=0):
    return [True if data[x:x + 1] != b'\x00' else False for x in range(data_size)]


def _ParseUnicodeStringArray(data, data_size, offset=0):
    return ReadUnicodeStringArray(data[:data_size])


def parse_windows_timestamp(qword):
    # see http://integriography.wordpress.com/2010/01/16/using-phython-to-parse-and-present-windows-64-bit-timestamps/
    return datetime.utcfromtimestamp(float(qword) * 1e-7 - 11644473600 )

def ReadUnicodeStringArray(buf):
    """Read a buffer containing an array of struct { int size; wchar string[size]; }
       Returns a list of utf8 encoded strings
    """
    strings = []
    buf_len = len(buf)
    pos = 0
    while pos < buf_len:
        if buf_len - pos < 4:
            raise CompositeValueParseError("truncated string array item length")
        item_byte_len = struct.unpack_from(str("<I"), buf, pos)[0]
        pos += 4
        if item_byte_len > buf_len - pos:
            raise CompositeValueParseError("string array item crosses its parent boundary")
        strings.append(buf[pos:pos+(item_byte_len)].decode('utf-16').rstrip('\0'))
        pos += item_byte_len
    return strings

def ReadGuid(buf):
    # mimics UUID(bytes_le=buf[0:16]) but without the security vulnerabilities of uuid (see https://bugs.python.org/issue40501 , normally fixed in python 3.9)
    b = buf[0:16]
    if isinstance(b, str):  # python 2
        return "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}".format(ord(b[3]),ord(b[2]),ord(b[1]),ord(b[0]),ord(b[5]),ord(b[4]),ord(b[7]),ord(b[6]),ord(b[8]),ord(b[9]),ord(b[10]),ord(b[11]),ord(b[12]),ord(b[13]), ord(b[14]),ord(b[15]))
    else:  # python 3
        return "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}".format(b[3],b[2],b[1],b[0],b[5],b[4],b[7],b[6],b[8],b[9],b[10],b[11],b[12],b[13],b[14],b[15])


#: Type code -> parser for one serialized item payload.
#:
#: A parser is called as ``parser(data, data_size, offset)`` and must not read
#: past ``data``.  Adding an entry here (or using
#: :func:`RegisterCompositeValueType`) is enough to teach the parser a new
#: type; already parsed streams can then reinterpret only their unknown items
#: with :func:`ReparseUnknownValues`.
COMPOSITE_VALUE_PARSERS = {
    RegUint8: _ParseFixedStruct("<B", 1),
    RegInt16: _ParseFixedStruct("<h", 2),
    RegUint16: _ParseFixedStruct("<H", 2),
    RegInt32: _ParseFixedStruct("<i", 4),
    RegUint32: _ParseFixedStruct("<I", 4),
    RegInt64: _ParseFixedStruct("<q", 8),
    RegUint64: _ParseFixedStruct("<Q", 8),
    RegFloat: _ParseFixedStruct("<f", 4),
    RegDouble: _ParseFixedStruct("<d", 8),
    RegUnicodeChar: _ParseUnicodeChar,
    RegBoolean: _ParseBoolean,
    RegUnicodeString: _ParseUnicodeString,
    RegCompositeValue: _ParseNestedComposite,
    RegDateTimeOffset: _ParseDateTimeOffset,
    RegTimeSpan: _ParseTimeSpan,
    RegGUID: _ParseGuid,
    RegBytesArray: _ParseBytesArray,
    RegInt16Array: _ParseInt16Array,
    RegUint16Array: _ParseUint16Array,
    RegInt32Array: _ParseInt32Array,
    RegUInt32Array: _ParseUInt32Array,
    RegInt64Array: _ParseInt64Array,
    RegUInt64Array: _ParseUInt64Array,
    RegFloatArray: _ParseFloatArray,
    RegDoubleArray: _ParseDoubleArray,
    RegUnicodeCharArray: _ParseUnicodeCharArray,
    RegBooleanArray: _ParseBooleanArray,
    RegUnicodeStringArray: _ParseUnicodeStringArray,
}


def RegisterCompositeValueType(type_code, parser):
    """Register a parser for a composite item type.

    ``parser(data, data_size, offset)`` receives only the item's payload and
    must return the decoded value.  Registering a type is a prerequisite for
    :func:`ReparseUnknownValues` to reinterpret previously preserved items.
    """
    if not callable(parser):
        raise TypeError("parser must be callable")
    COMPOSITE_VALUE_PARSERS[type_code] = parser
    return parser


def UnregisterCompositeValueType(type_code):
    """Remove a previously registered parser, if present."""
    return COMPOSITE_VALUE_PARSERS.pop(type_code, None)


# Snake_case aliases.
register_composite_value_type = RegisterCompositeValueType
unregister_composite_value_type = UnregisterCompositeValueType



def ParseAppDataCompositeValue(item_type, data, data_size, offset=0,
                               name=None, item_offset=None, item_byte_len=None,
                               original_type=None, outer_size=None):
    """
    Reads an individual Composite type entry from buffer
    Arguments:
        - `item_type`: composite data type (from registry value type)
        - `data`: Byte string containing a single CompositeData object
        - `data_size`: size of data
        - `offset`: byte offset of ``data`` in the enclosing stream, used for
          reporting raw ranges and for nested values
        - `name`, `item_offset`, `item_byte_len`: optional enclosing item
          metadata copied onto an unknown-value placeholder
        - `outer_size`: optional length declared by the enclosing registry
          value, recorded on a parsed composite stream so the outer length and
          the inner item count can be checked independently

    Known types are decoded normally.  An unrecognized type, or a known type
    whose payload is malformed, is returned as an
    :class:`UnknownCompositeValue` containing the exact payload bytes and its
    raw range instead of raising and discarding the enclosing object.
    """
    data = data[:data_size]
    parser = COMPOSITE_VALUE_PARSERS.get(item_type)
    if parser is not None:
        try:
            value = parser(data, len(data), offset)
        except Exception as e:
            return UnknownCompositeValue(
                original_type if original_type is not None else item_type,
                name, data, offset=offset, item_offset=item_offset,
                item_byte_len=item_byte_len, data_size=data_size,
                error=e,
                item_header_size=(item_byte_len - data_size
                                  if item_byte_len is not None else None),
                item_payload_size=data_size)
        if isinstance(value, CompositeValue) and outer_size is not None:
            value.outer_size = outer_size
        return value
    return UnknownCompositeValue(
        original_type if original_type is not None else item_type,
        name, data, offset=offset, item_offset=item_offset,
        item_byte_len=item_byte_len, data_size=data_size,
        item_header_size=(item_byte_len - data_size
                          if item_byte_len is not None else None),
        item_payload_size=data_size)


def _DecodeCompositeName(buf, start, name_length):
    """Decode a UTF-16 item name without reading past ``buf``."""
    name_bytes = buf[start:start + name_length * 2]
    try:
        return name_bytes.decode('utf-16')
    except (UnicodeDecodeError, LookupError):
        try:
            return name_bytes.decode('utf-16-le', 'replace')
        except Exception:
            return name_bytes.decode('latin-1', 'replace')


def _IsZeroPadding(buf, pos):
    """True if everything from ``pos`` through the end is zero padding."""
    return not any(buf[pos:])


def _TrailingUnknown(buf, start, buf_len, offset, declared_len,
                     item_type=None, name=None):
    """Preserve unparseable trailing bytes without crossing ``buf_len``."""
    return UnknownCompositeValue(
        item_type, name, buf[start:buf_len], offset=offset + start,
        item_offset=offset + start, item_byte_len=declared_len,
        data_size=buf_len - start, truncated=True)


def ParseAppDataCompositeStream(buf, offset=0, outer_size=None):
    """
    Read a buffer containing an ApplicationDataCompositeData binary object
    and returns a dictionary of items present.
    Arguments:
        - `buf`: Byte string containing the serialized ApplicationDataCompositeData object
        - `offset`: byte offset of ``buf`` in its parent buffer, recorded on
          each parsed value's raw range
        - `outer_size`: optional length declared by an enclosing value/item.
          When supplied it is recorded so the outer length and the inner item
          count can be checked independently.

    Returns a :class:`CompositeValue`.  It behaves like a ``dict`` and also
    reports the number of items (``count``) and the number of stream bytes
    accounted for (``size``).  Item lengths are always bounded by
    ``len(buf)``; a bad length is reported as a trailing unknown value and
    never causes a read past the parent buffer or removal of already parsed
    siblings.
    """
    composite_data = CompositeValue()
    composite_data.offset = offset
    composite_data.buffer_size = len(buf)
    composite_data.outer_size = len(buf) if outer_size is None else outer_size
    buf_len = len(buf)
    pos = 0

    while pos < buf_len:
        # Trailing zero bytes are stream alignment padding, not an item.  A
        # partial header that is not padding is preserved as trailing bytes.
        if buf_len - pos < 12 or struct.unpack_from(str("<I"), buf, pos)[0] == 0:
            if _IsZeroPadding(buf, pos):
                composite_data.padding_size = buf_len - pos
                composite_data.size = buf_len
            else:
                composite_data.trailing = _TrailingUnknown(
                    buf, pos, buf_len, offset, buf_len - pos)
                composite_data.truncated = True
                composite_data.size = pos
            break

        item_pos = pos
        item_byte_len, item_type, item_name_len = struct.unpack_from(str("<III"), buf, pos)

        minimum_len = 12 + (item_name_len + 1) * 2
        name_end = pos + 12 + item_name_len * 2
        name_offset = pos + 12
        if item_byte_len < minimum_len or name_end + 2 > buf_len:
            # The header is not usable: the length cannot cover the header and
            # name, or the name crosses the parent boundary.  There is no
            # trustworthy way to find the next item, so preserve the remainder
            # verbatim rather than guessing a length.
            partial_name = None
            if item_name_len and name_end <= buf_len:
                partial_name = _DecodeCompositeName(buf, name_offset, item_name_len)
            composite_data.trailing = _TrailingUnknown(
                buf, item_pos, buf_len, offset, item_byte_len,
                item_type=item_type, name=partial_name)
            composite_data.truncated = True
            composite_data.size = item_pos
            pos = buf_len
            break

        item_name = _DecodeCompositeName(buf, name_offset, item_name_len)

        data_start = name_end + 2
        declared_data_size = item_byte_len - minimum_len

        if data_start + declared_data_size > buf_len:
            # The declared item length crosses the end of the parent stream.
            # Keep only the bytes that actually exist and stop here; the next
            # item offset cannot be derived safely.
            available_data_size = buf_len - data_start
            payload = buf[data_start:data_start + available_data_size]
            value = UnknownCompositeValue(
                item_type, item_name, payload, offset=offset + data_start,
                item_offset=offset + item_pos, item_byte_len=item_byte_len,
                data_size=declared_data_size, truncated=True)
            composite_data[item_name] = value
            composite_data.count += 1
            composite_data.truncated = True
            composite_data.size = item_pos
            pos = buf_len
            break

        data = buf[data_start:data_start + declared_data_size]
        value = ParseAppDataCompositeValue(
            item_type, data, declared_data_size, offset=offset + data_start,
            name=item_name, item_offset=offset + item_pos,
            item_byte_len=item_byte_len)
        if isinstance(value, CompositeValue):
            value.item_byte_len = item_byte_len
            value.item_offset = offset + item_pos
            value.item_header_size = minimum_len

        composite_data[item_name] = value
        composite_data.count += 1

        item_end = item_pos + item_byte_len
        next_pos = item_end
        if next_pos % 8:  # alignment to 8 byte boundary
            next_pos += (8 - (next_pos % 8))

        if next_pos <= item_pos:
            # Defensive: a zero/negative step must not loop forever.
            composite_data.truncated = True
            composite_data.size = item_pos
            pos = buf_len
            break

        if item_end > buf_len:
            composite_data.size = buf_len
            pos = buf_len
        elif next_pos > buf_len:
            if _IsZeroPadding(buf, item_end):
                composite_data.padding_size = buf_len - item_end
                composite_data.size = buf_len
            else:
                composite_data.trailing = _TrailingUnknown(
                    buf, item_end, buf_len, offset, buf_len - item_end)
                composite_data.truncated = True
                composite_data.size = item_end
            pos = buf_len
        else:
            composite_data.size = next_pos
            pos = next_pos

    return composite_data


def ReparseUnknownValues(composite, parsers=None):
    """Reinterpret only the unknown items in an already parsed stream.

    Recognized values, including their original objects, are left untouched.
    ``parsers`` may be a ``{type_code: parser}`` mapping to use instead of
    :data:`COMPOSITE_VALUE_PARSERS`.  Nested composite values are visited
    recursively.  Returns the number of unknown items that were replaced.
    """
    if parsers is None:
        parsers = COMPOSITE_VALUE_PARSERS

    replaced = 0
    for name, value in list(composite.items()):
        if isinstance(value, UnknownCompositeValue):
            parser = parsers.get(value.item_type)
            if parser is not None:
                try:
                    new_value = parser(value.data, value.size, value.offset)
                except Exception as e:
                    value.error = e
                    continue
                if isinstance(new_value, CompositeValue):
                    new_value.item_byte_len = value.item_byte_len
                    new_value.item_offset = value.item_offset
                    new_value.outer_size = value.item_payload_size
                    if value.item_header_size is not None:
                        new_value.item_header_size = value.item_header_size
                composite[name] = new_value
                replaced += 1
        elif isinstance(value, CompositeValue):
            replaced += ReparseUnknownValues(value, parsers)

    return replaced


# Snake_case alias.
reparse_unknown_values = ReparseUnknownValues


def FormatCompositeValue(value, indent=0):
    """Return a readable rendering of a composite value.

    Unknown values are rendered with their type code and exact raw byte range.
    """
    pad = " " * indent
    if isinstance(value, CompositeValue):
        lines = ["{"]
        for name, item in value.items():
            lines.append("%s  %r: %s" % (pad, name,
                                         FormatCompositeValue(item, indent + 2)))
        if value.trailing is not None:
            lines.append("%s  <trailing>: %s" % (pad, value.trailing))
        lines.append("%s}" % pad)
        return "\n".join(lines)
    return str(value)


# Snake_case alias.
format_composite_value = FormatCompositeValue
