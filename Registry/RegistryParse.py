#!/bin/python

#    This file is part of python-registry.
#
#   Copyright 2011 Will Ballenthin <william.ballenthin@mandiant.com>
#                    while at Mandiant <http://www.mandiant.com>
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

# Added for python2-3 compatibility
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import struct
import datetime
import decimal
import binascii
from ctypes import c_uint32
from enum import Enum
from collections import namedtuple
from Registry import SettingsParse

# Constants
RegSZ = 0x0001
RegExpandSZ = 0x0002
RegBin = 0x0003
RegDWord = 0x0004
RegMultiSZ = 0x0007
RegQWord = 0x000B
RegNone = 0x0000
RegBigEndian = 0x0005
RegLink = 0x0006
RegResourceList = 0x0008
RegFullResourceDescriptor = 0x0009
RegResourceRequirementsList = 0x000A
RegFileTime = 0x0010
# Following are new types from settings.dat
RegUint8 = 0x101
RegInt16 = 0x102
RegUint16 = 0x103
RegInt32 = 0x104
RegUint32 = 0x105
RegInt64 = 0x106
RegUint64 = 0x107
RegFloat = 0x108
RegDouble = 0x109
RegUnicodeChar = 0x10A
RegBoolean = 0x10B
RegUnicodeString = 0x10C
RegCompositeValue = 0x10D
RegDateTimeOffset = 0x10E
RegTimeSpan = 0x10F
RegGUID = 0x110
RegUnk111 = 0x111
RegUnk112 = 0x112
RegUnk113 = 0x113
RegBytesArray = 0x114
RegInt16Array = 0x115
RegUint16Array = 0x116
RegInt32Array = 0x117
RegUInt32Array = 0x118
RegInt64Array = 0x119
RegUInt64Array = 0x11A
RegFloatArray = 0x11B
RegDoubleArray = 0x11C
RegUnicodeCharArray = 0x11D
RegBooleanArray = 0x11E
RegUnicodeStringArray = 0x11F

# Constants to support the transaction log files (new format)
LOG_ENTRY_SIZE_HEADER = 40
LOG_ENTRY_SIZE_ALIGNMENT = 0x200

class FileType(Enum):
    FILE_TYPE_PRIMARY = 0
    FILE_TYPE_LOG_OLD_1 = 1 # Starting from Windows XP
    FILE_TYPE_LOG_OLD_2 = 2 # Before Windows XP
    FILE_TYPE_LOG_NEW = 6 # Starting from Windows 8.1

# Added in Windows Vista. Must be applied to Registry type.
# see: http://msdn.microsoft.com/en-us/library/windows/hardware/ff543550%28v=vs.85%29.aspx
DEVPROP_MASK_TYPE = 0x00000FFF

# This named tuple describes the recovery operations to be performed on a hive.
RecoveryStatus = namedtuple('RecoveryStatus', ['recover_header', 'recover_data'])


def parse_timestamp(ticks, resolution, epoch, mode=decimal.ROUND_HALF_EVEN):
    """
    Generalized function for parsing timestamps

    :param ticks: number of time units since the epoch
    :param resolution: number of time units per second
    :param epoch: the datetime of this timestamp's epoch
    :param mode: decimal rounding mode
    :return: datetime.datetime
    """
    # python's datetime.datetime supports microsecond precision
    datetime_resolution = int(1e6)

    # convert ticks since epoch to microseconds since epoch
    us = int((decimal.Decimal(ticks * datetime_resolution) / decimal.Decimal(resolution)).quantize(1, mode))

    # convert to datetime
    return epoch + datetime.timedelta(microseconds=us)


def parse_windows_timestamp(qword):
    """
    :param qword: number of 100-nanoseconds since 1601-01-01
    :return: datetime.datetime
    """
    # see https://msdn.microsoft.com/en-us/library/windows/desktop/ms724290(v=vs.85).aspx
    return parse_timestamp(qword, int(1e7), datetime.datetime(1601, 1, 1))


class RegistryException(Exception):
    """
    Base Exception class for Windows Registry access.
    """

    def __init__(self, value):
        """
        Constructor.
        Arguments:
        - `value`: A string description.
        """
        super(RegistryException, self).__init__()
        self._value = value

    def __str__(self):
        return "Registry Exception: %s" % (self._value)


class RegistryStructureDoesNotExist(RegistryException):
    """
    Exception to be raised when a structure or block is requested which does not exist.
    For example, asking for the ValuesList structure of an NKRecord that has no values
    (and therefore no ValuesList) should result in this exception.
    """
    def __init__(self, value):
        """
        Constructor.
        Arguments:
        - `value`: A string description.
        """
        super(RegistryStructureDoesNotExist, self).__init__(value)

    def __str__(self):
        return "Registry Structure Does Not Exist Exception: %s" % (self._value)


class ParseException(RegistryException):
    """
    An exception to be thrown during Windows Registry parsing, such as
    when an invalid header is encountered.
    """
    def __init__(self, value):
        """
        Constructor.
        Arguments:
        - `value`: A string description.
        """
        super(ParseException, self).__init__(value)

    def __str__(self):
        return "Registry Parse Exception (%s)" % (self._value)


class UnknownTypeException(RegistryException):
    """
    An exception to be raised when an unknown data type is encountered.
    Supported data types current consist of
     - RegSZ
     - RegExpandSZ
     - RegBin
     - RegDWord
     - RegMultiSZ
     - RegQWord
     - RegNone
     - RegBigEndian
     - RegLink
     - RegResourceList
     - RegFullResourceDescriptor
     - RegResourceRequirementsList
     - RegFileTime
    """
    def __init__(self, value):
        """
        Constructor.
        Arguments:
        - `value`: A string description.
        """
        super(UnknownTypeException, self).__init__(value)

    def __str__(self):
        return "Unknown Type Exception (%s)" % (self._value)

class NotSupportedException(RegistryException):
    """
    An exception to be thrown during Windows Registry parsing, when something is not supported yet.
    """
    def __init__(self, value):
        """
        Constructor.
        Arguments:
        - `value`: A string description.
        """
        super(NotSupportedException, self).__init__(value)

    def __str__(self):
        return "Not Supported Exception (%s)" % (self._value)

class RegistryBlock(object):
    """
    Base class for structure blocks in the Windows Registry.
    A block is associated with a offset into a byte-string.

    All blocks (besides the root) also have a parent member, which refers to
    a RegistryBlock that contains a reference to this block, an is found at a
    hierarchically superior rank. Note, by following the parent links upwards,
    the root block should be accessible (aka. there should not be any loops)
    """
    def __init__(self, buf, offset, parent):
        """
        Constructor.
        Arguments:
        - `buf`: Byte string containing Windows Registry file.
        - `offset`: The offset into the buffer at which the block starts.
        - `parent`: The parent block, which links to this block.
        """
        self._buf = buf
        self._offset = offset
        self._parent = parent

    def unpack_binary(self, offset, length):
        return self._buf[self._offset + offset:self._offset + offset + length]

    def unpack_word(self, offset):
        """
        Returns a little-endian WORD (2 bytes) from the relative offset.
        Arguments:
        - `offset`: The relative offset from the start of the block.
        """
        return struct.unpack_from(str("<H"), self._buf, self._offset + offset)[0]

    def unpack_dword(self, offset):
        """
        Returns a little-endian DWORD (4 bytes) from the relative offset.
        Arguments:
        - `offset`: The relative offset from the start of the block.
        """
        return struct.unpack_from(str("<I"), self._buf, self._offset + offset)[0]

    def unpack_int(self, offset):
        """
        Returns a little-endian signed integer (4 bytes) from the relative offset.
        Arguments:
        - `offset`: The relative offset from the start of the block.
        """
        return struct.unpack_from(str("<i"), self._buf, self._offset + offset)[0]

    def unpack_qword(self, offset):
        """
        Returns a little-endian QWORD (8 bytes) from the relative offset.
        Arguments:
        - `offset`: The relative offset from the start of the block.
        """
        return struct.unpack_from(str("<Q"), self._buf, self._offset + offset)[0]

    def unpack_string(self, offset, length):
        """
        Returns a byte string from the relative offset with the given length.
        Arguments:
        - `offset`: The relative offset from the start of the block.
        - `length`: The length of the string.
        """
        return struct.unpack_from(str("<%ds") % (length), self._buf, self._offset + offset)[0]

    def absolute_offset(self, offset):
        """
        Get the absolute offset from an offset relative to this block
        Arguments:
        - `offset`: The relative offset into this block.
        """
        return self._offset + offset

    def parent(self):
        """
        Get the parent block. See the class documentation for what the parent link is.
        """
        return self._parent

    def offset(self):
        """
        Equivalent to self.absolute_offset(0x0), which is the starting offset of this block.
        """
        return self._offset


class REGFBlock(RegistryBlock):
    """
    The Windows Registry file header. This block has a length of 4k, although
    only the first 0x200 bytes are generally used.
    """
    def __init__(self, buf, offset, parent):
        """
        Constructor.
        Arguments:
        - `buf`: Byte string containing Windows Registry file.
        - `offset`: The offset into the buffer at which the block starts.
        - `parent`: The parent block, which links to this block.
        """
        super(REGFBlock, self).__init__(buf, offset, parent)

        _id = self.unpack_dword(0)
        if _id != 0x66676572:
            raise ParseException("Invalid REGF ID")

    def hive_sequence1(self):
        """
        Get first sequence number.
        This is incremented before writing to a primary file.
        """
        return self.unpack_dword(0x4)

    def hive_sequence2(self):
        """
        Get second sequence number.
        This is set to the same value as sequence1 after a primary files has been updated.
        """
        return self.unpack_dword(0x8)

    def validate_sequence_numbers(self):
        """
        Check if sequence numbers are equal.
        """
        return self.hive_sequence1() == self.hive_sequence2()

    def modification_timestamp(self):
        """
        Get the modified timestamp as a Python datetime.
        """
        return parse_windows_timestamp(self.unpack_qword(0xC))

    def reorganized_timestamp(self):
        """
        Get the last reorganized timestamp as a Python datetime.
        The field is used as of Windows 8, the value returned is garbage in previous versions of Windows.
        """
        return parse_windows_timestamp(self.unpack_qword(0xA8))

    def major_version(self):
        """
        Get the major version of the Windows Registry file format
        in use as an unsigned integer.
        """
        return self.unpack_dword(0x14)

    def minor_version(self):
        """
        Get the minor version of the Windows Registry file format
        in use as an unsigned integer.
        """
        return self.unpack_dword(0x18)

    def clustering_factor(self):
        """
        Get the clustering factor.
        """
        return self.unpack_dword(0x2C)

    def file_type(self):
        """
        Get the file type.
        """
        return FileType(self.unpack_dword(0x1C))

    def is_primary_file(self):
        """
        Check if this REGF block belongs to a primary (normal) file.
        """
        return self.file_type() == FileType.FILE_TYPE_PRIMARY

    def is_old_transaction_log_file(self):
        """
        Check if this REGF block belongs to an old transaction log file (used before Windows 8.1).
        """
        return (self.file_type() == FileType.FILE_TYPE_LOG_OLD_1) or (self.file_type() == FileType.FILE_TYPE_LOG_OLD_2)

    def is_new_transaction_log_file(self):
        """
        Check if this REGF block belongs to a new transaction log file (used as of Windows 8.1).
        """
        return self.file_type() == FileType.FILE_TYPE_LOG_NEW

    def file_format(self):
        """
        Get the file format.
        TODO: consider raising an exception if this isn't set to 1 (the only value possible).
        """
        return self.unpack_dword(0x20)

    def hive_flags(self):
        """
        Get the hive flags as an unsigned integer.
        """
        return self.unpack_dword(0x90)

    def hive_name(self):
        """
        Get the hive name of the open Windows Registry file as a string.
        """
        return self.unpack_string(0x30, 64).decode("utf-16le").rstrip("\x00")

    def first_hbin_offset(self):
        """
        Get the buffer offset of the first HBINBlock as an unsigned integer.
        Note: always returns 0x1000, nothing else is possible.
        """
        return 0x1000

    def hbins_size(self):
        """
        Size of all HBINBlock structures as an unsigned integer.
        """
        return self.unpack_dword(0x28)

    def last_hbin_offset(self):
        """
        Obsolete, use hbins_size instead.
        This doesn't return the offset of the last HBINBlock (as was believed before).
        """
        from warnings import warn
        warn("last_hbin_offset is obsolete, use hbins_size instead!")
        return self.unpack_dword(0x28)

    def calculate_checksum(self):
        """
        Checksum is calculated over the first 0x200 bytes:
        XOR of all D-Words from 0x00000000 to 0x000001FB with two edge cases.
        """
        xsum = 0
        idx = 0x0
        while idx <= 0x1FB:
            xsum ^= self.unpack_dword(idx)
            idx += 0x4
        if xsum == 0:
            return 1
        if xsum == 0xFFFFFFFF:
            return 0xFFFFFFFE
        return xsum

    def checksum(self):
        """
        Get the checksum stored in hive.
        """
        return self.unpack_dword(0x1FC)

    def validate_checksum(self):
        """
        Is the file checksum valid?
        """
        return self.calculate_checksum() == self.checksum()

    def validate(self):
        """
        Are the file checksum and sequence numbers valid?
        Obsolete, use recovery_required instead.
        """
        from warnings import warn
        warn("validate is obsolete, use recovery_required instead!")
        return self.validate_checksum() and self.validate_sequence_numbers()

    def recovery_required(self):
        """
        Are the file checksum and sequence numbers valid?
        Return a named tuple with two boolean values:
          - the recover_header is True when the REGF block recovery is required,
          - the recover_data is True when data recovery is required.
        """
        if not self.validate_checksum():
            # Header is invalid, this also implies data recovery
            return RecoveryStatus(recover_header = True, recover_data = True)

        if not self.validate_sequence_numbers():
            # Header is valid, data is in the mid-update state
            return RecoveryStatus(recover_header = False, recover_data = True)

        return RecoveryStatus(recover_header = False, recover_data = False)

    def first_key(self):
        first_hbin = next(self.hbins())

        key_offset = first_hbin.absolute_offset(self.unpack_dword(0x24))

        d = HBINCell(self._buf, key_offset, first_hbin)
        return NKRecord(self._buf, d.data_offset(), first_hbin)

    def hbins(self):
        """
        A generator that enumerates all HBIN (HBINBlock) structures in this Windows Registry.
        """
        h = HBINBlock(self._buf, self.first_hbin_offset(), self)
        yield h

        while h.has_next():
            h = h.next()
            yield h

    def first_log_entry_offset(self):
        """
        Get the offset of the first log entry as an unsigned integer.
        Note: always returns 0x200, nothing else is possible in new transaction log files.
        """
        return 0x200

    def log_entries(self):
        """
        A generator that enumerates all valid HvLE (HvLEBlock) structures in the transaction log file.
        """
        expected_seqnum = c_uint32(self.hive_sequence2())
        h = HvLEBlock(self._buf, self.first_log_entry_offset(), self)
        if h.sequence() == expected_seqnum.value and h.validate_log_entry():
            yield h

            while h.has_next():
                h = h.next()
                expected_seqnum.value += 1
                if h.sequence() == expected_seqnum.value and h.validate_log_entry():
                    yield h
                else:
                    break


class HBINCell(RegistryBlock):
    """
    HBIN data cell. An HBINBlock is continuously filled with HBINCell structures.
    The general structure is the length of the block, followed by a blob of data.
    """
    def __init__(self, buf, offset, parent):
        """
        Constructor.
        Arguments:
        - `buf`: Byte string containing Windows Registry file.
        - `offset`: The offset into the buffer at which the block starts.
        - `parent`: The parent block, which links to this block.
        """
        super(HBINCell, self).__init__(buf, offset, parent)
        self._size = self.unpack_int(0x0)

    def __str__(self):
        if self.is_free():
            return "HBIN Cell (free) at 0x%x" % (self._offset)
        else:
            return "HBIN Cell at 0x%x" % (self._offset)

    def is_free(self):
        """
        Is the cell free?
        """
        return self._size > 0

    def size(self):
        """
        Size of this cell, as an unsigned integer.
        """
        if self.is_free():
            return self._size
        else:
            return self._size * -1

    def next(self):
        """
        Returns the next HBINCell, which is located immediately after this.
        Note: This will always return an HBINCell starting at the next location
        whether or not the buffer is large enough. The calling function should
        check the offset of the next HBINCell to ensure it does not overrun the
        HBIN buffer.
        """
        try:
            return HBINCell(self._buf, self._offset + self.size(), self.parent())
        except:
            raise RegistryStructureDoesNotExist("HBINCell does not exist at 0x%x" % (self._offset + self.size()))

    def offset(self):
        """
        Accessor for absolute offset of this HBINCell.
        """
        return self._offset

    def data_offset(self):
        """
        Get the absolute offset of the data block of this HBINCell.
        """
        return self._offset + 0x4

    def raw_data(self):
        """
        Get the raw data from the buffer contained by this HBINCell.
        """
        return self._buf[self.data_offset():self.data_offset() + self.size()]

    def data_id(self):
        """
        Get the ID string of the data block of this HBINCell.
        """
        return self.unpack_string(0x4, 2)

    def abs_offset_from_hbin_offset(self, offset):
        """
        Offsets contained in HBIN cells are relative to the beginning of the first HBIN.
        This converts the relative offset into an absolute offset.
        """
        h = self.parent()
        while h.__class__.__name__ != "HBINBlock":
            h = h.parent()

        return h.first_hbin().offset() + offset

    def child(self):
        """
        Make a _guess_ as to the contents of this structure and
        return an instance of that class, or just a DataRecord
        otherwise.
        """
        if self.is_free():
            raise RegistryStructureDoesNotExist("HBINCell is free at 0x%x" % (self.offset()))

        id_ = self.data_id()

        if id_ == b"vk":
            return VKRecord(self._buf, self.data_offset(), self)
        elif id_ == b"nk":
            return NKRecord(self._buf, self.data_offset(), self)
        elif id_ == b"lf":
            return LFRecord(self._buf, self.data_offset(), self)
        elif id_ == b"lh":
            return LHRecord(self._buf, self.data_offset(), self)
        elif id_ == b"li":
            return LIRecord(self._buf, self.data_offset(), self)
        elif id_ == b"ri":
            return RIRecord(self._buf, self.data_offset(), self)
        elif id_ == b"sk":
            return SKRecord(self._buf, self.data_offset(), self)
        elif id_ == b"db":
            return DBRecord(self._buf, self.data_offset(), self)
        else:
            return DataRecord(self._buf, self.data_offset(), self)


class Record(RegistryBlock):
    """
    Abstract class for Records contained by cells in HBINs
    """
    def __init__(self, buf, offset, parent):
        """
        Constructor.
        Arguments:
        - `buf`: Byte string containing Windows Registry file.
        - `offset`: The offset into the buffer at which the block starts.
        - `parent`: The parent block, which links to this block. This SHOULD be an HBINCell.
        """
        super(Record, self).__init__(buf, offset, parent)

    def abs_offset_from_hbin_offset(self, offset):
        # TODO This violates DRY as this is a redefinition, see HBINCell.abs_offset_from_hbin_offset()
        """
        Offsets contained in HBIN cells are relative to the beginning of the first HBIN.
        This converts the relative offset into an absolute offset.
        """
        h = self.parent()
        while h.__class__.__name__ != "HBINBlock":
            h = h.parent()

        return h.first_hbin().offset() + offset


class DataRecord(Record):
    """
    A DataRecord is a HBINCell that does not contain any further structural data, but
    may contain, for example, the values pointed to by a VKRecord.
    """
    def __init__(self, buf, offset, parent):
        """
        Constructor.

        Arguments:
        - `buf`: Byte string containing Windows Registry file.
        - `offset`: The offset into the buffer at which the block starts.
        - `parent`: The parent block, which links to this block. This should be an HBINCell.
        """
        super(DataRecord, self).__init__(buf, offset, parent)

    def __str__(self):
        return "Data Record at 0x%x" % (self.offset())


class DBIndirectBlock(Record):
    """
    The DBIndirect block is a list of offsets to DataRecords with data
    size up to 0x3fd8.
    """
    def __init__(self, buf, offset, parent):
        """
        Constructor.
        Arguments:
        - `buf`: Byte string containing Windows Registry file.
        - `offset`: The offset into the buffer at which the block starts.
        - `parent`: The parent block, which links to this block. This should be an HBINCell.
        """
        super(DBIndirectBlock, self).__init__(buf, offset, parent)

    def __str__(self):
        return "Large Data Block at 0x%x" % (self.offset())

    def large_data(self, length):
        """
        Get the data pointed to by the indirect block. It may be large.
        Return a byte string.

        Each pointer references an HBIN cell whose body contributes up to
        0x3fd8 bytes.  Reading stops at the first unavailable or short
        fragment rather than taking bytes from an unrelated cell.
        """
        b = bytearray()
        count = 0
        pointer_count = (self.parent().size() - 4) // 4

        while length > 0 and count < pointer_count:
            off = self.abs_offset_from_hbin_offset(self.unpack_dword(4 * count))
            if off + 4 > len(self._buf):
                break

            cell = HBINCell(self._buf, off, self)
            if cell.is_free():
                break

            wanted = min(0x3fd8, length)
            available = max(0, cell.size() - 4)
            take = min(wanted, available)
            b += cell.raw_data()[0:take]
            count += 1
            length -= take

            if take < wanted:
                break

        return bytes(b)


class DBRecord(Record):
    """
    A DBRecord is a large data block, which is not thoroughly documented.
    Its similar to an inode in the Ext file systems.
    """
    def __init__(self, buf, offset, parent):
        """
        Constructor.
        Arguments:
        - `buf`: Byte string containing Windows Registry file.
        - `offset`: The offset into the buffer at which the block starts.
        - `parent`: The parent block, which links to this block. This should be an HBINCell.
        """
        super(DBRecord, self).__init__(buf, offset, parent)

        _id = self.unpack_string(0x0, 2)
        if _id != b"db":
            raise ParseException("Invalid DB Record ID")

    def __str__(self):
        return "Large Data Block at 0x%x" % (self.offset())

    def large_data(self, length):
        """
        Get the data described by the DBRecord. It may be large.
        Return a byte array.
        """
        off = self.abs_offset_from_hbin_offset(self.unpack_dword(0x4))
        cell = HBINCell(self._buf, off, self)
        dbi = DBIndirectBlock(self._buf, cell.data_offset(), cell)
        return dbi.large_data(length)


def decode_utf16le(s):
    """
    decode_utf16le attempts to decode a bytestring as UTF-16LE.
      If the string has an odd length, or some unexpected feature,
      this function does its best to handle the data. It does not
      catch any Unicode-related exceptions, such as UnicodeDecodeError,
      so these should be handled by the caller.

    @type s: bytes
    @param s: a bytestring to pase
    @rtype: unicode
    @return: the unicode string decoded from `s`
    @raises: this function does not attempt to catch any Unicode-related exception, so the caller should handle these.
    """
    if b"\x00\x00" in s:
        index = s.index(b"\x00\x00")
        if index > 2:
            if s[index - 2] != b"\x00"[0]: #py2+3
                #  61 00 62 00 63 64 00 00
                #                    ^  ^-- end of string
                #                    +-- index
                s = s[:index + 2]
            else:
                #  61 00 62 00 63 00 00 00
                #                 ^     ^-- end of string
                #                 +-- index
                s = s[:index + 3]
    if (len(s) % 2) != 0:
        s = s + b"\x00"
    s = s.decode("utf16")
    s = s.partition('\x00')[0]
    return s


class SourceRange(namedtuple("SourceRange", ["offset", "length"])):
    """
    A half-open byte range in the hive file.
    """
    __slots__ = ()

    @property
    def end(self):
        return self.offset + self.length

    @property
    def file_offset(self):
        return self.offset

    @property
    def physical_offset(self):
        return self.offset

    def to_dict(self):
        return {"offset": self.offset, "length": self.length, "end": self.end}


class MissingRange(namedtuple("MissingRange", ["offset", "length", "reason"])):
    """
    A half-open logical range of value data that could not be read from the hive.
    """
    __slots__ = ()

    @property
    def end(self):
        return self.offset + self.length

    def to_dict(self):
        return {
            "offset": self.offset,
            "length": self.length,
            "end": self.end,
            "reason": self.reason,
        }


class DataFragment(object):
    """
    One physically stored portion of a value's data.

    `logical_offset` is the offset of this fragment within the value's declared
    data.  `source_range` is where the bytes live in the hive file, when they
    are present.
    """
    def __init__(self, logical_offset, length, source_range, kind,
                 cell_offset=None, cell_size=None, available_length=None,
                 complete=True, reason=None):
        self.logical_offset = logical_offset
        self.length = length
        self.source_range = source_range
        self.kind = kind
        self.cell_offset = cell_offset
        self.cell_size = cell_size
        self.available_length = length if available_length is None else available_length
        self.complete = complete
        self.reason = reason

    @property
    def offset(self):
        return self.source_range.offset if self.source_range is not None else None

    @property
    def end(self):
        return self.logical_offset + self.length

    def to_dict(self):
        return {
            "logical_offset": self.logical_offset,
            "length": self.length,
            "logical_end": self.end,
            "offset": self.offset,
            "cell_offset": self.cell_offset,
            "cell_size": self.cell_size,
            "available_length": self.available_length,
            "kind": self.kind,
            "complete": self.complete,
            "reason": self.reason,
        }


class DataStorage(object):
    """
    A description of where a VK record's data lives.

    This models all three layouts used by the registry:
      - resident data packed into the VK record's data-offset field,
      - one HBIN cell containing the data,
      - a DB record plus DB indirect cells for large data.
    """
    def __init__(self, declared_raw_length, declared_length):
        self.declared_raw_length = declared_raw_length
        self.declared_length = declared_length
        self.inline = False
        self.storage_kind = "unavailable"
        self.data_offset = None
        self.cell_offset = None
        self.cell_size = None
        self.db_offset = None
        self.indirect_cell_offset = None
        self.fragments = []
        self.padding_ranges = []
        self.padding_start = None
        self.padding_data = b""
        self.problems = []
        self.data = b""
        self.proven_prefix_length = 0
        self.available_length = 0
        self.unexplained = []

    @property
    def resident(self):
        return self.inline

    @property
    def kind(self):
        return self.storage_kind

    @property
    def truncated(self):
        return self.proven_prefix_length < self.declared_length

    @property
    def complete(self):
        return not self.truncated

    @property
    def contradictions(self):
        return list(self.problems)

    @property
    def indirect_fragments(self):
        return [f for f in self.fragments if f.kind == "db-indirect"]

    @property
    def unexplained_ranges(self):
        return list(self.unexplained)

    def ranges_for(self, logical_start, logical_end):
        """
        Translate a half-open logical byte range into physical hive ranges.
        """
        if logical_end <= logical_start:
            return []

        ranges = []
        for fragment in self.fragments:
            if fragment.length <= 0 or fragment.source_range is None:
                continue
            frag_start = fragment.logical_offset
            frag_end = fragment.logical_offset + fragment.length
            start = max(logical_start, frag_start)
            end = min(logical_end, frag_end)
            if end <= start:
                continue
            offset = fragment.source_range.offset + (start - frag_start)
            length = end - start
            if ranges:
                previous = ranges[-1]
                if previous.end == offset:
                    ranges[-1] = SourceRange(previous.offset, previous.length + length)
                    continue
            ranges.append(SourceRange(offset, length))
        return ranges

    def missing_ranges_for(self, logical_start, logical_end):
        if logical_end <= logical_start:
            return []

        covered = []
        for source in self.ranges_for(logical_start, logical_end):
            for fragment in self.fragments:
                if fragment.length <= 0 or fragment.source_range is None:
                    continue
                if source.offset >= fragment.source_range.offset and \
                        source.offset + source.length <= fragment.source_range.end:
                    frag_start = fragment.logical_offset + \
                        (source.offset - fragment.source_range.offset)
                    covered.append((frag_start, frag_start + source.length))
                    break

        missing = []
        position = logical_start
        for start, end in sorted(covered):
            if start > position:
                missing.append(MissingRange(position, start - position, "unavailable bytes"))
            position = max(position, end)
        if position < logical_end:
            missing.append(MissingRange(position, logical_end - position, "unavailable bytes"))
        return missing

    def to_dict(self):
        return {
            "declared_raw_length": self.declared_raw_length,
            "declared_length": self.declared_length,
            "declared_length_hex": hex(self.declared_raw_length),
            "inline": self.inline,
            "resident": self.inline,
            "storage": self.storage_kind,
            "data_offset": self.data_offset,
            "cell_offset": self.cell_offset,
            "cell_size": self.cell_size,
            "db_offset": self.db_offset,
            "indirect_cell_offset": self.indirect_cell_offset,
            "available_length": self.available_length,
            "proven_prefix_length": self.proven_prefix_length,
            "complete": self.complete,
            "truncated": self.truncated,
            "problems": list(self.problems),
            "contradictions": list(self.problems),
            "fragments": [f.to_dict() for f in self.fragments],
            "indirect_fragments": [f.to_dict() for f in self.indirect_fragments],
            "padding": [r.to_dict() for r in self.padding_ranges],
            "unexplained": [r.to_dict() for r in self.unexplained],
            "unexplained_ranges": [r.to_dict() for r in self.unexplained],
        }


class MultiSZItem(object):
    """
    One string in a REG_MULTI_SZ sequence.
    """
    def __init__(self, index, value, logical_start, logical_end, ranges,
                 terminated, complete, separator_ranges=None):
        self.index = index
        self.value = value
        self.string = value
        self.text = value
        self.logical_start = logical_start
        self.logical_end = logical_end
        self.ranges = list(ranges)
        self.terminated = terminated
        self.complete = complete
        self.separator_ranges = list(separator_ranges or [])

    @property
    def start(self):
        return self.logical_start

    @property
    def end(self):
        return self.logical_end

    @property
    def physical_ranges(self):
        return list(self.ranges)

    @property
    def source_ranges(self):
        return list(self.ranges)

    @property
    def byte_length(self):
        return self.logical_end - self.logical_start

    def to_dict(self):
        return {
            "index": self.index,
            "value": self.value,
            "string": self.value,
            "logical_start": self.logical_start,
            "logical_end": self.logical_end,
            "byte_length": self.byte_length,
            "ranges": [r.to_dict() for r in self.ranges],
            "physical_ranges": [r.to_dict() for r in self.ranges],
            "source_ranges": [r.to_dict() for r in self.ranges],
            "terminated": self.terminated,
            "complete": self.complete,
            "separator_ranges": [r.to_dict() for r in self.separator_ranges],
        }


class MultiSZValue(list):
    """
    A REG_MULTI_SZ value.

    This is a list of strings for backwards compatibility, with forensic
    metadata attached:

      - `items`: the strings as MultiSZItem objects, including source ranges,
      - `terminated`: whether the empty-string sequence terminator was found,
      - `terminator_ranges`: physical ranges of that terminator,
      - `residual`: declared bytes following the terminator,
      - `padding`: allocated bytes outside the declared data,
      - `storage`: the DataStorage description.
    """
    def __init__(self, items, terminated, terminator_start=None,
                 terminator_end=None, storage=None, data=b"",
                 declared_length=None, residual=b"",
                 residual_ranges=None, padding=b"",
                 padding_ranges=None, unexplained=None,
                 problems=None, raw=b""):
        self.items = tuple(items)
        list.__init__(self, [item.value for item in items])
        self.terminated = terminated
        self.is_terminated = terminated
        self.terminator_start = terminator_start
        self.terminator_end = terminator_end
        self.storage = storage
        self.raw_data = raw
        self.data = data
        self.declared_length = declared_length
        self.residual = residual
        self.residual_bytes = residual
        self.residual_ranges = list(residual_ranges or [])
        self.padding = padding
        self.padding_bytes = padding
        self.padding_ranges = list(padding_ranges or [])
        self.unexplained = list(unexplained or [])
        self.problems = list(problems or [])

    @property
    def strings(self):
        return list(self)

    @property
    def sequence(self):
        return list(self)

    @property
    def terminated_by_empty_string(self):
        return self.terminated

    @property
    def terminator_ranges(self):
        if self.terminator_start is None or self.terminator_end is None:
            return []
        if self.storage is None:
            return [SourceRange(
                self.terminator_start,
                self.terminator_end - self.terminator_start)]
        return self.storage.ranges_for(self.terminator_start, self.terminator_end)

    @property
    def termination_ranges(self):
        return self.terminator_ranges

    @property
    def terminator_range(self):
        ranges = self.terminator_ranges
        return ranges[0] if ranges else None

    @property
    def residual_range(self):
        return list(self.residual_ranges)

    @property
    def physical_ranges(self):
        return [[r.to_dict() for r in item.ranges] for item in self.items]

    def item_ranges(self, index):
        return list(self.items[index].ranges)

    def to_dict(self):
        return {
            "strings": list(self),
            "sequence": list(self),
            "terminated": self.terminated,
            "terminator_start": self.terminator_start,
            "terminator_end": self.terminator_end,
            "terminator_ranges": [r.to_dict() for r in self.terminator_ranges],
            "items": [item.to_dict() for item in self.items],
            "declared_length": self.declared_length,
            "storage": self.storage.to_dict() if self.storage is not None else None,
            "residual": {
                "length": len(self.residual),
                "hex": binascii.hexlify(self.residual).decode("ascii"),
                "ranges": [r.to_dict() for r in self.residual_ranges],
            },
            "padding": {
                "length": len(self.padding),
                "hex": binascii.hexlify(self.padding).decode("ascii"),
                "ranges": [r.to_dict() for r in self.padding_ranges],
            },
            "unexplained": [r.to_dict() for r in self.unexplained],
            "unexplained_ranges": [r.to_dict() for r in self.unexplained],
            "problems": list(self.problems),
            "raw": binascii.hexlify(self.raw_data).decode("ascii"),
        }

    def export(self):
        return self.to_dict()

    def as_dict(self):
        return self.to_dict()

    def repair(self):
        """
        Return a newly constructed, legal sequence only when the repair is
        unambiguous.  The original bytes and metadata are never modified.
        """
        original = binascii.hexlify(self.raw_data).decode("ascii")
        if self.terminated:
            return {
                "applied": False,
                "reason": "sequence is already terminated",
                "original": original,
            }
        if self.storage is not None and not self.storage.complete:
            return {
                "applied": False,
                "reason": "value data is missing or contradictory",
                "original": original,
            }
        if self.unexplained:
            return {
                "applied": False,
                "reason": "unexplained ranges prevent an unambiguous repair",
                "original": original,
            }
        if len(self.raw_data) % 2 != 0:
            return {
                "applied": False,
                "reason": "odd trailing byte cannot be repaired unambiguously",
                "original": original,
            }
        repaired = self.raw_data + b"\x00\x00"
        parsed = parse_multisz(repaired)
        return {
            "applied": True,
            "reason": "appended the missing empty-string terminator",
            "original": original,
            "repaired": binascii.hexlify(repaired).decode("ascii"),
            "strings": list(parsed),
        }

    @property
    def repaired_value(self):
        repair = self.repair()
        return repair.get("strings") if repair.get("applied") else None


def _merge_ranges(ranges):
    merged = []
    for source in ranges:
        if merged and merged[-1].end == source.offset:
            merged[-1] = SourceRange(merged[-1].offset, merged[-1].length + source.length)
        else:
            merged.append(source)
    return merged


def _decode_utf16le_tolerant(data):
    try:
        return data.decode("utf-16le", "surrogatepass")
    except UnicodeDecodeError:
        return data.decode("utf-16le", "replace")


def parse_multisz(data, storage=None, declared_length=None):
    """
    Parse a REG_MULTI_SZ byte string.

    Strings are separated by one UTF-16LE NUL code unit.  The sequence ends at
    the first empty string; that empty string is the terminator and is never
    returned as a data element.  Bytes following the terminator are retained
    as `residual`, and are never parsed as additional strings.

    If `storage` is provided, it supplies the bytes, declared length, padding,
    and any missing ranges.  Parsing stops at the first missing fragment or
    contradiction, keeping the proven prefix and recording the rest of the
    logical data as unexplained.
    """
    if storage is not None:
        data = storage.data
        if declared_length is None:
            declared_length = storage.declared_length
    if declared_length is None:
        declared_length = len(data)

    gaps = []
    if storage is not None:
        gaps = sorted(storage.unexplained, key=lambda gap: gap.offset)
    first_gap = gaps[0].offset if gaps else None

    items = []
    terminated = False
    terminator_start = None
    terminator_end = None
    current_start = 0
    position = 0
    stopped_at_gap = False

    while position + 1 < len(data):
        if first_gap is not None and position >= first_gap:
            stopped_at_gap = True
            break

        code_unit = data[position:position + 2]
        position += 2

        if code_unit != b"\x00\x00":
            continue

        if position - 2 == current_start:
            # An empty string (one NUL code unit) is the sequence terminator.
            # Any further NUL bytes are residual and are not parsed.
            terminator_start = current_start
            terminator_end = position
            terminated = True
            break

        content_end = position - 2
        value = _decode_utf16le_tolerant(data[current_start:content_end])
        ranges = _logical_ranges(storage, current_start, content_end)
        separator_ranges = _logical_ranges(storage, content_end, position)
        items.append(MultiSZItem(
            len(items), value, current_start, content_end,
            _merge_ranges(ranges), True, True, separator_ranges))
        current_start = position

    if not terminated and not stopped_at_gap and first_gap is None and \
            current_start < len(data):
        # Data ended without a terminator.  Keep the bytes that form complete
        # code units as a final, explicitly unterminated item.
        content_end = len(data) - (len(data) % 2)
        if content_end > current_start:
            value = _decode_utf16le_tolerant(data[current_start:content_end])
            ranges = _logical_ranges(storage, current_start, content_end)
            items.append(MultiSZItem(
                len(items), value, current_start, content_end,
                _merge_ranges(ranges), False, False))

    if stopped_at_gap:
        proven_end = first_gap
    elif terminated:
        proven_end = terminator_end
    else:
        proven_end = len(data) - (len(data) % 2)

    residual = b""
    residual_ranges = []
    if terminated:
        residual = data[terminator_end:]
        residual_ranges = _logical_ranges(storage, terminator_end, len(data))

    padding = b""
    padding_ranges = []
    if storage is not None:
        padding_ranges = list(storage.padding_ranges)
        padding_parts = []
        for padding_range in padding_ranges:
            padding_parts.append(storage.padding_data[
                padding_range.offset - storage.padding_start:
                padding_range.end - storage.padding_start])
        padding = b"".join(padding_parts)

    unexplained = []
    for gap in gaps:
        if gap.offset >= proven_end:
            unexplained.append(gap)
    if stopped_at_gap:
        tail_start = max(proven_end, first_gap)
        if tail_start < declared_length:
            unexplained.append(MissingRange(
                tail_start, declared_length - tail_start,
                "parsing stopped at a missing or contradictory fragment"))
    if not terminated and first_gap is None and len(data) % 2 != 0:
        unexplained.append(MissingRange(
            len(data) - 1, 1, "incomplete UTF-16LE code unit"))

    problems = []
    if not terminated:
        problems.append("missing empty-string terminator")
    if len(data) % 2 != 0:
        problems.append("value data has an odd byte length")
    if stopped_at_gap:
        problems.append("value data is missing or contradictory")
    if storage is not None:
        for problem in storage.problems:
            if problem not in problems:
                problems.append(problem)
    if terminated and residual:
        problems.append("bytes remain after the sequence terminator")

    return MultiSZValue(
        items, terminated, terminator_start, terminator_end,
        storage=storage, data=data, declared_length=declared_length,
        residual=residual, residual_ranges=residual_ranges,
        padding=padding, padding_ranges=padding_ranges,
        unexplained=unexplained, problems=problems, raw=data)


def _logical_ranges(storage, logical_start, logical_end):
    if logical_end <= logical_start:
        return []
    if storage is not None:
        return storage.ranges_for(logical_start, logical_end)
    return [SourceRange(logical_start, logical_end - logical_start)]


decode_multisz = parse_multisz
decode_multi_sz = parse_multisz


class VKRecord(Record):
    """
    The VKRecord holds one name-value pair.  The data may be one of many types,
    including strings, integers, and binary data.
    """
    def __init__(self, buf, offset, parent):
        """
        Constructor.
        Arguments:
        - `buf`: Byte string containing Windows Registry file.
        - `offset`: The offset into the buffer at which the block starts.
        - `parent`: The parent block, which links to this block.
              This should be an HBINCell.
        """
        super(VKRecord, self).__init__(buf, offset, parent)

        _id = self.unpack_string(0x0, 2)
        if _id != b"vk":
            raise ParseException("Invalid VK Record ID")

    def data_type_str(self):
        """
        Get the value data's type as a string
        """
        data_type = self.data_type()
        if data_type == RegSZ:
            return "RegSZ"
        elif data_type == RegExpandSZ:
            return "RegExpandSZ"
        elif data_type == RegBin:
            return "RegBin"
        elif data_type == RegDWord:
            return "RegDWord"
        elif data_type == RegMultiSZ:
            return "RegMultiSZ"
        elif data_type == RegQWord:
            return "RegQWord"
        elif data_type == RegNone:
            return "RegNone"
        elif data_type == RegBigEndian:
            return "RegBigEndian"
        elif data_type == RegLink:
            return "RegLink"
        elif data_type == RegResourceList:
            return "RegResourceList"
        elif data_type == RegFullResourceDescriptor:
            return "RegFullResourceDescriptor"
        elif data_type == RegResourceRequirementsList:
            return "RegResourceRequirementsList"
        elif data_type == RegFileTime:
            return "RegFileTime"
        elif data_type == RegUint8:
            return "RegUint8"
        elif data_type == RegInt16:
            return "RegInt16"
        elif data_type == RegUint16:
            return "RegUint16"
        elif data_type == RegInt32:
            return "RegInt32"
        elif data_type == RegUint32:
            return "RegUint32"
        elif data_type == RegInt64:
            return "RegInt64"
        elif data_type == RegUint64:
            return "RegUint64"
        elif data_type == RegFloat:
            return "RegFloat"
        elif data_type == RegDouble:
            return "RegDouble"
        elif data_type == RegUnicodeChar:
            return "RegUnicodeChar"
        elif data_type == RegBoolean:
            return "RegBoolean"
        elif data_type == RegUnicodeString:
            return "RegUnicodeString"
        elif data_type == RegCompositeValue:
            return "RegCompositeValue"
        elif data_type == RegDateTimeOffset:
            return "RegDateTimeOffset"
        elif data_type == RegTimeSpan:
            return "RegTimeSpan"
        elif data_type == RegGUID:
            return "RegGUID"
        elif data_type == RegUnk111:
            return "RegUnk111"
        elif data_type == RegUnk112:
            return "RegUnk112"
        elif data_type == RegUnk113:
            return "RegUnk113"
        elif data_type == RegBytesArray:
            return "RegBytesArray"
        elif data_type == RegInt16Array:
            return "RegInt16Array"
        elif data_type == RegUint16Array:
            return "RegUint16Array"
        elif data_type == RegInt32Array:
            return "RegInt32Array"
        elif data_type == RegUInt32Array:
            return "RegUInt32Array"
        elif data_type == RegInt64Array:
            return "RegInt64Array"
        elif data_type == RegUInt64Array:
            return "RegUInt64Array"
        elif data_type == RegFloatArray:
            return "RegFloatArray"
        elif data_type == RegDoubleArray:
            return "RegDoubleArray"
        elif data_type == RegUnicodeCharArray:
            return "RegUnicodeCharArray"
        elif data_type == RegBooleanArray:
            return "RegBooleanArray"
        elif data_type == RegUnicodeStringArray:
            return "RegUnicodeStringArray"
        else:
            return "Unknown type: %s" % (hex(data_type))

    def __str__(self):
        if self.has_name():
            name = self.name()
        else:
            name = "(default)"

        data = ""
        data_type = self.data_type()
        if data_type == RegSZ or data_type == RegExpandSZ:
            data = self.data()[0:16] + "..."
        elif data_type == RegMultiSZ:
            data = str(len(self.data())) + " strings"
        elif data_type == RegDWord or data_type == RegQWord:
            data = str(hex(self.data()))
        elif data_type == RegNone:
            data = "(none)"
        elif data_type == RegBin:
            data = "(binary)"
        elif data_type in (RegFileTime, RegDateTimeOffset):
            data = self.data().isoformat("T") + "Z"
        elif data_type in (RegUint8, RegInt16, RegUint16, RegInt32, RegUint32,
                              RegInt64, RegUint64, RegFloat, RegDouble, RegUnicodeChar,
                              RegBoolean, RegUnicodeString, RegCompositeValue,
                              RegTimeSpan, RegGUID, RegUnk111, RegUnk112, RegUnk113, RegBytesArray,
                              RegInt16Array, RegUint16Array, RegInt32Array, RegUInt32Array,
                              RegInt64Array, RegUInt64Array, RegFloatArray, RegDoubleArray,
                              RegUnicodeCharArray, RegBooleanArray, RegUnicodeStringArray):
            data = str(self.data())
        else:
            data = "(unsupported)"

        return "VKRecord(Name: %s, Type: %s, Data: %s) at 0x%x" % (name,
                                                         self.data_type_str(),
                                                         data,
                                                         self.offset())

    def has_name(self):
        """
        Has a name? or perhaps we should use '(default)'
        """
        return self.unpack_word(0x2) != 0

    def has_ascii_name(self):
        """
        Is the name of this value in the ASCII charset?
        """
        return self.unpack_word(0x10) & 1 == 1

    def name(self):
        """
        Get the name, if it exists. If not, the empty string is returned.
        @return: unicode string containing the name
        """
        if not self.has_name():
            return ""
        name_length = self.unpack_word(0x2)
        unpacked_string = self.unpack_string(0x14, name_length)
        if self.has_ascii_name():
            return unpacked_string.decode("windows-1252")
        return unpacked_string.decode("utf-16le")

    def has_timestamp(self):
        """
        Has a timestamp? Only AppContainer settings.dat registry hive has this!
        """
        return (self.data_type() & 0x100 == 0x100) and (self.raw_data_length() >= 8)

    def timestamp(self):
        """
        Get the modified timestamp as a Python datetime. This is only valid for
        AppContainer settings.dat registry hive
        """
        if self.has_timestamp():
            return parse_windows_timestamp(struct.unpack_from(str("<Q"), self.raw_data()[-8:])[0])
        raise ValueError('value does not have a timestamp')

    def data_type(self):
        """
        Get the data type of this value data as an unsigned integer.
        """
        return self.unpack_dword(0xC) & DEVPROP_MASK_TYPE

    def data_length(self):
        """
        Get the length of this value data. This is the actual length of the data that should be parsed for the value.
        """
        size = self.unpack_dword(0x4)
        if size >= 0x80000000:
            size -= 0x80000000
        return size

    def raw_data_length(self):
        """
        Get the literal length of this value data. Some interpretation may be required to make sense of the value.
        """
        return self.unpack_dword(0x4)

    def data_offset(self):
        """
        Get the offset to the raw data associated with this value.
        """
        if self.raw_data_length() < 5 or self.raw_data_length() >= 0x80000000:
            return self.absolute_offset(0x8)
        else:
            return self.abs_offset_from_hbin_offset(self.unpack_dword(0x8))

    def data_storage(self):
        """
        Describe how the value's data is stored and where each byte comes from.

        The returned DataStorage contains the declared raw and effective
        lengths, the storage layout (resident, external cell, or DB indirect),
        the physical fragments, allocation padding, and any ranges that could
        not be explained.
        """
        storage = DataStorage(self.raw_data_length(), self.data_length())
        raw_length = storage.declared_raw_length
        length = storage.declared_length

        if raw_length >= 0x80000000 or raw_length < 5:
            # Small values and values with the high bit set keep their data
            # in the VK record's four-byte data-offset field.
            storage.inline = True
            storage.storage_kind = "resident"
            storage.data_offset = self.absolute_offset(0x8)
            storage.available_length = min(length, 4)
            storage.fragments.append(DataFragment(
                0, storage.available_length,
                SourceRange(storage.data_offset, storage.available_length),
                "resident", complete=storage.available_length == length))
            if storage.available_length < length:
                storage.problems.append(
                    "resident data is longer than the four bytes available in the VK record")
                storage.unexplained.append(MissingRange(
                    storage.available_length, length - storage.available_length,
                    "resident data exceeds the inline field"))
            storage.data = self._buf[storage.data_offset:storage.data_offset + storage.available_length]
            storage.proven_prefix_length = storage.available_length
            return storage

        storage.storage_kind = "external"
        storage.data_offset = self.abs_offset_from_hbin_offset(self.unpack_dword(0x8))

        if storage.data_offset + 4 > len(self._buf):
            storage.storage_kind = "unavailable"
            storage.problems.append("data cell is outside the hive")
            storage.unexplained.append(MissingRange(0, length, "data cell is outside the hive"))
            return storage

        cell = HBINCell(self._buf, storage.data_offset, self)
        storage.cell_offset = cell.offset()
        storage.cell_size = cell.size()
        available = max(0, cell.size() - 4)
        storage.available_length = available

        if cell.is_free():
            storage.storage_kind = "unavailable"
            storage.problems.append("data cell is free")
            storage.unexplained.append(MissingRange(0, length, "data cell is free"))
            return storage

        if 0x3fd8 < length < 0x80000000:
            storage.storage_kind = "db"
            self._read_indirect_data(storage, cell, length)
        else:
            data_start = cell.data_offset()
            if length > available:
                storage.problems.append(
                    "declared length exceeds the data cell's available bytes")
                storage.unexplained.append(MissingRange(
                    available, length - available, "data cell is too small"))
            read_length = min(length, available)
            storage.fragments.append(DataFragment(
                0, read_length, SourceRange(data_start, read_length),
                "external-cell", cell_offset=cell.offset(),
                cell_size=cell.size(), available_length=available,
                complete=read_length == length))
            storage.data = self._buf[data_start:data_start + read_length]
            storage.proven_prefix_length = read_length
            if available > length:
                storage.padding_start = data_start + length
                storage.padding_data = self._buf[
                    storage.padding_start:cell.data_offset() + available]
                storage.padding_ranges.append(
                    SourceRange(storage.padding_start, available - length))

        return storage

    def _read_indirect_data(self, storage, cell, length):
        """
        Read the DB record and its indirect cells, recording each fragment.
        """
        storage.db_offset = cell.offset()
        if cell.data_id() != b"db":
            storage.problems.append("large data is not referenced through a DB record")
            available = max(0, min(length, cell.size() - 4))
            data_start = cell.data_offset()
            storage.fragments.append(DataFragment(
                0, available, SourceRange(data_start, available),
                "external-cell", cell_offset=cell.offset(),
                cell_size=cell.size(), available_length=cell.size() - 4,
                complete=available == length))
            storage.data = self._buf[data_start:data_start + available]
            storage.proven_prefix_length = available
            if available < length:
                storage.unexplained.append(MissingRange(
                    available, length - available,
                    "large data not referenced through a DB record"))
            return

        db = DBRecord(self._buf, cell.data_offset(), cell)
        indirect_offset = db.abs_offset_from_hbin_offset(db.unpack_dword(0x4))
        storage.indirect_cell_offset = indirect_offset
        if indirect_offset + 4 > len(self._buf):
            storage.problems.append("DB indirect cell is outside the hive")
            storage.unexplained.append(MissingRange(
                0, length, "DB indirect cell is outside the hive"))
            return

        indirect_cell = HBINCell(self._buf, indirect_offset, db)
        if indirect_cell.is_free():
            storage.problems.append("DB indirect cell is free")
            storage.unexplained.append(MissingRange(0, length, "DB indirect cell is free"))
            return

        indirect = DBIndirectBlock(self._buf, indirect_cell.data_offset(), indirect_cell)
        pointer_count = (indirect_cell.size() - 4) // 4
        logical_offset = 0
        pieces = []

        for index in range(pointer_count):
            if logical_offset >= length:
                break
            relative_offset = indirect.unpack_dword(4 * index)
            fragment_offset = indirect.abs_offset_from_hbin_offset(relative_offset)
            wanted = min(0x3fd8, length - logical_offset)

            if fragment_offset + 4 > len(self._buf):
                storage.problems.append(
                    "DB indirect fragment %d is outside the hive" % index)
                storage.unexplained.append(MissingRange(
                    logical_offset, length - logical_offset,
                    "DB indirect fragment is outside the hive"))
                break

            fragment_cell = HBINCell(self._buf, fragment_offset, indirect)
            if fragment_cell.is_free():
                storage.problems.append(
                    "DB indirect fragment %d points to a free cell" % index)
                storage.unexplained.append(MissingRange(
                    logical_offset, length - logical_offset,
                    "DB indirect fragment is free"))
                break

            available = max(0, fragment_cell.size() - 4)
            read_length = min(wanted, available)
            data_start = fragment_cell.data_offset()
            storage.fragments.append(DataFragment(
                logical_offset, read_length, SourceRange(data_start, read_length),
                "db-indirect", cell_offset=fragment_cell.offset(),
                cell_size=fragment_cell.size(), available_length=available,
                complete=read_length == wanted))
            pieces.append(self._buf[data_start:data_start + read_length])
            logical_offset += read_length

            if read_length < wanted:
                storage.problems.append(
                    "DB indirect fragment %d does not contain its declared share" % index)
                storage.unexplained.append(MissingRange(
                    logical_offset, length - logical_offset,
                    "DB indirect fragment is too small"))
                break

        storage.data = b"".join(pieces)
        storage.proven_prefix_length = len(storage.data)
        if len(storage.data) < length:
            if not storage.unexplained:
                storage.unexplained.append(MissingRange(
                    len(storage.data), length - len(storage.data),
                    "DB indirect data ended before the declared length"))
            if not any("declared length" in problem for problem in storage.problems):
                storage.problems.append(
                    "declared length exceeds the DB indirect data")
        elif storage.fragments and storage.fragments[-1].cell_size is not None:
            last = storage.fragments[-1]
            if last.cell_size - 4 > last.length:
                padding_start = last.source_range.offset + last.length
                storage.padding_start = padding_start
                storage.padding_data = self._buf[
                    padding_start:last.cell_offset + last.cell_size]
                storage.padding_ranges.append(SourceRange(
                    padding_start, last.cell_size - 4 - last.length))

    def multisz(self):
        """
        Parse this value as REG_MULTI_SZ, preserving physical source ranges.
        """
        storage = self.data_storage()
        return parse_multisz(storage.data, storage=storage,
                             declared_length=storage.declared_length)

    def raw_data(self, overrun=0):
        """
        Get the unparsed raw data.
        """
        data_type = self.data_type()
        data_length = self.raw_data_length()
        data_offset = self.data_offset()
        ret = None

        if data_type == RegSZ or data_type == RegExpandSZ:
            if data_length >= 0x80000000:
                # data is contained in the data_offset field
                ret = self._buf[data_offset:data_offset + 0x4]
            elif 0x3fd8 < data_length < 0x80000000:
                d = HBINCell(self._buf, data_offset, self)
                if d.data_id() == b"db":
                    # this should always be the case
                    # but empirical testing does not confirm this
                    ret = d.child().large_data(data_length + overrun)
                else:
                    ret = d.raw_data()[:data_length + overrun]
            else:
                d = HBINCell(self._buf, data_offset, self)
                data_offset = d.data_offset()
                ret = self._buf[data_offset:data_offset + data_length]
        elif data_type == RegBin or data_type == RegNone \
             or data_type in (RegUint8, RegInt16, RegUint16, RegInt32, RegUint32, 
                              RegInt64, RegUint64, RegFloat, RegDouble, RegUnicodeChar, 
                              RegBoolean, RegUnicodeString, RegCompositeValue,RegDateTimeOffset, 
                              RegTimeSpan, RegGUID, RegUnk111, RegUnk112, RegUnk113, RegBytesArray, 
                              RegInt16Array, RegUint16Array, RegInt32Array, RegUInt32Array, 
                              RegInt64Array, RegUInt64Array, RegFloatArray, RegDoubleArray, 
                              RegUnicodeCharArray, RegBooleanArray, RegUnicodeStringArray):
            if data_length >= 0x80000000:
                data_length -= 0x80000000
                ret = self._buf[data_offset:data_offset + data_length + overrun]
            elif 0x3fd8 < data_length < 0x80000000:
                d = HBINCell(self._buf, data_offset, self)
                if d.data_id() == b"db":
                    # this should always be the case
                    # but empirical testing does not confirm this
                    ret = d.child().large_data(data_length + overrun)
                else:
                    ret = d.raw_data()[:data_length + overrun]
            else:
                ret = self._buf[data_offset + 4:data_offset + 4 + data_length + overrun]
        elif data_type == RegDWord:
            ret = self.unpack_binary(0x8, 0x4)
        elif data_type == RegMultiSZ:
            storage = self.data_storage()
            ret = storage.data
            if overrun > 0 and storage.fragments:
                last = storage.fragments[-1]
                if last.source_range is not None:
                    start = last.source_range.end
                    ret += self._buf[start:start + overrun]
        elif data_type == RegQWord:
            d = HBINCell(self._buf, data_offset, self)
            data_offset = d.data_offset()
            ret = self._buf[data_offset:data_offset + 0x8]
        elif data_type == RegBigEndian:
            d = HBINCell(self._buf, data_offset, self)
            data_offset = d.data_offset()
            ret = self._buf[data_offset:data_offset + 4]
        elif data_type == RegLink or \
                        data_type == RegResourceList or \
                        data_type == RegFullResourceDescriptor or \
                        data_type == RegResourceRequirementsList:
            if data_length >= 0x80000000:
                data_length -= 0x80000000
                ret = self._buf[data_offset:data_offset + data_length]
            elif 0x3fd8 < data_length < 0x80000000:
                d = HBINCell(self._buf, data_offset, self)
                if d.data_id() == b"db":
                    # this should always be the case
                    # but empirical testing does not confirm this
                    ret = d.child().large_data(data_length)
                else:
                    ret = d.raw_data()[:data_length]
            else:
                ret = self._buf[data_offset + 4:data_offset + 4 + data_length]
        elif data_type == RegFileTime:
            ret = self._buf[data_offset + 4:data_offset + 4 + data_length]
        elif data_length < 5 or data_length >= 0x80000000:
            ret = self.unpack_binary(0x8, 4)
        else:
            if data_length >= 0x80000000:
                data_length -= 0x80000000
                ret = self._buf[data_offset:data_offset + data_length]
            elif 0x3fd8 < data_length < 0x80000000:
                d = HBINCell(self._buf, data_offset, self)
                if d.data_id() == b"db":
                    # this should always be the case
                    # but empirical testing does not confirm this
                    ret = d.child().large_data(data_length)
                else:
                    ret = d.raw_data()[:data_length]
            else:
                ret = self._buf[data_offset + 4:data_offset + 4 + data_length]
        return ret

    def data(self, overrun=0):
        """
        Get the parsed data.
        This method will return various types based on the data type.

        RegSZ:
          Return a string containing the data, doing the best we can to convert it
          to ASCII or UNICODE.
        RegExpandSZ:
          Return a string containing the data, doing the best we can to convert it
          to ASCII or UNICODE. The special variables are not expanded.
        RegMultiSZ:
          Return a list of strings.
        RegNone:
          See RegBin
        RegDword:
          Return an unsigned integer containing the data.
        RegQword:
          Return an unsigned integer containing the data.
        RegBin:
          Return a sequence of bytes containing the binary data.
        RegBigEndian:
          Not currently supported. TODO.
        RegLink:
          Not currently supported. TODO.
        RegResourceList:
          Not currently supported. TODO.
        RegFullResourceDescriptor:
          Not currently supported. TODO.
        RegResourceRequirementsList:
          Not currently supported. TODO.
        RegFileTime:
          Return a datetime.datetime object
        """
        data_type = self.data_type()
        data_length = self.raw_data_length()
        d = self.raw_data(overrun=overrun)

        if data_type == RegSZ or data_type == RegExpandSZ:
            if overrun > 0:
                # decode_utf16le() only returns the first string, but if we explicitly
                # ask for overrun, let's make a best-effort to decode as much as possible.
                return d.decode('utf16')
            else:
                return decode_utf16le(d)
        elif data_type == RegBin or data_type == RegNone:
            return d
        elif data_type == RegDWord:
            return struct.unpack_from(str("<I"), d, 0)[0]
        elif data_type == RegMultiSZ:
            return self.multisz()
        elif data_type == RegQWord:
            return struct.unpack_from(str("<Q"), d, 0)[0]
        elif data_type == RegBigEndian:
            return struct.unpack_from(str(">I"), d, 0)[0]
        elif data_type == RegLink or \
                        data_type == RegResourceList or \
                        data_type == RegFullResourceDescriptor or \
                        data_type == RegResourceRequirementsList:
            # we don't really support these types, but can at least
            #  return raw binary for someone else to work with.
            return d
        elif data_type in (RegUint8, RegInt16, RegUint16, RegInt32, RegUint32, 
                        RegInt64, RegUint64, RegFloat, RegDouble, RegUnicodeChar, 
                        RegBoolean, RegUnicodeString, RegCompositeValue,RegDateTimeOffset, 
                        RegTimeSpan, RegGUID, RegUnk111, RegUnk112, RegUnk113, RegBytesArray, 
                        RegInt16Array, RegUint16Array, RegInt32Array, RegUInt32Array, 
                        RegInt64Array, RegUInt64Array, RegFloatArray, RegDoubleArray, 
                        RegUnicodeCharArray, RegBooleanArray, RegUnicodeStringArray):
            d = d[0:-8] # remove timestamp from end
            comp_type = data_type & 0xEFF # Apply mask for composite types
            return SettingsParse.ParseAppDataCompositeValue(comp_type, d, len(d))
        elif data_type == RegFileTime:
            return parse_windows_timestamp(struct.unpack_from(str("<Q"), d, 0)[0])
        elif data_length < 5 or data_length >= 0x80000000:
            return struct.unpack_from(str("<I"), d, 0)[0]
        else:
            raise UnknownTypeException("Unknown VK Record type 0x%x at 0x%x" % (data_type, self.offset()))


class SKRecord(Record):
    """
    Security Record. Contains Windows security descriptor,
    Which defines ownership and permissions for local values
    and subkeys.

    May be referenced by multiple NK records.
    """
    def __init__(self, buf, offset, parent):
        """
        Constructor.
        Arguments:
        - `buf`: Byte string containing Windows Registry file.
        - `offset`: The offset into the buffer at which the block starts.
        - `parent`: The parent block, which links to this block. This should be an HBINCell.
        """
        super(SKRecord, self).__init__(buf, offset, parent)

        _id = self.unpack_string(0x0, 2)
        if _id != b"sk":
            raise ParseException("Invalid SK Record ID")

        self._offset_prev_sk = self.unpack_dword(0x4)
        self._offset_next_sk = self.unpack_dword(0x8)

    def __str__(self):
        return "SK Record at 0x%x" % (self.offset())


class ValuesList(HBINCell):
    """
    A ValuesList is a simple structure of fixed length pointers/offsets to VKRecords.
    """
    def __init__(self, buf, offset, parent, number):
        """
        Constructor.
        Arguments:
        - `buf`: Byte string containing Windows Registry file.
        - `offset`: The offset into the buffer at which the block starts.
        - `parent`: The parent block, which links to this block. The parent of a ValuesList SHOULD be a NKRecord.
        """
        super(ValuesList, self).__init__(buf, offset, parent)
        self._number = number

    def __str__(self):
        return "ValueList(Length: %d) at 0x%x" % (self.parent().values_number(), self.offset())

    def values(self):
        """
        A generator that yields the VKRecords referenced by this list.
        """
        value_item = 0x0

        for _ in range(0, self._number):
            value_offset = self.abs_offset_from_hbin_offset(self.unpack_dword(value_item))

            d = HBINCell(self._buf, value_offset, self)
            v = VKRecord(self._buf, d.data_offset(), self)
            value_item += 4
            yield v


class SubkeyList(Record):
    """
    A base class for use by structures recording the subkeys of Registry key.
    The required overload is self.keys(), which is a generator for all the subkeys (NKRecords).
    The SubkeyList is not meant to be used directly.
    """
    def __init__(self, buf, offset, parent):
        """
        Constructor.
        Arguments:
        - `buf`: Byte string containing Windows Registry file.
        - `offset`: The offset into the buffer at which the block starts.
        - `parent`: The parent block, which links to this block. The parent of a SubkeyList SHOULD be a NKRecord.
        """
        super(SubkeyList, self).__init__(buf, offset, parent)

    def __str__(self):
        return "SubkeyList(Length: %d) at 0x%x" % (0, self.offset())

    def _keys_len(self):
        return self.unpack_word(0x2)

    def keys(self):
        """
        A generator that yields the NKRecords referenced by this list.
        The base SubkeyList class returns no NKRecords, since it should not be used directly.
        """
        return


class RIRecord(SubkeyList):
    """
    The RIRecord is a structure linking to structures containing
    a lists of offsets/pointers to subkey NKRecords. It is like a double (or more)
    indirect block.
    """
    def __init__(self, buf, offset, parent):
        """
        Constructor.
        Arguments:
        - `buf`: Byte string containing Windows Registry file.
        - `offset`: The offset into the buffer at which the block starts.
        - `parent`: The parent block, which links to this block.
        """
        super(RIRecord, self).__init__(buf, offset, parent)

    def __str__(self):
        return "RIRecord(Length: %d) at 0x%x" % (len(self.keys()), self.offset())

    def keys(self):
        """
        A generator that yields the NKRecords referenced by this list.
        ri style entry size.
        """
        key_index = 0x4

        for _ in range(0, self._keys_len()):
            key_offset = self.abs_offset_from_hbin_offset(self.unpack_dword(key_index))
            d = HBINCell(self._buf, key_offset, self)

            try:
                for k in d.child().keys():
                    yield k
            except RegistryStructureDoesNotExist:
                raise ParseException("Unsupported subkey list encountered.")

            key_index += 4


class DirectSubkeyList(SubkeyList):
    def __init__(self, buf, offset, parent):
        """
        Constructor.
        Arguments:
        - `buf`: Byte string containing Windows Registry file.
        - `offset`: The offset into the buffer at which the block starts.
        - `parent`: The parent block, which links to this block.
        """
        super(DirectSubkeyList, self).__init__(buf, offset, parent)

    def __str__(self):
        return "DirectSubkeyList(Length: %d) at 0x%x" % (self._keys_len(), self.offset())

    def keys(self):
        """
        A generator that yields the NKRecords referenced by this list.
        Assumes each entry is 0x8 bytes long (lf / lh style).
        """
        key_index = 0x4

        for _ in range(0, self._keys_len()):
            key_offset = self.abs_offset_from_hbin_offset(self.unpack_dword(key_index))

            d = HBINCell(self._buf, key_offset, self)
            yield NKRecord(self._buf, d.data_offset(), self)
            key_index += 8


class LIRecord(DirectSubkeyList):
    """
    The LIRecord is a simple structure containing a list of offsets/pointers
    to subkey NKRecords. It is a single indirect block.
    """
    def __init__(self, buf, offset, parent):
        """
        Constructor.
        Arguments:
        - `buf`: Byte string containing Windows Registry file.
        - `offset`: The offset into the buffer at which the block starts.
        - `parent`: The parent block, which links to this block.
        """
        super(LIRecord, self).__init__(buf, offset, parent)

    def __str__(self):
        return "LIRecord(Length: %d) at 0x%x" % (self._keys_len(), self.offset())

    def keys(self):
        """
        A generator that yields the NKRecords referenced by this list.
        li style entry size.
        """
        key_index = 0x4

        for _ in range(0, self._keys_len()):
            key_offset = self.abs_offset_from_hbin_offset(self.unpack_dword(key_index))

            d = HBINCell(self._buf, key_offset, self)
            yield NKRecord(self._buf, d.data_offset(), self)
            key_index += 4


class LFRecord(DirectSubkeyList):
    """
    The LFRecord is a simple structure containing a list of offsets/pointers
    to subkey NKRecords.
    The LFRecord also contains a hash for the name of the subkey pointed to
    by the offset, which enables more efficient searching of the Registry tree.
    """
    def __init__(self, buf, offset, parent):
        """
        Constructor.
        Arguments:
        - `buf`: Byte string containing Windows Registry file.
        - `offset`: The offset into the buffer at which the block starts.
        - `parent`: The parent block, which links to this block.
        """
        super(LFRecord, self).__init__(buf, offset, parent)
        _id = self.unpack_string(0x0, 2)
        if _id != b"lf":
            raise ParseException("Invalid LF Record ID")

    def __str__(self):
        return "LFRecord(Length: %d) at 0x%x" % (self._keys_len(), self.offset())


class LHRecord(DirectSubkeyList):
    """
    The LHRecord is a simple structure containing a list of offsets/pointers
    to subkey NKRecords.
    The LHRecord also contains a hash for the name of the subkey pointed to
    by the offset, which enables more efficient searching of the Registry tree.
    The LHRecord is analogous to the LFRecord, but it uses a different hashing function.
    """
    def __init__(self, buf, offset, parent):
        """
        Constructor.
        Arguments:
        - `buf`: Byte string containing Windows Registry file.
        - `offset`: The offset into the buffer at which the block starts.
        - `parent`: The parent block, which links to this block.
        """
        super(LHRecord, self).__init__(buf, offset, parent)
        _id = self.unpack_string(0x0, 2)
        if _id != b"lh":
            raise ParseException("Invalid LH Record ID")

    def __str__(self):
        return "LHRecord(Length: %d) at 0x%x" % (self._keys_len(), self.offset())


class NKRecord(Record):
    """
    The NKRecord defines the tree-like structure of the Windows Registry.
    It contains pointers/offsets to the ValueList (values associated with the given record),
    and to subkeys.
    """
    def __init__(self, buf, offset, parent):
        """
        Constructor.
        Arguments:
        - `buf`: Byte string containing Windows Registry file.
        - `offset`: The offset into the buffer at which the block starts.
        - `parent`: The parent block, which links to this block. This should be a HBINCell.
        """
        super(NKRecord, self).__init__(buf, offset, parent)
        _id = self.unpack_string(0x0, 2)
        if _id != b"nk":
            raise ParseException("Invalid NK Record ID")

    def __str__(self):
        classname = self.classname()
        if not self.has_classname():
            classname = "(none)"

        if self.is_root():
            return "Root NKRecord(Class: %s, Name: %s) at 0x%x" % (classname,
                                                                   self.name(),
                                                                   self.offset())
        else:
            return "NKRecord(Class: %s, Name: %s) at 0x%x" % (classname,
                                                              self.name(),
                                                              self.offset())

    def has_classname(self):
        """
        Does this have a classname?
        """
        return self.unpack_word(0x4A) > 0

    def classname(self):
        """
        If this has a classname, get it as a string. Otherwise, return the empty string.
        @return: unicode string containing the class name
        """
        if not self.has_classname():
            return ""

        classname_offset = self.unpack_dword(0x30)
        classname_length = self.unpack_word(0x4A)

        offset = self.abs_offset_from_hbin_offset(classname_offset)
        d = HBINCell(self._buf, offset, self)
        return struct.unpack_from(str("<%ds") % (classname_length), self._buf, d.data_offset())[0].decode("utf-16le").rstrip("\x00")

    def timestamp(self):
        """
        Get the modified timestamp as a Python datetime.
        """
        return parse_windows_timestamp(self.unpack_qword(0x4))

    def access_bits(self):
        """
        Get the access bits of the registry key as an unsigned integer.
        The field is used as of Windows 8.
        """
        return self.unpack_dword(0xC) & 0xFF

    def has_ascii_name(self):
        return self.unpack_word(0x2) & 0x0020 > 0

    def name(self):
        """
        Return the registry key name as a string.
        @return: unicode string containing the name
        """
        name_length = self.unpack_word(0x48)
        unpacked_string = self.unpack_string(0x4C, name_length)
        if self.has_ascii_name():
            return unpacked_string.decode("windows-1252")
        return unpacked_string.decode("utf-16le")

    def path(self):
          """
          Return the full path of the registry key as a unicode string
          @return: unicode string containing the path
          """
          p = self

          name = [p.name()]
          offsets = set([p._offset])
          while p.has_parent_key():
              p = p.parent_key()
              if p._offset in offsets:
                  name.append("[path cycle]")
                  break
              name.append(p.name())
              offsets.add(p._offset)
          return '\\'.join(reversed(name))

    def is_root(self):
        """
        Is this a root key?
        """
        return self.unpack_word(0x2) & 0x0004 > 0

    def has_parent_key(self):
        """
        Is there a parent key? There should always be a parent key, unless
        this is a root key (see self.is_root())
        """
        if self.is_root():
            return False
        try:
            self.parent_key()
            return True
        except ParseException:
            return False

    def parent_key(self):
        """
        Get the parent_key, which will be an NKRecord.
        """
        offset = self.abs_offset_from_hbin_offset(self.unpack_dword(0x10))

        d = HBINCell(self._buf, offset, self.parent())
        return NKRecord(self._buf, d.data_offset(), self.parent())

    def sk_record(self):
        """
        Get the security descriptor associated with this NKRecord as an SKRecord.
        """
        offset = self.abs_offset_from_hbin_offset(self.unpack_dword(0x2C))

        d = HBINCell(self._buf, offset, self)
        return SKRecord(self._buf, d.data_offset(), d)

    def values_number(self):
        """
        Get the number of values associated with this NKRecord/Key.
        """
        num = self.unpack_dword(0x24)
        if num == 0xFFFFFFFF:
            return 0
        return num

    def values_list(self):
        """
        Get the values as a ValuesList.
        Raises RegistryStructureDoesNotExist if this NKRecord has no values.
        """
        if self.values_number() == 0:
            raise RegistryStructureDoesNotExist("NK Record has no associated values.")

        values_list_offset = self.abs_offset_from_hbin_offset(self.unpack_dword(0x28))

        d = HBINCell(self._buf, values_list_offset, self)
        return ValuesList(self._buf, d.data_offset(), self, self.values_number())

    def subkey_number(self):
        """
        Get the number of subkeys of this key.
        """
        number = self.unpack_dword(0x14)
        if number == 0xFFFFFFFF:
            return 0
        return number

    def subkey_list(self):
        """
        Get the subkeys of this key as a descendant of SubkeyList.
        Raises RegistryStructureDoesNotExists if this NKRecord does not have any subkeys.
        See NKRecord.subkey_number() to check for the existence of subkeys.
        """
        if self.subkey_number() == 0:
            raise RegistryStructureDoesNotExist("NKRecord has no subkey list at 0x%x" % (self.offset()))

        subkey_list_offset = self.abs_offset_from_hbin_offset(self.unpack_dword(0x1C))

        d = HBINCell(self._buf, subkey_list_offset, self)
        id_ = d.data_id()

        if id_ == b"lf":
            l = LFRecord(self._buf, d.data_offset(), self)
        elif id_ == b"lh":
            l = LHRecord(self._buf, d.data_offset(), self)
        elif id_ == b"ri":
            l = RIRecord(self._buf, d.data_offset(), self)
        elif id_ == b"li":
            l = LIRecord(self._buf, d.data_offset(), self)
        else:
            raise ParseException("Subkey list with type 0x%s encountered, but not yet supported." %
                                 (binascii.hexlify(id_).decode('ascii')))

        return l


class HBINBlock(RegistryBlock):
    """
    A HBINBlock is the basic allocation block of the Windows Registry.
    It's length is multiple of 0x1000.
    """
    def __init__(self, buf, offset, parent):
        """
        Constructor.
        Arguments:
        - `buf`: Byte string containing Windows Registry file.
        - `offset`: The offset into the buffer at which the block starts.
        - `parent`: The parent block, which links to this block. The parent of the first HBINBlock
        should be the REGFBlock, and the parents of other HBINBlocks should be the preceding
        HBINBlocks.
        """
        super(HBINBlock, self).__init__(buf, offset, parent)

        _id = self.unpack_dword(0)
        if _id != 0x6E696268:
            raise ParseException("Invalid HBIN ID")

        self._reloffset_next_hbin = self.unpack_dword(0x8)
        self._offset_next_hbin = self._reloffset_next_hbin + self._offset

    def __str__(self):
        return "HBIN at 0x%x" % (self._offset)

    def first_hbin(self):
        """
        Get the first HBINBlock.
        """
        reloffset_from_first_hbin = self.unpack_dword(0x4)
        return HBINBlock(self._buf, (self.offset() - reloffset_from_first_hbin), self.parent())

    def has_next(self):
        """
        Does another HBINBlock exist after this one?
        """
        regf = self.first_hbin().parent()
        if regf.hbins_size() + regf.first_hbin_offset() == self._offset_next_hbin:
            return False

        try:
            self.next()
            return True
        except (ParseException, struct.error):
            return False

    def next(self):
        """
        Get the next HBIN after this one.
        Note: This blindly attempts to create it regardless of its existence.
        """
        return HBINBlock(self._buf, self._offset_next_hbin, self.parent())

    def cells(self):
        """
        Get a generator that yields each HBINCell contained in this HBIN.
        These are not necessarily in use, or linked to, from the root key.
        """
        c = HBINCell(self._buf, self._offset + 0x20, self)

        while c.offset() < self._offset_next_hbin:
            yield c
            if c.offset() + c.size() == self._offset_next_hbin:
                break
            c = c.next()

    def records(self):
        """
        Obsolete, use cells instead.
        """
        from warnings import warn
        warn("records is obsolete, use cells instead!")
        return self.cells()

class HvLEBlock(RegistryBlock):
    """
    A HvLEBlock is the log entry in a new transaction log file.
    It's length is multiple of 0x200.
    """
    def __init__(self, buf, offset, parent):
        """
        Constructor.
        Arguments:
        - `buf`: Byte string containing Windows Registry transaction log file.
        - `offset`: The offset into the file-like object at which the block starts.
        - `parent`: The parent block, which links to this block. The parent of the first HvLEBlock
        should be the REGFBlock, and the parents of other HvLEBlocks should be the preceding
        HvLEBlocks.
        """
        super(HvLEBlock, self).__init__(buf, offset, parent)

        _id = self.unpack_dword(0)
        if _id != 0x454C7648:
            raise ParseException("Invalid HvLE ID")

        self._offset_next_hvle = self._offset + self.size()
        self._marvin32seed = 0x82EF4D887A4E55C5

    def __str__(self):
        return "HvLE at 0x%x" % (self._offset)

    def marvin32_hash(self, buf):
        """
        Hash the buf using Marvin32 with a predefined seed.
        """
        def rotl(x, n, w):
            return (x.value << n) | (x.value >> (w - n))

        def to_uint32_le(four_bytes):
            b1, b2, b3, b4 = bytearray(four_bytes)
            return b1 | (b2 << 8) | (b3 << 16) | (b4 << 24)

        def marvin32_mix(state, val):
            lo, hi = state
            lo.value += val.value
            hi.value ^= lo.value
            lo.value = rotl(lo, 20, 32) + hi.value
            hi.value = rotl(hi, 9, 32) ^ lo.value
            lo.value = rotl(lo, 27, 32) + hi.value
            hi.value = rotl(hi, 19, 32)
            return (lo, hi)

        seed = self._marvin32seed
        lo = c_uint32(seed)
        hi = c_uint32(seed >> 32)
        state = (lo, hi)

        length = len(buf)
        pos = 0
        val = c_uint32()

        while length >= 4:
            val.value = to_uint32_le(buf[pos:pos+4])
            state = marvin32_mix(state, val)
            pos += 4
            length -= 4

        final = c_uint32(0x80)
        if length == 3:
            final.value = (final.value << 8) | buf[pos+2]
        elif length == 2:
            final.value = (final.value << 8) | buf[pos+1]
        elif length == 1:
            final.value = (final.value << 8) | buf[pos]

        state = marvin32_mix(state, final)
        state = marvin32_mix(state, c_uint32(0))
        lo, hi = state
        return (hi.value << 32 | lo.value)

    def size(self):
        """
        Get the size of this HvLEBlock.
        """
        return self.unpack_dword(0x4)

    def hash_1(self):
        """
        Get the value of Hash-1.
        """
        return self.unpack_qword(0x18)

    def calculate_hash_1(self):
        """
        Calculate the Hash-1.
        """
        return self.marvin32_hash(self._buf[self._offset+LOG_ENTRY_SIZE_HEADER:self._offset+self.size()])

    def hash_2(self):
        """
        Get the value of Hash-2.
        """
        return self.unpack_qword(0x20)

    def calculate_hash_2(self):
        """
        Calculate the Hash-2.
        """
        return self.marvin32_hash(self._buf[self._offset:self._offset+32])

    def validate_log_entry(self):
        """
        Check if this log entry is valid.
        """
        if (self.size() <= LOG_ENTRY_SIZE_HEADER) or (self.size() % LOG_ENTRY_SIZE_ALIGNMENT != 0):
            return False

        if self.hbins_size() % 0x1000 != 0:
            return False

        if self.hash_2() != self.calculate_hash_2() or self.hash_1() != self.calculate_hash_1():
            return False

        return True

    def hive_flags(self):
        """
        Get the hive flags as an unsigned integer.
        """
        return self.unpack_dword(0x8)

    def sequence(self):
        """
        Get the sequence number as an unsigned integer.
        """
        return self.unpack_dword(0xC)

    def hbins_size(self):
        """
        Get the size of all HBINBlock structures as an unsigned integer.
        """
        return self.unpack_dword(0x10)

    def dirty_pages_count(self):
        """
        Get the number of dirty pages in this log entry.
        """
        return self.unpack_dword(0x14)

    def dirty_pages_references(self):
        """
        Get a generator that yields dirty pages references in this log entry.
        """
        i = self.dirty_pages_count()
        rel_offset = 0
        while i > 0:
            c = DirtyPageReference(self._buf, self._offset + rel_offset + 0x28, self)
            yield c
            rel_offset += 8
            i -= 1

    def first_dirty_page_offset(self):
        """
        Get the offset of the first dirty page in this log entry.
        """
        return self._offset + LOG_ENTRY_SIZE_HEADER + 8*self.dirty_pages_count()

    def dirty_pages_with_references(self):
        """
        Get a generator that yields tuples with a DirtyPageReference and a DirtyPage.
        """
        current_offset = self.first_dirty_page_offset()
        for dirty_page_reference in self.dirty_pages_references():
            current_size = dirty_page_reference.size()
            dirty_page = DirtyPage(self._buf, current_offset, current_size, self)
            yield (dirty_page_reference, dirty_page)
            current_offset += dirty_page_reference.size()

    def has_next(self):
        """
        Does another HvLEBlock exist after this one?
        """
        try:
            self.next()
            return True
        except (ParseException, struct.error):
            return False

    def next(self):
        """
        Get the next HvLE after this one.
        Note: This blindly attempts to create it regardless of its existence.
        """
        return HvLEBlock(self._buf, self._offset_next_hvle, self.parent())


class DirtyPageReference(RegistryBlock):
    """
    A structure describing a single dirty page in the HvLEBlock.
    """
    def __init__(self, buf, offset, parent):
        """
        Constructor.
        Arguments:
        - `buf`: Byte string containing Windows Registry transaction log file.
        - `offset`: The offset into the buffer at which the block starts.
        - `parent`: The parent block, which links to this block.
        """
        super(DirtyPageReference, self).__init__(buf, offset, parent)

    def offset(self):
        """
        Offset of a dirty page in a primary file (relative from the first HBINBlock).
        """
        return self.unpack_dword(0x0)

    def size(self):
        """
        Size of a dirty page.
        """
        return self.unpack_dword(0x4)

class DirtyPage(RegistryBlock):
    """
    A a single dirty page in the HvLEBlock.
    """
    def __init__(self, buf, offset, size, parent):
        """
        Constructor.
        Arguments:
        - `buf`: Byte string containing Windows Registry transaction log file.
        - `offset`: The offset into the buffer at which the block starts.
        - `parent`: The parent block, which links to this block.
        """
        super(DirtyPage, self).__init__(buf, offset, parent)
        self._size = size

    def data(self):
        """
        Return the dirty page.
        """
        return self._buf[self._offset : self._offset + self._size]
