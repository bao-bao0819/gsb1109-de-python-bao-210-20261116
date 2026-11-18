"""
Read and write the SAS XPORT/XPT file format from SAS Version 5 or 6.

The SAS V5 Transport File format, also called XPORT, or simply XPT, ...
"""

# All "records" are 80 bytes long, padded if necessary.
# Character data are ASCII-encoded.
# Integer data are IBM-style integer format.
# Floating point data are IBM-style double format.

# Standard Library
import contextlib
import logging
import math
import re
import struct
import warnings
from collections.abc import Iterator, Mapping
from datetime import datetime

# Community Packages
import pandas as pd

# Xport Modules
import xport

__all__ = [
    'load',
    'loads',
    'dump',
    'dumps',
    'load_partial',
    'resume',
    'PartialLoad',
    'TruncatedDocumentError',
    '_encoding',
]

LOG = logging.getLogger(__name__)

# TODO: Make text encoding a method argument, not a global.
# TODO: Consider using Windows-1252 as the default, like HTML5.
TEXT_DATA_ENCODING = 'ISO-8859-1'
TEXT_METADATA_ENCODING = 'ascii'


@contextlib.contextmanager
def _encoding(data=TEXT_DATA_ENCODING, metadata=TEXT_METADATA_ENCODING):
    """
    Temporarily change the module's text encoding.
    """
    global TEXT_DATA_ENCODING
    global TEXT_METADATA_ENCODING
    stash = {'data': TEXT_DATA_ENCODING, 'metadata': TEXT_METADATA_ENCODING}
    try:
        TEXT_DATA_ENCODING = data
        TEXT_METADATA_ENCODING = metadata
        yield
    finally:
        LOG.debug(f'Reverting text encoding to {stash}')
        TEXT_DATA_ENCODING = stash['data']
        TEXT_METADATA_ENCODING = stash['metadata']


class Overflow(ArithmeticError):
    """Number too large to express."""


class Underflow(ArithmeticError):
    """Number too small to express, rounds to zero."""


class PartialLoad:
    """
    Outcome of reading a document that may end in a truncated member.

    ``library`` holds only the members whose full boundary was confirmed;
    each one has a matching :class:`xport.MemberReceipt` in ``receipts``.
    The undetermined ``tail`` is never folded into a member or mistaken for
    observations.  ``resume_offset`` is the byte position of the first
    unconfirmed byte, and ``prefix`` is every confirmed byte up to that
    position, allowing a later :func:`resume` to continue from proof.
    """

    def __init__(
        self,
        library,
        receipts,
        tail,
        resume_offset,
        prefix,
        incomplete_member=None,
        incomplete_header=None,
        incomplete_member_start=None,
        confirmed_observations=0,
        reason='',
        is_complete=None,
    ):
        """
        Bundle the confirmed library with its undetermined tail.

        ``is_complete`` defaults to "complete only when there is no tail",
        but the parser passes an explicit value because a cut can land on an
        observation or record boundary without any stray trailing bytes.
        """
        self.library = library
        self.receipts = receipts
        self.tail = tail
        self.resume_offset = resume_offset
        self.prefix = prefix
        self.incomplete_member = incomplete_member
        self.incomplete_header = incomplete_header
        self.incomplete_member_start = incomplete_member_start
        self.confirmed_observations = confirmed_observations
        self.reason = reason
        self._is_complete = not tail if is_complete is None else is_complete

    @property
    def complete(self):
        """
        Whether the document ended cleanly, with no undetermined bytes.
        """
        return self._is_complete

    def __repr__(self):
        """
        REPL-format string.
        """
        status = 'complete' if self.complete else f'truncated at byte {self.resume_offset}'
        return f'<PartialLoad {status}, members={list(self.receipts)}>'


class TruncatedDocumentError(ValueError):
    """
    An XPORT document ends in the middle of a member.

    The document cannot be marked as a complete, successful read because
    at least one member is missing bytes.  Everything that was confirmed --
    the received members, their receipts, and the undetermined tail -- is
    available on the wrapped :class:`PartialLoad` as :attr:`partial`.
    """

    def __init__(self, partial):
        """
        Wrap a :class:`PartialLoad` marked as incomplete.
        """
        self.partial = partial
        name = getattr(partial.incomplete_header, 'name', None) or 'a member'
        super().__init__(
            f'Truncated while reading {name!r} at byte {partial.resume_offset}: '
            f'{len(partial.tail)} undetermined byte(s); {partial.reason}'
        )

    @property
    def tail(self):
        """The undetermined trailing bytes."""
        return self.partial.tail

    @property
    def resume_offset(self):
        """Byte position of the first unconfirmed byte."""
        return self.partial.resume_offset

    @property
    def receipts(self):
        """Receipts for the fully received members."""
        return self.partial.receipts


class Namestr:
    """
    Variable metadata from a SAS Version 5 or 6 Transport (XPORT) file.
    """

    # Here is the C structure definition for the namestr record:
    #
    # struct NAMESTR {
    #    short ntype;       /* VARIABLE TYPE: 1=NUMERIC, 2=CHAR       */
    #    short nhfun;       /* HASH OF NNAME (always 0)               */
    #    short nlng;        /* LENGTH OF VARIABLE IN OBSERVATION      */
    #    short nvar0;       /* VARNUM                                 */
    #    char8 nname;       /* NAME OF VARIABLE                       */
    #    char40 nlabel;     /* LABEL OF VARIABLE                      */
    #    char8 nform;       /* NAME OF FORMAT                         */
    #    short nfl;          /* FORMAT FIELD LENGTH OR 0               */
    #    short nfd;         /* FORMAT NUMBER OF DECIMALS              */
    #    short nfj;         /* 0=LEFT JUSTIFICATION, 1=RIGHT JUST     */
    #    char nfill[2];      /* (UNUSED, FOR ALIGNMENT AND FUTURE)     */
    #    char8 niform;      /* NAME OF INPUT FORMAT                   */
    #    short nifl;         /* INFORMAT LENGTH ATTRIBUTE              */
    #    short nifd;        /* INFORMAT NUMBER OF DECIMALS            */
    #    long npos;         /* POSITION OF VALUE IN OBSERVATION       */
    #    char rest[52];     /* remaining fields are irrelevant         */
    #    };
    #
    # Note that the length given in the last 4 bytes of the member
    # header record indicates the actual number of bytes for the NAMESTR
    # structure. The size of the structure listed above is 140 bytes.
    # Under VAX/VMS, the size will be 136 bytes, meaning that the 'rest'
    # variable may be truncated.

    fmts = {
        140: '>hhhh8s40s8shhh2s8shhl52s',
        136: '>hhhh8s40s8shhh2s8shhl48s',
    }

    def __init__(self, vtype, length, number, name, label, format, informat, position):
        """
        Initialize a ``Namestr``.
        """
        self.vtype = vtype
        self.length = length
        self.number = number
        self.name = name
        self.label = label
        self.format = format
        self.informat = informat
        self.position = position

    def __eq__(self, other):
        """Equality."""
        attributes = [
            'vtype',
            'name',
            'label',
            'format',
            'informat',
            'position',
        ]
        return all(getattr(self, name) == getattr(other, name) for name in attributes)

    @classmethod
    def from_variable(cls, variable: xport.Variable, number=None, position=None):
        """
        Construct a ``Namestr`` from an ``xport.Variable``.
        """
        if variable.vtype is not None:
            vtype = variable.vtype
        elif variable.dtype.kind in {'f', 'i'}:
            vtype = xport.VariableType.NUMERIC
        elif variable.dtype.kind == 'O':
            vtype = xport.VariableType.CHARACTER
        elif variable.dtype.kind == 'b':
            # We'll encode Boolean columns as 1 if True else 0.
            vtype = xport.VariableType.NUMERIC
        else:
            raise TypeError(f'{type(variable).__name__}.dtype {variable.dtype} not supported')

        if variable.width is not None:
            length = variable.width
        elif vtype == xport.VariableType.NUMERIC:
            length = 8
        else:
            # TODO: Avoid encoding twice, once here and once in ``Observations``.
            length = variable.str.encode(TEXT_DATA_ENCODING).str.len().max()
        try:
            length = max(1, length)  # We need at least 1 byte per value.
        except TypeError:
            length = 1
        return cls(
            vtype=vtype,
            length=length,
            number=number,
            name=variable.name,
            label=variable.label,
            format=variable.format if variable.format is not None else xport.Format(),
            informat=variable.informat if variable.informat is not None else xport.Informat(),
            position=position,
        )

    @classmethod
    def from_bytes(cls, bytestring: bytes):
        """
        Construct a ``Namestr`` from an XPORT-format byte string.
        """
        LOG.debug(f'Decode {type(cls).__name__}')
        # dtype='float' if vtype == xport.VariableType.NUMERIC else 'string'
        size = len(bytestring)
        if size == 136:
            warnings.warn('File written on VAX/VMS, module behavior not tested')
        fmt = cls.fmts[size]
        tokens = struct.unpack(fmt, bytestring)
        return cls(
            vtype=xport.VariableType(tokens[0]),
            length=tokens[2],
            number=tokens[3],
            name=tokens[4].strip(b'\x00').decode(TEXT_METADATA_ENCODING).rstrip(),
            label=tokens[5].strip(b'\x00').decode(TEXT_METADATA_ENCODING).rstrip(),
            format=xport.Format.from_struct_tokens(*tokens[6:10]),
            informat=xport.Informat.from_struct_tokens(*tokens[11:14]),
            position=tokens[14],
        )

    def __bytes__(self):
        """
        Encode in XPORT-format.
        """
        LOG.debug(f'Encode {type(self).__name__}')
        fmt = self.fmts[140]
        format_name = self.format.name.encode('ascii')
        if len(format_name) > 8:
            raise ValueError(f'ASCII-encoded format name {format_name} longer than 8 characters')
        informat_name = self.informat.name.encode('ascii')
        if len(informat_name) > 8:
            raise ValueError(f'ASCII-encoded format name {informat_name} longer than 8 characters')
        if self.number is None:
            raise ValueError('Variable number not assigned')
        if self.position is None:
            raise ValueError('Variable position not assigned')
        return struct.pack(
            fmt,
            self.vtype,
            0,  # "Hash" of name, always 0.
            self.length,
            self.number,
            text_encode(self, 'name', 8),
            text_encode(self, 'label', 40),
            format_name.ljust(8),
            self.format.length,
            self.format.decimals,
            self.format.justify,
            b'',  # Unused
            informat_name.ljust(8),
            self.informat.length,
            self.informat.decimals,
            self.position,
            b'',  # Padding
        )


class MemberHeader(Mapping):
    """
    Dataset metadata from a SAS Version 5 or 6 Transport (XPORT) file.
    """

    # 4. Member header records
    #    Both of these records occur for every member in the file.
    #
    #    HEADER RECORD*******MEMBER HEADER RECORD!!!!!!!
    #    000000000000000001600000000140
    #    HEADER RECORD*******DSCRPTR HEADER RECORD!!!!!!!
    #    000000000000000000000000000000
    #
    #    Note the 0140 that appears in the member header record above.
    #    This value specifies the size of the variable descriptor
    #    (NAMESTR) record that is described later in this document. On
    #    the VAX/VMS operating system, the value will be 0136 instead of
    #    0140. This means that the descriptor will be only 136 bytes
    #    instead of 140.
    #
    # 5. Member header data ... as C structure:
    #
    #       struct REAL_HEADER {
    #          char sas_symbol[8];      /* "SAS"                    */
    #          char sas_dsname[8];      /* dataset name             */
    #          char sasdata[8];         /* "SASDATA"                */
    #          char sasver[8];          /* version of SAS used      */
    #          char sas_osname[8];      /* operating system used    */
    #          char blanks[24];
    #          char sas_create[16];     /* datetime created         */
    #          };
    #
    #    The second header record as C structure:
    #
    #       struct SECOND_HEADER {
    #          char dtmod[16];            /* date modified           */
    #          char padding[16];
    #          char dslabel[40];          /* dataset label          */
    #          char dstype[8]             /* dataset type           */
    #          };
    #
    # 6. Namestr header record
    #    One for each member
    #
    #       HEADER RECORD*******NAMESTR HEADER RECORD!!!!!!!
    #       000000xxxx0000000000000 0000000
    #
    #    In this header record, xxxx is the number of variables in the
    #    data set, displayed with blank-padded numeric characters. For
    #    example, for 2 variables, xxxx=0002.
    #
    # 7. Namestr records
    #    Each namestr field is 140 bytes long, but the fields are
    #    streamed together and broken in 80-byte pieces. If the last
    #    byte of the last namestr field does not fall in the last byte
    #    of the 80-byte record, the record is padded with ASCII blanks
    #    to 80 bytes.
    #
    # 8. Observation header
    #
    #       HEADER RECORD*******OBS HEADER RECORD!!!!!!!
    #       000000000000000000000000 000000

    # Literal opening of a member's first 80-byte header record.  Member
    # headers always begin on an 80-byte boundary, so this marker -- unlike
    # the full ``pattern`` -- can locate the next member even when the
    # current document cuts off inside that member's header.
    start_marker = b'HEADER RECORD*******MEMBER  HEADER RECORD'

    # The member header's first record is exactly one 80-byte record.  A
    # complete copy of this record proves a member boundary even when later
    # records of the same member (namestrs, OBS) are truncated.  Requiring
    # the full record -- including its numeric suffix -- makes accidental
    # matches inside observation data effectively impossible.
    first_record_pattern = re.compile(
        rb'HEADER RECORD\*{7}MEMBER  HEADER RECORD\!{7}0{17}160{8}(?:140|136)  '
    )

    # A truncated final record must share at least this much of the marker.
    # The prefix includes ``MEMBER`` so it cannot match the other record
    # types (NAMESTR, OBS, DSCRPTR, LIBRARY), which share only the shorter
    # ``HEADER RECORD*******`` opening.
    minimum_marker_prefix = len(b'HEADER RECORD*******MEMBER')

    @classmethod
    def next_member_start(cls, mview, search_from, end=None):
        """
        Find the 80-byte-aligned start of the next member header.

        Returns the offset, or ``None`` when no later member begins within
        the view.  Two cases are recognized.  A member header truncated by
        EOF can only begin in the final, incomplete 80-byte record, which is
        checked with the distinctive ``MEMBER`` marker prefix.  Every other
        boundary must be a complete first header record on an 80-byte
        boundary, so the marker appearing inside observation data is ignored.
        """
        if end is None:
            end = len(mview)
        marker = cls.start_marker

        # Case 1: the document ended inside the next member's header.  That
        # header can only start on the final, short 80-byte record.
        fragment_boundary = None
        if end % 80:
            aligned = (end // 80) * 80
            if aligned >= search_from:
                fragment = bytes(mview[aligned:end])
                shared = 0
                for a, b in zip(fragment, marker):
                    if a != b:
                        break
                    shared += 1
                if shared >= cls.minimum_marker_prefix:
                    fragment_boundary = aligned

        # Case 2: a complete member header.  Require a full first record on
        # an 80-byte boundary; observation text cannot reproduce that record.
        complete_boundary = None
        marker_pattern = re.compile(re.escape(marker))
        candidate = search_from
        while True:
            match = marker_pattern.search(mview, candidate, end)
            if match is None:
                break
            start = match.start()
            if start % 80 == 0 and start + 80 <= end:
                if cls.first_record_pattern.match(mview[start:start + 80]):
                    complete_boundary = start
                    break
            candidate = start + 1

        candidates = [b for b in (complete_boundary, fragment_boundary) if b is not None]
        return min(candidates) if candidates else None

    pattern = re.compile(
        # Header line 1
        rb'HEADER RECORD\*{7}MEMBER  HEADER RECORD\!{7}0{17}'
        rb'160{8}(?P<descriptor_size>140|136)  '
        # Header line 2
        rb'HEADER RECORD\*{7}DSCRPTR HEADER RECORD\!{7}0{30} {2}'
        # Header line 3
        rb'SAS {5}(?P<name>.{8})SASDATA '
        rb'(?P<version>.{8})(?P<os>.{8}) {24}(?P<created>.{16})'
        # Header line 4
        rb'(?P<modified>.{16}) {16}'
        rb'(?P<label>.{40})(?P<type>    DATA|    VIEW| {8})'
        # Namestrs
        rb'HEADER RECORD\*{7}NAMESTR HEADER RECORD\!{7}0{6}'
        rb'(?P<n_variables>.{4})0{20} {2}'
        rb'(?P<namestrs>.*?)'
        rb'HEADER RECORD\*{7}OBS {5}HEADER RECORD\!{7}0{30} {2}',
        # Observations ... until EOF or another Member.
        re.DOTALL,
    )

    def __init__(
        self,
        name,
        dataset_label,
        dataset_type,
        created,
        modified,
        sas_os,
        sas_version,
        namestrs=(),
    ):
        """
        Initialize a ``MemberHeader``.
        """
        self.name = name
        self.dataset_label = dataset_label
        self.dataset_type = dataset_type
        self.created = created
        self.modified = modified
        self.sas_os = sas_os
        self.sas_version = sas_version
        self.namestrs = {ns.name: ns for ns in namestrs}

    def __repr__(self):
        """
        Format for the REPL.
        """
        metadata = (name.strip('_') for name in xport.Dataset._metadata)
        metadata = {name: getattr(self, name) for name in metadata}
        metadata = (f'{k.title()}: {v}' for k, v in metadata.items() if v is not None)
        return f'<{type(self).__name__} {", ".join(metadata)}>'

    def __getitem__(self, name):
        """
        Get a namestr by variable name.
        """
        return self.namestrs[name]

    def __iter__(self):
        """
        Get an iterator of variable names.
        """
        return iter(self.namestrs)

    def __len__(self):
        """
        Get the number of variables/namestrs.
        """
        return len(self.namestrs)

    def __eq__(self, other):
        """
        Check equality.
        """
        if not isinstance(other, MemberHeader):
            raise TypeError(f"Can't compare {type(self).__name__} with {type(other).__name__}")
        metadata = (name.strip('_') for name in xport.Dataset._metadata)
        return super().__eq__(other) and all(
            getattr(self, name) == getattr(other, name) for name in metadata
        )

    @classmethod
    def from_dataset(cls, dataset: xport.Dataset, Namestr=Namestr, variable_enumeration_start=1):
        """
        Construct a ``MemberHeader`` from an ``xport.Dataset``.
        """
        namestrs = []
        p = 0
        for i, (k, v) in enumerate(dataset.items(), variable_enumeration_start):
            ns = Namestr.from_variable(v, number=i)
            ns.position = p
            p += ns.length
            namestrs.append(ns)
        return cls(
            name=dataset.name,
            dataset_label=dataset.dataset_label,
            dataset_type=dataset.dataset_type,
            created=dataset.created,
            modified=dataset.modified,
            sas_os=dataset.sas_os,
            sas_version=dataset.sas_version,
            namestrs=namestrs,
        )

    @classmethod
    def from_bytes(cls, bytestring: bytes, Namestr=Namestr):
        """
        Construct a ``MemberHeader`` from an XPORT-format byte string.
        """
        LOG.debug(f'Decode {type(cls).__name__}')
        mo = cls.pattern.search(bytestring)  # TODO: Why ``search``, not ``match``?
        if mo is None:
            raise ValueError('No member header found')

        def chunks():
            stride = int(mo['descriptor_size'])
            for i in range(0, len(mo['namestrs']), stride):
                chunk = mo['namestrs'][i:i + stride]
                if len(chunk) == stride:
                    yield chunk

        n = int(mo['n_variables'])
        namestrs = [Namestr.from_bytes(b) for b in chunks()]
        if len(namestrs) != n:
            raise ValueError(f'Expected {n}, got {len(namestrs)}')
        self = cls(
            name=mo['name'].strip(b'\x00').decode(TEXT_METADATA_ENCODING).strip(),
            dataset_label=mo['label'].strip(b'\x00').decode(TEXT_METADATA_ENCODING).strip(),
            dataset_type=mo['type'].strip(b'\x00').decode(TEXT_METADATA_ENCODING).strip(),
            sas_os=mo['os'].strip(b'\x00').decode(TEXT_METADATA_ENCODING).strip(),
            sas_version=mo['version'].strip().decode(TEXT_METADATA_ENCODING),
            created=strptime(mo['created']),
            modified=strptime(mo['modified']),
            namestrs=namestrs,
        )
        return self

    template = f'''\
HEADER RECORD{'*' * 7}MEMBER  HEADER RECORD{'!' * 7}{'0' * 17}16{'0' * 8}140  \
HEADER RECORD{'*' * 7}DSCRPTR HEADER RECORD{'!' * 7}{'0' * 30}  \
SAS     %(name)8bSASDATA %(version)8b%(os)8b{' ' * 24}%(created)16b\
%(modified)16b{' ' * 16}%(label)40b%(type)8b\
HEADER RECORD{'*' * 7}NAMESTR HEADER RECORD{'!' * 7}{'0' * 6}\
%(n_variables)04d{'0' * 20}  \
%(namestrs)b\
HEADER RECORD{'*' * 7}OBS     HEADER RECORD{'!' * 7}{'0' * 30}  \
'''.encode('ascii')

    def __bytes__(self):
        """
        Encode in XPORT format.
        """
        LOG.debug(f'Encode {type(self).__name__}')
        namestrs = b''.join(bytes(ns) for ns in self.values())
        if len(namestrs) % 80:
            namestrs += b' ' * (80 - len(namestrs) % 80)
        return self.template % {
            b'name': text_encode(self, 'name', 8),
            b'label': text_encode(self, 'dataset_label', 40),
            b'type': text_encode(self, 'dataset_type', 8),
            b'n_variables': len(self),
            b'os': text_encode(self, 'sas_os', 8),
            b'version': text_encode(self, 'sas_version', 8),
            b'created': strftime(self.created if self.created else datetime.now()),
            b'modified': strftime(self.modified if self.modified else datetime.now()),
            b'namestrs': namestrs,
        }


class ParseStats:
    """
    Bookkeeping for one member's observation block.

    ``confirmed_end_offset`` is the byte offset, relative to the start of
    the observation block, of the first byte not consumed by a confirmed
    full observation.  ``trailing_bytes`` counts the undetermined bytes at
    the end of the block; zero means the block ended cleanly on padding or
    at a member boundary.
    """

    def __init__(self, n_observations=0, confirmed_end_offset=0, trailing_bytes=0):
        """
        Initialize empty parse statistics.
        """
        self.n_observations = n_observations
        self.confirmed_end_offset = confirmed_end_offset
        self.trailing_bytes = trailing_bytes

    def __repr__(self):
        """
        REPL-format string.
        """
        return (
            f'ParseStats(n_observations={self.n_observations}, '
            f'confirmed_end_offset={self.confirmed_end_offset}, '
            f'trailing_bytes={self.trailing_bytes})'
        )


class Observations(Iterator):
    """
    Data from a SAS Version 5 or 6 Transport (XPORT) file.

    ``Observations`` is an iterator, yielding observations as tuples.
    """

    # 9. Data records
    #    Data records are streamed in the same way that namestrs are.
    #    There is ASCII blank padding at the end of the last record if
    #    necessary. There is no special trailing record.

    def __init__(self, observations, header=None, stats=None):
        """
        Initialize from an iterable of observations.
        """
        self.it = iter(observations)
        self.header = header
        self.stats = stats if stats is not None else ParseStats()

    def __next__(self):
        """
        Get the next item from the iterator.
        """
        return next(self.it)

    @classmethod
    def from_dataset(cls, dataset):
        """
        Yield observations from an ``xport.Dataset``.
        """
        return cls(
            observations=dataset.itertuples(index=False, name='Observation'),
            header=MemberHeader.from_dataset(dataset),
        )

    @classmethod
    def from_bytes(cls, bytestring, header):
        """
        Yield observations from an XPORT-format byte string.
        """
        LOG.debug(f'Decode {type(cls).__name__}')

        def character_decode(s):
            return s.decode(TEXT_DATA_ENCODING).rstrip()

        converters = []
        for namestr in header.values():
            if namestr.vtype == xport.VariableType.NUMERIC:
                converters.append(ibm_to_ieee)
            else:
                converters.append(character_decode)

        def iterator():
            sizes = [namestr.length for namestr in header.values()]
            fmt = ''.join(f'{x}s' for x in sizes)
            stride = sum(sizes)
            if stride == 0:
                stats.n_observations = 0
                stats.confirmed_end_offset = 0
                stats.trailing_bytes = 0
                return
            # TODO: The SAS Transport v5 specification says the sentinel
            #       character is b' ', but people report b'\x00' is used
            #       in some files.  Unfortunately, that would make rows
            #       with all zeros indistiguishable from the sentinel.
            sentinel = b' ' * stride
            mview = memoryview(bytestring)
            n_observations = 0
            offset = 0
            while offset < len(mview):
                chunk = mview[offset:offset + stride]
                if len(chunk) != stride:
                    break  # Partial observation: handled after the loop.
                if chunk == sentinel:
                    break  # Blank padding marks the end of observations.
                # TODO: If only characters and the last row is all empty
                #       or spaces, it's indistinguishable from padding.
                #       https://github.com/selik/xport/issues/46
                tokens = struct.unpack(fmt, chunk)
                yield tuple(f(v) for f, v in zip(converters, tokens))
                n_observations += 1
                offset += stride
            # Everything after the last full observation must be blank
            # ASCII padding that fills the final 80-byte record.  Any
            # non-blank byte is an undetermined, potentially truncated tail.
            remainder = bytes(mview[offset:])
            if remainder.strip(b' '):
                stats.trailing_bytes = len(remainder)
            stats.n_observations = n_observations
            stats.confirmed_end_offset = offset

        stats = ParseStats()
        return cls(iterator(), header, stats=stats)

    def to_bytes(self):
        """
        Get an iterator of XPORT-encoded observations.
        """

        def character_encoder(length):

            def encoder(s):
                try:
                    return s.encode(TEXT_DATA_ENCODING).ljust(length)
                except AttributeError:
                    return b' '
                # If handling errors from None, NAType, etc. is a
                # bottleneck, we should ``fillna`` before creating the
                # values iterator.

            return encoder

        fmt = ''.join(f'{namestr.length}s' for namestr in self.header.values())
        converters = []
        for namestr in self.header.values():
            if namestr.vtype == xport.VariableType.NUMERIC:
                converters.append(ieee_to_ibm)
            else:
                converters.append(character_encoder(namestr.length))
        for t in self:
            g = (f(v) for f, v in zip(converters, t))
            yield struct.pack(fmt, *g)

    def __bytes__(self):
        """
        Encode in XPORT-format.
        """
        LOG.debug(f'Encode {type(self).__name__}')
        observations = b''.join(self.to_bytes())
        if len(observations) % 80:
            observations += b' ' * (80 - len(observations) % 80)
        return observations


class Member(xport.Dataset):
    """
    Dataset from a SAS Version 5 or 6 Transport (XPORT) file.
    """

    @classmethod
    def from_header(cls, header):
        """
        Create an empty ``Member`` with metadata from a ``MemberHeader``.
        """
        variables = {
            namestr.name: xport.Variable(
                dtype='float' if namestr.vtype == xport.VariableType.NUMERIC else 'string',
                name=namestr.name,
                label=namestr.label,
                vtype=namestr.vtype,
                width=namestr.length,
                format=namestr.format,
                informat=namestr.informat,
            )
            for namestr in header.values()
        }
        public = (name.lstrip('_') for name in cls._metadata)
        self = cls(variables, **{name: getattr(header, name) for name in public})
        return self

    @classmethod
    def from_bytes(cls, bytestring, MemberHeader=MemberHeader, Observations=Observations):
        """
        Decode the first ``Member`` from an XPORT-format byte string.

        The member's :class:`ParseStats` are attached as the private
        ``_parse_stats`` attribute so the library parser can tell a complete
        member boundary from an undetermined truncated tail.
        """
        LOG.debug(f'Decode {type(cls).__name__}')
        mview = memoryview(bytestring)
        header_match = MemberHeader.pattern.match(mview)
        if header_match is None:
            raise ValueError('No member header found')
        i = header_match.end(0)

        # When decoding a standalone document, bound this member's
        # observations at the 80-byte-aligned start of the following member.
        # The library parser already passes an exact single-member region.
        boundary = MemberHeader.next_member_start(mview, i)

        header = MemberHeader.from_bytes(mview[:i])
        observations = Observations.from_bytes(mview[i:boundary], header)

        # This awkwardness works around Pandas subclasses misbehaving.
        # ``DataFrame.append`` discards subclass attributes.  Lame.
        head = cls.from_header(header)
        data = Member(pd.DataFrame.from_records(observations, columns=list(header)))
        data.copy_metadata(head)
        # ``from_records`` exhausts the iterator, which fills in the stats.
        object.__setattr__(data, '_parse_stats', observations.stats)
        object.__setattr__(data, '_member_header', header)
        LOG.info(f'Decoded XPORT dataset {data.name!r}')
        LOG.debug('%s', data)
        return data

    def __bytes__(self):
        """
        Encode in XPORT-format.
        """
        return self._bytes()

    def _bytes(self, MemberHeader=MemberHeader, Observations=Observations):
        LOG.debug(f'Encode {type(self).__name__}')
        dtype_kind_conversions = {
            'O': 'string',
            'b': 'float',
            'i': 'float',
        }
        dtypes = self.dtypes.to_dict()
        conversions = {}
        for column, dtype in dtypes.items():
            try:
                conversions[column] = dtype_kind_conversions[dtype.kind]
            except KeyError:
                continue
        if conversions:
            warnings.warn(f'Converting column dtypes {conversions}')
            # BUG: ``DataFrame.copy`` mutates and discards ``Variable`` metadata.
            # self = self.copy()  # Don't mutate!
            cpy = xport.Dataset({k: v for k, v in self.items()})
            for column, dtype in conversions.items():
                LOG.warning(f'Converting column {column!r} from {dtypes[column]} to {dtype}')
                try:
                    cpy[column] = cpy[column].astype(dtype)
                except Exception:
                    raise TypeError(f'Could not coerce column {column!r} to {dtype}')
            cpy.copy_metadata(self)
            self = cpy
        header = bytes(MemberHeader.from_dataset(self))
        observations = bytes(Observations.from_dataset(self))
        return header + observations


class Library(xport.Library):
    """
    Collection of datasets from a SAS Version 5 or 6 Transport file.
    """

    # 1. The first header record:
    #
    #   HEADER RECORD*******LIBRARY HEADER RECORD!!!!!!!
    #   000000000000000000000000000000
    #
    # 2. The first real header record ... as a C structure:
    #
    #   struct REAL_HEADER {
    #      char sas_symbol[2][8];       /* "SAS", twice             */
    #      char saslib[8];              /* "SASLIB"                 */
    #      char sasver[8];              /* version of SAS used      */
    #      char sas_os[8];              /* operating system used    */
    #      char blanks[24];
    #      char sas_create[16];         /* datetime created         */
    #      };
    #
    # 3. Second real header record
    #
    #       ddMMMyy:hh:mm:ss
    #
    #    In this record, the string is the datetime modified. Most
    #    often, the datetime created and datetime modified will always
    #    be the same. Pad with ASCII blanks to 80 bytes. Note that only
    #    a 2-digit year appears. If any program needs to read in this
    #    2-digit year, be prepared to deal with dates in the 1900s or
    #    the 2000s.

    pattern = re.compile(
        rb'HEADER RECORD\*{7}LIBRARY HEADER RECORD\!{7}0{30} {2}'
        rb'SAS {5}SAS {5}SASLIB {2}'
        rb'(?P<version>.{8})(?P<os>.{8}) {24}(?P<created>.{16})'
        rb'(?P<modified>.{16}) {64}'
        rb'(?P<members>.*)',
        re.DOTALL,
    )

    member_header_class = MemberHeader
    member_class = Member

    @classmethod
    def parse_partial(
        cls,
        bytestring,
        MemberHeader=None,
        Member=None,
        library=None,
        start=None,
    ):
        """
        Confirm member boundaries one at a time.

        Members are admitted to the returned :class:`PartialLoad` only when
        their whole header, every namestr, and an integer number of
        observations parse cleanly.  A member whose header is incomplete or
        whose observation block has an undetermined trailing fragment is
        kept out of the confirmed library and reported via the result's
        ``tail`` instead of marking the whole document a success.

        When resuming, pass the already-confirmed ``library`` and a ``start``
        offset at the first unconfirmed member so confirmed members are not
        registered a second time.
        """
        if MemberHeader is None:
            MemberHeader = cls.member_header_class
        if Member is None:
            Member = cls.member_class
        LOG.debug(f'Decoding {cls.__name__} from {len(bytestring)} bytes')
        mview = memoryview(bytestring)
        mo = cls.pattern.match(mview)
        if mo is None:
            lines = [mview[i * 80:(i + 1) * 80].tobytes() for i in range(8)]
            LOG.error('Document begins with' + '\n%s' * len(lines), *lines)
            raise ValueError(
                'Document does not match SAS Version 5 or 6 Transport (XPORT) format'
            )

        if library is None:
            library = cls(
                members=(),
                created=strptime(mo['created']),
                modified=strptime(mo['modified']),
                sas_os=mo['os'].strip(b'\x00').decode(TEXT_METADATA_ENCODING).strip(),
                sas_version=mo['version'].strip(b'\x00').decode(TEXT_METADATA_ENCODING).strip(),
            )
        receipts = dict(getattr(library, 'receipts', {}))

        pos = mo.start('members') if start is None else start
        end = len(mview)
        while pos < end:
            header_match = MemberHeader.pattern.match(mview, pos, end)
            if header_match is None:
                tail = bytes(mview[pos:])
                # Blank 80-byte record padding after the last member is not
                # a new observation or member; only accept it as clean end.
                if tail and not tail.strip(b' '):
                    pos = end
                    break
                return PartialLoad(
                    library=library,
                    receipts=receipts,
                    tail=tail,
                    resume_offset=pos,
                    prefix=bytes(mview[:pos]),
                    incomplete_member_start=pos,
                    reason='incomplete member header',
                )

            member_start = pos
            obs_start = header_match.end(0)
            # Locate the next member by its 80-byte-aligned start marker,
            # not by the full header pattern, so a truncated later header
            # still bounds this member's observations correctly.
            next_start = MemberHeader.next_member_start(mview, obs_start, end)
            is_last_member = next_start is None
            boundary = next_start if next_start is not None else end

            member = Member.from_bytes(mview[member_start:boundary])
            stats = object.__getattribute__(member, '_parse_stats')
            # A valid member's observation block (observations plus blank
            # record padding) always fills whole 80-byte records.  A final
            # member whose block is not 80-aligned was cut mid-record, even
            # if the cut happens to leave no non-blank trailing byte.
            misaligned = (boundary - obs_start) % 80 != 0
            if stats.trailing_bytes or (is_last_member and misaligned):
                confirmed_end = obs_start + stats.confirmed_end_offset
                return PartialLoad(
                    library=library,
                    receipts=receipts,
                    tail=bytes(mview[confirmed_end:]),
                    resume_offset=confirmed_end,
                    prefix=bytes(mview[:confirmed_end]),
                    incomplete_member=member,
                    incomplete_header=object.__getattribute__(member, '_member_header'),
                    incomplete_member_start=member_start,
                    confirmed_observations=stats.n_observations,
                    is_complete=False,
                    reason=(
                        f'{stats.trailing_bytes} trailing byte(s) do not form '
                        f'a complete observation'
                        if stats.trailing_bytes
                        else 'observation block ends before an 80-byte record boundary'
                    ),
                )

            receipt = xport.MemberReceipt(
                name=member.name,
                start_offset=member_start,
                end_offset=boundary,
                n_observations=stats.n_observations,
                n_variables=len(object.__getattribute__(member, '_member_header')),
            )
            if member.name not in library:
                library[member.name] = member
            library.confirm(receipt)
            receipts[member.name] = receipt
            pos = boundary

        return PartialLoad(
            library=library,
            receipts=receipts,
            tail=b'',
            resume_offset=pos,
            prefix=bytes(mview[:pos]),
        )

    @classmethod
    def from_bytes(cls, bytestring, MemberHeader=None, Member=None):
        """
        Parse a complete SAS XPORT document from a byte string.

        Raises :class:`TruncatedDocumentError` when the document ends inside
        a member, rather than silently dropping the incomplete member.  Use
        :func:`load_partial` to receive the confirmed members and tail.
        """
        partial = cls.parse_partial(
            bytestring,
            MemberHeader=MemberHeader,
            Member=Member,
        )
        if not partial.complete:
            raise TruncatedDocumentError(partial)
        LOG.info(f'Decoded {partial.library}')
        return partial.library

    template = f'''\
HEADER RECORD{'*' * 7}LIBRARY HEADER RECORD{'!' * 7}{'0' * 30}  \
SAS     SAS     SASLIB  \
%(version)8b%(os)8b{' ' * 24}%(created)16b\
%(modified)16b{' ' * 64}\
%(members)b\
'''.encode('ascii')

    def __bytes__(self):
        """
        XPORT-format bytes string.
        """
        return self._bytes()

    def _bytes(self, Member=Member):
        return self.template % {
            b'version': text_encode(self, 'sas_version', 8),
            b'os': text_encode(self, 'sas_os', 8),
            b'created': strftime(self.created if self.created else datetime.now()),
            b'modified': strftime(self.modified if self.modified else datetime.now()),
            b'members': b''.join(bytes(Member(member)) for member in self.values()),
        }


def text_encode(obj, name, n):
    """
    Encode and check resulting byte string length.
    """
    # The attribute name is a parameter to provide a useful error message.
    value = getattr(obj, name)
    if value is None:
        value = ''
    bytestring = value.encode(TEXT_METADATA_ENCODING).ljust(n)
    if len(bytestring) > n:
        raise ValueError(f'Encoded {name} {bytestring} longer than {n} characters')
    return bytestring


def strptime(timestring):
    """
    Parse a datetime from an XPT format string.

    All text in an XPT document are ASCII-encoded.  This function
    expects a bytes string in the "ddMMMyy:hh:mm:ss" format.  For
    example, ``b'16FEB11:10:07:55'``.  Note that XPT supports only
    2-digit years, which are expected to be either 1900s or 2000s.
    """
    text = timestring.decode('ascii')
    return datetime.strptime(text, '%d%b%y:%H:%M:%S')


def strftime(dt, minimum=datetime(1900, 1, 1), maximum=datetime(2100, 1, 1)):
    """
    Convert a datetime to an XPT format byte string.
    """
    if dt < minimum:
        raise ValueError(f'y2k never left! {dt} is in year {dt:%y}')
    if dt >= maximum:
        raise ValueError(f'y2k never left! {dt} is in year {dt:%y}')
    return dt.strftime('%d%b%y:%H:%M:%S').upper().encode('ascii')


def ibm_to_ieee(ibm: bytes) -> float:
    """
    Convert IBM-format floating point (bytes) to IEEE 754 64-bit (float).
    """
    # IBM mainframe:    sign * 0.mantissa * 16 ** (exponent - 64)
    # Python uses IEEE: sign * 1.mantissa * 2 ** (exponent - 1023)

    # Pad-out to 8 bytes if necessary. We expect 2 to 8 bytes, but
    # there's no need to check; bizarre sizes will cause a struct
    # module unpack error.
    ibm = ibm.ljust(8, b'\x00')

    # parse the 64 bits of IBM float as one 8-byte unsigned long long
    ulong, = struct.unpack('>Q', ibm)

    # IBM: 1-bit sign, 7-bits exponent, 56-bits mantissa
    sign = ulong & 0x8000000000000000
    exponent = (ulong & 0x7f00000000000000) >> 56
    mantissa = ulong & 0x00ffffffffffffff

    if mantissa == 0:
        if ibm[:1] == b'\x00':
            return 0.0
        elif ibm[:1] == b'\x80':
            return -0.0
        elif ibm[:1] == b'.':
            return float('nan')
        elif ibm[:1] in b'_ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            return getattr(xport.NaN, chr(ibm[0]))
        else:
            raise ValueError('Neither "true" zero nor NaN: %r' % ibm)

    # IBM-format exponent is base 16, so the mantissa can have up to 3
    # leading zero-bits in the binary mantissa. IEEE format exponent
    # is base 2, so we don't need any leading zero-bits and will shift
    # accordingly. This is one of the criticisms of IBM-format, its
    # wobbling precision.
    if ulong & 0x0080000000000000:
        shift = 3
    elif ulong & 0x0040000000000000:
        shift = 2
    elif ulong & 0x0020000000000000:
        shift = 1
    else:
        shift = 0
    mantissa >>= shift

    # clear the 1 bit to the left of the binary point
    # this is implicit in IEEE specification
    mantissa &= 0xffefffffffffffff

    # IBM exponent is excess 64, but we subtract 65, because of the
    # implicit 1 left of the radix point for the IEEE mantissa
    exponent -= 65
    # IBM exponent is base 16, IEEE is base 2, so we multiply by 4
    exponent <<= 2
    # IEEE exponent is excess 1023, but we also increment for each
    # right-shift when aligning the mantissa's first 1-bit
    exponent += shift + 1023

    # IEEE: 1-bit sign, 11-bits exponent, 52-bits mantissa
    # We didn't shift the sign bit, so it's already in the right spot
    ieee = sign | (exponent << 52) | mantissa
    return struct.unpack(">d", struct.pack(">Q", ieee))[0]


def ieee_to_ibm(ieee):
    """
    Convert Python floating point numbers to IBM-format (bytes).
    """
    # Python uses IEEE: sign * 1.mantissa * 2 ** (exponent - 1023)
    # IBM mainframe:    sign * 0.mantissa * 16 ** (exponent - 64)

    if ieee == 0.0:
        return b'\x00' * 8

    # The IBM hexadecimal floating point (HFP) format represents the number
    # zero with all zero bits.  All zero bits is the "true zero" or normalized
    # form of zero.  Any values for the sign and exponent can be used if the
    # mantissa portion of the encoding is all zero bits, but an IBM machine
    # might lose precision when performing arithmetic with alternative zero
    # representations.  With that in mind, and because this format was not
    # defined with a mechanism for not-a-number (NaN) values, SAS uses
    # alternative zero encodings to represent NaN.  By default, a SAS missing
    # value is encoded with an ASCII-encoded period (".") as the first byte.

    if isinstance(ieee, xport.NaN):
        return bytes(ieee)
    if math.isnan(ieee):
        return b'.' + b'\x00' * 7
    if math.isinf(ieee):
        raise NotImplementedError('Cannot convert infinity')

    bits = struct.pack('>d', ieee)
    ulong, = struct.unpack('>Q', bits)

    sign = (ulong & (1 << 63)) >> 63  # 1-bit     sign
    exponent = ((ulong & (0x7ff << 52)) >> 52) - 1023  # 11-bits   exponent
    mantissa = ulong & 0x000fffffffffffff  # 52-bits   mantissa/significand

    if exponent > 248:
        msg = 'Cannot store magnitude more than ~ 16 ** 63 as IBM-format'
        raise Overflow(msg)
    if exponent < -260:
        msg = 'Cannot store magnitude less than ~ 16 ** -65 as IBM-format'
        raise Underflow(msg)

    # IEEE mantissa has an implicit 1 left of the radix:    1.significand
    # IBM mantissa has an implicit 0 left of the radix:     0.significand
    # We must bitwise-or the implicit 1.mmm into the mantissa
    # later we will increment the exponent to account for this change
    mantissa = 0x0010000000000000 | mantissa

    # IEEE exponents are for base 2:    mantissa * 2 ** exponent
    # IBM exponents are for base 16:    mantissa * 16 ** exponent
    # We must divide the exponent by 4, since 16 ** x == 2 ** (4 * x)
    quotient, remainder = divmod(exponent, 4)
    exponent = quotient

    # We don't want to lose information;
    # the remainder from the divided exponent adjusts the mantissa
    mantissa <<= remainder

    # Increment exponent, because of earlier adjustment to mantissa
    # this corresponds to the 1.mantissa vs 0.mantissa implicit bit
    exponent += 1

    # IBM exponents are excess 64
    exponent += 64

    # IBM has 1-bit sign, 7-bits exponent, and 56-bits mantissa.
    # We must shift the sign and exponent into their places.
    sign <<= 63
    exponent <<= 56

    # We lose some precision, but who said floats were perfect?
    return struct.pack('>Q', sign | exponent | mantissa)


def load(fp):
    """
    Deserialize a SAS dataset library from a SAS Transport v5 (XPT) file.

        >>> with open('test/data/example.xpt', 'rb') as f:
        ...     library = load(f)
    """
    try:
        bytestring = fp.read()
    except UnicodeDecodeError:
        raise TypeError(f'Expected a BufferedReader in bytes-mode, got {type(fp).__name__}')
    return loads(bytestring)


def loads(bytestring):
    """
    Deserialize a SAS dataset library from an XPORT-format string.

        >>> with open('test/data/example.xpt', 'rb') as f:
        ...     bytestring = f.read()
        >>> library = loads(bytestring)
    """
    return Library.from_bytes(bytestring)


def load_partial(source):
    """
    Read a possibly truncated document without marking it all successful.

    ``source`` is either a bytes string or a bytes-mode file-like object.
    Always returns a :class:`PartialLoad`: fully received members appear in
    ``partial.library`` with receipts, while the undetermined trailing bytes
    stay in ``partial.tail`` and are never turned into observations.  Feed a
    grown document to :func:`resume` to continue from the proven position.
    """
    if hasattr(source, 'read'):
        try:
            bytestring = source.read()
        except UnicodeDecodeError:
            raise TypeError(
                f'Expected a BufferedReader in bytes-mode, got {type(source).__name__}'
            )
    else:
        bytestring = source
    return Library.parse_partial(bytestring)


def resume(partial, document):
    """
    Continue parsing after more bytes were appended to a truncated document.

    ``document`` is the full, grown byte buffer (the previously confirmed
    prefix plus the newly received bytes).  Parsing restarts at the proven
    boundary of the first unconfirmed member, so already-received members
    are not registered again and blank padding is not read as a new
    observation.  Returns a fresh :class:`PartialLoad`.
    """
    start = partial.incomplete_member_start
    if start is None:
        start = partial.resume_offset
    library_cls = type(partial.library)
    return library_cls.parse_partial(
        document,
        library=partial.library,
        start=start,
    )


def dump(library, fp):
    """
    Serialize a SAS dataset library to a SAS Transport v5 (XPORT) file.

        >>> library = Library()
        >>> with open('test/data/doctest.xpt', 'wb') as f:
        ...     dump(library, f)

    The input ``library`` can be either an ``xport.Library``, an
    ``xport.Dataset`` collection, or a single ``pandas.DataFrame``.
    An ``xport.Dataset`` is preferable, because that can be assigned a
    name, which SAS expects.

        >>> ds = xport.Dataset(name='EMPTY')
        >>> with open('test/data/doctest.xpt', 'wb') as f:
        ...     dump(ds, f)

    """
    fp.write(dumps(library))


def dumps(library):
    """
    Serialize a SAS dataset library to a string in XPORT-format.

        >>> library = Library()
        >>> bytestring = dumps(library)

    The input ``library`` can be either an ``xport.Library``, an
    ``xport.Dataset`` collection, or a single ``pandas.DataFrame``.
    An ``xport.Dataset`` is preferable, because that can be assigned a
    name, which SAS expects.

        >>> ds = xport.Dataset(name='EMPTY')
        >>> bytestring = dumps(ds)

    """
    return bytes(Library(library))
