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


class Observations(Iterator):
    """
    Data from a SAS Version 5 or 6 Transport (XPORT) file.

    ``Observations`` is an iterator, yielding observations as tuples.
    """

    # 9. Data records
    #    Data records are streamed in the same way that namestrs are.
    #    There is ASCII blank padding at the end of the last record if
    #    necessary. There is no special trailing record.

    def __init__(self, observations, header=None):
        """
        Initialize from an iterable of observations.
        """
        self.it = iter(observations)
        self.header = header
        self.raw = {}

    def __next__(self):
        """
        Get the next item from the iterator.
        """
        return next(self.it)

    @classmethod
    def from_dataset(cls, dataset):
        """
        Yield observations from an ``xport.Dataset``.

        If the dataset still has its original encodings, those encodings are
        attached so that byte-for-byte lossless writes remain possible.  A
        value that has been edited no longer matches its encoding, so it is
        freshly converted from its numeric view.
        """
        self = cls(
            observations=dataset.itertuples(index=False, name='Observation'),
            header=MemberHeader.from_dataset(dataset),
        )
        raw = getattr(dataset, '_raw_columns', None)
        if raw:
            index = dataset.index
            self.raw = {
                k: list(v.reindex(index)) if hasattr(v, 'reindex') and
                getattr(v, 'index', None) is not None else list(v)
                for k, v in raw.items()
            }
        return self

    @classmethod
    def from_bytes(cls, bytestring, header):
        """
        Yield observations from an XPORT-format byte string.
        """
        LOG.debug(f'Decode {type(cls).__name__}')

        def character_decode(s):
            return s.decode(TEXT_DATA_ENCODING).rstrip()

        converters = []
        numeric_names = []
        for namestr in header.values():
            if namestr.vtype == xport.VariableType.NUMERIC:
                converters.append(ibm_to_ieee)
                numeric_names.append(namestr.name)
            else:
                converters.append(character_decode)

        raw_columns = {name: [] for name in numeric_names}

        def iterator():
            sizes = [namestr.length for namestr in header.values()]
            fmt = ''.join(f'{x}s' for x in sizes)
            stride = sum(sizes)
            if stride == 0:
                return
            # TODO: The SAS Transport v5 specification says the sentinel
            #       character is b' ', but people report b'\x00' is used
            #       in some files.  Unfortunately, that would make rows
            #       with all zeros indistiguishable from the sentinel.
            sentinel = b' ' * stride
            mview = memoryview(bytestring)
            for i in range(0, len(mview), stride):
                chunk = mview[i:i + stride]
                if len(chunk) != stride or chunk == sentinel:
                    return
                # TODO: If only characters and the last row is all empty
                #       or spaces, it's indistinguishable from padding.
                #       https://github.com/selik/xport/issues/46
                tokens = struct.unpack(fmt, chunk)
                for namestr, token in zip(header.values(), tokens):
                    if namestr.vtype == xport.VariableType.NUMERIC:
                        raw_columns[namestr.name].append(bytes(token))
                yield tuple(f(v) for f, v in zip(converters, tokens))

        self = cls(iterator(), header)
        self.raw = raw_columns
        return self

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
        numeric_names = []
        for namestr in self.header.values():
            if namestr.vtype == xport.VariableType.NUMERIC:
                numeric_names.append(namestr.name)
                converters.append(None)  # Replaced per row to use raw bytes.
            else:
                converters.append(character_encoder(namestr.length))
        raws = {
            name: list(values) for name, values in getattr(self, 'raw', {}).items()
        }
        for i, t in enumerate(self):
            g = []
            n = 0
            for namestr, f, v in zip(self.header.values(), converters, t):
                if namestr.vtype == xport.VariableType.NUMERIC:
                    name = numeric_names[n]
                    n += 1
                    raw = raws[name][i] if name in raws and i < len(raws[name]) else None
                    g.append(_encode_field(v, raw))
                else:
                    g.append(f(v))
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


def _is_missing_encoding(raw):
    """
    Whether IBM bytes encode a SAS missing value rather than a number.
    """
    raw = bytes(raw).ljust(8, b'\x00')
    return raw[0] in b'._ABCDEFGHIJKLMNOPQRSTUVWXYZ' and not any(raw[1:])


def _encode_field(value, raw=None):
    """
    Encode a numeric field, reusing ``raw`` only when it still matches.

    This is deliberately conservative: an edited value must not inherit an
    unrelated original encoding.  When the bytes match the value's current
    numeric view, using them preserves the exact representation even where
    IEEE-754 cannot express the IBM fraction.  Missing markers are reused
    when the value is still missing, so tags such as ``.A`` survive writing.
    """
    if raw is not None:
        raw = bytes(raw)
        try:
            current = ibm_to_ieee(raw)
            if _is_missing_encoding(raw):
                # Pandas stores these as plain NaN in a float column, which
                # erases the SAS tag.  Retain the source bytes whenever the
                # value is still missing.  An explicitly different special
                # missing value is re-encoded instead.
                if _same_missing(value, raw):
                    return raw
            elif float(current) == float(value):
                return raw
        except (ValueError, TypeError):
            pass
    return encode_value(value)


def _same_missing(value, raw):
    """
    Whether ``value`` can reuse the missing-value tag encoded by ``raw``.

    A plain float NaN cannot express which SAS tag it came from, so it reuses
    the original bytes.  A special missing value must match the stored tag.
    """
    try:
        missing = math.isnan(value)
    except TypeError:
        return False
    if not missing:
        return False
    if isinstance(value, xport.SpecialMissingValue):
        return raw[:1] == value.name.encode('ascii')
    return True


def _set_raw_columns(dataset, raw):
    """
    Attach original encodings to a dataset and to its numeric variables.

    Encodings are stored as index-aligned Series so that filtering the dataset
    does not silently pair a value with another row's bytes.
    """
    columns = {}
    for name, values in raw.items():
        values = list(values)
        if len(values) != len(dataset.index):
            raise ValueError(f'Raw encodings for {name!r} have a different length')
        columns[name] = pd.Series(values, index=dataset.index, dtype='object', name=name)
    object.__setattr__(dataset, '_raw_columns', columns)
    for name, series in columns.items():
        if name in dataset.columns:
            object.__setattr__(dataset[name], '_raw', series)


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
        """
        LOG.debug(f'Decode {type(cls).__name__}')
        mview = memoryview(bytestring)
        matches = MemberHeader.pattern.finditer(mview)

        try:
            mo = next(matches)
        except StopIteration:
            raise ValueError('No member header found')
        i = mo.end(0)

        try:
            mo = next(matches)
        except StopIteration:
            j = None
        else:
            j = mo.start(0)

        header = MemberHeader.from_bytes(mview[:i])
        observations = Observations.from_bytes(mview[i:j], header)

        # This awkwardness works around Pandas subclasses misbehaving.
        # ``DataFrame.append`` discards subclass attributes.  Lame.
        head = cls.from_header(header)
        data = Member(pd.DataFrame.from_records(observations, columns=list(header)))
        data.copy_metadata(head)
        if observations.raw:
            _set_raw_columns(data, observations.raw)
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

    @classmethod
    def from_bytes(cls, bytestring, MemberHeader=MemberHeader, Member=Member):
        """
        Parse a SAS XPORT document from a byte string.
        """
        LOG.debug(f'Decoding {cls.__name__} from {len(bytestring)} bytes')
        mview = memoryview(bytestring)
        mo = cls.pattern.match(mview)
        if mo is None:
            lines = [mview[i * 80:(i + 1) * 80].tobytes() for i in range(8)]
            LOG.error('Document begins with' + '\n%s' * len(lines), *lines)
            raise ValueError('Document does not match SAS Version 5 or 6 Transport (XPORT) format')

        matches = MemberHeader.pattern.finditer(mview)
        indices = [m.start(0) for m in matches] + [None]
        chunks = (mview[i:j] for i, j in zip(indices, indices[1:]))
        self = cls(
            members=map(Member.from_bytes, chunks),
            created=strptime(mo['created']),
            modified=strptime(mo['modified']),
            sas_os=mo['os'].strip(b'\x00').decode(TEXT_METADATA_ENCODING).strip(),
            sas_version=mo['version'].strip(b'\x00').decode(TEXT_METADATA_ENCODING).strip(),
        )
        LOG.info(f'Decoded {self}')
        return self

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


def ibm_to_ieee(ibm: bytes) -> xport.IBMFloat:
    """
    Convert IBM-format floating point (bytes) to IEEE 754 64-bit (float).

    The return value is an ``xport.IBMFloat``: a regular Python ``float``
    that also remembers the bytes it came from.  That makes it possible to
    audit a read result without pretending that numeric equality proves the
    original encoding can be reproduced.

    IBM-format values are ``sign * 0.mantissa * 16 ** (exponent - 64)``.
    The fraction has 56 bits, which is three more than an IEEE-754 double's
    significand, so some values necessarily round when viewed as a float.
    """
    # Pad-out to 8 bytes if necessary. We expect 2 to 8 bytes, but
    # there's no need to check; bizarre sizes will cause a struct
    # module unpack error.
    raw = bytes(ibm)
    width = len(raw)
    ibm = raw.ljust(8, b'\x00')

    # Parse the 64 bits of IBM float as one 8-byte unsigned long long.
    ulong, = struct.unpack('>Q', ibm)

    # IBM: 1-bit sign, 7-bits exponent, 56-bits mantissa
    sign = -1.0 if ulong & 0x8000000000000000 else 1.0
    exponent = (ulong & 0x7f00000000000000) >> 56
    mantissa = ulong & 0x00ffffffffffffff

    if mantissa == 0:
        first = ibm[:1]
        if first == b'.':
            # SAS's default missing value, distinct from special missing.
            return xport.IBMFloat(float('nan'), raw=raw, width=width)
        if first in b'_ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            # Keep the tag.  The enum's ``raw`` property returns the bytes.
            return getattr(xport.NaN, chr(first[0]))
        # IBM considers any encoding with a zero fraction to be a zero.
        # Zero is not normalized, so the exponent and sign may be unusual.
        return xport.IBMFloat(sign * 0.0, raw=raw, width=width)

    # Avoid reconstructing the value through IEEE bit manipulation: the IBM
    # fraction can have up to 56 significant bits and may be unnormalized.
    # ``math.ldexp`` correctly rounds the exact value to an IEEE double.
    value = math.ldexp(mantissa, 4 * (exponent - 64) - 56)
    return xport.IBMFloat(sign * value, raw=raw, width=width)


def encode_value(value, preserve_raw=False):
    """
    Encode one numeric value in IBM-format.

    Special missing values keep their tag.  If ``preserve_raw`` is true and
    ``value`` carries an original encoding that still matches its numeric
    view, that encoding is returned instead of a fresh conversion.
    """
    if isinstance(value, xport.SpecialMissingValue):
        return bytes(value)
    if isinstance(value, xport.IBMFloat) and preserve_raw and value.raw is not None:
        return value.encoded
    return ieee_to_ibm(float(value))


def ieee_to_ibm(ieee):
    """
    Convert Python floating point numbers to IBM-format (bytes).
    """
    # Python uses IEEE: sign * 1.mantissa * 2 ** (exponent - 1023)
    # IBM mainframe:    sign * 0.mantissa * 16 ** (exponent - 64)

    if isinstance(ieee, xport.SpecialMissingValue):
        return bytes(ieee)

    if ieee == 0.0:
        # IBM's true zero is all zero bits, but the sign bit is meaningful
        # when reading.  Preserve negative zero so that a numeric view of
        # -0.0 does not silently claim to be the same encoding as +0.0.
        if math.copysign(1.0, ieee) < 0:
            return b'\x80' + b'\x00' * 7
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

    # IBM's largest normalized value is 0.ffffffffffffff * 16 ** 63,
    # which is just under 2 ** 252.  IEEE exponents of 251 and below fit.
    if exponent > 251:
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
