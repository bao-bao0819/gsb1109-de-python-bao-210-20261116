#!/usr/bin/env python3
# -*- coding: latin-1 -*-
"""
Offline consistency checks for indexed gzip mzML files.

Three independent views of a run are compared:

1. the gzip container (members, deflate stream ends, trailers and the
   uncompressed byte range of every member and element),
2. the native mzML index (``indexList`` offsets) and the generalized seekable
   gzip header index written by :class:`pymzml.utils.GSGW.GSGW`,
3. what :class:`pymzml.run.Reader` returns during sequential iteration and
   during random access by native identifier.

All checks are offline.  Findings carry the affected index key, gzip boundary,
identity interpretation and array evidence.  Equal array lengths alone are
never treated as a match: values (or, for opaque encodings, the encoded bytes)
must agree.

Typical use::

    from pymzml.utils.verify import verify_indexed_gzip

    report = verify_indexed_gzip("run.mzML.idx.gz")
    print(report.format())
    raise SystemExit(0 if report.ok else 1)
"""

import base64
import hashlib
import struct
import xml.etree.ElementTree as ElementTree
import xml.parsers.expat
import zlib
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

from .. import regex_patterns

__all__ = [
    "VerificationFinding",
    "IndexEntry",
    "GzipMember",
    "ArraySnapshot",
    "ItemSnapshot",
    "QueryObservation",
    "VerificationReport",
    "IndexedGzipVerifier",
    "verify_indexed_gzip",
    "compare_indexed_gzip",
    "DEFAULT_ARRAY_NAMES",
]

DEFAULT_ARRAY_NAMES = ("m/z array", "intensity array", "time array")

_DEFLATE_WBITS = -zlib.MAX_WBITS


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VerificationFinding:
    """One consistency finding with locators for every layer."""

    code: str
    severity: str
    message: str
    index: Any = None
    identity: Optional[dict] = None
    boundary: Optional[dict] = None
    arrays: tuple = ()
    details: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "index": self.index,
            "identity": self.identity,
            "boundary": self.boundary,
            "arrays": list(self.arrays),
            "details": self.details,
        }

    def __str__(self):
        parts = ["[{0}]".format(self.code)]
        if self.index is not None:
            parts.append("index={0!r}".format(self.index))
        if self.boundary:
            if self.boundary.get("member") is not None:
                parts.append("member={0}".format(self.boundary["member"]))
            if self.boundary.get("start") is not None:
                parts.append(
                    "offset={0}:{1}".format(
                        self.boundary.get("start"), self.boundary.get("end")
                    )
                )
        return "{0} {1}".format(" ".join(parts), self.message)


@dataclass
class IndexEntry:
    """A native ``indexList`` entry or a GSGW header index entry."""

    key: Any
    id_ref: str
    offset: int
    ordinal: int
    index_name: str = "spectrum"
    source: str = "native"

    def to_dict(self):
        return {
            "key": self.key,
            "id_ref": self.id_ref,
            "offset": self.offset,
            "ordinal": self.ordinal,
            "index_name": self.index_name,
            "source": self.source,
        }


@dataclass
class GzipMember:
    """A single gzip member with compressed and uncompressed boundaries."""

    number: int
    header_start: int
    deflate_start: int
    deflate_end: int
    trailer_end: int
    uncompressed_start: int
    uncompressed_end: int
    crc32: int
    isize: int
    crc32_ok: bool
    isize_ok: bool

    @property
    def uncompressed_length(self):
        return self.uncompressed_end - self.uncompressed_start

    def to_dict(self):
        return {
            "member": self.number,
            "header_start": self.header_start,
            "deflate_start": self.deflate_start,
            "deflate_end": self.deflate_end,
            "trailer_end": self.trailer_end,
            "start": self.deflate_start,
            "end": self.trailer_end,
            "uncompressed_start": self.uncompressed_start,
            "uncompressed_end": self.uncompressed_end,
            "length": self.uncompressed_length,
            "crc32_ok": self.crc32_ok,
            "isize_ok": self.isize_ok,
        }


@dataclass
class ArraySnapshot:
    """Raw and decoded evidence for one ``binaryDataArray``."""

    name: str
    cv_names: tuple
    encoded_length: Optional[int]
    encoded_bytes: bytes
    decoded_bytes: bytes
    values: tuple
    default_array_length: Optional[int]
    member: Optional[int] = None

    @property
    def count(self):
        if self.values:
            return len(self.values)
        return len(self.decoded_bytes)

    @property
    def value_digest(self):
        if self.values:
            return _sha256(repr(self.values).encode("utf-8"))
        return _sha256(self.decoded_bytes)

    @property
    def encoded_digest(self):
        return _sha256(self.encoded_bytes)

    def to_dict(self):
        return {
            "name": self.name,
            "cv_names": list(self.cv_names),
            "encoded_length": self.encoded_length,
            "encoded_digest": self.encoded_digest,
            "value_count": len(self.values),
            "value_digest": self.value_digest,
            "default_array_length": self.default_array_length,
            "member": self.member,
        }


@dataclass
class ItemSnapshot:
    """Identity and payload of a spectrum or chromatogram element."""

    key: Any
    index_key: Any
    tag: str
    index_attribute: Optional[int]
    id_attribute: str
    trailing_number: Optional[int]
    id_tokens: dict
    default_array_length: Optional[int]
    arrays: dict
    ordinal: Optional[int] = None
    offset: Optional[int] = None
    end: Optional[int] = None
    member: Optional[int] = None
    array_members: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "key": self.key,
            "index_key": self.index_key,
            "tag": self.tag,
            "index": self.index_attribute,
            "id": self.id_attribute,
            "trailing_number": self.trailing_number,
            "id_tokens": self.id_tokens,
            "default_array_length": self.default_array_length,
            "ordinal": self.ordinal,
            "offset": self.offset,
            "end": self.end,
            "member": self.member,
            "array_members": self.array_members,
            "arrays": {name: a.to_dict() for name, a in self.arrays.items()},
        }


@dataclass
class QueryObservation:
    """What a Reader call actually returned (or raised)."""

    requested: Any
    mode: str
    resolved_index: Optional[int] = None
    resolved_id: Optional[str] = None
    trailing_number: Optional[int] = None
    id_tokens: dict = field(default_factory=dict)
    array_values: dict = field(default_factory=dict)
    array_lengths: dict = field(default_factory=dict)
    array_digests: dict = field(default_factory=dict)
    exception: Optional[str] = None
    exception_type: Optional[str] = None
    sequential_position: Optional[int] = None

    def to_dict(self):
        return {
            "requested": self.requested,
            "mode": self.mode,
            "resolved_index": self.resolved_index,
            "resolved_id": self.resolved_id,
            "trailing_number": self.trailing_number,
            "id_tokens": self.id_tokens,
            "array_lengths": self.array_lengths,
            "exception": self.exception,
            "exception_type": self.exception_type,
            "sequential_position": self.sequential_position,
        }


@dataclass
class VerificationReport:
    """Result of one verification run."""

    path: str
    findings: list = field(default_factory=list)
    index_entries: list = field(default_factory=list)
    header_entries: list = field(default_factory=list)
    items: dict = field(default_factory=dict)
    members: list = field(default_factory=list)
    queries: list = field(default_factory=list)
    summary: dict = field(default_factory=dict)

    @property
    def ok(self):
        return not any(f.severity == "error" for f in self.findings)

    @property
    def errors(self):
        return [f for f in self.findings if f.severity == "error"]

    @property
    def warnings(self):
        return [f for f in self.findings if f.severity == "warning"]

    def codes(self):
        return {f.code for f in self.findings}

    def for_index(self, index):
        """All findings that mention ``index`` as target or affected range."""
        result = []
        for finding in self.findings:
            if finding.index == index:
                result.append(finding)
                continue
            affected = finding.details.get("affected_range")
            if affected and index in affected:
                result.append(finding)
                continue
            if finding.details.get("target_index") == index:
                result.append(finding)
                continue
            actual = finding.details.get("actual_target")
            if actual and index in (actual.get("key"), actual.get("id")):
                result.append(finding)
        return result

    def to_dict(self):
        return {
            "path": self.path,
            "ok": self.ok,
            "summary": self.summary,
            "findings": [f.to_dict() for f in self.findings],
            "index": [e.to_dict() for e in self.index_entries],
            "header_index": [e.to_dict() for e in self.header_entries],
            "items": {str(k): v.to_dict() for k, v in self.items.items()},
            "members": [m.to_dict() for m in self.members],
            "queries": [q.to_dict() for q in self.queries],
        }

    def explain(self, include_warnings=True):
        """Group findings by index key so affected ranges are easy to scan."""
        groups = OrderedDict()
        for finding in self.findings:
            if finding.severity == "info":
                continue
            if finding.severity == "warning" and not include_warnings:
                continue
            groups.setdefault(finding.index, []).append(finding)
        lines = []
        for index, findings in groups.items():
            affected = set()
            for finding in findings:
                affected.update(finding.details.get("affected_range", []))
            header = "index {0!r}".format(index)
            if affected:
                numeric = sorted(
                    (value for value in affected if isinstance(value, int))
                )
                if numeric:
                    header += "  affected_range={0}..{1}".format(
                        numeric[0], numeric[-1]
                    )
                else:
                    header += "  affected_range={0}".format(
                        sorted(affected, key=str)
                    )
            lines.append(header)
            for finding in findings:
                lines.append("  " + str(finding))
        return "\n".join(lines)

    def format(self, include_info=False):
        """Render the report as compact text."""
        lines = [
            "verification of {0}".format(self.path),
            "  status: {0}".format("OK" if self.ok else "FAILED"),
            "  findings: {0} error(s), {1} warning(s)".format(
                len(self.errors), len(self.warnings)
            ),
            "  summary: {0}".format(
                ", ".join(
                    "{0}={1}".format(k, v)
                    for k, v in self.summary.items()
                    if not isinstance(v, (dict, list))
                )
            ),
        ]
        for finding in self.findings:
            if not include_info and finding.severity == "info":
                continue
            lines.append("  " + str(finding))
            for label, payload in (
                ("identity", finding.identity),
                ("boundary", finding.boundary),
                ("arrays", list(finding.arrays) or None),
            ):
                if payload:
                    lines.append("    {0}: {1}".format(label, payload))
            for key, value in finding.details.items():
                if key in ("target_index",):
                    continue
                lines.append("    {0}: {1}".format(key, value))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _local_name(tag):
    return tag.rsplit("}", 1)[-1]


def _text(element):
    return (element.text or "").strip()


def _trailing_number(id_ref):
    """The integer ``Spectrum.ID`` derives from an ``id`` attribute."""
    if id_ref is None:
        return None
    match = regex_patterns.SPECTRUM_ID_PATTERN.search(id_ref)
    if match is None or match.group(1) == "":
        return None
    return int(match.group(1))


def _index_key(id_ref, tag="spectrum"):
    """Mirror pymzml's native index key derivation.

    ``StandardMzml`` prefers ``scan=<n>``/``scanId=<n>``; otherwise the whole
    ``idRef`` is used verbatim.  Chromatograms are always string keyed.
    """
    if id_ref is None:
        return None
    if tag == "chromatogram":
        return id_ref
    match = regex_patterns.SPECTRUM_ID_PATTERN2.search(id_ref)
    if match is not None:
        return int(match.group(2))
    return id_ref


def _id_tokens(id_ref):
    tokens = OrderedDict()
    if not id_ref:
        return tokens
    for chunk in id_ref.split():
        if "=" in chunk:
            key, value = chunk.split("=", 1)
            tokens[key] = value
    return tokens


def _read_member_header(blob, position):
    if blob[position : position + 2] != b"\x1f\x8b":
        raise ValueError(
            "expected gzip member at byte {0}, found {1!r}".format(
                position, blob[position : position + 2]
            )
        )
    if position + 10 > len(blob):
        raise ValueError("truncated gzip header at byte {0}".format(position))
    flags = blob[position + 3]
    cursor = position + 10
    if flags & 4:
        if cursor + 2 > len(blob):
            raise ValueError("truncated FEXTRA at byte {0}".format(cursor))
        xlen = struct.unpack("<H", blob[cursor : cursor + 2])[0]
        cursor += 2 + xlen
    for bit, label in ((8, "FNAME"), (16, "FCOMMENT")):
        if flags & bit:
            end = blob.find(b"\x00", cursor)
            if end < 0:
                raise ValueError(
                    "unterminated {0} at byte {1}".format(label, cursor)
                )
            cursor = end + 1
    if flags & 2:
        cursor += 2
    if cursor > len(blob):
        raise ValueError("truncated gzip header at byte {0}".format(position))
    return cursor


def scan_gzip_members(blob):
    """Scan every gzip member and validate its deflate end and trailer."""
    members = []
    problems = []
    position = 0
    uncompressed = 0
    number = 0
    while position < len(blob):
        try:
            deflate_start = _read_member_header(blob, position)
            decompressor = zlib.decompressobj(_DEFLATE_WBITS)
            payload = decompressor.decompress(blob[deflate_start:])
            if not decompressor.eof:
                problems.append(
                    {
                        "member": number,
                        "start": deflate_start,
                        "problem": "deflate stream ends without an end marker",
                    }
                )
                break
            consumed = len(blob) - deflate_start - len(decompressor.unused_data)
            deflate_end = deflate_start + consumed
            trailer_end = deflate_end + 8
            if trailer_end > len(blob):
                problems.append(
                    {
                        "member": number,
                        "start": deflate_start,
                        "problem": "gzip trailer is truncated",
                    }
                )
                break
            crc32, isize = struct.unpack("<II", blob[deflate_end:trailer_end])
            members.append(
                GzipMember(
                    number=number,
                    header_start=position,
                    deflate_start=deflate_start,
                    deflate_end=deflate_end,
                    trailer_end=trailer_end,
                    uncompressed_start=uncompressed,
                    uncompressed_end=uncompressed + len(payload),
                    crc32=crc32,
                    isize=isize,
                    crc32_ok=crc32 == (zlib.crc32(payload) & 0xFFFFFFFF),
                    isize_ok=isize == (len(payload) & 0xFFFFFFFF),
                )
            )
            uncompressed += len(payload)
            position = trailer_end
            number += 1
        except (ValueError, zlib.error) as exc:
            problems.append(
                {"member": number, "start": position, "problem": str(exc)}
            )
            break
    return members, problems


def _extract_spans(document):
    """Expat pass returning element, binaryDataArray and byte spans."""
    spectra = []
    stack = []
    current = None
    parser = xml.parsers.expat.ParserCreate()
    parser.buffer_text = True

    def start(name, attrs):
        nonlocal current
        local = name.rsplit(":", 1)[-1]
        position = parser.CurrentByteIndex
        if local in ("spectrum", "chromatogram"):
            entry = {
                "tag": local,
                "start": position,
                "attrs": dict(attrs),
                "array_spans": [],
            }
            stack.append(entry)
            current = entry
        elif local == "binaryDataArray" and current is not None:
            current["array_spans"].append({"start": position, "end": None})

    def end(name):
        nonlocal current
        local = name.rsplit(":", 1)[-1]
        if local == "binaryDataArray" and current is not None:
            if current["array_spans"]:
                current["array_spans"][-1]["end"] = parser.CurrentByteIndex + len(
                    "</{0}>".format(name)
                )
        elif local in ("spectrum", "chromatogram"):
            entry = stack.pop() if stack else None
            if entry is not None and entry["tag"] == local:
                entry["end"] = parser.CurrentByteIndex + len(
                    "</{0}>".format(name)
                )
                spectra.append(entry)
            current = stack[-1] if stack else None

    parser.StartElementHandler = start
    parser.EndElementHandler = end
    parser.Parse(document, True)
    spectra.sort(key=lambda entry: entry["start"])
    return spectra


def _decode_values(decoded_bytes, cv_names):
    """Decode plain little-endian numeric arrays; opaque encodings return ()"""
    names = set(cv_names)
    for name, code in (
        ("64-bit float", "d"),
        ("32-bit float", "f"),
        ("64-bit integer", "q"),
        ("32-bit integer", "i"),
    ):
        if name in names:
            size = struct.calcsize(code)
            count = len(decoded_bytes) // size
            if count == 0:
                return ()
            return struct.unpack("<" + code * count, decoded_bytes[: count * size])
    return ()


def snapshot_arrays(element):
    """Capture raw bytes, CV terms and decoded values for each array."""
    arrays = OrderedDict()
    try:
        default_length = int(element.get("defaultArrayLength"))
    except (TypeError, ValueError):
        default_length = None

    for position, array_element in enumerate(
        element.iterfind("{*}binaryDataArrayList/{*}binaryDataArray")
    ):
        cv_names = tuple(
            term.get("name", "")
            for term in array_element.findall("{*}cvParam")
        )
        name = next(
            (candidate for candidate in DEFAULT_ARRAY_NAMES if candidate in cv_names),
            None,
        )
        if name is None:
            name = cv_names[-1] if cv_names else "array[{0}]".format(position)

        binary = array_element.find("{*}binary")
        encoded_bytes = b""
        if binary is not None:
            try:
                encoded_bytes = base64.b64decode(_text(binary))
            except Exception:
                encoded_bytes = b""

        decoded_bytes = b""
        if encoded_bytes:
            try:
                decoded_bytes = zlib.decompress(encoded_bytes)
            except zlib.error:
                decoded_bytes = encoded_bytes

        try:
            encoded_length = int(array_element.get("encodedLength"))
        except (TypeError, ValueError):
            encoded_length = None

        arrays[name] = ArraySnapshot(
            name=name,
            cv_names=cv_names,
            encoded_length=encoded_length,
            encoded_bytes=encoded_bytes,
            decoded_bytes=decoded_bytes,
            values=_decode_values(decoded_bytes, cv_names),
            default_array_length=default_length,
        )
    return arrays


def parse_index_lists(document):
    """Return every ``indexList``/``offset`` entry from a complete document."""
    by_key = defaultdict(list)
    entries = []
    try:
        root = ElementTree.fromstring(document)
    except ElementTree.ParseError:
        return entries

    ordinal = 0
    for element in root.iter():
        if _local_name(element.tag) != "indexList":
            continue
        for index_element in element:
            if _local_name(index_element.tag) != "index":
                continue
            index_name = index_element.get("name", "")
            for offset_element in index_element:
                if _local_name(offset_element.tag) != "offset":
                    continue
                id_ref = offset_element.get("idRef", "")
                try:
                    offset = int(_text(offset_element))
                except (TypeError, ValueError):
                    continue
                tag = "chromatogram" if index_name == "chromatogram" else "spectrum"
                key = _index_key(id_ref, tag=tag)
                entries.append(
                    IndexEntry(
                        key=key,
                        id_ref=id_ref,
                        offset=offset,
                        ordinal=ordinal,
                        index_name=index_name,
                        source="native",
                    )
                )
                by_key[key].append(entries[-1])
                ordinal += 1
    return entries


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class IndexedGzipVerifier:
    """Compare container, index and Reader views of an mzML file.

    Args:
        path: input ``.mzML``, ``.mzML.gz`` or indexed ``.idx.gz`` file.
        encoding: XML encoding; defaults to ``latin-1``.
        array_names: binary array names whose values are compared.
        max_items: optional safety cap for the sequential scan.
    """

    def __init__(
        self,
        path,
        encoding="latin-1",
        array_names=DEFAULT_ARRAY_NAMES,
        max_items=None,
    ):
        self.path = str(path)
        self.encoding = encoding or "latin-1"
        self.array_names = tuple(array_names)
        self.max_items = max_items
        self.report = VerificationReport(path=self.path)
        self._blob = b""
        self._document = b""
        self._members = []
        self._member_problems = []
        self._records = []
        self._items = {}
        self._by_start = {}
        self._native_entries = []
        self._header_entries = []
        self._sequential = []
        self._indexed = False
        self._native_targets = {}
        self._header_targets = {}
        self._finding_keys = set()

    # -- public API ---------------------------------------------------------

    def run(self, query_ids=None, include_interleaved=True):
        """Execute all checks.

        Args:
            query_ids: explicit identifiers for random access.
            include_interleaved: run interleaved-query and reopen checks.

        Returns:
            VerificationReport
        """
        self._read_input()
        self._build_container_model()
        self._load_document_items()
        self._load_indexes()
        self._check_members()
        self._check_member_spanning()
        self._check_native_index()
        self._check_header_index()
        self._check_sequence()
        self._check_identity_interpretation()
        if self._indexed:
            self._check_random_access(query_ids)
            if include_interleaved:
                self._check_interleaved(query_ids)
                self._check_reopen(query_ids)
        else:
            self._add(
                "random_access_unsupported_input",
                "warning",
                "input has neither a native mzML index nor a GSGW header "
                "index; random access checks were skipped",
                details={
                    "native_index_entries": 0,
                    "header_index_entries": 0,
                    "sequential_items": len(self._sequential),
                },
            )
        self._check_sequence_duplicates()
        self._finalize()
        return self.report

    @classmethod
    def verify(cls, path, **kwargs):
        return cls(path, **kwargs).run()

    # -- input --------------------------------------------------------------

    def _read_input(self):
        with open(self.path, "rb") as handle:
            self._blob = handle.read()

    def _build_container_model(self):
        if self._blob[:2] != b"\x1f\x8b":
            self._document = self._blob
            self._members = []
            return
        self._members, self._member_problems = scan_gzip_members(self._blob)
        if self._member_problems:
            self._document = b""
            return
        payloads = []
        for member in self._members:
            payloads.append(
                zlib.decompress(
                    self._blob[member.deflate_start : member.deflate_end],
                    _DEFLATE_WBITS,
                )
            )
        self._document = b"".join(payloads)

    def _load_document_items(self):
        if not self._document:
            return
        self._records = _extract_spans(self._document)
        for ordinal, record in enumerate(self._records):
            record["ordinal"] = ordinal
            record["bytes"] = self._document[record["start"] : record["end"]]
            record["member"] = self._member_for(record["start"])
            record["id"] = record["attrs"].get("id", "")
            record["key"] = _index_key(
                record["id"],
                tag=record["tag"],
            )
            record["trailing"] = _trailing_number(record["id"])
            item = self._make_item(record)
            self._items[ordinal] = item
            self._by_start.setdefault(record["start"], item)
            record["item"] = item
        self.report.items = OrderedDict(self._items)

    def _make_item(self, record):
        try:
            element = ElementTree.fromstring(record["bytes"])
        except ElementTree.ParseError:
            element = None
        arrays = snapshot_arrays(element) if element is not None else OrderedDict()
        index_attr = record["attrs"].get("index")
        try:
            index_attr = int(index_attr)
        except (TypeError, ValueError):
            index_attr = None
        array_members = {}
        record["array_members"] = array_members
        for position, span in enumerate(record.get("array_spans", [])):
            if span.get("end") is None:
                continue
            array_members[position] = self._members_for(span["start"], span["end"])
            if arrays and position < len(arrays):
                arrays[list(arrays)[position]].member = array_members[position]
        return ItemSnapshot(
            key=record["key"],
            index_key=record["key"],
            tag=record["tag"],
            index_attribute=index_attr,
            id_attribute=record["id"],
            trailing_number=record["trailing"],
            id_tokens=_id_tokens(record["id"]),
            default_array_length=(
                int(record["attrs"]["defaultArrayLength"])
                if record["attrs"].get("defaultArrayLength", "").isdigit()
                else None
            ),
            arrays=arrays,
            ordinal=record["ordinal"],
            offset=record["start"],
            end=record["end"],
            member=record["member"],
            array_members=array_members,
        )

    # -- gzip boundary helpers ----------------------------------------------

    def _member_for(self, offset):
        for member in self._members:
            if member.uncompressed_start <= offset < member.uncompressed_end:
                return member.number
        return None

    def _members_for(self, start, end):
        return [
            member.number
            for member in self._members
            if member.uncompressed_start < end
            and member.uncompressed_end > start
        ]

    def _member_at_compressed(self, offset):
        for member in self._members:
            if member.header_start == offset or member.deflate_start == offset:
                return member
        return None

    # -- findings -----------------------------------------------------------

    def _add(
        self,
        code,
        severity,
        message,
        *,
        index=None,
        identity=None,
        boundary=None,
        arrays=(),
        details=None,
    ):
        key = (code, index, message)
        if key in self._finding_keys:
            return
        self._finding_keys.add(key)
        self.report.findings.append(
            VerificationFinding(
                code=code,
                severity=severity,
                message=message,
                index=index,
                identity=identity,
                boundary=boundary,
                arrays=tuple(arrays),
                details=details or {},
            )
        )

    # -- container checks ---------------------------------------------------

    def _check_members(self):
        if not self._members:
            return
        self.report.members = list(self._members)
        for member in self._members:
            if not member.crc32_ok or not member.isize_ok:
                self._add(
                    "gzip_member_trailer_mismatch",
                    "error",
                    "gzip member {0} fails its {1} check".format(
                        member.number,
                        "CRC32" if not member.crc32_ok else "ISIZE",
                    ),
                    boundary=member.to_dict(),
                    details={
                        "declared_crc32": hex(member.crc32),
                        "declared_isize": member.isize,
                        "actual_length": member.uncompressed_length,
                        "crc32_ok": member.crc32_ok,
                        "isize_ok": member.isize_ok,
                    },
                )
        for problem in self._member_problems:
            self._add(
                "gzip_member_unterminated",
                "error",
                "gzip container is malformed: {0}".format(problem["problem"]),
                boundary={
                    "member": problem.get("member"),
                    "start": problem.get("start"),
                },
                details=problem,
            )

    def _check_member_spanning(self):
        if not self._members:
            return
        for record, item in self._item_records():
            members = self._members_for(record["start"], record["end"])
            if len(members) > 1:
                self._add(
                    "element_spans_members",
                    "error",
                    "element {0!r} crosses compression block boundary {1}".format(
                        item.id_attribute, members
                    ),
                    index=item.key,
                    identity=self._identity(item),
                    boundary={
                        "start": record["start"],
                        "end": record["end"],
                        "members": members,
                    },
                    details={
                        "members": members,
                        "array_members": item.array_members,
                        "target_index": item.key,
                    },
                )
            for name, array in item.arrays.items():
                members = item.array_members.get(list(item.arrays).index(name), [])
                if len(members) > 1:
                    self._add(
                        "array_spans_members",
                        "error",
                        "array {0!r} of {1!r} is split across compression "
                        "blocks {2}".format(name, item.id_attribute, members),
                        index=item.key,
                        identity=self._identity(item),
                        boundary={
                            "member": members,
                            "start": record["start"],
                            "end": record["end"],
                        },
                        arrays=(name,),
                        details={
                            "members": members,
                            "target_index": item.key,
                        },
                    )

    # -- index loading ------------------------------------------------------

    def _load_indexes(self):
        self._native_entries = parse_index_lists(self._document)
        self.report.index_entries = list(self._native_entries)

        if self.path.endswith(".gz"):
            try:
                from .GSGR import GSGR

                reader = GSGR(self.path)
                try:
                    if not getattr(reader, "indexed", False):
                        reader.close()
                        self._add(
                            "header_index_missing",
                            "info",
                            "file is gzip but has no GSGW header index",
                        )
                        return
                    self._header_entries = [
                        IndexEntry(
                            key=key,
                            id_ref=str(key),
                            offset=offset,
                            ordinal=position,
                            index_name="header",
                            source="header",
                        )
                        for position, (key, offset) in enumerate(reader.index.items())
                    ]
                finally:
                    reader.close()
            except Exception as exc:
                self._add(
                    "header_index_unreadable",
                    "warning",
                    "could not read the GSGW header index: {0}".format(exc),
                )
        self.report.header_entries = list(self._header_entries)
        self._indexed = bool(self._native_entries or self._header_entries)

    # -- native index checks ------------------------------------------------

    def _item_records(self):
        return [(record, record["item"]) for record in self._records]

    def _check_native_index(self):
        if not self._records:
            return
        if not self._native_entries:
            self._add(
                "native_index_missing",
                "warning",
                "no native mzML indexList was found; only sequential access "
                "can be validated",
                details={"document_items": len(self._records)},
            )
            return

        # Duplicate keys shadow whole elements.
        grouped = defaultdict(list)
        for entry in self._native_entries:
            grouped[entry.key].append(entry)
        for key, entries in grouped.items():
            if len(entries) < 2:
                continue
            self._add(
                "native_index_duplicate_key",
                "error",
                "native index has {0} entries for key {1!r}".format(
                    len(entries), key
                ),
                index=key,
                details={
                    "offsets": [entry.offset for entry in entries],
                    "id_refs": [entry.id_ref for entry in entries],
                    "affected_range": [key],
                    "target_index": key,
                },
            )

        for entry in self._native_entries:
            exact = self._by_start.get(entry.offset)
            if exact is not None:
                self._compare_native_identity(entry, exact)
                continue

            previous = None
            for record in self._records:
                if record["start"] <= entry.offset:
                    previous = record
                else:
                    break
            if previous is None:
                self._add(
                    "native_index_offset_unmapped",
                    "error",
                    "native index offset {0} precedes the first element".format(
                        entry.offset
                    ),
                    index=entry.key,
                    details={
                        "id_ref": entry.id_ref,
                        "offset": entry.offset,
                        "target_index": entry.key,
                    },
                )
                continue
            self._add(
                "native_index_offset_misaligned",
                "error",
                "native index offset falls inside element {0!r}".format(
                    previous["id"]
                ),
                index=entry.key,
                identity=self._identity(previous["item"]),
                boundary={
                    "start": entry.offset,
                    "member": previous["member"],
                    "nearest_element_start": previous["start"],
                    "nearest_element_end": previous["end"],
                },
                details={
                    "id_ref": entry.id_ref,
                    "offset": entry.offset,
                    "delta": entry.offset - previous["start"],
                    "affected_range": [entry.key, previous["key"]],
                    "target_index": entry.key,
                },
            )

        # Completeness: every spectrum/chromatogram should have one entry.
        expected = Counter()
        spectrum_total = 0
        chromatogram_total = 0
        for record in self._records:
            expected[record["key"]] += 1
            if record["tag"] == "spectrum":
                spectrum_total += 1
            else:
                chromatogram_total += 1
        index_keys = Counter(entry.key for entry in self._native_entries)
        missing = [key for key, count in expected.items() if index_keys[key] < count]
        extra = [key for key in index_keys if key not in expected]
        if missing or extra:
            duplicates = {
                key: count for key, count in expected.items() if count > 1
            }
            if duplicates and set(missing) <= set(duplicates):
                code = "native_index_shadowed_keys"
                severity = "warning"
            else:
                code = "native_index_incomplete"
                severity = "error"
            self._add(
                code,
                severity,
                "native index does not cover the document: {0} key(s) missing, "
                "{1} extra".format(len(missing), len(extra)),
                details={
                    "missing_keys": missing[:50],
                    "extra_keys": extra[:50],
                    "duplicate_numbered_keys": duplicates,
                    "document_items": len(self._records),
                    "spectrum_items": spectrum_total,
                    "chromatogram_items": chromatogram_total,
                    "native_entries": len(self._native_entries),
                    "affected_range": sorted(
                        key for key in missing if isinstance(key, int)
                    )[:50],
                },
            )

    def _compare_native_identity(self, entry, item):
        if entry.id_ref.upper() == "TIC":
            if item.tag != "chromatogram":
                self._add(
                    "native_index_wrong_element",
                    "warning",
                    "native index TIC entry points at a {0}".format(item.tag),
                    index=entry.key,
                    identity=self._identity(item),
                    boundary={"start": item.offset, "member": item.member},
                )
            return
        self._native_targets[entry.key] = {
            "id": item.id_attribute,
            "key": item.index_key,
            "offset": item.offset,
            "member": item.member,
            "ordinal": item.ordinal,
            "source": "native",
        }
        if entry.key != item.index_key:
            self._add(
                "native_index_identity_mismatch",
                "error",
                "native index key {0!r} points at element with key {1!r}".format(
                    entry.key, item.index_key
                ),
                index=entry.key,
                identity=self._identity(item),
                boundary={"start": item.offset, "end": item.end, "member": item.member},
                details={
                    "id_ref": entry.id_ref,
                    "element_id": item.id_attribute,
                    "target_index": entry.key,
                },
            )
            return
        if entry.id_ref != item.id_attribute and entry.id_ref.upper() != "TIC":
            self._add(
                "native_index_id_ref_mismatch",
                "warning",
                "native index idRef {0!r} resolves to element id {1!r}".format(
                    entry.id_ref, item.id_attribute
                ),
                index=entry.key,
                identity=self._identity(item),
                details={
                    "id_ref": entry.id_ref,
                    "element_id": item.id_attribute,
                },
            )

    # -- GSGW header index checks ------------------------------------------

    def _check_header_index(self):
        if not self._header_entries:
            return

        offsets = sorted(entry.offset for entry in self._header_entries)
        numeric_header = {
            entry.key for entry in self._header_entries
            if isinstance(entry.key, int)
        }
        numeric_document = {
            record["key"] for record in self._records
            if record["tag"] == "spectrum" and isinstance(record["key"], int)
        }
        for missing in sorted(numeric_document - numeric_header):
            self._add(
                "header_index_incomplete",
                "error",
                "spectrum key {0!r} exists in the document but not in the "
                "GSGW header index".format(missing),
                index=missing,
                details={
                    "missing_keys": sorted(numeric_document - numeric_header),
                    "target_index": missing,
                },
            )
        for entry in self._header_entries:
            member = self._member_at_compressed(entry.offset)
            if member is None:
                self._add(
                    "header_index_offset_misaligned",
                    "error",
                    "header index key {0!r} offset {1} is not a gzip member "
                    "start".format(entry.key, entry.offset),
                    index=entry.key,
                    boundary={"start": entry.offset},
                    details={"target_index": entry.key},
                )
                continue

            following = [value for value in offsets if value > entry.offset]
            span_end = following[0] if following else len(self._blob)
            # GSGR reads [own offset, next offset) and lets zlib stop at the
            # deflate end, so a well-formed member consumes only its payload.
            clipped_end = min(span_end, member.trailer_end)
            payload = self._read_header_block(entry.offset, clipped_end)
            self._analyse_header_payload(entry, member, payload, span_end)

    def _read_header_block(self, start, end):
        """Reproduce ``GSGR.read_block`` byte-for-byte for one header key."""
        try:
            decompressor = zlib.decompressobj(_DEFLATE_WBITS)
            return decompressor.decompress(self._blob[start:end])
        except Exception:
            return b""

    def _header_payload_contiguous(self, member, payload):
        """True when this member starts at a document element boundary."""
        head = payload.lstrip()
        if not head:
            return True
        return head.startswith(b"<spectrum") or head.startswith(b"<chromatogram") or (
            head.startswith(b"<?xml")
        )

    def _analyse_header_payload(self, entry, member, payload, span_end=None):
        if not payload:
            self._add(
                "header_index_block_unreadable",
                "error",
                "header index key {0!r} at offset {1} does not decompress".format(
                    entry.key, entry.offset
                ),
                index=entry.key,
                boundary=member.to_dict(),
                details={"target_index": entry.key},
            )
            return

        text = payload.decode("utf-8", "replace")
        stripped = text.lstrip()
        if not stripped.startswith("<spectrum") and not stripped.startswith(
            "<chromatogram"
        ):
            contiguous = self._header_payload_contiguous(member, payload)
            structural = {
                "Head": "preamble",
                "junk": "preamble",
                "tail": "trailer",
            }.get(entry.key) if isinstance(entry.key, str) else None
            self._add(
                "header_index_points_to_non_element",
                "info" if structural or contiguous else "error",
                "header index key {0!r} holds the {1}, not an element".format(
                    entry.key, structural
                )
                if structural
                else "header index key {0!r} points at a fragment inside an "
                "element (continuation block {2})".format(
                    entry.key, entry.offset, member.number
                ),
                index=entry.key,
                boundary=member.to_dict(),
                details={
                    "head": stripped[:120],
                    "target_index": entry.key,
                    "structural": bool(structural),
                    "block_is_element_continuation": not contiguous,
                    "explanation": (
                        "this block continues an element that started in the "
                        "previous compression member; the native element offset "
                        "and the compression boundary disagree"
                        if not contiguous
                        else "structural chunk written by index_gzip"
                    ),
                },
            )
            return

        try:
            element = ElementTree.fromstring(
                '<mzML xmlns="http://psi.hupo.org/ms/mzml">' + text + "</mzML>"
            )[0]
        except ElementTree.ParseError as exc:
            if self._header_payload_contiguous(member, payload) and (
                span_end is not None and member.trailer_end < span_end
            ):
                # The member is intact but the next index offset cuts this
                # block short of the element end.
                self._add(
                    "header_index_block_short",
                    "error",
                    "header index key {0!r} block ends before the element is "
                    "complete (next offset {1})".format(entry.key, span_end),
                    index=entry.key,
                    boundary={**member.to_dict(), "next_offset": span_end},
                    details={
                        "head": stripped[:120],
                        "tail": stripped[-120:],
                        "payload_length": len(text),
                        "next_offset": span_end,
                        "target_index": entry.key,
                    },
                )
                return
            self._add(
                "header_index_block_truncated",
                "error",
                "header index key {0!r} block is not well formed: {1}".format(
                    entry.key, exc
                ),
                index=entry.key,
                boundary=member.to_dict(),
                details={
                    "head": stripped[:120],
                    "tail": stripped[-120:],
                    "payload_length": len(text),
                    "target_index": entry.key,
                },
            )
            return

        tag = _local_name(element.tag)
        id_attr = element.get("id", "")
        key = _index_key(id_attr, tag=tag)
        self._header_targets[entry.key] = {
            "id": id_attr,
            "key": key,
            "offset": entry.offset,
            "member": member.number,
            "payload_length": len(text),
            "source": "header",
        }
        sentinel = isinstance(entry.key, str) and entry.key in (
            "Head",
            "junk",
            "tail",
        )
        if not sentinel and key != entry.key:
            self._add(
                "header_index_wrong_element",
                "error",
                "header index key {0!r} decompresses to element {1!r} "
                "(key {2!r})".format(entry.key, id_attr, key),
                index=entry.key,
                identity=self._identity_from_values(
                    id_attr, tag, element.get("index")
                ),
                boundary=member.to_dict(),
                details={
                    "returned_id": id_attr,
                    "returned_key": key,
                    "target_index": entry.key,
                },
            )

    # -- sequential checks --------------------------------------------------

    def _check_sequence(self):
        import pymzml.run as run_module

        observed = []
        try:
            reader = run_module.Reader(self.path)
        except Exception as exc:
            self._add(
                "reader_open_failed",
                "error",
                "Reader could not open the file: {0}: {1}".format(
                    type(exc).__name__, exc
                ),
            )
            return
        try:
            for position, item in enumerate(reader):
                arrays = snapshot_arrays(item.element)
                tag = (
                    "spectrum"
                    if item.element.tag.endswith("}spectrum")
                    else "chromatogram"
                )
                key = (
                    getattr(item, "index", None)
                    if tag == "spectrum"
                    else item.ID
                )
                observed.append(
                    {
                        "position": position,
                        "key": key,
                        "id": item.element.get("id", ""),
                        "tag": tag,
                        "arrays": arrays,
                    }
                )
                if self.max_items and position + 1 >= self.max_items:
                    break
        except Exception as exc:
            self._add(
                "sequence_read_error",
                "error",
                "sequential iteration failed after {0} item(s): {1}: {2}".format(
                    len(observed), type(exc).__name__, exc
                ),
                details={
                    "items_read": len(observed),
                    "last_keys": [entry["key"] for entry in observed[-5:]],
                },
            )
        finally:
            try:
                reader.close()
            except Exception:
                pass
        self._sequential = observed

        declared = None
        try:
            probe = run_module.Reader(self.path)
            declared = probe.get_spectrum_count()
            probe.close()
        except Exception:
            declared = None
        self.report.summary["declared_spectrum_count"] = declared
        self.report.summary["sequential_items"] = len(observed)

        document_spectra = [
            record for record in self._records if record["tag"] == "spectrum"
        ]
        observed_spectra = [
            entry for entry in observed if entry["tag"] == "spectrum"
        ]
        if len(observed_spectra) != len(document_spectra):
            self._add(
                "sequence_document_mismatch",
                "error",
                "sequential iteration returned {0} spectra but the uncompressed "
                "document contains {1}".format(
                    len(observed_spectra), len(document_spectra)
                ),
                details={
                    "sequential_items": len(observed_spectra),
                    "document_items": len(document_spectra),
                    "document_keys": [r["key"] for r in document_spectra[:50]],
                    "sequential_keys": [s["key"] for s in observed_spectra[:50]],
                },
            )
        else:
            mismatches = []
            for record, seen in zip(document_spectra, observed_spectra):
                if record["id"] != seen["id"]:
                    mismatches.append(
                        {
                            "position": seen["position"],
                            "document_id": record["id"],
                            "observed_id": seen["id"],
                        }
                    )
            if mismatches:
                self._add(
                    "sequence_identity_mismatch",
                    "error",
                    "sequential order or identity differs from the document",
                    details={"mismatches": mismatches[:20]},
                )

        if declared is not None and declared != len(observed_spectra):
            # A wrong ``spectrumList/@count`` is a file-level metadata issue.
            # It only becomes a random-access problem if the document itself
            # contains more spectra than sequential iteration returned.
            severity = (
                "error" if declared == len(document_spectra) else "warning"
            )
            self._add(
                "sequence_count_mismatch",
                severity,
                "spectrumList declares {0} spectra, document contains {1} and "
                "sequential iteration returned {2}".format(
                    declared, len(document_spectra), len(observed_spectra)
                ),
                details={
                    "declared_spectrum_count": declared,
                    "document_spectra": len(document_spectra),
                    "sequential_spectra": len(observed_spectra),
                    "affected_range": [s["key"] for s in observed_spectra],
                },
            )

    def _check_sequence_duplicates(self):
        keys = [entry["key"] for entry in self._sequential]
        duplicates = {
            key: count for key, count in Counter(keys).items() if count > 1
        }
        if duplicates:
            self._add(
                "sequential_duplicate_key",
                "error",
                "sequential iteration yields keys more than once: {0}".format(
                    sorted(duplicates)
                ),
                details={
                    "duplicates": duplicates,
                    "affected_range": sorted(duplicates),
                },
            )

    # -- identity checks ----------------------------------------------------

    def _identity(self, item):
        return {
            "id": item.id_attribute,
            "index_key": item.index_key,
            "trailing_number": item.trailing_number,
            "tokens": dict(item.id_tokens),
            "element_index": item.index_attribute,
        }

    def _identity_from_values(self, id_attr, tag="spectrum", element_index=None):
        try:
            element_index = int(element_index)
        except (TypeError, ValueError):
            pass
        return {
            "id": id_attr,
            "index_key": _index_key(id_attr, tag=tag),
            "trailing_number": _trailing_number(id_attr),
            "tokens": dict(_id_tokens(id_attr)),
            "element_index": element_index,
        }

    def _check_identity_interpretation(self):
        grouped = defaultdict(list)
        for item in self._items.values():
            if item.tag == "spectrum":
                grouped[item.index_key].append(item)
        for key, items in grouped.items():
            if len(items) < 2:
                continue
            self._add(
                "identity_shadowed",
                "warning",
                "{0} spectra map to native key {1!r}; random access cannot "
                "select between them".format(len(items), key),
                index=key,
                identity={"key": key, "ids": [item.id_attribute for item in items]},
                details={
                    "ids": [item.id_attribute for item in items],
                    "ordinals": [item.ordinal for item in items],
                    "affected_range": [key],
                    "target_index": key,
                },
            )

        trailing_groups = defaultdict(list)
        for item in self._items.values():
            if item.tag == "spectrum" and item.trailing_number is not None:
                trailing_groups[item.trailing_number].append(item)
        for trailing, items in trailing_groups.items():
            keys = {item.index_key for item in items}
            if len(keys) < 2:
                continue
            self._add(
                "identity_trailing_collision",
                "warning",
                "Spectrum.ID {0} is shared by spectra with distinct native "
                "keys {1}".format(trailing, sorted(map(str, keys))),
                index=trailing,
                identity={
                    "trailing_number": trailing,
                    "ids": [item.id_attribute for item in items],
                    "index_keys": sorted(map(str, keys)),
                },
                details={
                    "ids": [item.id_attribute for item in items],
                    "index_keys": sorted(map(str, keys)),
                    "affected_range": sorted(map(str, keys)),
                    "target_index": trailing,
                },
            )

        for item in self._items.values():
            if item.tag != "spectrum":
                continue
            scan = item.id_tokens.get("scan") or item.id_tokens.get("scanId")
            if scan is not None and scan.isdigit() and int(scan) != item.trailing_number:
                self._add(
                    "identity_index_disagreement",
                    "warning",
                    "id {0!r} implies index key {1} but Spectrum.ID reports "
                    "{2}".format(item.id_attribute, int(scan), item.trailing_number),
                    index=item.index_key,
                    identity=self._identity(item),
                    details={
                        "scan_token": int(scan),
                        "trailing_number": item.trailing_number,
                        "target_index": item.index_key,
                    },
                )

    # -- random access ------------------------------------------------------

    def _document_items(self):
        return [(record, record["item"]) for record in self._records]

    def _default_query_ids(self):
        ids = []
        counts = Counter(record["key"] for record in self._records)
        for key in sorted(
            (key for key in counts if isinstance(key, int))
        )[:5]:
            ids.append(key)
        for key, count in counts.items():
            if count > 1 and key not in ids:
                ids.append(key)
                break
        for record in self._records:
            if record["key"] not in ids:
                ids.append(record["key"])
                break
        for record in self._records:
            if record["tag"] == "chromatogram" and record["key"] not in ids:
                ids.append(record["key"])
                break
        if self._records:
            last = self._records[-1]["key"]
            if last not in ids:
                ids.append(last)
        return ids

    def _observe(self, reader, requested, mode="random", position=None):
        observation = QueryObservation(requested=requested, mode=mode)
        try:
            item = reader[requested]
        except Exception as exc:
            message = str(exc)
            prefix = "{0}: ".format(type(exc).__name__)
            observation.exception = (
                message if message.startswith(prefix) else prefix + message
            )
            observation.exception_type = type(exc).__name__
            return observation
        observation.resolved_index = getattr(item, "index", None)
        observation.resolved_id = item.element.get("id", "")
        observation.trailing_number = _trailing_number(observation.resolved_id)
        observation.id_tokens = dict(_id_tokens(observation.resolved_id))
        if hasattr(item, "element"):
            arrays = snapshot_arrays(item.element)
            observation.array_values = {
                name: array.values for name, array in arrays.items()
            }
            observation.array_lengths = {
                name: array.count for name, array in arrays.items()
            }
            observation.array_digests = {
                name: array.value_digest for name, array in arrays.items()
            }
        observation.sequential_position = position
        return observation

    def _expected_items(self, key):
        return [
            (record, item)
            for record, item in self._document_items()
            if item.index_key == key
        ]

    def _check_random_access(self, query_ids):
        import pymzml.run as run_module

        ids = list(query_ids) if query_ids else self._default_query_ids()
        self.report.summary["query_ids"] = ids
        try:
            reader = run_module.Reader(self.path)
        except Exception as exc:
            self._add(
                "reader_open_failed",
                "error",
                "Reader could not open the file: {0}".format(exc),
            )
            return
        try:
            for key in ids:
                observation = self._observe(reader, key)
                self.report.queries.append(observation)
                self._evaluate_query(key, observation)
        finally:
            reader.close()

    def _evaluate_query(self, key, observation):
        expected = self._expected_items(key)
        if observation.exception:
            self._add(
                "random_access_raised",
                "error",
                "random access by {0!r} raised {1}: {2}".format(
                    key, observation.exception_type, observation.exception
                ),
                index=key,
                details={
                    "exception": observation.exception,
                    "expected": [
                        self._item_evidence(record, item)
                        for record, item in expected[:10]
                    ],
                    "target_index": key,
                },
            )
            return
        if not expected:
            return

        if not any(item.id_attribute == observation.resolved_id for record, item in expected):
            actual = self._record_for_returned(observation.resolved_id)
            array_evidence = self._compare_returned_to_expected(
                expected, observation
            )
            self._add(
                "random_access_wrong_element",
                "error",
                "random access by {0!r} returned id {1!r}, not the element "
                "keyed {0!r}".format(key, observation.resolved_id),
                index=key,
                identity={
                    "requested": key,
                    "returned_id": observation.resolved_id,
                    "returned_key": _index_key(
                        observation.resolved_id,
                        tag=actual["tag"] if actual else "spectrum",
                    ),
                    "returned_trailing_number": observation.trailing_number,
                },
                boundary={
                    "start": actual["start"] if actual else None,
                    "end": actual["end"] if actual else None,
                    "member": actual.get("member") if actual else None,
                },
                arrays=tuple(
                    name for name, payload in array_evidence.items()
                    if payload.get("lengths_equal") and not payload.get("values_equal")
                ),
                details={
                    "actual_target": (
                        self._item_evidence(actual, actual["item"])
                        if actual
                        else None
                    ),
                    "expected_ids": sorted(
                        {item.id_attribute for record, item in expected}
                    ),
                    "returned": self._returned_evidence(observation),
                    "expected": [
                        self._item_evidence(record, item)
                        for record, item in expected[:10]
                    ],
                    "array_evidence": array_evidence,
                    "affected_range": [
                        value
                        for value in (
                            key,
                            _index_key(
                                observation.resolved_id,
                                tag=actual["tag"] if actual else "spectrum",
                            ),
                        )
                        if isinstance(value, int)
                    ],
                    "target_index": key,
                },
            )
            return

        matches = [
            (record, item)
            for record, item in expected
            if item.id_attribute == observation.resolved_id
        ]
        if len(matches) > 1:
            self._add(
                "random_access_ambiguous",
                "warning",
                "random access by {0!r} returned id {1!r}, which occurs {2} "
                "times".format(key, observation.resolved_id, len(matches)),
                index=key,
                identity={
                    "returned_id": observation.resolved_id,
                    "occurrences": len(matches),
                },
                details={"target_index": key},
            )
        record, item = matches[0]
        self._compare_query_arrays(key, record, item, observation)

    def _record_for_returned(self, resolved_id):
        for record in self._records:
            if record["tag"] == "spectrum" and record["id"] == resolved_id:
                return record
        return None

    def _compare_returned_to_expected(self, expected, observation):
        """Evidence for a wrong-element result, keyed by array name.

        Lengths are recorded explicitly so an equal-length but different-value
        result can be reported rather than silently accepted.
        """
        payloads = {}
        for name in self.array_names:
            expected_arrays = [
                item.arrays[name]
                for record, item in expected
                if name in item.arrays
            ]
            if not expected_arrays:
                continue
            returned_digest = observation.array_digests.get(name)
            returned_count = observation.array_lengths.get(name)
            expected_digest = expected_arrays[0].value_digest
            expected_count = expected_arrays[0].count
            payloads[name] = {
                "expected_count": expected_count,
                "returned_count": returned_count,
                "lengths_equal": returned_count == expected_count,
                "values_equal": returned_digest == expected_digest,
                "expected_digest": expected_digest,
                "returned_digest": returned_digest,
            }
        return payloads

    def _item_evidence(self, record, item):
        return {
            "offset": record["start"],
            "member": record.get("member"),
            "id": item.id_attribute,
            "key": item.index_key,
            "ordinal": item.ordinal,
        }

    def _returned_evidence(self, observation):
        return {
            "id": observation.resolved_id,
            "element_index": observation.resolved_index,
            "trailing_number": observation.trailing_number,
            "array_lengths": observation.array_lengths,
        }

    def _compare_query_arrays(self, key, record, item, observation, label="random"):
        for name in self.array_names:
            expected_array = item.arrays.get(name)
            if expected_array is None:
                continue
            actual_digest = observation.array_digests.get(name)
            if actual_digest is None:
                continue
            if actual_digest == expected_array.value_digest:
                continue
            self._add(
                "array_value_mismatch",
                "error",
                "{0} access for {1!r} decoded different {2!r} values than the "
                "document element at offset {3}".format(
                    label, key, name, record["start"]
                ),
                index=key,
                identity=self._identity(item),
                boundary={
                    "start": record["start"],
                    "end": record["end"],
                    "member": record.get("member"),
                },
                arrays=(name,),
                details={
                    "expected_count": len(expected_array.values),
                    "returned_count": observation.array_lengths.get(name),
                    "expected_digest": expected_array.value_digest,
                    "returned_digest": actual_digest,
                    "note": "equal lengths are not sufficient; values differ",
                    "target_index": key,
                },
            )

    # -- interleaved and reopen checks -------------------------------------

    def _sequential_keys(self):
        return [entry["key"] for entry in self._sequential]

    def _check_interleaved(self, query_ids):
        import pymzml.run as run_module

        ids = list(query_ids) if query_ids else self._default_query_ids()
        if not ids:
            return
        try:
            reader = run_module.Reader(self.path)
        except Exception:
            return
        try:
            iterator = iter(reader)
            consumed = 0
            for _ in range(min(2, len(self._sequential))):
                next(iterator)
                consumed += 1
            if consumed < 2 or consumed >= len(self._sequential):
                return
            baseline = self._sequential_keys()
            for key in ids[:4]:
                observation = self._observe(
                    reader, key, mode="interleaved", position=consumed
                )
                self.report.queries.append(observation)
                if observation.exception:
                    self._add(
                        "interleaved_query_raised",
                        "error",
                        "query {0!r} during sequential reading raised {1}".format(
                            key, observation.exception_type
                        ),
                        index=key,
                        details={
                            "exception": observation.exception,
                            "after_items": consumed,
                            "target_index": key,
                        },
                    )
                    continue
                expected_items = self._expected_items(key)
                if expected_items and not any(
                    item.id_attribute == observation.resolved_id
                    for record, item in expected_items
                ):
                    actual = self._record_for_returned(observation.resolved_id)
                    self._add(
                        "interleaved_wrong_element",
                        "error",
                        "query {0!r} during sequential reading returned id "
                        "{1!r}".format(key, observation.resolved_id),
                        index=key,
                        identity={
                            "requested": key,
                            "returned_id": observation.resolved_id,
                            "returned_key": _index_key(
                                observation.resolved_id,
                                tag=actual["tag"] if actual else "spectrum",
                            ),
                        },
                        boundary={
                            "start": actual["start"] if actual else None,
                            "end": actual["end"] if actual else None,
                            "member": actual.get("member") if actual else None,
                        },
                        details={
                            "after_items": consumed,
                            "actual_target": (
                                self._item_evidence(actual, actual["item"])
                                if actual else None
                            ),
                            "expected_ids": sorted(
                                {item.id_attribute for record, item in expected_items}
                            ),
                            "affected_range": [
                                value
                                for value in (key, _index_key(
                                    observation.resolved_id,
                                    tag=actual["tag"] if actual else "spectrum",
                                ))
                                if isinstance(value, int)
                            ],
                            "target_index": key,
                        },
                    )
                try:
                    following = next(iterator)
                except StopIteration:
                    continue
                except Exception as exc:
                    self._add(
                        "interleaved_sequence_desync",
                        "error",
                        "sequential reading failed after query {0!r}: {1}".format(
                            key, exc
                        ),
                        index=key,
                        details={"after_items": consumed, "target_index": key},
                    )
                    continue
                observed_key = getattr(following, "index", None)
                expected_key = (
                    baseline[consumed] if consumed < len(baseline) else None
                )
                if expected_key is not None and observed_key != expected_key:
                    affected = [
                        value
                        for value in (expected_key, observed_key)
                        if isinstance(value, int)
                    ]
                    self._add(
                        "interleaved_sequence_jump",
                        "error",
                        "after query {0!r} sequential reading continued at "
                        "{1!r} instead of {2!r}".format(
                            key, observed_key, expected_key
                        ),
                        index=key,
                        identity={
                            "after_query": key,
                            "continued_with": observed_key,
                            "expected": expected_key,
                        },
                        details={
                            "after_items": consumed,
                            "continued_with": observed_key,
                            "expected": expected_key,
                            "affected_range": affected,
                            "target_index": key,
                        },
                    )
                consumed += 1
        finally:
            reader.close()

    def _check_reopen(self, query_ids):
        import pymzml.run as run_module

        ids = list(query_ids) if query_ids else self._default_query_ids()
        before = self._sequential_keys()[:3]
        try:
            reader = run_module.Reader(self.path)
        except Exception:
            return
        try:
            after = []
            for _ in range(3):
                item = next(reader)
                after.append(item.index)
        except Exception:
            after = after if "after" in locals() else []
        finally:
            reader.close()
        if before and after and before != after:
            self._add(
                "reopen_sequence_changed",
                "error",
                "sequential order differs on a fresh Reader",
                details={"before": before, "after": after},
            )
        try:
            reopened = run_module.Reader(self.path)
        except Exception as exc:
            self._add(
                "reopen_failed",
                "error",
                "Reader could not reopen the input: {0}".format(exc),
            )
            return
        try:
            for key in ids[:3]:
                observation = self._observe(reopened, key, mode="reopened")
                self.report.queries.append(observation)
                expected = self._expected_items(key)
                if observation.exception:
                    self._add(
                        "reopen_query_raised",
                        "error",
                        "query {0!r} after reopening raised {1}".format(
                            key, observation.exception_type
                        ),
                        index=key,
                        details={
                            "exception": observation.exception,
                            "target_index": key,
                        },
                    )
                elif expected and not any(
                    item.id_attribute == observation.resolved_id
                    for record, item in expected
                ):
                    self._add(
                        "reopen_wrong_element",
                        "error",
                        "after reopening, query {0!r} returned id {1!r}".format(
                            key, observation.resolved_id
                        ),
                        index=key,
                        details={
                            "expected_ids": sorted(
                                {item.id_attribute for record, item in expected}
                            ),
                            "returned_id": observation.resolved_id,
                            "target_index": key,
                        },
                    )
        finally:
            reopened.close()

    # -- cross-input comparison --------------------------------------------

    def targets(self):
        """Return ``{"native": ..., "header": ...}`` key -> actual target maps."""
        return {
            "native": dict(self._native_targets),
            "header": dict(self._header_targets),
        }

    # -- summary ------------------------------------------------------------

    def _finalize(self):
        codes = Counter(finding.code for finding in self.report.findings)
        self.report.summary.update(
            {
                "ok": self.report.ok,
                "errors": len(self.report.errors),
                "warnings": len(self.report.warnings),
                "codes": dict(codes),
                "native_index_entries": len(self._native_entries),
                "header_index_entries": len(self._header_entries),
                "indexed": self._indexed,
                "document_items": len(self._records),
                "gzip_members": len(self._members),
                "sequential_items": len(self._sequential),
            }
        )


def compare_indexed_gzip(reference, candidate, **kwargs):
    """Compare two inputs and report index entries whose target changed.

    This is intended for the "input was replaced" case: the candidate may be
    structurally readable while its index still addresses the reference
    layout.  Every changed key is reported with the reference identity, the
    candidate's *actual* target and the affected key range.

    Args:
        reference: path of the input the index was built for.
        candidate: path of the replacement input.

    Returns:
        VerificationReport with ``input_*`` findings.
    """
    report = VerificationReport(path="{0} -> {1}".format(reference, candidate))
    reference_verifier = IndexedGzipVerifier(reference, **kwargs)
    reference_report = reference_verifier.run(include_interleaved=False)
    candidate_verifier = IndexedGzipVerifier(candidate, **kwargs)
    candidate_report = candidate_verifier.run(include_interleaved=False)

    reference_targets = reference_verifier.targets()
    candidate_targets = candidate_verifier.targets()

    changed_keys = set()
    for source in ("native", "header"):
        ref_map = reference_targets[source]
        cand_map = candidate_targets[source]
        code = (
            "input_native_index_target_changed"
            if source == "native"
            else "input_header_index_target_changed"
        )
        for key in sorted(set(ref_map) | set(cand_map), key=lambda value: str(value)):
            expected = ref_map.get(key)
            actual = cand_map.get(key)
            if expected is None:
                changed_keys.add(key)
                report.findings.append(
                    VerificationFinding(
                        code="input_index_key_added",
                        severity="warning",
                        message="candidate index key {0!r} has no reference "
                        "counterpart".format(key),
                        index=key,
                        details={
                            "reference": None,
                            "candidate": actual,
                            "actual_target": actual,
                            "affected_range": [key] if isinstance(key, int) else [],
                            "target_index": key,
                        },
                    )
                )
                continue
            if actual is None:
                changed_keys.add(key)
                report.findings.append(
                    VerificationFinding(
                        code="input_index_key_missing",
                        severity="error",
                        message="candidate index lost key {0!r}".format(key),
                        index=key,
                        details={
                            "reference": expected,
                            "candidate": None,
                            "actual_target": None,
                            "affected_range": [key] if isinstance(key, int) else [],
                            "target_index": key,
                        },
                    )
                )
                continue
            if expected["id"] == actual["id"]:
                continue
            changed_keys.add(key)
            report.findings.append(
                VerificationFinding(
                    code=code,
                    severity="error",
                    message="index key {0!r} targeted {1!r} in the reference "
                    "but targets {2!r} after the input was replaced".format(
                        key, expected["id"], actual["id"]
                    ),
                    index=key,
                    identity={
                        "reference_id": expected["id"],
                        "candidate_id": actual["id"],
                        "source": source,
                    },
                    boundary={
                        "start": actual.get("offset"),
                        "member": actual.get("member"),
                    },
                    details={
                        "reference": expected,
                        "candidate": actual,
                        "actual_target": actual,
                        "affected_range": sorted(
                            value
                            for value in changed_keys
                            if isinstance(value, int)
                        ),
                        "target_index": key,
                    },
                )
            )

    numeric_changed = sorted(
        value for value in changed_keys if isinstance(value, int)
    )
    report.summary = {
        "reference": reference,
        "candidate": candidate,
        "reference_ok": reference_report.ok,
        "candidate_ok": candidate_report.ok,
        "changed_keys": numeric_changed,
        "affected_range": [
            min(numeric_changed), max(numeric_changed)
        ] if numeric_changed else [],
        "reference_errors": sorted(
            {finding.code for finding in reference_report.errors}
        ),
        "candidate_errors": sorted(
            {finding.code for finding in candidate_report.errors}
        ),
    }
    return report


def verify_indexed_gzip(path, **kwargs):
    """Verify an indexed gzip mzML file and return a :class:`VerificationReport`.

    Args:
        path: input file.

    Keyword Args:
        query_ids: explicit identifiers for random access.
        encoding: XML encoding.
        array_names: array names compared by value.
        max_items: safety cap for sequential scanning.
        include_interleaved: run interleaved and reopen checks.
    """
    include_interleaved = kwargs.pop("include_interleaved", True)
    query_ids = kwargs.pop("query_ids", None)
    verifier = IndexedGzipVerifier(path, **kwargs)
    return verifier.run(
        query_ids=query_ids, include_interleaved=include_interleaved
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="offline mzML index verification")
    parser.add_argument("path")
    parser.add_argument(
        "--replace",
        metavar="CANDIDATE",
        help="compare against a replacement input and report changed targets",
    )
    parser.add_argument("--query", action="append", default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--info", action="store_true")
    args = parser.parse_args()

    queries = None
    if args.query:
        queries = []
        for value in args.query:
            try:
                queries.append(int(value))
            except ValueError:
                queries.append(value)
    if args.replace:
        report = compare_indexed_gzip(args.path, args.replace)
    else:
        report = verify_indexed_gzip(args.path, query_ids=queries)
    if args.json:
        import json

        print(json.dumps(report.to_dict(), indent=2, default=str))
    else:
        print(report.format(include_info=args.info))
    raise SystemExit(0 if report.ok else 1)
