"""
Build small, self-contained mzML fixtures for index verification tests.

Everything is generated locally so the tests run offline and the exact byte
layout of every compression block is known.
"""

import base64
import struct

MZML_HEAD = (
    '<?xml version="1.0" encoding="ISO-8859-1"?>\n'
    '<mzML xmlns="http://psi.hupo.org/ms/mzml" '
    'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
    'xsi:schemaLocation="http://psi.hupo.org/ms/mzml '
    'http://psidev.info/files/ms/mzML/xsd/mzML1.1.0.xsd" '
    'version="1.1.0">\n'
    '<cvList count="1"><cv id="MS" fullName="PSI-MS" '
    'version="4.1.79" URI="http://psi.hupo.org/ms/mzml"/></cvList>\n'
    '<run id="fixture" defaultInstrumentConfigurationRef="ic1">\n'
)

MZML_TAIL = "</run>\n</mzML>\n"

ARRAY_TEMPLATE = (
    '<binaryDataArray encodedLength="{encoded_length}">'
    '<cvParam cvRef="MS" accession="MS:1000523" name="64-bit float" value=""/>'
    '<cvParam cvRef="MS" accession="MS:1000576" name="no compression" value=""/>'
    '<cvParam cvRef="MS" accession="{accession}" name="{name}" value=""/>'
    "<binary>{payload}</binary>"
    "</binaryDataArray>"
)


def encode(values):
    """Base64 of little-endian float64 values."""
    raw = b"".join(struct.pack("<d", value) for value in values)
    return base64.b64encode(raw).decode("ascii")


def spectrum_xml(
    index,
    native_id,
    mz,
    intensity,
    *,
    id_attribute="id",
    index_attribute="index",
    default_array_length=None,
    extra_attributes="",
):
    """Build one spectrum element with an m/z and an intensity array."""
    if default_array_length is None:
        default_array_length = len(mz)
    mz_payload = encode(mz)
    intensity_payload = encode(intensity)
    attributes = []
    if index_attribute is not None:
        attributes.append('{0}="{1}"'.format(index_attribute, index))
    if id_attribute is not None:
        attributes.append('{0}="{1}"'.format(id_attribute, native_id))
    attributes.append('defaultArrayLength="{0}"'.format(default_array_length))
    if extra_attributes:
        attributes.append(extra_attributes)
    return (
        "<spectrum {attrs}>"
        '<cvParam cvRef="MS" accession="MS:1000511" name="ms level" value="1"/>'
        '<binaryDataArrayList count="2">'
        "{mz_array}{intensity_array}"
        "</binaryDataArrayList>"
        "</spectrum>"
    ).format(
        attrs=" ".join(attributes),
        mz_array=ARRAY_TEMPLATE.format(
            encoded_length=len(mz_payload),
            accession="MS:1000514",
            name="m/z array",
            payload=mz_payload,
        ),
        intensity_array=ARRAY_TEMPLATE.format(
            encoded_length=len(intensity_payload),
            accession="MS:1000515",
            name="intensity array",
            payload=intensity_payload,
        ),
    )


def index_block(entries, index_name="spectrum"):
    """Build an ``indexList`` with one index."""
    offsets = "".join(
        '<offset idRef="{0}">{1}</offset>'.format(id_ref, offset)
        for id_ref, offset in entries
    )
    return (
        '<indexList count="1"><index name="{0}">{1}</index></indexList>'
    ).format(index_name, offsets)


def build_document(spectra, index_entries=None, preamble="<spectrumList"):
    """Assemble a document; offsets are computed from the actual layout.

    ``spectra`` is a list of ``{"key": ..., "xml": ...}`` dicts.
    ``index_entries`` is a list of ``(idRef, {"key": ..., "offset": ...})``.
    """
    count = '<spectrumList count="{0}">\n'.format(len(spectra))
    if preamble == "<spectrumList":
        head = MZML_HEAD + count
    else:
        head = preamble + count
    pieces = [head]
    offsets = {}
    cursor = len(head.encode("iso-8859-1"))
    for spectrum in spectra:
        offsets[spectrum["key"]] = cursor
        piece = spectrum["xml"] + "\n"
        pieces.append(piece)
        cursor += len(piece.encode("iso-8859-1"))
    pieces.append("</spectrumList>\n")
    index_start = cursor
    if index_entries is not None:
        resolved = [
            (id_ref, entry.get("offset", offsets[entry["key"]]))
            for id_ref, entry in index_entries
        ]
        block = index_block(resolved) + "\n"
        pieces.append(block)
        cursor += len(block.encode("iso-8859-1"))
    tail = '<indexListOffset>{0}</indexListOffset>\n'.format(index_start)
    pieces.append(tail)
    pieces.append(MZML_TAIL)
    return "".join(pieces), offsets


def write_indexed_gzip(path, chunks, write_index=True):
    """Write ``(key, text)`` chunks as a GSGW indexed gzip file."""
    from pymzml.utils.GSGW import GSGW

    writer = GSGW(
        output_path=str(path),
        max_idx=max(16, len(chunks) + 4),
        max_idx_len=8,
        max_offset_len=8,
    )
    for key, text in chunks:
        writer.add_data(text.encode("iso-8859-1"), key)
    if write_index:
        writer.write_index()
    writer.close()
    return str(path)


def spectrum_chunks(document, spectra):
    """Split a document so each spectrum is one block, plus Head and tail.

    The ``Head`` block ends exactly at ``<spectrumList ...>`` and the element
    blocks start with the newline that separates them, matching the behaviour
    of :func:`pymzml.utils.utils.index_gzip`.
    """
    chunks = []
    first = document.index("<spectrum ")
    chunks.append(("Head", document[:first]))
    cursor = first
    for position, spectrum in enumerate(spectra):
        element = spectrum["xml"]
        start = document.index(element, cursor)
        if position + 1 < len(spectra):
            following = document.index(spectra[position + 1]["xml"], start)
            chunks.append((spectrum["key"], document[cursor:following]))
            cursor = following
        else:
            end = document.index("</spectrumList>", start)
            chunks.append((spectrum["key"], document[cursor:end]))
            cursor = end
    chunks.append(("tail", document[cursor:]))
    return chunks


def split_array_chunks(document, spectra, split_key):
    """Split one spectrum so its first binary array straddles two blocks."""
    chunks = spectrum_chunks(document, spectra)
    result = []
    for key, text in chunks:
        if key != split_key:
            result.append((key, text))
            continue
        open_tag = "<binary>"
        close_tag = "</binary>"
        start = text.index(open_tag) + len(open_tag)
        end = text.index(close_tag, start)
        middle = (start + end) // 2
        result.append((key, text[:middle]))
        result.append(("split-{0}".format(key), text[middle:]))
    return result


def header_index_slots(path):
    """Return (index_offset, idx_len, offset_len, [(key, data_offset)]) for a GSGW file."""
    import struct

    with open(path, "rb") as handle:
        blob = handle.read()
    if blob[:2] != b"\x1f\x8b" or not (blob[3] & 16):
        raise ValueError("not a GSGW indexed gzip file")
    cursor = 10
    if blob[cursor : cursor + 3] != b"FU\x01":
        raise ValueError("missing GS index magic")
    cursor += 3
    idx_len, offset_len = struct.unpack("<BB", blob[cursor : cursor + 2])
    cursor += 2
    slots = []
    while cursor + idx_len + offset_len <= len(blob):
        raw_key = blob[cursor : cursor + idx_len]
        raw_offset = blob[cursor + idx_len : cursor + idx_len + offset_len]
        key_text = raw_key.decode("latin-1").strip("\xac").strip("\x00")
        if not key_text or raw_offset == b"\x01" * offset_len:
            break
        try:
            key = int(key_text)
        except ValueError:
            key = key_text
        slots.append((key, cursor + idx_len, raw_offset))
        cursor += idx_len + offset_len
    return slots


def tamper_header_offsets(path, mapping):
    """Rewrite GSGW header offsets for the given keys (in-place)."""
    slots = header_index_slots(path)
    with open(path, "rb+") as handle:
        for key, data_offset in mapping.items():
            slot = next(item for item in slots if item[0] == key)
            _, cursor, raw = slot
            width = len(raw)
            new = str(data_offset).encode("latin-1")
            if len(new) > width:
                raise ValueError("offset does not fit header slot")
            handle.seek(cursor)
            handle.write(b"\xac" * (width - len(new)) + new)
    return path
