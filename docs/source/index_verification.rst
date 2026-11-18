Verifying indexed gzip random access
====================================

Indexed gzip makes spectrum access fast, but it adds three layers that can
silently disagree with each other:

1. the gzip container (members, deflate ends, trailers and byte ranges),
2. the index (the mzML ``indexList`` and/or the GSGW header index),
3. the native identifier interpretation used by :class:`pymzml.run.Reader`.

:func:`pymzml.utils.verify.verify_indexed_gzip` checks all three offline and
returns a :class:`pymzml.utils.verify.VerificationReport`.  A report is
considered successful when it contains no ``error`` findings; ``warning``
findings point at ambiguities such as duplicate trailing integers.

Usage
-----

::

    from pymzml.utils.verify import verify_indexed_gzip

    report = verify_indexed_gzip("run.mzML.idx.gz")
    print(report.format())
    raise SystemExit(0 if report.ok else 1)

The same check is available from the command line::

    python -m pymzml.utils.verify run.mzML.idx.gz
    python -m pymzml.utils.verify run.mzML.idx.gz --query 42 --query TIC
    python -m pymzml.utils.verify reference.mzML.idx.gz --replace candidate.mzML.idx.gz
    python -m pymzml.utils.verify run.mzML.idx.gz --json

What is checked
---------------

- Every gzip member is decompressed with raw deflate; missing end markers,
  truncated trailers and CRC32/ISIZE mismatches are reported with the member
  number and compressed byte range.
- Every element and every ``binaryDataArray`` is mapped to the member(s) that
  contain it.  Elements or arrays crossing a compression boundary are errors,
  because a single-member random read cannot reconstruct them.
- Native ``indexList`` offsets are matched against element start offsets.  An
  offset that falls inside an element is reported as ``misaligned`` with the
  exact delta; an offset that lands on a different element is reported as an
  identity mismatch.
- GSGW header index keys are decompressed exactly like ``GSGR.read_block`` and
  compared with the element they actually address.
- Sequential iteration is compared with the uncompressed document, including
  order, declared count and duplicate keys.
- Random access is exercised for both default and explicit identifiers, then
  repeated in the middle of a sequential read and after reopening the file.
  Inputs with neither a native index nor a GSGW header index are reported as
  ``random_access_unsupported_input`` and only sequential checks run.

Evidence in a finding
---------------------

Every :class:`~pymzml.utils.verify.VerificationFinding` carries:

``index``
    The native index key affected by the finding.
``identity``
    The raw ``id`` attribute, the pymzml index key (``scan=<n>`` when present,
    otherwise the whole identifier), the trailing integer used by
    ``Spectrum.ID``, and the parsed ``key=value`` tokens.
``boundary``
    Byte offsets plus member numbers for both the compressed span and the
    uncompressed span.
``arrays``
    Names of arrays with value-level evidence.
``details``
    The *actual target* of a wrong lookup, the expected element(s), and array
    evidence that records both counts and digests.

Array equality
--------------

Arrays are never compared by length.  Values are compared for plain numeric
arrays, and the decoded bytes are compared for opaque encodings.  When a
lookup returns the wrong element whose arrays happen to have the same length,
the finding contains ``lengths_equal: True`` together with
``values_equal: False`` and differing digests, so an equal-length array is
explicitly shown *not* to be a hit.

Duplicate numeric identifiers
-----------------------------

``Spectrum.ID`` returns the last integer in the ``id`` attribute.  Identifiers
such as ``sample=1 scan=1`` and ``sample=2 scan=1`` therefore share
``Spectrum.ID == 1`` even though pymzml's native index keys them differently.
The verifier reports this as ``identity_trailing_collision``; random access by
the native key still works, but deduplicating spectra by ``Spectrum.ID`` would
collapse two distinct elements.  Duplicate *native* keys are reported as hard
errors with both offsets.

Replaced input
--------------

To check whether an index still describes a newly delivered input, compare a
reference and a candidate::

    from pymzml.utils.verify import compare_indexed_gzip

    report = compare_indexed_gzip("delivered.idx.gz", "replacement.idx.gz")

Changed keys are reported as ``input_native_index_target_changed`` or
``input_header_index_target_changed`` with the reference identity, the
candidate's actual target, and the affected key range in ``summary``.
