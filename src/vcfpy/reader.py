# -*- coding: utf-8 -*-
"""Parsing of VCF files from ``file``-like objects"""

import gzip
import os
import pathlib
import typing
from io import TextIOWrapper

from vcfpy import parser
from vcfpy.record import Record
from vcfpy.tabix import TabixFile, TabixInterval

__author__ = "Manuel Holtgrewe <manuel.holtgrewe@bihealth.de>"

#: Alias for the normalized interval type used by multi-interval fetches.
Interval = TabixInterval


class IntervalHit(Record):
    """A VCF record together with the query intervals that selected it.

    A hit is a normal :py:class:`~vcfpy.Record` for the selected source
    line, augmented with the query intervals that matched it.  Hits are
    unique per source record occurrence, identified by the BGZF virtual
    offset of the line.  Duplicate lines in the input therefore remain
    separate hits.
    """

    def __init__(
        self,
        record: Record,
        intervals: typing.Iterable[object],
        source_offset: int,
        source_path: str | None = None,
    ):
        super().__init__(
            record.CHROM,
            record.POS,
            record.ID,
            record.REF,
            record.ALT,
            record.QUAL,
            record.FILTER,
            record.INFO,
            record.FORMAT,
            record.calls,
        )
        self.begin = record.begin
        self.end = record.end
        self._record = record
        #: Query intervals (as supplied by the caller) that selected this occurrence.
        self.intervals: list[object] = list(intervals)
        #: Virtual BGZF offset of the source line.
        self.source_offset = source_offset
        #: Path of the file that contains the source line.
        self.source_path = source_path

    @property
    def record(self) -> Record:
        """The plain parsed record for the selected source line."""
        return self._record

    @property
    def source_identity(self) -> tuple[str | None, int]:
        """Identity of the source record occurrence."""
        return (self.source_path, self.source_offset)

    @property
    def offset(self) -> int:
        """Alias for :py:attr:`source_offset`."""
        return self.source_offset

    @property
    def interval(self):
        """Return the sole query interval, or all intervals if there are several."""
        return self.intervals[0] if len(self.intervals) == 1 else self.intervals

    def _core(self) -> tuple:
        return (
            self.CHROM,
            self.POS,
            self.ID,
            self.REF,
            self.ALT,
            self.QUAL,
            self.FILTER,
            self.INFO,
            self.FORMAT,
            self.calls,
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, IntervalHit):
            # Occurrences are distinct by source identity even when the record
            # content is byte-for-byte identical.
            return self._core() == other._core() and self.source_identity == other.source_identity
        if isinstance(other, Record):
            return self._core() == (
                other.CHROM,
                other.POS,
                other.ID,
                other.REF,
                other.ALT,
                other.QUAL,
                other.FILTER,
                other.INFO,
                other.FORMAT,
                other.calls,
            )
        return NotImplemented

    def __hash__(self) -> int:
        # Records contain lists and are not hashable themselves; use the
        # stable textual form so equal hits hash equally.
        return hash(str(self))

    def __repr__(self):
        return f"IntervalHit({self._core()!r}, intervals={self.intervals!r})"


def _fetch_many(reader: "Reader", intervals: typing.Iterable[object]) -> list["IntervalHit"]:
    """Fetch, parse, and merge hits for several intervals."""
    if isinstance(intervals, (str, dict, TabixInterval)):
        intervals = [intervals]
    reader._ensure_tabix_file()
    hits = []
    for tabix_hit in reader.tabix_file.fetch_many(intervals):
        hits.append(
            IntervalHit(
                record=reader.parser.parse_line(tabix_hit.line),
                intervals=tabix_hit.intervals,
                source_offset=tabix_hit.virtual_offset,
                source_path=reader.path,
            )
        )
    return hits


class Reader:
    """Class for parsing of files from ``file``-like objects

    Instead of using the constructor, use the class methods
    :py:meth:`~Reader.from_stream` and
    :py:meth:`~Reader.from_path`.

    On construction, the header will be read from the file which can cause
    problems.  After construction, :py:class:`~Reader` can be used as
    an iterable of :py:class:`~vcfpy.record.Record`.

    :raises: :py:class:`~vcfpy.exceptions.InvalidHeaderException` in the case
        of problems reading the header

    .. note::
        It is important to note that the ``header`` member is used during
        the parsing of the file.  **If you need a modified version then
        create a copy, e.g., using :py:method:`~vcfpy.header.Header.copy`**.

    .. note::
        If you use the ``parsed_samples`` feature and you write out
        records then you must not change the ``FORMAT`` of the record.
    """

    @classmethod
    def from_stream(
        cls,
        stream: TextIOWrapper,
        path: pathlib.Path | str | None = None,
        tabix_path: pathlib.Path | str | None = None,
        record_checks: list[typing.Literal["INFO", "FORMAT"]] | None = None,
        parsed_samples: list[str] | None = None,
    ):
        """Create new :py:class:`Reader` from file

        .. note::
            If you use the ``parsed_samples`` feature and you write out
            records then you must not change the ``FORMAT`` of the record.

        :param stream: ``file``-like object to read from
        :param path: optional string with path to store (for display only)
        :param list record_checks: record checks to perform, can contain
            'INFO' and 'FORMAT'
        :param list parsed_samples: ``list`` of ``str`` values with names of
            samples to parse call information for (for speedup); leave to
            ``None`` for ignoring
        """
        record_checks = record_checks or []
        if tabix_path and not path:  # pragma: no cover
            raise ValueError("Must give path if tabix_path is given")
        return Reader(
            stream=stream,
            path=path,
            tabix_path=tabix_path,
            record_checks=record_checks,
            parsed_samples=parsed_samples,
        )

    @classmethod
    def from_path(
        cls,
        path: pathlib.Path | str,
        tabix_path: pathlib.Path | str | None = None,
        record_checks: list[typing.Literal["INFO", "FORMAT"]] | None = None,
        parsed_samples: list[str] | None = None,
    ):
        """Create new :py:class:`Reader` from path

        .. note::
            If you use the ``parsed_samples`` feature and you write out
            records then you must not change the ``FORMAT`` of the record.

        :param path: the path to load from (converted to ``str`` for
            compatibility with ``path.py``)
        :param tabix_path: optional string with path to TBI index,
            automatic inferral from ``path`` will be tried on the fly
            if not given
        :param list record_checks: record checks to perform, can contain
            'INFO' and 'FORMAT'
        """
        record_checks = record_checks or []
        path = str(path)
        if path.endswith(".gz") or path.endswith(".bgz"):
            f = gzip.open(path, "rt")
            if not tabix_path:
                tabix_path = path + ".tbi"
                if not os.path.exists(tabix_path):
                    tabix_path = None  # guessing path failed
        else:
            f = open(path, "rt")
        return cls.from_stream(
            stream=f,
            path=path,
            tabix_path=tabix_path,
            record_checks=record_checks,
            parsed_samples=parsed_samples,
        )

    def __init__(
        self,
        stream: TextIOWrapper,
        path: pathlib.Path | str | None = None,
        tabix_path: pathlib.Path | str | None = None,
        record_checks: typing.Iterable[typing.Literal["FORMAT", "INFO"]] | None = None,
        parsed_samples: list[str] | None = None,
    ):
        #: stream (``file``-like object) to read from
        self.stream = stream
        #: optional ``str`` with the path to the stream
        self.path = None if path is None else str(path)
        #: optional ``str`` with path to tabix file
        self.tabix_path = None if tabix_path is None else str(tabix_path)
        #: checks to perform on records, can contain 'FORMAT' and 'INFO'
        self.record_checks = tuple(record_checks or [])
        #: if set, list of samples to parse for
        self.parsed_samples = parsed_samples
        #: the ``pysam.TabixFile`` used for reading from index bgzip-ed VCF;
        #: constructed on the fly
        self.tabix_file = None
        # the iterator through the Tabix file to use
        self.tabix_iter = None
        #: the parser to use
        self.parser = parser.Parser(stream, self.path, self.record_checks)
        #: the Header
        self.header = self.parser.parse_header(parsed_samples)

    def fetch(self, chrom_or_region, begin: int | None = None, end: int | None = None):
        """Jump to the start position of the given chromosomal position
        and limit iteration to the end position

        A collection of intervals is delegated to :py:meth:`fetch_many`.

        :param str chrom_or_region: name of the chromosome to jump to if
            begin and end are given and a samtools region string otherwise
            (e.g. "chr1:123,456-123,900").
        :param int begin: 0-based begin position (inclusive)
        :param int end: 0-based end position (exclusive)
        """
        if not isinstance(chrom_or_region, str):
            return self.fetch_many(chrom_or_region)
        if begin is not None and end is None:  # pragma: no cover
            raise ValueError("begin and end must both be None or neither")
        self._ensure_tabix_file()
        # jump to the next position
        if begin is None:
            self.tabix_iter = self.tabix_file.fetch(region=chrom_or_region)
        else:
            self.tabix_iter = self.tabix_file.fetch(reference=chrom_or_region, start=begin, end=end)
        return self

    def fetch_many(self, intervals=None, *more_intervals, **kwargs) -> list["IntervalHit"]:
        """Fetch records for several intervals and merge duplicate hits.

        Accepts one iterable of interval specifications, several positional
        specifications, or the ``intervals``/``regions`` keyword.  A record
        can be returned more than once for one query because index chunks
        overlap, and the file itself may contain identical lines.  Hits are
        therefore merged on the identity of the source record occurrence
        (its BGZF virtual offset), not on record content:

        * a line reached through multiple chunks/intervals is returned once,
          with every matching interval retained in ``IntervalHit.intervals``;
        * two identical lines in the file are two occurrences and both are
          returned;
        * results are ordered by source position and intervals are reported
          in a canonical order, so swapping the query intervals does not
          change which occurrence is retained or the output order.

        :param intervals: samtools region strings, ``(reference, begin,
            end)`` tuples, or mappings with ``reference``/``chrom``,
            optional ``start``/``begin`` and ``end``.  ``begin``/``end``
            follow the same convention as :py:meth:`fetch`.
        :return: list of :py:class:`IntervalHit` in source order.
        """
        if "regions" in kwargs:
            if intervals is not None:
                raise TypeError("Cannot combine 'intervals' and 'regions'")
            intervals = kwargs.pop("regions")
        if "intervals" in kwargs:  # pragma: no cover - guarded by signature
            if intervals is not None:
                raise TypeError("'intervals' given twice")
            intervals = kwargs.pop("intervals")
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {', '.join(sorted(kwargs))}")
        if more_intervals:
            specs = (intervals,) + more_intervals
        elif intervals is None:
            specs = []
        else:
            specs = intervals
            if isinstance(specs, (str, dict, TabixInterval)):
                specs = [specs]
            elif isinstance(specs, tuple) and (not specs or isinstance(specs[0], (str, dict, TabixInterval))):
                specs = [specs]
        return _fetch_many(self, specs)

    #: Aliases for multi-interval fetching.
    fetch_intervals = fetch_many
    fetch_regions = fetch_many
    fetch_records = fetch_many

    def _ensure_tabix_file(self):
        """Open the tabix file if needed and close a stale one."""
        if self.tabix_file and not self.tabix_file.closed:
            self.tabix_file.close()
        if not self.tabix_file or self.tabix_file.closed:
            if self.path is None:  # pragma: no cover
                raise ValueError("Cannot fetch without path")
            self.tabix_file = TabixFile(filename=self.path, index=self.tabix_path)

    def close(self):
        """Close underlying stream"""
        if self.tabix_file and not self.tabix_file.closed:
            self.tabix_file.close()
        if self.stream:
            self.stream.close()

    def __enter__(self) -> "Reader":
        return self

    def __exit__(self, type_: type[BaseException] | None, value: BaseException | None, traceback: typing.Any) -> None:
        self.close()

    def __iter__(self):
        return self

    def __next__(self):
        """Return next object from file

        :returns:
        :raises: ``vcfpy.exceptions.InvalidRecordException`` in the case of
            problems reading the record
        :raises: ``StopException`` if at end
        """
        if self.tabix_iter:
            return self.parser.parse_line(str(next(self.tabix_iter)))
        else:
            result = self.parser.parse_next_record()
            if result is None:
                raise StopIteration()
            else:
                return result
