# -*- coding: utf-8 -*-

from vcfpy.exceptions import (
    HeaderNotFound,
    IncorrectVCFFormat,
    InvalidHeaderException,
    InvalidRecordException,
    VCFPyException,
)
from vcfpy.header import (
    AltAlleleHeaderLine,
    CompoundHeaderLine,
    ContigHeaderLine,
    FieldInfo,
    FilterHeaderLine,
    FormatHeaderLine,
    Header,
    HeaderLine,
    InfoHeaderLine,
    MetaHeaderLine,
    PedigreeHeaderLine,
    SampleHeaderLine,
    SamplesInfos,
    SimpleHeaderLine,
    header_without_lines,
)
from vcfpy.reader import Reader
from vcfpy.record import (
    BND,
    DEL,
    FIVE_PRIME,
    FORWARD,
    HET,
    HOM_ALT,
    HOM_REF,
    INDEL,
    INS,
    MIXED,
    MNV,
    REVERSE,
    SNV,
    SV,
    SYMBOLIC,
    THREE_PRIME,
    AltRecord,
    BreakEnd,
    Call,
    Record,
    SingleBreakEnd,
    Substitution,
    SymbolicAllele,
    UnparsedCall,
)
from vcfpy.version import __version__
from vcfpy.writer import Writer

#: Names provided by :py:mod:`vcfpy.merge`, imported lazily so that
#: ``python -m vcfpy.merge`` does not find an already-imported submodule.
_MERGE_EXPORTS = {
    "MergeError",
    "MergeConflictError",
    "MergeConflict",
    "MergePlan",
    "FieldRename",
    "RenameCollection",
    "plan_merge",
    "merge_headers",
    "merge_records",
    "merge_vcf_files",
}


def __getattr__(name: str):
    if name in _MERGE_EXPORTS:
        from vcfpy import merge

        value = getattr(merge, name)
        globals()[name] = value
        return value
    raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))


def __dir__():
    return sorted(set(globals()) | _MERGE_EXPORTS)

__all__ = [
    "VCFPyException",
    "IncorrectVCFFormat",
    "InvalidHeaderException",
    "InvalidRecordException",
    "HeaderNotFound",
    "Header",
    "MetaHeaderLine",
    "SimpleHeaderLine",
    "CompoundHeaderLine",
    "InfoHeaderLine",
    "FilterHeaderLine",
    "FormatHeaderLine",
    "ContigHeaderLine",
    "AltAlleleHeaderLine",
    "PedigreeHeaderLine",
    "SampleHeaderLine",
    "FieldInfo",
    "SamplesInfos",
    "header_without_lines",
    "Reader",
    "HeaderLine",
    "Writer",
    "MergeError",
    "MergeConflictError",
    "MergeConflict",
    "MergePlan",
    "FieldRename",
    "RenameCollection",
    "plan_merge",
    "merge_headers",
    "merge_records",
    "merge_vcf_files",
    "Record",
    "Call",
    "UnparsedCall",
    "AltRecord",
    "Substitution",
    "SymbolicAllele",
    "BreakEnd",
    "SingleBreakEnd",
    "HOM_REF",
    "HET",
    "HOM_ALT",
    "SNV",
    "MNV",
    "INDEL",
    "INS",
    "DEL",
    "SV",
    "BND",
    "SYMBOLIC",
    "MIXED",
    "FORWARD",
    "REVERSE",
    "FIVE_PRIME",
    "THREE_PRIME",
    "__version__",
]
