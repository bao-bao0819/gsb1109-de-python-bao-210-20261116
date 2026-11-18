# -*- coding: utf-8 -*-
"""Utility functions for working with VCF records

Currently, this module provides :py:func:`recompute_ac_an` for recomputing
the reserved ``INFO`` fields ``AC`` and ``AN`` from genotype calls, e.g.,
after subsetting the samples of a VCF file.
"""

from typing import Iterable

from vcfpy.header import RESERVED_INFO, FieldInfo, Header
from vcfpy.record import Record, UnparsedCall

__author__ = "Manuel Holtgrewe <manuel.holtgrewe@bihealth.de>"


def _info_field_info(header: Header | None, key: str) -> FieldInfo | None:
    """Return :py:class:`~vcfpy.header.FieldInfo` for the INFO field ``key``

    The declaration from ``header`` takes precedence; if the field is not
    declared in the header then the reserved definition from the VCF
    specification is used, if any.
    """
    if header is not None and header.has_header_line("INFO", key):
        return header.get_info_field_info(key)
    return RESERVED_INFO.get(key)


def recompute_ac_an(record: Record, header: Header | None = None, samples: Iterable[str] | None = None) -> bool:
    """Recompute the ``AC`` and ``AN`` entries of ``record.INFO`` from the
    genotypes of the selected samples.

    When keeping only a subset of the samples of a VCF file, the ``AC``
    (allele count) and ``AN`` (total allele number) entries in ``INFO``
    still describe the full cohort.  This function recomputes both fields
    from the genotype (``GT``) calls of the selected samples, honouring the
    actual ploidy of each call.  Samples with partially missing genotypes
    (e.g., ``0/.``) still contribute their called alleles.

    The fields are updated in place and only if they are already present in
    ``record.INFO``.  Their representation follows the header declaration
    (e.g., one ``AC`` value per ALT allele for ``Number=A``, a scalar for
    ``Number=1``); if no declaration is found, the reserved definitions
    from the VCF specification are used.  Fields with declarations that are
    incompatible with allele counts (non-``Integer`` type or unsupported
    ``Number``) are left unchanged.  Other ``INFO`` fields -- in particular
    frequency annotations such as ``AF`` that generally stem from external
    annotation sources -- are never touched.

    If the allele counts cannot be determined from the record (no genotype
    calls for the selected samples, no ``GT`` in ``FORMAT``, or unparsed
    calls), the previous ``AC``/``AN`` values are kept unchanged.  As the
    values are recomputed from the genotypes from scratch on each call,
    repeated application (e.g., after repeated subsetting) is idempotent
    and does not shrink the ``AN`` denominator iteratively.

    :param record: The :py:class:`~vcfpy.record.Record` to update in place.
    :param header: Optional :py:class:`~vcfpy.header.Header` with the
        ``INFO`` field declarations to follow.
    :param samples: Optional iterable with the names of the samples to
        consider; defaults to all calls in ``record``.
    :return: ``True`` if the allele counts could be determined (and the
        fields were updated), ``False`` if the previous values were kept.
    """
    # Collect the calls of the selected samples.
    if samples is None:
        calls = list(record.calls)
    else:
        calls = []
        for name in samples:
            call = record.call_for_sample.get(name)
            if call is None:
                return False  # contribution of sample cannot be determined
            calls.append(call)
    # Cannot recompute without parsed genotype calls.
    if not calls or "GT" not in record.FORMAT:
        return False
    if any(isinstance(call, UnparsedCall) for call in calls):
        return False
    # Count the called alleles; partially missing genotypes contribute
    # their known alleles, fully missing genotypes contribute nothing.
    allele_counts = [0] * len(record.ALT)
    total_called = 0
    for call in calls:
        for allele in call.gt_alleles or []:
            if allele is None:
                continue
            total_called += 1
            if 1 <= allele <= len(record.ALT):
                allele_counts[allele - 1] += 1
    # Update the INFO fields according to their declaration.
    if "AC" in record.INFO:
        _update_ac(record, _info_field_info(header, "AC"), allele_counts)
    if "AN" in record.INFO:
        _update_an(record, _info_field_info(header, "AN"), total_called)
    return True


def _update_ac(record: Record, field_info: FieldInfo | None, allele_counts: list[int]) -> None:
    """Update ``record.INFO["AC"]`` from ``allele_counts`` if the
    declaration in ``field_info`` describes an allele count field.
    """
    if field_info is None or field_info.type != "Integer":
        return  # not declared as a count field, keep previous value
    if field_info.number in ("A", "."):
        record.INFO["AC"] = list(allele_counts)
    elif field_info.number == 1:
        record.INFO["AC"] = allele_counts[0] if allele_counts else 0
    # other declarations are incompatible, keep previous value


def _update_an(record: Record, field_info: FieldInfo | None, total_called: int) -> None:
    """Update ``record.INFO["AN"]`` to ``total_called`` if the declaration
    in ``field_info`` describes the total allele number field.
    """
    if field_info is None or field_info.type != "Integer":
        return  # not declared as a count field, keep previous value
    if field_info.number == 1:
        record.INFO["AN"] = total_called
    # other declarations are incompatible, keep previous value
