# -*- coding: utf-8 -*-
"""Splitting of multi-allelic records into bi-allelic records

The central entry point is :py:func:`split_multiallelic`.  It decomposes a
:py:class:`vcfpy.Record` with one or more alternative alleles into one
bi-allelic record per alternative allele.  All allele-dependent ``INFO`` and
``FORMAT`` values are remapped according to the ``Number`` definition of the
field in the header:

- ``Number=A`` values are reduced to the entry of the target allele,
- ``Number=R`` values are reduced to the reference and the target allele
  entry,
- ``Number=G`` values are reduced to the genotype combinations over
  ``{REF, target allele}``, using the actual ploidy of each sample's genotype
  (falling back to the ploidy implied by the value list length),
- all other values are copied unchanged.

Genotypes are remapped allele by allele: the reference allele becomes ``0``
and the target allele becomes ``1``.  Alleles pointing to *other* alternative
alleles cannot be expressed in a bi-allelic row.  They are **not** rewritten
to the reference allele; instead, the corresponding genotype position becomes
missing (``.``) and the original allele indices are listed separately in a
dedicated ``FORMAT`` field (``OA`` by default) so no information is lost.
Partially missing genotypes (e.g. ``./1``) and partially missing value lists
(shorter lists or ``.`` entries) are preserved as missing.
"""

import copy
import math
from typing import Any, Iterator

from vcfpy.header import Header
from vcfpy.record import Call, Record, UnparsedCall

__author__ = "Manuel Holtgrewe <manuel.holtgrewe@bihealth.de>"

#: Default FORMAT key for listing the original indices of alleles that could
#: not be represented in a bi-allelic record
OTHER_ALLELES_KEY = "OA"


def _num_genotypes(num_alleles: int, ploidy: int) -> int:
    """Return the number of distinct genotypes for ``num_alleles`` alleles and
    the given ``ploidy``.
    """
    return math.comb(num_alleles + ploidy - 1, ploidy)


def _genotype_index(alleles: tuple[int, ...]) -> int:
    """Return the index of the genotype ``alleles`` in the VCF ``Number=G``
    ordering.

    ``alleles`` must be a non-decreasing tuple of allele indices.  The
    ordering follows the VCF specification (combinatorial number system),
    e.g. ``0/0, 0/1, 1/1, 0/2, 1/2, 2/2, ...`` for diploid genotypes.
    """
    result = 0
    for pos, allele in enumerate(alleles, start=1):
        result += math.comb(allele + pos - 1, pos)
    return result


def _infer_ploidy(num_alleles: int, count: int) -> int | None:
    """Infer the ploidy from the number of ``Number=G`` values, return
    ``None`` if no ploidy in 1..8 explains ``count``.
    """
    for ploidy in range(1, 9):
        if _num_genotypes(num_alleles, ploidy) == count:
            return ploidy
    return None


def _remap_number_a(value: Any, alt_idx: int) -> Any:
    """Remap a ``Number=A`` value to the target allele ``alt_idx`` (1-based).

    The result is a one-element list; missing values and lists that are too
    short yield ``None`` entries.
    """
    if not isinstance(value, list):
        return copy.deepcopy(value)  # scalar, nothing to subset
    if not value:
        return []  # completely missing stays missing
    pos = alt_idx - 1
    return [value[pos] if pos < len(value) else None]


def _remap_number_r(value: Any, alt_idx: int) -> Any:
    """Remap a ``Number=R`` value to reference and target allele ``alt_idx``
    (1-based).

    Missing values and lists that are too short yield ``None`` entries.
    """
    if not isinstance(value, list):
        return copy.deepcopy(value)  # scalar, nothing to subset
    if not value:
        return []  # completely missing stays missing
    return [
        value[0] if len(value) > 0 else None,
        value[alt_idx] if alt_idx < len(value) else None,
    ]


def _remap_number_g(value: Any, alt_idx: int, ploidy: int | None) -> Any:
    """Remap a ``Number=G`` value to the genotypes over reference and target
    allele ``alt_idx`` (1-based) for the given ``ploidy``.

    Genotype combinations involving other alleles are dropped; value lists
    that are too short yield ``None`` entries.  If ``ploidy`` is ``None`` the
    value cannot be remapped and is copied unchanged.
    """
    if not isinstance(value, list) or not value or ploidy is None:
        return copy.deepcopy(value)
    result = []
    for num_alt in range(ploidy + 1):
        genotype = (0,) * (ploidy - num_alt) + (alt_idx,) * num_alt
        idx = _genotype_index(genotype)
        result.append(value[idx] if idx < len(value) else None)
    return result


def _remap_gt(call: Call, alt_idx: int) -> tuple[Any, list[int]]:
    """Remap the genotype of ``call`` to the bi-allelic row of ``alt_idx``.

    Returns the new GT value and the list of original allele indices that
    cannot be represented in the bi-allelic row (in genotype order).  Other
    alternative alleles become missing alleles; they are never rewritten to
    the reference allele.  Partially missing genotypes keep their missing
    alleles.
    """
    if call.gt_alleles is None:
        return copy.deepcopy(call.data.get("GT")), []
    other: list[int] = []
    new_alleles: list[int | None] = []
    for allele in call.gt_alleles:
        if allele is None:
            new_alleles.append(None)  # partially missing stays missing
        elif allele == 0:
            new_alleles.append(0)
        elif allele == alt_idx:
            new_alleles.append(1)
        else:
            other.append(allele)  # keep original allele index
            new_alleles.append(None)
    sep = call.gt_phase_char
    return sep.join("." if a is None else str(a) for a in new_alleles), other


def _remap_info_value(header: Header, key: str, value: Any, alt_idx: int, num_alleles: int) -> Any:
    """Remap one INFO value according to the header ``Number`` definition."""
    number = header.get_info_field_info(key).number
    if number == "A":
        return _remap_number_a(value, alt_idx)
    elif number == "R":
        return _remap_number_r(value, alt_idx)
    elif number == "G":
        # INFO has no per-sample ploidy; infer it from the value list length
        ploidy = _infer_ploidy(num_alleles, len(value)) if isinstance(value, list) else None
        return _remap_number_g(value, alt_idx, ploidy)
    else:
        return copy.deepcopy(value)


def _remap_call_data(
    header: Header, record: Record, call: Call, alt_idx: int, num_alleles: int
) -> tuple[dict[str, Any], list[int]]:
    """Remap all FORMAT values of ``call``, return new data dict and the list
    of original allele indices that could not be represented.
    """
    other: list[int] = []
    data: dict[str, Any] = {}
    for key in record.FORMAT:
        value = call.data.get(key)
        if key == "GT":
            data["GT"], other = _remap_gt(call, alt_idx)
            continue
        number = header.get_format_field_info(key).number
        if number == "A":
            data[key] = _remap_number_a(value, alt_idx)
        elif number == "R":
            data[key] = _remap_number_r(value, alt_idx)
        elif number == "G":
            # use the sample's actual ploidy, fall back to the ploidy implied
            # by the number of values
            ploidy = call.ploidy
            if ploidy is None and isinstance(value, list):
                ploidy = _infer_ploidy(num_alleles, len(value))
            data[key] = _remap_number_g(value, alt_idx, ploidy)
        else:
            data[key] = copy.deepcopy(value)
    return data, other


def split_multiallelic(
    record: Record,
    header: Header,
    *,
    other_alleles_key: str | None = OTHER_ALLELES_KEY,
    allele_index_info: str | None = None,
) -> Iterator[Record]:
    """Split ``record`` into one bi-allelic :py:class:`vcfpy.Record` per
    alternative allele.

    :param record: The record to split (may also be bi-allelic already, in
        which case a single remapped record is yielded).
    :param header: The VCF header, used for the ``Number`` definitions of the
        ``INFO`` and ``FORMAT`` fields.
    :param other_alleles_key: Key of the ``FORMAT`` field that lists the
        original allele indices that cannot be represented in a bi-allelic
        row (added only if at least one call requires it); use ``None`` to
        disable.
    :param allele_index_info: Optional key of an ``INFO`` field to store the
        original 1-based alternative allele index of each output record in.
    :return: Iterator over the bi-allelic records, in the order of the
        alternative alleles of ``record``.
    """
    num_alleles = len(record.ALT) + 1
    for alt_idx in range(1, len(record.ALT) + 1):
        # Remap INFO values according to their Number definition
        info = {
            key: _remap_info_value(header, key, value, alt_idx, num_alleles)
            for key, value in record.INFO.items()
        }
        if allele_index_info is not None:
            info[allele_index_info] = alt_idx
        # Remap the calls
        format_keys = list(record.FORMAT)
        calls: list[Call] = []
        others: list[list[int]] = []
        for call in record.calls:
            if isinstance(call, UnparsedCall):
                raise ValueError(
                    "Cannot split record with unparsed calls; parse the genotype fields first"
                )
            data, other = _remap_call_data(header, record, call, alt_idx, num_alleles)
            calls.append(Call(call.sample, data))
            others.append(other)
        # List allele contributions that cannot be expressed in the bi-allelic
        # row separately instead of rewriting them to the reference allele
        if other_alleles_key is not None and any(others):
            format_keys.append(other_alleles_key)
            for call, other in zip(calls, others, strict=True):
                call.data[other_alleles_key] = other if other else []
        yield Record(
            CHROM=record.CHROM,
            POS=record.POS,
            ID=list(record.ID),
            REF=record.REF,
            ALT=[copy.deepcopy(record.ALT[alt_idx - 1])],
            QUAL=record.QUAL,
            FILTER=list(record.FILTER),
            INFO=info,
            FORMAT=format_keys if record.FORMAT else None,
            calls=calls if record.FORMAT else None,
        )
