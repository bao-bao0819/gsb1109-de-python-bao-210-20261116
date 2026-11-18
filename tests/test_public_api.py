# -*- coding: utf-8 -*-
"""Test that the public API classes are available from the ``vcfpy`` package"""

import vcfpy


def test_from_exceptions():
    assert vcfpy.VCFPyException
    assert vcfpy.InvalidHeaderException
    assert vcfpy.InvalidRecordException
    assert vcfpy.IncorrectVCFFormat
    assert vcfpy.HeaderNotFound


def test_from_header():
    assert vcfpy.Header
    assert vcfpy.HeaderLine
    assert vcfpy.SimpleHeaderLine
    assert vcfpy.AltAlleleHeaderLine
    assert vcfpy.ContigHeaderLine
    assert vcfpy.FilterHeaderLine
    assert vcfpy.MetaHeaderLine
    assert vcfpy.PedigreeHeaderLine
    assert vcfpy.SampleHeaderLine
    assert vcfpy.CompoundHeaderLine
    assert vcfpy.InfoHeaderLine
    assert vcfpy.FormatHeaderLine
    assert vcfpy.SamplesInfos


def test_from_record():
    assert vcfpy.Record
    assert vcfpy.UnparsedCall
    assert vcfpy.Call
    assert vcfpy.AltRecord
    assert vcfpy.Substitution
    assert vcfpy.SV
    assert vcfpy.BreakEnd
    assert vcfpy.SymbolicAllele


def test_record_constants():
    assert vcfpy.SNV
    assert vcfpy.MNV
    assert vcfpy.DEL
    assert vcfpy.INS
    assert vcfpy.INDEL
    assert vcfpy.SV
    assert vcfpy.BND
    assert vcfpy.SYMBOLIC
    assert vcfpy.MIXED

    assert vcfpy.HOM_REF is not None
    assert vcfpy.HET
    assert vcfpy.HOM_ALT


def test_from_reader():
    assert vcfpy.Reader


def test_from_writer():
    assert vcfpy.Writer


def test_from_haplotype():
    assert vcfpy.reconstruct_haplotypes
    assert vcfpy.HaplotypeFragment
    assert vcfpy.SampleHaplotypes
    assert vcfpy.Provenance
    assert vcfpy.CrossBlockAnchor
    assert vcfpy.HaplotypeConflict
    assert vcfpy.HaplotypeConflictError
    assert vcfpy.UnknownRegion
    assert vcfpy.PhaseBlock
    assert vcfpy.UNKNOWN_SLOT
    assert vcfpy.OP_MATCH
    assert vcfpy.OP_SUBST
    assert vcfpy.OP_INS
    assert vcfpy.OP_DEL
