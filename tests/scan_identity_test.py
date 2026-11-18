#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test cases for the stable scan identity of spectra and the explicit
query entry points of pymzml.run.Reader.

Spectrum native ids may contain controllerType, controllerNumber and
scan (e.g. 'controllerType=0 controllerNumber=1 scan=1') next to the
sequential positional index. The full native id is the stable identity
of a spectrum, while the local scan number and the positional index are
distinct pieces of information. Spectra with the same local numbers
(e.g. scan=1 of controller 1 and controller 2) must not be merged and
random access as well as sequential reading of the same target must
resolve to the same raw spectrum.
"""
import os
import sys
import tempfile
import unittest

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import pymzml.run as run
from pymzml.spec import Spectrum
import test_file_paths


MULTI_CONTROLLER_SPECTRA = [
    ("controllerType=0 controllerNumber=1 scan=1", 111.0),
    ("controllerType=0 controllerNumber=1 scan=2", 112.0),
    ("controllerType=0 controllerNumber=2 scan=1", 211.0),
    ("controllerType=0 controllerNumber=2 scan=2", 212.0),
]


def build_multi_controller_mzml(path):
    """
    Write a minimal indexed mzML file with two controllers, where both
    controllers use the same local scan numbers (scan=1 and scan=2).
    Each spectrum carries a distinct total ion current so the identity
    of the raw spectrum can be verified.
    """
    header = (
        '<?xml version="1.0" encoding="ISO-8859-1"?>\n'
        '<indexedmzML xmlns="http://psi.hupo.org/ms/mzml" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://psi.hupo.org/ms/mzml http://psidev.info/files/ms/mzML/xsd/mzML1.1.1_idx.xsd">\n'
        '  <mzML xmlns="http://psi.hupo.org/ms/mzml" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://psi.hupo.org/ms/mzML http://psidev.info/files/ms/mzML/xsd/mzML1.1.0.xsd" id="multi_controller" version="1.1.0">\n'
        '    <cvList count="1">\n'
        '      <cv id="MS" fullName="Proteomics Standards Initiative Mass Spectrometry Ontology" version="4.1.79" URI="http://psidev.cvs.sourceforge.net/*checkout*/psidev/psi/psi-ms/mzML/controlledVocabulary/psi-ms.obo"/>\n'
        '    </cvList>\n'
        '    <fileDescription>\n'
        '      <fileContent>\n'
        '        <cvParam cvRef="MS" accession="MS:1000579" name="MS1 spectrum" value=""/>\n'
        '      </fileContent>\n'
        '    </fileDescription>\n'
        '    <run id="multi_controller_run">\n'
        '      <spectrumList count="{0}" defaultDataProcessingRef="dp1">\n'.format(
            len(MULTI_CONTROLLER_SPECTRA)
        )
    )
    body_parts = []
    offsets = []
    cursor = len(header.encode("latin-1"))
    for position, (native_id, tic) in enumerate(MULTI_CONTROLLER_SPECTRA):
        spec_xml = (
            '<spectrum index="{0}" id="{1}" defaultArrayLength="0">\n'
            '          <cvParam cvRef="MS" accession="MS:1000511" name="ms level" value="1"/>\n'
            '          <cvParam cvRef="MS" accession="MS:1000285" name="total ion current" value="{2}"/>\n'
            '        </spectrum>\n'.format(position, native_id, tic)
        )
        offsets.append((native_id, cursor))
        body_parts.append(spec_xml)
        cursor += len(spec_xml.encode("latin-1"))
    mid = "      </spectrumList>\n" "    </run>\n" "  </mzML>\n"
    index_entries = "".join(
        '      <offset idRef="{0}">{1}</offset>\n'.format(native_id, offset)
        for native_id, offset in offsets
    )
    index_block = (
        '  <indexList count="1">\n'
        '    <index name="spectrum">\n'
        + index_entries
        + "    </index>\n"
        "  </indexList>\n"
    )
    index_list_offset = cursor + len(mid.encode("latin-1"))
    footer = (
        "  <indexListOffset>{0}</indexListOffset>\n"
        "  <fileChecksum>0000000000000000000000000000000000000000</fileChecksum>\n"
        "</indexedmzML>\n".format(index_list_offset)
    )
    content = header + "".join(body_parts) + mid + index_block + footer
    with open(path, "wb") as fh:
        fh.write(content.encode("latin-1"))
    return offsets


class ScanIdentityTest(unittest.TestCase):
    """ """

    @classmethod
    def setUpClass(cls):
        cls.example_mzml = test_file_paths.paths[0]
        fd, cls.multi_controller_file = tempfile.mkstemp(suffix=".mzML")
        os.close(fd)
        cls.offsets = build_multi_controller_mzml(cls.multi_controller_file)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.multi_controller_file):
            os.remove(cls.multi_controller_file)

    def setUp(self):
        self.multi_reader = run.Reader(self.multi_controller_file)

    def tearDown(self):
        self.multi_reader.close()

    def test_native_id_is_stable_identity(self):
        """
        The full native id is exposed separately from the local scan
        number (ID) and the positional index.
        """
        reader = run.Reader(self.example_mzml)
        spectrum = reader.get_spectrum_by_native_id(
            "controllerType=0 controllerNumber=1 scan=1"
        )
        self.assertEqual(
            spectrum.native_id, "controllerType=0 controllerNumber=1 scan=1"
        )
        # local number of the native id
        self.assertEqual(spectrum.ID, 1)
        # positional index in the spectrumList
        self.assertEqual(spectrum.index, 0)
        self.assertEqual(
            spectrum.id_dict,
            {"controllerType": 0, "controllerNumber": 1, "scan": 1},
        )
        reader.close()

    def test_native_id_and_index_are_distinct(self):
        """
        The positional index and the local scan number of the native id
        are explicitly distinguished.
        """
        reader = run.Reader(self.example_mzml)
        # scans start at 1, positional indices start at 0
        spectrum = reader.get_spectrum_by_index(0)
        self.assertEqual(spectrum.index, 0)
        self.assertEqual(spectrum.ID, 1)
        self.assertEqual(
            spectrum.native_id, "controllerType=0 controllerNumber=1 scan=1"
        )
        spectrum = reader.get_spectrum_by_index(10)
        self.assertEqual(spectrum.index, 10)
        self.assertEqual(spectrum.ID, 11)
        self.assertEqual(
            spectrum.native_id, "controllerType=0 controllerNumber=1 scan=11"
        )
        reader.close()

    def test_same_local_numbers_are_not_merged(self):
        """
        Spectra with identical local scan numbers but different
        controllers keep their stable identity and are not merged.
        """
        spec_ctrl_1 = self.multi_reader.get_spectrum_by_native_id(
            "controllerType=0 controllerNumber=1 scan=1"
        )
        spec_ctrl_2 = self.multi_reader.get_spectrum_by_native_id(
            "controllerType=0 controllerNumber=2 scan=1"
        )
        self.assertIsInstance(spec_ctrl_1, Spectrum)
        self.assertIsInstance(spec_ctrl_2, Spectrum)
        # same local scan number ...
        self.assertEqual(spec_ctrl_1.ID, 1)
        self.assertEqual(spec_ctrl_2.ID, 1)
        # ... but distinct stable identities and distinct raw content
        self.assertNotEqual(spec_ctrl_1.native_id, spec_ctrl_2.native_id)
        self.assertEqual(spec_ctrl_1.TIC, 111.0)
        self.assertEqual(spec_ctrl_2.TIC, 211.0)
        # both native ids are indexed separately
        offset_dict = self.multi_reader.info["offset_dict"]
        self.assertIn("controllerType=0 controllerNumber=1 scan=1", offset_dict)
        self.assertIn("controllerType=0 controllerNumber=2 scan=1", offset_dict)
        # the ambiguous local number is not kept as a merging alias
        self.assertNotIn(1, offset_dict)
        self.assertNotIn(2, offset_dict)

    def test_ambiguous_local_number_raises(self):
        """
        Looking up an ambiguous local scan number by integer fails
        loudly instead of silently returning a merged spectrum.
        """
        with self.assertRaises(Exception):
            self.multi_reader[1]

    def test_unambiguous_local_number_still_works(self):
        """
        Files with unique scan numbers keep the legacy integer access.
        """
        reader = run.Reader(self.example_mzml)
        spectrum = reader[1]
        self.assertEqual(spectrum.ID, 1)
        self.assertEqual(
            spectrum.native_id, "controllerType=0 controllerNumber=1 scan=1"
        )
        reader.close()

    def test_get_spectrum_by_index(self):
        """
        The positional index resolves spectra in file order.
        """
        for position, (native_id, tic) in enumerate(MULTI_CONTROLLER_SPECTRA):
            spectrum = self.multi_reader.get_spectrum_by_index(position)
            self.assertEqual(spectrum.native_id, native_id)
            self.assertEqual(spectrum.index, position)
            self.assertEqual(spectrum.TIC, tic)

    def test_get_spectrum_by_index_out_of_range(self):
        with self.assertRaises(IndexError):
            self.multi_reader.get_spectrum_by_index(len(MULTI_CONTROLLER_SPECTRA))
        with self.assertRaises(IndexError):
            self.multi_reader.get_spectrum_by_index(-1)

    def test_random_and_sequential_access_match(self):
        """
        Random access (by native id and by positional index) and
        sequential iteration resolve to the same raw spectra.
        """
        sequential_spectra = []
        for spectrum in self.multi_reader:
            sequential_spectra.append(
                (spectrum.native_id, spectrum.index, spectrum.TIC)
            )
        self.assertEqual(len(sequential_spectra), len(MULTI_CONTROLLER_SPECTRA))
        for position, (native_id, index, tic) in enumerate(sequential_spectra):
            self.assertEqual(position, index)
            by_index = self.multi_reader.get_spectrum_by_index(position)
            by_native_id = self.multi_reader.get_spectrum_by_native_id(native_id)
            # same target via random access ...
            self.assertEqual(by_index.native_id, native_id)
            self.assertEqual(by_native_id.native_id, native_id)
            # ... is verifiably the same raw spectrum
            self.assertEqual(by_index.TIC, tic)
            self.assertEqual(by_native_id.TIC, tic)
            self.assertEqual(by_index.to_string(), by_native_id.to_string())

    def test_random_and_sequential_access_match_example_file(self):
        """
        Random access and sequential iteration over example.mzML
        resolve to the same raw spectra.
        """
        reader = run.Reader(self.example_mzml)
        sequential_ids = []
        for spectrum in reader:
            sequential_ids.append((spectrum.native_id, spectrum.index))
        reader.close()
        reader = run.Reader(self.example_mzml)
        for position, (native_id, index) in enumerate(sequential_ids):
            by_index = reader.get_spectrum_by_index(position)
            self.assertEqual(by_index.native_id, native_id)
            self.assertEqual(by_index.index, index)
            by_native_id = reader.get_spectrum_by_native_id(native_id)
            self.assertEqual(by_native_id.native_id, native_id)
            # same raw spectrum, independent of the access route
            self.assertEqual(by_index.ID, by_native_id.ID)
            self.assertEqual(by_index.TIC, by_native_id.TIC)
            self.assertEqual(list(by_index.mz), list(by_native_id.mz))
            self.assertEqual(list(by_index.i), list(by_native_id.i))
        reader.close()


if __name__ == "__main__":
    unittest.main(verbosity=3)
