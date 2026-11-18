# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from io import BytesIO
from unittest import TestCase
from xml.etree import ElementTree
import base64
import io
import struct

from hwp5 import binmodel
from hwp5 import xmlmodel
from hwp5 import treeop
from hwp5.binmodel import BinData
from hwp5.binmodel import ControlChar
from hwp5.binmodel import PageDef
from hwp5.binmodel import ParaCharShape
from hwp5.binmodel import ParaLineSeg
from hwp5.binmodel import ParaText
from hwp5.binmodel import ParaTextChunks
from hwp5.binmodel import SectionDef
from hwp5.tagids import HWPTAG_PARA_LINE_SEG
from hwp5.treeop import STARTEVENT, ENDEVENT
from hwp5.utils import cached_property
from hwp5.xmlmodel import DocInfo
from hwp5.xmlmodel import Hwp5File
from hwp5.xmlmodel import ModelEventStream
from hwp5.xmlmodel import Section
from hwp5.xmlmodel import XmlEvents
from hwp5.xmlmodel import embed_bindata
from hwp5.xmlmodel import line_segmented
from hwp5.xmlmodel import make_ranged_shapes
from hwp5.xmlmodel import merge_paragraph_text_charshape_lineseg
from hwp5.xmlmodel import split_and_shape

from . import test_binmodel
from .fixtures import get_fixture_path


class TestBase(test_binmodel.TestBase):

    @cached_property
    def hwp5file_xml(self):
        return Hwp5File(self.olestg)

    hwp5file = hwp5file_xml


class TestXmlEvents(TestBase):

    def test_dump_quoteattr_cr(self):
        sio = BytesIO()

        context = dict()
        attrs = dict(char='\r')
        events = [(STARTEVENT, (ControlChar, attrs, context)),
                  (ENDEVENT, (ControlChar, attrs, context))]
        xmlevents = XmlEvents(iter(events))
        xmlevents.dump(sio)

        data = sio.getvalue()
        self.assertTrue(b'&#13;' in data)

    def test_bytechunks_quoteattr_cr(self):

        context = dict()
        attrs = dict(char='\r')
        item = (ControlChar, attrs, context)
        modelevents = [(STARTEVENT, item),
                       (ENDEVENT, item)]
        xmlevents = XmlEvents(iter(modelevents))
        xml = b''.join(xmlevents.bytechunks())

        self.assertTrue(b'&#13;' in xml)


class TestModelEventStream(TestBase):

    @cached_property
    def docinfo(self):
        return ModelEventStream(self.hwp5file_bin['DocInfo'],
                                self.hwp5file_bin.header.version)

    def test_modelevents(self):
        self.assertEqual(len(list(self.docinfo.models())) * 2,
                         len(list(self.docinfo.modelevents())))
        # print len(list(self.docinfo.modelevents()))


class TestDocInfo(TestBase):

    @cached_property
    def docinfo(self):
        return DocInfo(self.hwp5file_bin['DocInfo'],
                       self.hwp5file_bin.header.version)

    def test_events(self):
        events = list(self.docinfo.events())
        self.assertEqual(136, len(events))
        # print len(events)

        # without embedbin, no <text> is embedded
        self.assertTrue('<text>' not in events[4][1][1]['bindata'])

    def test_events_with_embedbin(self):
        bindata = self.hwp5file_bin['BinData']
        events = list(self.docinfo.events(embedbin=bindata))
        self.assertTrue('<text>' in events[4][1][1]['bindata'])
        self.assertEqual(bindata['BIN0002.jpg'].open().read(),
                         base64.b64decode(events[4][1][1]
                                          ['bindata']['<text>']))


class TestSection(TestBase):

    def test_events(self):
        section = Section(self.hwp5file_bin['BodyText']['Section0'],
                          self.hwp5file_bin.fileheader.version)
        events = list(section.events())
        ev, (tag, attrs, ctx) = events[0]
        self.assertEqual((STARTEVENT, SectionDef), (ev, tag))
        self.assertFalse('section-id' in attrs)

        ev, (tag, attrs, ctx) = events[1]
        self.assertEqual((STARTEVENT, PageDef), (ev, tag))

        ev, (tag, attrs, ctx) = events[2]
        self.assertEqual((ENDEVENT, PageDef), (ev, tag))

        ev, (tag, attrs, ctx) = events[-1]
        self.assertEqual((ENDEVENT, SectionDef), (ev, tag))


class TestHwp5File(TestBase):

    def test_docinfo_class(self):
        self.assertTrue(isinstance(self.hwp5file.docinfo, DocInfo))

    def test_events(self):
        list(self.hwp5file.events())

    def test_events_embedbin_without_bindata(self):
        # see issue 76: https://github.com/mete0r/pyhwp/issues/76
        self.hwp5file_name = 'parashape.hwp'  # an hwp5file without BinData
        hwp5file = self.hwp5file
        self.assertTrue('BinData' not in hwp5file)
        list(hwp5file.events(embedbin=True))

    def test_xmlevents(self):
        events = iter(self.hwp5file.xmlevents())
        ev = next(events)
        self.assertEqual((STARTEVENT,
                          ('HwpDoc', dict(version='5.0.1.7'))), ev)
        list(events)

    def test_xmlevents_dump(self):
        with io.open(self.id() + '.xml', 'wb+') as outfile:
            self.hwp5file.xmlevents().dump(outfile)

            outfile.seek(0)
            doc = ElementTree.parse(outfile)

        self.assertEqual('HwpDoc', doc.getroot().tag)


class TestShapedText(TestCase):
    def test_make_shape_range(self):
        charshapes = [(0, 'A'), (4, 'B'), (6, 'C'), (10, 'D')]
        ranged_shapes = make_ranged_shapes(charshapes)
        self.assertEqual([((0, 4), 'A'), ((4, 6), 'B'), ((6, 10), 'C'),
                          ((10, 0x7fffffff), 'D')], list(ranged_shapes))

    def test_split(self):
        chunks = [((0, 3), None, 'aaa'), ((3, 6), None, 'bbb'),
                  ((6, 9), None, 'ccc'), ((9, 12), None, 'ddd')]
        charshapes = [(0, 'A'), (4, 'B'), (6, 'C'), (10, 'D')]
        shaped_chunks = split_and_shape(iter(chunks),
                                        make_ranged_shapes(charshapes))
        shaped_chunks = list(shaped_chunks)
        self.assertEqual([
            ((0, 3), ('A', None), 'aaa'),
            ((3, 4), ('A', None), 'b'),
            ((4, 6), ('B', None), 'bb'),
            ((6, 9), ('C', None), 'ccc'),
            ((9, 10), ('C', None), 'd'),
            ((10, 12), ('D', None), 'dd')],
            shaped_chunks)

        # split twice
        chunks = [((0, 112), None, 'x' * 112)]
        charshapes = [(0, 'a'), (3, 'b'), (5, 'c')]
        linesegs = [(0, 'A'), (51, 'B'), (103, 'C')]
        shaped = split_and_shape(iter(chunks), make_ranged_shapes(charshapes))
        shaped = list(shaped)
        self.assertEqual([((0, 3), ('a', None), 'xxx'),
                          ((3, 5), ('b', None), 'xx'),
                          ((5, 112), ('c', None), 'x' * 107)], shaped)
        lines = split_and_shape(iter(shaped), make_ranged_shapes(linesegs))
        lines = list(lines)
        self.assertEqual([
            ((0, 3), ('A', ('a', None)), 'xxx'),
            ((3, 5), ('A', ('b', None)), 'xx'),
            ((5, 51), ('A', ('c', None)), 'x' * (51 - 5)),
            ((51, 103), ('B', ('c', None)), 'x' * (103 - 51)),
            ((103, 112), ('C', ('c', None)), 'x' * (112 - 103))], lines)

    def test_split_keeps_surrogate_pair_intact(self):
        # UTF-16 unit layout: 'a'=0, high surrogate=1, low surrogate=2,
        # 'b'=3 ; the shape boundary at unit 2 lands inside the pair
        text = 'a\U0001F4A1b'
        chunks = [((0, 4), None, text)]
        charshapes = [(0, 'A'), (2, 'B')]
        shaped_chunks = list(split_and_shape(iter(chunks),
                                             make_ranged_shapes(charshapes)))
        self.assertEqual([
            ((0, 3), ('A', None), 'a\U0001F4A1'),
            ((3, 4), ('B', None), 'b')],
            shaped_chunks)
        # concatenation restores the full visible text
        self.assertEqual(text, ''.join(chunk
                                       for _, _, chunk in shaped_chunks))

    def test_split_with_several_supplementary_characters(self):
        # unit layout: a0 b1 pair2-3 c4 d5 pair6-7
        text = 'ab\U0001F4A1cd\U0001F600'
        chunks = [((0, 8), None, text)]

        shaped = list(split_and_shape(iter(chunks),
                                      make_ranged_shapes([(0, 'A'),
                                                          (5, 'B')])))
        self.assertEqual([
            ((0, 5), ('A', None), 'ab\U0001F4A1c'),
            ((5, 8), ('B', None), 'd\U0001F600')],
            shaped)

        # a boundary inside the second pair (unit 7) cannot split it
        shaped = list(split_and_shape(iter([((0, 8), None, text)]),
                                      make_ranged_shapes([(0, 'A'),
                                                          (7, 'B')])))
        self.assertEqual([((0, 8), ('A', None), text)], shaped)

    def test_control_marker_is_never_split(self):
        # an extended control marker occupies 8 UTF-16 code units but
        # renders no visible glyph; a shape boundary inside its span must
        # not split (or slice) the marker mapping
        marker = dict(code=2, chid='secd', param=b'')
        chunks = [((0, 8), None, marker), ((8, 10), None, 'xy')]
        charshapes = [(0, 'A'), (4, 'B')]
        shaped_chunks = list(split_and_shape(iter(chunks),
                                             make_ranged_shapes(charshapes)))
        self.assertEqual([
            ((0, 8), ('A', None), marker),
            ((8, 10), ('B', None), 'xy')],
            shaped_chunks)

    def test_control_marker_kept_alongside_surrogate_text(self):
        # units: marker 0..8, 'x'=8, pair 9..10, 'y'=11
        marker = dict(code=9,
                      param=dict(width=1, unknown0=0, unknown1=0,
                                 unknown2=b''))
        chunks = [((0, 8), None, marker),
                  ((8, 12), None, 'x\U0001F4A1y')]
        charshapes = [(0, 'S0'), (6, 'S1'), (10, 'S2')]
        shaped_chunks = list(split_and_shape(iter(chunks),
                                             make_ranged_shapes(charshapes)))
        self.assertEqual([
            ((0, 8), ('S0', None), marker),
            ((8, 11), ('S1', None), 'x\U0001F4A1'),
            ((11, 12), ('S2', None), 'y')],
            shaped_chunks)
        # the visible text is fully recoverable; markers are excluded
        visible = ''.join(chunk for _, _, chunk in shaped_chunks
                          if not isinstance(chunk, dict))
        self.assertEqual('x\U0001F4A1y', visible)

    def test_paratext_chunks_with_surrogate_and_extended_control(self):
        # UTF-16LE payload: 'a'(0) + U+1F4A1 pair(1..2) +
        # FIELD_START extended marker(3..10) + 'b'(11)
        marker = struct.pack('<H', 0x03) + b'\x00' * 14
        payload = (
            'a'.encode('utf-16-le') +
            '\U0001F4A1'.encode('utf-16-le') +
            marker +
            'b'.encode('utf-16-le')
        )
        chunks = list(ParaTextChunks.parse_chunks(payload))

        text_range, text = chunks[0]
        self.assertEqual((0, 3), text_range)
        self.assertEqual('a\U0001F4A1', text)

        control_range, control = chunks[1]
        self.assertEqual((3, 11), control_range)
        self.assertEqual(0x03, control['code'])
        self.assertTrue(isinstance(control, dict))

        self.assertEqual(((11, 12), 'b'), chunks[2])

        # shape boundary at unit 2 cuts through the surrogate pair and
        # another sits between the marker and the final visible character
        charshapes = [(0, 'A'), (2, 'B'), (11, 'C')]
        shaped_chunks = list(split_and_shape(
            iter((range, None, chunk) for range, chunk in chunks),
            make_ranged_shapes(charshapes)))

        self.assertEqual('a\U0001F4A1', shaped_chunks[0][2])
        self.assertEqual((0, 3), shaped_chunks[0][0])
        self.assertEqual((3, 11), shaped_chunks[1][0])
        self.assertEqual(0x03, shaped_chunks[1][2]['code'])
        self.assertEqual(((11, 12), ('C', None), 'b'), shaped_chunks[2])

        visible = ''.join(chunk for _, _, chunk in shaped_chunks
                          if not isinstance(chunk, dict))
        self.assertEqual('a\U0001F4A1b', visible)


class TestLineSeg(TestCase):
    def test_line_segmented(self):
        chunks = [((0, 3), None, 'aaa'), ((3, 6), None, 'bbb'),
                  ((6, 9), None, 'ccc'), ((9, 12), None, 'ddd')]
        linesegs = [(0, 'A'), (4, 'B'), (6, 'C'), (10, 'D')]
        lines = line_segmented(iter(chunks), make_ranged_shapes(linesegs))
        lines = list(lines)
        self.assertEqual([('A', [((0, 3), None, 'aaa'),
                                 ((3, 4), None, 'b')]),
                          ('B', [((4, 6), None, 'bb')]),
                          ('C', [((6, 9), None, 'ccc'),
                                 ((9, 10), None, 'd')]),
                          ('D', [((10, 12), None, 'dd')])], lines)

    def test_line_segmented_keeps_surrogate_pair_on_one_line(self):
        # unit layout: a0 b1 pair2-3 c4 d5 ; the line boundary at unit 3
        # lands inside the surrogate pair, so the pair stays on the first
        # line together with its char-shape attribution
        text = 'ab\U0001F4A1cd'
        chunks = [((0, 6), None, text)]
        shaped = split_and_shape(iter(chunks),
                                 make_ranged_shapes([(0, 'A'), (4, 'B')]))
        lines = list(line_segmented(iter(shaped),
                                    make_ranged_shapes([(0, 'L0'),
                                                        (3, 'L1')])))
        line0_text = ''.join(chunk for _, _, chunk in lines[0][1]
                             if not isinstance(chunk, dict))
        line1_text = ''.join(chunk for _, _, chunk in lines[1][1]
                             if not isinstance(chunk, dict))
        self.assertEqual('ab\U0001F4A1', line0_text)
        self.assertEqual('cd', line1_text)
        joined = ''.join(chunk for _, items in lines
                         for _, _, chunk in items
                         if not isinstance(chunk, dict))
        self.assertEqual(text, joined)


class TestDistributionBodyText(TestBase):

    hwp5file_name = 'viewtext.hwp'

    def test_issue33_missing_paralineseg(self):
        section0 = self.hwp5file_bin.bodytext.section(0)
        tagids = set(model['tagid'] for model in section0.models())
        types = set(model['type'] for model in section0.models())
        self.assertTrue(HWPTAG_PARA_LINE_SEG not in tagids)
        self.assertTrue(ParaLineSeg not in types)

        paratext = self.hwp5file_bin.bodytext.section(0).model(1)
        self.assertEqual(ParaText, paratext['type'])

        paracharshape = self.bodytext.section(0).model(2)
        self.assertEqual(ParaCharShape, paracharshape['type'])

        evs = merge_paragraph_text_charshape_lineseg(
            (paratext['type'], paratext['content'], dict()),
            (paracharshape['type'], paracharshape['content'], dict()),
            None
        )

        # we can merge events without a problem
        list(evs)


class TestMatchFieldStartEnd(TestCase):

    def test_match_field_start_end(self):

        records = \
            [{'level': 2,
              'payload': b'\x1c\x00\x00\x00\x18\x00\x00\x00\x04\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x00\x80',  # noqa
              'seqno': 196,
              'size': 22,
              'tagid': 66,
              'tagname': 'HWPTAG_PARA_HEADER'},
             {'level': 3,
              'payload': b'\x04\x00umf\x08\x00\x00\x00\x00\x00\x00\x00\x00\x04\x00\x03\x00umf%\x00\x00\x00\x00\x00\x00\x00\x00\x03\x002\x009\x009\x00\x04\x00umf\x08\x00\x00\x00\x00\x00\x00\x00\x00\x04\x00\r\x00',  # noqa
              'seqno': 197,
              'size': 56,
              'tagid': 67,
              'tagname': 'HWPTAG_PARA_TEXT'},
             {'level': 3,
              'payload': b'\x00\x00\x00\x00\x13\x00\x00\x00',
              'seqno': 198,
              'size': 8,
              'tagid': 68,
              'tagname': b'HWPTAG_PARA_CHAR_SHAPE'},
             {'level': 3,
              'payload': b'\x00\x00\x00\x00\x00\x00\x00\x00\x14\x05\x00\x00\x14\x05\x00\x00Q\x04\x00\x00\xfc\xfe\xff\xff\x00\x00\x00\x00X\x0b\x00\x00\x00\x00\x06\x00',  # noqa
              'seqno': 199,
              'size': 36,
              'tagid': 69,
              'tagname': 'HWPTAG_PARA_LINE_SEG'},
             {'level': 3,
              'payload': b'umf%\x00\x00\x00\x00\x08\x15\x00=\x00S\x00U\x00M\x00(\x00R\x00I\x00G\x00H\x00T\x00)\x00?\x00?\x00%\x00g\x00,\x00;\x00;\x002\x009\x009\x00P\x89\xa0z\x00\x00\x00\x00',  # noqa
              'seqno': 200,
              'size': 61,
              'tagid': 71,
              'tagname': 'HWPTAG_CTRL_HEADER'}]

        models = binmodel.parse_models(dict(), records)
        events = xmlmodel.prefix_binmodels_with_event(dict(), models)
        events = xmlmodel.make_texts_linesegmented_and_charshaped(events)
        events = xmlmodel.make_extended_controls_inline(events)
        events = xmlmodel.match_field_start_end(events)
        events = list(events)

    def test_issue144_fields_crossing_lineseg_boundary(self):

        name = 'issue144-fields-crossing-lineseg-boundary.hwp'
        path = get_fixture_path(name)
        hwp5file = xmlmodel.Hwp5File(path)
        xmlevents = hwp5file.bodytext.xmlevents()
        # pprint(list(enumerate(xmlevents)))

        stack_fields = []
        for ev, model in xmlevents:

            if ev is treeop.STARTEVENT:
                tag = model[0]
            else:
                tag = model

            if tag.startswith('Field'):
                if ev is treeop.STARTEVENT:
                    stack_fields.append(model)
                else:
                    stack_fields.pop()
            elif tag == 'LineSeg':
                # NO CROSSING
                if ev is treeop.STARTEVENT:
                    assert len(stack_fields) == 0
                else:
                    assert len(stack_fields) == 0


class TestEmbedBinData(TestBase):

    def test_embed_bindata(self):

        bindata = dict(flags=BinData.Flags(BinData.StorageType.EMBEDDING),
                       bindata=dict(storage_id=2, ext='jpg'))
        events = [(STARTEVENT, (BinData, bindata, dict())),
                  (ENDEVENT, (BinData, bindata, dict()))]
        events = list(embed_bindata(events, self.hwp5file_bin['BinData']))
        self.assertTrue('<text>' in bindata['bindata'])
