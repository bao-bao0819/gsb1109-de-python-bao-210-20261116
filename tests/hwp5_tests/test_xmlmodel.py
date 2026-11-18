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
from hwp5.binmodel import SectionDef
from hwp5.recordstream import encode_record_header
from hwp5.tagids import HWPTAG_PARA_LINE_SEG
from hwp5.treeop import STARTEVENT, ENDEVENT
from hwp5.utils import cached_property
from hwp5.xmlmodel import DocInfo
from hwp5.xmlmodel import Hwp5File
from hwp5.xmlmodel import ModelEventStream
from hwp5.xmlmodel import Section
from hwp5.xmlmodel import XmlEvents
from hwp5.xmlmodel import docinfo_defined_style_ids
from hwp5.xmlmodel import embed_bindata
from hwp5.xmlmodel import line_segmented
from hwp5.xmlmodel import make_ranged_shapes
from hwp5.xmlmodel import merge_paragraph_text_charshape_lineseg
from hwp5.xmlmodel import split_and_shape
from hwp5.xmlmodel import unresolved_style_segments

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


class MemoryStream(object):

    def __init__(self, data):
        self.data = data

    def open(self):
        return BytesIO(self.data)


def make_fileheader(version=(5, 0, 1, 7)):
    data = bytearray(256)
    data[:32] = b'HWP Document File' + b'\x00' * 15
    major, minor, micro, build = version
    data[32:36] = bytes([build, micro, minor, major])
    return bytes(data)


def make_record(tagid, level, payload):
    header = encode_record_header(dict(tagid=tagid, level=level,
                                       payload=payload))
    return header + payload


# a paragraph with a SectionDef control, taken from a real document;
# every section starts with one
SECTION0_PREFIX = bytes.fromhex(
    '4200600122000000040000000000000305000000010000000000430440040200'
    '64636573000000000000000002000200646c6f63000000000000000002005cd5'
    '00ae20003200300030003500200008c61cc820000cd37cc785c7c8b2e4b22e00'
    '0d00440480020000000007000000130000000800000017000000070000001800'
    '0000090000001a00000007000000450440020000000000000000d0070000d007'
    '0000a4060000b00400000000000018a600000000060047044002646365730000'
    '00006e0400000000401f00000100000000000000000000000000000000004908'
    '800288e80000dc4801003821000038210000241600009c1000009c1000009c10'
    '000000000000000000004a08c001000000000000000029000100ffffffff5203'
    '37021b010101000000004a08c001000000000000000029000100f82fe0005203'
    '370200000101000000004b08e00001000000890589058905890500004b08e000'
    '01000000890589058905890500004b08e0000100000089058905890589050000'
    '47040001646c6f63041000000000000000000000'
)

LINESEG_PAYLOAD = bytes.fromhex(
    '0000000000000000140500001405000051040000fcfeffff'
    '00000000580b000000000600'
)


def make_paragraph_records(parashape_id, style_id, charshape_id, text):
    para_header = struct.pack('<IIHBBHHHI',
                              len(text), 0, parashape_id, style_id, 0,
                              1, 0, 1, 0)
    para_charshape = struct.pack('<II', 0, charshape_id)
    return b''.join([
        make_record(66, 0, para_header),
        make_record(67, 1, text.encode('utf-16le')),
        make_record(68, 1, para_charshape),
        make_record(69, 1, LINESEG_PAYLOAD),
    ])


def make_section0():
    return b''.join([
        SECTION0_PREFIX,
        make_paragraph_records(0, 0, 0, '가나다라'),
        make_paragraph_records(7, 5, 9, '마바사아'),
    ])


def make_bstr(s):
    return struct.pack('<H', len(s)) + s.encode('utf-16le')


def make_docinfo_stream():
    ''' a minimal style sheet: 1 charshape, 1 parashape, 1 style '''
    document_properties = bytes(26)
    idmappings = struct.pack('<16I',
                             0, 0, 0, 0, 0, 0, 0, 0,
                             0, 1, 0, 0, 0, 1, 1, 0)
    charshape = bytes(68)
    parashape = bytes(46)
    style = (make_bstr('바탕글') + make_bstr('Normal')
             + struct.pack('<BBhHHH', 0, 0, 1042, 0, 0, 0))
    return b''.join([
        make_record(16, 0, document_properties),
        make_record(17, 0, idmappings),
        make_record(21, 0, charshape),
        make_record(25, 0, parashape),
        make_record(26, 0, style),
    ])


def make_storage(with_docinfo=False):
    storage = {
        'FileHeader': MemoryStream(make_fileheader()),
        'BodyText': {
            'Section0': MemoryStream(make_section0()),
        },
    }
    if with_docinfo:
        storage['DocInfo'] = MemoryStream(make_docinfo_stream())
    return storage


def findall_attributes(events, model):
    for event, item in events:
        item_model, attributes, context = item
        if event is STARTEVENT and item_model is model:
            yield attributes


class TestArchiveWithoutDocInfo(TestCase):
    ''' 아카이브에 본문만 있고 DocInfo 스타일이 없는 경우 '''

    @cached_property
    def hwp5file(self):
        return Hwp5File(make_storage())

    def dump_xml(self, hwp5file, **kwargs):
        out = BytesIO()
        hwp5file.xmlevents(**kwargs).dump(out)
        return out.getvalue().decode('utf-8')

    def test_events_without_docinfo(self):
        events = list(self.hwp5file.events())
        models = [item[0] for event, item in events if event is STARTEVENT]
        self.assertTrue(DocInfo in models)
        self.assertTrue(binmodel.Paragraph in models)

    def test_export_text_without_docinfo(self):
        xml = self.dump_xml(self.hwp5file)
        doc = ElementTree.fromstring(xml)
        (docinfo,) = doc.findall('DocInfo')
        # missing styles are not replaced with fabricated defaults
        self.assertEqual(0, len(list(docinfo)))
        bodytexts = doc.findall('BodyText')
        self.assertEqual(1, len(bodytexts))

    def test_style_references_preserved(self):
        events = list(self.hwp5file.events())
        paragraphs = list(findall_attributes(events, binmodel.Paragraph))
        self.assertEqual(3, len(paragraphs))
        self.assertEqual(0, paragraphs[1]['style_id'])
        self.assertEqual(0, paragraphs[1]['parashape_id'])
        self.assertEqual(5, paragraphs[2]['style_id'])
        self.assertEqual(7, paragraphs[2]['parashape_id'])

        texts = list(findall_attributes(events, binmodel.Text))
        charshape_ids = [t['charshape_id'] for t in texts]
        self.assertTrue(0 in charshape_ids)
        self.assertTrue(9 in charshape_ids)
        # determinable language information is preserved
        self.assertEqual('ko', texts[-1]['lang'])

    def test_docinfo_defined_style_ids_empty(self):
        events = self.hwp5file.docinfo_events()
        defined = docinfo_defined_style_ids(events)
        self.assertEqual(dict(styles=set(), charshapes=set(),
                              parashapes=set(), numberings=set()),
                         defined)

    def test_unresolved_segments_without_styles(self):
        events = list(self.hwp5file.events())
        defined = docinfo_defined_style_ids(self.hwp5file.docinfo_events())
        segments = list(unresolved_style_segments(events, defined))
        paragraphs = [item for item, unresolved in segments
                      if item[0] is binmodel.Paragraph]
        # every paragraph references undefined styles
        self.assertEqual(3, len(paragraphs))

    def test_docinfo_in_storage(self):
        hwp5file = Hwp5File(make_storage(with_docinfo=True))
        xml = self.dump_xml(hwp5file)
        self.assertTrue('<Style ' in xml)
        self.assertTrue('name="Normal"' in xml)

    def test_supplement_stylesheet_stream(self):
        docinfo = MemoryStream(make_docinfo_stream())
        xml = self.dump_xml(self.hwp5file, docinfo=docinfo)
        doc = ElementTree.fromstring(xml)
        (docinfo_element,) = doc.findall('DocInfo')
        (style,) = docinfo_element.findall('Style')
        # inheritance chain and language information of the supplemented
        # style sheet are preserved
        self.assertEqual('Normal', style.get('name'))
        self.assertEqual('0', style.get('next-style-id'))
        self.assertEqual('1042', style.get('lang-id'))
        self.assertEqual('0', style.get('parashape-id'))
        self.assertEqual('0', style.get('charshape-id'))

    def test_supplement_stylesheet_events(self):
        class SupplementedDocInfo(object):
            def events(self, **kwargs):
                item = (binmodel.Style,
                        dict(local_name='바탕글', name='Normal'), dict())
                yield STARTEVENT, item
                yield ENDEVENT, item

        xml = self.dump_xml(self.hwp5file, docinfo=SupplementedDocInfo())
        self.assertTrue('<Style ' in xml)

    def test_supplement_keeps_body_stable(self):
        xml_without = self.dump_xml(self.hwp5file)
        docinfo = MemoryStream(make_docinfo_stream())
        xml_with = self.dump_xml(self.hwp5file, docinfo=docinfo)
        body_without = xml_without[xml_without.index('<BodyText>'):]
        body_with = xml_with[xml_with.index('<BodyText>'):]
        # segments unrelated to the supplemented style sheet keep their
        # semantics and provenance
        self.assertEqual(body_without, body_with)

    def test_unresolved_segments_after_supplement(self):
        docinfo = MemoryStream(make_docinfo_stream())
        defined = docinfo_defined_style_ids(
            self.hwp5file.docinfo_events(docinfo))
        self.assertEqual(dict(styles=set([0]), charshapes=set([0]),
                              parashapes=set([0]), numberings=set()),
                         defined)

        events = list(self.hwp5file.events())
        segments = list(unresolved_style_segments(events, defined))
        paragraphs = [item for item, unresolved in segments
                      if item[0] is binmodel.Paragraph]
        # only the paragraph referencing still-undefined styles
        # needs recalculation
        self.assertEqual(1, len(paragraphs))
        attributes = paragraphs[0][1]
        self.assertEqual(5, attributes['style_id'])
        self.assertEqual(7, attributes['parashape_id'])

        texts = [item for item, unresolved in segments
                 if item[0] is binmodel.Text]
        charshape_ids = sorted(set(t[1]['charshape_id'] for t in texts))
        self.assertEqual([7, 8, 9], charshape_ids)
