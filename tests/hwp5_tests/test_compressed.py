# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from io import BytesIO
from unittest import TestCase
import os
import zlib

from hwp5.compressed import DecompressionStatus
from hwp5.compressed import ZLibIncrementalDecoder
from hwp5.compressed import decompress
from hwp5.compressed import decompress_gen
from hwp5.utils import cached_property


class ShortReadStream(object):
    ''' a file-like readable which never returns more than `maxchunk'
    bytes per read, even if more is requested and available '''

    def __init__(self, data, maxchunk):
        self.data = data
        self.maxchunk = maxchunk
        self.pos = 0

    def read(self, size=-1):
        remaining = len(self.data) - self.pos
        if size is None or size < 0 or size > remaining:
            size = remaining
        if size > self.maxchunk:
            size = self.maxchunk
        chunk = self.data[self.pos:self.pos + size]
        self.pos += size
        return chunk


class TestUncompress(TestCase):

    @cached_property
    def original_data(self):
        return os.urandom(16384)

    @cached_property
    def compressed_data(self):
        return zlib.compress(self.original_data)

    def test_incremental_decode(self):
        compressed_data = self.compressed_data

        dec = ZLibIncrementalDecoder(wbits=-15)
        data = dec.decode(compressed_data[2:2048])
        data += dec.decode(compressed_data[2048:2048 + 1024])
        data += dec.decode(compressed_data[2048 + 1024:2048 + 1024 + 4096])
        data += dec.decode(compressed_data[2048 + 1024 + 4096:], True)

        self.assertEqual(self.original_data, data)

    def test_decompress(self):

        gen = decompress_gen(BytesIO(self.compressed_data[2:]))
        self.assertEqual(self.original_data, b''.join(gen))

        # print '-----'

        f = decompress(BytesIO(self.compressed_data[2:]))
        g = BytesIO(self.original_data)

        self.assertEqual(f.read(2048), g.read(2048))
        self.assertEqual(f.read(1024), g.read(1024))
        self.assertEqual(f.read(4096), g.read(4096))
        self.assertEqual(f.read(), g.read())


class TestDecompressGen(TestCase):

    @cached_property
    def original_data(self):
        return os.urandom(16384)

    @cached_property
    def compressed_data(self):
        # raw deflate stream (wbits=-15), as stored in HWPv5 files
        c = zlib.compressobj(9, zlib.DEFLATED, -15)
        return c.compress(self.original_data) + c.flush()

    def test_decompressed_bytes_do_not_depend_on_bufsize(self):
        results = []
        for bufsize in (1, 2, 3, 7, 4096, 65536):
            source = BytesIO(self.compressed_data)
            gen = decompress_gen(source, bufsize)
            results.append(b''.join(gen))
        for result in results:
            self.assertEqual(self.original_data, result)

    def test_short_reads_do_not_terminate_decompression(self):
        source = ShortReadStream(self.compressed_data, 3)
        gen = decompress_gen(source, 4096)
        self.assertEqual(self.original_data, b''.join(gen))

    def test_status_of_a_complete_stream(self):
        status = DecompressionStatus()
        source = BytesIO(self.compressed_data)
        gen = decompress_gen(source, 4096, status)
        self.assertEqual(self.original_data, b''.join(gen))
        self.assertTrue(status.input_exhausted)
        self.assertTrue(status.eof)
        self.assertFalse(status.truncated)
        self.assertIsNone(status.error)

    def test_status_of_a_truncated_stream(self):
        truncated = self.compressed_data[:-7]
        status = DecompressionStatus()
        source = BytesIO(truncated)
        gen = decompress_gen(source, 4096, status)
        decompressed = b''.join(gen)
        # the bytes decompressed so far are reported...
        self.assertTrue(len(decompressed) > 0)
        self.assertEqual(self.original_data[:len(decompressed)], decompressed)
        # ...separately from the decompression status
        self.assertTrue(status.input_exhausted)
        self.assertFalse(status.eof)
        self.assertTrue(status.truncated)

    def test_truncated_stream_decompresses_the_same_for_any_bufsize(self):
        truncated = self.compressed_data[:-7]
        results = []
        statuses = []
        for bufsize in (1, 2, 3, 7, 4096, 65536):
            status = DecompressionStatus()
            source = BytesIO(truncated)
            gen = decompress_gen(source, bufsize, status)
            results.append(b''.join(gen))
            statuses.append(status)
        for result in results:
            self.assertEqual(results[0], result)
        for status in statuses:
            self.assertTrue(status.truncated)

    def test_decompress_returns_partial_output_of_a_truncated_stream(self):
        truncated = self.compressed_data[:-7]
        status = DecompressionStatus()
        f = decompress(BytesIO(truncated), status=status)
        decompressed = f.read()
        self.assertTrue(len(decompressed) > 0)
        self.assertEqual(self.original_data[:len(decompressed)], decompressed)
        self.assertTrue(status.truncated)

    def test_decompress_returns_nondeflate_input_as_is(self):
        data = b'this is not a deflate stream'
        f = decompress(BytesIO(data))
        self.assertEqual(data, f.read())
