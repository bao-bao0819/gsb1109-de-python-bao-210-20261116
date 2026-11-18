# -*- coding: utf-8 -*-
#
#   pyhwp : hwp file format parser in python
#   Copyright (C) 2010-2023 mete0r <https://github.com/mete0r>
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Affero General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Affero General Public License for more details.
#
#   You should have received a copy of the GNU Affero General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from io import BytesIO
import codecs
import zlib

from .utils import GeneratorReader


class ZLibIncrementalDecoder(codecs.IncrementalDecoder):
    def __init__(self, errors='strict', wbits=15):
        assert errors == 'strict'
        self.errors = errors
        self.wbits = wbits
        self.reset()

    def decode(self, input, final=False):
        c = self.decompressobj.decompress(input)
        if final:
            c += self.decompressobj.flush()
        return c

    def reset(self):
        self.decompressobj = zlib.decompressobj(self.wbits)


class DecompressionStatus(object):
    ''' status of an incremental decompression session

    - ``eof``: the end of the zlib stream has been reached
    - ``input_exhausted``: the input stream has been exhausted
    - ``error``: a zlib.error if decompression failed
    '''

    def __init__(self):
        self.eof = False
        self.input_exhausted = False
        self.error = None

    @property
    def truncated(self):
        ''' the zlib stream end has not been reached '''
        return not self.eof

    def __repr__(self):
        return ('<%s eof=%r input_exhausted=%r error=%r>'
                % (type(self).__name__, self.eof, self.input_exhausted,
                   self.error))


def decompress_gen(source, bufsize=4096, status=None):
    ''' generate decompressed chunks from a deflate stream

        source: a file-like readable
        bufsize: input chunk size; the resulting bytes do not depend on it
        status: an optional DecompressionStatus to be updated

        The input is read until an empty read occurs; short reads do not
        terminate the decompression, so the outcome does not depend on
        short-read timing.  If the input ends before the zlib stream end
        (i.e. truncated), the bytes decompressed so far are yielded and
        ``status.truncated`` reports True.
    '''
    dec = ZLibIncrementalDecoder(wbits=-15)
    try:
        while True:
            input = source.read(bufsize)
            if len(input) == 0:
                break
            yield dec.decode(input)
        yield dec.decode(b'', True)
    except zlib.error as e:
        if status is not None:
            status.error = e
        raise
    if status is not None:
        status.input_exhausted = True
        status.eof = dec.decompressobj.eof


def decompress_experimental(source, bufsize=4096):
    ''' decompress inputstream

        stream: a file-like readable
        returns a file-like readable
    '''
    return GeneratorReader(decompress_gen(source, bufsize))


def decompress(stream, bufsize=4096, status=None):
    ''' decompress inputstream

        stream: a file-like readable
        returns a file-like readable

        If the stream is truncated, the bytes decompressed so far are
        returned; pass ``status`` (a DecompressionStatus) to tell whether
        the zlib stream end has been reached.
    '''

    # #176 참고. #175의 임시방편을 사용한다.
    compressed_maybe = stream.read()
    try:
        chunks = decompress_gen(BytesIO(compressed_maybe), bufsize, status)
        decompressed = b''.join(chunks)
    except zlib.error:
        return BytesIO(compressed_maybe)
    else:
        return BytesIO(decompressed)
