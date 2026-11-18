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
''' Build provenance-preserving Hangul PUA search views.

Usage::

    hwp5proc searchview [--mapping=<json-table>] [--remap=<json-table>]
                        [--query=<text>] [--from-stdin]
                        [--loglevel=<loglevel>] [--logfile=<logfile>]
                        (<hwp5files>... | --from-stdin)
    hwp5proc searchview --help

Every paragraph is normalized through a versioned Hanyang-PUA mapping table.
The emitted view keeps the original codepoints, the mapping version and the
one-to-many source ranges; unknown codepoints stay distinct instead of being
unified.  With ``--query`` only search hits are printed, each traceable to the
concrete source glyphs.  With ``--remap`` the view is rebuilt with a second
table and the changed characters/hits (and their PUA ranges) are reported.
'''
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
import json
import logging
import sys

from ..binmodel import Hwp5File
from ..dataio import ParseError
from ..hypua_mappings import default_table
from ..pua import diff_views
from ..pua import iter_paratext_views
from ..pua import load_mapping_table
from ..pua import remap
from ..pua import search_hit_changes


PY2 = sys.version_info.major == 2
logger = logging.getLogger(__name__)


def _write(obj):
    line = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    if PY2:
        line = line.encode('utf-8')
        sys.stdout.write(line)
        sys.stdout.write(b'\n')
    else:
        sys.stdout.buffer.write(line.encode('utf-8'))
        sys.stdout.buffer.write(b'\n')


def main(args):
    table = _table_table(args)
    remap_table = _table_remap(args)

    for filename in filenames_from_args(args):
        try:
            process_file(filename, table, remap_table, args)
        except ParseError as e:
            logger.error('---- On processing %s:', filename)
            e.print_to_logger(logger)


def _table_table(args):
    if args.mapping:
        return load_mapping_table(args.mapping)
    return default_table()


def _table_remap(args):
    if args.remap:
        return load_mapping_table(args.remap)
    return None


def process_file(filename, table, remap_table, args):
    hwp5file = Hwp5File(filename)
    try:
        for view in iter_paratext_views(hwp5file, table):
            if remap_table is not None:
                emit_remap(filename, view, remap_table, args)
            elif args.query:
                emit_hits(filename, view, args)
            else:
                emit_view(filename, view)
    finally:
        hwp5file.close()


def emit_view(filename, view):
    obj = dict(filename=filename)
    obj.update(view.as_dict())
    _write(obj)


def emit_hits(filename, view, args):
    for hit in view.search(args.query):
        obj = dict(filename=filename)
        obj.update(hit.as_dict())
        obj['location'] = dict(stream=view.location[0],
                               seqno=view.location[1])
        obj['mapping_version'] = view.mapping_version
        _write(obj)


def emit_remap(filename, view, remap_table, args):
    new_view = remap(view, remap_table)
    if args.query:
        old_hits, new_hits, changed = search_hit_changes(
            view, new_view, args.query)
        obj = dict(
            filename=filename,
            location=dict(stream=view.location[0],
                          seqno=view.location[1]),
            query=args.query,
            old_mapping_version=view.mapping_version,
            new_mapping_version=new_view.mapping_version,
            old_hits=[hit.as_dict() for hit in old_hits],
            new_hits=[hit.as_dict() for hit in new_hits],
            changed_pua_ranges=changed,
        )
    else:
        obj = dict(filename=filename)
        obj.update(diff_views(view, new_view))
    _write(obj)


def filenames_from_args(args):
    if args.from_stdin:
        for line in sys.stdin:
            yield line[:-1]
    else:
        for filename in args.hwp5files:
            yield filename


def searchview_argparser(subparsers, _):
    parser = subparsers.add_parser(
        'searchview',
        help=_(
            'Build provenance-preserving Hangul PUA search views.'
        ),
        description=_(
            'Normalize Hanyang PUA text with a versioned mapping table and '
            'keep original codepoints, mapping version and one-to-many '
            'source ranges, so search hits trace back to source glyphs.'
        ),
    )
    parser.add_argument(
        'hwp5files',
        nargs='*',
        metavar='<hwp5files>',
        help=_('.hwp files to analyze'),
    )
    parser.add_argument(
        '--from-stdin',
        action='store_true',
        help=_('get filenames from stdin'),
    )
    parser.add_argument(
        '--mapping',
        metavar='<json-table>',
        help=_('JSON PUA mapping table to apply (bundled sample if omitted)'),
    )
    parser.add_argument(
        '--remap',
        metavar='<json-table>',
        help=_('rebuild the view with this table and report changed hits'),
    )
    parser.add_argument(
        '--query',
        metavar='<text>',
        help=_('report only search hits for this normalized text'),
    )
    parser.set_defaults(func=main)
    return parser
