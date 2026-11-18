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
import sys

from ..cli import parse_recordstream_name
from ..recordstream import Hwp5File as RecordStreamHwp5File
from ..recordstream import RecordStream
from ..storage import Open2Stream
from ..structured_import import format_stream_report
from ..structured_import import import_stream
from ..structured_import import report_to_json


PY2 = sys.version_info.major == 2


def main(args):
    if args.hwp5file and not args.record_stream:
        sys.stderr.write(
            'hwp5proc import: a <record-stream> is required with <hwp5file>\n'
        )
        raise SystemExit(2)

    stream, declared_version, declared_from = stream_from_args(args)

    report = import_stream(stream, version=declared_version,
                           include_candidates=not args.no_candidates)
    report['declared_version_source'] = declared_from

    if args.json:
        sys.stdout.write(report_to_json(report))
        sys.stdout.write('\n')
    else:
        print(format_stream_report(report, declared_from))

    if args.strict and report['summary']['records_with_contradictions']:
        raise SystemExit(1)
    return None


def stream_from_args(args):
    filename = args.hwp5file
    version = args.file_format_version

    if filename:
        records = RecordStreamHwp5File(filename)
        stream = parse_recordstream_name(records, args.record_stream)
        if version:
            declared = parse_version(version)
            source = '--file-format-version'
        else:
            declared = stream.version
            source = 'FileHeader'
        return stream, declared, source
    if version:
        declared = parse_version(version)
        source = '--file-format-version'
    else:
        declared = (5, 0, 0, 0)
        source = 'default'

    if PY2:
        stdin = sys.stdin
    else:
        stdin = sys.stdin.buffer
    stream = RecordStream(Open2Stream(lambda: stdin), declared)
    return stream, declared, source


def parse_version(version):
    return tuple(int(x) for x in version.split('.'))


def import_argparser(subparsers, _):
    parser = subparsers.add_parser(
        'import',
        help=_(
            'Import record streams with explicit structure extents.'
        ),
        description=_(
            'Import the records of the specified <record-stream> using the '
            'declared format version.  For each record, report the '
            'interpreted payload range, the unconsumed tail, and any '
            'contradiction between the declared version and the actual '
            'bytes.  When the declared structure does not explain the '
            'payload, candidate structure versions and the fields they '
            'affect are listed.  A candidate that merely consumes every byte '
            'is not considered verified.'
        ),
    )
    parser.add_argument(
        'hwp5file',
        nargs='?',
        metavar='<hwp5file>',
        help=_('.hwp file to analyze'),
    )
    parser.add_argument(
        'record_stream',
        nargs='?',
        metavar='<record-stream>',
        help=_(
            'Record-structured internal streams.\n'
            '(e.g. DocInfo, BodyText/*)\n'
        ),
    )
    parser.add_argument(
        '--file-format-version',
        '-V',
        metavar='<version>',
        help=_(
            'Specifies the declared HWPv5 file format version.  '
            'Defaults to the version in FileHeader, or 5.0.0.0 for stdin.'
        ),
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help=_('Print the full structured import report as json.'),
    )
    parser.add_argument(
        '--no-candidates',
        action='store_true',
        help=_(
            'Do not attempt alternative structure versions when the declared '
            'version does not explain the payload.'
        ),
    )
    parser.add_argument(
        '--strict',
        action='store_true',
        help=_(
            'Exit with status 1 if any record has a parse error or a '
            'contradiction between the declared version and the payload.'
        ),
    )
    parser.set_defaults(func=main)
    return parser


def list_models_argparser(subparsers, _):
    parser = subparsers.add_parser(
        'list-models',
        help=_(
            'List record models and their versioned members.'
        ),
        description=_(
            'List the structures that the importer knows for each record '
            'type, including the version at which each member appears.  '
            'This is the declaration side of a structured import.'
        ),
    )
    parser.add_argument(
        'models',
        nargs='*',
        metavar='<model>',
        help=_('Restrict the listing to these model names.'),
    )
    parser.set_defaults(func=list_models)
    return parser


def list_models(args):
    from ..binmodel._shared import tag_models
    from ..structured_import import _type_members
    from ..tagids import tagnames

    names = set(args.models)

    rows = []
    for tagid in sorted(tag_models):
        model_type = tag_models[tagid]
        if names and model_type.__name__ not in names:
            continue
        tagname = tagnames.get(tagid, 'HWPTAG%d' % tagid)
        for member in _type_members(model_type):
            version = member.get('version')
            if version is None:
                continue
            rows.append((model_type.__name__, tagname, member['name'],
                         '.'.join(str(x) for x in version)))

    for model_name, tagname, member_name, version in rows:
        print('%-24s %-32s %-32s %s' % (
            model_name, tagname, member_name, version))
    if not rows:
        print('no versioned members found')
    return None
