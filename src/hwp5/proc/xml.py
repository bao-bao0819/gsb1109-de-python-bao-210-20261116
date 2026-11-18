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
from functools import partial
import json
import logging
import os

from ..utils import make_open_dest_file
from ..utils import wrap_open_dest_for_tty
from ..utils import wrap_open_dest
from ..utils import pager
from ..utils import syntaxhighlight
from ..utils import xmllint
from ..stableid import StableIdAssigner
from ..stableid import index_from_jsonable
from ..stableid import index_to_jsonable
from ..stableid import reconcile
from ..xmldump_flat import xmldump_flat
from ..xmlmodel import Hwp5File


logger = logging.getLogger(__name__)


def xmldump_nested(hwp5file, output, embedbin=False, xml_declaration=True,
                   id_assigner=None):
    kwargs = dict(embedbin=embedbin)
    if id_assigner is not None:
        kwargs['id_assigner'] = id_assigner
    dump = hwp5file.xmlevents(**kwargs).dump
    dump = partial(dump, xml_declaration=xml_declaration)
    dump(output)


def load_baseline_index(path):
    if path is None or not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        document = json.load(f)
    return index_from_jsonable(document)


def main(args):
    ''' Transform <hwp5file> into an XML.
    '''

    baseline = load_baseline_index(args.index)
    assigner = StableIdAssigner(baseline)

    fmt = args.format or 'nested'
    if fmt == 'flat':
        xmldump = partial(
            xmldump_flat,
            xml_declaration=not args.no_xml_decl
        )
    elif fmt == 'nested':
        xmldump = partial(
            xmldump_nested,
            xml_declaration=not args.no_xml_decl,
            embedbin=args.embedbin,
            id_assigner=assigner,
        )

    open_dest = make_open_dest_file(args.output)
    open_dest = wrap_open_dest_for_tty(open_dest, [
        pager(),
        syntaxhighlight('application/xml'),
    ] + ([
        xmllint(format=True),
    ] if not args.no_validate_wellformed else []))
    open_dest = wrap_open_dest(open_dest, [
        xmllint(encode='utf-8'),
        xmllint(c14n=True),
    ] if not args.no_validate_wellformed else [])

    hwp5file = Hwp5File(args.hwp5file)
    with open_dest() as output:
        xmldump(hwp5file, output)

    if args.index:
        if not assigner.records:
            logger.warning('--index has no effect with --format=flat')
            return
        if baseline is not None and args.revision:
            correspondence = reconcile(baseline, assigner.records)
            with open(args.revision, 'w') as f:
                json.dump(correspondence, f, indent=2, sort_keys=True)
        index = index_to_jsonable(assigner.records)
        with open(args.index, 'w') as f:
            json.dump(index, f, indent=2, sort_keys=True)


def xml_argparser(subparsers, _):
    parser = subparsers.add_parser(
        'xml',
        help=_(
           'Transform .hwp files into an XML.'
        ),
        description=_(
           'Transform <hwp5file> into an XML.'
        ),
    )
    parser.add_argument(
        'hwp5file',
        metavar='<hwp5file>',
        help=_('.hwp file to analyze'),
    )
    parser.add_argument(
        '--embedbin',
        action='store_true',
        help=_('Embed BinData/* streams in the output XML.'),
    )
    parser.add_argument(
        '--no-xml-decl',
        action='store_true',
        help=_('Do not output <?xml ... ?> XML declaration.'),
    )
    parser.add_argument(
       '--output',
       metavar='<file>',
       help=_('Output filename.'),
    )
    parser.add_argument(
       '--format',
       metavar='<format>',
       help=_('"flat", "nested" (default: "nested")'),
    )
    parser.add_argument(
       '--no-validate-wellformed',
       action='store_true',
       help=_('Do not validate well-formedness of output.'),
    )
    parser.add_argument(
       '--index',
       metavar='<file>',
       help=_('Element index file of collected results. If it exists, '
              'it is used as the baseline so matched elements keep '
              'their original ids; it is then updated with the element '
              'records collected from this export.'),
    )
    parser.add_argument(
       '--revision',
       metavar='<file>',
       help=_('Write the revision correspondence between the '
              'pre-existing --index baseline and this export.'),
    )
    parser.set_defaults(func=main)
    return parser
