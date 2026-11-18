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
from contextlib import closing
import json

from ..attachments import inventory_from_hwp5file
from ..utils import make_open_dest_file
from ..xmlmodel import Hwp5File



def _json_bytes(inventory):
    text = json.dumps(inventory, indent=2, ensure_ascii=False,
                      default=str)
    if not text.endswith('\n'):
        text += '\n'
    return text.encode('utf-8')


def _table_text(inventory):
    lines = []
    header = ('ID', 'TYPE', 'STORAGE ID', 'NAME', 'STATUS', 'REFS')
    lines.append('%-4s %-10s %-11s %-16s %-10s %s' % header)
    for entry in inventory.get('attachments', ()):
        lines.append('%-4s %-10s %-11s %-16s %-10s %d' % (
            entry.get('bindata_id'),
            entry.get('storage_type'),
            entry.get('storage_id') or '-',
            entry.get('name') or '-',
            entry.get('status'),
            len(entry.get('references') or ()),
        ))
    unresolved = inventory.get('unresolved_references') or ()
    if unresolved:
        lines.append('')
        lines.append('unresolved references: %d' % len(unresolved))
        for item in unresolved:
            reference = item.get('reference') or {}
            lines.append('  bindata-id=%s %s' % (
                item.get('bindata_id'),
                reference.get('record_path') or '',
            ))
    pending = inventory.get('pending_links') or ()
    if pending:
        lines.append('')
        lines.append('pending external links: %d' % len(pending))
        for entry in pending:
            lines.append('  %s -> %s' % (
                entry.get('name') or '-',
                entry.get('resolved_path') or '-',
            ))
    return '\n'.join(lines) + '\n'


def main(args):
    hwp5file = Hwp5File(args.hwp5file)
    with closing(hwp5file):
        inventory = inventory_from_hwp5file(
            hwp5file,
            document_path=args.hwp5file,
            base_dir=args.base_dir,
            hash_content=not args.no_hash,
        )

    if args.format == 'table':
        output = _table_text(inventory)
    else:
        output = _json_bytes(inventory)

    open_dest = make_open_dest_file(args.output)
    with open_dest() as f:
        if not isinstance(output, bytes):
            output = output.encode('utf-8')
        f.write(output)


def attachments_argparser(subparsers, _):
    parser = subparsers.add_parser(
        'attachments',
        help=_('List binary attachments declared by .hwp files.'),
        description=_(
            'List the binary attachments declared in <hwp5file>.\n'
            'Each HWPTAG_BIN_DATA record becomes one entry; body '
            'references are kept per occurrence.'
        ),
    )
    parser.add_argument(
        'hwp5file',
        metavar='<hwp5file>',
        help=_('.hwp file to analyze'),
    )
    parser.add_argument(
        '--format',
        choices=('json', 'table'),
        default='json',
        help=_('Output format (default: json).'),
    )
    parser.add_argument(
        '--output',
        metavar='<file>',
        help=_('Output filename.'),
    )
    parser.add_argument(
        '--base-dir',
        metavar='<directory>',
        help=_(
            'Directory used to resolve document-relative external links. '
            'Defaults to the directory of <hwp5file>.'
        ),
    )
    parser.add_argument(
        '--no-hash',
        action='store_true',
        help=_('Do not read attachment content to compute SHA-256.'),
    )
    parser.set_defaults(func=main)
    return parser
