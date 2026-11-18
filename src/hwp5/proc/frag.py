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
import json
import sys

from .. import frag
from ..errors import InvalidHwp5FileError


def main(args):
    try:
        if args.frag_command == 'delete':
            return delete(args)
        return inspect(args)
    except (frag.FragError, InvalidHwp5FileError) as e:
        sys.stderr.write('%s\n' % e)
        raise SystemExit(1)


def inspect(args):
    inventory = frag.Inventory(frag.open_fragment(args.archive))
    if args.json:
        print(json.dumps(inventory.to_report(), indent=2, sort_keys=True))
    else:
        print(frag.format_inventory_text(inventory))
    return 0


def delete(args):
    selection = args.select or []
    report = frag.delete_annotations(
        args.archive,
        args.destination,
        selection=selection,
        select_all=args.all,
        texts=args.text or [],
        field_ids=args.field_id or [],
        dry_run=args.dry_run,
        in_place=args.in_place,
        delete_unreferenced_resources=args.delete_unreferenced_resources,
    )
    if args.json or args.dry_run:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print('deleted: %d' % len(report['deleted']))
        for item in report['deleted']:
            print('  %s' % item)
        for refusal in report['refused']:
            print('refused: %s (%s)' % (
                refusal.get('id'), '; '.join(refusal.get('reasons', []))))
    return 0


def frag_argparser(subparsers, _):
    parser = subparsers.add_parser(
        'frag',
        help=_('Inspect or edit HWP5 fragment archives'),
        description=_(
            'Inspect memo annotations in a HWP5 fragment archive and '
            'remove selected annotations without changing body text.'
        ),
    )
    commands = parser.add_subparsers(dest='frag_command')

    list_parser = commands.add_parser(
        'list',
        help=_('List annotations, their paired markers and resources'),
    )
    list_parser.add_argument(
        'archive',
        metavar='<archive>',
        help=_('Fragment directory or HWP5 file'),
    )
    list_parser.add_argument(
        '--json',
        action='store_true',
        help=_('Print the inventory as JSON'),
    )
    list_parser.set_defaults(func=main)

    delete_parser = commands.add_parser(
        'delete',
        help=_('Remove selected annotations into a new archive copy'),
    )
    delete_parser.add_argument(
        'archive',
        metavar='<archive>',
        help=_('Source fragment directory or HWP5 file'),
    )
    delete_parser.add_argument(
        'destination',
        nargs='?',
        metavar='<destination>',
        help=_('Destination fragment directory'),
    )
    delete_parser.add_argument(
        '--select',
        action='append',
        metavar='<annotation>',
        help=_(
            'Annotation id, index, text or field id. '
            'May be repeated or comma-separated.'
        ),
    )
    delete_parser.add_argument(
        '--all',
        action='store_true',
        help=_('Select every safely deletable annotation'),
    )
    delete_parser.add_argument(
        '--text',
        action='append',
        metavar='<text>',
        help=_('Select annotations whose anchor text contains <text>'),
    )
    delete_parser.add_argument(
        '--field-id',
        action='append',
        metavar='<id>',
        help=_('Select annotations with the given field id'),
    )
    delete_parser.add_argument(
        '--dry-run',
        action='store_true',
        help=_('Print the deletion plan without writing anything'),
    )
    delete_parser.add_argument(
        '--in-place',
        action='store_true',
        help=_('Edit the source archive in place'),
    )
    delete_parser.add_argument(
        '--delete-unreferenced-resources',
        action='store_true',
        help=_(
            'Also delete BinData streams that were only referenced by '
            'removed note bodies. Disabled by default because DocInfo '
            'resource tables are left unchanged.'
        ),
    )
    delete_parser.add_argument(
        '--json',
        action='store_true',
        help=_('Print the deletion report as JSON'),
    )
    delete_parser.set_defaults(func=main)
    return parser
