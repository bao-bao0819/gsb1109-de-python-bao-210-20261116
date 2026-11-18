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
''' Collect text occurrences with region evidence.

Page headers/footers are repeated on every applicable page but stored only
once in the document model, and they may carry exactly the same text as the
body. This subcommand reports every occurrence together with its structural
region (body/header/footer) and keeps repeated occurrences separate instead
of deduplicating them by content. When region evidence is missing, candidate
identities and min/max count ranges are reported.

Usage::

    hwp5proc texts [--text=<text>] [--output=<file>]
                   [--loglevel=<loglevel>] [--logfile=<logfile>]
                   <hwp5file>
    hwp5proc texts --help
'''
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
import logging
import sys

from ..dataio import ParseError
from ..textevents import build_report
from ..textevents import collect_header_footer_decorations
from ..textevents import collect_text_occurrences
from ..textevents import report_to_json
from ..utils import make_open_dest_file
from ..xmlmodel import Hwp5File


logger = logging.getLogger(__name__)


def main(args):
    hwp5file = Hwp5File(args.hwp5file)
    try:
        if args.text:
            payload = build_report(hwp5file, args.text)
        else:
            events = list(hwp5file.events())
            payload = {
                'occurrences': collect_text_occurrences(iter(events)),
                'header_footer_decorations':
                    collect_header_footer_decorations(iter(events)),
            }
    except ParseError as e:
        e.print_to_logger(logger)
        raise SystemExit(1)
    finally:
        close = getattr(hwp5file, 'close', None)
        if callable(close):
            close()

    output = report_to_json(payload)
    if args.output:
        open_dest = make_open_dest_file(args.output)
        with open_dest() as dest:
            dest.write(output.encode('utf-8'))
    else:
        if hasattr(sys.stdout, 'buffer'):
            sys.stdout.buffer.write(output.encode('utf-8'))
            sys.stdout.buffer.write(b'\n')
        else:
            sys.stdout.write(output + '\n')


def texts_argparser(subparsers, _):
    parser = subparsers.add_parser(
        'texts',
        help=_(
            'Collect text occurrences, distinguishing page headers/footers '
            'from body text.'
        ),
        description=_(
            'Collect every text occurrence with its structural region '
            '(body/header/footer), so identical text in repeated page '
            'decorations is not deduplicated from genuine body text.'
        ),
    )
    parser.add_argument(
        'hwp5file',
        metavar='<hwp5file>',
        help=_('.hwp file to analyze'),
    )
    parser.add_argument(
        '--text',
        metavar='<text>',
        help=_(
            'Only report occurrences containing this text, with identity '
            'candidates and count ranges.'
        ),
    )
    parser.add_argument(
        '--output',
        metavar='<file>',
        help=_('Output filename (default: standard output).'),
    )
    parser.set_defaults(func=main)
    return parser
