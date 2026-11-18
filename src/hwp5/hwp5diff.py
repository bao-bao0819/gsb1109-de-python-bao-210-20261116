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
'''Compare two revisions of an HWPv5 document.

The report separates three kinds of change that HWP stores in different
places:

* resource reordering in DocInfo (ids move, meaning stays),
* resource content changes in DocInfo (a font/style definition was edited),
* body changes (paragraph text or direct references were edited).

Paragraphs are only re-indexed when the comparison proves they were
affected; all other paragraphs keep their original language/style index.
'''
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from argparse import ArgumentParser
import gettext
import json
import logging
import os
import sys

from .cli import init_logger
from .dataio import ParseError
from .errors import InvalidHwp5FileError
from .revision.compare import compare_hwp5
from .utils import make_open_dest_file


PY3 = sys.version_info.major == 3
logger = logging.getLogger(__name__)
locale_dir = os.path.join(os.path.dirname(__file__), 'locale')
locale_dir = os.path.abspath(locale_dir)
t = gettext.translation('hwp5diff', locale_dir, fallback=True)
if PY3:
    _ = t.gettext
else:
    _ = t.ugettext  # pragma: no cover


def main():
    argparser = main_argparser()
    args = argparser.parse_args()
    init_logger(args)
    try:
        return run(args)
    except InvalidHwp5FileError as e:
        logger.error('%s', e)
        raise SystemExit(1)
    except ParseError as e:
        e.print_to_logger(logger)
        raise SystemExit(1)


def main_argparser():
    parser = ArgumentParser(
        prog='hwp5diff',
        description=_('Compare two revisions of HWPv5 documents.'),
    )

    def add_common(parser):
        parser.add_argument(
            '--loglevel',
            help=_('Set log level.'),
        )
        parser.add_argument(
            '--logfile',
            help=_('Set log file.'),
        )
        parser.add_argument(
            '--output',
            metavar='<file>',
            help=_('Output filename (default: standard output).'),
        )
        parser.add_argument(
            '--format',
            choices=('json', 'text'),
            default='text',
            help=_('Output format.'),
        )
        parser.add_argument(
            '--indices',
            action='store_true',
            help=_('Include rebuilt language/style indices per paragraph.'),
        )
        parser.add_argument(
            '--verbose',
            action='store_true',
            help=_('Include unchanged and remapped paragraphs in text '
                   'output.'),
        )

    add_common(parser)
    parser.add_argument(
        'old_hwp5file',
        metavar='<old-hwp5file>',
        help=_('Old revision (.hwp).'),
    )
    parser.add_argument(
        'new_hwp5file',
        metavar='<new-hwp5file>',
        help=_('New revision (.hwp).'),
    )
    parser.set_defaults(func=run)
    return parser


def diff_argparser(subparsers, _):
    '''Register the compare command under hwp5proc.'''
    parser = subparsers.add_parser(
        'diff',
        help=_('Compare two revisions of .hwp files.'),
        description=_(
            'Compare <old-hwp5file> and <new-hwp5file>, separating DocInfo '
            'resource reordering, resource content changes, and body text '
            'changes.'
        ),
    )
    parser.add_argument(
        '--loglevel',
        help=_('Set log level.'),
    )
    parser.add_argument(
        '--logfile',
        help=_('Set log file.'),
    )
    parser.add_argument(
        '--output',
        metavar='<file>',
        help=_('Output filename (default: standard output).'),
    )
    parser.add_argument(
        '--format',
        choices=('json', 'text'),
        default='text',
        help=_('Output format.'),
    )
    parser.add_argument(
        '--indices',
        action='store_true',
        help=_('Include rebuilt language/style indices per paragraph.'),
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help=_('Include unchanged and remapped paragraphs in text output.'),
    )
    parser.add_argument(
        'old_hwp5file',
        metavar='<old-hwp5file>',
        help=_('Old revision (.hwp).'),
    )
    parser.add_argument(
        'new_hwp5file',
        metavar='<new-hwp5file>',
        help=_('New revision (.hwp).'),
    )
    parser.set_defaults(func=run)
    return parser


def run(args):
    comparison = compare_hwp5(args.old_hwp5file, args.new_hwp5file)
    report = comparison.to_dict(include_indices=args.indices)

    open_dest = make_open_dest_file(args.output)
    with open_dest() as outfile:
        if args.format == 'json':
            _write_json(outfile, report)
        else:
            _write_text(outfile, report, verbose=args.verbose)
    return 0


def _write_json(outfile, report):
    text = json.dumps(report, ensure_ascii=False, indent=2,
                      sort_keys=True, default=_json_default)
    outfile.write(text.encode('utf-8'))
    outfile.write(b'\n')


def _json_default(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return repr(value)


def _write_text(outfile, report, verbose=False):
    lines = []
    summary = report['summary']
    categories = summary['categories']
    lines.append(
        'resources: reordered={resource_reorder} '
        'content-changed={resource_content_change} '
        'dependency-changed={resource_dependency_change} '
        'reorder-caused={resource_reorder_caused}'.format(**categories))
    lines.append('body: changed={body_change} '
                 'reorder-affected={resource_reorder_affected} '
                 'affected-paragraphs={affected_paragraphs}'.format(
                     **categories))

    if report['resource_reorder']:
        lines.append('')
        lines.append('RESOURCE REORDER')
        for item in report['resource_reorder']:
            lines.append('  {kind} {label!r}: {old_index} -> {new_index}'
                         .format(**item))

    if report['resource_content_change']:
        lines.append('')
        lines.append('RESOURCE CONTENT CHANGE')
        for item in report['resource_content_change']:
            lines.append('  {kind} {label!r}: {old_index} -> {new_index}'
                         .format(**item))
            for change in item.get('own_changes', []):
                lines.append('    {path}: {old!r} -> {new!r}'.format(**change))
            for user in item.get('used_by', []):
                lines.append('    used by section {section} paragraph '
                             '{old_ordinal}/{new_ordinal}'.format(**user))

    if report.get('reorder_affected'):
        lines.append('')
        lines.append('RESOURCE REORDER AFFECTED PARAGRAPHS')
        for item in report['reorder_affected']:
            lines.append('  section {section} paragraph '
                         '{old_ordinal}/{new_ordinal}'.format(**item))
            if item.get('numeric_ids_changed'):
                lines.append('    ids: {old_ids} -> {new_ids}'.format(**item))

    if report['resource_dependency_change']:
        lines.append('')
        lines.append('RESOURCE CONTENT CHANGE (VIA DEPENDENCY)')
        for item in report['resource_dependency_change']:
            lines.append('  {kind} {label!r}: {old_index} -> {new_index}'
                         .format(**item))
            for dependency in item.get('changed_dependencies', []):
                lines.append('    depends on changed {kind} '
                             '{old_index} -> {new_index}'.format(**dependency))

    paragraphs = report['paragraphs']
    displayed = [item for item in paragraphs
                 if verbose or item['affected']]
    if displayed:
        lines.append('')
        lines.append('BODY')
        for item in displayed:
            lines.extend(_paragraph_lines(item))

    text = '\n'.join(lines) + '\n'
    outfile.write(text.encode('utf-8'))


def _paragraph_lines(item):
    lines = []
    header = '  section {section} [{status}]'.format(**item)
    if item['status'] == 'remapped':
        header += ' ids rewritten, meaning preserved'
    if item.get('numeric_ids_changed'):
        header += ' (numeric ids changed)'
    lines.append(header)
    old_text = item.get('old_text')
    new_text = item.get('new_text')
    if old_text == new_text:
        lines.append('    text: {!r}'.format(new_text))
    else:
        if old_text is not None:
            lines.append('    - {!r}'.format(old_text))
        if new_text is not None:
            lines.append('    + {!r}'.format(new_text))
    for change in item.get('changes', [])[:20]:
        lines.append('    {path}: {old!r} -> {new!r}'.format(**change))
    if item.get('index_rebuilt'):
        style = item.get('style_index') or {}
        languages = [lang for lang, runs in
                     (item.get('language_index') or {}).items() if runs]
        lines.append('    rebuilt style={!r} parashape={!r} languages={}'
                     .format(style.get('style'),
                             style.get('parashape_id'),
                             ','.join(sorted(languages)) or '-'))
    return lines


if __name__ == '__main__':
    main()
