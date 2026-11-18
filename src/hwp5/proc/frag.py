# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import os.path

from ..frag import extract_fragment
from ..frag import validate_fragment


def parse_paragraph_indexes(spec):
    indexes = []
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        indexes.append(int(part))
    if not indexes:
        raise ValueError('no paragraph index given')
    return indexes


def main(args):
    filename = args.hwp5file
    paragraph_indexes = parse_paragraph_indexes(args.paragraphs)
    outdir = args.out_directory
    if outdir is None:
        outdir, ext = os.path.splitext(os.path.basename(filename))
        outdir += '.frag'
    manifest = extract_fragment(filename, outdir, paragraph_indexes,
                                section_idx=args.section)
    problems = validate_fragment(outdir)
    if problems:
        for problem in problems:
            print('validation: %s' % problem)
    print('fragment package: %s' % outdir)
    print('paragraphs: %s' % ', '.join(str(i) for i in paragraph_indexes))
    print('complete: %s' % manifest['complete'])
    for reason in manifest['incomplete_reasons']:
        print('incomplete: %s' % reason)
    for warning in manifest['warnings']:
        print('warning: %s' % warning)
    if problems:
        raise SystemExit(1)


def frag_argparser(subparsers, _):
    parser = subparsers.add_parser(
        'frag',
        help=_(
            'Extract selected paragraphs into a self-contained fragment '
            'package.'
        ),
        description=_(
            'Extract the given paragraphs of <hwp5file> (with their '
            'tables and other nested objects) into <out-directory>: a '
            'self-contained package with only the fonts, styles, '
            'numberings and embedded objects the selected paragraphs '
            'actually depend on.  Resource references are renumbered; '
            'the correspondence to the original resources is recorded '
            'in manifest.json.  If unknown controls may reference '
            'uncollected resources, the package is marked as incomplete '
            'instead of being claimed self-sufficient.'
        )
    )
    parser.add_argument(
        'hwp5file',
        metavar='<hwp5file>',
        help=_('.hwp file to extract paragraphs from'),
    )
    parser.add_argument(
        '-p', '--paragraphs',
        required=True,
        metavar='<indexes>',
        help=_('comma-separated 0-based paragraph indexes to extract, '
               'e.g. -p 0,1,2'),
    )
    parser.add_argument(
        '-s', '--section',
        type=int,
        default=0,
        metavar='<section>',
        help=_('section index to extract paragraphs from (default: 0)'),
    )
    parser.add_argument(
        '-o', '--out-directory',
        metavar='<out-directory>',
        help=_('output directory of the fragment package '
               '(default: <hwp5file>.frag)'),
    )
    parser.set_defaults(func=main)
    return parser
