#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
from io import BufferedReader
from typing import Union, cast

from pyshacl import __version__, shacl_rules
from pyshacl.cli import ShowVersion
from pyshacl.errors import (
    ConstraintLoadError,
    ReportableRuntimeError,
    RuleLoadError,
    ShapeLoadError,
    ValidationFailure,
)

parser = argparse.ArgumentParser(
    description='PySHACL {} SHACL Rules Expander command line tool.'.format(str(__version__))
)
parser.add_argument(
    'data',
    metavar='DataGraph',
    help='The file or endpoint containing the Target Data Graph.',
    default=None,
    nargs='?',
)
parser.add_argument(
    '-s',
    '--shapes',
    '--shacl',
    dest='shacl',
    action='store',
    nargs='?',
    help='A file containing the SHACL Shapes Graph.',
)
parser.add_argument(
    '-e',
    '--ont-graph',
    dest='ont',
    action='store',
    nargs='?',
    help='A file path or URL to a document containing extra ontological information. '
    'RDFS and OWL definitions from this are used to inoculate the DataGraph.',
)
parser.add_argument(
    '-i',
    '--inference',
    dest='inference',
    action='store',
    default='none',
    choices=('none', 'rdfs', 'owlrl', 'both'),
    help='Choose a type of inferencing to run against the Data Graph before validating.',
)
parser.add_argument(
    '-im',
    '--imports',
    dest='imports',
    action='store_true',
    default=False,
    help='Allow import of sub-graphs defined in statements with owl:imports.',
)
parser.add_argument(
    '-j',
    '--js',
    dest='js',
    action='store_true',
    default=False,
    help='Enable features from the SHACL-JS Specification.',
)
parser.add_argument(
    '-it',
    '--iterate-rules',
    dest='iterate_rules',
    action='store_true',
    default=False,
    help="Run all Shape's SHACL Rules in ordered rounds until the data_graph reaches a steady state.",
)
parser.add_argument(
    '-mr',
    '--max-rule-rounds',
    dest='max_rule_rounds',
    action='store',
    type=int,
    default=None,
    help='Maximum number of ordered SHACL Rules closure rounds. If this limit is reached while '
    'rules still add new triples, the run is reported as incomplete (exit code 4).',
)
parser.add_argument(
    '--rules-report',
    dest='rules_report',
    action='store',
    type=argparse.FileType('w'),
    default=None,
    help='Write a JSON report of the rule closure, including per-round evidence, to this file.',
)
parser.add_argument(
    '-d',
    '--debug',
    dest='debug',
    action='store_true',
    default=False,
    help='Output additional verbose runtime messages.',
)
parser.add_argument(
    '--focus',
    dest='focus',
    action='store',
    help='Optional IRIs of focus nodes from the DataGraph, the shapes will validate only these node. Comma-separated list.',
    nargs="?",
    default=None,
)
parser.add_argument(
    '--shape',
    dest='shape',
    action='store',
    help='Optional IRIs of a NodeShape or PropertyShape from the SHACL ShapesGraph, only these shapes will be used to validate the DataGraph. Comma-separated list.',
    nargs="?",
    default=None,
)
parser.add_argument(
    '-f',
    '--format',
    dest='format',
    action='store',
    help='Choose an output format. Default is "trig" for Datasets and "turtle" for Graphs.',
    default='auto',
    choices=('auto', 'turtle', 'xml', 'trig', 'json-ld', 'nt', 'n3', 'nquads'),
)
parser.add_argument(
    '-df',
    '--data-file-format',
    dest='data_file_format',
    action='store',
    help='Explicitly state the RDF File format of the input DataGraph file. Default="auto".',
    default='auto',
    choices=('auto', 'turtle', 'xml', 'trig', 'json-ld', 'nt', 'n3', 'nquads'),
)
parser.add_argument(
    '-sf',
    '--shacl-file-format',
    dest='shacl_file_format',
    action='store',
    help='Explicitly state the RDF File format of the input SHACL file. Default="auto".',
    default='auto',
    choices=('auto', 'turtle', 'xml', 'trig', 'json-ld', 'nt', 'n3', 'nquads'),
)
parser.add_argument(
    '-ef',
    '--ont-file-format',
    dest='ont_file_format',
    action='store',
    help='Explicitly state the RDF File format of the extra ontology file. Default="auto".',
    default='auto',
    choices=('auto', 'turtle', 'xml', 'trig', 'json-ld', 'nt', 'n3', 'nquads'),
)
parser.add_argument('-V', '--version', action=ShowVersion, help='Show PySHACL version and exit.')
parser.add_argument(
    '-o',
    '--output',
    dest='output',
    nargs='?',
    type=argparse.FileType('w'),
    help='Send output to a file (defaults to stdout).',
    default=sys.stdout,
)
# parser.add_argument('-h', '--help', action="help", help='Show this help text.')


def main(prog: Union[str, None] = None) -> None:
    if prog is not None and len(prog) > 0:
        parser.prog = prog

    args = parser.parse_args()
    if not args.data:
        # No datafile give, and not starting in server mode.
        sys.stderr.write('Input Error. No DataGraph file or endpoint supplied.\n')
        parser.print_usage(sys.stderr)
        sys.exit(1)
    runner_kwargs = {
        'debug': args.debug,
        'serialize_expanded_graph': True,
    }
    data_file = None
    data_graph: Union[BufferedReader, str]

    try:
        data_file = open(args.data, 'rb')
    except FileNotFoundError:
        sys.stderr.write('Input Error. DataGraph file not found.\n')
        sys.exit(1)
    except PermissionError:
        sys.stderr.write('Input Error. DataGraph file not readable.\n')
        sys.exit(1)
    else:
        # NOTE: This cast is not necessary in Python >= 3.10.
        data_graph = cast(BufferedReader, data_file)
    if args.shacl is not None:
        runner_kwargs['shacl_graph'] = args.shacl
    if args.ont is not None:
        runner_kwargs['ont_graph'] = args.ont
    if args.inference != 'none':
        runner_kwargs['inference'] = args.inference
    if args.imports:
        runner_kwargs['do_owl_imports'] = True
    if args.js:
        runner_kwargs['js'] = True
    if args.focus:
        runner_kwargs['focus_nodes'] = [_f.strip() for _f in args.focus.split(',')]
    if args.shape:
        runner_kwargs['use_shapes'] = [_s.strip() for _s in args.shape.split(',')]
    if args.iterate_rules:
        runner_kwargs['iterate_rules'] = True
    if args.max_rule_rounds is not None:
        runner_kwargs['max_rule_rounds'] = args.max_rule_rounds
    if args.rules_report is not None:
        runner_kwargs['return_rules_report'] = True
    if args.shacl_file_format:
        _f: str = args.shacl_file_format
        if _f != "auto":
            runner_kwargs['shacl_graph_format'] = _f
    if args.ont_file_format:
        _f = args.ont_file_format
        if _f != "auto":
            runner_kwargs['ont_graph_format'] = _f
    if args.data_file_format:
        _f = args.data_file_format
        if _f != "auto":
            runner_kwargs['data_graph_format'] = _f
    if args.format != "auto":
        runner_kwargs['serialize_expanded_graph_format'] = args.format
    exit_code: Union[int, None] = None
    rules_report = None
    try:
        output_txt = shacl_rules(data_graph, **runner_kwargs)
        if args.rules_report is not None and isinstance(output_txt, tuple) and len(output_txt) == 2:
            output_txt, rules_report = output_txt
        if isinstance(output_txt, BaseException):
            raise output_txt
    except ValidationFailure as vf:
        args.output.write("Rules Runner generated a Validation Failure result:\n")
        args.output.write(str(vf.message))
        args.output.write("\n")
        exit_code = 1
    except ShapeLoadError as sle:
        sys.stderr.write("Rules Runner encountered a Shape Load Error:\n")
        sys.stderr.write(str(sle))
        exit_code = 2
    except ConstraintLoadError as cle:
        sys.stderr.write("Rules Runner encountered a Constraint Load Error:\n")
        sys.stderr.write(str(cle))
        exit_code = 2
    except RuleLoadError as rle:
        sys.stderr.write("Rules Runner encountered a Rule Load Error:\n")
        sys.stderr.write(str(rle))
        exit_code = 2
    except ReportableRuntimeError as rre:
        sys.stderr.write("Rules Runner encountered a Runtime Error:\n")
        sys.stderr.write(str(rre.message))
        sys.stderr.write("\nIf you believe this is a bug in pyshacl, open an Issue on the pyshacl github page.\n")
        exit_code = 2
    except NotImplementedError as nie:
        sys.stderr.write("Rules Runner feature is not implemented:\n")
        if len(nie.args) > 0:
            sys.stderr.write(str(nie.args[0]))
        else:
            sys.stderr.write("No message provided.")
        sys.stderr.write("\nIf your use-case requires this feature, open an Issue on the pyshacl github page.\n")
        exit_code = 3
    except RuntimeError as re:
        import traceback

        traceback.print_tb(re.__traceback__)
        sys.stderr.write(
            "\n\nRules Runner encountered a Runtime Error. Please report this to the PySHACL issue tracker.\n"
        )
        exit_code = 2
    finally:
        if data_file is not None:
            try:
                data_file.close()
            except Exception as e:
                sys.stderr.write("Error closing data file:\n")
                sys.stderr.write(str(e))
        if exit_code is not None:
            sys.exit(exit_code)

        if isinstance(output_txt, bytes):
            output_unicode = output_txt.decode('utf-8')
        else:
            output_unicode = output_txt
        args.output.write(output_unicode)
    if args.rules_report is not None:
        if rules_report is None:
            # The report was requested but rules closure did not run (e.g. no
            # ``--iterate-rules``). Say so explicitly instead of emitting an
            # empty object that could be mistaken for a converged run.
            args.rules_report.write(
                '{\n  "status": "not_run",\n  "reason": "rules_closure_not_requested"\n}\n'
            )
        else:
            args.rules_report.write(rules_report.to_json(indent=2))
            args.rules_report.write("\n")
            if rules_report.get("status") == "incomplete":
                sys.stderr.write(
                    "Rules Runner reached the maximum number of rounds ({}) while rules were still adding "
                    "new triples. The expanded graph is incomplete.\n".format(rules_report.get("max_rounds"))
                )
                args.rules_report.close()
                args.output.close()
                sys.exit(4)
        args.rules_report.close()
    args.output.close()
    sys.exit(0)


if __name__ == "__main__":
    main()
