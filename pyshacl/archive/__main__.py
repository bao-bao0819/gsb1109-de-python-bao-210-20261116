# -*- coding: utf-8 -*-
"""Command line interface for offline validation archives.

Examples:
    python -m pyshacl.archive capture ./out \\
        --data data.ttl --shapes shapes/root.ttl --ontology ontology/root.ttl \\
        --inference rdfs --owl-imports
    python -m pyshacl.archive replay ./out
    python -m pyshacl.archive check ./out
"""
import argparse
import json
import sys
from typing import List, Optional

from .archive import capture_validation, replay_comparison
from .replay import IncompleteReplayError, detect_import_cycles, find_missing_imports
from .store import ArchiveStore


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m pyshacl.archive")
    sub = parser.add_subparsers(dest="command", required=True)

    capture = sub.add_parser("capture", help="Run a validation and write its archive")
    capture.add_argument("archive_dir")
    capture.add_argument("--data", required=True)
    capture.add_argument("--shapes", default=None)
    capture.add_argument("--ontology", default=None)
    capture.add_argument("--inference", default="none", choices=("none", "rdfs", "owlrl", "both"))
    capture.add_argument("--owl-imports", action="store_true")
    capture.add_argument("--advanced", action="store_true")
    capture.add_argument("--allow-infos", action="store_true")
    capture.add_argument("--allow-warnings", action="store_true")

    replay = sub.add_parser("replay", help="Re-run an archived validation offline")
    replay.add_argument("archive_dir")
    replay.add_argument("--exclude", action="append", default=[], help="Document id to treat as removed")

    check = sub.add_parser("check", help="Inspect import cycles and completeness without validating")
    check.add_argument("archive_dir")
    return parser


def _run_capture(args: argparse.Namespace) -> int:
    result = capture_validation(
        args.archive_dir,
        args.data,
        shacl_graph=args.shapes,
        ont_graph=args.ontology,
        inference=args.inference,
        do_owl_imports=args.owl_imports,
        advanced=args.advanced,
        allow_infos=args.allow_infos,
        allow_warnings=args.allow_warnings,
    )
    payload = {
        "archive_dir": str(result.archive_dir),
        "unique_blobs": result.unique_blobs,
        "stored_bytes": result.stored_bytes,
        "report": result.report.summary(),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def _run_replay(args: argparse.Namespace) -> int:
    try:
        verdict = replay_comparison(args.archive_dir, exclude_document_ids=args.exclude)
    except IncompleteReplayError as error:
        payload = {
            "complete": False,
            "reason": error.reason,
            "missing_imports": error.missing_imports,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 2
    print(json.dumps(verdict, indent=2, ensure_ascii=False))
    return 0 if verdict["report_matches"] and verdict["relative_expansions_match"] else 1


def _run_check(args: argparse.Namespace) -> int:
    store = ArchiveStore(args.archive_dir)
    manifest = store.read_manifest()
    payload = {
        "documents": len(manifest.documents),
        "load_events": len(manifest.load_events),
        "cycles": detect_import_cycles(manifest),
        "missing_imports": find_missing_imports(manifest, store),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if not payload["missing_imports"] else 2


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "capture":
        return _run_capture(args)
    if args.command == "replay":
        return _run_replay(args)
    return _run_check(args)


if __name__ == "__main__":
    sys.exit(main())
