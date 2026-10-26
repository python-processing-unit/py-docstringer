"""Command-line interface for py-docstringer."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from openrouter import OpenRouter

from .app import Summary, document_folder
from .batch import OpenRouterTransport
from .errors import ConfigurationError, DocstringerError
from .formatting import pluralize

DEFAULT_MODEL = "openai/gpt-6-luna"


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface."""

    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.poll_interval <= 0:
        parser.error("--poll-interval must be greater than zero")
    if args.timeout_hours <= 0:
        parser.error("--timeout-hours must be greater than zero")
    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than zero")
    if args.max_line_length <= 0:
        parser.error("--max-line-length must be greater than zero")

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(message)s",
    )

    try:
        summary = _run(args)
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130
    except DocstringerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    _print_summary(summary)
    return 1 if summary.failed or summary.skipped else 0


def _run(args: argparse.Namespace) -> Summary:
    """Run documentation generation using the supplied command-line arguments.

    Args:
        args (argparse.Namespace): Options for the folder, model, polling interval,
            timeout, batch size, overwrite behavior, and dry-run mode.

    Returns:
        Summary: The result returned by `document_folder`.

    Raises:
        ConfigurationError: If `OPENROUTER_API_KEY` is not set and dry-run mode is
            disabled.
    """
    options = {
        "poll_interval": args.poll_interval,
        "timeout": args.timeout_hours * 3600,
        "batch_size": args.batch_size,
        "max_line_length": args.max_line_length,
        "overwrite": args.overwrite,
    }

    if args.dry_run:
        return document_folder(args.folder, None, dry_run=True, **options)

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ConfigurationError("OPENROUTER_API_KEY is not set")

    with OpenRouter(api_key=api_key) as client:
        transport = OpenRouterTransport(client, args.model)
        return document_folder(args.folder, transport, **options)


def _print_summary(summary: Summary) -> None:
    """Print a summary of the documentation run.

    Args:
        summary (Summary): Counts and status details for the documentation run.
    """
    files = pluralize(summary.files, "file", "files")
    definitions = pluralize(summary.definitions, "definition", "definitions")
    docstrings = pluralize(summary.written, "docstring", "docstrings")
    skipped = (
        f"; {summary.skipped} {pluralize(summary.skipped, 'file', 'files')} skipped"
        if summary.skipped
        else ""
    )
    if summary.dry_run:
        print(
            f"Would document {summary.definitions} {definitions} "
            f"in {summary.files} {files}{skipped}."
        )
    else:
        print(
            f"Added {summary.written} {docstrings} across {summary.files} {files}; "
            f"{summary.failed} failed{skipped}."
        )


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the command-line argument parser.

    The parser defines options for the folder to scan, model, batch polling and timeout settings, batch size, overwriting docstrings, dry runs, and quiet output.

    Returns:
        argparse.ArgumentParser: The configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="add-docstrings",
        description=(
            "Add Google-style docstrings to every function, class, and method below "
            "a folder using OpenRouter batch requests."
        ),
    )
    parser.add_argument("folder", type=Path, help="folder to scan recursively for .py files")
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL),
        help=f"OpenRouter model slug (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=30.0,
        help="seconds between batch status checks (default: 30)",
    )
    parser.add_argument(
        "--timeout-hours",
        type=float,
        default=26.0,
        help="maximum hours to wait for each batch (default: 26)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="maximum requests per batch (default: 500)",
    )
    parser.add_argument(
        "--max-line-length",
        type=int,
        default=88,
        help="maximum docstring source line length (default: 88)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing docstrings instead of leaving them unchanged",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="list definitions without calling OpenRouter or editing files",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="only print warnings and errors")
    return parser
