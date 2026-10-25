"""Orchestrate discovery, batch generation, and source updates."""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import tokenize
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .batch import BatchRequest, BatchTransport, run_batches
from .discovery import Definition, discover_file, discover_source, read_source
from .editing import apply_docstrings, normalize_docstring
from .errors import ConfigurationError, DocstringApplicationError
from .formatting import pluralize
from .prompts import SYSTEM_PROMPT, build_prompt

logger = logging.getLogger(__name__)

_EXCLUDED_DIRECTORIES = frozenset(
    {
        ".git",
        ".hg",
        ".nox",
        ".pytest_cache",
        ".ruff_cache",
        ".svn",
        ".tox",
        ".ty_cache",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "env",
        "venv",
    }
)


@dataclass(frozen=True, slots=True)
class Summary:
    """Counts from one documentation run."""

    files: int
    definitions: int
    written: int
    failed: int
    skipped: int = 0
    dry_run: bool = False


@dataclass(frozen=True, slots=True)
class _PendingDocstring:
    definition: Definition
    content: str


def find_python_files(folder: Path) -> list[Path]:
    """Return Python files below a folder, excluding common generated directories."""

    files = []
    for path in folder.rglob("*.py"):
        relative_parts = path.relative_to(folder).parts[:-1]
        if any(part in _EXCLUDED_DIRECTORIES for part in relative_parts):
            continue
        files.append(path)
    return sorted(files)


def collect_definitions(
    folder: Path,
    *,
    include_documented: bool = False,
) -> tuple[list[Path], list[Definition], int]:
    """Collect definitions and count unreadable files."""

    files = find_python_files(folder)
    definitions: list[Definition] = []
    skipped = 0
    for path in files:
        try:
            definitions.extend(discover_file(path, include_documented=include_documented))
        except (OSError, SyntaxError, UnicodeDecodeError, tokenize.TokenError) as exc:
            skipped += 1
            logger.warning("Skipping %s: %s", path, exc)
    return files, definitions, skipped


def document_folder(
    folder: Path,
    transport: BatchTransport | None,
    *,
    poll_interval: float,
    timeout: float,
    batch_size: int,
    overwrite: bool = False,
    dry_run: bool = False,
) -> Summary:
    """Add docstrings to definitions below a folder, optionally replacing existing ones."""

    if not folder.is_dir():
        raise ConfigurationError(f"{folder} is not a directory")

    files, definitions, skipped = collect_definitions(
        folder,
        include_documented=overwrite,
    )
    logger.info(
        "Found %d %s in %d %s",
        len(definitions),
        pluralize(len(definitions), "definition", "definitions"),
        len(files),
        pluralize(len(files), "file", "files"),
    )
    if skipped:
        logger.warning(
            "Skipped %d %s that could not be parsed",
            skipped,
            pluralize(skipped, "file", "files"),
        )

    if dry_run:
        for definition in definitions:
            print(f"{definition.path}:{definition.lineno}: {definition.kind} {definition.qualname}")
        return Summary(
            files=len(files),
            definitions=len(definitions),
            written=0,
            failed=0,
            skipped=skipped,
            dry_run=True,
        )

    if transport is None:
        raise ConfigurationError("a batch transport is required unless --dry-run is used")
    if not definitions:
        return Summary(
            files=len(files),
            definitions=0,
            written=0,
            failed=0,
            skipped=skipped,
        )

    requests = [
        BatchRequest(
            custom_id=f"d{index}",
            messages=(
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(definition, overwrite=overwrite)},
            ),
        )
        for index, definition in enumerate(definitions)
    ]
    by_id = {
        request.custom_id: definition
        for request, definition in zip(requests, definitions, strict=True)
    }

    logger.info(
        "Submitting %d %s to OpenRouter",
        len(requests),
        pluralize(len(requests), "request", "requests"),
    )
    results = run_batches(
        transport,
        requests,
        poll_interval=poll_interval,
        timeout=timeout,
        batch_size=batch_size,
    )

    docstrings_by_path: dict[Path, list[_PendingDocstring]] = defaultdict(list)
    failed = 0
    for custom_id, definition in by_id.items():
        result = results[custom_id]
        if result.error:
            failed += 1
            logger.error(
                "%s:%d %s: %s",
                definition.path,
                definition.lineno,
                definition.qualname,
                result.error,
            )
            continue
        if not result.content:
            failed += 1
            logger.error(
                "%s:%d %s: model returned no content",
                definition.path,
                definition.lineno,
                definition.qualname,
            )
            continue
        try:
            docstrings_by_path[definition.path].append(
                _PendingDocstring(
                    definition=definition,
                    content=normalize_docstring(result.content),
                )
            )
        except DocstringApplicationError as exc:
            failed += 1
            logger.error(
                "%s:%d %s: %s",
                definition.path,
                definition.lineno,
                definition.qualname,
                exc,
            )

    written, write_failures = _write_files(docstrings_by_path, overwrite=overwrite)
    failed += write_failures
    return Summary(
        files=len(files),
        definitions=len(definitions),
        written=written,
        failed=failed,
        skipped=skipped,
    )


def _write_files(
    pending: dict[Path, list[_PendingDocstring]],
    *,
    overwrite: bool,
) -> tuple[int, int]:
    written = 0
    failed = 0
    for path, items in sorted(pending.items()):
        try:
            source, encoding = read_source(path)
            current_definitions = discover_source(
                source,
                path,
                include_documented=overwrite,
            )
        except (OSError, SyntaxError, ValueError) as exc:
            failed += len(items)
            logger.error("Could not re-read %s: %s", path, exc)
            continue

        available: dict[tuple[str, str, str], list[Definition]] = defaultdict(list)
        for definition in current_definitions:
            available[(definition.kind, definition.qualname, definition.source)].append(definition)

        insertions: dict[int, str] = {}
        for item in items:
            expected = item.definition
            matches = available.get(
                (expected.kind, expected.qualname, expected.source),
                [],
            )
            if len(matches) != 1:
                failed += 1
                logger.error(
                    "%s:%d %s: source changed while the batch was running",
                    path,
                    expected.lineno,
                    expected.qualname,
                )
                continue
            insertions[matches[0].lineno] = item.content

        if not insertions:
            continue

        try:
            updated = apply_docstrings(
                source,
                insertions,
                filename=str(path),
                overwrite=overwrite,
            )
            _write_source(path, updated, encoding)
            written += len(insertions)
            logger.info(
                "Updated %s (%d %s)",
                path,
                len(insertions),
                pluralize(len(insertions), "docstring", "docstrings"),
            )
        except (DocstringApplicationError, UnicodeError, OSError) as exc:
            failed += len(insertions)
            logger.error("Could not update %s: %s", path, exc)
    return written, failed


def _write_source(path: Path, source: str, encoding: str) -> None:
    data = source.encode(encoding)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        shutil.copymode(path, temporary)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
