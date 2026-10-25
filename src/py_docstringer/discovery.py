"""Discover classes, functions, and methods that lack docstrings."""

from __future__ import annotations

import ast
import tokenize
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

DefinitionKind = Literal["class", "function", "method"]

DefinitionNode = ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef


@dataclass(frozen=True, slots=True)
class Definition:
    """A Python definition that needs a docstring."""

    path: Path
    qualname: str
    kind: DefinitionKind
    lineno: int
    end_lineno: int
    source: str
    has_docstring: bool = False


def read_source(path: Path) -> tuple[str, str]:
    """Read a Python file using its declared encoding.

    Returns:
        The decoded source and the encoding that was detected.
    """

    with path.open("rb") as handle:
        encoding, _ = tokenize.detect_encoding(handle.readline)
    with path.open("r", encoding=encoding, newline="") as handle:
        return handle.read(), encoding


def discover_file(path: Path, *, include_documented: bool = False) -> list[Definition]:
    """Return definitions in a Python file, optionally including documented ones."""

    source, _ = read_source(path)
    return discover_source(source, path, include_documented=include_documented)


def discover_source(
    source: str,
    path: Path = Path("<memory>"),
    *,
    include_documented: bool = False,
) -> list[Definition]:
    """Return definitions in Python source, optionally including documented ones."""

    tree = ast.parse(source, filename=str(path))
    lines = source.splitlines(keepends=True)
    collector = _Collector(path, lines, include_documented=include_documented)
    collector.visit(tree)
    return collector.definitions


class _Collector(ast.NodeVisitor):
    def __init__(self, path: Path, lines: list[str], *, include_documented: bool) -> None:
        self._path = path
        self._lines = lines
        self._include_documented = include_documented
        self._scopes: list[DefinitionNode] = []
        self.definitions: list[Definition] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._record(node, "class")
        self._scopes.append(node)
        self.generic_visit(node)
        self._scopes.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        kind: DefinitionKind = (
            "method" if self._scopes and isinstance(self._scopes[-1], ast.ClassDef) else "function"
        )
        self._record(node, kind)
        self._scopes.append(node)
        self.generic_visit(node)
        self._scopes.pop()

    def _record(self, node: DefinitionNode, kind: DefinitionKind) -> None:
        has_docstring = ast.get_docstring(node, clean=False) is not None
        if has_docstring and not self._include_documented:
            return

        qualname = ".".join([*(scope.name for scope in self._scopes), node.name])
        start = min([node.lineno, *(decorator.lineno for decorator in node.decorator_list)])
        end = node.end_lineno or node.lineno
        self.definitions.append(
            Definition(
                path=self._path,
                qualname=qualname,
                kind=kind,
                lineno=node.lineno,
                end_lineno=end,
                source="".join(self._lines[start - 1 : end]),
                has_docstring=has_docstring,
            )
        )
