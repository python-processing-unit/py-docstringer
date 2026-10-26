"""Normalize model output and insert docstrings into Python source."""

from __future__ import annotations

import ast
import re
import textwrap
from collections.abc import Mapping
from dataclasses import dataclass

from .errors import DocstringApplicationError
from .formatting import pluralize

_DELIMITERS = ('"""', "'''")
_FENCE_RE = re.compile(
    r"\A```[A-Za-z0-9_-]*[ \t]*\r?\n(?P<body>.*?)\r?\n```[ \t]*\Z",
    re.DOTALL,
)


@dataclass(frozen=True, slots=True)
class _Edit:
    """Represent a text edit with a source range and replacement text.

    Attributes:
        start (int): Start offset of the range.
        end (int): End offset of the range.
        text (str): Replacement text.
    """

    start: int
    end: int
    text: str


def normalize_docstring(raw: str) -> str:
    """Turn a model response into clean docstring content without quotes."""

    text = raw.strip()
    fenced = _FENCE_RE.match(text)
    if fenced is not None:
        text = fenced.group("body")

    text = textwrap.dedent(text).strip()
    if not text:
        raise DocstringApplicationError("model returned an empty docstring")

    literal = _literal_docstring(text)
    text = literal if literal is not None else _strip_delimiters(text)
    text = textwrap.dedent(text).strip()

    lines = [line.rstrip() for line in text.splitlines()]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    if not lines:
        raise DocstringApplicationError("model returned an empty docstring")

    first = lines[0].strip()
    if len(lines) == 1:
        return first

    remainder = textwrap.dedent("\n".join(lines[1:]))
    return first + "\n" + remainder.rstrip("\n")


def _wrap_content(content: str, indent: str, max_line_length: int) -> str:
    """Wrap docstring content lines that would exceed the source line limit."""

    lines = content.split("\n")
    single_line = len(lines) == 1
    wrapped: list[str] = []

    for index, line in enumerate(lines):
        if not line:
            wrapped.append(line)
            continue
        if single_line:
            overhead = 6
        elif index == 0:
            overhead = 3
        else:
            overhead = 0
        width = max_line_length - len(indent) - overhead
        if width <= 0 or len(indent) + len(line) + overhead <= max_line_length:
            wrapped.append(line)
            continue

        leading = line[: len(line) - len(line.lstrip())]
        text = line[len(leading) :]
        if not text or width <= len(leading):
            wrapped.append(line)
            continue

        pieces = textwrap.wrap(
            text,
            width=width,
            initial_indent=leading,
            subsequent_indent=leading + "    ",
            break_long_words=False,
            break_on_hyphens=False,
            replace_whitespace=False,
            drop_whitespace=True,
        )
        wrapped.extend(pieces or [line])

    return "\n".join(wrapped)


def _ensure_summary_separator(content: str) -> str:
    """Insert a blank line after the summary when description text follows it."""

    lines = content.split("\n")
    if len(lines) > 1 and lines[1].strip():
        lines.insert(1, "")
    return "\n".join(lines)


def render_docstring(
    content: str,
    indent: str,
    *,
    newline: str = "\n",
    max_line_length: int | None = None,
) -> str:
    """Render docstring content as an indented Python string literal."""

    if max_line_length is not None:
        content = _wrap_content(content, indent, max_line_length)
    content = _ensure_summary_separator(content)
    lines = content.split("\n")
    if len(lines) == 1:
        escaped = _escape(lines[0])
        if escaped.endswith('"'):
            escaped = f'{escaped[:-1]}\\"'
        return f'{indent}"""{escaped}"""{newline}'

    rendered = [f'{indent}"""{_escape(lines[0])}']
    rendered.extend(f"{indent}{_escape(line)}" if line else "" for line in lines[1:])
    rendered.append(f'{indent}"""')
    return newline.join(rendered) + newline


def apply_docstrings(
    source: str,
    docstrings: Mapping[int, str],
    *,
    filename: str = "<unknown>",
    overwrite: bool = False,
    max_line_length: int | None = None,
) -> str:
    """Insert or replace docstrings keyed by definition line number."""

    if not docstrings:
        return source

    tree = ast.parse(source, filename=filename)
    lines = source.splitlines(keepends=True)
    newline = "\r\n" if "\r\n" in source else "\n"
    edits: list[_Edit] = []
    seen: set[int] = set()

    for node in ast.walk(tree):
        if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.lineno not in docstrings:
            continue
        seen.add(node.lineno)
        if ast.get_docstring(node, clean=False) is not None:
            if overwrite:
                edits.append(
                    _build_replacement_edit(
                        node,
                        lines,
                        docstrings[node.lineno],
                        filename,
                        newline,
                        max_line_length,
                    )
                )
            continue
        edits.append(
            _build_edit(
                node,
                lines,
                docstrings[node.lineno],
                filename,
                newline,
                max_line_length,
            )
        )

    missing = set(docstrings) - seen
    if missing:
        formatted = ", ".join(str(line) for line in sorted(missing))
        definitions_word = pluralize(len(missing), "definition", "definitions")
        lines_word = pluralize(len(missing), "line", "lines")
        verb = "was" if len(missing) == 1 else "were"
        raise DocstringApplicationError(
            f"{filename}: {definitions_word} at {lines_word} {formatted} {verb} not found"
        )

    edits.sort(key=lambda edit: (edit.start, edit.end > edit.start), reverse=True)
    for edit in edits:
        lines[edit.start : edit.end] = [edit.text]

    result = "".join(lines)
    try:
        ast.parse(result, filename=filename)
    except SyntaxError as exc:
        raise DocstringApplicationError(
            f"{filename}: generated docstrings produced invalid Python: {exc}"
        ) from exc
    return result


def _literal_docstring(text: str) -> str | None:
    """Parse text as a string literal when it has a recognized prefix.

    Args:
        text (str): Text to evaluate as a literal.

    Returns:
        str | None: The evaluated string, or None if the text does not begin with a
        recognized string-literal prefix or does not evaluate to a string.
    """
    lowered = text.lower()
    starts_with_literal = lowered.startswith(('"""', "'''", 'r"""', "r'''", 'u"""', "u'''"))
    if not starts_with_literal:
        return None
    try:
        value = ast.literal_eval(text)
    except SyntaxError, ValueError:
        return None
    return value if isinstance(value, str) else None


def _strip_delimiters(text: str) -> str:
    """Strip recognized string-literal delimiters from text.

    Recognize the configured delimiters with optional `r`, `R`, `u`, or `U` prefixes.

    Args:
        text (str): Text that may begin and end with a recognized delimiter.

    Returns:
        str: Text with a recognized opening delimiter and, if present, its matching closing delimiter removed.
    """
    for prefix in ("", "r", "R", "u", "U"):
        for delimiter in _DELIMITERS:
            opening = prefix + delimiter
            if text.startswith(opening):
                text = text[len(opening) :]
                if text.endswith(delimiter):
                    text = text[: -len(delimiter)]
                return text
    return text


def _escape(text: str) -> str:
    """Escape backslashes and triple double quotes in text.

    Args:
        text (str): The text to escape.

    Returns:
        str: The escaped text.
    """
    return text.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')


def _build_edit(
    node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
    lines: list[str],
    content: str,
    filename: str,
    newline: str,
    max_line_length: int | None,
) -> _Edit:
    """Build an edit that inserts a docstring into a definition.

    Determine the insertion point from the definition's first body statement and render the provided content using the specified newline convention. Handle both inline and indented suites.

    Args:
        node (ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef): Definition to edit.
        lines (list[str]): Source lines used to locate the insertion point.
        content (str): Docstring content to render.
        filename (str): Filename included in error messages.
        newline (str): Newline sequence to use in the rendered edit.

    Returns:
        _Edit: Edit describing the source range to replace and the replacement text.

    Raises:
        DocstringApplicationError: If the definition has no body, its body cannot be located, or the end of an inline signature cannot be found.
    """
    if not node.body:
        raise DocstringApplicationError(f"{filename}:{node.lineno}: definition has no body")

    first_statement = node.body[0]
    decorators = getattr(first_statement, "decorator_list", ())
    start_line = min([first_statement.lineno, *(decorator.lineno for decorator in decorators)])
    line_index = start_line - 1
    if line_index < 0 or line_index >= len(lines):
        raise DocstringApplicationError(
            f"{filename}:{node.lineno}: could not locate the definition body"
        )

    line = lines[line_index]
    statement_col = _character_column(line, first_statement.col_offset)
    prefix = line[:statement_col]

    if prefix.strip():
        colon = prefix.rfind(":")
        if colon == -1:
            raise DocstringApplicationError(
                f"{filename}:{node.lineno}: could not locate the end of the signature"
            )
        line_indent = re.match(r"[ \t]*", line)
        body_indent = (line_indent.group(0) if line_indent else "") + "    "
        rendered = render_docstring(
            content,
            body_indent,
            newline=newline,
            max_line_length=max_line_length,
        )
        text = f"{prefix[: colon + 1]}{newline}{rendered}{body_indent}{line[statement_col:]}"
        return _Edit(start=line_index, end=line_index + 1, text=text)

    indent = re.match(r"[ \t]*", line)
    docstring = render_docstring(
        content,
        indent.group(0) if indent else "",
        newline=newline,
        max_line_length=max_line_length,
    )
    return _Edit(start=line_index, end=line_index, text=docstring)


def _build_replacement_edit(
    node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
    lines: list[str],
    content: str,
    filename: str,
    newline: str,
    max_line_length: int | None,
) -> _Edit:
    """Build an edit that replaces an existing docstring.

    Render the replacement content using the existing docstring's location and
    indentation, preserving surrounding text.

    Args:
        node (ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef): Definition
            whose existing docstring will be replaced.
        lines (list[str]): Source lines containing the definition.
        content (str): Replacement docstring content.
        filename (str): Filename used in error messages.
        newline (str): Newline sequence used when rendering the replacement.

    Returns:
        _Edit: Edit describing the replacement's source range and text.

    Raises:
        DocstringApplicationError: If the definition has no body, its first
            statement is not a string docstring, or the docstring's source
            location cannot be determined.
    """
    if not node.body:
        raise DocstringApplicationError(f"{filename}:{node.lineno}: definition has no body")

    docstring = node.body[0]
    if not (
        isinstance(docstring, ast.Expr)
        and isinstance(docstring.value, ast.Constant)
        and isinstance(docstring.value.value, str)
    ):
        raise DocstringApplicationError(
            f"{filename}:{node.lineno}: could not locate the existing docstring"
        )

    end_lineno = docstring.end_lineno or docstring.lineno
    start_index = docstring.lineno - 1
    end_index = end_lineno - 1
    if (
        start_index < 0
        or end_index < start_index
        or end_index >= len(lines)
        or docstring.end_col_offset is None
    ):
        raise DocstringApplicationError(
            f"{filename}:{node.lineno}: could not locate the existing docstring"
        )

    start_line = lines[start_index]
    end_line = lines[end_index]
    start_column = _character_column(start_line, docstring.col_offset)
    end_column = _character_column(end_line, docstring.end_col_offset)
    prefix = start_line[:start_column]
    line_indent = re.match(r"[ \t]*", start_line)
    line_indent_text = line_indent.group(0) if line_indent else ""
    inline = bool(prefix.strip())

    if inline:
        colon = prefix.rfind(":")
        if colon == -1:
            raise DocstringApplicationError(
                f"{filename}:{node.lineno}: could not locate the end of the signature"
            )
        body_indent = line_indent_text + "    "
        rendered = render_docstring(
            content,
            body_indent,
            newline=newline,
            max_line_length=max_line_length,
        )
        rest = end_line[end_column:].lstrip(" \t")
        if rest.startswith(";"):
            rest = rest[1:].lstrip(" \t")
        rest = rest.rstrip("\r\n")
        text = f"{prefix[: colon + 1]}{newline}{rendered}"
        if rest:
            text += f"{body_indent}{rest}{newline}"
        return _Edit(start=start_index, end=end_index + 1, text=text)

    doc_indent = prefix
    rendered = render_docstring(
        content,
        doc_indent,
        newline=newline,
        max_line_length=max_line_length,
    )
    if rendered.endswith(newline):
        rendered = rendered[: -len(newline)]
    if doc_indent and rendered.startswith(doc_indent):
        rendered = rendered[len(doc_indent) :]

    return _Edit(
        start=start_index,
        end=end_index + 1,
        text=prefix + rendered + end_line[end_column:],
    )


def _character_column(line: str, byte_offset: int) -> int:
    """Return the character column corresponding to a UTF-8 byte offset.

    Args:
        line (str): The line to measure.
        byte_offset (int): The byte offset within the UTF-8 encoding of the line.

    Returns:
        int: The number of characters before the byte offset, or 0 if the offset is nonpositive.
    """
    if byte_offset <= 0:
        return 0
    encoded = line.encode("utf-8")
    return len(encoded[:byte_offset].decode("utf-8"))
