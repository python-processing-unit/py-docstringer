from __future__ import annotations

import ast
import shutil
import subprocess
from textwrap import dedent

import pytest

from py_docstringer.discovery import discover_source
from py_docstringer.editing import (
    apply_docstrings,
    normalize_docstring,
    render_docstring,
)
from py_docstringer.errors import DocstringApplicationError


def _module(source: str) -> ast.Module:
    return ast.parse(source)


def _find(source: str, name: str) -> ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef:
    for node in ast.walk(_module(source)):
        if (
            isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ):
            return node
    raise AssertionError(f"{name} not found")


def test_inserts_docstrings_for_function_class_and_method() -> None:
    source = dedent(
        """\
        def plain(value: int) -> str:
            return str(value)


        class Widget:
            def method(self) -> int:
                return 1
        """
    )

    updated = apply_docstrings(
        source,
        {
            1: "Return the value as a string.",
            5: "Represent a widget.",
            6: "Return the widget value.",
        },
    )
    tree = _module(updated)

    plain = tree.body[0]
    assert isinstance(plain, ast.FunctionDef)
    widget = tree.body[1]
    assert isinstance(widget, ast.ClassDef)
    assert ast.get_docstring(plain) == "Return the value as a string."
    assert ast.get_docstring(widget) == "Represent a widget."
    assert ast.get_docstring(_find(updated, "method")) == "Return the widget value."


def test_inserts_into_inline_suite() -> None:
    updated = apply_docstrings("def compact(): return 1\n", {1: "Return one."})
    function = _find(updated, "compact")

    assert ast.get_docstring(function) == "Return one."
    assert updated == dedent(
        '''\
        def compact():
            """Return one."""
            return 1
        '''
    )


def test_inserts_into_multiline_signature_with_inline_suite() -> None:
    source = dedent(
        """\
        def multiline(
            value: int,
        ) -> int: return value
        """
    )

    updated = apply_docstrings(source, {1: "Return the value."})

    assert ast.get_docstring(_find(updated, "multiline")) == "Return the value."
    assert "return value" in updated


def test_handles_unicode_before_inline_body() -> None:
    source = 'def greet(name: str = "é") -> str: return name\n'

    updated = apply_docstrings(source, {1: "Return the greeting."})

    assert ast.get_docstring(_find(updated, "greet")) == "Return the greeting."
    assert 'name: str = "é"' in updated


def test_preserves_decorators_and_inserts_after_signature() -> None:
    source = dedent(
        """\
        @decorator
        def decorated() -> None:
            return None
        """
    )

    updated = apply_docstrings(source, {2: "Do nothing."})

    assert updated.startswith("@decorator\n")
    assert ast.get_docstring(_find(updated, "decorated")) == "Do nothing."


def test_nested_class_and_inline_method_on_same_line() -> None:
    source = "class Outer:\n    def method(self): return 1\n"

    updated = apply_docstrings(
        source,
        {
            1: "Outer class.",
            2: "Method docstring.",
        },
    )

    assert ast.get_docstring(_find(updated, "Outer")) == "Outer class."
    assert ast.get_docstring(_find(updated, "method")) == "Method docstring."
    assert _find(updated, "method").lineno == 3


def test_multiline_docstring_keeps_indentation() -> None:
    content = "Return a value.\n\nArgs:\n    value: The input value.\n\nReturns:\n    The value."
    source = dedent(
        """\
        def function(value):
            return value
        """
    )

    updated = apply_docstrings(source, {1: content})
    function = _find(updated, "function")

    assert ast.get_docstring(function) == content
    assert '    """Return a value.' in updated
    assert "        value: The input value." in updated


def test_preserves_missing_trailing_newline() -> None:
    updated = apply_docstrings("def compact(): return 1", {1: "Return one."})

    assert not updated.endswith("\n")
    assert ast.get_docstring(_find(updated, "compact")) == "Return one."


def test_preserves_crlf_newlines() -> None:
    source = "def function():\r\n    return 1\r\n"

    updated = apply_docstrings(source, {1: "Return one.\n\nMore detail."})

    assert ast.get_docstring(_find(updated, "function")) == "Return one.\n\nMore detail."
    assert "\n" not in updated.replace("\r\n", "")


def test_inserts_around_decorated_first_member() -> None:
    source = dedent(
        """\
        class Widget:
            @property
            def value(self) -> int:
                return 1
        """
    )

    updated = apply_docstrings(
        source,
        {
            1: "Represent a widget.",
            3: "Return the value.",
        },
    )

    assert ast.get_docstring(_find(updated, "Widget")) == "Represent a widget."
    assert ast.get_docstring(_find(updated, "value")) == "Return the value."
    assert updated.startswith("class Widget:\n")
    assert "\n    @property\n" in updated


def test_inserts_into_function_with_decorated_first_statement() -> None:
    source = dedent(
        """\
        def outer() -> None:
            @decorator
            def inner() -> None:
                return None
        """
    )

    updated = apply_docstrings(
        source,
        {
            1: "Outer function.",
            3: "Inner function.",
        },
    )

    assert ast.get_docstring(_find(updated, "outer")) == "Outer function."
    assert ast.get_docstring(_find(updated, "inner")) == "Inner function."


def test_overwrite_replaces_existing_docstring() -> None:
    source = dedent(
        '''\
        def function() -> int:
            """Old summary.

            Old detail.
            """
            return 1
        '''
    )

    updated = apply_docstrings(
        source,
        {1: "New summary.\n\nArgs:\n    value: The input value."},
        overwrite=True,
    )

    assert ast.get_docstring(_find(updated, "function")) == (
        "New summary.\n\nArgs:\n    value: The input value."
    )
    assert "Old summary" not in updated
    assert updated.count('"""') == 2


def test_overwrite_replaces_inline_docstring() -> None:
    source = 'def function(): """Old summary."""; return 1\n'
    content = "New summary.\n\nArgs:\n    value: The input value."

    updated = apply_docstrings(
        source,
        {1: content},
        overwrite=True,
    )

    assert ast.get_docstring(_find(updated, "function")) == content
    assert "Old summary" not in updated
    assert "return 1" in updated
    assert updated.startswith('def function():\n    """New summary.')


def test_overwrite_adds_docstrings_that_are_missing() -> None:
    source = "def function():\n    return 1\n"

    updated = apply_docstrings(
        source,
        {1: "New summary."},
        overwrite=True,
    )

    assert ast.get_docstring(_find(updated, "function")) == "New summary."


def test_wraps_long_docstring_lines_at_spaces() -> None:
    content = (
        "This is a long summary that should be wrapped because it exceeds the "
        "configured line length by a lot."
    )
    source = "def function():\n    return 1\n"

    updated = apply_docstrings(source, {1: content}, max_line_length=40)

    docstring = ast.get_docstring(_find(updated, "function"))
    assert docstring is not None
    assert " ".join(docstring.split()) == content
    assert all(len(line) <= 40 for line in updated.splitlines())


def test_does_not_break_long_words() -> None:
    word = "x" * 60
    content = f"See {word}."
    source = "def function():\n    return 1\n"

    updated = apply_docstrings(source, {1: content}, max_line_length=40)

    assert word in updated
    assert all(len(line) <= 40 for line in updated.splitlines() if word not in line)


def test_wraps_google_style_section_content() -> None:
    content = (
        "Summary.\n\nArgs:\n"
        "    value: A long parameter description that needs wrapping because it is too long."
    )
    source = "def function(value):\n    return value\n"

    updated = apply_docstrings(source, {1: content}, max_line_length=50)

    docstring = ast.get_docstring(_find(updated, "function"))
    assert docstring is not None
    assert " ".join(docstring.split()) == " ".join(content.split())
    assert all(len(line) <= 50 for line in updated.splitlines())
    assert "\n            " in updated


def test_wraps_single_line_docstring_before_rendering() -> None:
    content = "A single long line that has enough words to need wrapping at this width."
    source = "def function():\n    return 1\n"

    updated = apply_docstrings(source, {1: content}, max_line_length=40)

    assert updated.count('"""') == 2
    assert updated.count("\n") > 3
    assert all(len(line) <= 40 for line in updated.splitlines())


def test_inserts_blank_line_after_summary_for_d205() -> None:
    content = "Return a value.\nMore detail without a blank line."
    source = "def function():\n    return 1\n"

    updated = apply_docstrings(source, {1: content})

    docstring = ast.get_docstring(_find(updated, "function"))
    assert docstring is not None
    assert docstring.splitlines()[0] == "Return a value."
    assert docstring.splitlines()[1] == ""
    assert docstring.splitlines()[2] == "More detail without a blank line."


def test_wrapped_summary_keeps_blank_line_after_first_line() -> None:
    content = (
        "This is a long summary that should be wrapped and still satisfy the "
        "docstring summary separator rule."
    )
    source = "def function():\n    return 1\n"

    updated = apply_docstrings(source, {1: content}, max_line_length=40)

    docstring = ast.get_docstring(_find(updated, "function"))
    assert docstring is not None
    assert docstring.splitlines()[1] == ""
    assert all(len(line) <= 40 for line in updated.splitlines())


def test_generated_docstring_passes_ruff_d205(tmp_path) -> None:
    ruff = shutil.which("ruff")
    if ruff is None:
        pytest.skip("ruff is not available")

    content = (
        "This is a long summary that should be wrapped and still satisfy the "
        "docstring summary separator rule."
    )
    source = "def function():\n    return 1\n"
    updated = apply_docstrings(source, {1: content}, max_line_length=40)
    path = tmp_path / "example.py"
    path.write_text(updated, encoding="utf-8")

    result = subprocess.run(
        [ruff, "check", "--isolated", "--select", "D205", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_skips_definitions_that_already_have_docstrings() -> None:
    source = dedent(
        '''\
        def documented() -> None:
            """Existing."""
        '''
    )

    assert apply_docstrings(source, {1: "Replacement."}) == source


def test_rejects_unknown_line_number() -> None:
    with pytest.raises(DocstringApplicationError, match="not found"):
        apply_docstrings("def function():\n    pass\n", {99: "Missing."})


def test_apply_is_idempotent_after_documenting() -> None:
    source = "def function():\n    return 1\n"
    updated = apply_docstrings(source, {1: "Return one."})

    assert discover_source(updated) == []


def test_normalize_plain_text() -> None:
    assert normalize_docstring("  Return a value.\n\nMore detail.  ") == (
        "Return a value.\n\nMore detail."
    )


def test_normalize_code_fence() -> None:
    raw = '```python\n"""Return a value.\n\nMore detail.\n"""\n```'

    assert normalize_docstring(raw) == "Return a value.\n\nMore detail."


def test_normalize_triple_quoted_docstring() -> None:
    raw = '"""Return a value.\n\nArgs:\n    x: The input.\n"""'

    assert normalize_docstring(raw) == "Return a value.\n\nArgs:\n    x: The input."


def test_normalize_raw_prefixed_docstring() -> None:
    raw = 'r"""Open C:\\Users\\example."""'

    assert normalize_docstring(raw) == "Open C:\\Users\\example."


def test_normalize_indented_quoted_docstring() -> None:
    raw = '    """Return a value.\n\n    Args:\n        value: The input.\n    """'

    assert normalize_docstring(raw) == ("Return a value.\n\nArgs:\n    value: The input.")


def test_normalize_rejects_empty_content() -> None:
    with pytest.raises(DocstringApplicationError, match="empty"):
        normalize_docstring('"""\n\n"""')


def test_render_escapes_backslashes_and_triple_quotes() -> None:
    content = 'Use C:\\Users\\me and """quoted""" text.'
    rendered = render_docstring(content, "    ")
    source = f"def function():\n{rendered}    return None\n"

    assert ast.get_docstring(_find(source, "function")) == content


def test_render_handles_single_line_ending_in_quote() -> None:
    content = 'Say "hello"'
    rendered = render_docstring(content, "    ")
    source = f"def function():\n{rendered}    return None\n"

    assert ast.get_docstring(_find(source, "function")) == content
