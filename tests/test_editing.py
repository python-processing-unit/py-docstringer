from __future__ import annotations

import ast
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
    assert updated.startswith('def function(): """New summary.\\n')


def test_overwrite_adds_docstrings_that_are_missing() -> None:
    source = "def function():\n    return 1\n"

    updated = apply_docstrings(
        source,
        {1: "New summary."},
        overwrite=True,
    )

    assert ast.get_docstring(_find(updated, "function")) == "New summary."


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
