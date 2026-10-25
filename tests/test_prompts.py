from __future__ import annotations

from pathlib import Path

from py_docstringer.discovery import Definition
from py_docstringer.prompts import SYSTEM_PROMPT, build_prompt


def test_build_prompt_includes_definition_context() -> None:
    definition = Definition(
        path=Path("src/example.py"),
        qualname="Widget.render",
        kind="method",
        lineno=12,
        end_lineno=14,
        source="    def render(self) -> str:\n        return 'widget'\n",
    )

    prompt = build_prompt(definition)

    assert "Definition type: method" in prompt
    assert "Qualified name: Widget.render" in prompt
    assert "File: src\\example.py" in prompt or "File: src/example.py" in prompt
    assert "return 'widget'" in prompt


def test_build_prompt_requests_rewrite_for_existing_docstring() -> None:
    definition = Definition(
        path=Path("example.py"),
        qualname="function",
        kind="function",
        lineno=1,
        end_lineno=2,
        source='def function():\n    """Old summary."""\n    return 1\n',
        has_docstring=True,
    )

    prompt = build_prompt(definition, overwrite=True)

    assert "already has a docstring" in prompt
    assert "Replace it" in prompt
    assert "Old summary" in prompt


def test_build_prompt_does_not_request_rewrite_without_overwrite() -> None:
    definition = Definition(
        path=Path("example.py"),
        qualname="function",
        kind="function",
        lineno=1,
        end_lineno=2,
        source='def function():\n    """Old summary."""\n    return 1\n',
        has_docstring=True,
    )

    assert "already has a docstring" not in build_prompt(definition)


def test_system_prompt_requests_google_style() -> None:
    assert "Google-style" in SYSTEM_PROMPT
    assert "Args" in SYSTEM_PROMPT
    assert "Returns" in SYSTEM_PROMPT
    assert "triple quotes" in SYSTEM_PROMPT
