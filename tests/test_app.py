from __future__ import annotations

import ast
from collections.abc import Callable, Sequence
from textwrap import dedent

from py_docstringer.app import document_folder, find_python_files
from py_docstringer.batch import BatchRequest, BatchResult, BatchSubmission


class _CompletedTransport:
    def __init__(
        self,
        failure_marker: str | None = None,
        *,
        content: str = "Generated docstring.",
        on_create: Callable[[], None] | None = None,
    ) -> None:
        self.failure_marker = failure_marker
        self.content = content
        self.on_create = on_create
        self.requests: list[BatchRequest] = []

    def create(self, requests: Sequence[BatchRequest]) -> BatchSubmission:
        self.requests.extend(requests)
        if self.on_create is not None:
            self.on_create()
        results = []
        for request in requests:
            prompt = request.messages[-1]["content"]
            if self.failure_marker and self.failure_marker in prompt:
                results.append(BatchResult(request.custom_id, error="model request failed"))
            else:
                results.append(BatchResult(request.custom_id, content=self.content))
        return BatchSubmission("batch_1", "completed", tuple(results))

    def fetch(self, batch_id: str) -> BatchSubmission:
        raise AssertionError("fetch should not be called for a completed batch")


def _run(folder, transport, *, overwrite: bool = False):
    return document_folder(
        folder,
        transport,
        poll_interval=1,
        timeout=60,
        batch_size=10,
        overwrite=overwrite,
    )


def test_document_folder_updates_undocumented_definitions(tmp_path) -> None:
    package = tmp_path / "pkg"
    package.mkdir()
    source = dedent(
        """\
        def alpha() -> int:
            return 1


        class Beta:
            def gamma(self) -> int:
                return 2
        """
    )
    source_path = package / "source.py"
    source_path.write_text(source, encoding="utf-8")
    documented_path = package / "documented.py"
    documented_path.write_text(
        'def already_documented():\n    """Already documented."""\n',
        encoding="utf-8",
    )
    broken_path = tmp_path / "broken.py"
    broken_path.write_text("def broken(:\n", encoding="utf-8")
    ignored_dir = tmp_path / ".venv"
    ignored_dir.mkdir()
    ignored_path = ignored_dir / "ignored.py"
    ignored_path.write_text("def ignored(): ...\n", encoding="utf-8")

    transport = _CompletedTransport()
    summary = _run(tmp_path, transport)

    assert summary.files == 3
    assert summary.definitions == 3
    assert summary.written == 3
    assert summary.failed == 0
    assert summary.skipped == 1

    updated = source_path.read_text(encoding="utf-8")
    tree = ast.parse(updated)
    assert ast.get_docstring(tree.body[0]) == "Generated docstring."
    beta = tree.body[1]
    assert isinstance(beta, ast.ClassDef)
    assert ast.get_docstring(beta) == "Generated docstring."
    method = next(node for node in beta.body if isinstance(node, ast.FunctionDef))
    assert ast.get_docstring(method) == "Generated docstring."
    assert documented_path.read_text(encoding="utf-8") == (
        'def already_documented():\n    """Already documented."""\n'
    )
    assert broken_path.read_text(encoding="utf-8") == "def broken(:\n"
    assert ignored_path.read_text(encoding="utf-8") == "def ignored(): ...\n"

    prompts = [request.messages[-1]["content"] for request in transport.requests]
    assert any("Qualified name: Beta.gamma" in prompt for prompt in prompts)
    assert all("already_documented" not in prompt for prompt in prompts)


def test_document_folder_dry_run_does_not_edit_files(tmp_path, capsys) -> None:
    path = tmp_path / "example.py"
    original = "def example():\n    return 1\n"
    path.write_text(original, encoding="utf-8")

    summary = document_folder(
        tmp_path,
        None,
        poll_interval=1,
        timeout=60,
        batch_size=10,
        dry_run=True,
    )

    assert summary.dry_run
    assert summary.definitions == 1
    assert "example" in capsys.readouterr().out
    assert path.read_text(encoding="utf-8") == original


def test_document_folder_applies_successes_when_one_request_fails(tmp_path) -> None:
    path = tmp_path / "example.py"
    path.write_text(
        dedent(
            """\
            def first() -> int:
                return 1


            class Second:
                def method(self) -> int:
                    return 2
            """
        ),
        encoding="utf-8",
    )
    transport = _CompletedTransport(failure_marker="Second.method")

    summary = _run(tmp_path, transport)

    assert summary.definitions == 3
    assert summary.written == 2
    assert summary.failed == 1
    updated = path.read_text(encoding="utf-8")
    tree = ast.parse(updated)
    assert ast.get_docstring(tree.body[0]) == "Generated docstring."
    second = tree.body[1]
    assert isinstance(second, ast.ClassDef)
    assert ast.get_docstring(second) == "Generated docstring."
    method = next(node for node in second.body if isinstance(node, ast.FunctionDef))
    assert ast.get_docstring(method) is None


def test_document_folder_only_overwrites_when_requested(tmp_path) -> None:
    path = tmp_path / "example.py"
    original = 'def function():\n    """Old summary."""\n    return 1\n'
    path.write_text(original, encoding="utf-8")

    default_summary = _run(tmp_path, _CompletedTransport())
    assert default_summary.definitions == 0
    assert default_summary.written == 0
    assert path.read_text(encoding="utf-8") == original

    overwrite_summary = _run(
        tmp_path,
        _CompletedTransport(content="New generated docstring."),
        overwrite=True,
    )
    assert overwrite_summary.definitions == 1
    assert overwrite_summary.written == 1
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = tree.body[0]
    assert isinstance(function, ast.FunctionDef)
    assert ast.get_docstring(function) == "New generated docstring."


def test_document_folder_overwrite_replaces_and_adds_docstrings(tmp_path) -> None:
    path = tmp_path / "example.py"
    path.write_text(
        dedent(
            '''\
            def documented() -> int:
                """Old summary."""
                return 1


            class Widget:
                def method(self) -> int:
                    return 2
            '''
        ),
        encoding="utf-8",
    )

    summary = _run(
        tmp_path,
        _CompletedTransport(content="New generated docstring."),
        overwrite=True,
    )

    assert summary.definitions == 3
    assert summary.written == 3
    assert summary.failed == 0
    updated = path.read_text(encoding="utf-8")
    assert "Old summary" not in updated
    tree = ast.parse(updated)
    assert ast.get_docstring(tree.body[0]) == "New generated docstring."
    widget = tree.body[1]
    assert isinstance(widget, ast.ClassDef)
    assert ast.get_docstring(widget) == "New generated docstring."
    method = next(node for node in widget.body if isinstance(node, ast.FunctionDef))
    assert ast.get_docstring(method) == "New generated docstring."


def test_document_folder_does_not_submit_when_nothing_is_missing(tmp_path) -> None:
    path = tmp_path / "example.py"
    path.write_text('def documented():\n    """Documented."""\n', encoding="utf-8")
    transport = _CompletedTransport()

    summary = _run(tmp_path, transport)

    assert summary.definitions == 0
    assert summary.written == 0
    assert summary.skipped == 0
    assert transport.requests == []


def test_document_folder_documents_decorated_first_member(tmp_path) -> None:
    path = tmp_path / "example.py"
    path.write_text(
        dedent(
            """\
            class Widget:
                @property
                def value(self) -> int:
                    return 1
            """
        ),
        encoding="utf-8",
    )

    summary = _run(tmp_path, _CompletedTransport())

    assert summary.written == 2
    assert summary.failed == 0
    tree = ast.parse(path.read_text(encoding="utf-8"))
    widget = tree.body[0]
    assert isinstance(widget, ast.ClassDef)
    assert ast.get_docstring(widget) == "Generated docstring."
    value = next(node for node in widget.body if isinstance(node, ast.FunctionDef))
    assert ast.get_docstring(value) == "Generated docstring."


def test_document_folder_realigns_definitions_after_line_shift(tmp_path) -> None:
    path = tmp_path / "example.py"
    path.write_text("def function():\n    return 1\n", encoding="utf-8")
    shifted = "# header\n\n\ndef function():\n    return 1\n"
    transport = _CompletedTransport(on_create=lambda: path.write_text(shifted, encoding="utf-8"))

    summary = _run(tmp_path, transport)

    assert summary.written == 1
    assert summary.failed == 0
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    assert ast.get_docstring(function) == "Generated docstring."


def test_document_folder_rejects_changed_definition(tmp_path) -> None:
    path = tmp_path / "example.py"
    path.write_text("def function():\n    return 1\n", encoding="utf-8")
    changed = "def function():\n    return 2\n"
    transport = _CompletedTransport(on_create=lambda: path.write_text(changed, encoding="utf-8"))

    summary = _run(tmp_path, transport)

    assert summary.written == 0
    assert summary.failed == 1
    assert path.read_text(encoding="utf-8") == changed


def test_document_folder_does_not_truncate_on_encoding_failure(tmp_path) -> None:
    path = tmp_path / "example.py"
    original = "# -*- coding: latin-1 -*-\ndef function():\n    return 1\n"
    path.write_bytes(original.encode("latin-1"))
    transport = _CompletedTransport(content="Return an arrow →.")

    summary = _run(tmp_path, transport)

    assert summary.written == 0
    assert summary.failed == 1
    assert path.read_bytes() == original.encode("latin-1")


def test_find_python_files_skips_generated_directories(tmp_path) -> None:
    (tmp_path / "package").mkdir()
    (tmp_path / "package" / "module.py").write_text("", encoding="utf-8")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "ignored.py").write_text("", encoding="utf-8")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "ignored.py").write_text("", encoding="utf-8")

    assert find_python_files(tmp_path) == [tmp_path / "package" / "module.py"]
