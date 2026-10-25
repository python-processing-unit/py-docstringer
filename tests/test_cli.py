from __future__ import annotations

from py_docstringer.app import Summary
from py_docstringer.cli import DEFAULT_MODEL, _build_parser, _print_summary, main


def test_default_model_is_gpt_6_luna(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)

    args = _build_parser().parse_args(["folder"])

    assert DEFAULT_MODEL == "openai/gpt-6-luna"
    assert args.model == "openai/gpt-6-luna"


def test_overwrite_flag(tmp_path) -> None:
    parser = _build_parser()

    assert parser.parse_args([str(tmp_path)]).overwrite is False
    assert parser.parse_args([str(tmp_path), "--overwrite"]).overwrite is True


def test_overwrite_dry_run_lists_existing_docstrings(tmp_path, capsys) -> None:
    (tmp_path / "example.py").write_text(
        'def function():\n    """Old summary."""\n    return 1\n',
        encoding="utf-8",
    )

    assert main([str(tmp_path), "--dry-run"]) == 0
    default_output = capsys.readouterr().out
    assert "function" not in default_output
    assert "Would document 0 definitions" in default_output

    assert main([str(tmp_path), "--dry-run", "--overwrite"]) == 0
    overwrite_output = capsys.readouterr().out
    assert "function function" in overwrite_output
    assert "Would document 1 definition" in overwrite_output


def test_dry_run_pluralizes_multiple_files(tmp_path, capsys) -> None:
    (tmp_path / "first.py").write_text("def first(): ...\n", encoding="utf-8")
    (tmp_path / "second.py").write_text("def second(): ...\n", encoding="utf-8")

    exit_code = main([str(tmp_path), "--dry-run"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "in 2 files." in captured.out
    assert "Would document 2 definitions" in captured.out


def test_summary_pluralizes_docstrings(capsys) -> None:
    _print_summary(Summary(files=1, definitions=1, written=1, failed=0))
    assert "Added 1 docstring across 1 file;" in capsys.readouterr().out

    _print_summary(Summary(files=2, definitions=3, written=2, failed=0))
    assert "Added 2 docstrings across 2 files;" in capsys.readouterr().out


def test_dry_run_lists_definitions_and_succeeds(tmp_path, capsys) -> None:
    (tmp_path / "example.py").write_text(
        "def example():\n    return 1\n",
        encoding="utf-8",
    )

    exit_code = main([str(tmp_path), "--dry-run"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "def" in captured.out
    assert "example" in captured.out
    assert "Would document 1 definition in 1 file." in captured.out


def test_skipped_files_produce_nonzero_exit(tmp_path, capsys) -> None:
    (tmp_path / "broken.py").write_text("def broken(:\n", encoding="utf-8")

    exit_code = main([str(tmp_path), "--dry-run"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "1 file skipped" in captured.out


def test_missing_api_key_fails_without_dry_run(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    (tmp_path / "example.py").write_text("def example(): ...\n", encoding="utf-8")

    exit_code = main([str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "OPENROUTER_API_KEY is not set" in captured.err


def test_missing_folder_fails(tmp_path, capsys) -> None:
    exit_code = main([str(tmp_path / "missing"), "--dry-run"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "is not a directory" in captured.err
