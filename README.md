# py-docstringer

Add Google-style docstrings to every undocumented function, class, and method below a
folder. Work is sent to [OpenRouter batch requests](https://openrouter.ai/docs/client-sdks/python/sdks/batch/README#batch),
which are asynchronous and cheaper than synchronous inference.

## Requirements

- Python `3.15.1` (installed automatically by uv)
- [uv](https://docs.astral.sh/uv/)
- `OPENROUTER_API_KEY` in the environment

## Usage

```powershell
uv sync
$env:OPENROUTER_API_KEY = "sk-or-..."
uv run add-docstrings .\src
```

The command scans the folder recursively for `.py` files, skips definitions that
already have a docstring, submits one batch request per definition, waits for the batch
to finish, and writes the generated docstrings back to the same files. Pass
`--overwrite` to rewrite existing docstrings as well as add missing ones. Generated
docstrings are wrapped at spaces to `--max-line-length` (default: 88 characters).

```powershell
uv run add-docstrings .\src --overwrite
```

Choose a model with a batch variant from the
[batch models list](https://openrouter.ai/models?variant=batch). The default is
[GPT-6 Luna](https://openrouter.ai/openai/gpt-6-luna)
(`openai/gpt-6-luna`). Your account also needs enough credits for the estimated
batch cost.

```text
usage: add-docstrings [-h] [--model MODEL] [--poll-interval POLL_INTERVAL]
                      [--timeout-hours TIMEOUT_HOURS] [--batch-size BATCH_SIZE]
                      [--max-line-length MAX_LINE_LENGTH] [--overwrite] [--dry-run]
                      [-q]
                      folder

positional arguments:
  folder                folder to scan recursively for .py files

options:
  --model MODEL         OpenRouter model slug (default: openai/gpt-6-luna)
  --poll-interval N     seconds between batch status checks (default: 30)
  --timeout-hours N     maximum hours to wait for each batch (default: 26)
  --batch-size N        maximum requests per batch (default: 500)
  --max-line-length N   maximum docstring source line length (default: 88)
  --overwrite           replace existing docstrings instead of leaving them unchanged
  --dry-run             list definitions without calling OpenRouter
  -q, --quiet           only print warnings and errors
```

Set `OPENROUTER_MODEL` to change the default model. Use `--dry-run` first to confirm
which definitions will be sent. The default timeout is deliberately above the 24-hour
batch completion window; providers with longer batch windows (up to 96 hours) need a
larger `--timeout-hours` value. The command exits nonzero if any request or file fails,
including files that could not be parsed.

## Development

```powershell
uv sync
uv run ty check .
uv run ruff check .
uv run ruff format --check .
uv run pytest -vv
```

`pyproject.toml` configures pytest with `-n auto`, so tests run in parallel. The GitHub
Actions workflow in `.github/workflows/ci.yml` runs the same four checks.
