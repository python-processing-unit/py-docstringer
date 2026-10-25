"""Submit and poll OpenRouter batch requests."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, TypedDict

from openrouter import OpenRouter
from openrouter.components.batchobject import BatchObject, BodyChatCompletion, Result
from openrouter.components.batchsubmitbody import BatchSubmitBody

from .errors import BatchError

FINAL_STATUSES = frozenset({"completed", "failed", "expired", "cancelled"})
_BATCH_BODY_FIELD_ORDER = (
    "endpoint",
    "model",
    "provider",
    "completion_window",
    "callback_url",
    "requests",
)


def _ensure_batch_body_field_order() -> None:
    """Keep ordering-sensitive batch fields ahead of requests for SDK 1.3.79."""

    fields = BatchSubmitBody.model_fields
    if list(fields) == list(_BATCH_BODY_FIELD_ORDER):
        return
    ordered_names = [name for name in _BATCH_BODY_FIELD_ORDER if name in fields]
    ordered_names.extend(name for name in fields if name not in ordered_names)
    ordered = {name: fields[name] for name in ordered_names}
    fields.clear()
    fields.update(ordered)


class Message(TypedDict):
    """A chat message sent to the model."""

    role: str
    content: str


@dataclass(frozen=True, slots=True)
class BatchRequest:
    """One request in an OpenRouter batch."""

    custom_id: str
    messages: tuple[Message, ...]


@dataclass(frozen=True, slots=True)
class BatchResult:
    """The outcome of one request in a batch."""

    custom_id: str
    content: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class BatchSubmission:
    """The state of an OpenRouter batch."""

    id: str
    status: str
    results: tuple[BatchResult, ...] | None = None
    error: str | None = None


class BatchTransport(Protocol):
    """The subset of the OpenRouter client used by this application."""

    def create(self, requests: Sequence[BatchRequest]) -> BatchSubmission:
        """Create a batch."""

    def fetch(self, batch_id: str) -> BatchSubmission:
        """Fetch a batch by id."""


class OpenRouterTransport:
    """A BatchTransport backed by the OpenRouter Python SDK."""

    def __init__(self, client: OpenRouter, model: str) -> None:
        _ensure_batch_body_field_order()
        self._client = client
        self._model = model

    def create(self, requests: Sequence[BatchRequest]) -> BatchSubmission:
        try:
            batch = self._client.batch.create_batches(
                endpoint="/v1/chat/completions",
                model=self._model,
                completion_window="24h",
                requests=[
                    {
                        "custom_id": request.custom_id,
                        "body": {
                            "messages": list(request.messages),
                            "temperature": 0.2,
                        },
                    }
                    for request in requests
                ],
            )
        except Exception as exc:
            raise BatchError(f"OpenRouter batch submission failed: {exc}") from exc
        return _to_submission(batch)

    def fetch(self, batch_id: str) -> BatchSubmission:
        try:
            batch = self._client.batch.get_batches(id=batch_id)
        except Exception as exc:
            raise BatchError(f"Could not fetch batch {batch_id}: {exc}") from exc
        return _to_submission(batch)


@dataclass(frozen=True, slots=True)
class _PendingBatch:
    requests: tuple[BatchRequest, ...]
    submission: BatchSubmission
    deadline: float


def run_batches(
    transport: BatchTransport,
    requests: Sequence[BatchRequest],
    *,
    poll_interval: float,
    timeout: float,
    batch_size: int = 500,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, BatchResult]:
    """Submit all request chunks, poll them together, and return every result.

    Batch-level failures are converted into per-request error results so that
    results from other batches are not discarded.
    """

    if not requests:
        return {}
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    results: dict[str, BatchResult] = {}
    pending: list[_PendingBatch] = []

    for chunk in _chunks(requests, batch_size):
        try:
            submission = transport.create(chunk)
        except BatchError as exc:
            results.update(_failures(chunk, str(exc)))
            continue
        pending.append(_PendingBatch(tuple(chunk), submission, monotonic() + timeout))

    while pending:
        remaining: list[_PendingBatch] = []
        completed = False
        for item in pending:
            if _is_complete(item.submission):
                _collect(item, results)
                completed = True
            else:
                remaining.append(item)
        if completed:
            pending = remaining
            continue

        now = monotonic()
        remaining = []
        timed_out = False
        for item in pending:
            if now >= item.deadline:
                results.update(
                    _failures(
                        item.requests,
                        f"Timed out waiting for batch {item.submission.id} "
                        f"(last status: {item.submission.status})",
                    )
                )
                timed_out = True
            else:
                remaining.append(item)
        if timed_out:
            pending = remaining
            continue

        wait_for = min(item.deadline - now for item in pending)
        sleep(min(poll_interval, max(wait_for, 0.0)))

        next_pending: list[_PendingBatch] = []
        for item in pending:
            try:
                submission = transport.fetch(item.submission.id)
            except BatchError as exc:
                results.update(_failures(item.requests, str(exc)))
            else:
                next_pending.append(_PendingBatch(item.requests, submission, item.deadline))
        pending = next_pending

    return results


def _chunks(requests: Sequence[BatchRequest], size: int) -> list[list[BatchRequest]]:
    return [list(requests[start : start + size]) for start in range(0, len(requests), size)]


def _is_complete(submission: BatchSubmission) -> bool:
    if submission.status not in FINAL_STATUSES:
        return False
    return not (submission.status == "completed" and submission.results is None)


def _collect(item: _PendingBatch, results: dict[str, BatchResult]) -> None:
    submission = item.submission
    if submission.status != "completed":
        detail = f": {submission.error}" if submission.error else ""
        results.update(
            _failures(
                item.requests,
                f"Batch {submission.id} ended with status {submission.status}{detail}",
            )
        )
        return

    chunk_results = {result.custom_id: result for result in submission.results or ()}
    for request in item.requests:
        results[request.custom_id] = chunk_results.get(
            request.custom_id,
            BatchResult(
                custom_id=request.custom_id,
                error=f"Batch {submission.id} returned no result",
            ),
        )


def _failures(
    requests: Sequence[BatchRequest],
    message: str,
) -> dict[str, BatchResult]:
    return {
        request.custom_id: BatchResult(custom_id=request.custom_id, error=message)
        for request in requests
    }


def _to_submission(batch: BatchObject) -> BatchSubmission:
    results = None
    if batch.results is not None:
        results = tuple(_to_result(result) for result in batch.results)
    error = batch.error.message if batch.error is not None else None
    return BatchSubmission(id=batch.id, status=batch.status, results=results, error=error)


def _to_result(result: Result) -> BatchResult:
    if result.error is not None:
        return BatchResult(custom_id=result.custom_id, error=result.error.message)
    return BatchResult(custom_id=result.custom_id, content=_chat_content(result))


def _chat_content(result: Result) -> str | None:
    if result.response is None:
        return None
    body = result.response.body
    if not isinstance(body, BodyChatCompletion):
        return None
    for choice in body.choices:
        if choice.message.content:
            return choice.message.content
    return None
