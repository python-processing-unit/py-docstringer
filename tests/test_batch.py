from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, cast

from openrouter import OpenRouter
from openrouter.components.batchobject import BatchObject
from openrouter.components.batchsubmitbody import BatchSubmitBody

from py_docstringer.batch import (
    BatchRequest,
    BatchResult,
    BatchSubmission,
    BatchTransport,
    OpenRouterTransport,
    run_batches,
)
from py_docstringer.errors import BatchError


def _request(custom_id: str = "d0") -> BatchRequest:
    return BatchRequest(
        custom_id=custom_id,
        messages=({"role": "user", "content": "Write the docstring."},),
    )


def _batch_object(
    *,
    content: str = "Generated docstring.",
    result_error: str | None = None,
    batch_error: str | None = None,
) -> BatchObject:
    result = {
        "custom_id": "d0",
        "error": (
            {"message": result_error, "param": None, "type": "invalid_request"}
            if result_error
            else None
        ),
        "id": "request_1",
        "response": None
        if result_error
        else {
            "body": {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "index": 0,
                        "logprobs": None,
                        "message": {
                            "content": content,
                            "refusal": None,
                            "role": "assistant",
                        },
                        "native_finish_reason": "stop",
                    }
                ],
                "created": 1,
                "id": "generation_1",
                "model": "test/model",
                "object": "chat.completion",
                "system_fingerprint": None,
            },
            "request_id": "request_1",
            "status_code": 200,
        },
    }
    return BatchObject.model_validate(
        {
            "completion_window": "24h",
            "created_at": 1,
            "endpoint": "/v1/chat/completions",
            "error": {"message": batch_error} if batch_error else None,
            "expires_at": None,
            "finalized_at": None,
            "id": "batch_1",
            "model": "test/model",
            "object": "batch",
            "provider": "Test",
            "request_counts": {"completed": 1, "failed": 0, "total": 1},
            "results": [result],
            "status": "completed",
            "usage": None,
        }
    )


class _FakeBatchEndpoint:
    def __init__(self, batch: BatchObject) -> None:
        self.batch = batch
        self.create_kwargs: dict[str, Any] | None = None
        self.fetched_id: str | None = None

    def create_batches(self, **kwargs: Any) -> BatchObject:
        self.create_kwargs = kwargs
        return self.batch

    def get_batches(self, id: str) -> BatchObject:
        self.fetched_id = id
        return self.batch


class _FakeOpenRouter:
    def __init__(self, batch: BatchObject) -> None:
        self.batch = _FakeBatchEndpoint(batch)


def test_openrouter_transport_converts_create_response() -> None:
    fake = _FakeOpenRouter(_batch_object(content="Return a value."))
    transport = OpenRouterTransport(cast(OpenRouter, fake), "test/model")

    submission = transport.create([_request()])

    assert submission.id == "batch_1"
    assert submission.status == "completed"
    assert submission.results == (BatchResult("d0", content="Return a value."),)
    assert fake.batch.create_kwargs is not None
    assert list(fake.batch.create_kwargs) == [
        "endpoint",
        "model",
        "completion_window",
        "requests",
    ]
    assert fake.batch.create_kwargs["endpoint"] == "/v1/chat/completions"
    assert fake.batch.create_kwargs["model"] == "test/model"
    assert fake.batch.create_kwargs["completion_window"] == "24h"
    assert fake.batch.create_kwargs["requests"][0]["custom_id"] == "d0"
    body = fake.batch.create_kwargs["requests"][0]["body"]
    assert body["messages"][0]["content"] == "Write the docstring."
    assert body["temperature"] == 0.2


def test_openrouter_transport_orders_batch_body_fields_for_api() -> None:
    fake = _FakeOpenRouter(_batch_object())
    OpenRouterTransport(cast(OpenRouter, fake), "test/model")

    body = BatchSubmitBody(
        endpoint="/v1/chat/completions",
        model="test/model",
        completion_window="24h",
        requests=[],
    )
    keys = list(json.loads(body.model_dump_json(exclude_none=True)))

    assert keys == ["endpoint", "model", "completion_window", "requests"]
    assert keys.index("completion_window") < keys.index("requests")


def test_openrouter_transport_converts_result_error() -> None:
    fake = _FakeOpenRouter(_batch_object(result_error="bad request"))
    transport = OpenRouterTransport(cast(OpenRouter, fake), "test/model")

    submission = transport.create([_request()])

    assert submission.results == (BatchResult("d0", error="bad request"),)


def test_openrouter_transport_converts_batch_error() -> None:
    fake = _FakeOpenRouter(_batch_object(batch_error="provider failed"))
    transport = OpenRouterTransport(cast(OpenRouter, fake), "test/model")

    submission = transport.create([_request()])

    assert submission.error == "provider failed"


def test_openrouter_transport_fetches_by_id() -> None:
    fake = _FakeOpenRouter(_batch_object())
    transport = OpenRouterTransport(cast(OpenRouter, fake), "test/model")

    submission = transport.fetch("batch_1")

    assert submission.id == "batch_1"
    assert fake.batch.fetched_id == "batch_1"


class _SequenceTransport:
    def __init__(self, states: Sequence[BatchSubmission]) -> None:
        self._states = iter(states)
        self.fetch_calls = 0

    def create(self, requests: Sequence[BatchRequest]) -> BatchSubmission:
        return next(self._states)

    def fetch(self, batch_id: str) -> BatchSubmission:
        self.fetch_calls += 1
        return next(self._states)


def test_run_batches_polls_until_completed() -> None:
    transport = _SequenceTransport(
        [
            BatchSubmission("batch_1", "validating"),
            BatchSubmission("batch_1", "in_progress"),
            BatchSubmission("batch_1", "completed", (BatchResult("d0", "Doc."),)),
        ]
    )
    sleeps: list[float] = []

    results = run_batches(
        cast(BatchTransport, transport),
        [_request()],
        poll_interval=5,
        timeout=60,
        sleep=sleeps.append,
        monotonic=lambda: 0.0,
    )

    assert results == {"d0": BatchResult("d0", "Doc.")}
    assert transport.fetch_calls == 2
    assert sleeps == [5, 5]


def test_run_batches_keeps_polling_when_results_are_not_ready() -> None:
    transport = _SequenceTransport(
        [
            BatchSubmission("batch_1", "validating"),
            BatchSubmission("batch_1", "completed"),
            BatchSubmission("batch_1", "completed", (BatchResult("d0", "Doc."),)),
        ]
    )

    results = run_batches(
        cast(BatchTransport, transport),
        [_request()],
        poll_interval=1,
        timeout=60,
        sleep=lambda _: None,
        monotonic=lambda: 0.0,
    )

    assert results["d0"].content == "Doc."
    assert transport.fetch_calls == 2


def test_run_batches_converts_terminal_failure_to_result_error() -> None:
    transport = _SequenceTransport(
        [BatchSubmission("batch_1", "failed", error="provider rejected the batch")]
    )

    results = run_batches(
        cast(BatchTransport, transport),
        [_request()],
        poll_interval=1,
        timeout=60,
        sleep=lambda _: None,
        monotonic=lambda: 0.0,
    )

    assert results["d0"].error is not None
    assert "provider rejected the batch" in results["d0"].error


def test_run_batches_times_out() -> None:
    class _StuckTransport:
        def create(self, requests: Sequence[BatchRequest]) -> BatchSubmission:
            return BatchSubmission("batch_1", "validating")

        def fetch(self, batch_id: str) -> BatchSubmission:
            return BatchSubmission("batch_1", "validating")

    clock = [0.0]
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock[0] += seconds

    results = run_batches(
        cast(BatchTransport, _StuckTransport()),
        [_request()],
        poll_interval=2,
        timeout=5,
        sleep=sleep,
        monotonic=lambda: clock[0],
    )

    assert results["d0"].error is not None
    assert "Timed out" in results["d0"].error
    assert sleeps == [2, 2, 1]


def test_run_batches_converts_missing_results_to_error() -> None:
    transport = _SequenceTransport(
        [
            BatchSubmission("batch_1", "completed", (BatchResult("other", "Doc."),)),
        ]
    )

    results = run_batches(
        cast(BatchTransport, transport),
        [_request()],
        poll_interval=1,
        timeout=60,
        sleep=lambda _: None,
        monotonic=lambda: 0.0,
    )

    assert results["d0"].error is not None
    assert "returned no result" in results["d0"].error


def test_run_batches_keeps_completed_chunks_when_another_fails() -> None:
    class _MixedTransport:
        def __init__(self) -> None:
            self.create_calls = 0

        def create(self, requests: Sequence[BatchRequest]) -> BatchSubmission:
            self.create_calls += 1
            if self.create_calls == 1:
                return BatchSubmission(
                    "good",
                    "completed",
                    (BatchResult(requests[0].custom_id, "Doc."),),
                )
            return BatchSubmission("bad", "failed", error="chunk failed")

        def fetch(self, batch_id: str) -> BatchSubmission:
            raise AssertionError("fetch should not be called for terminal batches")

    results = run_batches(
        cast(BatchTransport, _MixedTransport()),
        [_request("a"), _request("b")],
        poll_interval=1,
        timeout=60,
        batch_size=1,
        sleep=lambda _: None,
        monotonic=lambda: 0.0,
    )

    assert results["a"].content == "Doc."
    assert results["b"].error is not None
    assert "chunk failed" in results["b"].error


def test_run_batches_converts_submission_failure_to_errors() -> None:
    class _FailingTransport:
        def create(self, requests: Sequence[BatchRequest]) -> BatchSubmission:
            raise BatchError("submission rejected")

        def fetch(self, batch_id: str) -> BatchSubmission:
            raise AssertionError("fetch should not be called")

    results = run_batches(
        cast(BatchTransport, _FailingTransport()),
        [_request("a"), _request("b")],
        poll_interval=1,
        timeout=60,
        batch_size=1,
        sleep=lambda _: None,
        monotonic=lambda: 0.0,
    )

    assert set(results) == {"a", "b"}
    assert all(result.error == "submission rejected" for result in results.values())


def test_run_batches_submits_all_chunks_before_polling() -> None:
    class _Transport:
        def __init__(self) -> None:
            self.created: list[str] = []
            self.batches: dict[str, tuple[BatchRequest, ...]] = {}

        def create(self, requests: Sequence[BatchRequest]) -> BatchSubmission:
            batch_id = f"batch_{len(self.created) + 1}"
            self.created.append(batch_id)
            self.batches[batch_id] = tuple(requests)
            if batch_id == "batch_1":
                return BatchSubmission(batch_id, "validating")
            return BatchSubmission(
                batch_id,
                "completed",
                (BatchResult(requests[0].custom_id, "Doc."),),
            )

        def fetch(self, batch_id: str) -> BatchSubmission:
            assert self.created == ["batch_1", "batch_2"]
            request = self.batches[batch_id][0]
            return BatchSubmission(
                batch_id,
                "completed",
                (BatchResult(request.custom_id, "Doc."),),
            )

    results = run_batches(
        cast(BatchTransport, _Transport()),
        [_request("a"), _request("b")],
        poll_interval=1,
        timeout=60,
        batch_size=1,
        sleep=lambda _: None,
        monotonic=lambda: 0.0,
    )

    assert results == {"a": BatchResult("a", "Doc."), "b": BatchResult("b", "Doc.")}


def test_run_batches_chunks_requests() -> None:
    class _EchoTransport:
        def __init__(self) -> None:
            self.batches: list[tuple[str, ...]] = []

        def create(self, requests: Sequence[BatchRequest]) -> BatchSubmission:
            self.batches.append(tuple(request.custom_id for request in requests))
            return BatchSubmission(
                f"batch_{len(self.batches)}",
                "completed",
                tuple(BatchResult(request.custom_id, "Doc.") for request in requests),
            )

        def fetch(self, batch_id: str) -> BatchSubmission:
            raise AssertionError("fetch should not be called for completed batches")

    transport = _EchoTransport()
    requests = [_request("a"), _request("b"), _request("c")]

    results = run_batches(
        cast(BatchTransport, transport),
        requests,
        poll_interval=1,
        timeout=60,
        batch_size=1,
        sleep=lambda _: None,
        monotonic=lambda: 0.0,
    )

    assert transport.batches == [("a",), ("b",), ("c",)]
    assert set(results) == {"a", "b", "c"}


def test_run_batches_returns_empty_for_no_requests() -> None:
    class _UnusedTransport:
        def create(self, requests: Sequence[BatchRequest]) -> BatchSubmission:
            raise AssertionError("create should not be called")

        def fetch(self, batch_id: str) -> BatchSubmission:
            raise AssertionError("fetch should not be called")

    assert (
        run_batches(
            cast(BatchTransport, _UnusedTransport()),
            [],
            poll_interval=1,
            timeout=60,
            sleep=lambda _: None,
            monotonic=lambda: 0.0,
        )
        == {}
    )
