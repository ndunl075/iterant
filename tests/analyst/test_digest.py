from __future__ import annotations

from iterant.analyst import FailureAnalyst, HttpMessagesClient
from iterant.contracts import Role, RunKey, RunResult, Split
from iterant.proxy import FakeLLM, anthropic_message
from iterant.proxy.core import CallContent

MODEL = "claude-haiku-5-5"


def result(task: str, split: Split, score: float | None, *, attempt: int = 0) -> RunResult:
    refused = score is None
    return RunResult(
        key=RunKey(1, "c1", task, 0, attempt),
        split=split,
        score=score,
        in_tok=1,
        out_tok=1,
        usd_micros=0,
        wall_s=1.0,
        refusal=refused,
        infra_err=False,
    )


def call(call_id: int, task: str, prompt: str, reply: str, *, attempt: int = 0) -> CallContent:
    return CallContent(
        call_id=call_id,
        key=RunKey(1, "c1", task, 0, attempt),
        role=Role.TARGET,
        request={"messages": [{"role": "user", "content": prompt}]},
        response={"content": [{"type": "text", "text": reply}]},
    )


def test_fallback_digest_without_a_client() -> None:
    analyst = FailureAnalyst(None, model=MODEL, effort="medium")
    out = analyst.digest(
        [result("t1", Split.TRAIN, 0.0), result("t2", Split.TRAIN, 1.0)],
        [call(1, "t1", "add 2 and 2", "the answer is 5")],
    )
    assert "Train runs: 2" in out and "Failed: 1" in out
    assert "t1 rep0 score 0.00" in out and "the answer is 5" in out
    assert "t2" not in out


def test_other_splits_never_reach_the_digest_or_the_model() -> None:
    with FakeLLM() as fake:
        fake.add(None, anthropic_message("The agent answered wrongly."))
        client = HttpMessagesClient(fake.base_url, "tok")
        analyst = FailureAnalyst(client, model=MODEL, effort="medium")
        results = [
            result("train-task", Split.TRAIN, 0.0),
            result("sel-task", Split.SELECT, 0.0),
            result("hold-task", Split.HOLDOUT, 0.0),
            result("test-task", Split.TEST, 0.0),
        ]
        content = [
            call(1, "train-task", "TRAIN-PROMPT", "train reply"),
            call(2, "sel-task", "SELECT-SECRET", "select reply"),
            call(3, "hold-task", "HOLDOUT-SECRET", "holdout reply"),
            call(4, "test-task", "TEST-SECRET", "test reply"),
            call(5, "unknown-task", "ORPHAN-SECRET", "orphan reply"),
        ]
        out = analyst.digest(results, content)
    sent = repr([r.body for r in fake.requests])
    for secret in ("SELECT-SECRET", "HOLDOUT-SECRET", "TEST-SECRET", "ORPHAN-SECRET"):
        assert secret not in out and secret not in sent
    for name in ("sel-task", "hold-task", "test-task", "unknown-task"):
        assert name not in out and name not in sent
    assert "Train runs: 1" in out and "train-task" in out
    assert len(fake.requests) == 1


def test_llm_note_is_used_and_request_is_pinned() -> None:
    with FakeLLM() as fake:
        fake.add(None, anthropic_message("It never called the calculator tool."))
        analyst = FailureAnalyst(
            HttpMessagesClient(fake.base_url, "tok"), model=MODEL, effort="medium"
        )
        out = analyst.digest(
            [result("t1", Split.TRAIN, 0.25)], [call(1, "t1", "compute 17*23", "about 400")]
        )
    assert "It never called the calculator tool." in out
    body = fake.requests[0].body
    assert body["model"] == MODEL and body["output_config"] == {"effort": "medium"}
    assert "compute 17*23" in repr(body["messages"])
    assert fake.requests[0].headers["x-api-key"] == "tok"


def test_refusal_and_http_error_fall_back_to_the_deterministic_summary() -> None:
    with FakeLLM() as fake:
        fake.add(None, anthropic_message("no", stop_reason="refusal"), times=1)
        fake.add(None, {"error": {"message": "boom"}}, status=500)
        analyst = FailureAnalyst(
            HttpMessagesClient(fake.base_url, "tok"), model=MODEL, effort="medium"
        )
        out = analyst.digest(
            [result("t1", Split.TRAIN, 0.0), result("t2", Split.TRAIN, 0.5)],
            [call(1, "t1", "p1", "reply one"), call(2, "t2", "p2", "reply two")],
        )
    assert out.count("no analysis available") == 2
    assert "reply one" in out and "reply two" in out


def test_redact_strips_hidden_check_text_everywhere() -> None:
    with FakeLLM() as fake:
        fake.add(None, anthropic_message("It failed tests/check_secret.py."))
        analyst = FailureAnalyst(
            HttpMessagesClient(fake.base_url, "tok"),
            model=MODEL,
            effort="medium",
            redact=["tests/check_secret.py", "EXPECTED-42"],
        )
        out = analyst.digest(
            [result("t1", Split.TRAIN, 0.0)],
            [call(1, "t1", "see tests/check_secret.py", "EXPECTED-42")],
        )
    assert "check_secret" not in out and "EXPECTED-42" not in out
    assert "check_secret" not in repr(fake.requests[0].body)
    assert "EXPECTED-42" not in repr(fake.requests[0].body)


def test_latest_attempt_wins_and_refusals_are_counted_not_failures() -> None:
    analyst = FailureAnalyst(None, model=MODEL, effort="medium")
    out = analyst.digest(
        [
            result("t1", Split.TRAIN, None, attempt=0),
            result("t1", Split.TRAIN, 1.0, attempt=1),
            result("t2", Split.TRAIN, None),
        ],
        [],
    )
    assert "Train runs: 2 (scored 1, refused 1" in out
    assert "Failed: 0" in out


def test_digest_is_capped() -> None:
    analyst = FailureAnalyst(None, model=MODEL, effort="medium", max_chars=300)
    results = [result(f"task-{i}", Split.TRAIN, 0.0) for i in range(30)]
    out = analyst.digest(results, [])
    assert len(out) <= 300 and out.endswith("[truncated]")
