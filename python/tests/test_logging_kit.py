"""Tests for the 0.6.0 logging kit (Python SDK).

Covers phi_redact.integrations.logging_kit (stdlib logging filter,
structlog processor) and examples/lambda_log_scrubber.py (CloudWatch
subscription-filter handler). Everything runs against rules-only
Redactors — no model download — with synthetic fixtures.
"""

from __future__ import annotations

import base64
import gzip
import importlib.util
import io
import json
import logging
import os
import sys
import types
import urllib.request

import pytest

import phi_redact
from phi_redact.integrations.logging_kit import (
    RedactingFilter,
    add_redacting_filter,
    make_structlog_processor,
)

SSN = "234-56-7890"
PHONE = "415-555-0132"
EMAIL = "jane@example.com"
NAME = "Jane Doe"


def make_logger(name: str, redactor=None, attach_to: str = "logger",
                **filter_kw):
    """A logger writing formatted lines into an in-memory stream."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    logger = logging.getLogger(name)
    logger.handlers = [handler]
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    filt = add_redacting_filter(
        logger if attach_to == "logger" else handler,
        redactor if redactor is not None else phi_redact.Redactor(),
        **filter_kw)
    return logger, stream, filt


# --- RedactingFilter ---------------------------------------------------

def test_filter_scrubs_identifiers_from_message():
    logger, stream, _ = make_logger("t.filter.basic")
    logger.info("patient SSN %s phoned from %s", SSN, PHONE)
    out = stream.getvalue()
    assert SSN not in out and PHONE not in out
    assert "[SSN_1]" in out and "[PHONE_1]" in out


def test_filter_scrubs_literal_template_values():
    logger, stream, _ = make_logger("t.filter.literal")
    logger.info(f"card charged for {EMAIL}")
    out = stream.getvalue()
    assert EMAIL not in out
    assert "[EMAIL_1]" in out


def test_filter_passes_clean_line_unchanged():
    logger, stream, _ = make_logger("t.filter.clean")
    logger.info("startup complete in 42 ms")
    assert stream.getvalue() == "INFO startup complete in 42 ms\n"


def test_filter_attaches_to_handler_too():
    logger, stream, filt = make_logger("t.filter.handler",
                                       attach_to="handler")
    assert isinstance(filt, RedactingFilter)
    logger.info("call %s", PHONE)
    assert PHONE not in stream.getvalue()


def test_filter_scrubs_name_via_denylist_redactor():
    # Names are the neural layer's job; a denylist makes the rules-only
    # path deterministic for a known name in tests.
    redactor = phi_redact.Redactor(denylist=[NAME])
    logger, stream, _ = make_logger("t.filter.name", redactor=redactor)
    logger.info("discharge summary signed by %s", NAME)
    out = stream.getvalue()
    assert NAME not in out
    assert "[CUSTOM_1]" in out


def test_filter_scrubs_exception_traceback():
    logger, stream, _ = make_logger("t.filter.exc")
    try:
        raise ValueError(f"intake failed for SSN {SSN}")
    except ValueError:
        logger.exception("intake error")
    out = stream.getvalue()
    assert "intake error" in out
    assert "ValueError" in out  # traceback survived
    assert SSN not in out       # ...but its payload did not
    assert "[SSN_1]" in out


def test_filter_scrub_exc_disabled_leaves_traceback():
    logger, stream, _ = make_logger("t.filter.exc.off", scrub_exc=False)
    try:
        raise ValueError(f"intake failed for SSN {SSN}")
    except ValueError:
        logger.exception("intake error")
    assert SSN in stream.getvalue()


def test_filter_scrub_args_disabled_passes_message_through():
    logger, stream, _ = make_logger("t.filter.args.off", scrub_args=False)
    logger.info("patient SSN %s", SSN)
    assert SSN in stream.getvalue()


def test_filter_fails_open_on_redactor_error():
    class Broken:
        def redact(self, text):
            raise RuntimeError("boom")

    logger, stream, _ = make_logger("t.filter.failopen", redactor=Broken())
    logger.info("patient SSN %s", SSN)  # must not raise
    assert SSN in stream.getvalue()     # original record passed through


# --- structlog processor ----------------------------------------------

def test_structlog_processor_scrubs_event_and_nested_leaves():
    proc = make_structlog_processor(phi_redact.Redactor())
    event = {
        "event": f"call from {PHONE}",
        "patient": {"ssn": SSN, "visits": [f"call {PHONE}", 3, None]},
        "count": 2,
        "ok": True,
        "missing": None,
    }
    out = proc(None, "info", event)
    assert out["event"] == "call from [PHONE_1]"
    assert out["patient"]["ssn"] == "[SSN_1]"
    assert out["patient"]["visits"][0] == "call [PHONE_1]"
    assert out["patient"]["visits"][1] == 3
    assert out["patient"]["visits"][2] is None
    assert out["count"] == 2 and out["ok"] is True and out["missing"] is None
    # input not mutated
    assert event["event"] == f"call from {PHONE}"
    assert event["patient"]["ssn"] == SSN


def test_structlog_processor_scrubs_tuple_leaves():
    proc = make_structlog_processor(phi_redact.Redactor())
    out = proc(None, "info", {"event": "x", "pair": (PHONE, 7)})
    assert out["pair"] == ("[PHONE_1]", 7)


def test_structlog_processor_fails_open():
    class Broken:
        def redact(self, text):
            raise RuntimeError("boom")

    proc = make_structlog_processor(Broken())
    event = {"event": f"call from {PHONE}"}
    assert proc(None, "info", event) is event


# --- Lambda handler -----------------------------------------------------

EXAMPLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "examples", "lambda_log_scrubber.py")


def load_lambda_module():
    spec = importlib.util.spec_from_file_location("lambda_log_scrubber",
                                                  EXAMPLE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def subscription_event(messages, log_group="/aws/lambda/intake",
                       log_stream="2026/10/27/[$LATEST]abc"):
    payload = {
        "messageType": "DATA_MESSAGE",
        "owner": "123456789012",
        "logGroup": log_group,
        "logStream": log_stream,
        "subscriptionFilters": ["scrub-all"],
        "logEvents": [
            {"id": str(i + 1), "timestamp": 1792800000000 + i,
             "message": m}
            for i, m in enumerate(messages)
        ],
    }
    data = base64.b64encode(
        gzip.compress(json.dumps(payload).encode("utf-8"))).decode("ascii")
    return {"awslogs": {"data": data}}


@pytest.fixture()
def lam(monkeypatch):
    monkeypatch.delenv("PHI_REDACT_MODEL", raising=False)
    monkeypatch.delenv("PHI_REDACT_TOKENIZER", raising=False)
    monkeypatch.delenv("LOG_SCRUB_DEST", raising=False)
    return load_lambda_module()


def test_lambda_stdout_mode_scrubs_and_counts(lam, capsys):
    lam.DEST = "stdout"
    result = lam.lambda_handler(subscription_event(
        [f"patient SSN {SSN} phoned from {PHONE}", "routine heartbeat"]))
    assert result == {"scrubbed": 2, "forwarded": 2, "errors": 0}
    lines = [json.loads(l) for l in capsys.readouterr().out.splitlines()]
    assert len(lines) == 2
    assert SSN not in lines[0]["message"] and PHONE not in lines[0]["message"]
    assert "[SSN_1]" in lines[0]["message"]
    assert "[PHONE_1]" in lines[0]["message"]
    assert lines[0]["logGroup"] == "/aws/lambda/intake"
    assert lines[0]["timestamp"] == 1792800000000
    assert lines[1]["message"] == "routine heartbeat"


def test_lambda_datadog_mode_posts_scrubbed_batch(lam, monkeypatch):
    lam.DEST = "datadog"
    monkeypatch.setenv("DD_API_KEY", "test-key-123")
    calls = []

    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b"{}"

    def fake_urlopen(req, timeout=None):
        calls.append(req)
        return FakeResp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    result = lam.lambda_handler(subscription_event([f"SSN {SSN}"]))
    assert result == {"scrubbed": 1, "forwarded": 1, "errors": 0}
    assert len(calls) == 1
    req = calls[0]
    assert req.full_url == lam.DATADOG_INTAKE_URL
    assert req.get_header("Dd-api-key") == "test-key-123"
    body = json.loads(req.data.decode("utf-8"))
    assert len(body) == 1
    assert body[0]["message"] == "SSN [SSN_1]"
    assert body[0]["ddsource"] == "phi-redact-log-scrubber"
    assert "log_group:/aws/lambda/intake" in body[0]["ddtags"]


def test_lambda_cloudwatch_mode_puts_scrubbed_events(lam, monkeypatch):
    lam.DEST = "cloudwatch"
    monkeypatch.setenv("LOG_SCRUB_TARGET_LOG_GROUP", "/clean/intake")
    calls = {"put": [], "created": []}

    class AlreadyExists(Exception):
        pass

    class FakeExceptions:
        ResourceAlreadyExistsException = AlreadyExists

    class FakeLogsClient:
        exceptions = FakeExceptions()

        def create_log_stream(self, logGroupName, logStreamName):
            calls["created"].append((logGroupName, logStreamName))
            raise AlreadyExists()

        def describe_log_streams(self, logGroupName, logStreamNamePrefix):
            return {"logStreams": [
                {"logStreamName": logStreamNamePrefix,
                 "uploadSequenceToken": "tok-1"}]}

        def put_log_events(self, **kwargs):
            calls["put"].append(kwargs)
            return {"nextSequenceToken": "tok-2"}

    fake_boto3 = types.SimpleNamespace(
        client=lambda service: FakeLogsClient())
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)

    result = lam.lambda_handler(subscription_event([f"call {PHONE}"]))
    assert result == {"scrubbed": 1, "forwarded": 1, "errors": 0}
    assert calls["created"] == [("/clean/intake", "2026/10/27/[$LATEST]abc")]
    assert len(calls["put"]) == 1
    put = calls["put"][0]
    assert put["logGroupName"] == "/clean/intake"
    assert put["sequenceToken"] == "tok-1"
    assert put["logEvents"] == [
        {"timestamp": 1792800000000, "message": "call [PHONE_1]"}]


def test_lambda_control_message_is_a_noop(lam):
    payload = {"messageType": "CONTROL_MESSAGE", "logEvents": []}
    data = base64.b64encode(
        gzip.compress(json.dumps(payload).encode("utf-8"))).decode("ascii")
    result = lam.lambda_handler({"awslogs": {"data": data}})
    assert result == {"scrubbed": 0, "forwarded": 0, "errors": 0}


def test_lambda_malformed_event_counts_error_not_raise(lam):
    result = lam.lambda_handler({"awslogs": {"data": "!!!not-base64!!!"}})
    assert result == {"scrubbed": 0, "forwarded": 0, "errors": 1}
    result = lam.lambda_handler({})
    assert result == {"scrubbed": 0, "forwarded": 0, "errors": 1}


def test_lambda_one_bad_event_does_not_lose_the_batch(lam, capsys,
                                                      monkeypatch):
    # A redactor that blows up on one specific message: that event is
    # counted as an error, the rest of the batch still forwards.
    class Flaky:
        def redact(self, text):
            if "explode" in text:
                raise RuntimeError("boom")
            return phi_redact.Redactor().redact(text)

    lam.DEST = "stdout"
    monkeypatch.setattr(lam, "REDACTOR", Flaky())
    result = lam.lambda_handler(subscription_event(
        [f"call {PHONE}", "explode now", "all quiet"]))
    assert result == {"scrubbed": 2, "forwarded": 2, "errors": 1}
    out = capsys.readouterr().out
    assert "[PHONE_1]" in out and "all quiet" in out
