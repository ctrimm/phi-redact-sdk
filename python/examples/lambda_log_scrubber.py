#!/usr/bin/env python3
"""Lambda log scrubber — pipeline-side redaction for CloudWatch Logs.

Deploy as the target of a CloudWatch Logs subscription filter: every
batch of log events the source log group emits lands here first, each
event's message is scrubbed with phi-redact, and the scrubbed events
are forwarded to the destination chosen by LOG_SCRUB_DEST.

Placement semantics (read before deploying): this function scrubs the
*copy* in flight. The raw lines still exist in the SOURCE log group —
CloudWatch stores first and filters on delivery. Use this to protect
downstream systems (Datadog, archives, analytics, a longer-retention
clean log group). If the requirement is "PII must never be in
CloudWatch at all", scrub at emit time instead — see
phi_redact.integrations.logging_kit and docs/logging-integrations.md.

Configuration (environment variables):

    PHI_REDACT_MODEL / PHI_REDACT_TOKENIZER
        Optional ONNX model + tokenizer.json paths (bundle them in
        the deployment package or a layer and install the SDK's
        `onnx` extra). Unset = rules-only: deterministic identifiers
        (SSN, cards, phones, emails, dates...) are still scrubbed;
        names/addresses in prose need the neural layer.
    PHI_REDACT_OPERATOR   numbered (default) | label | mask | hash | remove
    PHI_REDACT_SALT       salt for the hash operator (use a secret —
        unsalted hashes of low-entropy values are dictionary-attackable)
    PHI_REDACT_THRESHOLD  neural score threshold (default 0.6)
    LOG_SCRUB_DEST        stdout (default) | datadog | cloudwatch
    DD_API_KEY            required for LOG_SCRUB_DEST=datadog
    LOG_SCRUB_TARGET_LOG_GROUP   required for LOG_SCRUB_DEST=cloudwatch

Destinations:

    stdout     print one JSON line per scrubbed event (logGroup,
               logStream, timestamp, message). The Lambda's own log
               group becomes the scrubbed store.
    datadog    POST batches to the Datadog Logs HTTP intake
               (https://http-intake.logs.datadoghq.com/api/v2/logs),
               stdlib urllib only.
    cloudwatch PutLogEvents into LOG_SCRUB_TARGET_LOG_GROUP, one
               stream per source stream (created on demand). boto3 is
               imported lazily — it ships with the Lambda Python
               runtime and is deliberately NOT an SDK dependency.

Robustness: a malformed batch never loses everything. Each event is
scrubbed under its own try/except; failures are counted, not raised.
The handler returns {"scrubbed": n, "forwarded": n, "errors": n}.
"""

from __future__ import annotations

import base64
import gzip
import json
import os
import sys
import urllib.request

# Allow running from a repo checkout without installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))

import phi_redact  # noqa: E402

DATADOG_INTAKE_URL = "https://http-intake.logs.datadoghq.com/api/v2/logs"
DATADOG_BATCH_SIZE = 500  # intake limit is 1000 entries / 5 MB per POST


def _build_redactor() -> phi_redact.Redactor:
    kwargs: dict = {}
    model = os.environ.get("PHI_REDACT_MODEL")
    tokenizer = os.environ.get("PHI_REDACT_TOKENIZER")
    if model:
        kwargs["model"] = model
    if tokenizer:
        kwargs["tokenizer"] = tokenizer
    operator = os.environ.get("PHI_REDACT_OPERATOR")
    if operator:
        kwargs["operator"] = operator
    salt = os.environ.get("PHI_REDACT_SALT")
    if salt:
        kwargs["salt"] = salt
    threshold = os.environ.get("PHI_REDACT_THRESHOLD")
    if threshold:
        kwargs["threshold"] = float(threshold)
    return phi_redact.Redactor(**kwargs)


# Built once per execution environment (cold start), reused across
# invocations — that is the point of keeping it at module level.
REDACTOR = _build_redactor()
DEST = os.environ.get("LOG_SCRUB_DEST", "stdout").strip().lower()


def _decode(event: dict) -> dict:
    raw = base64.b64decode(event["awslogs"]["data"])
    return json.loads(gzip.decompress(raw))


# --- destinations -----------------------------------------------------
# Each takes the scrubbed record list and returns how many records it
# forwarded. A record is {"logGroup", "logStream", "timestamp",
# "message"} with timestamp in epoch milliseconds (CloudWatch style).

def _forward_stdout(records: list[dict]) -> int:
    for rec in records:
        print(json.dumps(rec))
    return len(records)


def _forward_datadog(records: list[dict]) -> int:
    api_key = os.environ["DD_API_KEY"]
    forwarded = 0
    for i in range(0, len(records), DATADOG_BATCH_SIZE):
        chunk = records[i:i + DATADOG_BATCH_SIZE]
        entries = [{
            "message": rec["message"],
            "timestamp": rec["timestamp"],
            "ddsource": "phi-redact-log-scrubber",
            "ddtags": (f"log_group:{rec['logGroup']},"
                       f"log_stream:{rec['logStream']}"),
            "hostname": rec["logStream"],
        } for rec in chunk]
        req = urllib.request.Request(
            DATADOG_INTAKE_URL,
            data=json.dumps(entries).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json",
                     "DD-API-KEY": api_key})
        with urllib.request.urlopen(req, timeout=15) as resp:
            resp.read()
        forwarded += len(chunk)
    return forwarded


def _forward_cloudwatch(records: list[dict]) -> int:
    import boto3  # lazy: present in the Lambda runtime, not an SDK dep

    group = os.environ["LOG_SCRUB_TARGET_LOG_GROUP"]
    # One subscription payload carries exactly one source stream.
    stream = records[0]["logStream"] if records else "scrubbed"
    client = boto3.client("logs")
    try:
        client.create_log_stream(logGroupName=group, logStreamName=stream)
    except client.exceptions.ResourceAlreadyExistsException:
        pass
    token = None
    described = client.describe_log_streams(
        logGroupName=group, logStreamNamePrefix=stream)
    for s in described.get("logStreams", []):
        if s.get("logStreamName") == stream:
            token = s.get("uploadSequenceToken")
            break
    # PutLogEvents caps a batch at 10,000 events / 1 MB; subscription
    # payloads are far below that. Sort defensively — the API requires
    # chronological order.
    events = [{"timestamp": rec["timestamp"], "message": rec["message"]}
              for rec in sorted(records, key=lambda r: r["timestamp"] or 0)]
    kwargs = {"logGroupName": group, "logStreamName": stream,
              "logEvents": events}
    if token:
        kwargs["sequenceToken"] = token
    client.put_log_events(**kwargs)
    return len(events)


_FORWARDERS = {
    "stdout": _forward_stdout,
    "datadog": _forward_datadog,
    "cloudwatch": _forward_cloudwatch,
}


# --- handler ----------------------------------------------------------

def lambda_handler(event, context=None) -> dict:
    counts = {"scrubbed": 0, "forwarded": 0, "errors": 0}
    try:
        payload = _decode(event)
    except Exception:
        counts["errors"] = 1
        return counts
    if payload.get("messageType") == "CONTROL_MESSAGE":
        return counts  # CloudWatch health check; nothing to forward.

    log_group = payload.get("logGroup", "")
    log_stream = payload.get("logStream", "")
    records: list[dict] = []
    for log_event in payload.get("logEvents", []):
        try:
            message = REDACTOR.redact(log_event.get("message", "")).redacted
        except Exception:
            counts["errors"] += 1
            continue
        counts["scrubbed"] += 1
        records.append({"logGroup": log_group, "logStream": log_stream,
                        "timestamp": log_event.get("timestamp"),
                        "message": message})

    if records:
        try:
            counts["forwarded"] = _FORWARDERS[DEST](records)
        except Exception:
            counts["errors"] += len(records)
    return counts


if __name__ == "__main__":
    # Local smoke test: synthesize a subscription event and handle it.
    sample = {
        "messageType": "DATA_MESSAGE",
        "owner": "123456789012",
        "logGroup": "/aws/lambda/example",
        "logStream": "2026/10/27/[$LATEST]smoke",
        "subscriptionFilters": ["scrub"],
        "logEvents": [
            {"id": "1", "timestamp": 1792800000000,
             "message": "patient SSN 234-56-7890 phoned from 415-555-0132"},
        ],
    }
    envelope = {"awslogs": {"data": base64.b64encode(
        gzip.compress(json.dumps(sample).encode("utf-8"))).decode("ascii")}}
    print(lambda_handler(envelope))
