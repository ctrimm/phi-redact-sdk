//! phi-redact Rust SDK.
//!
//! Two layers, one call:
//!
//! ```no_run
//! use phi_redact::{redact, restore, RedactOptions};
//!
//! let r = redact("Call Jane Doe at 415-555-0132", &[], &RedactOptions::default()).unwrap();
//! assert_eq!(r.redacted, "Call Jane Doe at [PHONE_1]");
//! assert_eq!(restore(&r.redacted, &r.mapping), "Call Jane Doe at 415-555-0132");
//! ```
//!
//! The local pipeline is the deterministic rules layer (checksum-gated
//! scans for cards, SSNs, IBANs, phones, emails, ...) plus optional
//! injected neural spans — the same architecture as the TypeScript
//! SDK. With the cargo feature `neural` enabled, `Redactor` can also
//! load the ONNX model and produce the neural spans itself.
//!
//! Merge policy: precedence rules > recognizers > neural — rule spans
//! win every overlap, recognizer spans beat neural spans but yield to
//! rules, neural spans that overlap a kept span are dropped;
//! placeholders are numbered per label in document order. Keep the
//! mapping for pseudonymized round-trips; delete it and the copy is
//! anonymized.
//!
//! Feature parity with the Python/TypeScript SDKs (0.5.0 core
//! surface): anonymize operators ("numbered", "label", "mask",
//! "hash", "remove"), allow/deny lists, batch redaction, a
//! scrub/restore pair for scrubbing prompts before a cloud LLM call
//! (including ChunkedRestorer for streaming output), custom
//! recognizers (patterns with base scores, context-word boosting,
//! optional checksum/callable validators, custom labels), and
//! per-label thresholds with a needs-review mode (`review_margin`):
//! scored spans that miss their threshold but land within the margin
//! come back on `result.review` for a human to decide.
//!
//! All offsets are character indices (Unicode scalar values),
//! matching the Python SDK.
//!
//! This SDK redacts text. It is not a certified de-identification
//! solution and makes no HIPAA Expert Determination or Safe Harbor
//! claim.

pub mod engine;
#[cfg(feature = "neural")]
pub mod neural;
pub mod rules;
pub mod text;
pub mod types;

pub use engine::{
    apply_allowlist, apply_placeholders, denylist_spans, effective_threshold,
    gather_entities, hash_token, inject_denylist, mask_value, merge_entities,
    partition_by_score, process_entities, recognizer_spans, redact, redact_batch,
    restore, scan, score_floor, scrub, validate_recognizer, ChunkedRestorer, Redactor,
    DEFAULT_DENY_LABEL, DEFAULT_MAX_LEN, DEFAULT_THRESHOLD, NAMED_VALIDATORS, OPERATORS,
};
pub use types::{
    parse_operator, AnonymizeOperator, Entity, EntitySource, Error, Recognizer,
    RedactOptions, RedactionResult, Result, ReviewCandidate, Validator, ValidatorName,
};

/// Crate version — kept in lockstep with the Python SDK (0.6.0).
pub const VERSION: &str = "0.6.0";
