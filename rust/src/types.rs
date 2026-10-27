//! Shared types for the phi-redact Rust SDK.
//!
//! These mirror the Python (`phi_redact.engine`) and TypeScript
//! (`src/types.ts`) SDKs. All offsets are **character** indices into
//! the source text (Unicode scalar values), exactly like Python
//! string indices — not byte offsets, not UTF-16 code units.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::rc::Rc;

/// Where an entity span came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EntitySource {
    Rule,
    Neural,
    Denylist,
    Recognizer,
}

impl EntitySource {
    pub fn as_str(&self) -> &'static str {
        match self {
            EntitySource::Rule => "rule",
            EntitySource::Neural => "neural",
            EntitySource::Denylist => "denylist",
            EntitySource::Recognizer => "recognizer",
        }
    }
}

/// How entity spans are rendered in the output. Only `Numbered` is
/// reversible (it returns a mapping); the others return an empty
/// mapping by design.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AnonymizeOperator {
    Numbered,
    Label,
    Mask,
    Hash,
    Remove,
}

impl AnonymizeOperator {
    pub fn as_str(&self) -> &'static str {
        match self {
            AnonymizeOperator::Numbered => "numbered",
            AnonymizeOperator::Label => "label",
            AnonymizeOperator::Mask => "mask",
            AnonymizeOperator::Hash => "hash",
            AnonymizeOperator::Remove => "remove",
        }
    }
}

impl fmt::Display for AnonymizeOperator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// All the ways the SDK refuses a call. Mirrors the `ValueError`s the
/// Python SDK raises for the same misconfigurations.
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// Unknown anonymize operator name.
    UnknownOperator(String),
    /// Invalid recognizer configuration (bad label/score/validator/...).
    InvalidRecognizer(String),
    /// Invalid per-label threshold or review-margin value.
    InvalidScoreOption(String),
    /// A recognizer pattern the Rust `regex` crate cannot compile
    /// (e.g. it uses lookahead, which Python `re` allows but the
    /// finite-automaton `regex` crate does not support).
    Regex(String),
    /// Neural engine load/inference failure (feature `neural`).
    Neural(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::UnknownOperator(op) => write!(
                f,
                "unknown operator {op:?}; expected one of \
                 [\"numbered\", \"label\", \"mask\", \"hash\", \"remove\"]"
            ),
            Error::InvalidRecognizer(m) => write!(f, "{m}"),
            Error::InvalidScoreOption(m) => write!(f, "{m}"),
            Error::Regex(m) => write!(f, "{m}"),
            Error::Neural(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

/// Parse an operator name ("numbered" | "label" | "mask" | "hash" |
/// "remove"), mirroring the Python engine's validation.
pub fn parse_operator(name: &str) -> Result<AnonymizeOperator> {
    match name {
        "numbered" => Ok(AnonymizeOperator::Numbered),
        "label" => Ok(AnonymizeOperator::Label),
        "mask" => Ok(AnonymizeOperator::Mask),
        "hash" => Ok(AnonymizeOperator::Hash),
        "remove" => Ok(AnonymizeOperator::Remove),
        other => Err(Error::UnknownOperator(other.to_string())),
    }
}

/// A detected span. `start`/`end` are character indices.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Entity {
    pub start: usize,
    pub end: usize,
    pub label: String,
    pub text: String,
    pub source: EntitySource,
    /// Mean token confidence; present only for scored spans (neural,
    /// recognizer). `None` spans (rules, denylist) are never gated by
    /// thresholds and never become review candidates.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
    /// Assigned by `redact`, e.g. `"[PHONE_1]"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub placeholder: Option<String>,
}

impl Entity {
    pub fn new(
        start: usize,
        end: usize,
        label: impl Into<String>,
        text: impl Into<String>,
        source: EntitySource,
    ) -> Self {
        Entity {
            start,
            end,
            label: label.into(),
            text: text.into(),
            source,
            score: None,
            placeholder: None,
        }
    }

    pub fn overlaps(&self, other: &Entity) -> bool {
        !(self.end <= other.start || self.start >= other.end)
    }
}

/// A scored span held back by the threshold gate for human review:
/// NOT redacted, never in the mapping, never counted.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReviewCandidate {
    pub start: usize,
    pub end: usize,
    pub label: String,
    pub text: String,
    pub score: f64,
    pub source: EntitySource,
}

/// The result of a redaction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RedactionResult {
    pub redacted: String,
    pub entities: Vec<Entity>,
    /// placeholder -> original text, in document order. Keep for
    /// pseudonymized round-trips; delete it and the copy is
    /// anonymized. Populated only for the "numbered" operator.
    pub mapping: IndexMap<String, String>,
    /// label -> count, in first-occurrence order.
    pub counts: IndexMap<String, usize>,
    /// `"rules"` for the local engine without injected neural spans,
    /// `"rules+neural"` when neural spans were merged in.
    pub engine: String,
    pub model: Option<String>,
    /// Scored spans held back by the threshold gate for human review.
    /// Empty unless review mode is on (`review_margin` set).
    pub review: Vec<ReviewCandidate>,
}

/// Named checksum validators a recognizer can reference by string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidatorName {
    Luhn,
    Ssn,
    Iban,
    Aba,
}

impl ValidatorName {
    pub fn parse(name: &str) -> Option<ValidatorName> {
        match name {
            "luhn" => Some(ValidatorName::Luhn),
            "ssn" => Some(ValidatorName::Ssn),
            "iban" => Some(ValidatorName::Iban),
            "aba" => Some(ValidatorName::Aba),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            ValidatorName::Luhn => "luhn",
            ValidatorName::Ssn => "ssn",
            ValidatorName::Iban => "iban",
            ValidatorName::Aba => "aba",
        }
    }
}

/// A validator gate on a recognizer's matched value: a named built-in
/// checksum, or a user predicate.
#[derive(Clone)]
pub enum Validator {
    Named(ValidatorName),
    Func(Rc<dyn Fn(&str) -> bool>),
}

impl fmt::Debug for Validator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Validator::Named(n) => write!(f, "Validator::Named({n:?})"),
            Validator::Func(_) => write!(f, "Validator::Func(..)"),
        }
    }
}

impl From<ValidatorName> for Validator {
    fn from(n: ValidatorName) -> Self {
        Validator::Named(n)
    }
}

/// A user-defined recognizer (Presidio-style): regex patterns that
/// produce scored spans under a built-in or custom label. Mirrors
/// Python's `PatternRecognizer` dataclass field for field.
#[derive(Debug, Clone)]
pub struct Recognizer {
    /// Entity label for matches, e.g. `"EMPLOYEE_ID"` (uppercase by
    /// convention; new labels are allowed).
    pub label: String,
    /// One or more regex source strings. The entity span is match
    /// group `group` (default 0 = whole match). Patterns must stick
    /// to the subset the Rust `regex` crate supports (no lookahead /
    /// lookbehind / backreferences).
    pub patterns: Vec<String>,
    /// Base confidence for a match. Default 0.85.
    pub score: f64,
    /// Context words: if any appears within `context_window`
    /// characters on either side of a match (the match itself
    /// excluded, case-insensitive substring), the score rises by
    /// `context_boost`, capped at 1.0.
    pub context: Option<Vec<String>>,
    /// Characters of surrounding text searched for context words.
    /// Default 40 per side.
    pub context_window: usize,
    /// Score increase on a context hit. Default 0.10.
    pub context_boost: f64,
    /// When true, a match with no context word nearby is dropped
    /// instead of kept at its base score.
    pub require_context: bool,
    /// Gate on the matched value. A value that fails the validator
    /// is not a match.
    pub validator: Option<Validator>,
    /// Regex group whose span/text is the entity. Default 0.
    pub group: usize,
}

impl Recognizer {
    pub fn new(label: impl Into<String>, patterns: Vec<String>) -> Recognizer {
        Recognizer {
            label: label.into(),
            patterns,
            score: 0.85,
            context: None,
            context_window: 40,
            context_boost: 0.10,
            require_context: false,
            validator: None,
            group: 0,
        }
    }

    pub fn with_score(mut self, score: f64) -> Self {
        self.score = score;
        self
    }

    pub fn with_context(mut self, words: Vec<String>) -> Self {
        self.context = Some(words);
        self
    }

    pub fn with_context_window(mut self, window: usize) -> Self {
        self.context_window = window;
        self
    }

    pub fn with_context_boost(mut self, boost: f64) -> Self {
        self.context_boost = boost;
        self
    }

    pub fn with_require_context(mut self, require: bool) -> Self {
        self.require_context = require;
        self
    }

    pub fn with_validator(mut self, validator: Validator) -> Self {
        self.validator = Some(validator);
        self
    }

    pub fn with_named_validator(mut self, name: ValidatorName) -> Self {
        self.validator = Some(Validator::Named(name));
        self
    }

    pub fn with_group(mut self, group: usize) -> Self {
        self.group = group;
        self
    }
}

/// Options accepted by `redact` / `redact_batch` / `scrub` and the
/// `Redactor` pipeline. Mirrors TypeScript's `RedactOptions`.
#[derive(Debug, Clone, Default)]
pub struct RedactOptions {
    /// Placeholder style. Default: Numbered.
    pub operator: Option<AnonymizeOperator>,
    /// Salt for the "hash" operator. Default "".
    pub salt: Option<String>,
    /// Terms that are never redacted (case-insensitive exact match).
    pub allowlist: Option<Vec<String>>,
    /// Terms that are always redacted (case-insensitive,
    /// word-boundary aware), labeled with `deny_label`. Deny wins
    /// over allow.
    pub denylist: Option<Vec<String>>,
    /// Label for denylist spans. Default "CUSTOM".
    pub deny_label: Option<String>,
    /// User-defined recognizers.
    pub recognizers: Option<Vec<Recognizer>>,
    /// Per-label score gates: a label's threshold overrides the
    /// global threshold (0.6) for scored spans of that label.
    pub thresholds: Option<IndexMap<String, f64>>,
    /// Needs-review band width: a scored span that misses its
    /// threshold but scores at or above (threshold - review_margin)
    /// is returned on `result.review` instead of being dropped.
    /// `None` (the default) disables review mode.
    pub review_margin: Option<f64>,
}
