//! Merge + placeholder engine (Rust port of `phi_redact/engine.py`,
//! mirroring the TypeScript SDK's local architecture: the pipeline
//! accepts injected neural spans).
//!
//! Layers, merged into one entity list:
//!   1. rules       — deterministic checksum-gated scans (see
//!                    `crate::rules`). Always runs. Wins every overlap.
//!   2. recognizers — user-supplied patterns with scores (0.5.0).
//!                    Deterministic user intent: beats neural, yields
//!                    to rules.
//!   3. neural      — spans injected by the caller (or produced by
//!                    the `neural`-feature ONNX engine on `Redactor`).
//!
//! Precedence: rules > recognizers > neural.

use indexmap::IndexMap;
use sha2::{Digest, Sha256};

use crate::rules;
use crate::text::{casefold, CharIndex};
use crate::types::{
    AnonymizeOperator, Entity, EntitySource, Error, Recognizer, RedactOptions,
    RedactionResult, Result, ReviewCandidate, Validator, ValidatorName,
};

/// Min mean-token confidence for neural spans (the v9 ship gate).
pub const DEFAULT_THRESHOLD: f64 = 0.6;
/// Tokens per ONNX chunk (incl. specials) for the neural engine.
pub const DEFAULT_MAX_LEN: usize = 384;
/// Label given to denylist-injected spans.
pub const DEFAULT_DENY_LABEL: &str = "CUSTOM";
/// The anonymize operators, by name.
pub const OPERATORS: [&str; 5] = ["numbered", "label", "mask", "hash", "remove"];

fn overlaps(a: &Entity, b: &Entity) -> bool {
    a.overlaps(b)
}

// ---------------------------------------------------------------------------
// Merge
// ---------------------------------------------------------------------------

/// Merge the detection layers. Precedence: rules > recognizers >
/// neural.
///
/// Rules win every overlap (checksum-gated precision). Recognizer
/// spans win overlaps against neural spans but yield to rule spans —
/// the built-in rules are effectively "registered first".
/// `recognizer_ents` must arrive in precedence order
/// (`recognizer_spans` returns them so); among themselves the first
/// wins. Neural-neural overlaps keep the best score.
pub fn merge_entities(
    rule_ents: &[Entity],
    neural_ents: &[Entity],
    recognizer_ents: &[Entity],
) -> Vec<Entity> {
    let mut base: Vec<Entity> = rule_ents.to_vec();
    for r in recognizer_ents {
        if base.iter().any(|b| overlaps(r, b)) {
            continue;
        }
        base.push(r.clone());
    }
    let mut neural_sorted: Vec<&Entity> = neural_ents.iter().collect();
    // Stable sort by descending score: Python/JS semantics.
    neural_sorted.sort_by(|a, b| {
        b.score
            .unwrap_or(0.0)
            .partial_cmp(&a.score.unwrap_or(0.0))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut kept: Vec<Entity> = Vec::new();
    for n in neural_sorted {
        if base.iter().any(|b| overlaps(n, b)) {
            continue;
        }
        if kept.iter().any(|k| overlaps(n, k)) {
            continue;
        }
        kept.push(n.clone());
    }
    let mut out = base;
    out.extend(kept);
    out.sort_by_key(|e| e.start);
    out
}

// ---------------------------------------------------------------------------
// Operators
// ---------------------------------------------------------------------------

/// Consistent pseudonym token for the "hash" operator:
/// `[LABEL_<8 hex>]` from SHA-256(salt \x00 label \x00 value). The
/// same value under the same label and salt yields the same token in
/// every document, so pseudonymized datasets still join. Identical
/// construction in the Python and TypeScript SDKs (shared test
/// vectors pin it).
///
/// The salt defaults to "" but should be a secret in practice: SSNs
/// and other low-entropy values are dictionary-attackable from an
/// unsalted truncated hash.
pub fn hash_token(label: &str, value: &str, salt: &str) -> String {
    let mut data = Vec::with_capacity(salt.len() + label.len() + value.len() + 2);
    data.extend_from_slice(salt.as_bytes());
    data.push(0);
    data.extend_from_slice(label.as_bytes());
    data.push(0);
    data.extend_from_slice(value.as_bytes());
    let digest = Sha256::digest(&data);
    let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
    format!("[{}_{}]", label, &hex[..8])
}

/// Labels whose values are structured IDs: masking keeps the last 4
/// characters (the part support workflows quote) when the value is
/// long enough for that to be safe.
const MASK_KEEP_LAST: [&str; 13] = [
    "SSN", "CREDITCARD", "PHONE", "TAX_ID", "ROUTING_NUMBER", "BANK_ACCOUNT", "ACCOUNT",
    "IBAN", "PASSPORT", "DRIVERS_LICENSE", "LICENSE", "GOVERNMENT_ID", "MRN",
];

/// Masked rendering for the "mask" operator. Structured IDs (see
/// MASK_KEEP_LAST) of length >= 8 keep their last 4 characters;
/// preceding alphanumeric characters become "*" and separators are
/// preserved ("234-56-7890" -> "***-**-7890"). Every other value is
/// masked to its own length in asterisks.
pub fn mask_value(label: &str, value: &str) -> String {
    let chars: Vec<char> = value.chars().collect();
    if MASK_KEEP_LAST.contains(&label) && chars.len() >= 8 {
        let head: String = chars[..chars.len() - 4]
            .iter()
            .map(|c| if c.is_ascii_alphanumeric() { '*' } else { *c })
            .collect();
        let tail: String = chars[chars.len() - 4..].iter().collect();
        return head + &tail;
    }
    "*".repeat(chars.len())
}

fn replacement(e: &Entity, n: usize, operator: AnonymizeOperator, salt: &str) -> String {
    match operator {
        AnonymizeOperator::Numbered => format!("[{}_{}]", e.label, n),
        AnonymizeOperator::Label => format!("<{}>", e.label),
        AnonymizeOperator::Hash => hash_token(&e.label, &e.text, salt),
        AnonymizeOperator::Mask => mask_value(&e.label, &e.text),
        AnonymizeOperator::Remove => String::new(),
    }
}

/// Render entities into text under the chosen anonymize operator.
///
/// Replacements are computed in forward document order (so the
/// "numbered" operator numbers per label in reading order), then
/// spliced back-to-front so offsets stay valid. Sets each entity's
/// `placeholder`. Returns (redacted_text, mapping); the mapping is
/// populated only for "numbered" — the other operators are
/// irreversible by design.
pub fn apply_placeholders(
    text: &str,
    entities: &mut [Entity],
    operator: AnonymizeOperator,
    salt: &str,
) -> (String, IndexMap<String, String>) {
    let idx = CharIndex::new(text);
    let mut counters: IndexMap<String, usize> = IndexMap::new();
    let mut mapping: IndexMap<String, String> = IndexMap::new();

    let mut order: Vec<usize> = (0..entities.len()).collect();
    order.sort_by_key(|&i| entities[i].start); // stable
    let mut reps: Vec<String> = vec![String::new(); entities.len()];
    for &i in &order {
        let e = &entities[i];
        let n = {
            let c = counters.entry(e.label.clone()).or_insert(0);
            *c += 1;
            *c
        };
        let rep = replacement(e, n, operator, salt);
        if operator == AnonymizeOperator::Numbered {
            mapping.insert(rep.clone(), e.text.clone());
        }
        reps[i] = rep;
    }
    for (i, e) in entities.iter_mut().enumerate() {
        e.placeholder = Some(reps[i].clone());
    }
    let mut out = text.to_string();
    for &i in order.iter().rev() {
        let e = &entities[i];
        out.replace_range(idx.byte_of(e.start)..idx.byte_of(e.end), &reps[i]);
    }
    (out, mapping)
}

// ---------------------------------------------------------------------------
// Allow / deny lists
// ---------------------------------------------------------------------------

/// Drop spans whose text matches an allowlist term (case-insensitive
/// exact match — a term protects exactly the value it names).
pub fn apply_allowlist(entities: Vec<Entity>, allowlist: Option<&[String]>) -> Vec<Entity> {
    let Some(allowlist) = allowlist else { return entities };
    if allowlist.is_empty() {
        return entities;
    }
    let allowed: std::collections::HashSet<String> =
        allowlist.iter().map(|t| casefold(t)).collect();
    entities
        .into_iter()
        .filter(|e| !allowed.contains(&casefold(&e.text)))
        .collect()
}

/// Process-global compiled-pattern cache — the analogue of CPython's
/// `re` module cache, which the Python SDK implicitly relies on (it
/// "recompiles" recognizer and denylist patterns on every call but
/// pays for it once). `regex::Regex::new` costs ~0.2 ms per pattern
/// in Unicode tables; without the cache it would dominate every
/// redaction that uses recognizers or a denylist.
static PATTERN_CACHE: std::sync::LazyLock<
    std::sync::Mutex<std::collections::HashMap<String, std::sync::Arc<regex::Regex>>>,
> = std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::HashMap::new()));

fn cached_regex(
    pattern: &str,
) -> std::result::Result<std::sync::Arc<regex::Regex>, regex::Error> {
    if let Ok(cache) = PATTERN_CACHE.lock() {
        if let Some(re) = cache.get(pattern) {
            return Ok(re.clone());
        }
    }
    let re = regex::Regex::new(pattern)?;
    if let Ok(mut cache) = PATTERN_CACHE.lock() {
        cache.insert(pattern.to_string(), std::sync::Arc::new(re.clone()));
    }
    Ok(std::sync::Arc::new(re))
}

/// Exact-match spans for denylist terms: case-insensitive,
/// word-boundary aware (a term never matches inside a larger word).
///
/// Python implements this with a `(?<!\w)term(?!\w)` regex;
/// lookarounds do not exist in the Rust `regex` crate, so candidates
/// are found literally and their neighbours checked by hand. A
/// candidate rejected by a boundary advances the scan by one
/// character, mirroring the Python engine.
pub fn denylist_spans(
    text: &str,
    denylist: &[String],
    label: &str,
) -> Vec<Entity> {
    use crate::text::{is_word_char, next_char_boundary, prev_char};
    let idx = CharIndex::new(text);
    let mut spans = Vec::new();
    for term in denylist {
        if term.is_empty() {
            continue;
        }
        let re = match cached_regex(&format!("(?i){}", regex::escape(term))) {
            Ok(re) => re,
            Err(_) => continue,
        };
        let mut pos = 0usize;
        while pos <= text.len() {
            let Some(m) = re.find_at(text, pos) else { break };
            let prev_ok = match prev_char(text, m.start()) {
                None => true,
                Some(c) => !is_word_char(c),
            };
            let next_ok = match text[m.end()..].chars().next() {
                None => true,
                Some(c) => !is_word_char(c),
            };
            if prev_ok && next_ok {
                spans.push(Entity::new(
                    idx.char_of_byte(m.start()),
                    idx.char_of_byte(m.end()),
                    label,
                    &text[m.start()..m.end()],
                    EntitySource::Denylist,
                ));
                pos = if m.end() > m.start() {
                    m.end()
                } else {
                    next_char_boundary(text, m.end())
                };
            } else {
                pos = next_char_boundary(text, m.start());
            }
        }
    }
    spans
}

/// Add denylist spans to a detected entity list. A deny span wins
/// any overlap: a detected span it touches is dropped (the rest of
/// that span is left in the clear — the deny match is what the
/// operator asked to hide). Overlapping deny matches keep the first
/// in document order.
pub fn inject_denylist(
    text: &str,
    entities: Vec<Entity>,
    denylist: Option<&[String]>,
    label: &str,
) -> Vec<Entity> {
    let Some(denylist) = denylist else { return entities };
    if denylist.is_empty() {
        return entities;
    }
    let deny = denylist_spans(text, denylist, label);
    if deny.is_empty() {
        return entities;
    }
    let kept: Vec<Entity> = entities
        .into_iter()
        .filter(|e| !deny.iter().any(|d| overlaps(e, d)))
        .collect();
    let mut deny_sorted = deny;
    deny_sorted.sort_by_key(|d| d.start); // stable
    let mut merged: Vec<Entity> = Vec::new();
    for d in deny_sorted {
        if merged.iter().any(|k| overlaps(&d, k)) {
            continue;
        }
        merged.push(d);
    }
    let mut out = kept;
    out.extend(merged);
    out.sort_by_key(|e| e.start);
    out
}

/// Allowlist filter, then denylist injection. Deny wins over allow:
/// a term in both lists is redacted (as a deny span).
pub fn process_entities(
    text: &str,
    entities: Vec<Entity>,
    allowlist: Option<&[String]>,
    denylist: Option<&[String]>,
    deny_label: &str,
) -> Vec<Entity> {
    let entities = apply_allowlist(entities, allowlist);
    inject_denylist(text, entities, denylist, deny_label)
}

// ---------------------------------------------------------------------------
// Custom recognizers (0.5.0, Presidio parity)
// ---------------------------------------------------------------------------

/// The named checksum validators a recognizer can reference.
pub const NAMED_VALIDATORS: [ValidatorName; 4] = [
    ValidatorName::Luhn,
    ValidatorName::Ssn,
    ValidatorName::Iban,
    ValidatorName::Aba,
];

fn validator_fn(v: &Validator) -> Box<dyn Fn(&str) -> bool + '_> {
    match v {
        Validator::Named(ValidatorName::Luhn) => Box::new(rules::luhn_ok),
        Validator::Named(ValidatorName::Ssn) => Box::new(rules::ssn_valid),
        Validator::Named(ValidatorName::Iban) => Box::new(rules::iban_valid),
        Validator::Named(ValidatorName::Aba) => Box::new(rules::aba_routing_ok),
        Validator::Func(f) => Box::new(move |s: &str| f(s)),
    }
}

/// Configuration validation, mirroring `PatternRecognizer.__post_init__`.
pub fn validate_recognizer(rec: &Recognizer) -> Result<()> {
    if rec.label.is_empty() {
        return Err(Error::InvalidRecognizer(
            "recognizer label must be non-empty".into(),
        ));
    }
    if rec.patterns.is_empty() {
        return Err(Error::InvalidRecognizer(
            "recognizer needs at least one pattern".into(),
        ));
    }
    if !(0.0..=1.0).contains(&rec.score) {
        return Err(Error::InvalidRecognizer(format!(
            "recognizer score {} outside [0, 1]",
            rec.score
        )));
    }
    if rec.context_boost < 0.0 {
        return Err(Error::InvalidRecognizer("context_boost must be >= 0".into()));
    }
    if rec.require_context && rec.context.as_ref().map_or(true, |c| c.is_empty()) {
        return Err(Error::InvalidRecognizer(
            "require_context=true needs context words".into(),
        ));
    }
    Ok(())
}

fn round4(x: f64) -> f64 {
    (x * 10000.0).round() / 10000.0
}

fn context_present(text: &str, idx: &CharIndex, s: usize, e: usize, rec: &Recognizer) -> bool {
    let lo = s.saturating_sub(rec.context_window);
    let hi = (e + rec.context_window).min(idx.len());
    let window = format!(
        "{}{}",
        idx.slice(text, lo, s),
        idx.slice(text, e, hi)
    )
    .to_lowercase();
    rec.context
        .as_ref()
        .map(|words| {
            words
                .iter()
                .any(|w| !w.is_empty() && window.contains(&w.to_lowercase()))
        })
        .unwrap_or(false)
}

/// Run user recognizers over text.
///
/// Returns scored spans (source "recognizer") in PRECEDENCE order —
/// recognizer registration order, then pattern order, then document
/// order — with overlaps among them already resolved first-wins in
/// that order. `merge_entities` relies on this order; the final
/// entity list is sorted by start there.
pub fn recognizer_spans(text: &str, recognizers: &[Recognizer]) -> Result<Vec<Entity>> {
    let idx = CharIndex::new(text);
    let mut accepted: Vec<Entity> = Vec::new();
    for rec in recognizers {
        validate_recognizer(rec)?;
        let validate = rec.validator.as_ref().map(validator_fn);
        for pattern in &rec.patterns {
            let re = cached_regex(pattern).map_err(|err| {
                Error::Regex(format!(
                    "recognizer {:?}: invalid pattern {pattern:?}: {err}",
                    rec.label
                ))
            })?;
            let ngroups = re.capture_names().count();
            if rec.group > ngroups {
                return Err(Error::InvalidRecognizer(format!(
                    "recognizer {:?}: group {} but pattern {:?} has {} group(s)",
                    rec.label, rec.group, pattern, ngroups
                )));
            }
            for caps in re.captures_iter(text) {
                let Some(g) = caps.get(rec.group) else { continue };
                if g.start() == g.end() {
                    continue; // group did not participate / empty match
                }
                let value = g.as_str();
                if let Some(v) = &validate {
                    if !v(value) {
                        continue;
                    }
                }
                let s = idx.char_of_byte(g.start());
                let e = idx.char_of_byte(g.end());
                let mut score = rec.score;
                if rec.context.as_ref().map_or(false, |c| !c.is_empty()) {
                    if context_present(text, &idx, s, e, rec) {
                        score = (score + rec.context_boost).min(1.0);
                    } else if rec.require_context {
                        continue;
                    }
                }
                let span = Entity {
                    start: s,
                    end: e,
                    label: rec.label.clone(),
                    text: value.to_string(),
                    source: EntitySource::Recognizer,
                    score: Some(round4(score)),
                    placeholder: None,
                };
                if accepted.iter().any(|a| overlaps(&span, a)) {
                    continue;
                }
                accepted.push(span);
            }
        }
    }
    Ok(accepted)
}

// ---------------------------------------------------------------------------
// Score gate: per-label thresholds + needs-review mode (0.5.0)
// ---------------------------------------------------------------------------

/// The score gate for one label: its per-label override when set,
/// else the global threshold.
pub fn effective_threshold(
    label: &str,
    global_threshold: f64,
    thresholds: Option<&IndexMap<String, f64>>,
) -> f64 {
    if let Some(t) = thresholds {
        if let Some(v) = t.get(label) {
            return *v;
        }
    }
    global_threshold
}

fn validate_score_options(
    thresholds: Option<&IndexMap<String, f64>>,
    review_margin: Option<f64>,
) -> Result<()> {
    if let Some(t) = thresholds {
        for (label, v) in t {
            if !(0.0..=1.0).contains(v) {
                return Err(Error::InvalidScoreOption(format!(
                    "threshold for {label:?} is {v}, outside [0, 1]"
                )));
            }
        }
    }
    if let Some(m) = review_margin {
        if !(0.0..=1.0).contains(&m) {
            return Err(Error::InvalidScoreOption(format!(
                "review_margin {m} outside [0, 1]"
            )));
        }
    }
    Ok(())
}

/// Lowest score a span can survive with. Detectors are asked for
/// spans down to this floor so the partition — not the detector —
/// decides what is redacted, reviewed, or dropped.
pub fn score_floor(
    global_threshold: f64,
    thresholds: Option<&IndexMap<String, f64>>,
    review_margin: Option<f64>,
) -> f64 {
    let mut floor = global_threshold;
    if let Some(t) = thresholds {
        for v in t.values() {
            floor = floor.min(*v);
        }
    }
    if let Some(m) = review_margin {
        floor -= m;
    }
    floor.max(0.0)
}

/// Split merged entities into (redact, review) by score.
///
/// Only spans carrying a score are gated — neural spans and
/// recognizer spans. Spans WITHOUT a score (the checksum-gated rules
/// layer, denylist injections) are deterministic by construction and
/// always redact: a per-label threshold on "SSN" does not stop the
/// SSN rule, and a threshold never turns a rule span into a review
/// candidate.
///
/// A scored span redacts when score >= its effective threshold.
/// Below that, when `review_margin` is set, a span scoring at or
/// above (threshold - review_margin) becomes a review candidate:
/// NOT redacted, returned for a human to decide. Anything lower is
/// dropped. With `review_margin = None` (the default) the review
/// list is always empty.
pub fn partition_by_score(
    entities: Vec<Entity>,
    global_threshold: f64,
    thresholds: Option<&IndexMap<String, f64>>,
    review_margin: Option<f64>,
) -> Result<(Vec<Entity>, Vec<ReviewCandidate>)> {
    validate_score_options(thresholds, review_margin)?;
    let mut kept = Vec::new();
    let mut review = Vec::new();
    for e in entities {
        let Some(score) = e.score else {
            kept.push(e);
            continue;
        };
        let eff = effective_threshold(&e.label, global_threshold, thresholds);
        if score >= eff {
            kept.push(e);
        } else if let Some(margin) = review_margin {
            if score >= eff - margin {
                review.push(ReviewCandidate {
                    start: e.start,
                    end: e.end,
                    label: e.label,
                    text: e.text,
                    score: e.score.unwrap(),
                    source: e.source,
                });
            }
        }
    }
    review.sort_by_key(|c| c.start);
    Ok((kept, review))
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

/// Full detection pipeline: rules scan + recognizer scan + injected
/// neural spans, merged rules > recognizers > neural, allow/deny
/// applied, then the score gate. Returns (entities_to_redact,
/// review_candidates).
pub fn gather_entities(
    text: &str,
    neural_spans: &[Entity],
    options: &RedactOptions,
    global_threshold: f64,
) -> Result<(Vec<Entity>, Vec<ReviewCandidate>)> {
    validate_score_options(options.thresholds.as_ref(), options.review_margin)?;
    let rec_ents = match &options.recognizers {
        Some(recs) if !recs.is_empty() => recognizer_spans(text, recs)?,
        _ => Vec::new(),
    };
    let merged = merge_entities(&rules::scan(text), neural_spans, &rec_ents);
    let processed = process_entities(
        text,
        merged,
        options.allowlist.as_deref(),
        options.denylist.as_deref(),
        options.deny_label.as_deref().unwrap_or(DEFAULT_DENY_LABEL),
    );
    partition_by_score(
        processed,
        global_threshold,
        options.thresholds.as_ref(),
        options.review_margin,
    )
}

fn finish(
    text: &str,
    mut kept: Vec<Entity>,
    review: Vec<ReviewCandidate>,
    options: &RedactOptions,
    engine: &str,
    model: Option<String>,
) -> RedactionResult {
    let (redacted, mapping) = apply_placeholders(
        text,
        &mut kept,
        options.operator.unwrap_or(AnonymizeOperator::Numbered),
        options.salt.as_deref().unwrap_or(""),
    );
    let mut counts: IndexMap<String, usize> = IndexMap::new();
    for e in &kept {
        *counts.entry(e.label.clone()).or_insert(0) += 1;
    }
    RedactionResult {
        redacted,
        entities: kept,
        mapping,
        counts,
        engine: engine.to_string(),
        model,
        review,
    }
}

/// Rules-only entity scan. Returns entities sorted by start.
pub fn scan(text: &str) -> Vec<Entity> {
    rules::scan(text)
}

/// Redact text with the local pipeline plus injected neural spans
/// (mirrors the TypeScript SDK's `redact`). The score gate is the
/// SDK default threshold (0.6), overridable per label via
/// `options.thresholds`.
pub fn redact(
    text: &str,
    neural_spans: &[Entity],
    options: &RedactOptions,
) -> Result<RedactionResult> {
    let (kept, review) = gather_entities(text, neural_spans, options, DEFAULT_THRESHOLD)?;
    let engine = if neural_spans.is_empty() { "rules" } else { "rules+neural" };
    Ok(finish(text, kept, review, options, engine, None))
}

/// Redact many texts, one result per input, in input order — the
/// batch-pipeline entry point.
pub fn redact_batch(
    texts: &[String],
    options: &RedactOptions,
) -> Result<Vec<RedactionResult>> {
    texts.iter().map(|t| redact(t, &[], options)).collect()
}

/// Scrub text for a trip outside the trust boundary (e.g. a cloud
/// LLM call): returns (redacted_text, mapping). Send the redacted
/// text out, keep the mapping local, restore the response with
/// `restore` — or with `ChunkedRestorer` when the response streams.
/// Round-trips require the default "numbered" operator; the other
/// operators return an empty mapping by design.
pub fn scrub(
    text: &str,
    neural_spans: &[Entity],
    options: &RedactOptions,
) -> Result<(String, IndexMap<String, String>)> {
    let r = redact(text, neural_spans, options)?;
    Ok((r.redacted, r.mapping))
}

/// Put real values back using a mapping from `redact`.
pub fn restore(redacted: &str, mapping: &IndexMap<String, String>) -> String {
    rules::restore_text(redacted, mapping)
}

// ---------------------------------------------------------------------------
// Streaming restore (LLM round-trips)
// ---------------------------------------------------------------------------

/// Length (in characters) of the longest buffer suffix that is a
/// proper prefix of some mapping key — text that might still grow
/// into a placeholder when the next chunk arrives, so it must not be
/// emitted yet.
fn partial_suffix_len(buf: &str, keys: &[&String]) -> usize {
    if keys.is_empty() {
        return 0;
    }
    let buf_chars: Vec<char> = buf.chars().collect();
    let key_chars: Vec<Vec<char>> = keys.iter().map(|k| k.chars().collect()).collect();
    let maxk = key_chars.iter().map(|k| k.len()).max().unwrap_or(0);
    let mut best = 0;
    let limit = buf_chars.len().min(maxk.saturating_sub(1));
    for n in 1..=limit {
        let suffix = &buf_chars[buf_chars.len() - n..];
        if key_chars
            .iter()
            .any(|k| k.len() > n && k[..n] == suffix[..])
        {
            best = n;
        }
    }
    best
}

/// Restore placeholders from text that arrives in chunks (e.g. a
/// streaming LLM response built on scrubbed input).
///
/// `feed` returns the restored text that is safe to emit now; a
/// trailing partial placeholder is buffered until it completes or
/// `flush` proves it was never a placeholder. Placeholders split
/// across chunks are restored whole; unknown or partial placeholders
/// pass through untouched — feed/flush never panic and never invent
/// text.
pub struct ChunkedRestorer {
    mapping: IndexMap<String, String>,
    buf: String,
}

impl ChunkedRestorer {
    pub fn new(mapping: IndexMap<String, String>) -> ChunkedRestorer {
        ChunkedRestorer { mapping, buf: String::new() }
    }

    pub fn feed(&mut self, chunk: &str) -> String {
        self.buf.push_str(chunk);
        let keys: Vec<&String> = self.mapping.keys().collect();
        let hold = partial_suffix_len(&self.buf, &keys);
        if hold > 0 {
            let total = self.buf.chars().count();
            let emit_end_char = total - hold;
            let byte = crate::text::CharIndex::new(&self.buf).byte_of(emit_end_char);
            let emit: String = self.buf.drain(..byte).collect();
            rules::restore_text(&emit, &self.mapping)
        } else {
            let emit = std::mem::take(&mut self.buf);
            rules::restore_text(&emit, &self.mapping)
        }
    }

    pub fn flush(&mut self) -> String {
        let buf = std::mem::take(&mut self.buf);
        rules::restore_text(&buf, &self.mapping)
    }
}

// ---------------------------------------------------------------------------
// Redactor
// ---------------------------------------------------------------------------

/// Stateful redactor: this instance's options (operator, salt,
/// allow/deny lists, recognizers, thresholds, review margin, global
/// threshold) applied on every call — the Rust counterpart of the
/// Python SDK's `Redactor`. Without the `neural` feature (or without
/// a model) it is rules + recognizers; with a model loaded, neural
/// spans join the merge.
pub struct Redactor {
    options: RedactOptions,
    threshold: f64,
    #[cfg(feature = "neural")]
    neural: Option<crate::neural::OnnxEngine>,
}

impl Default for Redactor {
    fn default() -> Self {
        Redactor::new()
    }
}

impl Redactor {
    /// Rules-only redactor with default options.
    pub fn new() -> Redactor {
        Redactor {
            options: RedactOptions::default(),
            threshold: DEFAULT_THRESHOLD,
            #[cfg(feature = "neural")]
            neural: None,
        }
    }

    /// Redactor with the given options. Validates score options and
    /// recognizers eagerly, like the Python constructor.
    pub fn with_options(options: RedactOptions) -> Result<Redactor> {
        validate_score_options(options.thresholds.as_ref(), options.review_margin)?;
        if let Some(recs) = &options.recognizers {
            for rec in recs {
                validate_recognizer(rec)?;
            }
        }
        Ok(Redactor {
            options,
            threshold: DEFAULT_THRESHOLD,
            #[cfg(feature = "neural")]
            neural: None,
        })
    }

    /// Load the ONNX neural layer (feature `neural`).
    #[cfg(feature = "neural")]
    pub fn with_model(
        mut self,
        model: &str,
        tokenizer: &str,
        labels: Option<&str>,
        max_len: usize,
    ) -> Result<Redactor> {
        let engine =
            crate::neural::OnnxEngine::new(model, tokenizer, labels, max_len, self.threshold)?;
        self.neural = Some(engine);
        Ok(self)
    }

    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    pub fn set_threshold(&mut self, threshold: f64) {
        self.threshold = threshold;
        #[cfg(feature = "neural")]
        if let Some(n) = &mut self.neural {
            n.threshold = threshold;
        }
    }

    pub fn engine_name(&self) -> &'static str {
        #[cfg(feature = "neural")]
        if self.neural.is_some() {
            return "rules+onnx";
        }
        "rules"
    }

    /// Neural spans for `text` from the loaded engine, asked down to
    /// the score floor (empty without a model).
    fn neural_spans(&self, _text: &str) -> Result<Vec<Entity>> {
        #[cfg(feature = "neural")]
        if let Some(engine) = &self.neural {
            let floor = score_floor(
                self.threshold,
                self.options.thresholds.as_ref(),
                self.options.review_margin,
            );
            return engine.predict(_text, Some(floor));
        }
        Ok(Vec::new())
    }

    fn model_path(&self) -> Option<String> {
        #[cfg(feature = "neural")]
        if let Some(engine) = &self.neural {
            return Some(engine.model_path.clone());
        }
        None
    }

    /// Entities that would be redacted, with this Redactor's lists,
    /// recognizers, and score gate applied. Review candidates are a
    /// redact()-time concept and are not returned here.
    pub fn scan(&self, text: &str) -> Result<Vec<Entity>> {
        let spans = self.neural_spans(text)?;
        let (kept, _review) =
            gather_entities(text, &spans, &self.options, self.threshold)?;
        Ok(kept)
    }

    /// Redact text with this Redactor's options.
    pub fn redact(&self, text: &str) -> Result<RedactionResult> {
        let spans = self.neural_spans(text)?;
        let (kept, review) =
            gather_entities(text, &spans, &self.options, self.threshold)?;
        Ok(finish(
            text,
            kept,
            review,
            &self.options,
            self.engine_name(),
            self.model_path(),
        ))
    }

    /// Redact text with a per-call global threshold override. Like
    /// the Python Redactor, the override is sticky: it becomes this
    /// Redactor's threshold.
    pub fn redact_with_threshold(
        &mut self,
        text: &str,
        threshold: f64,
    ) -> Result<RedactionResult> {
        self.set_threshold(threshold);
        self.redact(text)
    }

    /// Redact many texts with this one instance, in input order.
    pub fn redact_batch(&self, texts: &[String]) -> Result<Vec<RedactionResult>> {
        texts.iter().map(|t| self.redact(t)).collect()
    }

    /// Scrub text for a trip outside the trust boundary: returns
    /// (redacted_text, mapping). See the free `scrub` function.
    pub fn scrub(&self, text: &str) -> Result<(String, IndexMap<String, String>)> {
        let r = self.redact(text)?;
        Ok((r.redacted, r.mapping))
    }

    /// Put real values back using a mapping from `redact`.
    pub fn restore(redacted: &str, mapping: &IndexMap<String, String>) -> String {
        rules::restore_text(redacted, mapping)
    }
}
