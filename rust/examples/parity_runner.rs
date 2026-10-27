//! E2E parity runner — the Rust counterpart of
//! e2e/parity/run_ts.mjs (mode "basic") and
//! e2e/parity/run_ts_features.mjs (mode "features"), printing
//! byte-comparable JSON in exactly their shapes.
//!
//! ```text
//! cargo build --release --example parity_runner
//! ./target/release/examples/parity_runner basic    ../e2e/parity/notes.jsonl
//! ./target/release/examples/parity_runner features ../e2e/parity/notes.jsonl
//! ```
//!
//! The config list in "features" mode must stay in lockstep with
//! test_parity.py's FEATURE_CONFIGS — same names, same semantics.
//! "batch_hash" and "batch_recognizer_review" exercise
//! redact_batch(); every other config calls redact() per note.

use phi_redact::{
    redact, redact_batch, AnonymizeOperator, Entity, Recognizer, RedactOptions,
    RedactionResult, ValidatorName,
};
use serde_json::{json, Map, Value};

fn notes(path: &str) -> Vec<String> {
    let raw = std::fs::read_to_string(path).expect("notes.jsonl");
    raw.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let v: Value = serde_json::from_str(l).unwrap();
            v["note"].as_str().unwrap().to_string()
        })
        .collect()
}

fn mapping_json(m: &indexmap::IndexMap<String, String>) -> Value {
    let mut obj = Map::new();
    for (k, v) in m {
        obj.insert(k.clone(), Value::String(v.clone()));
    }
    Value::Object(obj)
}

fn counts_json(c: &indexmap::IndexMap<String, usize>) -> Value {
    let mut obj = Map::new();
    for (k, v) in c {
        obj.insert(k.clone(), Value::from(*v));
    }
    Value::Object(obj)
}

fn basic(path: &str) {
    let empty: Vec<Entity> = Vec::new();
    let cfg = RedactOptions::default();
    let out: Vec<Value> = notes(path)
        .iter()
        .map(|note| {
            let r = redact(note, &empty, &cfg).unwrap();
            json!({
                "redacted": r.redacted,
                "entities": r.entities.iter().map(|e| json!({
                    "start": e.start, "end": e.end, "label": e.label,
                })).collect::<Vec<_>>(),
                "mapping": mapping_json(&r.mapping),
            })
        })
        .collect();
    println!("{}", Value::Array(out));
}

fn thresholds(pairs: &[(&str, f64)]) -> indexmap::IndexMap<String, f64> {
    pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
}

fn feature_configs() -> Vec<(&'static str, RedactOptions)> {
    let base = RedactOptions::default();
    let refc = |score: f64| {
        Recognizer::new("REF_CODE", vec![r"[A-Z]{2}-\d{4}".to_string()]).with_score(score)
    };
    vec![
        (
            "label",
            RedactOptions { operator: Some(AnonymizeOperator::Label), ..base.clone() },
        ),
        (
            "mask",
            RedactOptions { operator: Some(AnonymizeOperator::Mask), ..base.clone() },
        ),
        (
            "hash_salted",
            RedactOptions {
                operator: Some(AnonymizeOperator::Hash),
                salt: Some("parity-salt".into()),
                ..base.clone()
            },
        ),
        (
            "hash_unsalted",
            RedactOptions { operator: Some(AnonymizeOperator::Hash), ..base.clone() },
        ),
        (
            "remove",
            RedactOptions { operator: Some(AnonymizeOperator::Remove), ..base.clone() },
        ),
        (
            "allowlist",
            RedactOptions {
                allowlist: Some(vec!["415-555-0132".into(), "jane@example.com".into()]),
                ..base.clone()
            },
        ),
        (
            "denylist",
            RedactOptions {
                denylist: Some(vec!["Maria".into(), "Jane".into()]),
                ..base.clone()
            },
        ),
        (
            "denylist_label",
            RedactOptions {
                denylist: Some(vec!["Santos".into()]),
                deny_label: Some("NAME_PATIENT".into()),
                ..base.clone()
            },
        ),
        (
            "allow_and_deny",
            RedactOptions {
                allowlist: Some(vec!["415-555-0132".into()]),
                denylist: Some(vec!["Maria".into()]),
                ..base.clone()
            },
        ),
        (
            "recognizer_basic",
            RedactOptions {
                recognizers: Some(vec![
                    Recognizer::new("EMPLOYEE_ID", vec![r"EMP\d{6}".to_string()])
                        .with_score(0.9),
                ]),
                ..base.clone()
            },
        ),
        (
            "recognizer_context",
            RedactOptions {
                recognizers: Some(vec![
                    Recognizer::new("REF_CODE", vec![r"[A-Z]{2}-\d{4}".to_string()])
                        .with_score(0.5)
                        .with_context(vec!["ref".into()])
                        .with_context_boost(0.45),
                ]),
                ..base.clone()
            },
        ),
        (
            "recognizer_require_context",
            RedactOptions {
                recognizers: Some(vec![
                    Recognizer::new("REF_CODE", vec![r"[A-Z]{2}-\d{4}".to_string()])
                        .with_score(0.9)
                        .with_context(vec!["ref".into()])
                        .with_require_context(true),
                ]),
                ..base.clone()
            },
        ),
        (
            "recognizer_validator",
            RedactOptions {
                recognizers: Some(vec![
                    Recognizer::new("MEMBER_ID", vec![r"\b\d{9}\b".to_string()])
                        .with_score(0.9)
                        .with_named_validator(ValidatorName::Aba),
                ]),
                ..base.clone()
            },
        ),
        (
            "thresholds_lower",
            RedactOptions {
                recognizers: Some(vec![refc(0.5)]),
                thresholds: Some(thresholds(&[("REF_CODE", 0.4)])),
                ..base.clone()
            },
        ),
        (
            "thresholds_raise_review",
            RedactOptions {
                recognizers: Some(vec![refc(0.8)]),
                thresholds: Some(thresholds(&[("REF_CODE", 0.9)])),
                review_margin: Some(0.2),
                ..base.clone()
            },
        ),
        (
            "review_margin_global",
            RedactOptions {
                recognizers: Some(vec![refc(0.55)]),
                review_margin: Some(0.1),
                ..base.clone()
            },
        ),
    ]
}

fn shape(r: &RedactionResult) -> Value {
    json!({
        "redacted": r.redacted,
        "entities": r.entities.iter().map(|e| json!({
            "start": e.start, "end": e.end, "label": e.label,
            "source": e.source.as_str(),
            "placeholder": e.placeholder, "score": e.score,
        })).collect::<Vec<_>>(),
        "mapping": mapping_json(&r.mapping),
        "counts": counts_json(&r.counts),
        "review": r.review.iter().map(|c| json!({
            "start": c.start, "end": c.end, "label": c.label,
            "text": c.text, "score": c.score,
            "source": c.source.as_str(),
        })).collect::<Vec<_>>(),
    })
}

fn features(path: &str) {
    let notes = notes(path);
    let empty: Vec<Entity> = Vec::new();
    let mut out: Vec<Value> = feature_configs()
        .iter()
        .map(|(name, cfg)| {
            json!({
                "name": name,
                "results": notes.iter()
                    .map(|n| shape(&redact(n, &empty, cfg).unwrap()))
                    .collect::<Vec<_>>(),
            })
        })
        .collect();
    let hash_cfg = RedactOptions {
        operator: Some(AnonymizeOperator::Hash),
        salt: Some("parity-salt".into()),
        ..Default::default()
    };
    out.push(json!({
        "name": "batch_hash",
        "results": redact_batch(&notes, &hash_cfg).unwrap()
            .iter().map(shape).collect::<Vec<_>>(),
    }));
    let review_cfg = RedactOptions {
        recognizers: Some(vec![
            Recognizer::new("REF_CODE", vec![r"[A-Z]{2}-\d{4}".to_string()])
                .with_score(0.8),
        ]),
        thresholds: Some(thresholds(&[("REF_CODE", 0.9)])),
        review_margin: Some(0.2),
        ..Default::default()
    };
    out.push(json!({
        "name": "batch_recognizer_review",
        "results": phi_redact::Redactor::with_options(review_cfg).unwrap()
            .redact_batch(&notes).unwrap()
            .iter().map(shape).collect::<Vec<_>>(),
    }));
    println!("{}", Value::Array(out));
}

fn main() {
    let mode = std::env::args().nth(1);
    let path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "../e2e/parity/notes.jsonl".to_string());
    match mode.as_deref() {
        Some("basic") => basic(&path),
        Some("features") => features(&path),
        other => {
            eprintln!("usage: parity_runner basic|features [notes.jsonl] (got {other:?})");
            std::process::exit(2);
        }
    }
}
