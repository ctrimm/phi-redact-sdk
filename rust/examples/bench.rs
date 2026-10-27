//! Rust bench harness — the counterpart of bench/bench_python.py and
//! bench/bench_ts.mjs. Prints one JSON line per run.
//!
//! ```text
//! cargo build --release --example bench
//! ./target/release/examples/bench rules
//! ./target/release/examples/bench features
//! ./target/release/examples/bench batch
//! ./target/release/examples/bench neural MODEL TOKENIZER LABELS   (feature `neural`)
//! ```
//!
//! Run from the rust/ directory. Same method as the other harnesses:
//! per-note timing over the 15 parity fixture notes (2 warm-up
//! passes, 7 measured; pooled latencies, median + nearest-rank p95);
//! batch over the 1,000-doc fixture-derived corpus (1 warm-up, 5
//! reps).

use phi_redact::{AnonymizeOperator, Recognizer, RedactOptions, Redactor};
use std::time::Instant;

fn load_notes(path: &str) -> Vec<String> {
    let raw = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    raw.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let v: serde_json::Value = serde_json::from_str(l).unwrap();
            v["note"].as_str().unwrap().to_string()
        })
        .collect()
}

fn median(xs: &[f64]) -> f64 {
    let mut s = xs.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[s.len() / 2]
}

fn p95(xs: &[f64]) -> f64 {
    let mut s = xs.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[((0.95 * s.len() as f64).ceil() as usize).max(1) - 1]
}

fn round4(x: f64) -> f64 {
    (x * 10000.0).round() / 10000.0
}

fn per_note<F: Fn(&str)>(f: F, notes: &[String]) -> String {
    for _ in 0..2 {
        for n in notes {
            f(n);
        }
    }
    let mut samples: Vec<f64> = Vec::new();
    for _ in 0..7 {
        for n in notes {
            let t0 = Instant::now();
            f(n);
            samples.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
    }
    format!(
        "\"median_ms\": {}, \"p95_ms\": {}, \"samples\": {}, \"notes\": {}",
        round4(median(&samples)),
        round4(p95(&samples)),
        samples.len(),
        notes.len()
    )
}

fn feature_options() -> RedactOptions {
    let mut thresholds = indexmap::IndexMap::new();
    thresholds.insert("REF_CODE".to_string(), 0.9);
    RedactOptions {
        operator: Some(AnonymizeOperator::Hash),
        salt: Some("bench".into()),
        denylist: Some(vec!["Maria".into()]),
        recognizers: Some(vec![
            Recognizer::new("EMPLOYEE_ID", vec![r"EMP\d{6}".to_string()]).with_score(0.9),
            Recognizer::new("REF_CODE", vec![r"[A-Z]{2}-\d{4}".to_string()])
                .with_score(0.8),
        ]),
        thresholds: Some(thresholds),
        review_margin: Some(0.2),
        ..Default::default()
    }
}

fn main() {
    let mode = std::env::args().nth(1).unwrap_or_default();
    let notes = load_notes("../e2e/parity/notes.jsonl");
    match mode.as_str() {
        "rules" => {
            let r = Redactor::new();
            let body = per_note(|n| {
                let _ = r.redact(n).unwrap();
            }, &notes);
            println!("{{\"language\": \"rust\", \"mode\": \"rules\", {body}}}");
        }
        "features" => {
            let r = Redactor::with_options(feature_options()).unwrap();
            let body = per_note(|n| {
                let _ = r.redact(n).unwrap();
            }, &notes);
            println!("{{\"language\": \"rust\", \"mode\": \"features\", {body}}}");
        }
        "batch" => {
            let corpus = load_notes("../bench/corpus.jsonl");
            let r = Redactor::new();
            let _ = r.redact_batch(&corpus).unwrap();
            let mut totals: Vec<f64> = Vec::new();
            for _ in 0..5 {
                let t0 = Instant::now();
                let _ = r.redact_batch(&corpus).unwrap();
                totals.push(t0.elapsed().as_secs_f64());
            }
            let med = median(&totals);
            println!(
                "{{\"language\": \"rust\", \"mode\": \"batch\", \"docs\": {}, \
                 \"median_total_s\": {}, \"notes_per_sec\": {}, \
                 \"per_note_ms\": {}, \"reps\": {}}}",
                corpus.len(),
                round4(med),
                (corpus.len() as f64 / med * 10.0).round() / 10.0,
                round4(med / corpus.len() as f64 * 1000.0),
                totals.len()
            );
        }
        "neural" => neural_mode(&notes),
        other => {
            eprintln!("usage: bench rules|features|batch|neural (got {other:?})");
            std::process::exit(2);
        }
    }
}

#[cfg(feature = "neural")]
fn neural_mode(notes: &[String]) {
    let model = std::env::args().nth(2).expect("model path");
    let tok = std::env::args().nth(3).expect("tokenizer path");
    let labels = std::env::args().nth(4).expect("labels path");
    let t0 = Instant::now();
    let r = Redactor::new()
        .with_model(&model, &tok, Some(&labels), phi_redact::DEFAULT_MAX_LEN)
        .unwrap();
    let load_s = t0.elapsed().as_secs_f64();
    let body = per_note(|n| {
        let _ = r.redact(n).unwrap();
    }, notes);
    println!(
        "{{\"language\": \"rust\", \"mode\": \"neural\", {body}, \
         \"model_load_s\": {}}}",
        round4(load_s)
    );
}

#[cfg(not(feature = "neural"))]
fn neural_mode(_notes: &[String]) {
    eprintln!("neural mode needs a build with --features neural");
    std::process::exit(2);
}
