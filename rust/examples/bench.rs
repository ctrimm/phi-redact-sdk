//! Rust bench harness — the counterpart of bench/bench_python.py and
//! bench/bench_ts.mjs. Prints one JSON line per invocation.
//!
//! ```text
//! cargo build --release --features neural --example bench
//! ./target/release/examples/bench scan|rules|features|batch|batch-features
//! ./target/release/examples/bench neural MODEL TOKENIZER LABELS     (feature `neural`)
//! ./target/release/examples/bench neural-stages MODEL TOKENIZER LABELS
//! ```
//!
//! Run from the rust/ directory. Driven by bench/run_bench.py, which
//! interleaves languages round-by-round; this harness just executes
//! BENCH_REPS measured passes (after BENCH_WARMUPS discarded passes)
//! in-process and reports raw samples plus SHA-256 digests of its
//! inputs and outputs so the driver can verify that all three SDKs
//! read byte-identical corpora and produce identical outputs under
//! identical configs. Digest payload format is shared with the
//! Python and TypeScript harnesses (see bench/run_bench.py).

use phi_redact::{scan, AnonymizeOperator, Recognizer, RedactOptions, Redactor};
use sha2::{Digest, Sha256};
use std::time::Instant;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

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

fn score_str(score: Option<f64>, with_score: bool) -> String {
    if !with_score {
        return String::new();
    }
    match score {
        Some(s) => format!("{s:.4}"),
        None => String::new(),
    }
}

/// Canonical output payload shared by all three bench harnesses:
/// redacted text, one line per entity, one line per review candidate.
fn result_payload(r: &phi_redact::RedactionResult, with_score: bool) -> String {
    let mut out = String::new();
    out.push_str(&r.redacted);
    for e in &r.entities {
        out.push_str(&format!(
            "\n{},{},{},{},{},{}",
            e.start,
            e.end,
            e.label,
            score_str(e.score, with_score),
            e.source.as_str(),
            e.placeholder.as_deref().unwrap_or("")
        ));
    }
    for c in &r.review {
        out.push_str(&format!(
            "\nR{},{},{},{},{}",
            c.start,
            c.end,
            c.label,
            score_str(Some(c.score), with_score),
            c.source.as_str()
        ));
    }
    out
}

fn spans_payload(spans: &[phi_redact::Entity], with_score: bool) -> String {
    let mut out = String::new();
    for (i, e) in spans.iter().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        out.push_str(&format!(
            "{},{},{},{},{},{}",
            e.start,
            e.end,
            e.label,
            score_str(e.score, with_score),
            e.source.as_str(),
            e.placeholder.as_deref().unwrap_or("")
        ));
    }
    out
}

fn digest_payloads(payloads: &[String]) -> String {
    let mut h = Sha256::new();
    for p in payloads {
        h.update(p.as_bytes());
        h.update(b"\x00");
    }
    format!("{:x}", h.finalize())
}

fn input_digest(notes: &[String]) -> String {
    let mut h = Sha256::new();
    for n in notes {
        h.update(n.as_bytes());
        h.update(b"\n");
    }
    format!("{:x}", h.finalize())
}

fn json_samples(xs: &[f64]) -> String {
    let parts: Vec<String> = xs.iter().map(|x| format!("{x:.4}")).collect();
    format!("[{}]", parts.join(","))
}

fn round4(x: f64) -> f64 {
    (x * 10000.0).round() / 10000.0
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

/// Time `f` per note (warmups discarded) and return raw samples in
/// milliseconds. Output digests are computed by the callers from a
/// separate untimed pass.
fn per_note<F: Fn(&str)>(f: F, notes: &[String]) -> Vec<f64> {
    let warmups = env_usize("BENCH_WARMUPS", 2);
    let reps = env_usize("BENCH_REPS", 7);
    for _ in 0..warmups {
        for n in notes {
            f(n);
        }
    }
    let mut samples: Vec<f64> = Vec::new();
    for _ in 0..reps {
        for n in notes {
            let t0 = Instant::now();
            f(n);
            samples.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
    }
    samples
}

fn main() {
    let mode = std::env::args().nth(1).unwrap_or_default();
    let notes = load_notes("../e2e/parity/notes.jsonl");
    let idig = input_digest(&notes);
    match mode.as_str() {
        "scan" => {
            let f = |n: &str| spans_payload(&scan(n), true);
            // Digest pass first (untimed full pass -> steady state).
            let payloads: Vec<String> = notes.iter().map(|n| f(n)).collect();
            let samples = per_note(|n| { f(n); }, &notes);
            println!(
                "{{\"language\": \"rust\", \"mode\": \"scan\", \"samples_ms\": {}, \
                 \"digest\": \"{}\", \"digest_noscore\": \"{}\", \
                 \"input_digest\": \"{}\", \"notes\": {}}}",
                json_samples(&samples),
                digest_payloads(&payloads),
                digest_payloads(&payloads),
                idig,
                notes.len()
            );
        }
        "rules" | "features" => {
            let r = if mode == "rules" {
                Redactor::new()
            } else {
                Redactor::with_options(feature_options()).unwrap()
            };
            // Digest passes first (untimed full passes -> steady state).
            let scored: Vec<String> = notes
                .iter()
                .map(|n| result_payload(&r.redact(n).unwrap(), true))
                .collect();
            let plain: Vec<String> = notes
                .iter()
                .map(|n| result_payload(&r.redact(n).unwrap(), false))
                .collect();
            let samples = per_note(
                |n| {
                    let _ = r.redact(n).unwrap();
                },
                &notes,
            );
            println!(
                "{{\"language\": \"rust\", \"mode\": \"{mode}\", \"samples_ms\": {}, \
                 \"digest\": \"{}\", \"digest_noscore\": \"{}\", \
                 \"input_digest\": \"{}\", \"notes\": {}}}",
                json_samples(&samples),
                digest_payloads(&scored),
                digest_payloads(&plain),
                idig,
                notes.len()
            );
        }
        "batch" | "batch-features" => {
            let corpus = load_notes("../bench/corpus.jsonl");
            let r = if mode == "batch" {
                Redactor::new()
            } else {
                Redactor::with_options(feature_options()).unwrap()
            };
            let warmups = env_usize("BENCH_WARMUPS", 1);
            let reps = env_usize("BENCH_REPS", 5);
            // Digest from a dedicated untimed pass (not tied to
            // warmups, so BENCH_WARMUPS=0 invocations still report it).
            let digest;
            let digest_ns;
            {
                let results = r.redact_batch(&corpus).unwrap();
                digest = digest_payloads(
                    &results.iter().map(|x| result_payload(x, true)).collect::<Vec<_>>(),
                );
                digest_ns = digest_payloads(
                    &results.iter().map(|x| result_payload(x, false)).collect::<Vec<_>>(),
                );
            }
            for _ in 0..warmups {
                let _ = r.redact_batch(&corpus).unwrap();
            }
            let mut totals: Vec<f64> = Vec::new();
            for _ in 0..reps {
                let t0 = Instant::now();
                let _ = r.redact_batch(&corpus).unwrap();
                totals.push(t0.elapsed().as_secs_f64());
            }
            let tot: Vec<String> = totals.iter().map(|x| format!("{x:.4}")).collect();
            println!(
                "{{\"language\": \"rust\", \"mode\": \"{mode}\", \"docs\": {}, \
                 \"totals_s\": [{}], \"digest\": \"{digest}\", \
                 \"digest_noscore\": \"{digest_ns}\", \"input_digest\": \"{}\"}}",
                corpus.len(),
                tot.join(","),
                input_digest(&corpus)
            );
        }
        "neural" => neural_mode(&notes, &idig, false),
        "neural-stages" => neural_mode(&notes, &idig, true),
        other => {
            eprintln!("usage: bench scan|rules|features|batch|batch-features|neural|neural-stages (got {other:?})");
            std::process::exit(2);
        }
    }
}

#[cfg(feature = "neural")]
fn neural_mode(notes: &[String], idig: &str, stages_mode: bool) {
    let model = std::env::args().nth(2).expect("model path");
    let tok = std::env::args().nth(3).expect("tokenizer path");
    let labels = std::env::args().nth(4).expect("labels path");
    if !stages_mode {
        let t0 = Instant::now();
        let r = Redactor::new()
            .with_model(&model, &tok, Some(&labels), phi_redact::DEFAULT_MAX_LEN)
            .unwrap();
        let load_s = t0.elapsed().as_secs_f64();
        // Digest passes first (untimed full passes -> steady state).
        let scored: Vec<String> = notes
            .iter()
            .map(|n| result_payload(&r.redact(n).unwrap(), true))
            .collect();
        let plain: Vec<String> = notes
            .iter()
            .map(|n| result_payload(&r.redact(n).unwrap(), false))
            .collect();
        let samples = per_note(
            |n| {
                let _ = r.redact(n).unwrap();
            },
            notes,
        );
        println!(
            "{{\"language\": \"rust\", \"mode\": \"neural\", \"samples_ms\": {}, \
             \"digest\": \"{}\", \"digest_noscore\": \"{}\", \
             \"input_digest\": \"{}\", \"notes\": {}, \"model_load_s\": {}}}",
            json_samples(&samples),
            digest_payloads(&scored),
            digest_payloads(&plain),
            idig,
            notes.len(),
            round4(load_s)
        );
        return;
    }
    // Stage decomposition via OnnxEngine::predict_timed.
    let t0 = Instant::now();
    let engine = phi_redact::neural::OnnxEngine::new(
        &model,
        &tok,
        Some(&labels),
        phi_redact::DEFAULT_MAX_LEN,
        phi_redact::DEFAULT_THRESHOLD,
    )
    .unwrap();
    let load_s = t0.elapsed().as_secs_f64();
    // Digest passes first (untimed full passes -> steady state).
    let scored: Vec<String> = notes
        .iter()
        .map(|n| spans_payload(&engine.predict(n, None).unwrap(), true))
        .collect();
    let plain: Vec<String> = notes
        .iter()
        .map(|n| spans_payload(&engine.predict(n, None).unwrap(), false))
        .collect();
    let warmups = env_usize("BENCH_WARMUPS", 2);
    let reps = env_usize("BENCH_REPS", 7);
    for _ in 0..warmups {
        for n in notes {
            let _ = engine.predict_timed(n, None).unwrap();
        }
    }
    let (mut tok_s, mut inf_s, mut post_s, mut tot_s) = (vec![], vec![], vec![], vec![]);
    for _ in 0..reps {
        for n in notes {
            let t0 = Instant::now();
            let (_spans, st) = engine.predict_timed(n, None).unwrap();
            tot_s.push(t0.elapsed().as_secs_f64() * 1000.0);
            tok_s.push(st.tokenize_ms);
            inf_s.push(st.inference_ms);
            post_s.push(st.post_ms);
        }
    }
    println!(
        "{{\"language\": \"rust\", \"mode\": \"neural-stages\", \
         \"stages_ms\": {{\"tokenize\": {}, \"inference\": {}, \"post\": {}}}, \
         \"samples_ms\": {}, \"digest\": \"{}\", \"digest_noscore\": \"{}\", \
         \"input_digest\": \"{}\", \"notes\": {}, \"model_load_s\": {}}}",
        json_samples(&tok_s),
        json_samples(&inf_s),
        json_samples(&post_s),
        json_samples(&tot_s),
        digest_payloads(&scored),
        digest_payloads(&plain),
        idig,
        notes.len(),
        round4(load_s)
    );
}

#[cfg(not(feature = "neural"))]
fn neural_mode(_notes: &[String], _idig: &str, _stages_mode: bool) {
    eprintln!("neural modes need a build with --features neural");
    std::process::exit(2);
}
