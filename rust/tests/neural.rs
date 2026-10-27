//! Neural-engine equivalence with the Python SDK (feature `neural`).
//!
//! Pins the Python `OnnxEngine.predict(threshold=0.0)` spans for three
//! fixture notes against the real v9 int8 model — same spans, same
//! texts, same rounded scores. Skips when the model assets are not
//! present (set PHI_REDACT_V9_DIR to the directory containing
//! phi-redact-v9-int8.onnx and mac-model-dir/).
#![cfg(feature = "neural")]

use phi_redact::neural::OnnxEngine;
use std::path::PathBuf;

fn engine() -> Option<OnnxEngine> {
    let dir = std::env::var("PHI_REDACT_V9_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("../../phi-redact/work/v9"));
    let model = dir.join("phi-redact-v9-int8.onnx");
    let tok = dir.join("mac-model-dir/tokenizer.json");
    let tags = dir.join("mac-model-dir/tags.json");
    if !model.exists() || !tok.exists() || !tags.exists() {
        eprintln!("v9 assets not found under {}; skipping", dir.display());
        return None;
    }
    Some(
        OnnxEngine::new(
            model.to_str().unwrap(),
            tok.to_str().unwrap(),
            tags.to_str(),
            phi_redact::DEFAULT_MAX_LEN,
            phi_redact::DEFAULT_THRESHOLD,
        )
        .expect("engine loads"),
    )
}

fn spans(e: &OnnxEngine, note: &str) -> Vec<(usize, usize, String, String, f64)> {
    e.predict(note, Some(0.0))
        .unwrap()
        .iter()
        .map(|s| (s.start, s.end, s.label.clone(), s.text.clone(), s.score.unwrap()))
        .collect()
}

#[test]
fn neural_spans_match_python() {
    let Some(e) = engine() else { return };
    // Expected values: Python OnnxEngine.predict(note, threshold=0.0)
    // on the v9 int8 model, recorded 2026-10-27.
    let cases: Vec<(&str, Vec<(usize, usize, &str, &str, f64)>)> = vec![
        (
            "Call Jane Doe at 415-555-0132 about case 9918.",
            vec![
                (5, 9, "NAME_PATIENT", "Jane", 0.9994),
                (10, 13, "NAME_PATIENT", "Doe", 0.971),
                (17, 29, "PHONE", "415-555-0132", 0.9997),
            ],
        ),
        (
            "Pt Maria Santos DOB 03/14/1978 SSN 234-56-7890, ph 415-555-0132, \
             jane@example.com, card 4111111111111111, MRN 48291375, ZIP 90210.",
            vec![
                (0, 15, "NAME_PATIENT", "Pt Maria Santos", 0.9646),
                (20, 30, "DATE", "03/14/1978", 0.9993),
                (35, 46, "SSN", "234-56-7890", 1.0),
                (54, 63, "PHONE", "-555-0132", 0.8314),
                (65, 81, "EMAIL", "jane@example.com", 0.9999),
                (88, 104, "CREDITCARD", "4111111111111111", 1.0),
                (124, 129, "ADDRESS", "90210", 0.9708),
            ],
        ),
        (
            "Call 415-555-0132 then 212-555-0199 about SSN 234-56-7890.",
            vec![
                (5, 17, "PHONE", "415-555-0132", 0.9998),
                (23, 35, "PHONE", "212-555-0199", 0.9852),
                (46, 57, "SSN", "234-56-7890", 1.0),
            ],
        ),
    ];
    for (note, expected) in cases {
        let got = spans(&e, note);
        assert_eq!(got.len(), expected.len(), "span count for {note:?}: {got:?}");
        for (g, x) in got.iter().zip(expected.iter()) {
            assert_eq!(
                (g.0, g.1, g.2.as_str(), g.3.as_str()),
                (x.0, x.1, x.2, x.3),
                "span mismatch for {note:?}"
            );
            assert!(
                (g.4 - x.4).abs() <= 0.0002,
                "score mismatch for {note:?}: got {}, python {}",
                g.4,
                x.4
            );
        }
    }
}

#[test]
fn neural_redactor_end_to_end() {
    let Some(e) = engine() else { return };
    let dir = std::env::var("PHI_REDACT_V9_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("../../phi-redact/work/v9"));
    let r = phi_redact::Redactor::new()
        .with_model(
            dir.join("phi-redact-v9-int8.onnx").to_str().unwrap(),
            dir.join("mac-model-dir/tokenizer.json").to_str().unwrap(),
            dir.join("mac-model-dir/tags.json").to_str(),
            phi_redact::DEFAULT_MAX_LEN,
        )
        .unwrap();
    assert_eq!(r.engine_name(), "rules+onnx");
    let _ = e;
    let out = r
        .redact("Pt Maria Santos DOB 03/14/1978 SSN 234-56-7890.")
        .unwrap();
    // Neural NAME_PATIENT span and rule spans merged; rules win ties.
    assert!(out.redacted.contains("[NAME_PATIENT_1]"), "{}", out.redacted);
    assert!(out.redacted.contains("[SSN_1]"), "{}", out.redacted);
    assert_eq!(
        phi_redact::restore(&out.redacted, &out.mapping),
        "Pt Maria Santos DOB 03/14/1978 SSN 234-56-7890."
    );
}
