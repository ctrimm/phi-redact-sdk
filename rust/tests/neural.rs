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


/// Regression: chunk-boundary span scoring must match Python.
///
/// Python's `OnnxEngine.predict` shifts *every* tokenizer offset by
/// the chunk offset — including the (0,0) offsets of the [CLS]/[SEP]
/// specials, which become (coff, coff) — and its decode skips only
/// offsets exactly equal to (0,0). For chunks after the first, the
/// [CLS]-position prediction therefore participates in span building
/// and score averaging. The Rust port originally stored (0,0) for
/// those positions unshifted, so its decode always skipped them.
///
/// Doc 59 of bench/lengths/tier_2000.jsonl is the bisected case:
/// chunk 2 starts at char 886, its [CLS] position predicts
/// B-ADDRESS at 0.7096, and the "Apt. 36" span that opens there
/// scores 0.9202 in Python (mean over the boundary token + three
/// content tokens) vs 0.9903 in the old Rust decode (content tokens
/// only). The score gap flipped the keep-highest-score merge
/// downstream: Python redacted the full street address, Rust left
/// it in the clear. Expected values below are Python
/// `OnnxEngine.predict(note, threshold=0.0)` output, recorded
/// 2026-10-30.
#[test]
fn chunk_boundary_span_scores_match_python() {
    let Some(e) = engine() else { return };
    let note: String = concat!(
        "Badge EMP160053 documented the home visit for case 24881302 on 04/12/2026. Emily Johnson ",
        "prefers portal messages; alternate contact Ahmed Johnson at 312-555-0135. Encouraged daily ",
        "walking as tolerated and a follow-up call in one week. Hearing is grossly intact to ",
        "conversational voice in a quiet room. Reviewed red-flag symptoms that should prompt an ",
        "urgent call or visit. Badge EMP209478 documented the home visit for case 58479808 on ",
        "05/22/2025. Billing flagged account 38011869: card 4111111111111111 declined, patient ",
        "updated payment by phone. Call Jane Doe at 415-555-0132 about case 9918. Voicemail left ",
        "for Ahmed Hall at 415-555-0153; will retry on 09/02/2023. Badge EMP411733 documented the ",
        "home visit for case 41876952 on 12/12/2026. Email from David Harris ",
        "(david.harris@example.com) asks whether the 12/03/2026 slot can move to the afternoon. ",
        "Address on file: 77 Harbor Blvd, Apt. 36, San Diego, CA 92101. Phone 312-555-0121. Address ",
        "on file: 77 Harbor Blvd, Apt. 34, San Diego, CA 92101. Phone 512-555-0167. Worker ",
        "EMP390551 left a door hanger at 48 Elm Avenue on 12/25/2024 after no answer. Mental ",
        "status: alert and oriented times three, affect appropriate, speech fluent. Discussed ",
        "advance directive paperwork; the patient will discuss with family first. Mental status: ",
        "alert and oriented times three, affect appropriate, speech fluent.\nThe patient reports ",
        "sleeping six to seven hours nightly with one awakening.\nNo new medications were started ",
        "today; the current list was reconciled with the patient. Please contact Susan Lewis at ",
        "susan.lewis@example.org or 303-555-0111 to reschedule the appointment. The rash has faded ",
        "considerably since the steroid cream was started last week. The rash has faded ",
        "considerably since the steroid cream was started last week.\nRemittance for invoice ",
        "84802073 charged to card 4111111111111111; receipt emailed to brian.miller@example.com. ",
        "Mental status: alert and oriented times three, affect appropriate, speech fluent."
    )
    .to_string();
    assert_eq!(note.len(), 1967, "embedded doc 59 drifted");
    let got = spans(&e, &note);
    let apt: Vec<_> = got
        .iter()
        .filter(|s| s.0 == 887 && s.1 == 894 && s.2 == "ADDRESS")
        .collect();
    assert_eq!(apt.len(), 1, "Apt. 36 span missing: {got:?}");
    assert_eq!(apt[0].3, "Apt. 36");
    assert!(
        (apt[0].4 - 0.9202).abs() <= 0.0002,
        "chunk-boundary span score: got {}, python 0.9202          (0.9903 = boundary token dropped from the mean)",
        apt[0].4
    );
    // The full street-address span from the overlap region is also
    // pinned: it is the span the merge must prefer downstream.
    let full: Vec<_> = got
        .iter()
        .filter(|s| s.0 == 871 && s.1 == 894 && s.2 == "ADDRESS")
        .collect();
    assert_eq!(full.len(), 1, "full address span missing: {got:?}");
    assert_eq!(full[0].3, "77 Harbor Blvd, Apt. 36");
    assert!((full[0].4 - 0.9788).abs() <= 0.0002);
}
