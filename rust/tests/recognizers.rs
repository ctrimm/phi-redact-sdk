//! Ports of python/tests/test_recognizers_thresholds.py (0.5.0):
//! custom recognizers and per-label thresholds with needs-review
//! mode. The Python suite's scripted-OnnxEngine cases are ported
//! through injected neural spans (the local pipeline's architecture,
//! same as the TypeScript SDK).

use phi_redact::{
    merge_entities, redact, recognizer_spans, validate_recognizer, Entity, EntitySource,
    Recognizer, RedactOptions, Redactor, Validator,
};
use std::rc::Rc;

fn badge() -> Recognizer {
    Recognizer::new("EMPLOYEE_ID", vec![r"EMP\d{6}".to_string()])
}

const BADGE_TEXT: &str = "Badge EMP123456 issued, call 415-555-0132.";

fn opts() -> RedactOptions {
    RedactOptions::default()
}

fn with_recognizers(recs: Vec<Recognizer>) -> RedactOptions {
    RedactOptions {
        recognizers: Some(recs),
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// recognizer basics
// ---------------------------------------------------------------------------

#[test]
fn custom_label_redacted_with_recognizer_source() {
    let r = redact(BADGE_TEXT, &[], &with_recognizers(vec![badge()])).unwrap();
    assert_eq!(r.redacted, "Badge [EMPLOYEE_ID_1] issued, call [PHONE_1].");
    let emp: Vec<&Entity> = r.entities.iter().filter(|e| e.label == "EMPLOYEE_ID").collect();
    assert_eq!(emp.len(), 1);
    assert_eq!(emp[0].source, EntitySource::Recognizer);
    assert_eq!(emp[0].score, Some(0.85)); // default score
    assert_eq!(emp[0].text, "EMP123456");
    assert_eq!(phi_redact::restore(&r.redacted, &r.mapping), BADGE_TEXT);
}

#[test]
fn multiple_patterns_one_recognizer() {
    let rec = Recognizer::new(
        "EMPLOYEE_ID",
        vec![r"EMP\d{6}".to_string(), r"STF-\d{4}".to_string()],
    );
    let r = redact("EMP123456 and STF-7788", &[], &with_recognizers(vec![rec])).unwrap();
    assert_eq!(r.redacted, "[EMPLOYEE_ID_1] and [EMPLOYEE_ID_2]");
}

#[test]
fn redactor_constructor_and_scan() {
    let r = Redactor::with_options(with_recognizers(vec![badge()])).unwrap();
    let scanned = r.scan(BADGE_TEXT).unwrap();
    let labels: Vec<&str> = scanned.iter().map(|e| e.label.as_str()).collect();
    assert_eq!(labels, ["EMPLOYEE_ID", "PHONE"]);
    assert_eq!(
        r.redact(BADGE_TEXT).unwrap().redacted,
        "Badge [EMPLOYEE_ID_1] issued, call [PHONE_1]."
    );
}

#[test]
fn no_recognizers_no_change() {
    let r = redact(BADGE_TEXT, &[], &opts()).unwrap();
    assert_eq!(r.redacted, "Badge EMP123456 issued, call [PHONE_1].");
}

#[test]
fn group_selects_value_span() {
    let rec = Recognizer::new("BADGE_ID", vec![r"(?:badge|Badge)\s+(\d{6})".to_string()])
        .with_group(1);
    let r = redact("badge 482913 verified", &[], &with_recognizers(vec![rec])).unwrap();
    assert_eq!(r.redacted, "badge [BADGE_ID_1] verified");
    assert_eq!(r.entities[0].text, "482913");
}

#[test]
fn first_registered_recognizer_wins_overlap() {
    let a = Recognizer::new("SIX_DIGITS", vec![r"\b\d{6}\b".to_string()]).with_score(0.9);
    let b = Recognizer::new("EXACT_CODE", vec!["482913".to_string()]).with_score(0.99);
    let r = redact("code 482913 here", &[], &with_recognizers(vec![a.clone(), b.clone()])).unwrap();
    assert_eq!(r.redacted, "code [SIX_DIGITS_1] here");
    let r = redact("code 482913 here", &[], &with_recognizers(vec![b, a])).unwrap();
    assert_eq!(r.redacted, "code [EXACT_CODE_1] here");
}

#[test]
fn empty_matches_are_skipped() {
    let rec = Recognizer::new("DIGITS", vec![r"\d*".to_string()]).with_score(0.9);
    let r = redact("abc 123 def", &[], &with_recognizers(vec![rec])).unwrap();
    assert_eq!(r.redacted, "abc [DIGITS_1] def");
}

// ---------------------------------------------------------------------------
// recognizers x operators
// ---------------------------------------------------------------------------

#[test]
fn recognizer_label_operator() {
    let mut o = with_recognizers(vec![badge()]);
    o.operator = Some(phi_redact::AnonymizeOperator::Label);
    let r = redact(BADGE_TEXT, &[], &o).unwrap();
    assert_eq!(r.redacted, "Badge <EMPLOYEE_ID> issued, call <PHONE>.");
    assert!(r.mapping.is_empty());
}

#[test]
fn recognizer_hash_operator_consistent() {
    let mut o = with_recognizers(vec![badge()]);
    o.operator = Some(phi_redact::AnonymizeOperator::Hash);
    o.salt = Some("s".into());
    let a = redact("EMP123456", &[], &o).unwrap();
    let b = redact("again EMP123456", &[], &o).unwrap();
    assert_eq!(a.redacted, b.redacted.split(' ').last().unwrap());
    assert!(a.redacted.starts_with("[EMPLOYEE_ID_"));
}

#[test]
fn recognizer_remove_operator() {
    let mut o = with_recognizers(vec![badge()]);
    o.operator = Some(phi_redact::AnonymizeOperator::Remove);
    let r = redact("EMP123456 gone", &[], &o).unwrap();
    assert_eq!(r.redacted, " gone");
}

// ---------------------------------------------------------------------------
// context
// ---------------------------------------------------------------------------

fn case_rec() -> Recognizer {
    Recognizer::new("CASE_ID", vec![r"\b\d{6}\b".to_string()])
        .with_score(0.5)
        .with_context(vec!["case".to_string()])
        .with_context_boost(0.4)
}

#[test]
fn context_boost_raises_score_over_threshold() {
    let r = redact("case 482913 open", &[], &with_recognizers(vec![case_rec()])).unwrap();
    assert_eq!(r.redacted, "case [CASE_ID_1] open");
    assert_eq!(r.entities[0].score, Some(0.9));
}

#[test]
fn no_context_stays_below_global_threshold() {
    let r = redact("number 482913 open", &[], &with_recognizers(vec![case_rec()])).unwrap();
    assert_eq!(r.redacted, "number 482913 open");
    assert!(r.entities.is_empty() && r.review.is_empty());
}

#[test]
fn context_outside_window_does_not_boost() {
    let rec = case_rec().with_context_window(5);
    let r = redact("case number is far away: 482913", &[], &with_recognizers(vec![rec])).unwrap();
    assert!(r.entities.is_empty());
}

#[test]
fn boost_caps_at_one() {
    let rec = Recognizer::new("CASE_ID", vec![r"\b\d{6}\b".to_string()])
        .with_score(0.95)
        .with_context(vec!["case".to_string()])
        .with_context_boost(0.4);
    let spans = recognizer_spans("case 482913", &[rec]).unwrap();
    assert_eq!(spans.len(), 1);
    assert_eq!(spans[0].score, Some(1.0));
}

#[test]
fn require_context_drops_uncontextualized_match() {
    let rec = Recognizer::new("CASE_ID", vec![r"\b\d{6}\b".to_string()])
        .with_score(0.99)
        .with_context(vec!["case".to_string()])
        .with_require_context(true);
    let r = redact("number 482913 open", &[], &with_recognizers(vec![rec.clone()])).unwrap();
    assert!(r.entities.is_empty());
    let r = redact("case 482913 open", &[], &with_recognizers(vec![rec])).unwrap();
    assert_eq!(r.redacted, "case [CASE_ID_1] open");
}

// ---------------------------------------------------------------------------
// validators
// ---------------------------------------------------------------------------

#[test]
fn callable_validator() {
    let rec = Recognizer::new("ACCT_CODE", vec![r"ACCT-[A-Z]{2}\d{4}".to_string()])
        .with_validator(Validator::Func(Rc::new(|v: &str| v.ends_with("99"))));
    let r = redact("ACCT-XY1299 and ACCT-XY1234", &[], &with_recognizers(vec![rec])).unwrap();
    assert_eq!(r.redacted, "[ACCT_CODE_1] and ACCT-XY1234");
}

#[test]
fn named_luhn_validator() {
    let rec = Recognizer::new("CARD_LIKE", vec![r"\b\d{16}\b".to_string()])
        .with_named_validator(phi_redact::ValidatorName::Luhn);
    let spans = recognizer_spans("4111111111111111 vs 1234567812345678", &[rec]).unwrap();
    assert_eq!(
        spans.iter().map(|s| s.text.as_str()).collect::<Vec<_>>(),
        ["4111111111111111"]
    );
}

#[test]
fn named_iban_validator_accepts_spaced_iban() {
    let rec = Recognizer::new(
        "IBAN",
        vec![r"[A-Z]{2}\d{2}(?: \d{4}){4} \d{2}".to_string()],
    )
    .with_named_validator(phi_redact::ValidatorName::Iban);
    let good = redact(
        "IBAN DE89 3704 0044 0532 0130 00 end",
        &[],
        &with_recognizers(vec![rec.clone()]),
    )
    .unwrap();
    assert_eq!(good.redacted, "IBAN [IBAN_1] end");
    let bad = redact(
        "IBAN DE89 3704 0044 0532 0130 01 end",
        &[],
        &with_recognizers(vec![rec]),
    )
    .unwrap();
    assert!(bad.entities.iter().all(|e| e.label != "IBAN"));
}

#[test]
fn invalid_configs_raise() {
    assert!(validate_recognizer(&Recognizer::new("", vec!["x".into()])).is_err());
    assert!(validate_recognizer(&Recognizer::new("X", vec![])).is_err());
    assert!(validate_recognizer(
        &Recognizer::new("X", vec!["x".into()]).with_score(1.5)
    )
    .is_err());
    assert!(validate_recognizer(
        &Recognizer::new("X", vec!["x".into()]).with_require_context(true)
    )
    .is_err());
    // (Python also rejects validator="nope"; in Rust the validator is
    // an enum, so an unknown name is unrepresentable — ValidatorName::
    // parse returns None instead.)
    assert!(phi_redact::ValidatorName::parse("nope").is_none());
    assert!(redact("x", &[], &with_recognizers(vec![
        Recognizer::new("", vec!["x".into()])
    ]))
    .is_err());
}

// ---------------------------------------------------------------------------
// precedence
// ---------------------------------------------------------------------------

#[test]
fn recognizer_yields_to_overlapping_rule_span() {
    let rec = Recognizer::new("EMPLOYEE_ID", vec![r"EMP-\d{5}".to_string()]).with_score(0.95);
    let r = redact("Badge EMP-12345 issued.", &[], &with_recognizers(vec![rec])).unwrap();
    assert_eq!(r.redacted, "Badge EMP-[ZIP_1] issued.");
}

#[test]
fn merge_recognizer_beats_neural_rule_beats_recognizer() {
    let rule = Entity::new(0, 5, "ZIP", "90210", EntitySource::Rule);
    let mut rec = Entity::new(3, 9, "EMPLOYEE_ID", "123456", EntitySource::Recognizer);
    rec.score = Some(0.9);
    let mut neural = Entity::new(3, 9, "NAME", "123456", EntitySource::Neural);
    neural.score = Some(0.99);
    let merged = merge_entities(&[rule], &[neural.clone()], &[rec.clone()]);
    assert_eq!(
        merged.iter().map(|e| e.label.as_str()).collect::<Vec<_>>(),
        ["ZIP"]
    );
    let merged = merge_entities(&[], &[neural], &[rec]);
    assert_eq!(
        merged.iter().map(|e| e.label.as_str()).collect::<Vec<_>>(),
        ["EMPLOYEE_ID"]
    );
}

#[test]
fn allowlist_drops_recognizer_span() {
    let mut o = with_recognizers(vec![badge()]);
    o.allowlist = Some(vec!["EMP123456".into()]);
    let r = redact(BADGE_TEXT, &[], &o).unwrap();
    assert_eq!(r.redacted, "Badge EMP123456 issued, call [PHONE_1].");
}

#[test]
fn denylist_wins_overlap_with_recognizer() {
    let mut o = with_recognizers(vec![badge()]);
    o.denylist = Some(vec!["EMP123456".into()]);
    let r = redact("id EMP123456 ok", &[], &o).unwrap();
    assert_eq!(r.redacted, "id [CUSTOM_1] ok");
    assert_eq!(r.entities[0].source, EntitySource::Denylist);
}

// ---------------------------------------------------------------------------
// per-label thresholds + needs-review
// ---------------------------------------------------------------------------

fn ref_rec() -> Recognizer {
    Recognizer::new("REF_CODE", vec![r"[A-Z]{2}-\d{4}".to_string()]).with_score(0.8)
}

const REF_TEXT: &str = "ref ZX-8842 on file";

fn thresholds(pairs: &[(&str, f64)]) -> indexmap::IndexMap<String, f64> {
    pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
}

#[test]
fn per_label_threshold_lowers_gate() {
    let low = Recognizer::new("CASE_ID", vec![r"\b\d{6}\b".to_string()]).with_score(0.5);
    let mut o = with_recognizers(vec![low]);
    o.thresholds = Some(thresholds(&[("CASE_ID", 0.4)]));
    let r = redact("code 482913 here", &[], &o).unwrap();
    assert_eq!(r.redacted, "code [CASE_ID_1] here");
}

#[test]
fn per_label_threshold_raises_gate_and_drops() {
    let mut o = with_recognizers(vec![ref_rec()]);
    o.thresholds = Some(thresholds(&[("REF_CODE", 0.95)]));
    let r = redact(REF_TEXT, &[], &o).unwrap();
    assert_eq!(r.redacted, REF_TEXT);
    assert!(r.entities.is_empty() && r.review.is_empty());
}

#[test]
fn score_exactly_at_threshold_redacts() {
    let mut o = with_recognizers(vec![ref_rec()]);
    o.thresholds = Some(thresholds(&[("REF_CODE", 0.8)]));
    let r = redact(REF_TEXT, &[], &o).unwrap();
    assert_eq!(r.redacted, "ref [REF_CODE_1] on file");
}

#[test]
fn other_labels_unaffected_by_override() {
    let mut o = with_recognizers(vec![ref_rec()]);
    o.thresholds = Some(thresholds(&[("REF_CODE", 0.95)]));
    let r = redact("call 415-555-0132, ref ZX-8842", &[], &o).unwrap();
    assert_eq!(r.redacted, "call [PHONE_1], ref ZX-8842");
}

#[test]
fn invalid_threshold_values_raise() {
    let mut o = opts();
    o.thresholds = Some(thresholds(&[("X", 1.5)]));
    assert!(redact("x", &[], &o).is_err());
    let mut o = opts();
    o.review_margin = Some(-0.1);
    assert!(redact("x", &[], &o).is_err());
    let mut o = opts();
    o.thresholds = Some(thresholds(&[("X", -0.5)]));
    assert!(Redactor::with_options(o).is_err());
}

#[test]
fn span_in_band_becomes_review_candidate() {
    let mut o = with_recognizers(vec![ref_rec()]);
    o.thresholds = Some(thresholds(&[("REF_CODE", 0.9)]));
    o.review_margin = Some(0.2);
    let r = redact(REF_TEXT, &[], &o).unwrap();
    assert_eq!(r.redacted, REF_TEXT); // NOT redacted
    assert!(r.entities.is_empty() && r.counts.is_empty() && r.mapping.is_empty());
    assert_eq!(r.review.len(), 1);
    let c = &r.review[0];
    assert_eq!(
        (c.start, c.end, c.label.as_str(), c.text.as_str(), c.score, c.source),
        (4, 11, "REF_CODE", "ZX-8842", 0.8, EntitySource::Recognizer)
    );
}

#[test]
fn span_below_floor_is_dropped_not_reviewed() {
    let low = Recognizer::new("REF_CODE", vec![r"[A-Z]{2}-\d{4}".to_string()])
        .with_score(0.5);
    let mut o = with_recognizers(vec![low]);
    o.thresholds = Some(thresholds(&[("REF_CODE", 0.9)]));
    o.review_margin = Some(0.2);
    let r = redact(REF_TEXT, &[], &o).unwrap();
    assert!(r.review.is_empty() && r.entities.is_empty());
}

#[test]
fn span_exactly_at_floor_is_reviewed() {
    let mut o = with_recognizers(vec![ref_rec()]);
    o.thresholds = Some(thresholds(&[("REF_CODE", 0.9)]));
    o.review_margin = Some(0.1);
    let r = redact(REF_TEXT, &[], &o).unwrap();
    assert_eq!(r.review.len(), 1);
}

#[test]
fn global_margin_without_per_label_thresholds() {
    let low = Recognizer::new("REF_CODE", vec![r"[A-Z]{2}-\d{4}".to_string()])
        .with_score(0.55);
    let mut o = with_recognizers(vec![low.clone()]);
    o.review_margin = Some(0.1);
    let r = redact(REF_TEXT, &[], &o).unwrap();
    assert_eq!(r.review.len(), 1);
    let mut o = with_recognizers(vec![low]);
    o.review_margin = Some(0.04);
    let r = redact(REF_TEXT, &[], &o).unwrap();
    assert!(r.review.is_empty() && r.entities.is_empty());
}

#[test]
fn zero_margin_reviews_nothing() {
    let mut o = with_recognizers(vec![ref_rec()]);
    o.thresholds = Some(thresholds(&[("REF_CODE", 0.9)]));
    o.review_margin = Some(0.0);
    let r = redact(REF_TEXT, &[], &o).unwrap();
    assert!(r.review.is_empty() && r.entities.is_empty());
}

#[test]
fn review_candidates_sorted_by_start() {
    let mut o = with_recognizers(vec![ref_rec()]);
    o.thresholds = Some(thresholds(&[("REF_CODE", 0.9)]));
    o.review_margin = Some(0.2);
    let r = redact("QW-1093 then ZX-8842", &[], &o).unwrap();
    assert_eq!(
        r.review.iter().map(|c| c.text.as_str()).collect::<Vec<_>>(),
        ["QW-1093", "ZX-8842"]
    );
}

#[test]
fn restore_ignores_review_candidates() {
    let text = "call 415-555-0132, ref ZX-8842";
    let mut o = with_recognizers(vec![ref_rec()]);
    o.thresholds = Some(thresholds(&[("REF_CODE", 0.9)]));
    o.review_margin = Some(0.2);
    let r = redact(text, &[], &o).unwrap();
    assert_eq!(r.review.len(), 1);
    assert_eq!(phi_redact::restore(&r.redacted, &r.mapping), text);
}

#[test]
fn rules_spans_are_never_gated() {
    let mut o = opts();
    o.thresholds = Some(thresholds(&[("SSN", 0.99)]));
    o.review_margin = Some(0.5);
    let r = redact("SSN 234-56-7890", &[], &o).unwrap();
    assert_eq!(r.redacted, "SSN [SSN_1]");
    assert!(r.review.is_empty());
}

#[test]
fn denylist_spans_are_never_gated() {
    let mut o = opts();
    o.denylist = Some(vec!["Codral".into()]);
    o.thresholds = Some(thresholds(&[("CUSTOM", 0.99)]));
    o.review_margin = Some(0.5);
    let r = redact("Codral", &[], &o).unwrap();
    assert_eq!(r.redacted, "[CUSTOM_1]");
    assert!(r.review.is_empty());
}

#[test]
fn batch_results_carry_review() {
    let mut o = with_recognizers(vec![ref_rec()]);
    o.thresholds = Some(thresholds(&[("REF_CODE", 0.9)]));
    o.review_margin = Some(0.2);
    let r = Redactor::with_options(o).unwrap();
    let texts: Vec<String> = vec![REF_TEXT.to_string(), "nothing here".to_string()];
    let batch = r.redact_batch(&texts).unwrap();
    assert_eq!(batch[0].review.len(), 1);
    assert!(batch[1].review.is_empty());
    assert_eq!(batch[0].redacted, REF_TEXT);
}

// ---------------------------------------------------------------------------
// injected neural spans (ports of the Python scripted-engine cases)
// ---------------------------------------------------------------------------

fn name_span(text: &str, name: &str, score: f64) -> Entity {
    let s = text.find(name).unwrap();
    let mut e = Entity::new(s, s + name.len(), "NAME", name, EntitySource::Neural);
    e.score = Some(score);
    e
}

#[test]
fn neural_spans_partitioned_by_label_threshold() {
    let text = "Dr. Alice Ferreira reviewed Ana Souza and Bob Lindqvist.";
    let spans = vec![
        name_span(text, "Alice Ferreira", 0.95),
        name_span(text, "Ana Souza", 0.75),
        name_span(text, "Bob Lindqvist", 0.45),
    ];
    let mut o = opts();
    o.thresholds = Some(thresholds(&[("NAME", 0.8)]));
    o.review_margin = Some(0.3);
    let r = redact(text, &spans, &o).unwrap();
    // 0.95 redacts, 0.75 lands in the review band [0.5, 0.8),
    // 0.45 is below the floor and drops.
    assert!(r.redacted.contains("[NAME_1]"));
    assert!(r.redacted.contains("Ana Souza")); // left in the clear
    assert_eq!(
        r.review.iter().map(|c| c.text.as_str()).collect::<Vec<_>>(),
        ["Ana Souza"]
    );
    assert!(r.redacted.contains("Bob Lindqvist"));
    assert!(r.review.iter().all(|c| c.text != "Bob Lindqvist"));
}

#[test]
fn recognizer_beats_neural_in_full_pipeline() {
    let text = "Alice Ferreira called.";
    let spans = vec![name_span(text, "Alice Ferreira", 0.99)];
    let rec = Recognizer::new("STAFF", vec!["Alice".to_string()]).with_score(0.9);
    let r = redact(text, &spans, &with_recognizers(vec![rec])).unwrap();
    assert_eq!(r.redacted, "[STAFF_1] Ferreira called.");
}
