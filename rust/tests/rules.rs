//! Ports of python/tests/test_rules.py (validators, scan, merge,
//! placeholders, phone boundaries, the 2026-10-19 deterministic
//! rules).

use phi_redact::rules;
use phi_redact::{
    merge_entities, redact, scan, Entity, EntitySource, RedactOptions, Redactor,
};

const NOTE: &str = "Pt Maria Santos DOB 03/14/1978 SSN 234-56-7890, ph 415-555-0132, \
                    jane@example.com, card 4111111111111111, MRN 48291375, ZIP 90210.";

// ---------------------------------------------------------------------------
// validators
// ---------------------------------------------------------------------------

#[test]
fn luhn() {
    assert!(rules::luhn_ok("4111111111111111"));
    assert!(!rules::luhn_ok("1234567812345678"));
    assert!(!rules::luhn_ok("123"));
}

#[test]
fn ssn() {
    assert!(rules::ssn_valid("234-56-7890"));
    assert!(!rules::ssn_valid("000-12-3456")); // bad area
    assert!(!rules::ssn_valid("666-12-3456")); // bad area
    assert!(!rules::ssn_valid("900-12-3456")); // bad area
    assert!(!rules::ssn_valid("234-00-7890")); // bad group
    assert!(!rules::ssn_valid("234-56-0000")); // bad serial
}

#[test]
fn iban() {
    assert!(rules::iban_valid("GB29NWBK60161331926819"));
    assert!(!rules::iban_valid("GB29NWBK60161331926818"));
}

#[test]
fn aba() {
    assert!(rules::aba_routing_ok("021000021")); // JPMorgan Chase
    assert!(rules::aba_routing_ok("494969870")); // eval case
    assert!(rules::aba_routing_ok("336229607"));
    assert!(rules::aba_routing_ok("481566888"));
    assert!(rules::aba_routing_ok("746787360"));
    assert!(!rules::aba_routing_ok("123456789")); // checksum fails
    assert!(!rules::aba_routing_ok("809096606")); // plain SSN, not routing
    assert!(!rules::aba_routing_ok("12345678")); // wrong length
    assert!(!rules::aba_routing_ok("1234567890"));
}

// ---------------------------------------------------------------------------
// scan
// ---------------------------------------------------------------------------

#[test]
fn scan_finds_all_entity_types() {
    let scanned = scan(NOTE);
    let mut labels: Vec<&str> = scanned.iter().map(|e| e.label.as_str()).collect();
    labels.sort();
    assert_eq!(
        labels,
        ["CREDITCARD", "DATE", "EMAIL", "MRN", "PHONE", "SSN", "ZIP"]
    );
}

#[test]
fn scan_skips_luhn_invalid_card() {
    let ents = scan("card 1234567812345678 charged");
    assert!(!ents.iter().any(|e| e.text == "1234567812345678"));
    assert!(!ents.iter().any(|e| e.label == "CREDITCARD"));
}

#[test]
fn scan_spans_exact_and_sorted() {
    let text = "call 415-555-0132 or jane@example.com";
    let ents = scan(text);
    let starts: Vec<usize> = ents.iter().map(|e| e.start).collect();
    let mut sorted = starts.clone();
    sorted.sort();
    assert_eq!(starts, sorted);
    let chars: Vec<char> = text.chars().collect();
    for e in &ents {
        let slice: String = chars[e.start..e.end].iter().collect();
        assert_eq!(slice, e.text);
        assert_eq!(e.source, EntitySource::Rule);
    }
}

#[test]
fn scan_golden_spans() {
    let scanned = scan(NOTE);
    let got: Vec<(usize, usize, &str)> = scanned
        .iter()
        .map(|e| (e.start, e.end, e.label.as_str()))
        .collect();
    assert_eq!(
        got,
        [
            (20, 30, "DATE"),
            (35, 46, "SSN"),
            (51, 63, "PHONE"),
            (65, 81, "EMAIL"),
            (88, 104, "CREDITCARD"),
            (106, 118, "MRN"),
            (124, 129, "ZIP"),
        ]
    );
}

#[test]
fn scan_empty_and_clean_text() {
    assert!(scan("").is_empty());
    assert!(scan("nothing to see here, just words").is_empty());
}

#[test]
fn scan_url_ip_iban() {
    let ents = scan("see https://example.com/x from 10.0.0.1, IBAN GB29NWBK60161331926819");
    let by_label = |label: &str| ents.iter().find(|e| e.label == label).map(|e| e.text.as_str());
    assert_eq!(by_label("URL"), Some("https://example.com/x"));
    assert_eq!(by_label("IP"), Some("10.0.0.1"));
    assert_eq!(by_label("IBAN"), Some("GB29NWBK60161331926819"));
}

// ---------------------------------------------------------------------------
// merge policy
// ---------------------------------------------------------------------------

fn ent(start: usize, end: usize, label: &str, source: EntitySource, score: f64) -> Entity {
    let mut e = Entity::new(start, end, label, "x", source);
    e.score = Some(score);
    e
}

#[test]
fn merge_rules_win_every_overlap() {
    let rule = ent(5, 17, "PHONE", EntitySource::Rule, 0.0);
    let neural = ent(5, 10, "NAME_PATIENT", EntitySource::Neural, 0.99);
    let merged = merge_entities(&[rule], &[neural], &[]);
    assert_eq!(
        merged.iter().map(|e| &e.label).collect::<Vec<_>>(),
        ["PHONE"]
    );
}

#[test]
fn merge_neural_neural_keeps_best_score() {
    let a = ent(0, 4, "NAME_PATIENT", EntitySource::Neural, 0.6);
    let b = ent(0, 4, "NAME_FAMILY", EntitySource::Neural, 0.9);
    let merged = merge_entities(&[], &[a, b], &[]);
    assert_eq!(
        merged.iter().map(|e| &e.label).collect::<Vec<_>>(),
        ["NAME_FAMILY"]
    );
}

#[test]
fn merge_disjoint_union_sorted() {
    let rule = ent(10, 15, "PHONE", EntitySource::Rule, 0.0);
    let neural = ent(0, 4, "NAME_PATIENT", EntitySource::Neural, 0.8);
    let merged = merge_entities(&[rule], &[neural], &[]);
    assert_eq!(
        merged.iter().map(|e| &e.label).collect::<Vec<_>>(),
        ["NAME_PATIENT", "PHONE"]
    );
}

#[test]
fn merge_touching_spans_do_not_overlap() {
    let rule = ent(0, 4, "PHONE", EntitySource::Rule, 0.0);
    let neural = ent(4, 8, "NAME", EntitySource::Neural, 0.8);
    assert_eq!(merge_entities(&[rule], &[neural], &[]).len(), 2);
}

// ---------------------------------------------------------------------------
// phone boundaries (the lookbehind re-implementation)
// ---------------------------------------------------------------------------

#[test]
fn phone_paren_fully_consumed() {
    let ents = scan("Reach her at (415) 555-0132.");
    assert_eq!(ents.len(), 1);
    assert_eq!(ents[0].label, "PHONE");
    assert_eq!(ents[0].text, "(415) 555-0132");
}

#[test]
fn phone_country_code_fully_consumed() {
    let ents = scan("Call +1 415-555-0132.");
    assert_eq!(ents.len(), 1);
    assert_eq!(ents[0].text, "+1 415-555-0132");
}

#[test]
fn phone_redact_leaves_no_stray_chars() {
    let r = redact(
        "Call (415) 555-0132 or +1 212-555-0199.",
        &[],
        &RedactOptions::default(),
    )
    .unwrap();
    assert_eq!(r.redacted, "Call [PHONE_1] or [PHONE_2].");
    assert_eq!(r.mapping.get("[PHONE_1]").map(String::as_str), Some("(415) 555-0132"));
    assert_eq!(r.mapping.get("[PHONE_2]").map(String::as_str), Some("+1 212-555-0199"));
}

#[test]
fn phone_glued_to_word_not_matched() {
    let ents = scan("abc415-555-0132");
    assert!(!ents.iter().any(|e| e.label == "PHONE"));
}

#[test]
fn phone_plain_still_matches() {
    let ents = scan("Call 415-555-0132.");
    assert_eq!(
        ents.iter().map(|e| e.text.as_str()).collect::<Vec<_>>(),
        ["415-555-0132"]
    );
}

#[test]
fn phone_paren_glued_to_letter_fully_consumed() {
    let ents = scan("x(312) 555-0167");
    assert_eq!(ents.len(), 1);
    assert_eq!(ents[0].text, "(312) 555-0167");
    let r = redact("x(312) 555-0167", &[], &RedactOptions::default()).unwrap();
    assert_eq!(r.redacted, "x[PHONE_1]");
}

#[test]
fn phone_paren_glued_without_phone_no_match() {
    assert!(scan("x(abc)").is_empty());
}

// ---------------------------------------------------------------------------
// new deterministic rules (EIN, routing, passport, DL, gov id, bank acct)
// ---------------------------------------------------------------------------

fn labels_of(text: &str) -> Vec<(String, String)> {
    scan(text).iter().map(|e| (e.label.clone(), e.text.clone())).collect()
}

#[test]
fn tax_id_ein() {
    let found = labels_of("EIN 18-6721452 on file.");
    assert!(found.contains(&("TAX_ID".to_string(), "18-6721452".to_string())));
}

#[test]
fn tax_id_not_confused_with_ssn_or_date() {
    let ents = scan("SSN 234-56-7890, EIN 18-6721452, DOB 03/14/1978.");
    let texts = |label: &str| -> Vec<&str> {
        ents.iter().filter(|e| e.label == label).map(|e| e.text.as_str()).collect()
    };
    assert_eq!(texts("SSN"), ["234-56-7890"]);
    assert_eq!(texts("TAX_ID"), ["18-6721452"]);
    assert_eq!(texts("DATE"), ["03/14/1978"]);
}

#[test]
fn routing_number() {
    let found = labels_of("Routing number 494969870 for direct deposit.");
    assert!(found.contains(&("ROUTING_NUMBER".to_string(), "494969870".to_string())));
}

#[test]
fn routing_rejects_non_checksum() {
    let ents = scan("Code 123456789 is not a routing number.");
    assert!(!ents.iter().any(|e| e.label == "ROUTING_NUMBER"));
}

#[test]
fn bank_account_context_anchored() {
    let ents = scan("Account 713591432266 at First Bank.");
    assert_eq!(ents.len(), 1);
    assert_eq!(ents[0].label, "BANK_ACCOUNT");
    assert_eq!(ents[0].text, "713591432266");
}

#[test]
fn bank_account_span_excludes_anchor() {
    let text = "Account 713591432266 at First Bank.";
    let ents = scan(text);
    let chars: Vec<char> = text.chars().collect();
    let slice: String = chars[ents[0].start..ents[0].end].iter().collect();
    assert_eq!(slice, "713591432266");
}

#[test]
fn bank_account_needs_anchor() {
    let ents = scan("Number 713591432266 appears alone.");
    assert!(!ents.iter().any(|e| e.label == "BANK_ACCOUNT"));
}

#[test]
fn bank_account_anchor_variants() {
    for (text, expected) in [
        ("Acct #987654321098 for payroll.", "987654321098"),
        ("ACCT-112233445566 on file.", "112233445566"),
        ("Account number 112233445566 on file.", "112233445566"),
    ] {
        let scanned = scan(text);
        let ents: Vec<&Entity> = scanned.iter().filter(|e| e.label == "BANK_ACCOUNT").collect();
        assert_eq!(ents.len(), 1, "{text}");
        assert_eq!(ents[0].text, expected, "{text}");
    }
}

#[test]
fn passport() {
    let found = labels_of("Passport Y12973198 verified.");
    assert!(found.contains(&("PASSPORT".to_string(), "Y12973198".to_string())));
}

#[test]
fn drivers_license() {
    let scanned = scan("DL D1315565 and CA-L123456 on file.");
    let texts: Vec<&str> = scanned
        .iter()
        .filter(|e| e.label == "DRIVERS_LICENSE")
        .map(|e| e.text.as_str())
        .collect();
    assert_eq!(texts, ["D1315565", "CA-L123456"]);
}

#[test]
fn government_id() {
    let ents = scan("USCIS LIN2190012345; alien A123456789.");
    let texts: Vec<&str> = ents
        .iter()
        .filter(|e| e.label == "GOVERNMENT_ID")
        .map(|e| e.text.as_str())
        .collect();
    assert_eq!(texts, ["LIN2190012345", "A123456789"]);
}

#[test]
fn new_rules_redact_round_trip() {
    let text = "EIN 18-6721452, routing 494969870, passport Y12973198.";
    let r = redact(text, &[], &RedactOptions::default()).unwrap();
    assert_eq!(
        r.redacted,
        "EIN [TAX_ID_1], routing [ROUTING_NUMBER_1], passport [PASSPORT_1]."
    );
    assert_eq!(phi_redact::restore(&r.redacted, &r.mapping), text);
}

// ---------------------------------------------------------------------------
// standalone rules::redact_text quirk: counters run back-to-front,
// like redact/rules.py (the engine numbers in document order instead)
// ---------------------------------------------------------------------------

#[test]
fn rules_redact_text_numbers_back_to_front() {
    let (out, mapping) = rules::redact_text("Call 415-555-0132 then 212-555-0199", None);
    assert_eq!(out, "Call [PHONE_2] then [PHONE_1]");
    assert_eq!(mapping.get("[PHONE_1]").map(String::as_str), Some("212-555-0199"));
    assert_eq!(phi_redact::restore(&out, &mapping), "Call 415-555-0132 then 212-555-0199");
}

// ---------------------------------------------------------------------------
// Redactor basics
// ---------------------------------------------------------------------------

#[test]
fn redactor_defaults_and_version() {
    assert_eq!(phi_redact::DEFAULT_THRESHOLD, 0.6);
    assert_eq!(phi_redact::DEFAULT_MAX_LEN, 384);
    assert_eq!(phi_redact::VERSION, "0.6.0");
    let r = Redactor::new();
    assert_eq!(r.engine_name(), "rules");
    assert_eq!(
        r.redact(NOTE).unwrap().redacted,
        redact(NOTE, &[], &RedactOptions::default()).unwrap().redacted
    );
}
