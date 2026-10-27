//! Ports of python/tests/test_parity_features.py (0.4.0): anonymize
//! operators (incl. the SHARED hash vectors), allow/deny lists,
//! scrub/restore + ChunkedRestorer, batch.

use phi_redact::{
    hash_token, redact, restore, AnonymizeOperator, ChunkedRestorer, RedactOptions,
    Redactor,
};

const NOTE: &str = "Call 415-555-0132 about SSN 234-56-7890.";

fn opts() -> RedactOptions {
    RedactOptions::default()
}

// ---------------------------------------------------------------------------
// hash: shared vectors, pinned in all three SDK suites
// ---------------------------------------------------------------------------

#[test]
fn hash_shared_vectors() {
    // (salt, label, value, expected token)
    let vectors = [
        ("", "SSN", "234-56-7890", "[SSN_5f6c8b16]"),
        ("pepper", "SSN", "234-56-7890", "[SSN_e2e9f611]"),
        ("", "EMAIL", "jane@example.com", "[EMAIL_34da503d]"),
        ("s3cret-salt", "NAME_PATIENT", "Maria Santos", "[NAME_PATIENT_09a6b457]"),
        ("parity-salt", "PHONE", "415-555-0132", "[PHONE_954b7ae0]"),
    ];
    for (salt, label, value, expected) in vectors {
        assert_eq!(hash_token(label, value, salt), expected, "{salt}/{label}/{value}");
    }
}

#[test]
fn hash_via_redact() {
    let o = RedactOptions {
        operator: Some(AnonymizeOperator::Hash),
        salt: Some("pepper".into()),
        ..Default::default()
    };
    let r = redact("SSN 234-56-7890", &[], &o).unwrap();
    assert_eq!(r.redacted, "SSN [SSN_e2e9f611]");
    assert!(r.mapping.is_empty()); // hash is irreversible by design
}

#[test]
fn hash_consistent_across_documents() {
    let o = RedactOptions {
        operator: Some(AnonymizeOperator::Hash),
        salt: Some("s".into()),
        ..Default::default()
    };
    let a = redact("SSN 234-56-7890", &[], &o).unwrap();
    let b = redact("again 234-56-7890 here", &[], &o).unwrap();
    assert_eq!(
        a.redacted.split(' ').last().unwrap(),
        b.redacted.split(' ').nth(1).unwrap()
    );
}

#[test]
fn hash_salt_changes_token() {
    let mk = |salt: &str| RedactOptions {
        operator: Some(AnonymizeOperator::Hash),
        salt: Some(salt.into()),
        ..Default::default()
    };
    let a = redact("SSN 234-56-7890", &[], &mk("a")).unwrap();
    let b = redact("SSN 234-56-7890", &[], &mk("b")).unwrap();
    assert_ne!(a.redacted, b.redacted);
}

// ---------------------------------------------------------------------------
// operators
// ---------------------------------------------------------------------------

#[test]
fn label_operator_presidio_style() {
    let o = RedactOptions {
        operator: Some(AnonymizeOperator::Label),
        ..Default::default()
    };
    let r = redact(NOTE, &[], &o).unwrap();
    assert_eq!(r.redacted, "Call <PHONE> about SSN <SSN>.");
    assert!(r.mapping.is_empty());
    assert_eq!(r.counts.get("PHONE"), Some(&1));
    assert_eq!(r.counts.get("SSN"), Some(&1));
    assert_eq!(r.entities[0].placeholder.as_deref(), Some("<PHONE>"));
}

#[test]
fn mask_structured_ids_keep_last_four() {
    let o = RedactOptions {
        operator: Some(AnonymizeOperator::Mask),
        ..Default::default()
    };
    let r = redact(
        "SSN 234-56-7890, card 4111111111111111, ph (415) 555-0132",
        &[],
        &o,
    )
    .unwrap();
    assert_eq!(
        r.redacted,
        "SSN ***-**-7890, card ************1111, ph (***) ***-0132"
    );
    assert!(r.mapping.is_empty());
}

#[test]
fn mask_other_values_same_length_asterisks() {
    let o = RedactOptions {
        operator: Some(AnonymizeOperator::Mask),
        ..Default::default()
    };
    let r = redact("ZIP 90210, see https://example.com/x", &[], &o).unwrap();
    assert_eq!(r.redacted, "ZIP *****, see *********************");
}

#[test]
fn remove_operator_deletes_spans() {
    let o = RedactOptions {
        operator: Some(AnonymizeOperator::Remove),
        ..Default::default()
    };
    let r = redact("Call 415-555-0132 now", &[], &o).unwrap();
    assert_eq!(r.redacted, "Call  now");
    assert!(r.mapping.is_empty());
    assert_eq!(r.entities[0].placeholder.as_deref(), Some(""));
}

#[test]
fn unknown_operator_rejected() {
    assert!(phi_redact::parse_operator("bogus").is_err());
    assert_eq!(
        phi_redact::parse_operator("hash").unwrap(),
        AnonymizeOperator::Hash
    );
}

#[test]
fn redactor_operator_option() {
    let r = Redactor::with_options(RedactOptions {
        operator: Some(AnonymizeOperator::Label),
        ..Default::default()
    })
    .unwrap();
    assert_eq!(r.redact(NOTE).unwrap().redacted, "Call <PHONE> about SSN <SSN>.");
}

// ---------------------------------------------------------------------------
// allow / deny lists
// ---------------------------------------------------------------------------

#[test]
fn allowlisted_value_not_redacted() {
    let o = RedactOptions {
        allowlist: Some(vec!["415-555-0132".into()]),
        ..Default::default()
    };
    let r = redact(NOTE, &[], &o).unwrap();
    assert_eq!(r.redacted, "Call 415-555-0132 about SSN [SSN_1].");
    assert_eq!(r.counts.len(), 1);
    assert_eq!(r.counts.get("SSN"), Some(&1));
}

#[test]
fn allowlist_is_case_insensitive() {
    let o = RedactOptions {
        allowlist: Some(vec!["jane@example.com".into()]),
        ..Default::default()
    };
    let r = redact("mail JANE@EXAMPLE.COM", &[], &o).unwrap();
    assert_eq!(r.redacted, "mail JANE@EXAMPLE.COM");
}

#[test]
fn redactor_allowlist_applies_to_scan() {
    let r = Redactor::with_options(RedactOptions {
        allowlist: Some(vec!["415-555-0132".into()]),
        ..Default::default()
    })
    .unwrap();
    let scanned = r.scan(NOTE).unwrap();
    let labels: Vec<&str> = scanned.iter().map(|e| e.label.as_str()).collect();
    assert_eq!(labels, ["SSN"]);
}

#[test]
fn denylist_term_redacted_as_custom() {
    let o = RedactOptions {
        denylist: Some(vec!["Codral".into()]),
        ..Default::default()
    };
    let r = redact("Patient takes Codral daily.", &[], &o).unwrap();
    assert_eq!(r.redacted, "Patient takes [CUSTOM_1] daily.");
    assert_eq!(r.entities.len(), 1);
    assert_eq!(r.entities[0].label, "CUSTOM");
    assert_eq!(r.entities[0].source, phi_redact::EntitySource::Denylist);
    assert_eq!(restore(&r.redacted, &r.mapping), "Patient takes Codral daily.");
}

#[test]
fn denylist_custom_label() {
    let o = RedactOptions {
        denylist: Some(vec!["Codral".into()]),
        deny_label: Some("NAME_PATIENT".into()),
        ..Default::default()
    };
    let r = redact("Patient takes Codral daily.", &[], &o).unwrap();
    assert_eq!(r.redacted, "Patient takes [NAME_PATIENT_1] daily.");
}

#[test]
fn denylist_case_insensitive_word_boundaries() {
    let o = RedactOptions {
        denylist: Some(vec!["Codral".into()]),
        ..Default::default()
    };
    let r = redact("codral and CODRAL, not Codraline.", &[], &o).unwrap();
    assert_eq!(r.redacted, "[CUSTOM_1] and [CUSTOM_2], not Codraline.");
}

#[test]
fn denylist_span_wins_overlap_with_detected() {
    // The deny match sits inside the MRN rule span; the rule span is
    // dropped and only the denied digits are hidden.
    let o = RedactOptions {
        denylist: Some(vec!["48291375".into()]),
        ..Default::default()
    };
    let r = redact("MRN 48291375", &[], &o).unwrap();
    assert_eq!(r.redacted, "MRN [CUSTOM_1]");
}

#[test]
fn deny_wins_over_allow() {
    let o = RedactOptions {
        allowlist: Some(vec!["Codral".into()]),
        denylist: Some(vec!["Codral".into()]),
        ..Default::default()
    };
    let r = redact("Codral", &[], &o).unwrap();
    assert_eq!(r.redacted, "[CUSTOM_1]");
}

#[test]
fn denylist_no_match_is_noop() {
    let o = RedactOptions {
        denylist: Some(vec!["Zebra".into()]),
        ..Default::default()
    };
    let r = redact(NOTE, &[], &o).unwrap();
    assert_eq!(r.redacted, redact(NOTE, &[], &opts()).unwrap().redacted);
}

// ---------------------------------------------------------------------------
// scrub / restore + ChunkedRestorer
// ---------------------------------------------------------------------------

#[test]
fn scrub_returns_redacted_and_mapping() {
    let r = Redactor::new();
    let (redacted, mapping) = r.scrub(NOTE).unwrap();
    let base = redact(NOTE, &[], &opts()).unwrap();
    assert_eq!(redacted, base.redacted);
    assert_eq!(mapping, base.mapping);
    assert_eq!(Redactor::restore(&redacted, &mapping), NOTE);
}

#[test]
fn scrub_with_label_operator_has_empty_mapping() {
    let r = Redactor::with_options(RedactOptions {
        operator: Some(AnonymizeOperator::Label),
        ..Default::default()
    })
    .unwrap();
    let (redacted, mapping) = r.scrub(NOTE).unwrap();
    assert_eq!(redacted, "Call <PHONE> about SSN <SSN>.");
    assert!(mapping.is_empty());
}

#[test]
fn restore_many_same_label_placeholders() {
    let phones: Vec<String> = (0..12).map(|i| format!("415-555-01{i:02}")).collect();
    let text = phones.join(" ");
    let r = redact(&text, &[], &opts()).unwrap();
    assert_eq!(r.counts.get("PHONE"), Some(&12));
    assert_eq!(restore(&r.redacted, &r.mapping), text);
}

#[test]
fn chunked_char_by_char_round_trip() {
    let original = "The number is 415-555-0132 and SSN 234-56-7890.";
    let (redacted, mapping) = Redactor::new().scrub(original).unwrap();
    let mut cr = ChunkedRestorer::new(mapping);
    let mut out = String::new();
    for c in redacted.chars() {
        out.push_str(&cr.feed(&c.to_string()));
    }
    out.push_str(&cr.flush());
    assert_eq!(out, original);
}

#[test]
fn chunked_every_split_point() {
    let original = "Call 415-555-0132.";
    let (redacted, mapping) = Redactor::new().scrub(original).unwrap();
    let chars: Vec<char> = redacted.chars().collect();
    for i in 0..=chars.len() {
        let mut cr = ChunkedRestorer::new(mapping.clone());
        let a: String = chars[..i].iter().collect();
        let b: String = chars[i..].iter().collect();
        let out = cr.feed(&a) + &cr.feed(&b) + &cr.flush();
        assert_eq!(out, original, "split at {i}");
    }
}

#[test]
fn chunked_unknown_placeholder_untouched() {
    let (_, mapping) = Redactor::new().scrub(NOTE).unwrap();
    let mut cr = ChunkedRestorer::new(mapping);
    let mut out = cr.feed("ref [FOO_9] and [PHONE_");
    out.push_str(&cr.feed("1] done"));
    out.push_str(&cr.flush());
    assert_eq!(out, "ref [FOO_9] and 415-555-0132 done");
}

#[test]
fn chunked_partial_placeholder_at_flush_untouched() {
    let (_, mapping) = Redactor::new().scrub(NOTE).unwrap();
    let mut cr = ChunkedRestorer::new(mapping);
    let out = cr.feed("ends with [PHO") + &cr.flush();
    assert_eq!(out, "ends with [PHO");
}

#[test]
fn chunked_empty_mapping_passes_text_through() {
    let mut cr = ChunkedRestorer::new(indexmap::IndexMap::new());
    assert_eq!(cr.feed("plain [TEXT_1] here"), "plain [TEXT_1] here");
    assert_eq!(cr.flush(), "");
}

// ---------------------------------------------------------------------------
// batch
// ---------------------------------------------------------------------------

#[test]
fn batch_preserves_order_and_matches_single() {
    let texts: Vec<String> = [NOTE, "nothing to see here", "SSN 234-56-7890"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let r = Redactor::new();
    let batch = r.redact_batch(&texts).unwrap();
    let singles: Vec<String> = texts.iter().map(|t| r.redact(t).unwrap().redacted).collect();
    assert_eq!(
        batch.iter().map(|b| b.redacted.clone()).collect::<Vec<_>>(),
        singles
    );
    assert!(batch[1].entities.is_empty());
}

#[test]
fn batch_empty() {
    assert!(Redactor::new().redact_batch(&[]).unwrap().is_empty());
    assert!(phi_redact::redact_batch(&[], &opts()).unwrap().is_empty());
}
