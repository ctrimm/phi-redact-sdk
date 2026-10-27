//! Streaming line-delimited JSON redaction (0.4.0) — the Rust
//! counterpart of python/examples/batch_jsonl.py.
//!
//! ```text
//! cargo run --release --example batch_jsonl -- [input] [output] [--field text]
//!                                                    [--operator OPERATOR] [--salt SALT]
//! ```
//!
//! Input/output default to "-" (stdin/stdout). Each non-blank line is
//! a JSON value; records whose `field` value is a string are redacted
//! in place and gain a `counts` object (label -> count). Records
//! without a string field pass through untouched, like the other SDKs.
//! Hash tokens are sorted-irreversible, so this is safe to run on
//! exports you do not want to keep a mapping for.

use phi_redact::{parse_operator, RedactOptions, Redactor};
use serde_json::{Map, Value};
use std::io::{BufRead, BufReader, Write};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut positional: Vec<String> = Vec::new();
    let mut field = "text".to_string();
    let mut operator: Option<String> = None;
    let mut salt: Option<String> = None;
    let mut model: Option<String> = None;
    let mut tokenizer: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--" => {}
            "--field" => {
                i += 1;
                field = args.get(i).cloned().unwrap_or_default();
            }
            "--operator" => {
                i += 1;
                operator = args.get(i).cloned();
            }
            "--salt" => {
                i += 1;
                salt = args.get(i).cloned();
            }
            "--model" => {
                i += 1;
                model = args.get(i).cloned();
            }
            "--tokenizer" => {
                i += 1;
                tokenizer = args.get(i).cloned();
            }
            other => positional.push(other.to_string()),
        }
        i += 1;
    }
    let input = positional.first().map(String::as_str).unwrap_or("-");
    let output = positional.get(1).map(String::as_str).unwrap_or("-");

    let mut options = RedactOptions::default();
    if let Some(op) = &operator {
        options.operator = Some(parse_operator(op).unwrap_or_else(|e| {
            eprintln!("batch_jsonl: {e}");
            std::process::exit(2);
        }));
    }
    options.salt = salt;
    let redactor = build_redactor(options, model, tokenizer);

    let reader: Box<dyn BufRead> = if input == "-" {
        Box::new(BufReader::new(std::io::stdin()))
    } else {
        match std::fs::File::open(input) {
            Ok(f) => Box::new(BufReader::new(f)),
            Err(e) => {
                eprintln!("batch_jsonl: cannot open {input}: {e}");
                std::process::exit(1);
            }
        }
    };
    let mut writer: Box<dyn Write> = if output == "-" {
        Box::new(std::io::stdout())
    } else {
        match std::fs::File::create(output) {
            Ok(f) => Box::new(f),
            Err(e) => {
                eprintln!("batch_jsonl: cannot create {output}: {e}");
                std::process::exit(1);
            }
        }
    };

    for line in reader.lines() {
        let line = line.unwrap_or_default();
        if line.trim().is_empty() {
            continue;
        }
        // Like the Python processor, a malformed line is a hard error.
        let mut record: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("batch_jsonl: invalid JSON line: {e}");
                std::process::exit(1);
            }
        };
        if let Value::Object(map) = &mut record {
            process_record(map, &field, &redactor);
        }
        let _ = writeln!(writer, "{}", serde_json::to_string(&record).unwrap());
    }
    let _ = writer.flush();
}

fn build_redactor(
    options: RedactOptions,
    model: Option<String>,
    tokenizer: Option<String>,
) -> Redactor {
    let plain = |options: RedactOptions| {
        Redactor::with_options(options).unwrap_or_else(|e| {
            eprintln!("batch_jsonl: {e}");
            std::process::exit(2);
        })
    };
    match (model, tokenizer) {
        (None, None) => plain(options),
        (Some(m), Some(t)) => {
            #[cfg(feature = "neural")]
            {
                // Labels come from a config.json (id2label) or a
                // tags.json-style BIOES array beside the model when
                // present; with_model also accepts a config path —
                // here we look for config.json next to the model.
                let dir = std::path::Path::new(&m)
                    .parent()
                    .map(|p| p.to_path_buf())
                    .unwrap_or_default();
                let cfg = ["config.json", "tags.json"]
                    .iter()
                    .map(|f| dir.join(f))
                    .find(|p| p.exists());
                match cfg {
                    Some(c) => plain(options)
                        .with_model(&m, &t, c.to_str(), phi_redact::DEFAULT_MAX_LEN)
                        .unwrap_or_else(|e| {
                            eprintln!("batch_jsonl: {e}");
                            std::process::exit(2);
                        }),
                    None => {
                        eprintln!(
                            "batch_jsonl: no config.json/tags.json beside the model"
                        );
                        std::process::exit(2);
                    }
                }
            }
            #[cfg(not(feature = "neural"))]
            {
                let _ = (m, t);
                eprintln!(
                    "batch_jsonl: --model needs a build with --features neural"
                );
                std::process::exit(2);
            }
        }
        _ => {
            eprintln!("batch_jsonl: --model and --tokenizer go together");
            std::process::exit(2);
        }
    }
}

fn process_record(map: &mut Map<String, Value>, field: &str, redactor: &Redactor) {
    let value = match map.get(field) {
        Some(Value::String(s)) => s.clone(),
        _ => return,
    };
    if let Ok(result) = redactor.redact(&value) {
        map.insert(field.to_string(), Value::String(result.redacted));
        let counts: Map<String, Value> = result
            .counts
            .iter()
            .map(|(k, v)| (k.clone(), Value::from(*v)))
            .collect();
        map.insert("counts".to_string(), Value::Object(counts));
    }
}
