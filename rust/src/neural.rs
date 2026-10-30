//! ONNX neural layer (cargo feature `neural`) — a port of the Python
//! SDK's `OnnxEngine`: token-classification inference, chunked for
//! long texts, BIOES decode.
//!
//! Faithfulness notes (mirroring the Python implementation):
//! - The tokenizer's serialized truncation cap is stripped before
//!   load (the published tokenizer.json caps at 256 tokens, which
//!   would silently defeat this engine's own chunking).
//! - `encode` runs with special tokens ON and the engine then wraps
//!   with its own bos/eos ids — the marker ids come from the
//!   tokenizer's vocabulary ([CLS]/[SEP] when present, else 0/2).
//! - Only the inputs the graph declares are fed (some graphs require
//!   token_type_ids, older ones do not have that input).
//! - Offsets: the `tokenizers` crate reports BYTE offsets; all spans
//!   leaving this module are CHARACTER offsets, like every other SDK.
//! - Softmax runs in f64 (Python: numpy float64 over float32 logits),
//!   so scores track Python's to the last decimal in practice.

use crate::text::CharIndex;
use crate::types::{Entity, EntitySource, Error, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use ort::session::Session;
use ort::value::Tensor;
use tokenizers::Tokenizer;

/// Char overlap when a text is split for chunking.
const OVERLAP_CHARS: usize = 96;
/// Fallback markers for BPE vocabs without [CLS]/[SEP].
const BOS: u32 = 0;
const EOS: u32 = 2;

pub struct OnnxEngine {
    pub model_path: String,
    pub max_len: usize,
    pub threshold: f64,
    tok: Tokenizer,
    bos_id: u32,
    eos_id: u32,
    session: Mutex<Session>,
    input_names: Vec<String>,
    id2label: HashMap<i64, String>,
}

fn neural_err(msg: impl Into<String>) -> Error {
    Error::Neural(msg.into())
}

fn load_tokenizer(path: &str) -> Result<Tokenizer> {
    // Mirror _load_tokenizer: strip a serialized truncation block
    // before loading; fall back to from_file on any read/parse issue.
    if let Ok(raw) = std::fs::read_to_string(path) {
        if let Ok(mut cfg) = serde_json::from_str::<serde_json::Value>(&raw) {
            let has_truncation = cfg
                .get("truncation")
                .map(|t| !t.is_null())
                .unwrap_or(false);
            if has_truncation {
                if let Some(obj) = cfg.as_object_mut() {
                    obj.remove("truncation");
                }
                let stripped = serde_json::to_string(&cfg)
                    .map_err(|e| neural_err(e.to_string()))?;
                return Tokenizer::from_bytes(stripped.as_bytes())
                    .map_err(|e| neural_err(format!("tokenizer load failed: {e}")));
            }
        }
    }
    Tokenizer::from_file(path)
        .map_err(|e| neural_err(format!("tokenizer load failed: {e}")))
}

fn special_ids(tok: &Tokenizer) -> (u32, u32) {
    if let (Some(cls), Some(sep)) = (tok.token_to_id("[CLS]"), tok.token_to_id("[SEP]")) {
        return (cls, sep);
    }
    (BOS, EOS)
}

fn load_labels(path: &Path) -> Result<HashMap<i64, String>> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| neural_err(format!("cannot read labels {}: {e}", path.display())))?;
    let cfg: serde_json::Value = serde_json::from_str(&raw)
        .map_err(|e| neural_err(format!("bad labels JSON: {e}")))?;
    let mut out = HashMap::new();
    match &cfg {
        // tags.json style: ["O", "B-NAME", ...], index = id.
        serde_json::Value::Array(items) => {
            for (i, v) in items.iter().enumerate() {
                if let Some(s) = v.as_str() {
                    out.insert(i as i64, s.to_string());
                }
            }
        }
        serde_json::Value::Object(map) => {
            if let Some(id2label) = map.get("id2label").and_then(|v| v.as_object()) {
                for (k, v) in id2label {
                    if let (Ok(id), Some(s)) = (k.parse::<i64>(), v.as_str()) {
                        out.insert(id, s.to_string());
                    }
                }
            } else if let Some(label2id) = map.get("label2id").and_then(|v| v.as_object()) {
                for (k, v) in label2id {
                    if let Some(id) = v.as_i64() {
                        out.insert(id, k.clone());
                    }
                }
            }
        }
        _ => {}
    }
    if out.is_empty() {
        return Err(neural_err(format!(
            "no labels found in {}",
            path.display()
        )));
    }
    Ok(out)
}

/// Wall-clock stage split of one `predict` call, in milliseconds.
/// Buckets (the same definitions the Python bench harness uses):
/// `tokenize` = all tokenizer work including the split length
/// checks; `inference` = the ONNX `session.run` call only;
/// `post` = everything else (tensor assembly, output extraction,
/// softmax/argmax, BIOES decode, dedupe/sort).
#[derive(Debug, Clone, Copy, Default)]
pub struct StageTimings {
    pub tokenize_ms: f64,
    pub inference_ms: f64,
    pub post_ms: f64,
}

impl OnnxEngine {
    pub fn new(
        model_path: &str,
        tokenizer_path: &str,
        labels_path: Option<&str>,
        max_len: usize,
        threshold: f64,
    ) -> Result<Self> {
        let tok = load_tokenizer(tokenizer_path)?;
        let (bos_id, eos_id) = special_ids(&tok);
        let session = Session::builder()
            .map_err(|e| neural_err(e.to_string()))?
            .commit_from_file(model_path)
            .map_err(|e| neural_err(format!("cannot load model {model_path}: {e}")))?;
        let input_names: Vec<String> =
            session.inputs.iter().map(|i| i.name.clone()).collect();
        let labels: PathBuf = match labels_path {
            Some(p) => PathBuf::from(p),
            None => match std::env::var("PHI_REDACT_LABELS") {
                Ok(p) if !p.is_empty() => PathBuf::from(p),
                _ => Path::new(model_path)
                    .parent()
                    .map(|d| d.join("config.json"))
                    .unwrap_or_else(|| PathBuf::from("config.json")),
            },
        };
        let id2label = load_labels(&labels)?;
        Ok(OnnxEngine {
            model_path: model_path.to_string(),
            max_len,
            threshold,
            tok,
            bos_id,
            eos_id,
            session: Mutex::new(session),
            input_names,
            id2label,
        })
    }

    // -- chunking ------------------------------------------------------

    fn split(&self, text: &str, off: usize, out: &mut Vec<(String, usize)>) {
        let n_tokens = self
            .tok
            .encode(text, true)
            .map(|e| e.get_ids().len())
            .unwrap_or(usize::MAX);
        if n_tokens <= self.max_len.saturating_sub(2) {
            out.push((text.to_string(), off));
            return;
        }
        let chars: Vec<char> = text.chars().collect();
        let n = chars.len();
        let mid = n / 2;
        let mut cut = mid;
        'search: for i in 1..(n / 2) {
            for j in [mid + i, mid - i] {
                if j > 0 && j < n && matches!(chars[j], ' ' | '\t' | '\n') {
                    cut = j;
                    break 'search;
                }
            }
        }
        let left: String = chars[..cut].iter().collect();
        self.split(&left, off, out);
        let mut back = cut.saturating_sub(OVERLAP_CHARS);
        if back == 0 {
            back = cut;
        }
        let right: String = chars[back..].iter().collect();
        self.split(&right, off + back, out);
    }

    // -- inference -----------------------------------------------------

    /// Neural spans for `text`. `threshold` overrides the engine's
    /// confidence gate for this call only.
    pub fn predict(&self, text: &str, threshold: Option<f64>) -> Result<Vec<Entity>> {
        self.predict_inner(text, threshold, None)
    }

    /// `predict` plus a stage split (see `StageTimings`). Used by the
    /// bench harness to decompose the neural pipeline; timings are
    /// measured inside the real code path.
    pub fn predict_timed(
        &self,
        text: &str,
        threshold: Option<f64>,
    ) -> Result<(Vec<Entity>, StageTimings)> {
        let mut stages = StageTimings::default();
        let out = self.predict_inner(text, threshold, Some(&mut stages))?;
        Ok((out, stages))
    }

    fn predict_inner(
        &self,
        text: &str,
        threshold: Option<f64>,
        mut stages: Option<&mut StageTimings>,
    ) -> Result<Vec<Entity>> {
        let gate = threshold.unwrap_or(self.threshold);
        let t0 = std::time::Instant::now();
        let mut chunks: Vec<(String, usize)> = Vec::new();
        self.split(text, 0, &mut chunks);
        if let Some(s) = stages.as_deref_mut() {
            s.tokenize_ms += t0.elapsed().as_secs_f64() * 1000.0;
        }
        let text_index = CharIndex::new(text);
        let mut entities: Vec<Entity> = Vec::new();
        for (chunk, coff) in &chunks {
            entities.extend(self.predict_chunk(
                text,
                &text_index,
                chunk,
                *coff,
                gate,
                stages.as_deref_mut(),
            )?);
        }
        let t0 = std::time::Instant::now();
        // Dedupe identical spans from chunk overlap, keep best score.
        let mut best: HashMap<(usize, usize, String), Entity> = HashMap::new();
        let mut order: Vec<(usize, usize, String)> = Vec::new();
        for e in entities {
            let key = (e.start, e.end, e.label.clone());
            match best.get(&key) {
                Some(prev) if prev.score.unwrap_or(0.0) >= e.score.unwrap_or(0.0) => {}
                Some(_) => {
                    best.insert(key, e);
                }
                None => {
                    order.push(key.clone());
                    best.insert(key, e);
                }
            }
        }
        let mut out: Vec<Entity> = order.into_iter().map(|k| best.remove(&k).unwrap()).collect();
        out.sort_by_key(|e| e.start);
        if let Some(s) = stages.as_deref_mut() {
            s.post_ms += t0.elapsed().as_secs_f64() * 1000.0;
        }
        Ok(out)
    }

    fn predict_chunk(
        &self,
        text: &str,
        text_index: &CharIndex,
        chunk: &str,
        coff: usize,
        gate: f64,
        mut stages: Option<&mut StageTimings>,
    ) -> Result<Vec<Entity>> {
        let t0 = std::time::Instant::now();
        let enc = self
            .tok
            .encode(chunk, true)
            .map_err(|e| neural_err(format!("tokenize failed: {e}")))?;
        if let Some(s) = stages.as_deref_mut() {
            s.tokenize_ms += t0.elapsed().as_secs_f64() * 1000.0;
        }
        let t_post = std::time::Instant::now();
        let chunk_index = CharIndex::new(chunk);
        let ids: Vec<i64> = std::iter::once(self.bos_id as i64)
            .chain(enc.get_ids().iter().map(|&i| i as i64))
            .chain(std::iter::once(self.eos_id as i64))
            .collect();
        let n = ids.len();
        // Global CHAR offsets, built exactly like the Python engine
        // (see `global_offsets`): tokenizer offsets are byte ranges
        // within the chunk; convert to chunk chars, then shift.
        let offs = global_offsets(enc.get_offsets(), &chunk_index, coff);

        let mask: Vec<i64> = vec![1; n];
        let types: Vec<i64> = vec![0; n];
        let mut inputs: Vec<(String, Tensor<i64>)> = Vec::new();
        for name in &self.input_names {
            let data = match name.as_str() {
                "input_ids" => ids.clone(),
                "attention_mask" => mask.clone(),
                "token_type_ids" => types.clone(),
                _ => continue,
            };
            let tensor = Tensor::<i64>::from_array(([1usize, n], data))
                .map_err(|e| neural_err(e.to_string()))?;
            inputs.push((name.clone(), tensor));
        }
        let mut session = self
            .session
            .lock()
            .map_err(|_| neural_err("session lock poisoned"))?;
        if let Some(s) = stages.as_deref_mut() {
            // Tensor assembly counts as post, matching the bucket
            // definitions (inference = session.run only).
            s.post_ms += t_post.elapsed().as_secs_f64() * 1000.0;
        }
        let t_run = std::time::Instant::now();
        let outputs = session
            .run(inputs)
            .map_err(|e| neural_err(format!("inference failed: {e}")))?;
        if let Some(s) = stages.as_deref_mut() {
            s.inference_ms += t_run.elapsed().as_secs_f64() * 1000.0;
        }
        let t_post = std::time::Instant::now();
        let logits = outputs
            .get("logits")
            .ok_or_else(|| neural_err("model has no 'logits' output"))?;
        let (shape, data) = logits
            .try_extract_tensor::<f32>()
            .map_err(|e| neural_err(format!("logits not f32: {e}")))?;
        let dims: Vec<i64> = shape.iter().copied().collect();
        if dims.len() != 3 || dims[0] != 1 {
            return Err(neural_err(format!("unexpected logits shape {dims:?}")));
        }
        let n_labels = dims[2] as usize;
        if data.len() != n * n_labels {
            return Err(neural_err("logits length mismatch"));
        }
        // softmax (f64, like numpy over float32 logits) + argmax
        let mut pred: Vec<i64> = Vec::with_capacity(n);
        let mut probs: Vec<Vec<f64>> = Vec::with_capacity(n);
        for t in 0..n {
            let row = &data[t * n_labels..(t + 1) * n_labels];
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            let mut exps: Vec<f64> = row.iter().map(|&x| (x as f64 - max).exp()).collect();
            let sum: f64 = exps.iter().sum();
            for v in exps.iter_mut() {
                *v /= sum;
            }
            let mut best_i = 0usize;
            let mut best_v = f64::NEG_INFINITY;
            for (i, &v) in exps.iter().enumerate() {
                if v > best_v {
                    best_v = v;
                    best_i = i;
                }
            }
            pred.push(best_i as i64);
            probs.push(exps);
        }
        let out = self.decode(text, text_index, &offs, &pred, &probs, gate);
        if let Some(s) = stages.as_deref_mut() {
            // Extraction + softmax/argmax + BIOES decode count as post.
            s.post_ms += t_post.elapsed().as_secs_f64() * 1000.0;
        }
        Ok(out)
    }

    fn decode(
        &self,
        text: &str,
        index: &CharIndex,
        offs: &[(usize, usize)],
        pred: &[i64],
        probs: &[Vec<f64>],
        gate: f64,
    ) -> Vec<Entity> {
        let mut entities: Vec<Entity> = Vec::new();
        // [label, start, end, confidences]
        let mut cur: Option<(String, usize, usize, Vec<f64>)> = None;
        let flush = |cur: &mut Option<(String, usize, usize, Vec<f64>)>,
                     entities: &mut Vec<Entity>| {
            if let Some((label, s, e, confs)) = cur.take() {
                let (s, e) = trim(text, index, s, e);
                if s < e {
                    let score =
                        round4(confs.iter().sum::<f64>() / confs.len() as f64);
                    entities.push(Entity {
                        start: s,
                        end: e,
                        label,
                        text: index.slice(text, s, e).to_string(),
                        source: EntitySource::Neural,
                        score: Some(score),
                        placeholder: None,
                    });
                }
            }
        };
        for (i, (&(s, e), &lid)) in offs.iter().zip(pred.iter()).enumerate() {
            if s == 0 && e == 0 {
                flush(&mut cur, &mut entities);
                continue;
            }
            let tag = self.id2label.get(&lid).map(String::as_str).unwrap_or("O");
            if tag == "O" {
                flush(&mut cur, &mut entities);
                continue;
            }
            let (prefix, typ) = match tag.split_once('-') {
                Some((p, t)) => (p, t.to_string()),
                None => ("", tag.to_string()),
            };
            let conf = probs[i].iter().copied().fold(f64::NEG_INFINITY, f64::max);
            match prefix {
                "S" => {
                    flush(&mut cur, &mut entities);
                    let (s2, e2) = trim(text, index, s, e);
                    if s2 < e2 {
                        entities.push(Entity {
                            start: s2,
                            end: e2,
                            label: typ,
                            text: index.slice(text, s2, e2).to_string(),
                            source: EntitySource::Neural,
                            score: Some(round4(conf)),
                            placeholder: None,
                        });
                    }
                }
                "B" => {
                    flush(&mut cur, &mut entities);
                    cur = Some((typ, s, e, vec![conf]));
                }
                _ => {
                    // I / E, lenient on malformed transitions
                    match &mut cur {
                        Some((label, _, end, confs)) if *label == typ => {
                            *end = e;
                            confs.push(conf);
                        }
                        _ => {
                            cur = Some((typ, s, e, vec![conf]));
                        }
                    }
                    if prefix == "E" {
                        flush(&mut cur, &mut entities);
                    }
                }
            }
        }
        flush(&mut cur, &mut entities);
        entities.retain(|e| e.score.unwrap_or(0.0) >= gate);
        entities
    }
}

fn round4(x: f64) -> f64 {
    (x * 10000.0).round() / 10000.0
}

/// Global char offsets for one chunk's encoding, mirroring the
/// Python engine exactly (`OnnxEngine.predict`):
/// `[(0, 0)] + [(s + coff, e + coff) for s, e in enc.offsets] + [(0, 0)]`.
///
/// EVERY encoding offset is shifted — including the `(0, 0)` offsets
/// the tokenizer reports for the [CLS]/[SEP] specials, which become
/// `(coff, coff)`. The decoder's special-token test skips only
/// offsets exactly equal to `(0, 0)`, so for chunks after the first
/// those positions are decoded as content: their predictions can
/// open or extend spans, and their confidences enter span score
/// means. This is load-bearing for parity with Python (see the
/// chunk-boundary regression test in tests/neural.rs):
/// special-casing `(0, 0)` here drops the boundary token from the
/// mean — on one bisected document that moved a span score from
/// 0.9202 to 0.9903 and flipped the keep-highest-score merge,
/// leaving a street address unredacted.
fn global_offsets(
    enc_offsets: &[(usize, usize)],
    chunk_index: &CharIndex,
    coff: usize,
) -> Vec<(usize, usize)> {
    let mut offs: Vec<(usize, usize)> = Vec::with_capacity(enc_offsets.len() + 2);
    offs.push((0, 0));
    for &(bs, be) in enc_offsets {
        offs.push((
            chunk_index.char_of_byte(bs) + coff,
            chunk_index.char_of_byte(be) + coff,
        ));
    }
    offs.push((0, 0));
    offs
}

/// Whitespace trim on char spans (Python `_trim`).
fn trim(text: &str, index: &CharIndex, mut s: usize, mut e: usize) -> (usize, usize) {
    let chars = |i: usize| index.slice(text, i, i + 1).chars().next();
    while s < e && chars(s).map(|c| c.is_whitespace()).unwrap_or(false) {
        s += 1;
    }
    while e > s && chars(e - 1).map(|c| c.is_whitespace()).unwrap_or(false) {
        e -= 1;
    }
    (s, e)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The chunk-boundary parity mechanism, pinned at the offset
    /// layer: encoding offsets of (0, 0) (the [CLS]/[SEP] specials)
    /// must come out shifted by the chunk offset, exactly like
    /// Python's `[(s + coff, e + coff) for s, e in enc.offsets]` —
    /// NOT pinned to (0, 0), which is what the decoder treats as a
    /// skippable sentinel. Reverting `global_offsets` to the old
    /// special-case fails this test.
    #[test]
    fn global_offsets_shift_specials_like_python() {
        let chunk = "x Apt. 36 done";
        let index = CharIndex::new(chunk);
        // Byte ranges as the tokenizer reports them for one chunk:
        // [CLS] (0,0), four content tokens, [SEP] (0,0).
        let enc = [(0, 0), (0, 1), (1, 5), (5, 6), (6, 9), (0, 0)];
        let offs = global_offsets(&enc, &index, 886);
        assert_eq!(
            offs,
            vec![
                (0, 0),     // the engine's own bos sentinel
                (886, 886), // [CLS]: (0,0) shifted — decoded as content
                (886, 887),
                (887, 891),
                (891, 892),
                (892, 895),
                (886, 886), // [SEP]: same treatment
                (0, 0),     // the engine's own eos sentinel
            ]
        );
    }

    /// First chunk (coff = 0): shifted specials land on (0, 0) and
    /// stay skippable — the one case where the old and new
    /// constructions agree.
    #[test]
    fn global_offsets_first_chunk_specials_stay_zero() {
        let chunk = "abc";
        let index = CharIndex::new(chunk);
        let enc = [(0, 0), (0, 3), (0, 0)];
        assert_eq!(
            global_offsets(&enc, &index, 0),
            vec![(0, 0), (0, 0), (0, 3), (0, 0), (0, 0)]
        );
    }
}
