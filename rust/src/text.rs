//! Character-index plumbing.
//!
//! Python string indices count Unicode code points; the Rust `regex`
//! crate reports byte ranges. `CharIndex` converts between the two
//! so every public offset in this crate is a character index, exactly
//! matching the Python SDK.

/// Maps between byte offsets and character indices for one string.
#[derive(Debug, Clone)]
pub struct CharIndex {
    /// `bytes[i]` is the byte offset where character `i` starts;
    /// a final sentinel entry holds the string's byte length.
    bytes: Vec<usize>,
}

impl CharIndex {
    pub fn new(text: &str) -> CharIndex {
        let mut bytes: Vec<usize> = text.char_indices().map(|(b, _)| b).collect();
        bytes.push(text.len());
        CharIndex { bytes }
    }

    /// Number of characters.
    pub fn len(&self) -> usize {
        self.bytes.len() - 1
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Byte offset where character `char_idx` starts (`len()` maps to
    /// the string's byte length).
    pub fn byte_of(&self, char_idx: usize) -> usize {
        self.bytes[char_idx]
    }

    /// Character index for a byte offset. The offset must fall on a
    /// character boundary (regex match ranges always do).
    pub fn char_of_byte(&self, byte: usize) -> usize {
        debug_assert!(self.bytes.binary_search(&byte).is_ok());
        self.bytes.partition_point(|&b| b < byte)
    }

    /// The substring spanning character range `[start, end)`.
    pub fn slice<'a>(&self, text: &'a str, start: usize, end: usize) -> &'a str {
        &text[self.bytes[start]..self.bytes[end]]
    }
}

/// A `\w` word character, Python `re` semantics: Unicode alphanumeric
/// or underscore (Python also counts some combining marks; the
/// difference does not arise on realistic text).
pub fn is_word_char(c: char) -> bool {
    c == '_' || c.is_alphanumeric()
}

/// Case-fold a string the way Python's `str.casefold()` does, for the
/// allowlist comparison. Rust's `to_lowercase` covers the common
/// cases; a few multi-char foldings Python performs are spelled out.
pub fn casefold(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            'ß' | 'ẞ' => out.push_str("ss"),
            'ﬀ' => out.push_str("ff"),
            'ﬁ' => out.push_str("fi"),
            'ﬂ' => out.push_str("fl"),
            'ﬃ' => out.push_str("ffi"),
            'ﬄ' => out.push_str("ffl"),
            'ﬅ' | 'ﬆ' => out.push_str("st"),
            _ => out.extend(c.to_lowercase()),
        }
    }
    out
}

/// Byte offset of the start of the character after the one beginning
/// at byte offset `from` (or the string length if there is none).
pub fn next_char_boundary(text: &str, from: usize) -> usize {
    if from >= text.len() {
        return text.len();
    }
    let mut i = from + 1;
    while i < text.len() && !text.is_char_boundary(i) {
        i += 1;
    }
    i
}

/// The character immediately before byte offset `byte`, if any.
pub fn prev_char(text: &str, byte: usize) -> Option<char> {
    if byte == 0 {
        return None;
    }
    text[..byte].chars().next_back()
}
