use std::fmt;

use tiktoken_rs::CoreBPE;

pub const VOCAB_SIZE: usize = 50_257;
pub const END_OF_TEXT_ID: u32 = 50_256;

#[derive(Debug)]
pub struct TokenizerError(String);

impl fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tokenizer-Fehler: {}", self.0)
    }
}

impl std::error::Error for TokenizerError {}

// ANCHOR: struct
/// Dünner Wrapper um den GPT-2-Tokenizer (`r50k_base`) aus `tiktoken-rs`.
pub struct Gpt2Tokenizer {
    bpe: &'static CoreBPE,
}
// ANCHOR_END: struct

impl Gpt2Tokenizer {
    // ANCHOR: new
    pub fn new() -> Self {
        Self { bpe: tiktoken_rs::r50k_base_singleton() }
    }
    // ANCHOR_END: new

    // ANCHOR: encode_decode
    /// Text → Token-IDs. `<|endoftext|>` wird als Sondertoken erkannt.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.bpe.encode_with_special_tokens(text)
    }

    /// Token-IDs → Text. Schlägt fehl, wenn die Bytes kein gültiges UTF-8 ergeben.
    pub fn decode(&self, ids: &[u32]) -> Result<String, TokenizerError> {
        self.bpe
            .decode(ids)
            .map_err(|e| TokenizerError(e.to_string()))
    }
    // ANCHOR_END: encode_decode

    // ANCHOR: pieces
    /// Text jedes einzelnen Tokens. Ein Token kann ein halbes UTF-8-Zeichen sein
    /// und wird dann mit � dargestellt.
    pub fn pieces(&self, text: &str) -> Vec<String> {
        self.encode(text)
            .into_iter()
            .map(|id| self.piece(id))
            .collect()
    }
    // ANCHOR_END: pieces

    fn piece(&self, id: u32) -> String {
        let bytes = self.bpe.decode_bytes(&[id]).unwrap_or_default();
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

impl Default for Gpt2Tokenizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ANCHOR: test_roundtrip
    #[test]
    fn roundtrip_erhaelt_text() {
        let tokenizer = Gpt2Tokenizer::new();
        for text in ["Hello, world!", "Grüezi öäü ß", "Emoji 🙂 ok", ""] {
            let ids = tokenizer.encode(text);
            assert_eq!(tokenizer.decode(&ids).unwrap(), text);
        }
    }
    // ANCHOR_END: test_roundtrip

    #[test]
    fn bekannte_gpt2_ids() {
        let tokenizer = Gpt2Tokenizer::new();
        assert_eq!(tokenizer.encode("Hello world"), [15496, 995]);
    }

    #[test]
    fn endoftext_ist_sondertoken() {
        let tokenizer = Gpt2Tokenizer::new();
        assert_eq!(tokenizer.encode("<|endoftext|>"), [END_OF_TEXT_ID]);
    }

    #[test]
    fn alle_ids_liegen_im_vokabular() {
        let tokenizer = Gpt2Tokenizer::new();
        let ids = tokenizer.encode("Ein längerer Satz mit Zahlen 12345 und 🙂.");
        assert!(ids.iter().all(|&id| (id as usize) < VOCAB_SIZE));
    }

    #[test]
    fn halbes_utf8_zeichen_wird_nicht_dekodiert() {
        let tokenizer = Gpt2Tokenizer::new();
        let ids = tokenizer.encode("🙂");
        assert!(ids.len() > 1);
        assert!(tokenizer.decode(&ids[..1]).is_err());
    }
}
