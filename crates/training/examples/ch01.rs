//! Kapitel 1: Was macht der GPT-2-Tokenizer mit Text?
//! Ausführen: cargo run -p training --example ch01

use training::tokenizer::{Gpt2Tokenizer, VOCAB_SIZE};

const BEISPIELE: [&str; 7] = [
    "Hello world",
    "The quick brown fox jumps over the lazy dog.",
    "Grüezi mit Umlauten: äöü",
    "Emoji: 🙂",
    "Antidisestablishmentarianism",
    "Ende<|endoftext|>",
    " 🙂",
];

fn main() {
    let tokenizer = Gpt2Tokenizer::new();
    println!("Vokabulargrösse: {VOCAB_SIZE}\n");
    for text in BEISPIELE {
        zeige(&tokenizer, text);
    }
}

fn zeige(tokenizer: &Gpt2Tokenizer, text: &str) {
    let ids = token_ids(tokenizer, text);
    let tokens = token_count(tokenizer, text);
    let roundtrip = tokenizer.decode(&ids).expect("gültiges UTF-8");
    println!("Text:      {text:?}");
    println!("IDs:       {ids:?}");
    println!("Tokens:     {tokens}");
    println!("Stücke:    {:?}", tokenizer.pieces(text));
    println!(
        "Zeichen/Bytes/Token: {}/{}/{}",
        text.chars().count(),
        text.len(),
        ids.len()
    );
    println!("Roundtrip: {}\n", roundtrip == text);
}

fn token_ids(tokenizer: &Gpt2Tokenizer, text: &str) -> Vec<u32> {
    tokenizer.encode(text)
}

fn token_count(tokenizer: &Gpt2Tokenizer, text: &str) -> usize {
    let count = token_ids(tokenizer, text);
    count.len()
}

#[cfg(test)]
mod tests {
    use training::tokenizer::Gpt2Tokenizer;
    use crate::{token_count, token_ids};

    #[test]
    fn test_ch01() {
        let tokenizer = Gpt2Tokenizer::new();
        for text in ["Hello, world!"] {
            let count = token_count(&tokenizer, text);
            assert_eq!(count, 4);
        }
    }

    #[test]
    fn test_ch02() {
        let tokenizer = Gpt2Tokenizer::new();
        for text in ["Hello world"] {
            let count = token_count(&tokenizer, text);
            assert_eq!(count, 2);
        }
    }

    #[test]
    fn leerer_text_hat_keine_tokens() {
        let tokenizer = Gpt2Tokenizer::new();
        let count = token_count(&tokenizer, "");
        assert_eq!(count, 0);
    }

    #[test]
    fn roundtrip_erhaelt_umlaute() {
        let tokenizer = Gpt2Tokenizer::new();
        let text = "Grüezi";
        let ids = token_ids(&tokenizer, &text);
        let roundtrip = tokenizer.decode(&ids).expect("gültiges UTF-8");
        assert_eq!(text, roundtrip);
    }

    #[test]
    fn endoftext_hat_eine_token_id() {
        let tokenizer = Gpt2Tokenizer::new();
        let text = "<|endoftext|>";
        let ids = token_ids(&tokenizer, &text);
        assert_eq!(ids, [training::tokenizer::END_OF_TEXT_ID]);
    }

}