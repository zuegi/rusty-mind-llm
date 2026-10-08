//! Kapitel 1: Was macht der GPT-2-Tokenizer mit Text?
//! Ausführen: cargo run -p training --example ch01

use training::tokenizer::{Gpt2Tokenizer, VOCAB_SIZE};

const BEISPIELE: [&str; 6] = [
    "Hello world",
    "The quick brown fox jumps over the lazy dog.",
    "Grüezi mit Umlauten: äöü",
    "Emoji: 🙂",
    "Antidisestablishmentarianism",
    "Ende<|endoftext|>",
];

fn main() {
    let tokenizer = Gpt2Tokenizer::new();
    println!("Vokabulargrösse: {VOCAB_SIZE}\n");
    for text in BEISPIELE {
        zeige(&tokenizer, text);
    }
}

fn zeige(tokenizer: &Gpt2Tokenizer, text: &str) {
    let ids = tokenizer.encode(text);
    let roundtrip = tokenizer.decode(&ids).expect("gültiges UTF-8");
    println!("Text:      {text:?}");
    println!("IDs:       {ids:?}");
    println!("Stücke:    {:?}", tokenizer.pieces(text));
    println!(
        "Zeichen/Bytes/Token: {}/{}/{}",
        text.chars().count(),
        text.len(),
        ids.len()
    );
    println!("Roundtrip: {}\n", roundtrip == text);
}
