//! Kapitel 2: Vom Token-Strom zu Trainingsbatches.
//! Ausführen: cargo run -p training --example ch02 [-- <textdatei>]

use std::error::Error;

use training::dataset::{Dataset, split_tokens};
use training::tokenizer::Gpt2Tokenizer;

const BEISPIELTEXT: &str = "Der Hund läuft durch den Park. Die Katze schläft auf dem Sofa. \
Am Abend regnet es, und alle bleiben zu Hause. Morgen scheint wieder die Sonne.";
const WIEDERHOLUNGEN: usize = 6;
const CONTEXT_LENGTH: usize = 8;
const BATCH_SIZE: usize = 2;
const VALIDATION_FRACTION: f64 = 0.1;
const SEED: u64 = 42;

fn main() -> Result<(), Box<dyn Error>> {
    let text = match std::env::args().nth(1) {
        Some(path) => std::fs::read_to_string(path)?,
        None => BEISPIELTEXT.repeat(WIEDERHOLUNGEN),
    };
    let tokenizer = Gpt2Tokenizer::new();
    let tokens = tokenizer.encode(&text);

    let (train, validation) = split_tokens(&tokens, VALIDATION_FRACTION);
    println!("Token gesamt: {}, Training: {}, Validierung: {}", tokens.len(), train.len(), validation.len());

    let dataset = Dataset::new(train, CONTEXT_LENGTH)?;
    println!("Fenster (Länge {CONTEXT_LENGTH}): {}\n", dataset.len());

    let window = dataset.window(0);
    println!("Fenster 0, Eingabe: {:?}", tokenizer.decode(&window.input)?);
    println!("Fenster 0, Ziel:    {:?}\n", tokenizer.decode(&window.target)?);

    let batches = dataset.epoch(BATCH_SIZE, SEED);
    println!("Batches pro Epoche (Grösse {BATCH_SIZE}): {}", batches.len());
    println!("Erster Batch, Eingabe: {:?}", batches[0].inputs);
    println!("Erster Batch, Ziel:    {:?}", batches[0].targets);
    Ok(())
}
