//! Kapitel 16: Verteilungen und Generierung mit Temperatur, Top-k und Top-p.

use std::path::Path;

use burn::tensor::Device;
use model::gpt::{GptModel, GptModelConfig};
use training::{
    checkpoint::load_model,
    generation::{GenerationOptions, generate_with_options},
    sampling::{SamplingFilter, SamplingStrategy},
    tokenizer::{END_OF_TEXT_ID, Gpt2Tokenizer, VOCAB_SIZE},
};

type Demo = Result<(), Box<dyn std::error::Error>>;

const CONTEXT_LENGTH: usize = 8;
const EMBEDDING_DIM: usize = 8;
const NUM_HEADS: usize = 2;
const NUM_BLOCKS: usize = 1;
const MAX_NEW_TOKENS: usize = 64;
const USAGE: &str = "Aufruf: ch16 [<checkpoint.bpk> <starttext> <neue_tokens:1..64> <seed>]";

// ANCHOR: variants
const VARIANTS: [(&str, SamplingStrategy); 5] = [
    ("argmax", SamplingStrategy::Greedy),
    (
        "Temperatur 0.5",
        SamplingStrategy::Multinomial {
            temperature: 0.5,
            filter: SamplingFilter::None,
        },
    ),
    (
        "Temperatur 1.5",
        SamplingStrategy::Multinomial {
            temperature: 1.5,
            filter: SamplingFilter::None,
        },
    ),
    (
        "Top-k 3, Temperatur 1",
        SamplingStrategy::Multinomial {
            temperature: 1.0,
            filter: SamplingFilter::TopK(3),
        },
    ),
    (
        "Top-p 0.9, Temperatur 1",
        SamplingStrategy::Multinomial {
            temperature: 1.0,
            filter: SamplingFilter::TopP(0.9),
        },
    ),
];
// ANCHOR_END: variants

fn show_probabilities() -> Demo {
    let logits = [3.0, 2.0, 1.0, 0.0, -1.0];
    println!("Feste Logits für Token-IDs 0 bis 4: {logits:?}");
    for (name, strategy) in VARIANTS {
        let probabilities = strategy.probabilities(&logits)?;
        let display: Vec<_> = probabilities.iter().map(|p| format!("{p:.4}")).collect();
        println!("{name}: [{}]", display.join(", "));
    }
    Ok(())
}

fn config() -> GptModelConfig {
    GptModelConfig::new(
        VOCAB_SIZE,
        CONTEXT_LENGTH,
        EMBEDDING_DIM,
        NUM_HEADS,
        NUM_BLOCKS,
    )
}

fn prompt_ids(prompt: &str) -> Result<Vec<i32>, Box<dyn std::error::Error>> {
    let ids = Gpt2Tokenizer::new().encode(prompt);
    if ids.is_empty() {
        return Err("Starttext darf nicht leer sein".into());
    }
    Ok(ids
        .into_iter()
        .map(i32::try_from)
        .collect::<Result<_, _>>()?)
}

fn check_count(count: usize) -> Demo {
    if count == 0 || count > MAX_NEW_TOKENS {
        return Err(format!("Neue Tokens müssen zwischen 1 und {MAX_NEW_TOKENS} liegen").into());
    }
    Ok(())
}

// ANCHOR: compare
fn compare(model: &GptModel, prompt: &[i32], count: usize, seed: u64, device: &Device) -> Demo {
    for (name, strategy) in VARIANTS {
        let options = GenerationOptions {
            strategy,
            seed,
            max_new_tokens: count,
            stop_token: Some(i32::try_from(END_OF_TEXT_ID)?),
        };
        let ids = generate_with_options(model, prompt.to_vec(), CONTEXT_LENGTH, options, device)?;
        report(name, prompt.len(), ids)?;
    }
    Ok(())
}
// ANCHOR_END: compare

fn report(name: &str, prompt_length: usize, ids: Vec<i32>) -> Demo {
    let ids: Vec<u32> = ids
        .into_iter()
        .map(u32::try_from)
        .collect::<Result<_, _>>()?;
    println!("\n=== {name} ===");
    println!("Neue Tokens: {}", ids.len() - prompt_length);
    let text = Gpt2Tokenizer::new().decode(&ids[prompt_length..]);
    match &text {
        Ok(text) => println!("\nVom Modell erzeugte Fortsetzung:\n{text}"),
        Err(_) => println!("\nFortsetzung konnte nicht decodiert werden."),
    }
    println!("\nToken-IDs (Starttext + Fortsetzung):\n{ids:?}");
    text?;
    Ok(())
}

fn compare_checkpoint(path: &Path, prompt: &str, count: usize, seed: u64) -> Demo {
    check_count(count)?;
    let ids = prompt_ids(prompt)?;
    let device = Device::flex();
    let model = load_model(&config(), path, &device)?;
    println!("Gleiche Gewichte, gleicher Prompt; kein Training, keine Qualitätsgarantie.");
    println!("Checkpoint: {}", path.display());
    println!("Dein Starttext: {prompt:?}");
    println!("Seed: {seed}");
    println!("Maximal neue Tokens je Strategie: {count}");
    compare(&model, &ids, count, seed, &device)
}

fn main() -> Demo {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [] => show_probabilities(),
        [checkpoint, prompt, count, seed] => {
            compare_checkpoint(Path::new(checkpoint), prompt, count.parse()?, seed.parse()?)
        }
        _ => Err(USAGE.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use training::checkpoint::save_model;

    #[test]
    fn checkpoint_roundtrip_preserves_sampled_output() -> Demo {
        let device = Device::flex();
        let config = GptModelConfig::new(5, CONTEXT_LENGTH, EMBEDDING_DIM, NUM_HEADS, NUM_BLOCKS);
        let model = config.init(&device);
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("model.bpk");
        save_model(&model, &path)?;
        let loaded = load_model(&config, &path, &device)?;
        for (_, strategy) in VARIANTS {
            let options = GenerationOptions {
                strategy,
                seed: 42,
                max_new_tokens: 6,
                stop_token: None,
            };
            let run = |model| {
                generate_with_options(model, vec![0, 1, 2], CONTEXT_LENGTH, options, &device)
            };
            assert_eq!(run(&model)?, run(&loaded)?);
        }
        Ok(())
    }

    #[test]
    fn checkpoint_demo_rejects_empty_prompt_limits_and_missing_file() {
        assert!(prompt_ids("").is_err());
        assert!(check_count(0).is_err());
        assert!(check_count(MAX_NEW_TOKENS + 1).is_err());
        let directory = tempfile::tempdir().unwrap();
        assert!(compare_checkpoint(&directory.path().join("missing.bpk"), "Hello", 1, 42).is_err());
    }
}
