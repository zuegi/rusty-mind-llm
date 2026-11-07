//! Kapitel 12: Ganze Epochen und Generierung mit GPT-2-IDs.

use std::{path::Path, time::Instant};

use burn::{module::Module, optim::SgdConfig, tensor::Device};
use model::gpt::{GptModel, GptModelConfig};
use training::{
    checkpoint::{load_model, save_model},
    dataset::{Dataset, split_tokens},
    generation::generate,
    runner::{evaluate, train_epoch},
    tokenizer::{Gpt2Tokenizer, VOCAB_SIZE},
};

type Demo = Result<(), Box<dyn std::error::Error>>;

const CONTEXT_LENGTH: usize = 8;
const EMBEDDING_DIM: usize = 8;
const NUM_HEADS: usize = 2;
const NUM_BLOCKS: usize = 1;
const BATCH_SIZE: usize = 2;
const MAX_EPOCHS: usize = 3;
const MAX_NEW_TOKENS: usize = 64;
const VALIDATION_FRACTION: f64 = 0.1;
const LEARNING_RATE: f64 = 0.05;
const SEED: u64 = 42;
const DEFAULT_TEXT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/the-verdict.txt");
const USAGE: &str = "Aufruf: ch12_full train <checkpoint.bpk> <epochen:1..3> [textdatei] | generate <checkpoint.bpk> <starttext> <neue_tokens:1..64>";

fn config() -> GptModelConfig {
    GptModelConfig::new(
        VOCAB_SIZE,
        CONTEXT_LENGTH,
        EMBEDDING_DIM,
        NUM_HEADS,
        NUM_BLOCKS,
    )
}

// ANCHOR: full_data
fn datasets(text: &str) -> Result<(Dataset, Dataset), Box<dyn std::error::Error>> {
    let tokens = Gpt2Tokenizer::new().encode(text);
    let (train, validation) = split_tokens(&tokens, VALIDATION_FRACTION);
    Ok((
        Dataset::new(train, CONTEXT_LENGTH)?,
        Dataset::new(validation, CONTEXT_LENGTH)?,
    ))
}
// ANCHOR_END: full_data

fn report(model: &GptModel, train: &Dataset, validation: &Dataset, epoch: usize) {
    let fixed = model.valid();
    let device = Device::flex();
    let train_loss = evaluate(&fixed, train, BATCH_SIZE, &device);
    let validation_loss = evaluate(&fixed, validation, BATCH_SIZE, &device);
    println!(
        "Epoche {epoch}: Trainings-Loss {train_loss:.6}, Validierungs-Loss {validation_loss:.6}"
    );
}

fn save(model: &GptModel, path: &Path) -> Demo {
    if let Some(parent) = path.parent().filter(|path| !path.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let mut pending_name = path.as_os_str().to_os_string();
    pending_name.push(format!(".{}.pending.bpk", std::process::id()));
    let pending = Path::new(&pending_name);
    save_model(&model.valid(), pending)?;
    std::fs::rename(pending, path)?;
    println!("Modellgewichte gespeichert: {}", path.display());
    Ok(())
}

// ANCHOR: full_training
fn train_and_save(
    text: &str,
    path: &Path,
    epochs: usize,
) -> Result<GptModel, Box<dyn std::error::Error>> {
    check_limit(epochs, MAX_EPOCHS)?;
    let (train, validation) = datasets(text)?;
    let device = Device::flex().autodiff();
    device.seed(SEED);
    let mut model = config().init(&device);
    let mut optimizer = SgdConfig::new().init();
    describe(&train, &validation, epochs)?;
    report(&model, &train, &validation, 0);
    let started = Instant::now();
    for epoch in 1..=epochs {
        let batches = train.epoch(BATCH_SIZE, SEED + epoch as u64);
        let (updated, steps) = train_epoch(model, &batches, LEARNING_RATE, &mut optimizer, &device);
        model = updated;
        report(&model, &train, &validation, epoch);
        save(&model, path)?;
        report_epoch(epoch, steps, &started);
    }
    Ok(model.valid())
}
// ANCHOR_END: full_training

fn report_epoch(epoch: usize, steps: usize, started: &Instant) {
    println!(
        "Epoche {epoch} abgeschlossen: {steps} Updates, Gesamtzeit {:.2}s",
        started.elapsed().as_secs_f64()
    );
}

fn describe(train: &Dataset, validation: &Dataset, epochs: usize) -> Demo {
    let batches = train.len() / BATCH_SIZE;
    if batches == 0 {
        return Err("Keine vollständigen Trainingsbatches: mehr Text nötig".into());
    }
    println!(
        "Fenster: Training {}, Validierung {}",
        train.len(),
        validation.len()
    );
    println!(
        "{epochs} Epochen, {batches} Updates pro Epoche, {} Updates insgesamt",
        epochs * batches
    );
    println!(
        "Volle Epochen: kein Zeitabbruch. Abbrechen mit Ctrl+C; letzte abgeschlossene Epoche bleibt gespeichert."
    );
    Ok(())
}

fn check_limit(value: usize, maximum: usize) -> Demo {
    if value == 0 || value > maximum {
        return Err(format!("Wert muss zwischen 1 und {maximum} liegen, erhalten: {value}").into());
    }
    Ok(())
}

// ANCHOR: text_generation
fn generated_ids(
    model: &GptModel,
    prompt: &str,
    count: usize,
    device: &Device,
) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    check_limit(count, MAX_NEW_TOKENS)?;
    let ids = Gpt2Tokenizer::new().encode(prompt);
    if ids.is_empty() {
        return Err("Starttext darf nicht leer sein".into());
    }
    let input = ids
        .into_iter()
        .map(i32::try_from)
        .collect::<Result<Vec<_>, _>>()?;
    generate(model, input, count, CONTEXT_LENGTH, device)
        .into_iter()
        .map(u32::try_from)
        .collect::<Result<Vec<_>, _>>()
        .map_err(Into::into)
}

fn generate_text(path: &Path, prompt: &str, count: usize) -> Demo {
    let device = Device::flex();
    let model = load_model(&config(), path, &device)?;
    let ids = generated_ids(&model, prompt, count, &device)?;
    println!("Token-IDs (Starttext + Fortsetzung): {ids:?}");
    let text = Gpt2Tokenizer::new().decode(&ids)?;
    println!("Text (keine Qualitätsgarantie):\n{text}");
    Ok(())
}
// ANCHOR_END: text_generation

fn train_file(text: &str, checkpoint: &str, epochs: &str) -> Demo {
    let epochs = epochs.parse()?;
    check_limit(epochs, MAX_EPOCHS)?;
    train_and_save(
        &std::fs::read_to_string(text)?,
        Path::new(checkpoint),
        epochs,
    )?;
    Ok(())
}

fn main() -> Demo {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [mode, checkpoint, epochs] if mode == "train" => {
            train_file(DEFAULT_TEXT, checkpoint, epochs)
        }
        [mode, checkpoint, epochs, text] if mode == "train" => train_file(text, checkpoint, epochs),
        [mode, checkpoint, prompt, count] if mode == "generate" => {
            generate_text(Path::new(checkpoint), prompt, count.parse()?)
        }
        _ => Err(USAGE.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_text_uses_all_split_windows() -> Demo {
        let text = std::fs::read_to_string(DEFAULT_TEXT)?;
        let (train, validation) = datasets(&text)?;
        assert_eq!((train.len(), validation.len()), (578, 64));
        assert_eq!(train.epoch(BATCH_SIZE, SEED).len(), 289);
        Ok(())
    }

    #[test]
    fn rejects_unbounded_runs_and_empty_prompts() {
        for value in [0, MAX_EPOCHS + 1] {
            assert!(check_limit(value, MAX_EPOCHS).is_err());
        }
        let device = Device::flex();
        let model = config().init(&device);
        assert!(generated_ids(&model, "", 1, &device).is_err());
        assert!(generated_ids(&model, "Hello", MAX_NEW_TOKENS + 1, &device).is_err());
    }

    #[test]
    fn tiny_corpus_epoch_checkpoint_and_generation_roundtrip() -> Demo {
        let text = "Hello world. ".repeat(32);
        let path = std::env::temp_dir().join(format!(
            "rusty-mind-llm-ch12-full-{}.bpk",
            std::process::id()
        ));
        let trained = train_and_save(&text, &path, 1)?;
        let device = Device::flex();
        let loaded = load_model(&config(), &path, &device)?;
        let first = generated_ids(&loaded, "Hello world. Hello world.", 2, &device)?;
        assert_eq!(
            first,
            generated_ids(&trained, "Hello world. Hello world.", 2, &device)?
        );
        let reloaded = load_model(&config(), &path, &device)?;
        assert_eq!(
            first,
            generated_ids(&reloaded, "Hello world. Hello world.", 2, &device)?
        );
        assert!(first.iter().all(|&id| (id as usize) < VOCAB_SIZE));
        assert_eq!(
            first.len(),
            Gpt2Tokenizer::new()
                .encode("Hello world. Hello world.")
                .len()
                + 2
        );
        std::fs::remove_file(path)?;
        Ok(())
    }

    #[test]
    fn incompatible_checkpoint_is_an_error() -> Demo {
        let device = Device::flex();
        let tiny = GptModelConfig::new(5, CONTEXT_LENGTH, EMBEDDING_DIM, NUM_HEADS, NUM_BLOCKS)
            .init(&device);
        let path = std::env::temp_dir().join(format!(
            "rusty-mind-llm-ch12-incompatible-{}.bpk",
            std::process::id()
        ));
        save_model(&tiny, &path)?;
        let result = load_model(&config(), &path, &device);
        std::fs::remove_file(path)?;
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn loaded_weights_generate_decodable_text() -> Demo {
        let device = Device::flex();
        let mut model = config().init(&device);
        model.output_projection.weight = burn::module::Param::from_tensor(
            burn::tensor::Tensor::<2>::zeros([EMBEDDING_DIM, VOCAB_SIZE], &device),
        );
        let path = std::env::temp_dir().join(format!(
            "rusty-mind-llm-ch12-decode-{}.bpk",
            std::process::id()
        ));
        save_model(&model, &path)?;
        let loaded = load_model(&config(), &path, &device)?;
        let ids = generated_ids(&loaded, "Hello", 2, &device)?;
        assert_eq!(Gpt2Tokenizer::new().decode(&ids)?, "Hello!!");
        generate_text(&path, "Hello", 2)?;
        std::fs::remove_file(&path)?;
        assert!(generate_text(&path, "Hello", 2).is_err());
        Ok(())
    }
}
