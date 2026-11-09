//! Kapitel 13: Epochen messen, Perplexity berechnen, besten Stand speichern.
//! Ausführen: cargo run --release -p training --example ch13 -- <checkpoint.bpk> <epochen:1..20> [textdatei]

use std::path::Path;

use burn::{module::Module, optim::SgdConfig, tensor::Device};
use model::gpt::{GptModel, GptModelConfig};
use training::{
    checkpoint::save_model,
    dataset::{Dataset, split_tokens},
    runner::{evaluate, train_epoch},
    tokenizer::{Gpt2Tokenizer, VOCAB_SIZE},
};

type Demo = Result<(), Box<dyn std::error::Error>>;

const CONTEXT_LENGTH: usize = 8;
const EMBEDDING_DIM: usize = 8;
const NUM_HEADS: usize = 2;
const NUM_BLOCKS: usize = 1;
const BATCH_SIZE: usize = 2;
const MAX_EPOCHS: usize = 20;
const VALIDATION_FRACTION: f64 = 0.1;
const LEARNING_RATE: f64 = 0.05;
const SEED: u64 = 42;
const DEFAULT_TEXT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/the-verdict.txt");
const USAGE: &str =
    "Aufruf: ch13 <bester_checkpoint.bpk> <epochen:1..20> [textdatei] [--learning-rate <rate>]";

#[derive(Debug, Clone, Copy, PartialEq)]
struct EpochResult {
    epoch: usize,
    train_loss: f64,
    validation_loss: f64,
}

impl EpochResult {
    fn train_perplexity(&self) -> f64 {
        self.train_loss.exp()
    }

    fn validation_perplexity(&self) -> f64 {
        self.validation_loss.exp()
    }
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

fn datasets(text: &str) -> Result<(Dataset, Dataset), Box<dyn std::error::Error>> {
    let tokens = Gpt2Tokenizer::new().encode(text);
    let (train, validation) = split_tokens(&tokens, VALIDATION_FRACTION);
    Ok((
        Dataset::new(train, CONTEXT_LENGTH)?,
        Dataset::new(validation, CONTEXT_LENGTH)?,
    ))
}

fn measure(model: &GptModel, train: &Dataset, validation: &Dataset, epoch: usize) -> EpochResult {
    let fixed = model.valid();
    let device = Device::flex();
    EpochResult {
        epoch,
        train_loss: evaluate(&fixed, train, BATCH_SIZE, &device),
        validation_loss: evaluate(&fixed, validation, BATCH_SIZE, &device),
    }
}

// ANCHOR: best_rule
/// Ersetzt den besten Stand nur bei strikt kleinerem Validierungs-Loss.
fn is_new_best(best: Option<&EpochResult>, candidate: &EpochResult) -> bool {
    best.is_none_or(|best| candidate.validation_loss < best.validation_loss)
}
// ANCHOR_END: best_rule

// ANCHOR: overfitting_signal
/// Warnsignal: Training wurde seit dem besten Stand besser, Validierung nicht.
fn shows_overfitting_signal(best: &EpochResult, last: &EpochResult) -> bool {
    last.train_loss < best.train_loss && last.validation_loss > best.validation_loss
}
// ANCHOR_END: overfitting_signal

fn save_atomically(model: &GptModel, path: &Path) -> Demo {
    if let Some(parent) = path.parent().filter(|path| !path.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let mut pending_name = path.as_os_str().to_os_string();
    pending_name.push(format!(".{}.pending.bpk", std::process::id()));
    let pending = Path::new(&pending_name);
    save_model(&model.valid(), pending)?;
    std::fs::rename(pending, path)?;
    Ok(())
}

fn print_result(result: &EpochResult, saved: bool) {
    let marker = if saved { "  <- neuer bester Stand" } else { "" };
    println!(
        "Epoche {:>2}: Training {:.4} (PPL {:.1}), Validierung {:.4} (PPL {:.1}){marker}",
        result.epoch,
        result.train_loss,
        result.train_perplexity(),
        result.validation_loss,
        result.validation_perplexity(),
    );
}

fn check_epochs(epochs: usize) -> Demo {
    if epochs == 0 || epochs > MAX_EPOCHS {
        return Err(format!(
            "Epochen müssen zwischen 1 und {MAX_EPOCHS} liegen, erhalten: {epochs}"
        )
        .into());
    }
    Ok(())
}

fn check_learning_rate(rate: f64) -> Demo {
    if !rate.is_finite() || rate <= 0.0 {
        return Err("Lernrate muss endlich und positiv sein".into());
    }
    Ok(())
}

fn save_if_best(
    model: &GptModel,
    result: EpochResult,
    best: &mut Option<EpochResult>,
    path: &Path,
) -> Result<bool, Box<dyn std::error::Error>> {
    let improved = is_new_best(best.as_ref(), &result);
    if improved {
        save_atomically(model, path)?;
        *best = Some(result);
    }
    Ok(improved)
}

fn run(
    text: &str,
    path: &Path,
    epochs: usize,
    learning_rate: f64,
) -> Result<Vec<EpochResult>, Box<dyn std::error::Error>> {
    check_epochs(epochs)?;
    check_learning_rate(learning_rate)?;
    let (train, validation) = datasets(text)?;
    if train.len() < BATCH_SIZE {
        return Err("Keine vollständigen Trainingsbatches: mehr Text nötig".into());
    }
    let device = Device::flex().autodiff();
    device.seed(SEED);
    let model = config().init(&device);
    train_epochs(
        model,
        &train,
        &validation,
        path,
        epochs,
        learning_rate,
        &device,
    )
}

// ANCHOR: training_run
fn train_epochs(
    mut model: GptModel,
    train: &Dataset,
    validation: &Dataset,
    path: &Path,
    epochs: usize,
    learning_rate: f64,
    device: &Device,
) -> Result<Vec<EpochResult>, Box<dyn std::error::Error>> {
    let mut optimizer = SgdConfig::new().init();
    let mut history = vec![measure(&model, train, validation, 0)];
    let mut best = None;
    save_if_best(&model, history[0], &mut best, path)?;
    print_result(&history[0], true);
    for epoch in 1..=epochs {
        let batches = train.epoch(BATCH_SIZE, SEED + epoch as u64);
        (model, _) = train_epoch(model, &batches, learning_rate, &mut optimizer, device);
        let result = measure(&model, train, validation, epoch);
        let improved = save_if_best(&model, result, &mut best, path)?;
        print_result(&result, improved);
        history.push(result);
    }
    Ok(history)
}
// ANCHOR_END: training_run

fn best_of(history: &[EpochResult]) -> Option<&EpochResult> {
    history.iter().fold(None, |best, result| {
        if is_new_best(best, result) {
            Some(result)
        } else {
            best
        }
    })
}

fn summarize(history: &[EpochResult], path: &Path) {
    let best = best_of(history).expect("Ausgangsmessung vorhanden");
    let last = history.last().expect("Ausgangsmessung vorhanden");
    println!(
        "\nBester Stand: Epoche {} (Validierung {:.4}, PPL {:.1}) in {}",
        best.epoch,
        best.validation_loss,
        best.validation_perplexity(),
        path.display()
    );
    if shows_overfitting_signal(best, last) {
        println!(
            "Warnsignal: Seit Epoche {} wurde Training besser, Validierung schlechter.",
            best.epoch
        );
    } else {
        println!("Das geprüfte Warnsignal liegt nicht vor; das schliesst Overfitting nicht aus.");
    }
}

fn main() -> Demo {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (checkpoint, epochs, text, rate) = match args.as_slice() {
        [checkpoint, epochs] => (checkpoint, epochs, DEFAULT_TEXT, LEARNING_RATE),
        [checkpoint, epochs, flag, rate] if flag == "--learning-rate" => {
            (checkpoint, epochs, DEFAULT_TEXT, rate.parse()?)
        }
        [checkpoint, epochs, text, flag, rate] if flag == "--learning-rate" => {
            (checkpoint, epochs, text.as_str(), rate.parse()?)
        }
        [checkpoint, epochs, text] => (checkpoint, epochs, text.as_str(), LEARNING_RATE),
        _ => return Err(USAGE.into()),
    };
    let epochs = epochs.parse()?;
    check_epochs(epochs)?;
    check_learning_rate(rate)?;
    println!("Text: {text}, Epochen: {epochs}, Lernrate: {rate}, Seed: {SEED}");
    let path = Path::new(checkpoint);
    let history = run(&std::fs::read_to_string(text)?, path, epochs, rate)?;
    summarize(&history, path);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use training::checkpoint::load_model;

    fn result(epoch: usize, train_loss: f64, validation_loss: f64) -> EpochResult {
        EpochResult {
            epoch,
            train_loss,
            validation_loss,
        }
    }

    #[test]
    fn perplexity_is_exp_of_loss() {
        let r = result(1, 0.0, 0.7);
        assert_eq!(r.train_perplexity(), 1.0);
        assert!((r.validation_perplexity() - 2.0138).abs() < 1e-3);
    }

    #[test]
    fn only_strict_improvement_replaces_best() {
        let best = result(2, 3.5, 3.8);
        assert!(is_new_best(None, &best));
        assert!(is_new_best(Some(&best), &result(3, 3.0, 3.6)));
        assert!(!is_new_best(Some(&best), &result(3, 3.0, 4.1)));
        assert!(!is_new_best(Some(&best), &result(3, 3.0, 3.8)));
    }

    #[test]
    fn overfitting_signal_needs_better_training_and_worse_validation() {
        let best = result(2, 3.5, 3.8);
        assert!(shows_overfitting_signal(&best, &result(3, 3.0, 4.1)));
        assert!(!shows_overfitting_signal(&best, &result(3, 3.0, 3.6)));
        assert!(!shows_overfitting_signal(&best, &result(3, 3.6, 4.1)));
    }

    #[test]
    fn rejects_epochs_outside_limit() {
        assert!(check_epochs(0).is_err());
        assert!(check_epochs(MAX_EPOCHS + 1).is_err());
        assert!(check_epochs(MAX_EPOCHS).is_ok());
    }

    #[test]
    fn rejects_invalid_learning_rates() {
        for rate in [0.0, -0.1, f64::NAN, f64::INFINITY] {
            assert!(check_learning_rate(rate).is_err());
        }
        assert!(check_learning_rate(0.02).is_ok());
    }

    #[test]
    fn best_includes_baseline_and_keeps_first_tie() {
        let history = [
            result(0, 4.0, 4.0),
            result(1, 3.5, 4.1),
            result(2, 3.0, 4.0),
        ];
        assert_eq!(best_of(&history), Some(&history[0]));
        assert!(!shows_overfitting_signal(&history[0], &result(3, 4.1, 4.2)));
    }

    #[test]
    fn worse_or_equal_loss_keeps_checkpoint_bytes() -> Demo {
        let device = Device::flex();
        let config = GptModelConfig::new(5, CONTEXT_LENGTH, EMBEDDING_DIM, NUM_HEADS, NUM_BLOCKS);
        let model = config.init(&device);
        let other = config.init(&device);
        let path = std::env::temp_dir().join(format!(
            "rusty-mind-llm-ch13-selection-{}.bpk",
            std::process::id()
        ));
        let mut best = None;
        assert!(save_if_best(&model, result(0, 4.0, 4.0), &mut best, &path)?);
        let original = std::fs::read(&path)?;
        for candidate in [result(1, 3.0, 4.1), result(2, 2.0, 4.0)] {
            assert!(!save_if_best(&other, candidate, &mut best, &path)?);
            assert_eq!(std::fs::read(&path)?, original);
        }
        assert_eq!(best, Some(result(0, 4.0, 4.0)));
        assert!(save_if_best(&other, result(3, 2.0, 3.9), &mut best, &path)?);
        let loaded = load_model(&config, &path, &device)?;
        let data = Dataset::new(&[0; CONTEXT_LENGTH + 1], CONTEXT_LENGTH)?;
        assert_eq!(
            evaluate(&loaded, &data, 1, &device),
            evaluate(&other, &data, 1, &device)
        );
        std::fs::remove_file(path)?;
        Ok(())
    }

    #[test]
    fn saved_checkpoint_matches_best_validation_loss() -> Demo {
        let text = "Hello world. ".repeat(64);
        let path =
            std::env::temp_dir().join(format!("rusty-mind-llm-ch13-{}.bpk", std::process::id()));
        let history = run(&text, &path, 3, LEARNING_RATE)?;
        let best = best_of(&history).expect("Ausgangsmessung");
        let (_, validation) = datasets(&text)?;
        let loaded = load_model(&config(), &path, &Device::flex())?;
        let loss = evaluate(&loaded, &validation, BATCH_SIZE, &Device::flex());
        assert!((loss - best.validation_loss).abs() < 1e-6);
        std::fs::remove_file(path)?;
        Ok(())
    }

    #[test]
    fn needs_complete_training_batch() {
        let path = std::env::temp_dir().join("rusty-mind-llm-ch13-never.bpk");
        assert!(run("Hello world", &path, 1, LEARNING_RATE).is_err());
        assert!(!path.exists());
    }
}
