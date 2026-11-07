//! Kapitel 12: Begrenzter CPU-Trainingsversuch mit echtem Text.
//! Ausführen: cargo run -p training --example ch12 [-- <textdatei>]

use std::time::{Duration, Instant};

use burn::module::Module;
use burn::optim::{ModuleOptimizer, SgdConfig};
use burn::tensor::Device;
use model::gpt::{GptModel, GptModelConfig};
use training::dataset::{Dataset, split_tokens};
use training::runner::evaluate;
use training::step::train_step;
use training::tokenizer::{Gpt2Tokenizer, VOCAB_SIZE};

type Demo = Result<(), Box<dyn std::error::Error>>;

const CONTEXT_LENGTH: usize = 8;
const EMBEDDING_DIM: usize = 8;
const NUM_HEADS: usize = 2;
const NUM_BLOCKS: usize = 1;
const BATCH_SIZE: usize = 2;
const TRAIN_WINDOWS: usize = 8;
const VALIDATION_WINDOWS: usize = 2;
const MAX_UPDATES: usize = 3;
const TRAINING_TIME_LIMIT: Duration = Duration::from_secs(30);
const VALIDATION_FRACTION: f64 = 0.1;
const LEARNING_RATE: f64 = 0.05;
const SEED: u64 = 42;
const BYTES_PER_F32: usize = std::mem::size_of::<f32>();

// ANCHOR: data
fn datasets(text: &str) -> Result<(Dataset, Dataset), Box<dyn std::error::Error>> {
    let tokens = Gpt2Tokenizer::new().encode(text);
    let (train, validation) = split_tokens(&tokens, VALIDATION_FRACTION);
    println!(
        "Token gesamt: {}, Training: {}, Validierung: {}",
        tokens.len(),
        train.len(),
        validation.len()
    );
    Ok((
        limited_dataset(train, TRAIN_WINDOWS)?,
        limited_dataset(validation, VALIDATION_WINDOWS)?,
    ))
}

fn limited_dataset(
    tokens: &[u32],
    max_windows: usize,
) -> Result<Dataset, training::dataset::DatasetError> {
    let count = tokens.len().min(max_windows * CONTEXT_LENGTH + 1);
    Dataset::new(&tokens[..count], CONTEXT_LENGTH)
}
// ANCHOR_END: data

fn model(device: &Device) -> GptModel {
    device.seed(SEED);
    GptModelConfig::new(
        VOCAB_SIZE,
        CONTEXT_LENGTH,
        EMBEDDING_DIM,
        NUM_HEADS,
        NUM_BLOCKS,
    )
    .init(device)
}

fn describe(train: &Dataset, validation: &Dataset) {
    println!(
        "Verwendete Fenster: Training {}, Validierung {}",
        train.len(),
        validation.len()
    );
    println!(
        "Vokabular: {VOCAB_SIZE}, Logits pro vollem Batch: [{BATCH_SIZE}, {CONTEXT_LENGTH}, {VOCAB_SIZE}]"
    );
    let values = BATCH_SIZE * CONTEXT_LENGTH * VOCAB_SIZE;
    println!(
        "Logits: {values} f32-Werte, {} Bytes (nicht gesamter RAM-Bedarf)",
        values * BYTES_PER_F32
    );
    println!(
        "Budget: höchstens {MAX_UPDATES} Updates, {} Sekunden zwischen Updates geprüft",
        TRAINING_TIME_LIMIT.as_secs()
    );
}

// ANCHOR: report
fn report(model: &GptModel, train: &Dataset, validation: &Dataset, label: &str) {
    let device = Device::flex();
    let fixed = model.valid();
    let train_loss = evaluate(&fixed, train, BATCH_SIZE, &device);
    let validation_loss = evaluate(&fixed, validation, BATCH_SIZE, &device);
    println!("{label}: Trainings-Loss {train_loss:.6}, Validierungs-Loss {validation_loss:.6}");
}
// ANCHOR_END: report

// ANCHOR: bounded_training
fn train(
    mut model: GptModel,
    train: &Dataset,
    device: &Device,
) -> Result<(GptModel, usize), Box<dyn std::error::Error>> {
    let batches = train.epoch(BATCH_SIZE, SEED);
    if batches.is_empty() {
        return Err("Keine vollständigen Trainingsbatches: mehr Text nötig".into());
    }
    let mut optimizer = SgdConfig::new().init();
    let started = Instant::now();
    let mut updates = 0;
    for (index, batch) in batches.iter().take(MAX_UPDATES).enumerate() {
        if !within_time_budget(&started, index + 1) {
            break;
        }
        let (updated, loss) = update(model, batch, &mut optimizer, device)?;
        model = updated;
        updates += 1;
        report_update(index + 1, loss, &started);
    }
    Ok((model, updates))
}
// ANCHOR_END: bounded_training

fn within_time_budget(started: &Instant, next_update: usize) -> bool {
    if started.elapsed() >= TRAINING_TIME_LIMIT {
        println!("Zeitbudget erreicht: Abbruch vor Update {next_update}");
        return false;
    }
    true
}

fn report_update(index: usize, loss: f32, started: &Instant) {
    println!(
        "Update {index}: Loss vor Update {loss:.6}, Trainingszeit {:.2}s",
        started.elapsed().as_secs_f64()
    );
}

fn update(
    model: GptModel,
    batch: &training::dataset::Batch,
    optimizer: &mut ModuleOptimizer,
    device: &Device,
) -> Result<(GptModel, f32), Box<dyn std::error::Error>> {
    let (inputs, targets) = batch.to_tensors(device);
    let (model, loss) = train_step(model, inputs, targets, LEARNING_RATE, optimizer);
    let loss = loss.into_scalar::<f32>();
    if !loss.is_finite() {
        return Err("Nicht endlicher Trainings-Loss".into());
    }
    Ok((model, loss))
}

fn main() -> Demo {
    let path = std::env::args_os().nth(1).unwrap_or_else(|| {
        concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/the-verdict.txt").into()
    });
    let text = std::fs::read_to_string(path)?;
    let (train_data, validation) = datasets(&text)?;
    describe(&train_data, &validation);
    let device = Device::flex().autodiff();
    let model = model(&device);
    report(&model, &train_data, &validation, "Vor Training");
    let (model, updates) = train(model, &train_data, &device)?;
    println!("Ausgeführte SGD-Updates: {updates}");
    report(&model, &train_data, &validation, "Nach Training");
    println!(
        "Nur Lernversuch: keine volle Epoche, kein vortrainiertes GPT-2, kein Checkpoint gespeichert."
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEXT: &str = include_str!("../../../data/the-verdict.txt");

    #[test]
    fn real_text_windows_preserve_ids_and_split_boundary() -> Demo {
        let tokens = Gpt2Tokenizer::new().encode(TEXT);
        let (train_tokens, validation_tokens) = split_tokens(&tokens, VALIDATION_FRACTION);
        let (train, validation) = datasets(TEXT)?;
        assert_eq!(
            (train.len(), validation.len()),
            (TRAIN_WINDOWS, VALIDATION_WINDOWS)
        );
        for (dataset, source) in [(&train, train_tokens), (&validation, validation_tokens)] {
            for index in 0..dataset.len() {
                verify_window(dataset, source, index);
            }
        }
        assert!(train_tokens.iter().any(|&id| id > 4));
        Ok(())
    }

    fn verify_window(dataset: &Dataset, source: &[u32], index: usize) {
        let start = index * CONTEXT_LENGTH;
        let end = start + CONTEXT_LENGTH;
        let window = dataset.window(index);
        assert_eq!(window.input.as_slice(), &source[start..end]);
        assert_eq!(window.target.as_slice(), &source[start + 1..end + 1]);
        assert!(window.target.iter().all(|&id| (id as usize) < VOCAB_SIZE));
    }

    #[test]
    fn rejects_text_without_validation_window() {
        assert!(datasets("Hello world").is_err());
    }

    #[test]
    fn real_vocab_tensor_shapes() -> Demo {
        let (train, _) = datasets(TEXT)?;
        let device = Device::flex();
        let batch = &train.epoch(BATCH_SIZE, SEED)[0];
        let (inputs, targets) = batch.to_tensors(&device);
        assert_eq!(
            model(&device).forward(inputs).dims(),
            [BATCH_SIZE, CONTEXT_LENGTH, VOCAB_SIZE]
        );
        assert_eq!(targets.dims(), [BATCH_SIZE, CONTEXT_LENGTH]);
        Ok(())
    }

    #[test]
    fn training_updates_weights_and_evaluation_is_read_only() -> Demo {
        let (train, validation) = datasets(TEXT)?;
        let device = Device::flex().autodiff();
        let model = model(&device);
        let batch = &train.epoch(BATCH_SIZE, SEED)[0];
        let (inputs, targets) = batch.to_tensors(&device);
        let before = projection(&model);
        let mut optimizer = SgdConfig::new().init();
        let (updated, loss) = train_step(model, inputs, targets, LEARNING_RATE, &mut optimizer);
        assert!(loss.into_scalar::<f32>().is_finite());
        assert_ne!(projection(&updated), before);
        let before_evaluation = projection(&updated);
        assert!(evaluate(&updated.valid(), &validation, BATCH_SIZE, &Device::flex()).is_finite());
        assert_eq!(projection(&updated), before_evaluation);
        Ok(())
    }

    fn projection(model: &GptModel) -> Vec<f32> {
        model
            .output_projection
            .weight
            .val()
            .into_data()
            .try_to_vec::<f32>()
            .unwrap()
    }

    #[test]
    fn rejects_training_without_complete_batch() -> Demo {
        let train = limited_dataset(&[0; CONTEXT_LENGTH + 1], TRAIN_WINDOWS)?;
        let device = Device::flex().autodiff();
        assert!(super::train(model(&device), &train, &device).is_err());
        Ok(())
    }

    #[test]
    fn stops_after_update_budget() -> Demo {
        let (train, _) = datasets(TEXT)?;
        assert!(train.epoch(BATCH_SIZE, SEED).len() > MAX_UPDATES);
        let device = Device::flex().autodiff();
        let (_, updates) = super::train(model(&device), &train, &device)?;
        assert_eq!(updates, MAX_UPDATES);
        Ok(())
    }

    #[test]
    fn elapsed_time_budget_prevents_next_update() {
        let expired = Instant::now() - TRAINING_TIME_LIMIT;
        assert!(!within_time_budget(&expired, 1));
        assert!(within_time_budget(&Instant::now(), 1));
    }
}
