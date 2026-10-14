//! Kapitel 9: Trainingsschleife und getrennte Validierung auf der CPU.
//! Ausführen: cargo run -p training --example ch09

use burn::module::Module;
use burn::optim::SgdConfig;
use burn::tensor::{Device, Int, Tensor};
use model::gpt::{GptModel, GptModelConfig};
use training::dataset::{Dataset, split_tokens};
use training::runner::{evaluate, train_epoch};

type Demo = Result<(), Box<dyn std::error::Error>>;

const VOCAB_SIZE: usize = 5;
const TOKEN_COUNT: usize = 101;
const CONTEXT_LENGTH: usize = 3;
const EMBEDDING_DIM: usize = 8;
const NUM_HEADS: usize = 2;
const NUM_BLOCKS: usize = 1;
const BATCH_SIZE: usize = 4;
const EPOCHS: usize = 10;
const SEED: u64 = 42;
const LEARNING_RATE: f64 = 0.05;
const VALIDATION_FRACTION: f64 = 0.2;

// ANCHOR: datasets
fn datasets() -> Result<(Dataset, Dataset), Box<dyn std::error::Error>> {
    let tokens: Vec<u32> = (0..TOKEN_COUNT).map(|i| (i % VOCAB_SIZE) as u32).collect();
    let (train, validation) = split_tokens(&tokens, VALIDATION_FRACTION);
    Ok((
        Dataset::new(train, CONTEXT_LENGTH)?,
        Dataset::new(validation, CONTEXT_LENGTH)?,
    ))
}
// ANCHOR_END: datasets

fn model(device: &Device) -> GptModel {
    device.seed(SEED);
    let model = GptModelConfig::new(
        VOCAB_SIZE,
        CONTEXT_LENGTH,
        EMBEDDING_DIM,
        NUM_HEADS,
        NUM_BLOCKS,
    )
    .init(device);
    let inputs = Tensor::<2, Int>::zeros([1, CONTEXT_LENGTH], device);
    let _ = model.forward(inputs).into_data();
    model
}

// ANCHOR: report
fn report(model: &GptModel, train: &Dataset, validation: &Dataset, epoch: usize) {
    let device = Device::flex();
    let fixed = model.valid();
    let train_loss = evaluate(&fixed, train, BATCH_SIZE, &device);
    let validation_loss = evaluate(&fixed, validation, BATCH_SIZE, &device);
    println!("Epoche {epoch:2}: Training {train_loss:.6}, Validierung {validation_loss:.6}");
}
// ANCHOR_END: report

// ANCHOR: loop
fn main() -> Demo {
    let device = Device::flex().autodiff();
    let (train, validation) = datasets()?;
    let mut model = model(&device);
    let mut optimizer = SgdConfig::new().init();
    let mut updates = 0;
    report(&model, &train, &validation, 0);
    for epoch in 1..=EPOCHS {
        let batches = train.epoch(BATCH_SIZE, SEED + epoch as u64);
        let (updated, steps) = train_epoch(model, &batches, LEARNING_RATE, &mut optimizer, &device);
        model = updated;
        updates += steps;
        report(&model, &train, &validation, epoch);
    }
    println!("SGD-Updates: {updates}");
    Ok(())
}
// ANCHOR_END: loop
