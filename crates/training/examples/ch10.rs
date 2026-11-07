//! Kapitel 10: Modellgewichte speichern und laden.
//! Ausführen: cargo run -p training --example ch10

use std::{fs, path::PathBuf};

use burn::{
    module::Module,
    optim::SgdConfig,
    tensor::{Device, Int, Tensor},
};
use model::gpt::{GptModel, GptModelConfig};
use training::checkpoint::{load_model, save_model};
use training::{
    dataset::{Dataset, split_tokens},
    runner::{evaluate, train_epoch},
};

type Demo = Result<(), Box<dyn std::error::Error>>;

const VOCAB_SIZE: usize = 5;
const CONTEXT_LENGTH: usize = 3;
const EMBEDDING_DIM: usize = 8;
const NUM_HEADS: usize = 2;
const NUM_BLOCKS: usize = 1;
const TOKEN_COUNT: usize = 101;
const BATCH_SIZE: usize = 4;
const EPOCHS: usize = 10;
const SEED: u64 = 42;
const LEARNING_RATE: f64 = 0.05;
const VALIDATION_FRACTION: f64 = 0.2;

fn config() -> GptModelConfig {
    GptModelConfig::new(
        VOCAB_SIZE,
        CONTEXT_LENGTH,
        EMBEDDING_DIM,
        NUM_HEADS,
        NUM_BLOCKS,
    )
}

fn checkpoint_path() -> PathBuf {
    std::env::temp_dir().join("rusty-mind-llm-ch10-model.bpk")
}

fn training_model(device: &Device) -> GptModel {
    device.seed(SEED);
    let model = config().init(device);
    let inputs = Tensor::<2, Int>::zeros([1, CONTEXT_LENGTH], device);
    let _ = model.forward(inputs).into_data();
    model
}

fn report(model: &GptModel, train: &Dataset, validation: &Dataset, epoch: usize) {
    let device = Device::flex();
    let fixed = model.valid();
    let train_loss = evaluate(&fixed, train, BATCH_SIZE, &device);
    let validation_loss = evaluate(&fixed, validation, BATCH_SIZE, &device);
    println!("Epoche {epoch:2}: Training {train_loss:.6}, Validierung {validation_loss:.6}");
}

fn train_and_save(path: PathBuf) -> Demo {
    let device = Device::flex().autodiff();
    let tokens: Vec<u32> = (0..TOKEN_COUNT).map(|i| (i % VOCAB_SIZE) as u32).collect();
    let (train, validation) = split_tokens(&tokens, VALIDATION_FRACTION);
    let train = Dataset::new(train, CONTEXT_LENGTH)?;
    let validation = Dataset::new(validation, CONTEXT_LENGTH)?;
    let mut model = training_model(&device);
    let mut optimizer = SgdConfig::new().init();
    report(&model, &train, &validation, 0);
    for epoch in 1..=EPOCHS {
        let batches = train.epoch(BATCH_SIZE, SEED + epoch as u64);
        (model, _) = train_epoch(model, &batches, LEARNING_RATE, &mut optimizer, &device);
        report(&model, &train, &validation, epoch);
    }
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    save_model(&model.valid(), &path)?;
    println!("Trainierte Modellgewichte gespeichert: {}", path.display());
    Ok(())
}

fn main() -> Demo {
    if let Some(path) = std::env::args_os().nth(1) {
        return train_and_save(PathBuf::from(path));
    }
    let device = Device::flex();
    let config = config();
    let model = config.init(&device);
    let inputs = Tensor::<2, Int>::from_ints([[0, 1, 2]], &device);
    let before = model.forward(inputs.clone()).into_data();

    let path = checkpoint_path();
    save_model(&model, &path)?;
    let loaded = load_model(&config, &path, &device)?;
    let after = loaded.forward(inputs).into_data();

    assert_eq!(before, after);
    println!("Modellgewichte gespeichert und identisch geladen.");
    println!("Datei: {}", path.display());
    fs::remove_file(path)?;
    Ok(())
}
