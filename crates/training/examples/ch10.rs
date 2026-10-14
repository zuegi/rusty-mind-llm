//! Kapitel 10: Modellgewichte speichern und laden.
//! Ausführen: cargo run -p training --example ch10

use std::{fs, path::PathBuf};

use burn::tensor::{Device, Int, Tensor};
use model::gpt::GptModelConfig;
use training::checkpoint::{load_model, save_model};

type Demo = Result<(), Box<dyn std::error::Error>>;

const VOCAB_SIZE: usize = 5;
const CONTEXT_LENGTH: usize = 3;
const EMBEDDING_DIM: usize = 8;
const NUM_HEADS: usize = 2;
const NUM_BLOCKS: usize = 1;

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

fn main() -> Demo {
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
