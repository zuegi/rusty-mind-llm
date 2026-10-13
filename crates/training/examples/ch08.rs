//! Kapitel 8: Genau ein SGD-Schritt für das GPT-Modell.
//! Ausführen: cargo run -p training --example ch08

// ANCHOR: setup
use burn::backend::{Autodiff, NdArray};
use burn::module::AutodiffModule;
use burn::tensor::backend::Backend;
use model::gpt::{GptModel, GptModelConfig};
use training::dataset::{Batch, Dataset};
use training::loss::cross_entropy;
use training::step::sgd_step;

type Cpu = NdArray<f32>;
type Ad = Autodiff<Cpu>;
type Demo = Result<(), Box<dyn std::error::Error>>;

const VOCAB_SIZE: usize = 5;
const CONTEXT_LENGTH: usize = 3;
const EMBEDDING_DIM: usize = 4;
const NUM_HEADS: usize = 2;
const NUM_BLOCKS: usize = 1;
const BATCH_SIZE: usize = 2;
const SEED: u64 = 42;
const LEARNING_RATE: f64 = 0.01;
const TOKEN_IDS: [u32; 7] = [0, 1, 2, 3, 4, 0, 1];
// ANCHOR_END: setup

// ANCHOR: batch
fn batch() -> Result<Batch, Box<dyn std::error::Error>> {
    let dataset = Dataset::new(&TOKEN_IDS, CONTEXT_LENGTH)?;
    dataset
        .epoch(BATCH_SIZE, SEED)
        .into_iter()
        .next()
        .ok_or_else(|| "Kein vollständiger Batch".into())
}
// ANCHOR_END: batch

// ANCHOR: model
fn model<B: Backend>(device: &B::Device) -> GptModel<B> {
    B::seed(device, SEED);
    GptModelConfig::new(
        VOCAB_SIZE,
        CONTEXT_LENGTH,
        EMBEDDING_DIM,
        NUM_HEADS,
        NUM_BLOCKS,
    )
    .init(device)
}
// ANCHOR_END: model

// ANCHOR: main
fn main() -> Demo {
    let device = Default::default();
    let batch = batch()?;
    println!("Eingaben: {:?}", batch.inputs);
    println!("Ziele:    {:?}", batch.targets);
    let (inputs, targets) = batch.to_tensors::<Ad>(&device);
    println!("Tensorformen: {:?}, {:?}", inputs.dims(), targets.dims());
    let (updated, before) = sgd_step(model::<Ad>(&device), inputs, targets, LEARNING_RATE);
    let (inputs, targets) = batch.to_tensors::<Cpu>(&device);
    let after = cross_entropy(updated.valid().forward(inputs), targets);
    println!("Lernrate: {LEARNING_RATE}");
    println!("Loss vor SGD:  {:.6}", before.into_scalar());
    println!("Loss nach SGD: {:.6}", after.into_scalar());
    Ok(())
}
// ANCHOR_END: main
