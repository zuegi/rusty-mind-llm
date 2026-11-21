//! Kapitel 17: Regularisierung mit Dropout und AdamW.
//! Ausführen: cargo run --release -p training --example ch17 -- [textdatei]

use burn::{
    module::Module,
    optim::{AdamWConfig, ModuleOptimizer},
    tensor::Device,
};
use model::gpt::{GptModel, GptModelConfig};
use training::{
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
const TRAIN_WINDOWS: usize = 8;
const VALIDATION_WINDOWS: usize = 2;
const VALIDATION_FRACTION: f64 = 0.1;
const LEARNING_RATE: f64 = 0.01;
const EPOCHS: usize = 60;
const REPORT_EVERY: usize = 10;
const SEEDS: [u64; 3] = [42, 43, 44];
const WEIGHT_DECAY: f32 = 0.1;
const DROPOUT: f64 = 0.2;
const DEFAULT_TEXT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/the-verdict.txt");
const VARIANTS: [Regularization; 4] = [
    Regularization::None,
    Regularization::WeightDecay,
    Regularization::Dropout,
    Regularization::Both,
];

#[derive(Debug, Clone, Copy)]
enum Regularization {
    None,
    WeightDecay,
    Dropout,
    Both,
}

impl Regularization {
    fn name(self) -> &'static str {
        match self {
            Self::None => "Keine Regularisierung",
            Self::WeightDecay => "AdamW mit Weight Decay",
            Self::Dropout => "Dropout 0.2",
            Self::Both => "AdamW mit Weight Decay und Dropout 0.2",
        }
    }

    fn dropout(self) -> f64 {
        match self {
            Self::Dropout | Self::Both => DROPOUT,
            _ => 0.0,
        }
    }

    // ANCHOR: optimizer
    fn optimizer(self) -> ModuleOptimizer {
        match self {
            Self::WeightDecay | Self::Both => {
                AdamWConfig::new().with_weight_decay(WEIGHT_DECAY).init()
            }
            _ => AdamWConfig::new().with_weight_decay(0.0).init(),
        }
    }
    // ANCHOR_END: optimizer

    // ANCHOR: model
    fn model(self, initial: &GptModel) -> GptModel {
        let mut model = initial.clone();
        for block in &mut model.blocks {
            block.attention_dropout.prob = self.dropout();
            block.residual_dropout.prob = self.dropout();
        }
        model
    }
    // ANCHOR_END: model
}

type Losses = Vec<(f64, f64)>;

fn initial_model(seed: u64, device: &Device) -> GptModel {
    device.seed(seed);
    GptModelConfig::new(
        VOCAB_SIZE,
        CONTEXT_LENGTH,
        EMBEDDING_DIM,
        NUM_HEADS,
        NUM_BLOCKS,
    )
    .init(device)
}

fn datasets(text: &str) -> Result<(Dataset, Dataset), Box<dyn std::error::Error>> {
    let tokens = Gpt2Tokenizer::new().encode(text);
    let (train_tokens, validation_tokens) = split_tokens(&tokens, VALIDATION_FRACTION);
    let train_count = train_tokens.len().min(TRAIN_WINDOWS * CONTEXT_LENGTH + 1);
    let validation_count = validation_tokens
        .len()
        .min(VALIDATION_WINDOWS * CONTEXT_LENGTH + 1);
    let train = Dataset::new(&train_tokens[..train_count], CONTEXT_LENGTH)?;
    let validation = Dataset::new(&validation_tokens[..validation_count], CONTEXT_LENGTH)?;
    if train.len() < BATCH_SIZE {
        return Err("Keine vollständigen Trainingsbatches".into());
    }
    Ok((train, validation))
}

// ANCHOR: run
fn run_variant(
    regularization: Regularization,
    initial: &GptModel,
    seed: u64,
    data: (&Dataset, &Dataset),
    device: &Device,
) -> Losses {
    let (train, validation) = data;
    let mut model = regularization.model(initial);
    let mut optimizer = regularization.optimizer();
    device.seed(seed);
    let mut losses = vec![measure(&model, train, validation)];
    for epoch in 1..=EPOCHS {
        let batches = train.epoch(BATCH_SIZE, seed + epoch as u64);
        (model, _) = train_epoch(model, &batches, LEARNING_RATE, &mut optimizer, device);
        if epoch % REPORT_EVERY == 0 {
            losses.push(measure(&model, train, validation));
        }
    }
    losses
}
// ANCHOR_END: run

// ANCHOR: measure
fn measure(model: &GptModel, train: &Dataset, validation: &Dataset) -> (f64, f64) {
    let device = Device::flex();
    let fixed = model.valid();
    (
        evaluate(&fixed, train, BATCH_SIZE, &device),
        evaluate(&fixed, validation, BATCH_SIZE, &device),
    )
}
// ANCHOR_END: measure

fn mean_losses(runs: &[Losses]) -> Losses {
    let count = runs.len() as f64;
    (0..runs[0].len())
        .map(|point| {
            let train = runs.iter().map(|run| run[point].0).sum::<f64>() / count;
            let validation = runs.iter().map(|run| run[point].1).sum::<f64>() / count;
            (train, validation)
        })
        .collect()
}

fn print_variant(regularization: Regularization, losses: &Losses, updates_per_epoch: usize) {
    println!("\n{}", regularization.name());
    for (point, (train, validation)) in losses.iter().enumerate() {
        let updates = point * REPORT_EVERY * updates_per_epoch;
        println!(
            "Updates {updates:>4}: Training {train:.4}, Validierung {validation:.4}, Lücke {:+.4}",
            validation - train
        );
    }
}

fn main() -> Demo {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| DEFAULT_TEXT.to_owned());
    let text = std::fs::read_to_string(path)?;
    let (train, validation) = datasets(&text)?;
    let device = Device::flex().autodiff();
    let initials: Vec<_> = SEEDS
        .iter()
        .map(|&seed| initial_model(seed, &device))
        .collect();
    println!(
        "Mittel über Seeds {SEEDS:?}, gleiche Anfangsgewichte je Seed, \
         konstante Lernrate {LEARNING_RATE}, {EPOCHS} Epochen"
    );
    for regularization in VARIANTS {
        let runs: Vec<_> = SEEDS
            .iter()
            .zip(&initials)
            .map(|(&seed, initial)| {
                run_variant(
                    regularization,
                    initial,
                    seed,
                    (&train, &validation),
                    &device,
                )
            })
            .collect();
        let updates_per_epoch = train.len() / BATCH_SIZE;
        print_variant(regularization, &mean_losses(&runs), updates_per_epoch);
    }
    println!("\nKleines Experiment: Lücke und Reihenfolge sind kein Beweis für Generalisierung.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Int, Tensor};

    #[test]
    fn variants_share_initial_predictions_and_training_keeps_initial_weights() -> Demo {
        let device = Device::flex().autodiff();
        let initial = GptModelConfig::new(5, CONTEXT_LENGTH, EMBEDDING_DIM, NUM_HEADS, NUM_BLOCKS)
            .init(&device);
        let input = Tensor::<2, Int>::zeros([BATCH_SIZE, CONTEXT_LENGTH], &Device::flex());
        let expected = initial.valid().forward(input.clone()).into_data();
        let data = Dataset::new(&[0, 1, 2, 3, 4].repeat(4), CONTEXT_LENGTH)?;
        for variant in VARIANTS {
            let model = variant.model(&initial);
            assert_eq!(model.valid().forward(input.clone()).into_data(), expected);
            let (trained, _) = train_epoch(
                model,
                &data.batches(BATCH_SIZE),
                LEARNING_RATE,
                &mut variant.optimizer(),
                &device,
            );
            assert_ne!(trained.valid().forward(input.clone()).into_data(), expected);
            assert_eq!(initial.valid().forward(input.clone()).into_data(), expected);
        }
        Ok(())
    }
}
