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
const LEARNING_RATE: f64 = 0.001;
const WEIGHT_DECAY: f32 = 0.1;
const DROPOUT: f64 = 0.2;
const SEED: u64 = 42;
const DEFAULT_TEXT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/the-verdict.txt");
const VARIANTS: [Regularization; 3] = [
    Regularization::None,
    Regularization::WeightDecay,
    Regularization::Dropout,
];

#[derive(Debug, Clone, Copy)]
enum Regularization {
    None,
    WeightDecay,
    Dropout,
}

impl Regularization {
    fn name(self) -> &'static str {
        match self {
            Self::None => "Keine Regularisierung",
            Self::WeightDecay => "AdamW mit Weight Decay",
            Self::Dropout => "Dropout 0.2",
        }
    }

    fn dropout(self) -> f64 {
        match self {
            Self::Dropout => DROPOUT,
            _ => 0.0,
        }
    }

    fn optimizer(self) -> ModuleOptimizer {
        match self {
            Self::WeightDecay => AdamWConfig::new().with_weight_decay(WEIGHT_DECAY).init(),
            _ => AdamWConfig::new().with_weight_decay(0.0).init(),
        }
    }

    fn model(self, initial: &GptModel) -> GptModel {
        let mut model = initial.clone();
        for block in &mut model.blocks {
            block.attention_dropout.prob = self.dropout();
            block.residual_dropout.prob = self.dropout();
        }
        model
    }
}

fn initial_model(device: &Device) -> GptModel {
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

fn run_variant(
    regularization: Regularization,
    initial: &GptModel,
    train: &Dataset,
    validation: &Dataset,
    device: &Device,
) {
    let mut model = regularization.model(initial);
    let mut optimizer = regularization.optimizer();
    let training_batches = train.batches(BATCH_SIZE);
    println!("\n{}", regularization.name());
    report(&model, train, validation, 0);
    let updates;
    (model, updates) = train_epoch(
        model,
        &training_batches,
        LEARNING_RATE,
        &mut optimizer,
        device,
    );
    report(&model, train, validation, updates);
}

fn report(model: &GptModel, train: &Dataset, validation: &Dataset, updates: usize) {
    let device = Device::flex();
    let fixed = model.valid();
    let train_loss = evaluate(&fixed, train, BATCH_SIZE, &device);
    let validation_loss = evaluate(&fixed, validation, BATCH_SIZE, &device);
    println!(
        "Updates {updates}: Trainings-Loss {train_loss:.6}, Validierungs-Loss {validation_loss:.6}"
    );
}

fn main() -> Demo {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| DEFAULT_TEXT.to_owned());
    let text = std::fs::read_to_string(path)?;
    let (train, validation) = datasets(&text)?;
    let device = Device::flex().autodiff();
    let initial = initial_model(&device);
    println!("Identische Anfangsgewichte (Seed {SEED}), konstante Lernrate {LEARNING_RATE}");
    for regularization in VARIANTS {
        run_variant(regularization, &initial, &train, &validation, &device);
    }
    println!("\nWenige Updates sind kein Nachweis für Overfitting oder bessere Generalisierung.");
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
