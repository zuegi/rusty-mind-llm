//! Kapitel 14: SGD, Momentum, Adam und ein Lernratenplan auf denselben Daten.
//! Ausführen: cargo run --release -p training --example ch14 -- [textdatei]

use burn::{
    grad_clipping::GradientClippingConfig,
    module::Module,
    optim::{AdamConfig, ModuleOptimizer, SgdConfig, momentum::MomentumConfig},
    tensor::Device,
};
use model::gpt::{GptModel, GptModelConfig};
use training::{
    dataset::{Batch, Dataset, split_tokens},
    runner::evaluate,
    step::train_step,
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
const EPOCHS: usize = 3;
const VALIDATION_FRACTION: f64 = 0.1;
const SGD_RATE: f64 = 0.05;
const MOMENTUM_RATE: f64 = 0.01;
const ADAM_RATE: f64 = 0.001;
const MOMENTUM: f64 = 0.9;
const WARMUP_UPDATES: usize = 4;
const FINAL_RATE_FRACTION: f64 = 0.1;
const CLIP_NORM: f32 = 0.1;
const SEED: u64 = 42;
const DEFAULT_TEXT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/the-verdict.txt");
const EXPERIMENTS: [Experiment; 5] = [
    Experiment::Sgd,
    Experiment::Momentum,
    Experiment::Adam,
    Experiment::AdamSchedule,
    Experiment::AdamScheduleClipping,
];

#[derive(Debug, Clone, Copy)]
enum Experiment {
    Sgd,
    Momentum,
    Adam,
    AdamSchedule,
    AdamScheduleClipping,
}

impl Experiment {
    // ANCHOR: optimizer
    fn optimizer(self) -> ModuleOptimizer {
        match self {
            Self::Sgd => SgdConfig::new().init(),
            Self::Momentum => SgdConfig::new()
                .with_momentum(Some(
                    MomentumConfig::new()
                        .with_momentum(MOMENTUM)
                        .with_dampening(0.0),
                ))
                .init(),
            Self::Adam | Self::AdamSchedule => AdamConfig::new().init(),
            Self::AdamScheduleClipping => AdamConfig::new()
                .with_grad_clipping(Some(GradientClippingConfig::Norm(CLIP_NORM)))
                .init(),
        }
    }
    // ANCHOR_END: optimizer

    fn learning_rate(self, update: usize, total: usize) -> f64 {
        match self {
            Self::Sgd => SGD_RATE,
            Self::Momentum => MOMENTUM_RATE,
            Self::Adam => ADAM_RATE,
            Self::AdamSchedule | Self::AdamScheduleClipping => scheduled_rate(update, total),
        }
    }
}

// ANCHOR: schedule
fn scheduled_rate(update: usize, total: usize) -> f64 {
    assert!(
        update > 0 && update <= total,
        "Update muss in 1..=total liegen"
    );
    let warmup = WARMUP_UPDATES.min(total);
    if update <= warmup {
        return ADAM_RATE * update as f64 / warmup as f64;
    }
    let progress = (update - warmup) as f64 / (total - warmup) as f64;
    ADAM_RATE * (1.0 - progress * (1.0 - FINAL_RATE_FRACTION))
}
// ANCHOR_END: schedule

fn datasets(text: &str) -> Result<(Dataset, Dataset), Box<dyn std::error::Error>> {
    let tokens = Gpt2Tokenizer::new().encode(text);
    let (train, validation) = split_tokens(&tokens, VALIDATION_FRACTION);
    let train_count = train.len().min(TRAIN_WINDOWS * CONTEXT_LENGTH + 1);
    let validation_count = validation
        .len()
        .min(VALIDATION_WINDOWS * CONTEXT_LENGTH + 1);
    let train = Dataset::new(&train[..train_count], CONTEXT_LENGTH)?;
    let validation = Dataset::new(&validation[..validation_count], CONTEXT_LENGTH)?;
    if train.len() < BATCH_SIZE {
        return Err("Keine vollständigen Trainingsbatches: mehr Text nötig".into());
    }
    Ok((train, validation))
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

fn report(model: &GptModel, train: &Dataset, validation: &Dataset, epoch: usize) {
    let device = Device::flex();
    let fixed = model.valid();
    let train_loss = evaluate(&fixed, train, BATCH_SIZE, &device);
    let validation_loss = evaluate(&fixed, validation, BATCH_SIZE, &device);
    println!("Epoche {epoch}: Training {train_loss:.6}, Validierung {validation_loss:.6}");
}

// ANCHOR: epoch
fn train_batches(
    mut model: GptModel,
    batches: &[Batch],
    optimizer: &mut ModuleOptimizer,
    rates: &[f64],
    device: &Device,
) -> Result<GptModel, Box<dyn std::error::Error>> {
    assert_eq!(
        batches.len(),
        rates.len(),
        "Jeder Batch braucht eine Lernrate"
    );
    for (batch, &rate) in batches.iter().zip(rates) {
        let (inputs, targets) = batch.to_tensors(device);
        let (updated, loss) = train_step(model, inputs, targets, rate, optimizer);
        if !loss.into_scalar::<f32>().is_finite() {
            return Err("Nicht endlicher Trainings-Loss".into());
        }
        model = updated;
    }
    Ok(model)
}
// ANCHOR_END: epoch

// ANCHOR: run
fn run(
    mut model: GptModel,
    train: &Dataset,
    validation: &Dataset,
    experiment: Experiment,
    device: &Device,
) -> Demo {
    let mut optimizer = experiment.optimizer();
    let updates_per_epoch = train.len() / BATCH_SIZE;
    let total = EPOCHS * updates_per_epoch;
    println!("\n{experiment:?}: {total} Updates");
    report(&model, train, validation, 0);
    for epoch in 1..=EPOCHS {
        let batches = train.epoch(BATCH_SIZE, SEED + epoch as u64);
        let first = (epoch - 1) * updates_per_epoch + 1;
        let rates = epoch_rates(experiment, first, batches.len(), total);
        println!("Lernraten für Updates ab {first}: {rates:.8?}");
        model = train_batches(model, &batches, &mut optimizer, &rates, device)?;
        report(&model, train, validation, epoch);
    }
    Ok(())
}
// ANCHOR_END: run

fn epoch_rates(experiment: Experiment, first: usize, count: usize, total: usize) -> Vec<f64> {
    (first..first + count)
        .map(|update| experiment.learning_rate(update, total))
        .collect()
}

fn describe(train: &Dataset, validation: &Dataset) {
    println!(
        "Seed {SEED}, Fenster: Training {}, Validierung {}, Epochen {EPOCHS}",
        train.len(),
        validation.len()
    );
    println!(
        "SGD {SGD_RATE}, Momentum {MOMENTUM_RATE}, Adam {ADAM_RATE}; Clipping-Norm je Tensor {CLIP_NORM}"
    );
}

fn main() -> Demo {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = match args.as_slice() {
        [] => DEFAULT_TEXT,
        [path] => path.as_str(),
        _ => return Err("Aufruf: ch14 [textdatei]".into()),
    };
    let (train, validation) = datasets(&std::fs::read_to_string(path)?)?;
    let device = Device::flex().autodiff();
    let model = initial_model(&device);
    describe(&train, &validation);
    for experiment in EXPERIMENTS {
        run(model.clone(), &train, &validation, experiment, &device)?;
    }
    println!("\nBegrenzter Lernversuch; kein Checkpoint, kein Resume, kein Qualitätsnachweis.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn warmup_and_decay_use_global_updates() {
        let rates: Vec<_> = (1..=12).map(|update| scheduled_rate(update, 12)).collect();
        assert_eq!(&rates[..4], &[0.00025, 0.0005, 0.00075, 0.001]);
        assert!(rates[4..].windows(2).all(|pair| pair[1] < pair[0]));
        assert!((rates[11] - 0.0001).abs() < 1e-12);
        assert!(rates.iter().all(|rate| rate.is_finite() && *rate > 0.0));
        assert_eq!(scheduled_rate(1, 1), ADAM_RATE);
        assert_eq!(epoch_rates(Experiment::AdamSchedule, 5, 4, 12), rates[4..8]);
    }

    fn fixture() -> Result<(GptModel, Dataset), Box<dyn std::error::Error>> {
        let device = Device::flex().autodiff();
        device.seed(SEED);
        let model = GptModelConfig::new(5, CONTEXT_LENGTH, EMBEDDING_DIM, NUM_HEADS, NUM_BLOCKS)
            .init(&device);
        let data = Dataset::new(&[0, 1, 2, 3, 4].repeat(4), CONTEXT_LENGTH)?;
        Ok((model, data))
    }

    fn one_update(
        model: GptModel,
        data: &Dataset,
        optimizer: &mut ModuleOptimizer,
    ) -> Result<GptModel, Box<dyn std::error::Error>> {
        let device = Device::flex().autodiff();
        let batches = data.epoch(BATCH_SIZE, SEED);
        train_batches(model, &batches, optimizer, &[ADAM_RATE], &device)
    }

    fn measured_loss(model: &GptModel, data: &Dataset) -> f64 {
        evaluate(&model.valid(), data, BATCH_SIZE, &Device::flex())
    }

    #[test]
    fn optimizer_memory_changes_second_update() -> Demo {
        let (model, data) = fixture()?;
        let mut optimizer = Experiment::Adam.optimizer();
        let first = one_update(model, &data, &mut optimizer)?;
        let continued = one_update(first.clone(), &data, &mut optimizer)?;
        let reset = one_update(first, &data, &mut Experiment::Adam.optimizer())?;
        let retained_loss = measured_loss(&continued, &data);
        let reset_loss = measured_loss(&reset, &data);
        assert!((retained_loss - reset_loss).abs() > 1e-6);
        Ok(())
    }

    #[test]
    fn configured_clipping_changes_training_update() -> Demo {
        let (model, data) = fixture()?;
        let unclipped = one_update(
            model.clone(),
            &data,
            &mut Experiment::AdamSchedule.optimizer(),
        )?;
        let clipped = one_update(
            model,
            &data,
            &mut Experiment::AdamScheduleClipping.optimizer(),
        )?;
        assert!((measured_loss(&unclipped, &data) - measured_loss(&clipped, &data)).abs() > 1e-6);
        Ok(())
    }

    #[test]
    fn rejects_text_without_enough_validation_tokens() {
        assert!(datasets("Hello world").is_err());
    }
}
