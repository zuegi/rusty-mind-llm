use burn::module::Module;
use burn::optim::SgdConfig;
use burn::tensor::{Device, Int, Tensor};
use model::gpt::{GptModel, GptModelConfig};
use training::dataset::{Batch, Dataset, split_tokens};
use training::loss::cross_entropy;
use training::runner::{evaluate, train_epoch};
use training::step::train_step;


const VOCAB_SIZE: usize = 5;
const CONTEXT_LENGTH: usize = 3;
const EMBEDDING_DIM: usize = 8;
const NUM_HEADS: usize = 2;
const NUM_BLOCKS: usize = 1;
const BATCH_SIZE: usize = 4;
const LEARNING_RATE: f64 = 0.05;
const TOLERANCE: f64 = 1e-5;
const TOKEN_COUNT: usize = 101;
const EPOCHS: usize = 10;
const SEED: u64 = 42;

fn model() -> GptModel {
    let model = GptModelConfig::new(
        VOCAB_SIZE,
        CONTEXT_LENGTH,
        EMBEDDING_DIM,
        NUM_HEADS,
        NUM_BLOCKS,
    )
    .init(&Device::flex().autodiff());
    let inputs = Tensor::<2, Int>::zeros([1, CONTEXT_LENGTH], &Device::flex().autodiff());
    let _ = model.forward(inputs).into_data();
    model
}

fn datasets() -> (Dataset, Dataset) {
    let tokens: Vec<u32> = (0..TOKEN_COUNT).map(|i| (i % VOCAB_SIZE) as u32).collect();
    let (train, validation) = split_tokens(&tokens, 0.2);
    (
        Dataset::new(train, CONTEXT_LENGTH).unwrap(),
        Dataset::new(validation, CONTEXT_LENGTH).unwrap(),
    )
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

fn window_loss(model: &GptModel, window: &training::dataset::Window) -> f64 {
    let batch = Batch {
        inputs: vec![window.input.clone()],
        targets: vec![window.target.clone()],
    };
    let (inputs, targets) = batch.to_tensors(&Device::flex().autodiff());
    cross_entropy(model.forward(inputs), targets).into_scalar::<f32>() as f64
}

#[test]
fn evaluation_batches_keep_order_and_include_partial_batch() {
    let tokens: Vec<u32> = (0..11).collect();
    let dataset = Dataset::new(&tokens, CONTEXT_LENGTH).unwrap();
    let batches = dataset.batches(2);
    assert_eq!(
        batches.iter().map(|b| b.inputs.len()).collect::<Vec<_>>(),
        [2, 1]
    );
    let inputs: Vec<_> = batches.into_iter().flat_map(|b| b.inputs).collect();
    for (index, input) in inputs.iter().enumerate() {
        assert_eq!(input, &dataset.window(index).input);
    }
}

#[test]
fn evaluate_weights_partial_batch_by_positions_and_never_updates_model() {
    let model = model();
    let (train, _) = datasets();
    let before = projection(&model);
    let fixed = model.valid();
    let expected = (0..train.len())
        .map(|i| window_loss(&fixed, train.window(i)))
        .sum::<f64>()
        / train.len() as f64;
    for batch_size in [1, 4, 100] {
        let actual = evaluate(&fixed, &train, batch_size, &Device::flex());
        assert!((actual - expected).abs() < TOLERANCE);
    }
    assert_eq!(projection(&model), before);
}

#[test]
fn epoch_matches_one_manual_update_per_batch() {
    let model = model();
    let (train, _) = datasets();
    let batches = train.epoch(BATCH_SIZE, SEED);
    let mut optimizer = SgdConfig::new().init();
    let mut manual = model.clone();
    for batch in &batches {
        let (inputs, targets) = batch.to_tensors(&Device::flex());
        (manual, _) = train_step(manual, inputs, targets, LEARNING_RATE, &mut optimizer);
    }
    let mut optimizer = SgdConfig::new().init();
    let (updated, steps) = train_epoch(
        model,
        &batches,
        LEARNING_RATE,
        &mut optimizer,
        &Device::flex().autodiff(),
    );
    assert_eq!(steps, train.len() / BATCH_SIZE);
    assert_eq!(projection(&updated), projection(&manual));
}

#[test]
fn repeated_epochs_reduce_train_and_held_out_pattern_loss() {
    let mut model = model();
    let (train, validation) = datasets();
    let device = Device::flex();
    let before_train = evaluate(&model.valid(), &train, BATCH_SIZE, &device);
    let before_validation = evaluate(&model.valid(), &validation, BATCH_SIZE, &device);
    let mut optimizer = SgdConfig::new().init();
    for epoch in 1..=EPOCHS {
        let batches = train.epoch(BATCH_SIZE, SEED + epoch as u64);
        (model, _) = train_epoch(
            model, &batches, LEARNING_RATE, &mut optimizer, &device.clone().autodiff(),
        );
    }
    let after_train = evaluate(&model.valid(), &train, BATCH_SIZE, &device);
    let after_validation = evaluate(&model.valid(), &validation, BATCH_SIZE, &device);
    assert!(after_train < before_train * 0.75);
    assert!(after_validation < before_validation * 0.75);
}

#[test]
#[should_panic(expected = "Keine vollständigen Trainingsbatches")]
fn rejects_epoch_without_complete_batch() {
    let mut optimizer = SgdConfig::new().init();
    train_epoch(
        model(),
        &[],
        LEARNING_RATE,
        &mut optimizer,
        &Device::flex().autodiff(),
    );
}

#[test]
#[should_panic(expected = "Auswertung braucht ein Backend ohne Autodiff")]
fn rejects_evaluation_on_autodiff_backend() {
    let (train, _) = datasets();
    evaluate(&model(), &train, BATCH_SIZE, &Device::flex());
}

#[test]
#[should_panic(expected = "Auswertung braucht ein Backend ohne Autodiff")]
fn rejects_autodiff_input_device_even_for_validation_model() {
    let (train, _) = datasets();
    evaluate(&model().valid(), &train, BATCH_SIZE, &Device::flex().autodiff());
}

#[test]
#[should_panic(expected = "batch_size muss größer als 0 sein")]
fn rejects_zero_evaluation_batch_size() {
    let (train, _) = datasets();
    evaluate(&model().valid(), &train, 0, &Device::flex());
}

#[test]
fn validation_windows_stay_within_their_original_split() {
    let tokens: Vec<u32> = (0..TOKEN_COUNT as u32).collect();
    let (train, validation) = split_tokens(&tokens, 0.2);
    let dataset = Dataset::new(validation, CONTEXT_LENGTH).unwrap();
    for batch in dataset.batches(BATCH_SIZE) {
        for id in batch.inputs.iter().chain(&batch.targets).flatten() {
            assert!(validation.contains(id));
            assert!(!train.contains(id));
        }
    }
}

#[test]
#[should_panic(expected = "Nicht endlicher Loss bei der Auswertung")]
fn rejects_non_finite_evaluation_loss() {
    let mut fixed = model().valid();
    fixed.output_projection.weight = burn::module::Param::from_tensor(Tensor::<2>::full(
        [EMBEDDING_DIM, VOCAB_SIZE],
        f32::NAN,
        &Device::flex(),
    ));
    let (train, _) = datasets();
    evaluate(&fixed, &train, BATCH_SIZE, &Device::flex());
}
