use burn::module::Module;
use burn::tensor::{Device, Int, Tensor};
use model::gpt::{GptModel, GptModelConfig};
use training::dataset::{Batch, Dataset};
use training::loss::cross_entropy;
use training::step::sgd_step;

const TOLERANCE: f32 = 1e-5;
const LEARNING_RATE: f64 = 0.001;
const VOCAB_SIZE: usize = 5;
const CONTEXT_LENGTH: usize = 3;
const EMBEDDING_DIM: usize = 4;
const NUM_HEADS: usize = 2;
const NUM_BLOCKS: usize = 1;
const TOKEN_IDS: [u32; 7] = [0, 1, 2, 3, 4, 0, 1];
const EXPECTED_GRADIENTS: [f32; 6] = [
    1.0 / 6.0,
    1.0 / 6.0,
    -1.0 / 3.0,
    -1.0 / 3.0,
    1.0 / 6.0,
    1.0 / 6.0,
];

fn values<const D: usize>(tensor: Tensor<D>) -> Vec<f32> {
    tensor.into_data().try_to_vec::<f32>().unwrap()
}

fn token_values(rows: &[Vec<u32>]) -> Vec<i64> {
    rows.iter().flatten().map(|&id| i64::from(id)).collect()
}

fn batch() -> Batch {
    Dataset::new(&TOKEN_IDS, CONTEXT_LENGTH)
        .unwrap()
        .epoch(2, 42)
        .remove(0)
}

fn model() -> GptModel {
    GptModelConfig::new(
        VOCAB_SIZE,
        CONTEXT_LENGTH,
        EMBEDDING_DIM,
        NUM_HEADS,
        NUM_BLOCKS,
    )
    .init(&Device::flex().autodiff())
}

#[test]
fn batch_tensors_preserve_window_order_and_shifted_targets() {
    let batch = batch();
    let (inputs, targets) = batch.to_tensors(&Device::flex());
    assert_eq!(inputs.dims(), [2, CONTEXT_LENGTH]);
    assert_eq!(targets.dims(), [2, CONTEXT_LENGTH]);
    let inputs = inputs.into_data().convert::<i64>().try_to_vec::<i64>().unwrap();
    let targets = targets.into_data().convert::<i64>().try_to_vec::<i64>().unwrap();
    assert_eq!(inputs, token_values(&batch.inputs));
    assert_eq!(targets, token_values(&batch.targets));
    for (inputs, targets) in batch.inputs.iter().zip(&batch.targets) {
        assert_eq!(&inputs[1..], &targets[..CONTEXT_LENGTH - 1]);
    }
}

#[test]
#[should_panic(expected = "Batch darf nicht leer sein")]
fn rejects_empty_batch() {
    Batch {
        inputs: vec![],
        targets: vec![],
    }
    .to_tensors(&Device::flex());
}

#[test]
#[should_panic(expected = "Fenster darf nicht leer sein")]
fn rejects_empty_windows() {
    Batch {
        inputs: vec![vec![]],
        targets: vec![vec![]],
    }
    .to_tensors(&Device::flex());
}

#[test]
#[should_panic(expected = "Batch-Größen ungleich")]
fn rejects_missing_target_windows() {
    Batch {
        inputs: vec![vec![1, 2]],
        targets: vec![],
    }
    .to_tensors(&Device::flex());
}

#[test]
#[should_panic(expected = "Alle Eingabe- und Zielfenster müssen gleich lang sein")]
fn rejects_ragged_inputs() {
    Batch {
        inputs: vec![vec![1, 2], vec![3]],
        targets: vec![vec![2, 3], vec![4, 0]],
    }
    .to_tensors(&Device::flex());
}

#[test]
#[should_panic(expected = "Alle Eingabe- und Zielfenster müssen gleich lang sein")]
fn rejects_mismatched_target_lengths() {
    Batch {
        inputs: vec![vec![1, 2]],
        targets: vec![vec![2]],
    }
    .to_tensors(&Device::flex());
}

#[test]
fn cross_entropy_averages_aligned_positions_across_multiple_windows() {
    let device = Device::flex();
    let weights = [
        [[2.0, 3.0, 5.0], [5.0, 3.0, 2.0]],
        [[3.0, 2.0, 5.0], [3.0, 5.0, 2.0]],
    ];
    let logits = Tensor::<3>::from_data(weights, &device).log();
    let targets = Tensor::<2, Int>::from_data([[2, 1], [0, 2]], &device);
    let loss = cross_entropy(logits, targets);
    let expected = -[0.5_f32, 0.3, 0.3, 0.2].iter().map(|p| p.ln()).sum::<f32>() / 4.0;
    assert_eq!(loss.dims(), [1]);
    assert!((loss.into_scalar::<f32>() - expected).abs() < TOLERANCE);
}

#[test]
fn equal_logits_give_log_vocabulary_size() {
    let device = Device::flex();
    let logits = Tensor::<3>::zeros([2, 3, VOCAB_SIZE], &device);
    let targets = Tensor::<2, Int>::from_data([[0, 1, 2], [3, 4, 0]], &device);
    let loss = cross_entropy(logits, targets).into_scalar::<f32>();
    assert!((loss - (VOCAB_SIZE as f32).ln()).abs() < TOLERANCE);
}

#[test]
fn cross_entropy_remains_finite_for_large_logits() {
    let device = Device::flex();
    let logits = Tensor::<3>::from_data([[[1000.0, 1000.0, -1000.0]]], &device);
    let targets = Tensor::<2, Int>::from_data([[0]], &device);
    let loss = cross_entropy(logits, targets).into_scalar::<f32>();
    assert!(loss.is_finite());
    assert!((loss - 2.0_f32.ln()).abs() < TOLERANCE);
}

#[test]
#[should_panic(expected = "Zielform passt nicht zu Logits")]
fn rejects_target_shape_even_when_element_count_matches() {
    let device = Device::flex();
    let logits = Tensor::<3>::zeros([2, 2, VOCAB_SIZE], &device);
    let targets = Tensor::<2, Int>::zeros([1, 4], &device);
    cross_entropy(logits, targets);
}

#[test]
fn cross_entropy_gradients_match_softmax_minus_target_divided_by_positions() {
    let device = Device::flex().autodiff();
    let logits = Tensor::<3>::zeros([1, 2, 3], &device).require_grad();
    let targets = Tensor::<2, Int>::from_data([[2, 0]], &device);
    let gradients = cross_entropy(logits.clone(), targets).backward();
    let actual = values(logits.grad(&gradients).expect("Logits ohne Gradient"));
    for (actual, expected) in actual.into_iter().zip(EXPECTED_GRADIENTS) {
        assert!((actual - expected).abs() < TOLERANCE);
    }
}

fn parameter_tensors(model: &GptModel) -> [Tensor<2>; 3] {
    [
        model.gpt_embedding.token.weight.val(),
        model.blocks[0].attention.query.weight.val(),
        model.output_projection.weight.val(),
    ]
}

fn assert_sgd_update(
    before: Tensor<2>,
    after: Tensor<2>,
    gradients: &burn::tensor::Gradients,
) {
    let gradient = values(before.grad(gradients).expect("Parameter ohne Gradient"));
    let before = values(before);
    let after = values(after);
    assert!(gradient.iter().all(|g| g.is_finite()));
    assert!(gradient.iter().any(|g| g.abs() > TOLERANCE));
    assert_ne!(before, after, "SGD muss Gewichte verändern");
    for ((before, gradient), after) in before.into_iter().zip(gradient).zip(after) {
        let expected = before - LEARNING_RATE as f32 * gradient;
        assert!((after - expected).abs() < TOLERANCE);
    }
}

#[test]
fn one_sgd_step_updates_embedding_block_and_output_and_reduces_this_batch_loss() {
    let model = model();
    let (inputs, targets) = batch().to_tensors(&Device::flex().autodiff());
    let before = cross_entropy(model.forward(inputs.clone()), targets.clone());
    let gradients = before.clone().backward();
    let (updated, reported) = sgd_step(model.clone(), inputs, targets, LEARNING_RATE);
    assert!(!reported.is_autodiff());
    assert_model_update(&model, &updated, &gradients);
    let (inputs, targets) = batch().to_tensors(&Device::flex());
    let after = cross_entropy(updated.valid().forward(inputs), targets).into_scalar::<f32>();
    assert!((reported.into_scalar::<f32>() - before.clone().into_scalar::<f32>()).abs() < TOLERANCE);
    assert!(after.is_finite());
    assert!(after < before.into_scalar::<f32>(), "SGD soll hier Loss senken");
}

fn assert_model_update(
    before: &GptModel,
    after: &GptModel,
    gradients: &burn::tensor::Gradients,
) {
    for (old, new) in parameter_tensors(before)
        .into_iter()
        .zip(parameter_tensors(after))
    {
        assert_sgd_update(old, new, gradients);
    }
}

#[test]
fn rejects_non_positive_and_non_finite_learning_rates() {
    for learning_rate in [0.0, -0.01, f64::NAN, f64::INFINITY] {
        let (inputs, targets) = batch().to_tensors(&Device::flex().autodiff());
        let failure = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            sgd_step(model(), inputs, targets, learning_rate)
        }));
        assert!(failure.is_err(), "Ungültige Lernrate {learning_rate}");
    }
}

#[test]
#[should_panic(expected = "Training braucht ein Autodiff-Device")]
fn rejects_training_without_autodiff() {
    let device = Device::flex();
    let (inputs, targets) = batch().to_tensors(&device);
    sgd_step(model().valid(), inputs, targets, LEARNING_RATE);
}
