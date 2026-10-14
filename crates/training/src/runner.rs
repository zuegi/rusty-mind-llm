use burn::tensor::Device;
use burn::optim::{LearningRate, ModuleOptimizer};
use model::gpt::GptModel;

use crate::{dataset::Dataset, loss::cross_entropy, step::train_step};

// ANCHOR: train_epoch
/// Trainiert einmal auf allen vollständigen, gemischten Batches.
pub fn train_epoch(
    mut model: GptModel,
    batches: &[crate::dataset::Batch],
    learning_rate: LearningRate,
    optimizer: &mut ModuleOptimizer,
    device: &Device,
) -> (GptModel, usize) {
    assert!(!batches.is_empty(), "Keine vollständigen Trainingsbatches");
    for batch in batches {
        let (inputs, targets) = batch.to_tensors(device);
        (model, _) = train_step(model, inputs, targets, learning_rate, optimizer);
    }
    (model, batches.len())
}
// ANCHOR_END: train_epoch

// ANCHOR: evaluate
/// Mittlerer Loss aller Positionen mit festen Gewichten, ohne Autodiff.
pub fn evaluate(
    model: &GptModel,
    dataset: &Dataset,
    batch_size: usize,
    device: &Device,
) -> f64 {
    assert!(
        !device.is_autodiff(),
        "Auswertung braucht ein Backend ohne Autodiff"
    );
    assert!(!dataset.is_empty(), "Keine Fenster für die Auswertung");
    let mut total_loss = 0.0;
    let mut positions = 0;
    for batch in dataset.batches(batch_size) {
        let count = batch.inputs.len() * dataset.context_length();
        let (inputs, targets) = batch.to_tensors(device);
        let logits = model.forward(inputs);
        assert!(
            !logits.is_autodiff(),
            "Auswertung braucht ein Backend ohne Autodiff"
        );
        let loss = f64::from(cross_entropy(logits, targets).into_scalar::<f32>());
        assert!(loss.is_finite(), "Nicht endlicher Loss bei der Auswertung");
        total_loss += loss * count as f64;
        positions += count;
    }
    total_loss / positions as f64
}
// ANCHOR_END: evaluate
