use burn::optim::{LearningRate, Optimizer};
use burn::tensor::{
    ElementConversion,
    backend::{AutodiffBackend, Backend},
};
use model::gpt::GptModel;

use crate::{dataset::Dataset, loss::cross_entropy, step::train_step};

// ANCHOR: train_epoch
/// Trainiert einmal auf allen vollständigen, gemischten Batches.
pub fn train_epoch<B: AutodiffBackend, O: Optimizer<GptModel<B>, B>>(
    mut model: GptModel<B>,
    batches: &[crate::dataset::Batch],
    learning_rate: LearningRate,
    optimizer: &mut O,
    device: &B::Device,
) -> (GptModel<B>, usize) {
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
pub fn evaluate<B: Backend>(
    model: &GptModel<B>,
    dataset: &Dataset,
    batch_size: usize,
    device: &B::Device,
) -> f64 {
    assert!(
        !B::ad_enabled(device),
        "Auswertung braucht ein Backend ohne Autodiff"
    );
    assert!(!dataset.is_empty(), "Keine Fenster für die Auswertung");
    let mut total_loss = 0.0;
    let mut positions = 0;
    for batch in dataset.batches(batch_size) {
        let count = batch.inputs.len() * dataset.context_length();
        let (inputs, targets) = batch.to_tensors(device);
        let loss = cross_entropy(model.forward(inputs), targets)
            .into_scalar()
            .elem::<f64>();
        assert!(loss.is_finite(), "Nicht endlicher Loss bei der Auswertung");
        total_loss += loss * count as f64;
        positions += count;
    }
    total_loss / positions as f64
}
// ANCHOR_END: evaluate
