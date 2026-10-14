use burn::optim::{GradientsParams, LearningRate, Optimizer, SgdConfig};
use burn::tensor::{Int, Tensor, backend::AutodiffBackend};
use model::gpt::GptModel;

use crate::loss::cross_entropy;

// ANCHOR: sgd_step
/// Genau ein SGD-Update ohne Momentum; zurück kommt auch der Loss vor dem Update.
pub fn sgd_step<B: AutodiffBackend>(
    model: GptModel<B>,
    inputs: Tensor<B, 2, Int>,
    targets: Tensor<B, 2, Int>,
    learning_rate: LearningRate,
) -> (GptModel<B>, Tensor<B::InnerBackend, 1>) {
    let mut optimizer = SgdConfig::new().init();
    train_step(model, inputs, targets, learning_rate, &mut optimizer)
}
// ANCHOR_END: sgd_step

// ANCHOR: train_step
/// Ein Update mit einem vom Aufrufer erhaltenen Optimizer.
pub fn train_step<B: AutodiffBackend, O: Optimizer<GptModel<B>, B>>(
    model: GptModel<B>,
    inputs: Tensor<B, 2, Int>,
    targets: Tensor<B, 2, Int>,
    learning_rate: LearningRate,
    optimizer: &mut O,
) -> (GptModel<B>, Tensor<B::InnerBackend, 1>) {
    let valid_rate = learning_rate.is_finite() && learning_rate > 0.0;
    assert!(valid_rate, "Lernrate muss endlich und positiv sein");
    let loss = cross_entropy(model.forward(inputs), targets);
    let gradients = GradientsParams::from_grads(loss.clone().backward(), &model);
    let model = optimizer.step(learning_rate, model, gradients);
    (model, loss.inner())
}
// ANCHOR_END: train_step
