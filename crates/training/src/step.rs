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
    let valid_rate = learning_rate.is_finite() && learning_rate > 0.0;
    assert!(valid_rate, "Lernrate muss endlich und positiv sein");
    let loss = cross_entropy(model.forward(inputs), targets);
    let gradients = GradientsParams::from_grads(loss.clone().backward(), &model);
    let mut optimizer = SgdConfig::new().init();
    let model = optimizer.step(learning_rate, model, gradients);
    (model, loss.inner())
}
// ANCHOR_END: sgd_step
