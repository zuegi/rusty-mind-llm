use burn::optim::{GradientsParams, LearningRate, ModuleOptimizer, SgdConfig};
use burn::tensor::{Int, Tensor};
use model::gpt::GptModel;

use crate::loss::cross_entropy;

// ANCHOR: sgd_step
/// Genau ein SGD-Update ohne Momentum; zurück kommt auch der Loss vor dem Update.
pub fn sgd_step(
    model: GptModel,
    inputs: Tensor<2, Int>,
    targets: Tensor<2, Int>,
    learning_rate: LearningRate,
) -> (GptModel, Tensor<1>) {
    let mut optimizer = SgdConfig::new().init();
    train_step(model, inputs, targets, learning_rate, &mut optimizer)
}
// ANCHOR_END: sgd_step

// ANCHOR: train_step
/// Ein Update mit einem vom Aufrufer erhaltenen Optimizer.
pub fn train_step(
    model: GptModel,
    inputs: Tensor<2, Int>,
    targets: Tensor<2, Int>,
    learning_rate: LearningRate,
    optimizer: &mut ModuleOptimizer,
) -> (GptModel, Tensor<1>) {
    let valid_rate = learning_rate.is_finite() && learning_rate > 0.0;
    assert!(valid_rate, "Lernrate muss endlich und positiv sein");
    let loss = cross_entropy(model.forward(inputs), targets);
    assert!(loss.is_autodiff(), "Training braucht ein Autodiff-Device");
    let gradients = GradientsParams::from_grads(loss.clone().backward(), &model);
    assert!(!gradients.is_empty(), "Modellparameter ohne Gradienten");
    let model = optimizer.step(learning_rate, model, gradients);
    (model, loss.without_autodiff())
}
// ANCHOR_END: train_step
