use burn::module::Param;
use burn::tensor::{Device, Tensor};
use model::layer_norm::LayerNormConfig;


fn values(tensor: Tensor<3>) -> Vec<f32> {
    tensor.into_data().try_to_vec::<f32>().unwrap()
}

#[test]
fn normalizes_each_token_to_mean_zero_and_unit_variance() {
    let device = Device::flex();
    let norm = LayerNormConfig::new(4).init(&device);
    let input = Tensor::<3>::from_data([[[1.0, 2.0, 3.0, 4.0], [10.0, 10.0, 30.0, 30.0]]], &device);
    let output = values(norm.forward(input));
    for token in output.chunks(4) {
        let mean = token.iter().sum::<f32>() / 4.0;
        let variance = token.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5, "mean {mean}");
        assert!((variance - 1.0).abs() < 1e-3, "variance {variance}");
    }
}

// ANCHOR: compare
const COMPARE_TOLERANCE: f32 = 1e-4;

#[test]
fn matches_burn_layer_norm() {
    let device = Device::flex();
    let input = Tensor::<3>::from_data([[[1.0, 2.0, 3.0, 4.0], [0.5, -1.0, 2.0, 0.0]]], &device);
    let ours = LayerNormConfig::new(4).init(&device).forward(input.clone());
    let burns = burn::nn::LayerNormConfig::new(4).init(&device).forward(input);
    for (a, b) in values(ours).iter().zip(values(burns)) {
        assert!((a - b).abs() < COMPARE_TOLERANCE, "{a} != {b}");
    }
}

#[test]
fn matches_burn_layer_norm_with_learned_scale_and_shift() {
    let device = Device::flex();
    let input = Tensor::<3>::from_data([[[1.0, 2.0, 3.0, 4.0], [0.5, -1.0, 2.0, 0.0]]], &device);
    let gamma = Tensor::<1>::from_data([2.0, 0.5, 1.0, -1.0], &device);
    let beta = Tensor::<1>::from_data([0.1, 0.2, 0.3, 0.4], &device);

    let mut ours = LayerNormConfig::new(4).init(&device);
    ours.scale = Param::from_tensor(gamma.clone());
    ours.shift = Param::from_tensor(beta.clone());
    let mut burns = burn::nn::LayerNormConfig::new(4).init(&device);
    burns.gamma = Param::from_tensor(gamma);
    burns.beta = Some(Param::from_tensor(beta));

    let ours = values(ours.forward(input.clone()));
    for (a, b) in ours.iter().zip(values(burns.forward(input))) {
        assert!((a - b).abs() < COMPARE_TOLERANCE, "{a} != {b}");
    }
}
// ANCHOR_END: compare

#[test]
fn propagates_gradients_to_scale_and_shift() {
    let device = Device::flex().autodiff();
    let norm = LayerNormConfig::new(3).init(&device);
    let input = Tensor::<3>::from_data([[[1.0, 2.0, 4.0]]], &device);
    let grads = (norm.forward(input) * 2.0).sum().backward();
    assert!(norm.scale.val().grad(&grads).is_some());
    assert!(norm.shift.val().grad(&grads).is_some());
}

#[test]
#[should_panic(expected = "embedding dimension mismatch")]
fn rejects_wrong_embedding_dimension() {
    let device = Device::flex();
    let norm = LayerNormConfig::new(4).init(&device);
    norm.forward(Tensor::<3>::zeros([1, 2, 3], &device));
}
