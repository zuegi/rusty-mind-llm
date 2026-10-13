use burn::backend::{Autodiff, NdArray};
use burn::module::{Module, Param};
use burn::tensor::{Tensor, TensorData};
use model::attention::{SelfAttention, SelfAttentionConfig};
use model::embedding::GptEmbeddingConfig;

type Cpu = NdArray<f32>;
const TOLERANCE: f32 = 1e-5;
const QUERY_WEIGHTS: [[f32; 2]; 2] = [[1.0, 0.0], [1.0, 1.0]];
const KEY_WEIGHTS: [[f32; 2]; 2] = [[0.0, 1.0], [1.0, 0.0]];
const VALUE_WEIGHTS: [[f32; 2]; 2] = [[2.0, 0.0], [0.0, 3.0]];

fn fixed_attention<B: burn::tensor::backend::Backend>() -> SelfAttention<B> {
    let device = Default::default();
    let mut attention = SelfAttentionConfig::new(2, 2).init(&device);
    attention.query.weight = Param::from_tensor(Tensor::from_data(QUERY_WEIGHTS, &device));
    attention.key.weight = Param::from_tensor(Tensor::from_data(KEY_WEIGHTS, &device));
    attention.value.weight = Param::from_tensor(Tensor::from_data(VALUE_WEIGHTS, &device));
    attention
}

#[test]
fn projects_each_role_and_uses_scaled_causal_scores() {
    let attention = fixed_attention::<Cpu>();
    let input = Tensor::from_data([[[1.0, 0.0], [0.0, 1.0]]], &Default::default());
    let query = attention.query.forward(input.clone());
    let key = attention.key.forward(input.clone());
    let value = attention.value.forward(input.clone());
    query
        .into_data()
        .assert_eq(&TensorData::from([QUERY_WEIGHTS]), true);
    key.into_data()
        .assert_eq(&TensorData::from([KEY_WEIGHTS]), true);
    value
        .into_data()
        .assert_eq(&TensorData::from([VALUE_WEIGHTS]), true);
    let output = attention
        .forward(input)
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    // Mit kausaler Maske:
    // Position 0 sieht nur Position 0 -> [2.0, 0.0]
    // Position 1 sieht Position 0 und 1 -> [1.0, 1.5]
    let expected = [2.0, 0.0, 1.0, 1.5];
    for (actual, expected) in output.iter().zip(expected) {
        assert!(
            (actual - expected).abs() < TOLERANCE,
            "{actual} != {expected}"
        );
    }
}

#[test]
fn batches_do_not_share_context() {
    let attention = fixed_attention::<Cpu>();
    let first = [[1.0_f32, 0.0], [0.0, 1.0]];
    let second = [[2.0_f32, -1.0], [0.5, 3.0]];
    let device = Default::default();
    let batch = attention.forward(Tensor::from_data([first, second], &device));
    for (index, window) in [first, second].into_iter().enumerate() {
        let single = attention.forward(Tensor::from_data([window], &device));
        let [_, context_length, head_dim] = single.dims();
        batch
            .clone()
            .slice([index..index + 1, 0..context_length, 0..head_dim])
            .into_data()
            .assert_eq(&single.into_data(), true);
    }
}

#[test]
fn accepts_embedding_output_and_different_head_width() {
    let device = Default::default();
    let embedding = GptEmbeddingConfig::new(5, 3, 4).init::<Cpu>(&device);
    let attention = SelfAttentionConfig::new(4, 2).init(&device);
    let input = embedding.forward(Tensor::from_data([[1, 2, 3], [0, 1, 2]], &device));
    assert_eq!(attention.forward(input).dims(), [2, 3, 2]);
    assert_eq!(attention.num_params(), 3 * 4 * 2);
}

#[test]
fn propagates_finite_nonzero_gradients_to_all_projections_and_input() {
    let attention = fixed_attention::<Autodiff<Cpu>>();
    let input = Tensor::from_data([[[1.0, 0.0], [0.0, 1.0]]], &Default::default()).require_grad();
    let grads = attention.forward(input.clone()).sum().backward();
    for projection in [&attention.query, &attention.key, &attention.value] {
        let values = projection
            .weight
            .grad(&grads)
            .expect("Projektionsgradient fehlt")
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert!(values.iter().all(|value| value.is_finite()));
        assert!(values.iter().any(|value| value.abs() > TOLERANCE));
    }
    let values = input
        .grad(&grads)
        .expect("Eingabegradient fehlt")
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    assert!(values.iter().all(|value| value.is_finite()));
    assert!(values.iter().any(|value| value.abs() > TOLERANCE));
}

#[test]
fn one_token_returns_its_value() {
    let attention = fixed_attention::<Cpu>();
    let input = Tensor::from_data([[[2.0, 1.0]]], &Default::default());
    attention
        .forward(input.clone())
        .into_data()
        .assert_eq(&attention.value.forward(input).into_data(), true);
}

// ANCHOR: test_causal
#[test]
fn future_tokens_do_not_influence_earlier_outputs() {
    let attention = fixed_attention::<Cpu>();
    // Erste Sequenz: zwei Tokens
    let seq_a = Tensor::from_data([[[1.0, 0.0], [0.0, 1.0]]], &Default::default());
    // Zweite Sequenz: dieselben ersten zwei Tokens, aber ein drittes beliebiges Token angehängt
    let seq_b = Tensor::from_data(
        [[[1.0, 0.0], [0.0, 1.0], [99.0, -99.0]]],
        &Default::default(),
    );
    let out_a = attention.forward(seq_a);
    let out_b = attention.forward(seq_b);

    // Position 0 und 1 müssen bei beiden Sequenzen exakt identisch sein!
    let out_a_vec = out_a.into_data().to_vec::<f32>().unwrap();
    let out_b_vec = out_b.into_data().to_vec::<f32>().unwrap();
    for i in 0..4 {
        assert!(
            (out_a_vec[i] - out_b_vec[i]).abs() < TOLERANCE,
            "Zukunft beeinflusst Vergangenheit bei Index {i}"
        );
    }
}
// ANCHOR_END: test_causal

#[test]
#[should_panic(expected = "embedding_dim must be positive")]
fn rejects_zero_embedding_dimension() {
    SelfAttentionConfig::new(0, 2).init::<Cpu>(&Default::default());
}

#[test]
#[should_panic(expected = "head_dim must be positive")]
fn rejects_zero_head_dimension() {
    SelfAttentionConfig::new(2, 0).init::<Cpu>(&Default::default());
}

#[test]
#[should_panic(expected = "input must not be empty")]
fn rejects_empty_sequence() {
    fixed_attention::<Cpu>().forward(Tensor::zeros([1, 0, 2], &Default::default()));
}

#[test]
#[should_panic(expected = "input must not be empty")]
fn rejects_empty_batch() {
    fixed_attention::<Cpu>().forward(Tensor::zeros([0, 2, 2], &Default::default()));
}

#[test]
#[should_panic(expected = "embedding dimension mismatch")]
fn rejects_wrong_embedding_dimension() {
    fixed_attention::<Cpu>().forward(Tensor::zeros([1, 2, 3], &Default::default()));
}
