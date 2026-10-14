use burn::module::{Module, Param};
use burn::tensor::{Device, Tensor, TensorData};
use model::attention::{
    MultiHeadAttentionConfig, NaiveMultiHeadAttentionConfig, SelfAttention, SelfAttentionConfig,
};
use model::embedding::GptEmbeddingConfig;

const TOLERANCE: f32 = 1e-5;
const QUERY_WEIGHTS: [[f32; 2]; 2] = [[1.0, 0.0], [1.0, 1.0]];
const KEY_WEIGHTS: [[f32; 2]; 2] = [[0.0, 1.0], [1.0, 0.0]];
const VALUE_WEIGHTS: [[f32; 2]; 2] = [[2.0, 0.0], [0.0, 3.0]];

fn fixed_attention(device: &Device) -> SelfAttention {
    let mut attention = SelfAttentionConfig::new(2, 2).init(device);
    attention.query.weight = Param::from_tensor(Tensor::from_data(QUERY_WEIGHTS, device));
    attention.key.weight = Param::from_tensor(Tensor::from_data(KEY_WEIGHTS, device));
    attention.value.weight = Param::from_tensor(Tensor::from_data(VALUE_WEIGHTS, device));
    attention
}

#[test]
fn projects_each_role_and_uses_scaled_causal_scores() {
    let attention = fixed_attention(&Device::ndarray());
    let input = Tensor::from_data([[[1.0, 0.0], [0.0, 1.0]]], &Device::ndarray());
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
        .try_to_vec::<f32>()
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
    let attention = fixed_attention(&Device::ndarray());
    let first = [[1.0_f32, 0.0], [0.0, 1.0]];
    let second = [[2.0_f32, -1.0], [0.5, 3.0]];
    let device = Device::ndarray();
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
    let device = Device::ndarray();
    let embedding = GptEmbeddingConfig::new(5, 3, 4).init(&device);
    let attention = SelfAttentionConfig::new(4, 2).init(&device);
    let input = embedding.forward(Tensor::from_data([[1, 2, 3], [0, 1, 2]], &device));
    assert_eq!(attention.forward(input).dims(), [2, 3, 2]);
    assert_eq!(attention.num_params(), 3 * 4 * 2);
}

#[test]
fn propagates_finite_nonzero_gradients_to_all_projections_and_input() {
    let attention = fixed_attention(&Device::ndarray().autodiff());
    let input = Tensor::from_data([[[1.0, 0.0], [0.0, 1.0]]], &Device::ndarray().autodiff()).require_grad();
    let grads = attention.forward(input.clone()).sum().backward();
    for projection in [&attention.query, &attention.key, &attention.value] {
        let values = projection
            .weight
            .grad(&grads)
            .expect("Projektionsgradient fehlt")
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        assert!(values.iter().all(|value| value.is_finite()));
        assert!(values.iter().any(|value| value.abs() > TOLERANCE));
    }
    let values = input
        .grad(&grads)
        .expect("Eingabegradient fehlt")
        .into_data()
        .try_to_vec::<f32>()
        .unwrap();
    assert!(values.iter().all(|value| value.is_finite()));
    assert!(values.iter().any(|value| value.abs() > TOLERANCE));
}

#[test]
fn one_token_returns_its_value() {
    let attention = fixed_attention(&Device::ndarray());
    let input = Tensor::from_data([[[2.0, 1.0]]], &Device::ndarray());
    attention
        .forward(input.clone())
        .into_data()
        .assert_eq(&attention.value.forward(input).into_data(), true);
}

// ANCHOR: test_causal
#[test]
fn future_tokens_do_not_influence_earlier_outputs() {
    let attention = fixed_attention(&Device::ndarray());
    // Erste Sequenz: zwei Tokens
    let seq_a = Tensor::from_data([[[1.0, 0.0], [0.0, 1.0]]], &Device::ndarray());
    // Zweite Sequenz: dieselben ersten zwei Tokens, aber ein drittes beliebiges Token angehängt
    let seq_b = Tensor::from_data(
        [[[1.0, 0.0], [0.0, 1.0], [99.0, -99.0]]],
        &Device::ndarray(),
    );
    let out_a = attention.forward(seq_a);
    let out_b = attention.forward(seq_b);

    // Position 0 und 1 müssen bei beiden Sequenzen exakt identisch sein!
    let out_a_vec = out_a.into_data().try_to_vec::<f32>().unwrap();
    let out_b_vec = out_b.into_data().try_to_vec::<f32>().unwrap();
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
    SelfAttentionConfig::new(0, 2).init(&Device::ndarray());
}

#[test]
#[should_panic(expected = "head_dim must be positive")]
fn rejects_zero_head_dimension() {
    SelfAttentionConfig::new(2, 0).init(&Device::ndarray());
}

#[test]
#[should_panic(expected = "input must not be empty")]
fn rejects_empty_sequence() {
    fixed_attention(&Device::ndarray()).forward(Tensor::zeros([1, 0, 2], &Device::ndarray()));
}

#[test]
#[should_panic(expected = "input must not be empty")]
fn rejects_empty_batch() {
    fixed_attention(&Device::ndarray()).forward(Tensor::zeros([0, 2, 2], &Device::ndarray()));
}

#[test]
#[should_panic(expected = "embedding dimension mismatch")]
fn rejects_wrong_embedding_dimension() {
    fixed_attention(&Device::ndarray()).forward(Tensor::zeros([1, 2, 3], &Device::ndarray()));
}

#[test]
fn multi_head_attention_preserves_shape_and_propagates_gradients() {
    let device = Device::ndarray().autodiff();
    let mha = MultiHeadAttentionConfig::new(4, 2).init(&device);

    let input = Tensor::<3>::ones([2, 3, 4], &device).require_grad();
    let output = mha.forward(input.clone());

    assert_eq!(output.dims(), [2, 3, 4]);

    let loss = output.sum();
    let grads = loss.backward();
    let input_grad = input.grad(&grads).expect("Gradient für Eingabe fehlt");
    assert_eq!(input_grad.dims(), [2, 3, 4]);
    for val in input_grad.into_data().try_to_vec::<f32>().unwrap() {
        assert!(val.is_finite());
    }
}

// ANCHOR: test_mha_equivalence
#[test]
fn naive_and_batched_multi_head_attention_are_numerically_equivalent() {
    let device = Device::ndarray();
    let embedding_dim = 4;
    let num_heads = 2;
    let head_dim = 2;

    let batched = MultiHeadAttentionConfig::new(embedding_dim, num_heads).init(&device);
    let mut naive =
        NaiveMultiHeadAttentionConfig::new(embedding_dim, num_heads).init(&device);

    naive.output_projection.weight = Param::from_tensor(batched.output_projection.weight.val());

    // Kopiere Teilgewichte der gemeinsamen Projektionen in die einzelnen Köpfe
    let q_w = batched.query.weight.val();
    let k_w = batched.key.weight.val();
    let v_w = batched.value.weight.val();

    for h in 0..num_heads {
        let start = h * head_dim;
        let end = start + head_dim;
        naive.heads[h].query.weight =
            Param::from_tensor(q_w.clone().slice([0..embedding_dim, start..end]));
        naive.heads[h].key.weight =
            Param::from_tensor(k_w.clone().slice([0..embedding_dim, start..end]));
        naive.heads[h].value.weight =
            Param::from_tensor(v_w.clone().slice([0..embedding_dim, start..end]));
    }

    let input = Tensor::<3>::from_data(
        [[[1.0, 0.5, -0.2, 0.8], [0.2, -0.7, 0.3, 1.1]]],
        &device,
    );

    let out_batched = batched.forward(input.clone()).into_data().try_to_vec::<f32>().unwrap();
    let out_naive = naive.forward(input).into_data().try_to_vec::<f32>().unwrap();

    assert_eq!(out_batched.len(), out_naive.len());
    for (i, (b, n)) in out_batched.iter().zip(out_naive.iter()).enumerate() {
        assert!(
            (b - n).abs() < TOLERANCE,
            "Abweichung bei Index {i}: batched={b}, naive={n}"
        );
    }
}
// ANCHOR_END: test_mha_equivalence

#[test]
#[should_panic(expected = "embedding_dim must be divisible by num_heads")]
fn rejects_indivisible_num_heads() {
    MultiHeadAttentionConfig::new(5, 2).init(&Device::ndarray());
}
