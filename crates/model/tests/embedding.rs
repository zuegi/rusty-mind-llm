use burn::backend::{Autodiff, NdArray};
use burn::module::{Module, Param};
use burn::tensor::{Int, Tensor, TensorData};
use model::embedding::{GptEmbedding, GptEmbeddingConfig};

type Cpu = NdArray<f32>;

fn fixed_embedding() -> GptEmbedding<Cpu> {
    let device = Default::default();
    let mut embedding = GptEmbeddingConfig::new(3, 2, 2).init(&device);
    embedding.token.weight = Param::from_tensor(Tensor::from_data(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        &device,
    ));
    embedding.position.weight =
        Param::from_tensor(Tensor::from_data([[10.0, 20.0], [30.0, 40.0]], &device));
    embedding
}

#[test]
fn repeated_ids_select_same_token_vector() {
    let embedding = fixed_embedding();
    let ids = Tensor::<Cpu, 2, Int>::from_data([[1, 1]], &Default::default());
    embedding
        .token
        .forward(ids)
        .into_data()
        .assert_eq(&TensorData::from([[[3.0_f32, 4.0], [3.0, 4.0]]]), true);
}

#[test]
fn adds_positions_and_restarts_them_for_each_window() {
    let embedding = fixed_embedding();
    let ids = Tensor::<Cpu, 2, Int>::from_data([[1, 1], [0, 2]], &Default::default());
    let output = embedding.forward(ids);
    assert_eq!(output.dims(), [2, 2, 2]);
    output.into_data().assert_eq(
        &TensorData::from([
            [[13.0_f32, 24.0], [33.0, 44.0]],
            [[11.0, 22.0], [35.0, 46.0]],
        ]),
        true,
    );
}

#[test]
fn supports_shorter_windows() {
    let ids = Tensor::<Cpu, 2, Int>::from_data([[2]], &Default::default());
    fixed_embedding()
        .forward(ids)
        .into_data()
        .assert_eq(&TensorData::from([[[15.0_f32, 26.0]]]), true);
}

#[test]
fn initializes_two_trainable_tables() {
    let device = Default::default();
    let embedding = GptEmbeddingConfig::new(5, 3, 4).init::<Autodiff<Cpu>>(&device);
    assert_eq!(embedding.token.weight.dims(), [5, 4]);
    assert_eq!(embedding.position.weight.dims(), [3, 4]);
    assert_eq!(embedding.num_params(), (5 + 3) * 4);
    let ids = Tensor::from_data([[1, 1]], &device);
    let gradients = embedding.forward(ids).sum().backward();
    assert!(embedding.token.weight.val().grad(&gradients).is_some());
    assert!(embedding.position.weight.val().grad(&gradients).is_some());
}

#[test]
#[should_panic(expected = "context exceeds position table")]
fn rejects_long_windows() {
    let ids = Tensor::<Cpu, 2, Int>::from_data([[0, 1, 2]], &Default::default());
    fixed_embedding().forward(ids);
}

#[test]
#[should_panic(expected = "input must not be empty")]
fn rejects_empty_windows() {
    let ids = Tensor::<Cpu, 2, Int>::zeros([1, 0], &Default::default());
    fixed_embedding().forward(ids);
}

#[test]
#[should_panic(expected = "vocab_size must be positive")]
fn rejects_empty_vocabulary() {
    GptEmbeddingConfig::new(0, 2, 2).init::<Cpu>(&Default::default());
}

#[test]
#[should_panic(expected = "max_context_length must be positive")]
fn rejects_zero_context_length() {
    GptEmbeddingConfig::new(3, 0, 2).init::<Cpu>(&Default::default());
}

#[test]
#[should_panic(expected = "embedding_dim must be positive")]
fn rejects_zero_embedding_dimension() {
    GptEmbeddingConfig::new(3, 2, 0).init::<Cpu>(&Default::default());
}
