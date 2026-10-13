use burn::backend::{Autodiff, NdArray};
use burn::module::Module;
use burn::tensor::{Int, Tensor};
use model::gpt::GptModelConfig;

type Cpu = NdArray<f32>;
type Ad = Autodiff<NdArray<f32>>;

const VOCAB_SIZE: usize = 10;
const CONTEXT_LENGTH: usize = 3;
const EMBEDDING_DIM: usize = 4;
const NUM_HEADS: usize = 2;
const NUM_BLOCKS: usize = 2;

fn config() -> GptModelConfig {
    GptModelConfig::new(
        VOCAB_SIZE,
        CONTEXT_LENGTH,
        EMBEDDING_DIM,
        NUM_HEADS,
        NUM_BLOCKS,
    )
}

#[test]
fn forward_returns_one_logit_per_vocabulary_entry() {
    let device = Default::default();
    let model = config().init::<Cpu>(&device);
    let ids = Tensor::<Cpu, 2, Int>::from_data([[1, 2, 3], [3, 2, 1]], &device);

    assert_eq!(model.forward(ids).dims(), [2, CONTEXT_LENGTH, VOCAB_SIZE]);
}

#[test]
fn output_projection_is_separate_and_counted() {
    let model = config().init::<Cpu>(&Default::default());

    assert_eq!(
        model.output_projection.weight.dims(),
        [EMBEDDING_DIM, VOCAB_SIZE]
    );
    // Embedding: (10 + 3) * 4; blocks: 2 * 228; output: 4 * 10.
    assert_eq!(model.num_params(), 52 + 456 + 40);
}

#[test]
fn later_tokens_do_not_change_earlier_logits() {
    let device = Default::default();
    let model = config().init::<Cpu>(&device);
    let before = model
        .forward(Tensor::<Cpu, 2, Int>::from_data([[1, 2, 3]], &device))
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let after = model
        .forward(Tensor::<Cpu, 2, Int>::from_data([[1, 2, 9]], &device))
        .into_data()
        .to_vec::<f32>()
        .unwrap();

    for (old, new) in before[..2 * VOCAB_SIZE]
        .iter()
        .zip(&after[..2 * VOCAB_SIZE])
    {
        assert!((old - new).abs() < 1e-5);
    }
}

#[test]
fn gradients_reach_gpt_embedding_blocks_and_output_projection() {
    let device = Default::default();
    let model = config().init::<Ad>(&device);
    let ids = Tensor::<Ad, 2, Int>::from_data([[1, 2, 3]], &device);
    let gradients = model.forward(ids).powf_scalar(2.0).sum().backward();

    assert!(
        model
            .gpt_embedding
            .token
            .weight
            .val()
            .grad(&gradients)
            .is_some()
    );
    assert!(
        model.blocks[0]
            .attention
            .query
            .weight
            .val()
            .grad(&gradients)
            .is_some()
    );
    assert!(
        model
            .output_projection
            .weight
            .val()
            .grad(&gradients)
            .is_some()
    );
}

#[test]
#[should_panic(expected = "num_blocks must be positive")]
fn rejects_zero_blocks() {
    GptModelConfig::new(VOCAB_SIZE, CONTEXT_LENGTH, EMBEDDING_DIM, NUM_HEADS, 0)
        .init::<Cpu>(&Default::default());
}
