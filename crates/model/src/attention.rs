use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{Tensor, activation::softmax, backend::Backend};

// ANCHOR: config
#[derive(Debug)]
pub struct SelfAttentionConfig {
    pub embedding_dim: usize,
    pub head_dim: usize,
}

#[derive(Module, Debug)]
pub struct SelfAttention<B: Backend> {
    pub query: Linear<B>,
    pub key: Linear<B>,
    pub value: Linear<B>,
}
// ANCHOR_END: config

// ANCHOR: config_impl
impl SelfAttentionConfig {
    pub fn new(embedding_dim: usize, head_dim: usize) -> Self {
        Self {
            embedding_dim,
            head_dim,
        }
    }

    // ANCHOR: init
    pub fn init<B: Backend>(&self, device: &B::Device) -> SelfAttention<B> {
        assert!(self.embedding_dim > 0, "embedding_dim must be positive");
        assert!(self.head_dim > 0, "head_dim must be positive");
        let projection = LinearConfig::new(self.embedding_dim, self.head_dim).with_bias(false);
        SelfAttention {
            query: projection.init(device),
            key: projection.init(device),
            value: projection.init(device),
        }
    }
    // ANCHOR_END: init
}
// ANCHOR_END: config_impl

// ANCHOR: model_impl
impl<B: Backend> SelfAttention<B> {
    // ANCHOR: forward
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, context_length, embedding_dim] = input.dims();
        let [expected_dim, head_dim] = self.query.weight.dims();
        assert!(
            batch_size > 0 && context_length > 0,
            "input must not be empty"
        );
        assert_eq!(embedding_dim, expected_dim, "embedding dimension mismatch");
        let query = self.query.forward(input.clone());
        let key = self.key.forward(input.clone());
        let value = self.value.forward(input);
        let scores = query.matmul(key.swap_dims(1, 2)) / (head_dim as f64).sqrt();
        let weights = softmax(scores, 2);
        weights.matmul(value)
    }
    // ANCHOR_END: model_impl
    // ANCHOR_END: forward
}
