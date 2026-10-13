use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{Tensor, activation::gelu, backend::Backend};

const EXPANSION_FACTOR: usize = 4;

// ANCHOR: config
#[derive(Debug)]
pub struct FeedForwardConfig {
    pub embedding_dim: usize,
}

#[derive(Module, Debug)]
pub struct FeedForward<B: Backend> {
    pub expand: Linear<B>,
    pub contract: Linear<B>,
}
// ANCHOR_END: config

impl FeedForwardConfig {
    pub fn new(embedding_dim: usize) -> Self {
        Self { embedding_dim }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> FeedForward<B> {
        assert!(self.embedding_dim > 0, "embedding_dim must be positive");
        let hidden_dim = EXPANSION_FACTOR * self.embedding_dim;
        FeedForward {
            expand: LinearConfig::new(self.embedding_dim, hidden_dim).init(device),
            contract: LinearConfig::new(hidden_dim, self.embedding_dim).init(device),
        }
    }
}

impl<B: Backend> FeedForward<B> {
    // ANCHOR: forward
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let hidden = gelu(self.expand.forward(input));
        self.contract.forward(hidden)
    }
    // ANCHOR_END: forward
}
