use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{Device, Tensor, activation::gelu};

const EXPANSION_FACTOR: usize = 4;

// ANCHOR: config
#[derive(Debug)]
pub struct FeedForwardConfig {
    pub embedding_dim: usize,
}

#[derive(Module, Debug)]
pub struct FeedForward {
    pub expand: Linear,
    pub contract: Linear,
}
// ANCHOR_END: config

impl FeedForwardConfig {
    pub fn new(embedding_dim: usize) -> Self {
        Self { embedding_dim }
    }

    // ANCHOR: init
    pub fn init(&self, device: &Device) -> FeedForward {
        assert!(self.embedding_dim > 0, "embedding_dim must be positive");
        let hidden_dim = EXPANSION_FACTOR * self.embedding_dim;
        FeedForward {
            expand: LinearConfig::new(self.embedding_dim, hidden_dim).init(device),
            contract: LinearConfig::new(hidden_dim, self.embedding_dim).init(device),
        }
    }
    // ANCHOR_END: init
}

impl FeedForward {
    // ANCHOR: forward
    pub fn forward(&self, input: Tensor<3>) -> Tensor<3> {
        // ANCHOR: expand_activate
        let hidden = gelu(self.expand.forward(input));
        // ANCHOR_END: expand_activate
        // ANCHOR: contract
        self.contract.forward(hidden)
        // ANCHOR_END: contract
    }
    // ANCHOR_END: forward
}
