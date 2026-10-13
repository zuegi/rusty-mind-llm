use crate::attention::{MultiHeadAttention, MultiHeadAttentionConfig};
use crate::feed_forward::{FeedForward, FeedForwardConfig};
use crate::layer_norm::{LayerNorm, LayerNormConfig};
use burn::module::Module;
use burn::tensor::{Tensor, backend::Backend};

// ANCHOR: config
#[derive(Debug)]
pub struct TransformerBlockConfig {
    pub embedding_dim: usize,
    pub num_heads: usize,
}

#[derive(Module, Debug)]
pub struct TransformerBlock<B: Backend> {
    pub attention_norm: LayerNorm<B>,
    pub attention: MultiHeadAttention<B>,
    pub feed_forward_norm: LayerNorm<B>,
    pub feed_forward: FeedForward<B>,
}
// ANCHOR_END: config

impl TransformerBlockConfig {
    pub fn new(embedding_dim: usize, num_heads: usize) -> Self {
        Self {
            embedding_dim,
            num_heads,
        }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> TransformerBlock<B> {
        TransformerBlock {
            attention_norm: LayerNormConfig::new(self.embedding_dim).init(device),
            attention: MultiHeadAttentionConfig::new(self.embedding_dim, self.num_heads)
                .init(device),
            feed_forward_norm: LayerNormConfig::new(self.embedding_dim).init(device),
            feed_forward: FeedForwardConfig::new(self.embedding_dim).init(device),
        }
    }
}

impl<B: Backend> TransformerBlock<B> {
    // ANCHOR: forward
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let normed = self.attention_norm.forward(input.clone());
        // Residual 1: Eingabe + Attention-Ausgabe
        let after_attention = input + self.attention.forward(normed);
        let normed = self.feed_forward_norm.forward(after_attention.clone());
        // Residual 2: x₁ + Feed-Forward-Ausgabe
        after_attention + self.feed_forward.forward(normed)
    }
    // ANCHOR_END: forward
}
