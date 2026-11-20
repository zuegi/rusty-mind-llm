use crate::attention::{MultiHeadAttention, MultiHeadAttentionConfig};
use crate::feed_forward::{FeedForward, FeedForwardConfig};
use crate::layer_norm::{LayerNorm, LayerNormConfig};
use burn::module::Module;
use burn::nn::{Dropout, DropoutConfig};
use burn::tensor::{Device, Tensor};

// ANCHOR: config
#[derive(Debug)]
pub struct TransformerBlockConfig {
    pub embedding_dim: usize,
    pub num_heads: usize,
    pub dropout: f64,
}

#[derive(Module, Debug)]
pub struct TransformerBlock {
    pub attention_norm: LayerNorm,
    pub attention: MultiHeadAttention,
    pub attention_dropout: Dropout,
    pub feed_forward_norm: LayerNorm,
    pub feed_forward: FeedForward,
    pub residual_dropout: Dropout,
}
// ANCHOR_END: config

impl TransformerBlockConfig {
    pub fn new(embedding_dim: usize, num_heads: usize) -> Self {
        Self {
            embedding_dim,
            num_heads,
            dropout: 0.0,
        }
    }

    pub fn with_dropout(mut self, dropout: f64) -> Self {
        self.dropout = dropout;
        self
    }

    pub fn init(&self, device: &Device) -> TransformerBlock {
        TransformerBlock {
            attention_norm: LayerNormConfig::new(self.embedding_dim).init(device),
            attention: MultiHeadAttentionConfig::new(self.embedding_dim, self.num_heads)
                .init(device),
            attention_dropout: DropoutConfig::new(self.dropout).init(),
            feed_forward_norm: LayerNormConfig::new(self.embedding_dim).init(device),
            feed_forward: FeedForwardConfig::new(self.embedding_dim).init(device),
            residual_dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

impl TransformerBlock {
    // ANCHOR: forward
    pub fn forward(&self, input: Tensor<3>) -> Tensor<3> {
        let normed = self.attention_norm.forward(input.clone());
        let attention = self
            .attention_dropout
            .forward(self.attention.forward(normed));
        let after_attention = input + attention;
        let normed = self.feed_forward_norm.forward(after_attention.clone());
        let feed_forward = self
            .residual_dropout
            .forward(self.feed_forward.forward(normed));
        after_attention + feed_forward
    }
    // ANCHOR_END: forward
}
