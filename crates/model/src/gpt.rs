use crate::embedding::{GptEmbedding, GptEmbeddingConfig};
use crate::transformer_block::{TransformerBlock, TransformerBlockConfig};
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{Int, Tensor, backend::Backend};

// ANCHOR: config
#[derive(Debug)]
pub struct GptModelConfig {
    pub vocab_size: usize,
    pub max_context_length: usize,
    pub embedding_dim: usize,
    pub num_heads: usize,
    pub num_blocks: usize,
}

#[derive(Module, Debug)]
pub struct GptModel<B: Backend> {
    pub gpt_embedding: GptEmbedding<B>,
    pub blocks: Vec<TransformerBlock<B>>,
    pub output_projection: Linear<B>,
}
// ANCHOR_END: config

impl GptModelConfig {
    pub fn new(
        vocab_size: usize,
        max_context_length: usize,
        embedding_dim: usize,
        num_heads: usize,
        num_blocks: usize,
    ) -> Self {
        Self {
            vocab_size,
            max_context_length,
            embedding_dim,
            num_heads,
            num_blocks,
        }
    }

    // ANCHOR: init
    pub fn init<B: Backend>(&self, device: &B::Device) -> GptModel<B> {
        assert!(self.num_blocks > 0, "num_blocks must be positive");
        let gpt_embedding =
            GptEmbeddingConfig::new(self.vocab_size, self.max_context_length, self.embedding_dim)
                .init(device);
        let block_config = TransformerBlockConfig::new(self.embedding_dim, self.num_heads);
        let blocks = (0..self.num_blocks)
            .map(|_| block_config.init(device))
            .collect();
        let output_projection = LinearConfig::new(self.embedding_dim, self.vocab_size)
            .with_bias(false)
            .init(device);

        GptModel {
            gpt_embedding,
            blocks,
            output_projection,
        }
    }
    // ANCHOR_END: init
}

impl<B: Backend> GptModel<B> {
    // ANCHOR: forward
    pub fn forward(&self, token_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let hidden = self
            .blocks
            .iter()
            .fold(self.gpt_embedding.forward(token_ids), |hidden, block| {
                block.forward(hidden)
            });
        self.output_projection.forward(hidden)
    }
    // ANCHOR_END: forward
}
