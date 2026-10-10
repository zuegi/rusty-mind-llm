use burn::module::Module;
use burn::nn::{Embedding, EmbeddingConfig};
use burn::tensor::{Int, Tensor, backend::Backend};

// ANCHOR: config
#[derive(Debug)]
pub struct GptEmbeddingConfig {
    pub vocab_size: usize,
    pub max_context_length: usize,
    pub embedding_dim: usize,
}

#[derive(Module, Debug)]
pub struct GptEmbedding<B: Backend> {
    pub token: Embedding<B>,
    pub position: Embedding<B>,
}
// ANCHOR_END: config

// ANCHOR: init
impl GptEmbeddingConfig {
    pub fn new(vocab_size: usize, max_context_length: usize, embedding_dim: usize) -> Self {
        Self {
            vocab_size,
            max_context_length,
            embedding_dim,
        }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> GptEmbedding<B> {
        assert!(self.vocab_size > 0, "vocab_size must be positive");
        assert!(
            self.max_context_length > 0,
            "max_context_length must be positive"
        );
        assert!(self.embedding_dim > 0, "embedding_dim must be positive");
        GptEmbedding {
            token: EmbeddingConfig::new(self.vocab_size, self.embedding_dim).init(device),
            position: EmbeddingConfig::new(self.max_context_length, self.embedding_dim)
                .init(device),
        }
    }
}
// ANCHOR_END: init

// ANCHOR: forward
impl<B: Backend> GptEmbedding<B> {
    pub fn forward(&self, token_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [batch_size, context_length] = token_ids.dims();
        let [max_context_length, _] = self.position.weight.dims();
        assert!(
            batch_size > 0 && context_length > 0,
            "input must not be empty"
        );
        assert!(
            context_length <= max_context_length,
            "context exceeds position table"
        );
        let positions = Tensor::<B, 1, Int>::arange(0..context_length as i64, &token_ids.device())
            .reshape([1, context_length]);
        let token_vectors = self.token.forward(token_ids);
        let position_vectors = self.position.forward(positions);
        token_vectors + position_vectors
    }
}
// ANCHOR_END: forward
