use burn::module::{Module, Param};
use burn::tensor::{Tensor, backend::Backend};

const DEFAULT_EPSILON: f64 = 1e-5;

// ANCHOR: config
#[derive(Debug)]
pub struct LayerNormConfig {
    pub embedding_dim: usize,
    pub epsilon: f64,
}

#[derive(Module, Debug)]
pub struct LayerNorm<B: Backend> {
    pub scale: Param<Tensor<B, 1>>,
    pub shift: Param<Tensor<B, 1>>,
    epsilon: f64,
}
// ANCHOR_END: config

impl LayerNormConfig {
    pub fn new(embedding_dim: usize) -> Self {
        Self {
            embedding_dim,
            epsilon: DEFAULT_EPSILON,
        }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> LayerNorm<B> {
        assert!(self.embedding_dim > 0, "embedding_dim must be positive");
        LayerNorm {
            scale: Param::from_tensor(Tensor::ones([self.embedding_dim], device)),
            shift: Param::from_tensor(Tensor::zeros([self.embedding_dim], device)),
            epsilon: self.epsilon,
        }
    }
}

impl<B: Backend> LayerNorm<B> {
    // ANCHOR: forward
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let [_, _, embedding_dim] = input.dims();
        assert_eq!(
            embedding_dim,
            self.scale.dims()[0],
            "embedding dimension mismatch"
        );
        let mean = input.clone().mean_dim(2);
        let centered = input - mean;
        let variance = centered.clone().powf_scalar(2.0).mean_dim(2);
        let normalized = centered / (variance + self.epsilon).sqrt();
        let scale = self.scale.val().reshape([1, 1, embedding_dim]);
        let shift = self.shift.val().reshape([1, 1, embedding_dim]);
        normalized * scale + shift
    }
    // ANCHOR_END: forward
}
