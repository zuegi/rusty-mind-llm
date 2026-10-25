use burn::module::{Module, Param};
use burn::tensor::{Device, Tensor};

const DEFAULT_EPSILON: f64 = 1e-5;

// ANCHOR: config
#[derive(Debug)]
pub struct LayerNormConfig {
    pub embedding_dim: usize,
    pub epsilon: f64,
}

#[derive(Module, Debug)]
pub struct LayerNorm {
    pub scale: Param<Tensor<1>>,
    pub shift: Param<Tensor<1>>,
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

    // ANCHOR: init
    pub fn init(&self, device: &Device) -> LayerNorm {
        assert!(self.embedding_dim > 0, "embedding_dim must be positive");
        LayerNorm {
            scale: Param::from_tensor(Tensor::ones([self.embedding_dim], device)),
            shift: Param::from_tensor(Tensor::zeros([self.embedding_dim], device)),
            epsilon: self.epsilon,
        }
    }
    // ANCHOR_END: init
}

impl LayerNorm {
    // ANCHOR: forward
    pub fn forward(&self, input: Tensor<3>) -> Tensor<3> {
        let [_, _, embedding_dim] = input.dims();
        assert_eq!(
            embedding_dim,
            self.scale.dims()[0],
            "embedding dimension mismatch"
        );
        // ANCHOR: normalize
        let mean = input.clone().mean_dim(2);
        let centered = input - mean;
        let variance = centered.clone().powf_scalar(2.0).mean_dim(2);
        let normalized = centered / (variance + self.epsilon).sqrt();
        // ANCHOR_END: normalize
        // ANCHOR: affine
        let scale = self.scale.val().reshape([1, 1, embedding_dim]);
        let shift = self.shift.val().reshape([1, 1, embedding_dim]);
        normalized * scale + shift
        // ANCHOR_END: affine
    }
    // ANCHOR_END: forward
}
