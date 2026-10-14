use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{Device, Tensor, activation::softmax};

// ANCHOR: config
#[derive(Debug)]
pub struct SelfAttentionConfig {
    pub embedding_dim: usize,
    pub head_dim: usize,
}

#[derive(Module, Debug)]
pub struct SelfAttention {
    pub query: Linear,
    pub key: Linear,
    pub value: Linear,
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
    pub fn init(&self, device: &Device) -> SelfAttention {
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
impl SelfAttention {
    // ANCHOR: forward
    pub fn forward(&self, input: Tensor<3>) -> Tensor<3> {
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
// ANCHOR: causal_mask
        let mut scores = query.matmul(key.swap_dims(1, 2)) / (head_dim as f64).sqrt();
        if context_length > 1 {
            let mask = Tensor::<2>::ones([context_length, context_length], &scores.device())
                .triu(1)
                .bool()
                .unsqueeze();
            scores = scores.mask_fill(mask, f32::NEG_INFINITY);
        }
// ANCHOR_END: causal_mask
        let weights = softmax(scores, 2);
        weights.matmul(value)
    }
    // ANCHOR_END: forward
}
// ANCHOR_END: model_impl

// ANCHOR: naive_mha
#[derive(Debug)]
pub struct NaiveMultiHeadAttentionConfig {
    pub embedding_dim: usize,
    pub num_heads: usize,
}

#[derive(Module, Debug)]
pub struct NaiveMultiHeadAttention {
    pub heads: Vec<SelfAttention>,
    pub output_projection: Linear,
}

impl NaiveMultiHeadAttentionConfig {
    pub fn new(embedding_dim: usize, num_heads: usize) -> Self {
        Self {
            embedding_dim,
            num_heads,
        }
    }

    pub fn init(&self, device: &Device) -> NaiveMultiHeadAttention {
        assert!(self.embedding_dim > 0, "embedding_dim must be positive");
        assert!(self.num_heads > 0, "num_heads must be positive");
        assert_eq!(
            self.embedding_dim % self.num_heads,
            0,
            "embedding_dim must be divisible by num_heads"
        );
        let head_dim = self.embedding_dim / self.num_heads;
        let head_config = SelfAttentionConfig::new(self.embedding_dim, head_dim);
        let heads = (0..self.num_heads)
            .map(|_| head_config.init(device))
            .collect();
        let output_projection =
            LinearConfig::new(self.embedding_dim, self.embedding_dim).with_bias(false).init(device);
        NaiveMultiHeadAttention {
            heads,
            output_projection,
        }
    }
}

impl NaiveMultiHeadAttention {
    pub fn forward(&self, input: Tensor<3>) -> Tensor<3> {
        let [batch_size, context_length, embedding_dim] = input.dims();
        let [expected_dim, _] = self.output_projection.weight.dims();
        assert!(
            batch_size > 0 && context_length > 0,
            "input must not be empty"
        );
        assert_eq!(embedding_dim, expected_dim, "embedding dimension mismatch");

        let head_outputs: Vec<Tensor<3>> = self
            .heads
            .iter()
            .map(|head| head.forward(input.clone()))
            .collect();

        let concatenated = Tensor::cat(head_outputs, 2);
        self.output_projection.forward(concatenated)
    }
}
// ANCHOR_END: naive_mha

// ANCHOR: mha_config
#[derive(Debug)]
pub struct MultiHeadAttentionConfig {
    pub embedding_dim: usize,
    pub num_heads: usize,
}

#[derive(Module, Debug)]
pub struct MultiHeadAttention {
    pub query: Linear,
    pub key: Linear,
    pub value: Linear,
    pub output_projection: Linear,
    pub num_heads: usize,
    pub head_dim: usize,
}
// ANCHOR_END: mha_config

// ANCHOR: mha_config_impl
impl MultiHeadAttentionConfig {
    pub fn new(embedding_dim: usize, num_heads: usize) -> Self {
        Self {
            embedding_dim,
            num_heads,
        }
    }

    pub fn init(&self, device: &Device) -> MultiHeadAttention {
        assert!(self.embedding_dim > 0, "embedding_dim must be positive");
        assert!(self.num_heads > 0, "num_heads must be positive");
        assert_eq!(
            self.embedding_dim % self.num_heads,
            0,
            "embedding_dim must be divisible by num_heads"
        );
        let head_dim = self.embedding_dim / self.num_heads;
        let proj = LinearConfig::new(self.embedding_dim, self.embedding_dim).with_bias(false);
        let output_projection =
            LinearConfig::new(self.embedding_dim, self.embedding_dim).with_bias(false).init(device);
        MultiHeadAttention {
            query: proj.init(device),
            key: proj.init(device),
            value: proj.init(device),
            output_projection,
            num_heads: self.num_heads,
            head_dim,
        }
    }
}
// ANCHOR_END: mha_config_impl

// ANCHOR: mha_forward
impl MultiHeadAttention {
    pub fn forward(&self, input: Tensor<3>) -> Tensor<3> {
        let [batch_size, context_length, embedding_dim] = input.dims();
        let [expected_dim, _] = self.query.weight.dims();
        assert!(
            batch_size > 0 && context_length > 0,
            "input must not be empty"
        );
        assert_eq!(embedding_dim, expected_dim, "embedding dimension mismatch");

        let q = self.project_and_split(self.query.forward(input.clone()), batch_size, context_length);
        let k = self.project_and_split(self.key.forward(input.clone()), batch_size, context_length);
        let v = self.project_and_split(self.value.forward(input), batch_size, context_length);

        let mut scores = q.matmul(k.swap_dims(2, 3)) / (self.head_dim as f64).sqrt();
        if context_length > 1 {
            let mask = Tensor::<2>::ones([context_length, context_length], &scores.device())
                .triu(1)
                .bool()
                .unsqueeze::<3>()
                .unsqueeze::<4>();
            scores = scores.mask_fill(mask, f32::NEG_INFINITY);
        }
        let weights = softmax(scores, 3);
        let context = weights.matmul(v);

        let merged = context
            .swap_dims(1, 2)
            .reshape([batch_size, context_length, embedding_dim]);
        self.output_projection.forward(merged)
    }

    pub fn project_and_split(
        &self,
        tensor: Tensor<3>,
        batch_size: usize,
        context_length: usize,
    ) -> Tensor<4> {
        tensor
            .reshape([batch_size, context_length, self.num_heads, self.head_dim])
            .swap_dims(1, 2)
    }
}
// ANCHOR_END: mha_forward
