use burn::module::Module;
use burn::tensor::{Device, Int, Tensor};
use model::embedding::GptEmbeddingConfig;
use model::feed_forward::FeedForwardConfig;
use model::transformer_block::TransformerBlockConfig;


#[test]
fn feed_forward_keeps_shape() {
    let device = Device::flex();
    let ff = FeedForwardConfig::new(4).init(&device);
    assert_eq!(ff.forward(Tensor::zeros([2, 3, 4], &device)).dims(), [2, 3, 4]);
}

#[test]
fn block_keeps_shape() {
    let device = Device::flex();
    let block = TransformerBlockConfig::new(4, 2).init(&device);
    assert_eq!(block.forward(Tensor::ones([2, 3, 4], &device)).dims(), [2, 3, 4]);
}

#[test]
fn block_does_not_look_into_the_future() {
    let device = Device::flex();
    let block = TransformerBlockConfig::new(4, 2).init(&device);
    let first = Tensor::<3>::from_data([[[1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 0.0, 1.0]]], &device);
    let changed = Tensor::<3>::from_data([[[1.0, 2.0, 3.0, 4.0], [9.0, 8.0, 7.0, 6.0]]], &device);
    let a = block.forward(first).into_data().try_to_vec::<f32>().unwrap();
    let b = block.forward(changed).into_data().try_to_vec::<f32>().unwrap();
    for (x, y) in a[..4].iter().zip(&b[..4]) {
        assert!((x - y).abs() < 1e-5);
    }
}

#[test]
fn accepts_embedding_output() {
    let device = Device::flex();
    let embedding = GptEmbeddingConfig::new(10, 3, 4).init(&device);
    let block = TransformerBlockConfig::new(4, 2).init(&device);
    let ids = Tensor::<2, Int>::from_data([[1, 2, 3]], &device);
    assert_eq!(block.forward(embedding.forward(ids)).dims(), [1, 3, 4]);
}

#[test]
fn propagates_gradients_to_all_sublayers() {
    let device = Device::flex().autodiff();
    let block = TransformerBlockConfig::new(4, 2).init(&device);
    let input = Tensor::<3>::from_data([[[1.0, 2.0, 3.0, 4.0], [0.5, -1.0, 2.0, 0.0]]], &device);
    let grads = block.forward(input).powf_scalar(2.0).sum().backward();
    assert!(block.attention.query.weight.val().grad(&grads).is_some());
    assert!(block.feed_forward.expand.weight.val().grad(&grads).is_some());
    assert!(block.feed_forward_norm.scale.val().grad(&grads).is_some());
    assert!(block.attention_norm.shift.val().grad(&grads).is_some());
}

#[test]
fn module_counts_parameters() {
    let block = TransformerBlockConfig::new(4, 2).init(&Device::flex());
    // 4 Attention-Matrizen (16) + 2 LayerNorm (8) + FFN (4*16+16 + 16*4+4)
    assert_eq!(block.num_params(), 64 + 16 + 148);
}
