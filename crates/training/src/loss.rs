use burn::nn::loss::CrossEntropyLossConfig;
use burn::tensor::{Int, Tensor, backend::Backend};

// ANCHOR: cross_entropy
/// Mittlere Cross-Entropy über alle Positionen und Fenster, aus rohen Logits.
pub fn cross_entropy<B: Backend>(logits: Tensor<B, 3>, targets: Tensor<B, 2, Int>) -> Tensor<B, 1> {
    let [batch_size, context_length, vocab_size] = logits.dims();
    assert!(
        batch_size > 0 && context_length > 0 && vocab_size > 0,
        "Leere Logits"
    );
    let expected = [batch_size, context_length];
    assert_eq!(targets.dims(), expected, "Zielform passt nicht zu Logits");
    let positions = batch_size * context_length;
    CrossEntropyLossConfig::new()
        .init(&logits.device())
        .forward(
            logits.reshape([positions, vocab_size]),
            targets.reshape([positions]),
        )
}
// ANCHOR_END: cross_entropy
