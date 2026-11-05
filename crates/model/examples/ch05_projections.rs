//! Kapitel 5.8: Projektionen, skalierte Scores und kausale Maske.
//! Ausführen: cargo run -p model --example ch05_projections

use burn::module::Param;
use burn::tensor::{Device, Tensor};
use model::attention::{SelfAttention, SelfAttentionConfig};

type Demo = Result<(), Box<dyn std::error::Error>>;

const EMBEDDING_DIM: usize = 2;
const HEAD_DIM: usize = 2;
const INPUT: [[[f32; EMBEDDING_DIM]; 2]; 1] = [[[1.0, 0.0], [0.0, 1.0]]];
const QUERY_WEIGHTS: [[f32; HEAD_DIM]; EMBEDDING_DIM] = [[1.0, 0.0], [1.0, 1.0]];
const KEY_WEIGHTS: [[f32; HEAD_DIM]; EMBEDDING_DIM] = [[0.0, 1.0], [1.0, 0.0]];
const VALUE_WEIGHTS: [[f32; HEAD_DIM]; EMBEDDING_DIM] = [[2.0, 0.0], [0.0, 3.0]];

// ANCHOR: fixed
fn fixed_attention() -> SelfAttention {
    let device = Device::flex();
    let mut attention = SelfAttentionConfig::new(EMBEDDING_DIM, HEAD_DIM).init(&device);
    attention.query.weight = Param::from_tensor(Tensor::from_data(QUERY_WEIGHTS, &device));
    attention.key.weight = Param::from_tensor(Tensor::from_data(KEY_WEIGHTS, &device));
    attention.value.weight = Param::from_tensor(Tensor::from_data(VALUE_WEIGHTS, &device));
    attention
}
// ANCHOR_END: fixed

fn main() -> Demo {
    let input = Tensor::<3>::from_data(INPUT, &Device::flex());
    let attention = fixed_attention();
    print_vectors("Eingabe", input.clone())?;
    print_vectors("Query", attention.query.forward(input.clone()))?;
    print_vectors("Key", attention.key.forward(input.clone()))?;
    print_vectors("Value", attention.value.forward(input.clone()))?;
    print_vectors("Kontext", attention.forward(input))?;
    Ok(())
}

fn print_vectors(label: &str, tensor: Tensor<3>) -> Demo {
    let [batch_size, context_length, head_dim] = tensor.dims();
    let values = tensor.into_data().try_to_vec::<f32>()?;
    println!("{label}, Form [{batch_size}, {context_length}, {head_dim}]:");
    for (position, vector) in values.chunks(head_dim).enumerate() {
        let numbers: Vec<_> = vector.iter().map(|value| format!("{value:.3}")).collect();
        println!("  Token {position}: [{}]", numbers.join(", "));
    }
    Ok(())
}
