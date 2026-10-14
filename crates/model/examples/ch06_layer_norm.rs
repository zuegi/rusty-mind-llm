//! Kapitel 6, Schritt 1: LayerNorm mit festen Zahlen.
//! Ausführen: cargo run -p model --example ch06_layer_norm

use burn::tensor::{Device, Tensor};
use model::layer_norm::LayerNormConfig;

type Demo = Result<(), Box<dyn std::error::Error>>;

const EMBEDDING_DIM: usize = 4;
const INPUT: [[[f32; EMBEDDING_DIM]; 2]; 1] =
    [[[1.0, 2.0, 3.0, 4.0], [10.0, 10.0, 30.0, 30.0]]];

fn main() -> Demo {
    let device = Device::ndarray();
    let input = Tensor::<3>::from_data(INPUT, &device);
    print_vectors("Eingabe", input.clone())?;
    let norm = LayerNormConfig::new(EMBEDDING_DIM).init(&device);
    print_vectors("Nach LayerNorm", norm.forward(input))?;
    Ok(())
}

fn print_vectors(label: &str, tensor: Tensor<3>) -> Demo {
    let [batch_size, context_length, dim] = tensor.dims();
    let values = tensor.into_data().try_to_vec::<f32>()?;
    println!("{label}, Form [{batch_size}, {context_length}, {dim}]:");
    for (position, vector) in values.chunks(dim).enumerate() {
        let mean = vector.iter().sum::<f32>() / dim as f32;
        let variance = vector.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / dim as f32;
        let numbers: Vec<_> = vector.iter().map(|value| format!("{value:.3}")).collect();
        println!(
            "  Token {position}: [{}]  Mittelwert {mean:.3}, Varianz {variance:.3}",
            numbers.join(", ")
        );
    }
    Ok(())
}
