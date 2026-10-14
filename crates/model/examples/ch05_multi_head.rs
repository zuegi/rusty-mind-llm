//! Kapitel 5, Schritt 4: Multi-Head Attention mit Burn.
//! Ausführen: cargo run -p model --example ch05_multi_head

use burn::tensor::{Device, Tensor};
use model::attention::{MultiHeadAttentionConfig, NaiveMultiHeadAttentionConfig};

type Demo = Result<(), Box<dyn std::error::Error>>;

const EMBEDDING_DIM: usize = 4;
const NUM_HEADS: usize = 2;
const INPUT: [[[f32; EMBEDDING_DIM]; 2]; 1] = [
    [[1.0, 0.0, 0.5, -0.5], [0.0, 1.0, -0.5, 0.5]],
];

fn main() -> Demo {
    let device = Device::flex();
    let input = Tensor::<3>::from_data(INPUT, &device);

    // 1. Didaktische Variante mit einzelnen Köpfen
    let naive = NaiveMultiHeadAttentionConfig::new(EMBEDDING_DIM, NUM_HEADS).init(&device);
    print_vectors("Eingabe", input.clone())?;

    for (index, head) in naive.heads.iter().enumerate() {
        let head_output = head.forward(input.clone());
        print_vectors(&format!("Didaktisch: Kopf {index} (Head-Dim 2)"), head_output)?;
    }

    let naive_out = naive.forward(input.clone());
    print_vectors("Didaktisch: Gesamtausgabe", naive_out)?;

    // 2. Produktive / optimierte 4D-Batched-Variante
    let batched = MultiHeadAttentionConfig::new(EMBEDDING_DIM, NUM_HEADS).init(&device);
    let batched_out = batched.forward(input);
    print_vectors("Optimiert: Gesamtausgabe (4D-Batched)", batched_out)?;

    Ok(())
}

fn print_vectors(label: &str, tensor: Tensor<3>) -> Demo {
    let [batch_size, context_length, dim] = tensor.dims();
    let values = tensor.into_data().try_to_vec::<f32>()?;
    println!("{label}, Form [{batch_size}, {context_length}, {dim}]:");
    for (position, vector) in values.chunks(dim).enumerate() {
        let numbers: Vec<_> = vector.iter().map(|value| format!("{value:.3}")).collect();
        println!("  Token {position}: [{}]", numbers.join(", "));
    }
    Ok(())
}
