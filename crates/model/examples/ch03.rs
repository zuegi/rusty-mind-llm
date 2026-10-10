//! Kapitel 3: Token- und Positions-Embedding mit nachrechenbaren Gewichten.
//! Ausführen: cargo run -p model --example ch03

use burn::backend::NdArray;
use burn::module::Param;
use burn::tensor::{Int, Tensor};
use model::embedding::{GptEmbedding, GptEmbeddingConfig};

type Cpu = NdArray<f32>;

const VOCAB_SIZE: usize = 5;
const MAX_CONTEXT_LENGTH: usize = 3;
const EMBEDDING_DIM: usize = 4;
const TOKEN_WEIGHTS: [[f32; EMBEDDING_DIM]; VOCAB_SIZE] = [
    [0.1, 0.0, 0.2, -0.1],
    [0.2, -0.1, 0.4, 0.3],
    [-0.2, 0.3, 0.1, 0.0],
    [0.0, 0.2, -0.3, 0.4],
    [0.1, -0.2, 0.0, 0.2],
];
const POSITION_WEIGHTS: [[f32; EMBEDDING_DIM]; MAX_CONTEXT_LENGTH] = [
    [0.0, 0.2, -0.1, 0.1],
    [0.1, 0.0, 0.2, 0.0],
    [0.0, -0.1, 0.0, 0.2],
];

fn example_embedding() -> GptEmbedding<Cpu> {
    let device = Default::default();
    let config = GptEmbeddingConfig::new(VOCAB_SIZE, MAX_CONTEXT_LENGTH, EMBEDDING_DIM);
    let mut embedding = config.init(&device);
    embedding.token.weight = Param::from_tensor(Tensor::from_data(TOKEN_WEIGHTS, &device));
    embedding.position.weight = Param::from_tensor(Tensor::from_data(POSITION_WEIGHTS, &device));
    embedding
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Default::default();
    let embedding = example_embedding();
    let token_ids = Tensor::<Cpu, 2, Int>::from_data([[1, 1, 3], [0, 2, 4]], &device);
    println!("Vokabular: 0=Der, 1=Hund, 2=Katze, 3=rennt, 4=.");
    println!("Eingabe: {:?}", token_ids.to_data().to_vec::<i64>()?);
    println!("Eingabeform: {:?}", token_ids.dims());
    print_vectors("Token-Vektoren", embedding.token.forward(token_ids.clone()))?;
    let positions = Tensor::<Cpu, 2, Int>::from_data([[0, 1, 2]], &device);
    print_vectors(
        "Positions-Vektoren (für jedes Fenster)",
        embedding.position.forward(positions),
    )?;
    print_vectors("Summe", embedding.forward(token_ids))?;
    Ok(())
}

fn print_vectors(label: &str, tensor: Tensor<Cpu, 3>) -> Result<(), Box<dyn std::error::Error>> {
    let [batch_size, context_length, embedding_dim] = tensor.dims();
    let values = tensor.into_data().to_vec::<f32>()?;
    println!("{label}, Form [{batch_size}, {context_length}, {embedding_dim}]:");
    for (batch, window) in values.chunks(context_length * embedding_dim).enumerate() {
        for (position, vector) in window.chunks(embedding_dim).enumerate() {
            let numbers: Vec<_> = vector.iter().map(|value| format!("{value:.1}")).collect();
            println!(
                "  Fenster {batch}, Position {position}: [{}]",
                numbers.join(", ")
            );
        }
    }
    Ok(())
}
