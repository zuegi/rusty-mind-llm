//! Kapitel 6, Schritt 6: Transformer-Blöcke hinter dem Embedding.
//! Ausführen: cargo run -p model --example ch06

use burn::module::Module;
use burn::tensor::{Device, Int, Tensor};
use model::embedding::GptEmbeddingConfig;
use model::transformer_block::TransformerBlockConfig;

type Demo = Result<(), Box<dyn std::error::Error>>;

const VOCAB_SIZE: usize = 50;
const CONTEXT_LENGTH: usize = 4;
const EMBEDDING_DIM: usize = 8;
const NUM_HEADS: usize = 2;
const NUM_BLOCKS: usize = 3;
const SHOWN_VALUES: usize = 4;
const TOKEN_IDS: [[i32; CONTEXT_LENGTH]; 1] = [[1, 2, 3, 4]];
const CHANGED_LAST_TOKEN: [[i32; CONTEXT_LENGTH]; 1] = [[1, 2, 3, 9]];

fn main() -> Demo {
    let device = Device::flex();
    let embedding =
        GptEmbeddingConfig::new(VOCAB_SIZE, CONTEXT_LENGTH, EMBEDDING_DIM).init(&device);
    let blocks: Vec<_> = (0..NUM_BLOCKS)
        .map(|_| TransformerBlockConfig::new(EMBEDDING_DIM, NUM_HEADS).init(&device))
        .collect();

    let ids = Tensor::<2, Int>::from_data(TOKEN_IDS, &device);
    let embedded = embedding.forward(ids);
    show("Embedding x", &embedded)?;

    let mut hidden = embedded;
    for (index, block) in blocks.iter().enumerate() {
        hidden = block.forward(hidden);
        show(&format!("nach Block {}", index + 1), &hidden)?;
    }

    let params: usize = blocks.iter().map(|block| block.num_params()).sum();
    println!("Parameter in {NUM_BLOCKS} Blöcken: {params}");
    print_causality(&embedding, &blocks)
}

// ANCHOR: stack
fn run(
    embedding: &model::embedding::GptEmbedding,
    blocks: &[model::transformer_block::TransformerBlock],
    token_ids: [[i32; CONTEXT_LENGTH]; 1],
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let ids = Tensor::<2, Int>::from_data(token_ids, &Device::flex());
    let output = blocks
        .iter()
        .fold(embedding.forward(ids), |hidden, block| block.forward(hidden));
    Ok(output.into_data().try_to_vec::<f32>()?)
}
// ANCHOR_END: stack

fn print_causality(
    embedding: &model::embedding::GptEmbedding,
    blocks: &[model::transformer_block::TransformerBlock],
) -> Demo {
    let before = run(embedding, blocks, TOKEN_IDS)?;
    let after = run(embedding, blocks, CHANGED_LAST_TOKEN)?;
    println!("Letztes Token geändert (4 → 9):");
    for (position, (old, new)) in before
        .chunks(EMBEDDING_DIM)
        .zip(after.chunks(EMBEDDING_DIM))
        .enumerate()
    {
        let same = old.iter().zip(new).all(|(a, b)| (a - b).abs() < 1e-5);
        println!("  Token {position}: Ausgabe unverändert: {same}");
    }
    Ok(())
}

fn show(label: &str, tensor: &Tensor<3>) -> Demo {
    let values = tensor.clone().into_data().try_to_vec::<f32>()?;
    let shown: Vec<_> = values[..SHOWN_VALUES]
        .iter()
        .map(|value| format!("{value:.3}"))
        .collect();
    println!("{label}, Form {:?}, Token 0 (erste {SHOWN_VALUES}): [{}]", tensor.dims(), shown.join(", "));
    Ok(())
}
