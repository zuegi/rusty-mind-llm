//! Kapitel 7: Ein kleines GPT-Sprachmodell zusammensetzen.
//! Ausführen: cargo run -p model --example ch07

use burn::backend::NdArray;
use burn::module::Module;
use burn::tensor::{Int, Tensor};
use model::gpt::GptModelConfig;

type Cpu = NdArray<f32>;
type Demo = Result<(), Box<dyn std::error::Error>>;

const VOCAB_SIZE: usize = 50;
const CONTEXT_LENGTH: usize = 4;
const EMBEDDING_DIM: usize = 8;
const NUM_HEADS: usize = 2;
const NUM_BLOCKS: usize = 2;
const TOKEN_IDS: [[i32; CONTEXT_LENGTH]; 1] = [[1, 2, 3, 4]];

fn main() -> Demo {
    let device = Default::default();
    let model = GptModelConfig::new(
        VOCAB_SIZE,
        CONTEXT_LENGTH,
        EMBEDDING_DIM,
        NUM_HEADS,
        NUM_BLOCKS,
    )
    .init::<Cpu>(&device);
    let ids = Tensor::<Cpu, 2, Int>::from_data(TOKEN_IDS, &device);
    let embedded = model.gpt_embedding.forward(ids.clone());
    let logits = model.forward(ids);

    let values = logits.clone().into_data().to_vec::<f32>()?;
    let [batch_size, context_length, vocab_size] = logits.dims();
    println!("Eingabe-Form: [{batch_size}, {context_length}]");
    println!("GptEmbedding:  {:?}", embedded.dims());
    println!("Logits-Form:  [{batch_size}, {context_length}, {vocab_size}]");
    println!("Parameter:    {}", model.num_params());
    println!(
        "Erste 5 Logits für Position 3: {:?}",
        &values[3 * VOCAB_SIZE..3 * VOCAB_SIZE + 5]
    );
    Ok(())
}
