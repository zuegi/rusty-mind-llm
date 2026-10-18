use burn::Tensor;
use burn::prelude::Device;
use burn::tensor::Int;
use model::gpt::GptModelConfig;

type Demo = Result<(), Box<dyn std::error::Error>>;

const VOCAB_SIZE: usize = 5;
const CONTEXT_LENGTH: usize = 3;
const EMBEDDING_DIM: usize = 8;
const NUM_HEADS: usize = 2;
const NUM_BLOCKS: usize = 1;
const TOKEN_IDS: [[i32; CONTEXT_LENGTH]; 1] = [[0, 1, 3]];
const NEW_TOKENS: usize = 5;
const VOCABULARY: [&str; VOCAB_SIZE] = ["Der", "Hund", "Katze", "rennt", "."];

fn main() -> Demo {
    let device = Device::flex();
    let model = GptModelConfig::new(
        VOCAB_SIZE,
        CONTEXT_LENGTH,
        EMBEDDING_DIM,
        NUM_HEADS,
        NUM_BLOCKS,
    )
    .init(&device);

    let mut token_ids = TOKEN_IDS[0].to_vec();
    for _ in 0..NEW_TOKENS {
        // Tensor erstellen
        let context_start = token_ids.len().saturating_sub(CONTEXT_LENGTH);
        let context = &token_ids[context_start..];
        assert!(context.len() <= CONTEXT_LENGTH);
        let context_tensor = Tensor::<1, Int>::from_ints(context, &device);
        let ids = context_tensor.reshape([1, context.len()]);

        // forward
        let logits = model.forward(ids);
        let [batch_size, position_count, vocab_size] = logits.dims();

        // letzte Position + argmax
        let last_logits = logits.slice([
            0..batch_size,
            (position_count - 1)..position_count,
            0..vocab_size,
        ]);
        let next_token = last_logits.argmax(2);

        // id anhängen
        let next_id = next_token.into_scalar::<i32>();
        println!("context: {:?}", context);
        token_ids.push(next_id);
        println!("token_ids: {:?}", token_ids);

    }

    assert_eq!(token_ids.len(), TOKEN_IDS[0].len() + NEW_TOKENS);

    for token_id in &token_ids {
        print!("{} ", VOCABULARY[*token_id as usize]);
    }
    println!();
    Ok(())
}

#[test]
fn selects_token_from_last_position() {
    // Feste Logits, Auswahl, erwartete ID vergleichen.
}