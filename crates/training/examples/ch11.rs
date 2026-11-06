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

fn context_window(token_ids: &[i32], context_length: usize) -> &[i32] {
    assert!(context_length > 0, "context_length muss größer als 0 sein");
    let start = token_ids.len().saturating_sub(context_length);
    &token_ids[start..]
}

fn next_token(model: &model::gpt::GptModel, context: &[i32], device: &Device) -> i32 {
    let context_tensor = Tensor::<1, Int>::from_ints(context, device);
    let ids = context_tensor.reshape([1, context.len()]);
    let logits = model.forward(ids);
    let [batch_size, position_count, vocab_size] = logits.dims();
    let last_logits = logits.slice([
        0..batch_size,
        (position_count - 1)..position_count,
        0..vocab_size,
    ]);
    last_logits.argmax(2).into_scalar::<i32>()
}

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
        let context = context_window(&token_ids, CONTEXT_LENGTH);
        let next_id = next_token(&model, context, &device);
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
fn keeps_only_the_latest_context_tokens() {
    let tokens = [0, 1, 3, 4, 4];
    assert_eq!(context_window(&tokens, CONTEXT_LENGTH), &[3, 4, 4]);
    assert_eq!(context_window(&tokens, 10), &tokens);
}
