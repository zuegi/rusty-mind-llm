use burn::Tensor;
use burn::prelude::Device;
use burn::tensor::Int;
use model::gpt::GptModelConfig;
use training::checkpoint::load_model;

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
    select_next_token(logits)
}

fn select_next_token(logits: Tensor<3>) -> i32 {
    let [batch_size, position_count, vocab_size] = logits.dims();
    assert_eq!(batch_size, 1, "Generierung erwartet ein Fenster");
    assert!(position_count > 0, "Generierung braucht einen Kontext");
    let last_logits = logits.slice([
        0..batch_size,
        (position_count - 1)..position_count,
        0..vocab_size,
    ]);
    last_logits.argmax(2).into_scalar::<i32>()
}

fn generate(
    model: &model::gpt::GptModel,
    mut token_ids: Vec<i32>,
    new_tokens: usize,
    context_length: usize,
    device: &Device,
) -> Vec<i32> {
    assert!(!token_ids.is_empty(), "Generierung braucht Start-Token");
    for _ in 0..new_tokens {
        let context = context_window(&token_ids, context_length);
        let next_id = next_token(model, context, device);
        println!("context: {:?}", context);
        token_ids.push(next_id);
        println!("token_ids: {:?}", token_ids);
    }
    token_ids
}

fn main() -> Demo {
    let device = Device::flex();
    let config = GptModelConfig::new(
        VOCAB_SIZE,
        CONTEXT_LENGTH,
        EMBEDDING_DIM,
        NUM_HEADS,
        NUM_BLOCKS,
    );
    let model = match std::env::args_os().nth(1) {
        Some(path) => load_model(&config, path, &device)?,
        None => {
            println!("Kein Checkpoint: Demo mit zufälligen Gewichten.");
            config.init(&device)
        }
    };
    let token_ids = generate(
        &model,
        TOKEN_IDS[0].to_vec(),
        NEW_TOKENS,
        CONTEXT_LENGTH,
        &device,
    );

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

#[test]
fn selects_token_from_last_position() {
    let logits = Tensor::<3>::from_floats(
        [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 1.0, 2.0]]],
        &Device::flex(),
    );
    assert_eq!(select_next_token(logits), 2);
}

#[test]
fn generates_requested_tokens_with_bounded_context() {
    let device = Device::flex();
    let config = GptModelConfig::new(5, 3, 8, 2, 1);
    let model = config.init(&device);
    let prompt = vec![0, 1, 2, 3, 4];
    let expected = next_token(&model, &[2, 3, 4], &device);
    let tokens = generate(&model, prompt.clone(), 5, 3, &device);
    assert_eq!(tokens.len(), prompt.len() + 5);
    assert_eq!(&tokens[..prompt.len()], &prompt);
    assert_eq!(tokens[prompt.len()], expected);
    assert!(tokens.iter().all(|id| (0..5).contains(id)));
    assert_eq!(generate(&model, prompt.clone(), 0, 3, &device), prompt);
}

#[test]
fn loaded_checkpoint_generates_same_tokens() -> Demo {
    let device = Device::flex();
    let config = GptModelConfig::new(5, 3, 8, 2, 1);
    let model = config.init(&device);
    let expected = generate(&model, vec![0, 1, 3], 5, 3, &device);
    let path = std::env::temp_dir().join(format!("rusty-mind-llm-ch11-{}.bpk", std::process::id()));
    training::checkpoint::save_model(&model, &path)?;
    let loaded = load_model(&config, &path, &device)?;
    let actual = generate(&loaded, vec![0, 1, 3], 5, 3, &device);
    std::fs::remove_file(path)?;
    assert_eq!(actual, expected);
    Ok(())
}
