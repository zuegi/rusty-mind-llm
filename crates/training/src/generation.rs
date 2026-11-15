use burn::tensor::{Device, Int, Tensor};
use model::gpt::GptModel;
use rand::{SeedableRng, rngs::StdRng};

use crate::sampling::{SamplingError, SamplingStrategy};

pub fn context_window(token_ids: &[i32], context_length: usize) -> &[i32] {
    assert!(context_length > 0, "context_length muss größer als 0 sein");
    let start = token_ids.len().saturating_sub(context_length);
    &token_ids[start..]
}

pub fn next_token(model: &GptModel, context: &[i32], device: &Device) -> i32 {
    select_next_token(context_logits(model, context, device))
}

fn context_logits(model: &GptModel, context: &[i32], device: &Device) -> Tensor<3> {
    assert!(!context.is_empty(), "Generierung braucht einen Kontext");
    let context_tensor = Tensor::<1, Int>::from_ints(context, device);
    let ids = context_tensor.reshape([1, context.len()]);
    model.forward(ids)
}

pub fn select_next_token(logits: Tensor<3>) -> i32 {
    last_position_logits(logits).argmax(2).into_scalar::<i32>()
}

fn last_position_logits(logits: Tensor<3>) -> Tensor<3> {
    let [batch_size, position_count, vocab_size] = logits.dims();
    assert_eq!(batch_size, 1, "Generierung erwartet ein Fenster");
    assert!(position_count > 0, "Generierung braucht einen Kontext");
    logits.slice([
        0..batch_size,
        (position_count - 1)..position_count,
        0..vocab_size,
    ])
}

/// Greedy-Generierung; die zurückgegebene Folge enthält auch den Startkontext.
pub fn generate(
    model: &GptModel,
    mut token_ids: Vec<i32>,
    new_tokens: usize,
    context_length: usize,
    device: &Device,
) -> Vec<i32> {
    assert!(!token_ids.is_empty(), "Generierung braucht Start-Token");
    assert!(context_length > 0, "context_length muss größer als 0 sein");
    for _ in 0..new_tokens {
        let context = context_window(&token_ids, context_length);
        token_ids.push(next_token(model, context, device));
    }
    token_ids
}

// ANCHOR: sampling_options
#[derive(Debug, Clone, Copy)]
pub struct GenerationOptions {
    pub strategy: SamplingStrategy,
    pub seed: u64,
    pub max_new_tokens: usize,
    pub stop_token: Option<i32>,
}
// ANCHOR_END: sampling_options

// ANCHOR: sampled_generation
/// Enthält den Startkontext und ein erzeugtes Stopp-Token; stoppt nur auf neuen Tokens.
pub fn generate_with_options(
    model: &GptModel,
    mut token_ids: Vec<i32>,
    context_length: usize,
    options: GenerationOptions,
    device: &Device,
) -> Result<Vec<i32>, SamplingError> {
    validate_generation(model, &token_ids, context_length, options)?;
    let mut rng = StdRng::seed_from_u64(options.seed);
    for _ in 0..options.max_new_tokens {
        let context = context_window(&token_ids, context_length);
        let logits = context_values(model, context, device)?;
        let id = options.strategy.select(&logits, &mut rng)?;
        token_ids.push(id);
        if options.stop_token == Some(id) {
            break;
        }
    }
    Ok(token_ids)
}
// ANCHOR_END: sampled_generation

fn context_values(
    model: &GptModel,
    context: &[i32],
    device: &Device,
) -> Result<Vec<f32>, SamplingError> {
    last_position_logits(context_logits(model, context, device))
        .into_data()
        .try_to_vec::<f32>()
        .map_err(|error| SamplingError::new(error.to_string()))
}

fn validate_generation(
    model: &GptModel,
    token_ids: &[i32],
    context_length: usize,
    options: GenerationOptions,
) -> Result<(), SamplingError> {
    let vocab_size = model.output_projection.weight.dims()[1];
    options.strategy.validate(vocab_size)?;
    let max_context = model.gpt_embedding.position.weight.dims()[0];
    if token_ids.is_empty() || context_length == 0 || context_length > max_context {
        return Err(SamplingError::new(
            "Startkontext leer oder Kontextlänge außerhalb des Modells",
        ));
    }
    validate_token_ids(token_ids, options.stop_token, vocab_size)
}

fn validate_token_ids(
    token_ids: &[i32],
    stop_token: Option<i32>,
    vocab_size: usize,
) -> Result<(), SamplingError> {
    if token_ids
        .iter()
        .copied()
        .chain(stop_token)
        .any(|id| id < 0 || id as usize >= vocab_size)
    {
        return Err(SamplingError::new("Token-ID außerhalb des Vokabulars"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use model::gpt::GptModelConfig;

    #[test]
    fn generation_keeps_prompt_and_uses_latest_context() {
        let device = Device::flex();
        let model = GptModelConfig::new(5, 3, 8, 2, 1).init(&device);
        let prompt = vec![0, 1, 2, 3, 4];
        let expected = next_token(&model, &[2, 3, 4], &device);
        let output = generate(&model, prompt.clone(), 5, 3, &device);
        assert_eq!(&output[..prompt.len()], &prompt);
        assert_eq!(output.len(), prompt.len() + 5);
        assert_eq!(output[prompt.len()], expected);
        assert!(output.iter().all(|id| (0..5).contains(id)));
        assert_eq!(generate(&model, prompt.clone(), 0, 3, &device), prompt);
    }

    #[test]
    #[should_panic(expected = "Generierung braucht Start-Token")]
    fn rejects_empty_prompt() {
        let device = Device::flex();
        let model = GptModelConfig::new(5, 3, 8, 2, 1).init(&device);
        generate(&model, vec![], 1, 3, &device);
    }

    #[test]
    fn sampled_greedy_matches_existing_generation_and_stops_on_new_token() {
        let device = Device::flex();
        let model = GptModelConfig::new(5, 3, 8, 2, 1).init(&device);
        let prompt = vec![0, 1, 2, 3, 4];
        let options = GenerationOptions {
            strategy: SamplingStrategy::Greedy,
            seed: 42,
            max_new_tokens: 5,
            stop_token: None,
        };
        let expected = generate(&model, prompt.clone(), 5, 3, &device);
        assert_eq!(
            generate_with_options(&model, prompt.clone(), 3, options, &device).unwrap(),
            expected
        );
        let stopped = GenerationOptions {
            stop_token: Some(expected[prompt.len()]),
            ..options
        };
        let output = generate_with_options(&model, prompt.clone(), 3, stopped, &device).unwrap();
        assert_eq!(output, expected[..prompt.len() + 1]);
        let zero = GenerationOptions {
            max_new_tokens: 0,
            ..options
        };
        assert_eq!(
            generate_with_options(&model, prompt.clone(), 3, zero, &device).unwrap(),
            prompt
        );
    }

    #[test]
    fn sampled_generation_reuses_rng_and_latest_context() {
        use crate::sampling::SamplingFilter;
        let device = Device::flex();
        let model = GptModelConfig::new(5, 3, 8, 2, 1).init(&device);
        let options = GenerationOptions {
            strategy: SamplingStrategy::Multinomial {
                temperature: 1.0,
                filter: SamplingFilter::None,
            },
            seed: 42,
            max_new_tokens: 8,
            stop_token: None,
        };
        let mut expected = vec![0, 1, 2, 3, 4];
        let mut rng = StdRng::seed_from_u64(options.seed);
        let actual = generate_with_options(&model, expected.clone(), 3, options, &device).unwrap();
        for _ in 0..options.max_new_tokens {
            let context = context_window(&expected, 3);
            let input = Tensor::<1, Int>::from_ints(context, &device).reshape([1, context.len()]);
            let logits = last_position_logits(model.forward(input))
                .into_data()
                .try_to_vec::<f32>()
                .unwrap();
            expected.push(options.strategy.select(&logits, &mut rng).unwrap());
        }
        assert_eq!(actual, expected);
    }

    #[test]
    fn sampled_generation_rejects_invalid_context_and_ids() {
        let device = Device::flex();
        let model = GptModelConfig::new(5, 3, 8, 2, 1).init(&device);
        let options = GenerationOptions {
            strategy: SamplingStrategy::Greedy,
            seed: 42,
            max_new_tokens: 0,
            stop_token: None,
        };
        for (prompt, context) in [
            (vec![], 3),
            (vec![0], 0),
            (vec![0], 4),
            (vec![-1], 3),
            (vec![5], 3),
        ] {
            assert!(generate_with_options(&model, prompt, context, options, &device).is_err());
        }
        let invalid_stop = GenerationOptions {
            stop_token: Some(5),
            ..options
        };
        assert!(generate_with_options(&model, vec![0], 3, invalid_stop, &device).is_err());
    }
}
