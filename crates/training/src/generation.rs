use burn::tensor::{Device, Int, Tensor};
use model::gpt::GptModel;

pub fn context_window(token_ids: &[i32], context_length: usize) -> &[i32] {
    assert!(context_length > 0, "context_length muss größer als 0 sein");
    let start = token_ids.len().saturating_sub(context_length);
    &token_ids[start..]
}

pub fn next_token(model: &GptModel, context: &[i32], device: &Device) -> i32 {
    assert!(!context.is_empty(), "Generierung braucht einen Kontext");
    let context_tensor = Tensor::<1, Int>::from_ints(context, device);
    let ids = context_tensor.reshape([1, context.len()]);
    select_next_token(model.forward(ids))
}

pub fn select_next_token(logits: Tensor<3>) -> i32 {
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
}
