use std::fs;

use burn::{store::RecordError, tensor::{Device, Int, Tensor}};
use model::gpt::GptModelConfig;
use training::checkpoint::{load_model, save_model};

#[test]
fn saved_weights_produce_same_output_after_loading() {
    let device = Device::ndarray();
    let config = GptModelConfig::new(5, 3, 8, 2, 1);
    let model = config.init(&device);
    let inputs = Tensor::<2, Int>::from_ints([[0, 1, 2]], &device);
    let expected = model.forward(inputs.clone()).into_data();
    let path =
        std::env::temp_dir().join(format!("rusty-mind-llm-ch10-test-{}.bpk", std::process::id()));

    save_model(&model, &path).unwrap();
    assert!(path.is_file());
    let loaded = load_model(&config, &path, &device).unwrap();
    let actual = loaded.forward(inputs).into_data();

    assert_eq!(expected, actual);
    fs::remove_file(path).unwrap();
}

fn checkpoint_path(label: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("rusty-mind-llm-ch10-{label}-{}.bpk", std::process::id()))
}

#[test]
fn missing_checkpoint_returns_io_error() {
    let config = GptModelConfig::new(5, 3, 8, 2, 1);
    let result = load_model(&config, checkpoint_path("missing"), &Device::ndarray());
    assert!(matches!(result, Err(RecordError::Io(_))));
}

#[test]
fn corrupt_checkpoint_returns_error() {
    let path = checkpoint_path("corrupt");
    fs::write(&path, b"not a burnpack file").unwrap();
    let config = GptModelConfig::new(5, 3, 8, 2, 1);
    let result = load_model(&config, &path, &Device::ndarray());
    fs::remove_file(path).unwrap();
    assert!(matches!(result, Err(RecordError::Io(_))));
}

#[test]
fn incompatible_checkpoint_returns_validation_error() {
    let device = Device::ndarray();
    let path = checkpoint_path("incompatible");
    let model = GptModelConfig::new(5, 3, 8, 2, 1).init(&device);
    save_model(&model, &path).unwrap();
    let config = GptModelConfig::new(6, 3, 8, 2, 1);
    let result = load_model(&config, &path, &device);
    fs::remove_file(path).unwrap();
    assert!(matches!(result, Err(RecordError::Validation(_))));
}

#[test]
fn saving_to_directory_returns_io_error() {
    let model = GptModelConfig::new(5, 3, 8, 2, 1).init(&Device::ndarray());
    let path = checkpoint_path("directory");
    fs::create_dir(&path).unwrap();
    let result = save_model(&model, &path);
    fs::remove_dir(path).unwrap();
    assert!(matches!(result, Err(RecordError::Io(_))));
}

#[test]
fn loading_on_autodiff_device_preserves_trainable_parameters() {
    let device = Device::ndarray();
    let config = GptModelConfig::new(5, 3, 8, 2, 1);
    let path = checkpoint_path("autodiff");
    save_model(&config.init(&device), &path).unwrap();
    let device = device.autodiff();
    let loaded = load_model(&config, &path, &device).unwrap();
    fs::remove_file(path).unwrap();
    let inputs = Tensor::<2, Int>::from_ints([[0, 1, 2]], &device);
    let gradients = loaded.forward(inputs).sum().backward();
    assert!(
        loaded.gpt_embedding.token.weight.val().grad(&gradients).is_some()
    );
    assert!(
        loaded.output_projection.weight.val().grad(&gradients).is_some()
    );
}
