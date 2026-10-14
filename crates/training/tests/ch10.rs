use std::fs;

use burn::{
    backend::NdArray,
    tensor::{Int, Tensor},
};
use model::gpt::GptModelConfig;
use training::checkpoint::{load_model, save_model};

type Cpu = NdArray<f32>;

#[test]
fn saved_weights_produce_same_output_after_loading() {
    let device = Default::default();
    let config = GptModelConfig::new(5, 3, 8, 2, 1);
    let model = config.init::<Cpu>(&device);
    let inputs = Tensor::<Cpu, 2, Int>::from_ints([[0, 1, 2]], &device);
    let expected = model.forward(inputs.clone()).into_data();
    let path =
        std::env::temp_dir().join(format!("rusty-mind-llm-ch10-test-{}", std::process::id()));

    save_model(&model, &path).unwrap();
    let loaded = load_model::<Cpu>(&config, &path, &device).unwrap();
    let actual = loaded.forward(inputs).into_data();

    assert_eq!(expected, actual);
    fs::remove_file(path.with_extension("mpk")).unwrap();
}
