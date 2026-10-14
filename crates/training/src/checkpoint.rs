use std::path::Path;

use burn::{
    module::Module,
    record::{DefaultRecorder, RecorderError},
    tensor::backend::Backend,
};
use model::gpt::{GptModel, GptModelConfig};

/// Speichert die Modellgewichte als Burn-Record.
pub fn save_model<B: Backend>(
    model: &GptModel<B>,
    path: impl AsRef<Path>,
) -> Result<(), RecorderError> {
    model
        .clone()
        .save_file(path.as_ref(), &DefaultRecorder::new())
}

/// Initialisiert Modell mit `config` und lädt danach gespeicherte Gewichte.
pub fn load_model<B: Backend>(
    config: &GptModelConfig,
    path: impl AsRef<Path>,
    device: &B::Device,
) -> Result<GptModel<B>, RecorderError> {
    let model = config.init(device);
    model.load_file(path.as_ref(), &DefaultRecorder::new(), device)
}
