use std::path::Path;

use burn::{module::Module, store::RecordError, tensor::Device};
use model::gpt::{GptModel, GptModelConfig};

/// Speichert die Modellgewichte als Burnpack-Datei am angegebenen Pfad.
pub fn save_model(model: &GptModel, path: impl AsRef<Path>) -> Result<(), RecordError> {
    model.clone().save_file(path)
}

/// Initialisiert Modell mit `config` und lädt danach gespeicherte Gewichte.
pub fn load_model(
    config: &GptModelConfig,
    path: impl AsRef<Path>,
    device: &Device,
) -> Result<GptModel, RecordError> {
    let model = config.init(device);
    model.try_load_file(path)
}
