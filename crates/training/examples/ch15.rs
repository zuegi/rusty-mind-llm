//! Kapitel 15: Adam-Training über Prozessgrenzen hinweg fortsetzen.
//! Aufruf: ch15 train <neuer_ordner> <bis_update>
//!         ch15 resume <checkpoint_ordner> <neuer_ordner> <bis_update>

use std::{fs, io::Write, path::Path};

use burn::{
    module::{Module, list_param_ids},
    optim::{AdamConfig, ModuleOptimizer},
    tensor::Device,
};
use model::gpt::{GptModel, GptModelConfig};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use training::{
    checkpoint::{load_model, save_model},
    dataset::{Batch, Dataset},
    schedule::WarmupDecay,
    step::train_step,
    tokenizer::{Gpt2Tokenizer, VOCAB_SIZE},
};

type Demo<T = ()> = Result<T, Box<dyn std::error::Error>>;

const FORMAT_VERSION: u32 = 1;
const COMPATIBILITY: &str = "ch15-burn-0.22-rand-0.10-v1";
const CONTEXT_LENGTH: usize = 8;
const EMBEDDING_DIM: usize = 8;
const NUM_HEADS: usize = 2;
const NUM_BLOCKS: usize = 1;
const BATCH_SIZE: usize = 2;
const TRAIN_WINDOWS: usize = 8;
const EPOCHS: usize = 3;
const SEED: u64 = 42;
const PEAK_RATE: f64 = 0.001;
const WARMUP_UPDATES: usize = 4;
const FINAL_RATE_FRACTION: f64 = 0.1;
const ADAM_MOMENT_TENSORS: usize = 2;
const MODEL_FILE: &str = "model.bpk";
const OPTIMIZER_FILE: &str = "optimizer.bpk";
const STATE_FILE: &str = "state.json";
const DEFAULT_TEXT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/the-verdict.txt");
const USAGE: &str = "Aufruf: ch15 train <neuer_ordner> <bis_update> | resume <checkpoint_ordner> <neuer_ordner> <bis_update>";

// ANCHOR: state
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Settings {
    model: ModelSettings,
    tokens: Vec<u32>,
    batch_size: usize,
    epochs: usize,
    seed: u64,
    adam: AdamConfig,
    schedule: WarmupDecay,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Snapshot {
    version: u32,
    compatibility: String,
    settings: Settings,
    completed_updates: usize,
    model_sha256: String,
    optimizer_sha256: String,
}

struct Training {
    model: GptModel,
    optimizer: ModuleOptimizer,
    snapshot: Snapshot,
}
// ANCHOR_END: state

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelSettings {
    vocab_size: usize,
    context_length: usize,
    embedding_dim: usize,
    num_heads: usize,
    num_blocks: usize,
}

impl ModelSettings {
    fn config(&self) -> GptModelConfig {
        GptModelConfig::new(
            self.vocab_size,
            self.context_length,
            self.embedding_dim,
            self.num_heads,
            self.num_blocks,
        )
    }

    fn validate(&self) -> Demo {
        if self.vocab_size == 0
            || self.context_length == 0
            || self.embedding_dim == 0
            || self.num_heads == 0
            || self.num_blocks == 0
            || !self.embedding_dim.is_multiple_of(self.num_heads)
        {
            return Err("Ungültige Modellkonfiguration".into());
        }
        Ok(())
    }
}

impl Settings {
    fn new(model: ModelSettings, tokens: Vec<u32>) -> Self {
        Self {
            model,
            tokens,
            batch_size: BATCH_SIZE,
            epochs: EPOCHS,
            seed: SEED,
            adam: AdamConfig::new(),
            schedule: WarmupDecay {
                peak_rate: PEAK_RATE,
                warmup_updates: WARMUP_UPDATES,
                final_fraction: FINAL_RATE_FRACTION,
            },
        }
    }

    fn dataset(&self) -> Demo<Dataset> {
        Ok(Dataset::new(&self.tokens, self.model.context_length)?)
    }

    fn total_updates(&self) -> usize {
        self.epochs * ((self.tokens.len() - 1) / self.model.context_length / self.batch_size)
    }

    fn validate(&self) -> Demo {
        self.model.validate()?;
        self.validate_data()?;
        let batches = self.dataset()?.len() / self.batch_size;
        batches
            .checked_mul(self.epochs)
            .filter(|&n| n > 0)
            .ok_or("Keine vollständigen Batches oder zu viele Updates")?;
        self.seed
            .checked_add(self.epochs as u64)
            .ok_or("Epochen-Seed läuft über")?;
        Ok(self.schedule.validate()?)
    }

    fn validate_data(&self) -> Demo {
        let valid_tokens = self
            .tokens
            .iter()
            .all(|&token| (token as usize) < self.model.vocab_size);
        if self.batch_size == 0 || self.epochs == 0 || !valid_tokens {
            return Err("Ungültige Trainingsdaten oder Batch-/Epochenzahl".into());
        }
        if self.model.context_length >= self.tokens.len() {
            return Err("Zu wenige Token für ein Trainingsfenster".into());
        }
        Ok(())
    }
}

impl Snapshot {
    fn new(settings: Settings) -> Self {
        Self {
            version: FORMAT_VERSION,
            compatibility: COMPATIBILITY.into(),
            settings,
            completed_updates: 0,
            model_sha256: String::new(),
            optimizer_sha256: String::new(),
        }
    }
}

fn model_settings() -> ModelSettings {
    ModelSettings {
        vocab_size: VOCAB_SIZE,
        context_length: CONTEXT_LENGTH,
        embedding_dim: EMBEDDING_DIM,
        num_heads: NUM_HEADS,
        num_blocks: NUM_BLOCKS,
    }
}

fn settings() -> Demo<Settings> {
    let text = fs::read_to_string(DEFAULT_TEXT)?;
    let tokens = Gpt2Tokenizer::new()
        .encode(&text)
        .into_iter()
        .take(TRAIN_WINDOWS * CONTEXT_LENGTH + 1)
        .collect();
    Ok(Settings::new(model_settings(), tokens))
}

impl Training {
    fn new(settings: Settings) -> Demo<Self> {
        settings.validate()?;
        let device = Device::flex().autodiff();
        device.seed(settings.seed);
        let model = settings.model.config().init(&device);
        let optimizer = settings.adam.init();
        let snapshot = Snapshot::new(settings);
        Ok(Self {
            model,
            optimizer,
            snapshot,
        })
    }

    // ANCHOR: resume_batches
    fn next_batch(&self, dataset: &Dataset) -> Batch {
        let settings = &self.snapshot.settings;
        let per_epoch = dataset.len() / settings.batch_size;
        let epoch = self.snapshot.completed_updates / per_epoch;
        let offset = self.snapshot.completed_updates % per_epoch;
        let seed = settings.seed + epoch as u64 + 1;
        dataset.epoch(settings.batch_size, seed).remove(offset)
    }
    // ANCHOR_END: resume_batches

    // ANCHOR: resume_update
    fn update(mut self, dataset: &Dataset) -> Demo<Self> {
        let batch = self.next_batch(dataset);
        let next = self.snapshot.completed_updates + 1;
        let settings = &self.snapshot.settings;
        let rate = settings.schedule.rate(next, settings.total_updates());
        let (inputs, targets) = batch.to_tensors(&Device::flex().autodiff());
        let (model, loss) = train_step(self.model, inputs, targets, rate, &mut self.optimizer);
        let loss = loss.into_scalar::<f32>();
        if !loss.is_finite() {
            return Err("Nicht endlicher Trainings-Loss; Checkpoint wird nicht gespeichert".into());
        }
        self.model = model;
        self.snapshot.completed_updates = next;
        println!("Update {next}: Lernrate {rate:.8}, Loss {loss:.6}");
        Ok(self)
    }
    // ANCHOR_END: resume_update

    fn advance(mut self, until: usize) -> Demo<Self> {
        let total = self.snapshot.settings.total_updates();
        if until <= self.snapshot.completed_updates || until > total {
            return Err(format!(
                "Ziel muss nach dem aktuellen Update und höchstens bei {total} liegen"
            )
            .into());
        }
        let dataset = self.snapshot.settings.dataset()?;
        while self.snapshot.completed_updates < until {
            self = self.update(&dataset)?;
        }
        Ok(self)
    }
}

fn write_new(path: &Path, bytes: &[u8]) -> Demo {
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn digest(path: &Path) -> Demo<String> {
    Ok(format!("{:x}", Sha256::digest(fs::read(path)?)))
}

// ANCHOR: save
fn save(mut training: Training, path: &Path) -> Demo {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    fs::create_dir(path)?;
    save_model(&training.model.valid(), path.join(MODEL_FILE))?;
    training.optimizer.save(path.join(OPTIMIZER_FILE))?;
    for name in [MODEL_FILE, OPTIMIZER_FILE] {
        fs::File::open(path.join(name))?.sync_all()?;
    }
    training.snapshot.model_sha256 = digest(&path.join(MODEL_FILE))?;
    training.snapshot.optimizer_sha256 = digest(&path.join(OPTIMIZER_FILE))?;
    write_new(
        &path.join(STATE_FILE),
        &serde_json::to_vec_pretty(&training.snapshot)?,
    )?;
    println!("Trainingszustand gespeichert: {}", path.display());
    Ok(())
}
// ANCHOR_END: save

fn read_snapshot(path: &Path) -> Demo<Snapshot> {
    let snapshot: Snapshot = serde_json::from_slice(&fs::read(path.join(STATE_FILE))?)?;
    if snapshot.version != FORMAT_VERSION || snapshot.compatibility != COMPATIBILITY {
        return Err("Inkompatibler Trainings-Checkpoint".into());
    }
    snapshot.settings.validate()?;
    if snapshot.completed_updates == 0
        || snapshot.completed_updates > snapshot.settings.total_updates()
    {
        return Err("Ungültiger Update-Zähler im Checkpoint".into());
    }
    if digest(&path.join(MODEL_FILE))? != snapshot.model_sha256
        || digest(&path.join(OPTIMIZER_FILE))? != snapshot.optimizer_sha256
    {
        return Err("Checkpoint unvollständig oder verändert: SHA-256 stimmt nicht".into());
    }
    Ok(snapshot)
}

// ANCHOR: load
fn load(path: &Path) -> Demo<Training> {
    let snapshot = read_snapshot(path)?;
    let device = Device::flex().autodiff();
    let model = load_model(
        &snapshot.settings.model.config(),
        path.join(MODEL_FILE),
        &device,
    )?;
    let optimizer = load_optimizer(&snapshot.settings, path, &model)?;
    Ok(Training {
        model,
        optimizer,
        snapshot,
    })
}
// ANCHOR_END: load

fn load_optimizer(settings: &Settings, path: &Path, model: &GptModel) -> Demo<ModuleOptimizer> {
    let optimizer = settings.adam.init().load(path.join(OPTIMIZER_FILE))?;
    let expected_tensors = list_param_ids(model).len() * ADAM_MOMENT_TENSORS;
    if optimizer.to_record().len() != expected_tensors {
        return Err("Adam-Zustand ist nicht vollständig; kein stiller Neustart".into());
    }
    Ok(optimizer)
}

fn main() -> Demo {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (training, destination, until) = match args.as_slice() {
        [command, destination, until] if command == "train" => {
            (Training::new(settings()?)?, destination, until)
        }
        [command, source, destination, until] if command == "resume" => {
            (load(Path::new(source))?, destination, until)
        }
        _ => return Err(USAGE.into()),
    };
    let destination = Path::new(destination);
    if destination.exists() {
        return Err(
            "Zielordner existiert bereits; vorhandene Checkpoints bleiben unverändert".into(),
        );
    }
    save(training.advance(until.parse()?)?, destination)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::{
        module::{ModuleVisitor, Param},
        tensor::{Tensor, TensorData},
    };
    use std::sync::Mutex;

    static TRAINING_LOCK: Mutex<()> = Mutex::new(());

    fn fixture() -> Settings {
        let model = ModelSettings {
            vocab_size: 5,
            context_length: 2,
            ..model_settings()
        };
        let tokens = vec![0, 1, 4, 2, 1, 3, 0, 4, 2, 3, 4, 1, 2, 0, 3, 1, 4];
        Settings::new(model, tokens)
    }

    #[derive(Default)]
    struct Weights(Vec<TensorData>);

    impl ModuleVisitor for Weights {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
            self.0.push(param.val().into_data());
        }
    }

    fn weights(model: &GptModel) -> Vec<TensorData> {
        let mut weights = Weights::default();
        model.visit(&mut weights);
        weights.0
    }

    fn roundtrip(partial: Training) -> Demo<(Training, tempfile::TempDir)> {
        let completed = partial.snapshot.completed_updates;
        let expected_batch = partial.next_batch(&partial.snapshot.settings.dataset()?);
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("checkpoint");
        save(partial, &path)?;
        let restored = load(&path)?;
        assert_eq!(restored.snapshot.completed_updates, completed);
        assert_eq!(
            restored.next_batch(&restored.snapshot.settings.dataset()?),
            expected_batch
        );
        Ok((restored, directory))
    }

    #[test]
    fn resume_matches_all_weights_inside_and_between_epochs() -> Demo {
        let _lock = TRAINING_LOCK.lock().unwrap();
        let initial = Training::new(fixture())?;
        let baseline = initial.model.clone();
        let expected = initial.advance(12)?;
        for interruption in [3, 4, 5] {
            let mut partial = Training::new(fixture())?;
            partial.model = baseline.clone();
            let (restored, _directory) = roundtrip(partial.advance(interruption)?)?;
            let actual = restored.advance(12)?;
            assert_eq!(weights(&expected.model), weights(&actual.model));
        }
        Ok(())
    }

    #[test]
    fn resetting_adam_is_not_resume() -> Demo {
        let _lock = TRAINING_LOCK.lock().unwrap();
        let partial = Training::new(fixture())?.advance(3)?;
        let mut reset = Training::new(fixture())?;
        reset.model = partial.model.clone();
        reset.snapshot.completed_updates = partial.snapshot.completed_updates;
        assert_ne!(
            weights(&partial.advance(4)?.model),
            weights(&reset.advance(4)?.model)
        );
        Ok(())
    }

    #[test]
    fn incomplete_corrupt_or_incompatible_checkpoints_are_errors() -> Demo {
        let _lock = TRAINING_LOCK.lock().unwrap();
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("checkpoint");
        save(Training::new(fixture())?.advance(3)?, &path)?;
        let optimizer = fs::read(path.join(OPTIMIZER_FILE))?;
        fs::write(path.join(OPTIMIZER_FILE), b"corrupt")?;
        let error = load(&path)
            .err()
            .ok_or("Beschädigter Checkpoint wurde geladen")?;
        assert!(error.to_string().contains("SHA-256"));
        fs::write(path.join(OPTIMIZER_FILE), optimizer)?;
        let mut snapshot = read_snapshot(&path)?;
        snapshot.version += 1;
        fs::write(path.join(STATE_FILE), serde_json::to_vec(&snapshot)?)?;
        assert!(load(&path).is_err());
        fs::remove_file(path.join(STATE_FILE))?;
        assert!(load(&path).is_err());
        Ok(())
    }

    #[test]
    fn missing_adam_moments_are_not_silently_reset() -> Demo {
        let _lock = TRAINING_LOCK.lock().unwrap();
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("checkpoint");
        save(Training::new(fixture())?.advance(3)?, &path)?;
        let mut snapshot = read_snapshot(&path)?;
        AdamConfig::new().init().save(path.join(OPTIMIZER_FILE))?;
        snapshot.optimizer_sha256 = digest(&path.join(OPTIMIZER_FILE))?;
        fs::write(path.join(STATE_FILE), serde_json::to_vec(&snapshot)?)?;
        let error = load(&path)
            .err()
            .ok_or("Leerer Adam-Zustand wurde geladen")?;
        assert!(
            error
                .to_string()
                .contains("Adam-Zustand ist nicht vollständig")
        );
        Ok(())
    }

    #[test]
    fn invalid_targets_and_overwrites_are_errors() -> Demo {
        let _lock = TRAINING_LOCK.lock().unwrap();
        assert!(Training::new(fixture())?.advance(0).is_err());
        assert!(Training::new(fixture())?.advance(13).is_err());
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("checkpoint");
        save(Training::new(fixture())?.advance(3)?, &path)?;
        let original = fs::read(path.join(STATE_FILE))?;
        assert!(save(Training::new(fixture())?.advance(4)?, &path).is_err());
        assert_eq!(fs::read(path.join(STATE_FILE))?, original);
        assert!(load(&path)?.advance(3).is_err());
        Ok(())
    }
}
