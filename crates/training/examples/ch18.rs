//! Kapitel 18: Vollständiger Trainingslauf mit AdamW, Lernratenplan, Dropout,
//! Validierung, Bestmodell und Resume.
//! Aufruf: ch18 train <neuer_ordner> <bis_update>
//!         ch18 resume <checkpoint_ordner> <neuer_ordner> <bis_update>

use std::{fs, io::Write, path::Path};

use burn::{
    module::{Module, list_param_ids},
    optim::{AdamWConfig, ModuleOptimizer},
    tensor::{Device, Int, Tensor},
};
use model::gpt::{GptModel, GptModelConfig};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use training::{
    checkpoint::{load_model, save_model},
    dataset::{Batch, Dataset, split_tokens},
    runner::evaluate,
    schedule::WarmupDecay,
    step::train_step,
    tokenizer::{Gpt2Tokenizer, VOCAB_SIZE},
};

type Demo<T = ()> = Result<T, Box<dyn std::error::Error>>;

const FORMAT_VERSION: u32 = 1;
const COMPATIBILITY: &str = "ch18-burn-0.22-rand-0.10-v1";
const CONTEXT_LENGTH: usize = 8;
const EMBEDDING_DIM: usize = 8;
const NUM_HEADS: usize = 2;
const NUM_BLOCKS: usize = 1;
const BATCH_SIZE: usize = 2;
const TRAIN_WINDOWS: usize = 8;
const VALIDATION_WINDOWS: usize = 2;
const VALIDATION_FRACTION: f64 = 0.1;
const EPOCHS: usize = 10;
const SEED: u64 = 42;
const PEAK_RATE: f64 = 0.005;
const WARMUP_UPDATES: usize = 4;
const FINAL_RATE_FRACTION: f64 = 0.1;
const WEIGHT_DECAY: f32 = 0.1;
const DROPOUT: f64 = 0.1;
const ADAM_MOMENT_TENSORS: usize = 2;
const LAST_DIR: &str = "last";
const BEST_DIR: &str = "best";
const MODEL_FILE: &str = "model.bpk";
const OPTIMIZER_FILE: &str = "optimizer.bpk";
const STATE_FILE: &str = "state.json";
const BEST_FILE: &str = "best.json";
const DEFAULT_TEXT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/the-verdict.txt");
const USAGE: &str = "Aufruf: ch18 train <neuer_ordner> <bis_update> | resume <checkpoint_ordner> <neuer_ordner> <bis_update>";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelSettings {
    vocab_size: usize,
    context_length: usize,
    embedding_dim: usize,
    num_heads: usize,
    num_blocks: usize,
    dropout: f64,
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
        .with_dropout(self.dropout)
    }

    fn validate(&self) -> Demo {
        let sizes = [
            self.vocab_size,
            self.context_length,
            self.embedding_dim,
            self.num_heads,
            self.num_blocks,
        ];
        let valid_dropout = (0.0..1.0).contains(&self.dropout);
        if sizes.contains(&0)
            || !self.embedding_dim.is_multiple_of(self.num_heads)
            || !valid_dropout
        {
            return Err("Ungültige Modellkonfiguration".into());
        }
        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Settings {
    model: ModelSettings,
    train_tokens: Vec<u32>,
    validation_tokens: Vec<u32>,
    batch_size: usize,
    epochs: usize,
    seed: u64,
    optimizer: AdamWConfig,
    schedule: WarmupDecay,
}

// ANCHOR: state
/// Bester Validierungsstand; steht in `state.json` und in `best/best.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct BestInfo {
    validation_loss: f64,
    update: usize,
    model: ModelSettings,
    model_sha256: String,
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
    best: Option<BestInfo>,
}

struct Training {
    model: GptModel,
    optimizer: ModuleOptimizer,
    snapshot: Snapshot,
    best_model: Option<GptModel>,
}
// ANCHOR_END: state

impl Settings {
    fn new(model: ModelSettings, train_tokens: Vec<u32>, validation_tokens: Vec<u32>) -> Self {
        Self {
            model,
            train_tokens,
            validation_tokens,
            batch_size: BATCH_SIZE,
            epochs: EPOCHS,
            seed: SEED,
            optimizer: AdamWConfig::new().with_weight_decay(WEIGHT_DECAY),
            schedule: WarmupDecay {
                peak_rate: PEAK_RATE,
                warmup_updates: WARMUP_UPDATES,
                final_fraction: FINAL_RATE_FRACTION,
            },
        }
    }

    fn train_dataset(&self) -> Demo<Dataset> {
        Ok(Dataset::new(&self.train_tokens, self.model.context_length)?)
    }

    fn validation_dataset(&self) -> Demo<Dataset> {
        Ok(Dataset::new(
            &self.validation_tokens,
            self.model.context_length,
        )?)
    }

    fn updates_per_epoch(&self) -> usize {
        (self.train_tokens.len() - 1) / self.model.context_length / self.batch_size
    }

    fn total_updates(&self) -> usize {
        self.epochs * self.updates_per_epoch()
    }

    fn validate(&self) -> Demo {
        self.model.validate()?;
        let in_vocab =
            |tokens: &[u32]| tokens.iter().all(|&t| (t as usize) < self.model.vocab_size);
        if self.batch_size == 0 || self.epochs == 0 {
            return Err("Ungültige Batch- oder Epochenzahl".into());
        }
        if !in_vocab(&self.train_tokens) || !in_vocab(&self.validation_tokens) {
            return Err("Token ausserhalb des Vokabulars".into());
        }
        let too_short = |tokens: &[u32]| tokens.len() <= self.model.context_length;
        if too_short(&self.train_tokens) || too_short(&self.validation_tokens) {
            return Err("Zu wenige Token für Trainings- oder Validierungsfenster".into());
        }
        if self.updates_per_epoch() == 0 {
            return Err("Keine vollständigen Trainingsbatches".into());
        }
        self.seed
            .checked_add(self.total_updates() as u64 + self.epochs as u64 + 1)
            .ok_or("Seed läuft über")?;
        Ok(self.schedule.validate()?)
    }
}

fn model_settings() -> ModelSettings {
    ModelSettings {
        vocab_size: VOCAB_SIZE,
        context_length: CONTEXT_LENGTH,
        embedding_dim: EMBEDDING_DIM,
        num_heads: NUM_HEADS,
        num_blocks: NUM_BLOCKS,
        dropout: DROPOUT,
    }
}

fn settings() -> Demo<Settings> {
    let text = fs::read_to_string(DEFAULT_TEXT)?;
    let tokens = Gpt2Tokenizer::new().encode(&text);
    let (train, validation) = split_tokens(&tokens, VALIDATION_FRACTION);
    let train = &train[..train.len().min(TRAIN_WINDOWS * CONTEXT_LENGTH + 1)];
    let validation = &validation[..validation
        .len()
        .min(VALIDATION_WINDOWS * CONTEXT_LENGTH + 1)];
    Ok(Settings::new(
        model_settings(),
        train.to_vec(),
        validation.to_vec(),
    ))
}

/// Burn erzeugt Parameter erst bei der ersten Verwendung. Hier, damit Update 1
/// keine Zufallszahlen für die Initialisierung verbraucht, die Resume nicht kennt.
fn materialize_parameters(model: &GptModel) {
    let input = Tensor::<2, Int>::zeros([1, 1], &Device::flex());
    model.valid().forward(input).into_data();
}

impl Training {
    fn new(settings: Settings) -> Demo<Self> {
        settings.validate()?;
        let device = Device::flex().autodiff();
        device.seed(settings.seed);
        let model = settings.model.config().init(&device);
        materialize_parameters(&model);
        Ok(Self {
            model,
            optimizer: settings.optimizer.init(),
            snapshot: Snapshot {
                version: FORMAT_VERSION,
                compatibility: COMPATIBILITY.into(),
                settings,
                completed_updates: 0,
                model_sha256: String::new(),
                optimizer_sha256: String::new(),
                best: None,
            },
            best_model: None,
        })
    }

    fn next_batch(&self, dataset: &Dataset) -> Batch {
        let settings = &self.snapshot.settings;
        let per_epoch = settings.updates_per_epoch();
        let epoch = self.snapshot.completed_updates / per_epoch;
        let offset = self.snapshot.completed_updates % per_epoch;
        let seed = settings.seed + epoch as u64 + 1;
        dataset.epoch(settings.batch_size, seed).remove(offset)
    }

    // ANCHOR: update
    fn update(mut self, dataset: &Dataset) -> Demo<Self> {
        let batch = self.next_batch(dataset);
        let next = self.snapshot.completed_updates + 1;
        let settings = &self.snapshot.settings;
        let rate = settings.schedule.rate(next, settings.total_updates());
        let device = Device::flex().autodiff();
        device.seed(settings.seed + next as u64);
        let (inputs, targets) = batch.to_tensors(&device);
        let (model, loss) = train_step(self.model, inputs, targets, rate, &mut self.optimizer);
        let loss = loss.into_scalar::<f32>();
        if !loss.is_finite() {
            return Err("Nicht endlicher Trainings-Loss; Checkpoint wird nicht gespeichert".into());
        }
        self.model = model;
        self.snapshot.completed_updates = next;
        println!("Update {next}: Lernrate {rate:.8}, Loss {loss:.6}");
        if next.is_multiple_of(self.snapshot.settings.updates_per_epoch()) {
            self.validate_epoch()?;
        }
        Ok(self)
    }
    // ANCHOR_END: update

    // ANCHOR: validate
    /// Bewertet mit festen Gewichten und ersetzt das Bestmodell nur bei echter Verbesserung.
    fn validate_epoch(&mut self) -> Demo {
        let settings = &self.snapshot.settings;
        let fixed = self.model.valid();
        let device = Device::flex();
        let train = evaluate(
            &fixed,
            &settings.train_dataset()?,
            settings.batch_size,
            &device,
        );
        let validation = evaluate(
            &fixed,
            &settings.validation_dataset()?,
            settings.batch_size,
            &device,
        );
        let update = self.snapshot.completed_updates;
        let improved = self
            .snapshot
            .best
            .as_ref()
            .is_none_or(|best| validation < best.validation_loss);
        println!(
            "Epoche {}: Training {train:.4}, Validierung {validation:.4}{}",
            update / settings.updates_per_epoch(),
            if improved {
                "  <- neues Bestmodell"
            } else {
                ""
            }
        );
        if improved {
            self.snapshot.best = Some(BestInfo {
                validation_loss: validation,
                update,
                model: settings.model.clone(),
                model_sha256: String::new(),
            });
            self.best_model = Some(fixed);
        }
        Ok(())
    }
    // ANCHOR_END: validate

    fn advance(mut self, until: usize) -> Demo<Self> {
        let total = self.snapshot.settings.total_updates();
        if until <= self.snapshot.completed_updates || until > total {
            return Err(format!(
                "Ziel muss nach dem aktuellen Update und höchstens bei {total} liegen"
            )
            .into());
        }
        let dataset = self.snapshot.settings.train_dataset()?;
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

fn sync_file(path: &Path) -> Demo {
    Ok(fs::File::open(path)?.sync_all()?)
}

// ANCHOR: save
fn save(mut training: Training, path: &Path) -> Demo {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    fs::create_dir(path)?;
    save_best(&mut training, &path.join(BEST_DIR))?;
    save_last(&mut training, &path.join(LAST_DIR))?;
    println!("Lauf gespeichert: {}", path.display());
    Ok(())
}

fn save_best(training: &mut Training, dir: &Path) -> Demo {
    let (Some(info), Some(model)) = (training.snapshot.best.as_mut(), &training.best_model) else {
        return Ok(());
    };
    fs::create_dir(dir)?;
    save_model(model, dir.join(MODEL_FILE))?;
    sync_file(&dir.join(MODEL_FILE))?;
    info.model_sha256 = digest(&dir.join(MODEL_FILE))?;
    write_new(&dir.join(BEST_FILE), &serde_json::to_vec_pretty(info)?)
}

fn save_last(training: &mut Training, dir: &Path) -> Demo {
    fs::create_dir(dir)?;
    save_model(&training.model.valid(), dir.join(MODEL_FILE))?;
    training.optimizer.save(dir.join(OPTIMIZER_FILE))?;
    sync_file(&dir.join(MODEL_FILE))?;
    sync_file(&dir.join(OPTIMIZER_FILE))?;
    training.snapshot.model_sha256 = digest(&dir.join(MODEL_FILE))?;
    training.snapshot.optimizer_sha256 = digest(&dir.join(OPTIMIZER_FILE))?;
    write_new(
        &dir.join(STATE_FILE),
        &serde_json::to_vec_pretty(&training.snapshot)?,
    )
}
// ANCHOR_END: save

fn read_snapshot(dir: &Path) -> Demo<Snapshot> {
    let snapshot: Snapshot = serde_json::from_slice(&fs::read(dir.join(STATE_FILE))?)?;
    if snapshot.version != FORMAT_VERSION || snapshot.compatibility != COMPATIBILITY {
        return Err("Inkompatibler Trainings-Checkpoint".into());
    }
    snapshot.settings.validate()?;
    if snapshot.completed_updates == 0
        || snapshot.completed_updates > snapshot.settings.total_updates()
    {
        return Err("Ungültiger Update-Zähler im Checkpoint".into());
    }
    if digest(&dir.join(MODEL_FILE))? != snapshot.model_sha256
        || digest(&dir.join(OPTIMIZER_FILE))? != snapshot.optimizer_sha256
    {
        return Err("Checkpoint unvollständig oder verändert: SHA-256 stimmt nicht".into());
    }
    Ok(snapshot)
}

// ANCHOR: load
fn load(path: &Path) -> Demo<Training> {
    let snapshot = read_snapshot(&path.join(LAST_DIR))?;
    let device = Device::flex().autodiff();
    let last = path.join(LAST_DIR);
    let model = load_model(
        &snapshot.settings.model.config(),
        last.join(MODEL_FILE),
        &device,
    )?;
    let optimizer = load_optimizer(&snapshot.settings, &last, &model)?;
    let best_model = load_best(&snapshot, &path.join(BEST_DIR))?;
    Ok(Training {
        model,
        optimizer,
        snapshot,
        best_model,
    })
}

fn load_best(snapshot: &Snapshot, dir: &Path) -> Demo<Option<GptModel>> {
    let Some(info) = &snapshot.best else {
        return if dir.exists() {
            Err("Bestmodell vorhanden, aber nicht im Trainingszustand eingetragen".into())
        } else {
            Ok(None)
        };
    };
    let stored: BestInfo = serde_json::from_slice(&fs::read(dir.join(BEST_FILE))?)?;
    let consistent = stored == *info
        && digest(&dir.join(MODEL_FILE))? == info.model_sha256
        && info.update <= snapshot.completed_updates;
    if !consistent {
        return Err("Bestmodell passt nicht zum Trainingszustand".into());
    }
    let device = Device::flex();
    Ok(Some(load_model(
        &info.model.config(),
        dir.join(MODEL_FILE),
        &device,
    )?))
}
// ANCHOR_END: load

fn load_optimizer(settings: &Settings, dir: &Path, model: &GptModel) -> Demo<ModuleOptimizer> {
    let optimizer = settings.optimizer.init().load(dir.join(OPTIMIZER_FILE))?;
    let expected_tensors = list_param_ids(model).len() * ADAM_MOMENT_TENSORS;
    if optimizer.to_record().len() != expected_tensors {
        return Err("AdamW-Zustand ist nicht vollständig; kein stiller Neustart".into());
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
            dropout: 0.2,
            ..model_settings()
        };
        let train = vec![0, 1, 4, 2, 1, 3, 0, 4, 2, 3, 4, 1, 2, 0, 3, 1, 4];
        let validation = vec![1, 2, 3, 4, 0, 2, 1];
        Settings {
            epochs: 3,
            ..Settings::new(model, train, validation)
        }
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

    fn best_weights(training: &Training) -> Vec<TensorData> {
        weights(training.best_model.as_ref().expect("Bestmodell fehlt"))
    }

    fn saved(training: Training) -> Demo<(Training, tempfile::TempDir)> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("lauf");
        save(training, &path)?;
        Ok((load(&path)?, directory))
    }

    #[test]
    fn resume_with_dropout_matches_uninterrupted_run_including_best() -> Demo {
        let _lock = TRAINING_LOCK.lock().unwrap();
        let initial = Training::new(fixture())?;
        let baseline = initial.model.clone();
        let expected = initial.advance(12)?;
        for interruption in [3, 4, 5, 8] {
            let mut partial = Training::new(fixture())?;
            partial.model = baseline.clone();
            let (restored, _directory) = saved(partial.advance(interruption)?)?;
            let actual = restored.advance(12)?;
            assert_eq!(weights(&expected.model), weights(&actual.model));
            let best = |t: &Training| {
                t.snapshot
                    .best
                    .as_ref()
                    .map(|b| (b.validation_loss, b.update))
            };
            assert_eq!(best(&expected), best(&actual));
            assert_eq!(best_weights(&expected), best_weights(&actual));
        }
        Ok(())
    }

    #[test]
    fn resume_keeps_previous_best_unless_validation_improves() -> Demo {
        let _lock = TRAINING_LOCK.lock().unwrap();
        let mut partial = Training::new(fixture())?.advance(4)?;
        let first_best = partial.snapshot.best.clone().ok_or("Kein Bestwert")?;
        assert_eq!(first_best.update, 4);
        partial.snapshot.best.as_mut().unwrap().validation_loss = -1.0;
        let (restored, _directory) = saved(partial)?;
        let before = best_weights(&restored);
        let resumed = restored.advance(12)?;
        let best = resumed.snapshot.best.as_ref().ok_or("Bestwert verloren")?;
        assert_eq!(best.validation_loss, -1.0);
        assert_eq!(best.update, 4);
        assert_eq!(before, best_weights(&resumed));
        Ok(())
    }

    #[test]
    fn best_and_last_are_saved_separately() -> Demo {
        let _lock = TRAINING_LOCK.lock().unwrap();
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("lauf");
        let training = Training::new(fixture())?.advance(12)?;
        let best_update = training
            .snapshot
            .best
            .as_ref()
            .ok_or("Kein Bestwert")?
            .update;
        save(training, &path)?;
        assert_eq!(read_snapshot(&path.join(LAST_DIR))?.completed_updates, 12);
        let best: BestInfo =
            serde_json::from_slice(&fs::read(path.join(BEST_DIR).join(BEST_FILE))?)?;
        assert_eq!(best.update, best_update);
        let config = best.model.config();
        let loaded = load_model(
            &config,
            path.join(BEST_DIR).join(MODEL_FILE),
            &Device::flex(),
        )?;
        let settings = fixture();
        let loss = evaluate(
            &loaded,
            &settings.validation_dataset()?,
            settings.batch_size,
            &Device::flex(),
        );
        assert!((loss - best.validation_loss).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn weight_decay_is_applied_and_survives_resume() -> Demo {
        let _lock = TRAINING_LOCK.lock().unwrap();
        let without_decay = || {
            let mut settings = fixture();
            settings.optimizer = AdamWConfig::new().with_weight_decay(0.0);
            settings
        };
        let decayed = Training::new(fixture())?.advance(4)?;
        let plain = Training::new(without_decay())?.advance(4)?;
        assert_ne!(weights(&decayed.model), weights(&plain.model));
        let expected = serde_json::to_string(&decayed.snapshot.settings.optimizer)?;
        let (restored, _directory) = saved(decayed)?;
        let actual = serde_json::to_string(&restored.snapshot.settings.optimizer)?;
        assert_eq!(expected, actual);
        Ok(())
    }

    #[test]
    fn dropout_only_applies_in_training_mode() -> Demo {
        let _lock = TRAINING_LOCK.lock().unwrap();
        let device = Device::flex().autodiff();
        let model = fixture().model.config().init(&device);
        let input = Tensor::<2, Int>::zeros([2, 2], &Device::flex());
        let fixed = model.valid();
        assert_eq!(
            fixed.forward(input.clone()).into_data(),
            fixed.forward(input).into_data()
        );
        Ok(())
    }

    #[test]
    fn inconsistent_checkpoints_are_errors() -> Demo {
        let _lock = TRAINING_LOCK.lock().unwrap();
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("lauf");
        save(Training::new(fixture())?.advance(5)?, &path)?;
        let best_model = fs::read(path.join(BEST_DIR).join(MODEL_FILE))?;
        fs::write(path.join(BEST_DIR).join(MODEL_FILE), b"corrupt")?;
        assert!(load(&path).is_err());
        fs::write(path.join(BEST_DIR).join(MODEL_FILE), best_model)?;
        fs::remove_file(path.join(BEST_DIR).join(BEST_FILE))?;
        assert!(load(&path).is_err());
        fs::remove_dir_all(path.join(BEST_DIR))?;
        assert!(load(&path).is_err());
        fs::write(path.join(LAST_DIR).join(OPTIMIZER_FILE), b"corrupt")?;
        let error = load(&path).err().ok_or("Beschädigter Zustand geladen")?;
        assert!(error.to_string().contains("SHA-256"));
        Ok(())
    }

    #[test]
    fn invalid_targets_and_overwrites_are_errors() -> Demo {
        let _lock = TRAINING_LOCK.lock().unwrap();
        assert!(Training::new(fixture())?.advance(0).is_err());
        assert!(Training::new(fixture())?.advance(13).is_err());
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("lauf");
        save(Training::new(fixture())?.advance(4)?, &path)?;
        let original = fs::read(path.join(LAST_DIR).join(STATE_FILE))?;
        assert!(save(Training::new(fixture())?.advance(5)?, &path).is_err());
        assert_eq!(fs::read(path.join(LAST_DIR).join(STATE_FILE))?, original);
        assert!(load(&path)?.advance(4).is_err());
        Ok(())
    }
}
