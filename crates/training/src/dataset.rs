use std::fmt;

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

// ANCHOR: types
/// Ein Trainingsbeispiel: `target` ist `input`, um ein Token nach links verschoben.
#[derive(Debug, Clone, PartialEq)]
pub struct Window {
    pub input: Vec<u32>,
    pub target: Vec<u32>,
}

/// Mehrere Fenster, Form je `[batch_size][context_length]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Batch {
    pub inputs: Vec<Vec<u32>>,
    pub targets: Vec<Vec<u32>>,
}

#[derive(Debug, PartialEq)]
pub enum DatasetError {
    TooShort { needed: usize, got: usize },
}

impl fmt::Display for DatasetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooShort { needed, got } => {
                write!(f, "Zu wenige Token: mindestens {needed} nötig, {got} vorhanden")
            }
        }
    }
}

impl std::error::Error for DatasetError {}
// ANCHOR_END: types

// ANCHOR: split
/// Teilt den Token-Strom in (Training, Validierung). Muss VOR dem Fenstern geschehen.
pub fn split_tokens(tokens: &[u32], validation_fraction: f64) -> (&[u32], &[u32]) {
    assert!(
        (0.0..1.0).contains(&validation_fraction),
        "validation_fraction muss in [0, 1) liegen"
    );
    let train_len = (tokens.len() as f64 * (1.0 - validation_fraction)) as usize;
    tokens.split_at(train_len)
}
// ANCHOR_END: split

pub struct Dataset {
    windows: Vec<Window>,
    context_length: usize,
}

impl Dataset {
    // ANCHOR: new
    /// Schneidet den Strom lückenlos in Fenster der Länge `context_length`.
    pub fn new(tokens: &[u32], context_length: usize) -> Result<Self, DatasetError> {
        assert!(context_length > 0, "context_length muss größer als 0 sein");
        let needed = context_length + 1;
        if tokens.len() < needed {
            return Err(DatasetError::TooShort { needed, got: tokens.len() });
        }
        let inputs = tokens[..tokens.len() - 1].chunks_exact(context_length);
        let targets = tokens[1..].chunks_exact(context_length);
        let windows = inputs
            .zip(targets)
            .map(|(input, target)| Window { input: input.to_vec(), target: target.to_vec() })
            .collect();
        Ok(Self { windows, context_length })
    }
    // ANCHOR_END: new

    pub fn len(&self) -> usize {
        self.windows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.windows.is_empty()
    }

    pub fn context_length(&self) -> usize {
        self.context_length
    }

    pub fn window(&self, index: usize) -> &Window {
        &self.windows[index]
    }

    // ANCHOR: epoch
    /// Eine Epoche: Fenster mit festem Seed mischen und zu Batches bündeln.
    /// Ein unvollständiger letzter Batch wird verworfen.
    pub fn epoch(&self, batch_size: usize, seed: u64) -> Vec<Batch> {
        assert!(batch_size > 0, "batch_size muss größer als 0 sein");
        let mut order: Vec<usize> = (0..self.windows.len()).collect();
        order.shuffle(&mut StdRng::seed_from_u64(seed));
        order
            .chunks_exact(batch_size)
            .map(|indices| self.batch_of(indices))
            .collect()
    }

    fn batch_of(&self, indices: &[usize]) -> Batch {
        Batch {
            inputs: indices.iter().map(|&i| self.windows[i].input.clone()).collect(),
            targets: indices.iter().map(|&i| self.windows[i].target.clone()).collect(),
        }
    }
    // ANCHOR_END: epoch
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stream(n: u32) -> Vec<u32> {
        (0..n).collect()
    }

    // ANCHOR: test_shift
    #[test]
    fn ziel_ist_eingabe_um_eins_verschoben() {
        let dataset = Dataset::new(&stream(9), 4).unwrap();
        assert_eq!(dataset.len(), 2);
        assert_eq!(dataset.window(0).input, [0, 1, 2, 3]);
        assert_eq!(dataset.window(0).target, [1, 2, 3, 4]);
        assert_eq!(dataset.window(1).input, [4, 5, 6, 7]);
        assert_eq!(dataset.window(1).target, [5, 6, 7, 8]);
    }
    // ANCHOR_END: test_shift

    #[test]
    fn rest_am_ende_faellt_weg() {
        assert_eq!(Dataset::new(&stream(12), 4).unwrap().len(), 2);
    }

    #[test]
    fn zu_kurzer_strom_ist_ein_fehler() {
        let result = Dataset::new(&stream(4), 4);
        assert_eq!(result.err(), Some(DatasetError::TooShort { needed: 5, got: 4 }));
    }

    // ANCHOR: test_split
    #[test]
    fn split_teilt_ohne_ueberlappung() {
        let tokens = stream(100);
        let (train, validation) = split_tokens(&tokens, 0.1);
        assert_eq!((train.len(), validation.len()), (90, 10));
        assert!(train.iter().all(|t| !validation.contains(t)));
    }
    // ANCHOR_END: test_split

    #[test]
    fn epoche_hat_vollstaendige_batches_der_richtigen_form() {
        let dataset = Dataset::new(&stream(41), 4).unwrap();
        let batches = dataset.epoch(3, 1);
        assert_eq!(batches.len(), 3);
        for batch in &batches {
            assert_eq!(batch.inputs.len(), 3);
            assert!(batch.inputs.iter().all(|row| row.len() == 4));
            assert_eq!(batch.targets.len(), 3);
        }
    }

    #[test]
    fn epoche_nutzt_jedes_fenster_hoechstens_einmal() {
        let dataset = Dataset::new(&stream(37), 4).unwrap();
        assert_eq!(dataset.len(), 9);
        let mut firsts: Vec<u32> = dataset
            .epoch(2, 7)
            .iter()
            .flat_map(|b| b.inputs.iter().map(|row| row[0]))
            .collect();
        firsts.sort();
        firsts.dedup();
        assert_eq!(firsts.len(), 8);
    }

    #[test]
    fn gleicher_seed_gleiche_reihenfolge() {
        let dataset = Dataset::new(&stream(101), 4).unwrap();
        assert_eq!(dataset.epoch(4, 42), dataset.epoch(4, 42));
        assert_ne!(dataset.epoch(4, 42), dataset.epoch(4, 43));
    }
}
