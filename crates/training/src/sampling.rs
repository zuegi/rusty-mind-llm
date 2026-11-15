use std::fmt;

use rand::{
    Rng,
    distr::{Distribution, weighted::WeightedIndex},
};

// ANCHOR: strategy
#[derive(Debug, Clone, Copy)]
pub enum SamplingFilter {
    None,
    TopK(usize),
    TopP(f64),
}

#[derive(Debug, Clone, Copy)]
pub enum SamplingStrategy {
    Greedy,
    Multinomial {
        temperature: f64,
        filter: SamplingFilter,
    },
}
// ANCHOR_END: strategy

#[derive(Debug)]
pub struct SamplingError(String);

impl SamplingError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for SamplingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Sampling-Fehler: {}", self.0)
    }
}

impl std::error::Error for SamplingError {}

impl SamplingStrategy {
    pub fn validate(self, vocab_size: usize) -> Result<(), SamplingError> {
        if vocab_size == 0 {
            return Err(SamplingError::new("Vokabular darf nicht leer sein"));
        }
        if let Self::Multinomial {
            temperature,
            filter,
        } = self
        {
            if !temperature.is_finite() || temperature <= 0.0 {
                return Err(SamplingError::new(
                    "Temperatur muss endlich und größer als 0 sein",
                ));
            }
            validate_filter(filter, vocab_size)?;
        }
        Ok(())
    }

    pub fn probabilities(self, logits: &[f32]) -> Result<Vec<f64>, SamplingError> {
        self.validate(logits.len())?;
        validate_logits(logits)?;
        match self {
            Self::Greedy => Ok(greedy_probabilities(logits)),
            Self::Multinomial {
                temperature,
                filter,
            } => {
                let mut probabilities = softmax(logits, temperature);
                apply_filter(&mut probabilities, filter);
                Ok(probabilities)
            }
        }
    }

    pub fn select(self, logits: &[f32], rng: &mut impl Rng) -> Result<i32, SamplingError> {
        let probabilities = self.probabilities(logits)?;
        let index = match self {
            Self::Greedy => probabilities
                .iter()
                .position(|&p| p == 1.0)
                .ok_or_else(|| SamplingError::new("Kein Greedy-Kandidat"))?,
            Self::Multinomial { .. } => WeightedIndex::new(&probabilities)
                .map_err(|error| SamplingError::new(error.to_string()))?
                .sample(rng),
        };
        i32::try_from(index).map_err(|error| SamplingError::new(error.to_string()))
    }
}

fn validate_filter(filter: SamplingFilter, vocab_size: usize) -> Result<(), SamplingError> {
    match filter {
        SamplingFilter::TopK(k) if k == 0 || k > vocab_size => Err(SamplingError::new(format!(
            "Top-k muss zwischen 1 und {vocab_size} liegen"
        ))),
        SamplingFilter::TopP(p) if !p.is_finite() || p <= 0.0 || p > 1.0 => Err(
            SamplingError::new("Top-p muss endlich sein und in (0, 1] liegen"),
        ),
        _ => Ok(()),
    }
}

fn validate_logits(logits: &[f32]) -> Result<(), SamplingError> {
    if logits.iter().any(|logit| !logit.is_finite()) {
        return Err(SamplingError::new("Logits müssen endlich sein"));
    }
    if logits.len() > i32::MAX as usize {
        return Err(SamplingError::new("Vokabular zu groß für i32-Token-IDs"));
    }
    Ok(())
}

fn ranked_ids(values: &[f64]) -> Vec<usize> {
    let mut ids: Vec<_> = (0..values.len()).collect();
    ids.sort_by(|&a, &b| values[b].total_cmp(&values[a]).then(a.cmp(&b)));
    ids
}

fn greedy_probabilities(logits: &[f32]) -> Vec<f64> {
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let id = logits
        .iter()
        .position(|&logit| logit == maximum)
        .expect("Logits validiert");
    let mut probabilities = vec![0.0; logits.len()];
    probabilities[id] = 1.0;
    probabilities
}

// ANCHOR: softmax
fn softmax(logits: &[f32], temperature: f64) -> Vec<f64> {
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut weights: Vec<_> = logits
        .iter()
        .map(|&logit| ((f64::from(logit) - f64::from(maximum)) / temperature).exp())
        .collect();
    normalize(&mut weights);
    weights
}
// ANCHOR_END: softmax

fn normalize(weights: &mut [f64]) {
    let sum: f64 = weights.iter().sum();
    for weight in weights {
        *weight /= sum;
    }
}

// ANCHOR: filter
fn apply_filter(probabilities: &mut [f64], filter: SamplingFilter) {
    let keep = match filter {
        SamplingFilter::None | SamplingFilter::TopP(1.0) => return,
        SamplingFilter::TopK(k) => k,
        SamplingFilter::TopP(p) => nucleus_size(probabilities, p),
    };
    let ids = ranked_ids(probabilities);
    for &id in &ids[keep..] {
        probabilities[id] = 0.0;
    }
    normalize(probabilities);
}

fn nucleus_size(probabilities: &[f64], threshold: f64) -> usize {
    let ids = ranked_ids(probabilities);
    let mut cumulative = 0.0;
    for (position, &id) in ids.iter().enumerate() {
        cumulative += probabilities[id];
        if cumulative >= threshold {
            return position + 1;
        }
    }
    ids.len()
}
// ANCHOR_END: filter

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, rngs::StdRng};

    fn strategy(temperature: f64, filter: SamplingFilter) -> SamplingStrategy {
        SamplingStrategy::Multinomial {
            temperature,
            filter,
        }
    }

    #[test]
    fn temperature_changes_distribution_without_changing_ranking() {
        let logits = [0.0, 1.0, 2.0];
        let cold = strategy(0.5, SamplingFilter::None)
            .probabilities(&logits)
            .unwrap();
        let warm = strategy(2.0, SamplingFilter::None)
            .probabilities(&logits)
            .unwrap();
        assert!(cold[2] > warm[2]);
        assert!(cold[0] < warm[0]);
        assert!((cold.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        let shifted = strategy(0.5, SamplingFilter::None)
            .probabilities(&[1000.0, 1001.0, 1002.0])
            .unwrap();
        assert_eq!(cold, shifted);
        assert!(warm[2] > warm[1] && warm[1] > warm[0]);
    }

    #[test]
    fn top_k_keeps_exactly_k_tokens_and_breaks_ties_by_id() {
        let probabilities = strategy(1.0, SamplingFilter::TopK(2))
            .probabilities(&[0.0, 3.0, 3.0, 3.0])
            .unwrap();
        assert_eq!(probabilities, [0.0, 0.5, 0.5, 0.0]);
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..100 {
            assert_eq!(
                strategy(2.0, SamplingFilter::TopK(1))
                    .select(&[1.0, 3.0, 2.0], &mut rng)
                    .unwrap(),
                1
            );
        }
    }

    #[test]
    fn top_p_keeps_token_crossing_threshold_and_renormalizes() {
        let logits = [0.5_f32.ln(), 0.3_f32.ln(), 0.2_f32.ln()];
        let probabilities = strategy(1.0, SamplingFilter::TopP(0.7))
            .probabilities(&logits)
            .unwrap();
        assert!((probabilities[0] - 0.625).abs() < 1e-7);
        assert!((probabilities[1] - 0.375).abs() < 1e-7);
        assert_eq!(probabilities[2], 0.0);
        assert_eq!(nucleus_size(&[0.5, 0.25, 0.25], 0.5), 1);
        assert_eq!(nucleus_size(&[0.5, 0.25, 0.25], 0.75), 2);
        let all = strategy(1.0, SamplingFilter::TopP(1.0))
            .probabilities(&logits)
            .unwrap();
        assert!(all.iter().all(|&p| p > 0.0));
    }

    #[test]
    fn seeded_sampling_is_reproducible_and_follows_weights() {
        let logits = [0.75_f32.ln(), 0.25_f32.ln()];
        let draw = |seed| {
            let mut rng = StdRng::seed_from_u64(seed);
            (0..10_000)
                .map(|_| {
                    strategy(1.0, SamplingFilter::None)
                        .select(&logits, &mut rng)
                        .unwrap()
                })
                .collect::<Vec<_>>()
        };
        let first = draw(42);
        assert_eq!(first, draw(42));
        assert_ne!(first, draw(43));
        let frequency = first.iter().filter(|&&id| id == 0).count() as f64 / first.len() as f64;
        assert!((frequency - 0.75).abs() < 0.02);
    }

    #[test]
    fn rejects_invalid_configuration_and_logits() {
        for temperature in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(
                strategy(temperature, SamplingFilter::None)
                    .validate(3)
                    .is_err()
            );
        }
        for p in [0.0, -0.1, 1.1, f64::NAN, f64::INFINITY] {
            assert!(strategy(1.0, SamplingFilter::TopP(p)).validate(3).is_err());
        }
        for k in [0, 4] {
            assert!(strategy(1.0, SamplingFilter::TopK(k)).validate(3).is_err());
        }
        for logits in [
            vec![],
            vec![f32::NAN],
            vec![f32::INFINITY],
            vec![f32::NEG_INFINITY],
        ] {
            assert!(SamplingStrategy::Greedy.probabilities(&logits).is_err());
        }
    }

    #[test]
    fn extreme_finite_logits_and_temperatures_remain_normalized() {
        for temperature in [f64::MIN_POSITIVE, f64::MAX] {
            let probabilities = strategy(temperature, SamplingFilter::None)
                .probabilities(&[-f32::MAX, f32::MAX])
                .unwrap();
            assert!(probabilities.iter().all(|p| p.is_finite()));
            assert_eq!(probabilities.iter().sum::<f64>(), 1.0);
        }
    }
}
