use serde::{Deserialize, Serialize};

/// Linearer Warmup mit anschliessender linearer Absenkung, gezählt pro Update.
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WarmupDecay {
    pub peak_rate: f64,
    pub warmup_updates: usize,
    pub final_fraction: f64,
}

impl WarmupDecay {
    pub fn validate(&self) -> Result<(), &'static str> {
        if !self.peak_rate.is_finite() || self.peak_rate <= 0.0 || self.warmup_updates == 0 {
            return Err("Lernrate und Warmup müssen positiv sein");
        }
        let valid_fraction = self.final_fraction.is_finite()
            && self.final_fraction > 0.0
            && self.final_fraction <= 1.0;
        if !valid_fraction {
            return Err("Endanteil der Lernrate muss in (0, 1] liegen");
        }
        Ok(())
    }

    pub fn rate(&self, update: usize, total: usize) -> f64 {
        assert!(self.validate().is_ok(), "Ungültiger Lernratenplan");
        assert!(
            update > 0 && update <= total,
            "Update muss in 1..=total liegen"
        );
        let warmup = self.warmup_updates.min(total);
        if update <= warmup {
            return self.peak_rate * update as f64 / warmup as f64;
        }
        let progress = (update - warmup) as f64 / (total - warmup) as f64;
        self.peak_rate * (1.0 - progress * (1.0 - self.final_fraction))
    }
}
