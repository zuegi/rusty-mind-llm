//! Kapitel 8: Cross-Entropy mit festen, nachrechenbaren Logits.
//! Ausführen: cargo run -p training --example ch08_loss

use burn::tensor::{Device, Int, Tensor, activation::softmax};
use training::loss::cross_entropy;

type Demo = Result<(), Box<dyn std::error::Error>>;

const WEIGHTS: [[[f32; 3]; 2]; 1] = [[[2.0, 3.0, 5.0], [5.0, 3.0, 2.0]]];
const LIKELY_TARGETS: [[i32; 2]; 1] = [[2, 0]];
const UNLIKELY_TARGETS: [[i32; 2]; 1] = [[0, 2]];

fn main() -> Demo {
    let device = Device::flex();
    let logits = Tensor::<3>::from_data(WEIGHTS, &device).log();
    print_predictions(logits.clone())?;
    let likely = Tensor::<2, Int>::from_data(LIKELY_TARGETS, &device);
    let unlikely = Tensor::<2, Int>::from_data(UNLIKELY_TARGETS, &device);
    let likely_loss = cross_entropy(logits.clone(), likely).into_scalar::<f32>();
    let unlikely_loss = cross_entropy(logits, unlikely).into_scalar::<f32>();
    println!("Loss für Ziele [2, 0]: {likely_loss:.6}");
    println!("Loss für Ziele [0, 2]: {unlikely_loss:.6}");
    Ok(())
}

fn print_predictions(logits: Tensor<3>) -> Demo {
    let probabilities = softmax(logits.clone(), 2).into_data().try_to_vec::<f32>()?;
    println!("Logits: {:?}", logits.into_data().try_to_vec::<f32>()?);
    println!("Wahrscheinlichkeiten: {probabilities:?}");
    Ok(())
}
