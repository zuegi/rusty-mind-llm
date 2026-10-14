//! Kapitel 4: Gradienten mit Burn-Autodiff berechnen.
//! Ausführen: cargo run -p model --example ch04

use burn::tensor::{Device, Int, Tensor};
use model::embedding::GptEmbeddingConfig;


const VOCAB_SIZE: usize = 5;
const MAX_CONTEXT_LENGTH: usize = 3;
const EMBEDDING_DIM: usize = 4;

type Demo = Result<(), Box<dyn std::error::Error>>;

fn main() -> Demo {
    square()?;
    chain_rule()?;
    weighted_sum()?;
    embedding_gradients()?;
    Ok(())
}

// ANCHOR: square
fn square() -> Demo {
    let device = Device::flex().autodiff();
    let x = Tensor::<1>::from_data([3.0], &device).require_grad();
    let y = x.clone() * x.clone();
    let grads = y.backward();
    let x_grad = x.grad(&grads).ok_or("x hat keinen Gradienten")?;
    println!("Beispiel 1: y = x², x = 3");
    println!("  y = {:?}", y.into_data().try_to_vec::<f32>()?);
    println!("  dy/dx = {:?}", x_grad.into_data().try_to_vec::<f32>()?);
    Ok(())
}
// ANCHOR_END: square

// ANCHOR: chain_rule
fn chain_rule() -> Demo {
    let device = Device::flex().autodiff();
    let x = Tensor::<1>::from_data([3.0], &device).require_grad();
    let u = x.clone() * 2.0 + 1.0;
    let y = u.clone() * u;
    let grads = y.backward();
    let x_grad = x.grad(&grads).ok_or("x hat keinen Gradienten")?;
    println!("Beispiel 2: y = (2x + 1)², x = 3");
    println!("  y = {:?}", y.into_data().try_to_vec::<f32>()?);
    println!("  dy/dx = {:?}", x_grad.into_data().try_to_vec::<f32>()?);
    Ok(())
}
// ANCHOR_END: chain_rule

// ANCHOR: weighted_sum
fn weighted_sum() -> Demo {
    let device = Device::flex().autodiff();
    let w = Tensor::<1>::from_data([0.5, -1.0, 2.0], &device).require_grad();
    let x = Tensor::<1>::from_data([1.0, 2.0, 3.0], &device);
    let loss = (w.clone() * x).sum();
    let grads = loss.backward();
    let w_grad = w.grad(&grads).ok_or("w hat keinen Gradienten")?;
    println!("Beispiel 3: loss = sum(w * x), x = [1, 2, 3]");
    println!("  loss = {:?}", loss.into_data().try_to_vec::<f32>()?);
    println!("  dloss/dw = {:?}", w_grad.into_data().try_to_vec::<f32>()?);
    Ok(())
}
// ANCHOR_END: weighted_sum

// ANCHOR: embedding
fn embedding_gradients() -> Demo {
    let device = Device::flex().autodiff();
    let config = GptEmbeddingConfig::new(VOCAB_SIZE, MAX_CONTEXT_LENGTH, EMBEDDING_DIM);
    let embedding = config.init(&device);
    let token_ids = Tensor::<2, Int>::from_data([[1, 1, 3]], &device);
    let loss = embedding.forward(token_ids).sum();
    let grads = loss.backward();
    let token_grad = embedding.token.weight.grad(&grads);
    let position_grad = embedding.position.weight.grad(&grads);
    println!("Beispiel 4: loss = sum(embedding([[1, 1, 3]]))");
    print_rows(
        "Token-Gradient",
        token_grad.ok_or("Token-Tabelle ohne Gradient")?,
    )?;
    print_rows(
        "Positions-Gradient",
        position_grad.ok_or("Positions-Tabelle ohne Gradient")?,
    )?;
    Ok(())
}
// ANCHOR_END: embedding

fn print_rows(label: &str, tensor: Tensor<2>) -> Demo {
    let [rows, columns] = tensor.dims();
    let values = tensor.into_data().try_to_vec::<f32>()?;
    println!("  {label}, Form [{rows}, {columns}]:");
    for (row, vector) in values.chunks(columns).enumerate() {
        let numbers: Vec<_> = vector.iter().map(|value| format!("{value:.0}")).collect();
        println!("    Zeile {row}: [{}]", numbers.join(", "));
    }
    Ok(())
}
