//! Kapitel 6, Schritt 3: Gradient durch viele Schichten, mit und ohne Residual.
//! Ausführen: cargo run -p model --example ch06_residual

use burn::tensor::{Device, Tensor};

type Demo = Result<(), Box<dyn std::error::Error>>;

const LAYER_SCALE: f32 = 0.1;
const LAYER_COUNTS: [usize; 4] = [1, 2, 5, 10];

fn main() -> Demo {
    println!("Schichten  Gradient ohne Residual  Gradient mit Residual");
    for layers in LAYER_COUNTS {
        let plain = gradient_at_input(layers, false)?;
        let residual = gradient_at_input(layers, true)?;
        println!("{layers:9}  {plain:21.3e}  {residual:21.3}");
    }
    Ok(())
}

// ANCHOR: stack
fn gradient_at_input(layers: usize, residual: bool) -> Result<f32, Box<dyn std::error::Error>> {
    let x = Tensor::<1>::from_data([1.0], &Device::ndarray().autodiff()).require_grad();
    let mut hidden = x.clone();
    for _ in 0..layers {
        let update = hidden.clone() * LAYER_SCALE;
        hidden = if residual { hidden + update } else { update };
    }
    let grads = hidden.backward();
    let gradient = x.grad(&grads).ok_or("kein Gradient")?;
    Ok(gradient.into_data().try_to_vec::<f32>()?[0])
}
// ANCHOR_END: stack
