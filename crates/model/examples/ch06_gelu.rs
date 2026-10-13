//! Kapitel 6, Schritt 2: GELU im Vergleich zu ReLU und zur Identität.
//! Ausführen: cargo run -p model --example ch06_gelu

use burn::backend::NdArray;
use burn::tensor::{
    Tensor,
    activation::{gelu, relu},
};

type Cpu = NdArray<f32>;
type Demo = Result<(), Box<dyn std::error::Error>>;

const INPUT: [f32; 9] = [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0];

fn main() -> Demo {
    let input = Tensor::<Cpu, 1>::from_data(INPUT, &Default::default());
    let relu_values = relu(input.clone()).into_data().to_vec::<f32>()?;
    let gelu_values = gelu(input).into_data().to_vec::<f32>()?;
    println!("     x     ReLU     GELU");
    for ((x, r), g) in INPUT.iter().zip(relu_values).zip(gelu_values) {
        println!("{x:6.1}  {r:7.3}  {g:7.3}");
    }
    Ok(())
}
