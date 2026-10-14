//! Kapitel 6, Schritt 2: Feed-Forward-Netz Schritt für Schritt.
//! Ausführen: cargo run -p model --example ch06_feed_forward

use burn::module::Module;
use burn::tensor::{Device, Tensor, activation::gelu};
use model::feed_forward::FeedForwardConfig;

type Demo = Result<(), Box<dyn std::error::Error>>;

const EMBEDDING_DIM: usize = 2;
// Dasselbe Token (Vektor [1, -1]) steht an Position 0 und 2.
const INPUT: [[[f32; EMBEDDING_DIM]; 3]; 1] = [[[1.0, -1.0], [0.5, 2.0], [1.0, -1.0]]];

fn main() -> Demo {
    let device = Device::flex();
    let ff = FeedForwardConfig::new(EMBEDDING_DIM).init(&device);
    let input = Tensor::<3>::from_data(INPUT, &device);
    println!("Eingabe:        Form {:?}", input.dims());

    let widened = ff.expand.forward(input.clone());
    println!("nach expand:    Form {:?}", widened.dims());
    let activated = gelu(widened);
    println!("nach GELU:      Form {:?}", activated.dims());
    let output = ff.contract.forward(activated);
    println!("nach contract:  Form {:?}", output.dims());

    let values = ff.forward(input).into_data().try_to_vec::<f32>()?;
    let first = &values[..EMBEDDING_DIM];
    let third = &values[2 * EMBEDDING_DIM..];
    println!("Token 0 und Token 2 gleich: {}", first == third);
    println!("Parameter: {}", ff.num_params());
    Ok(())
}
