//! Kapitel 5: Ein Attention-Kopf mit festen Vektoren.
//! Ausführen: cargo run -p model --example ch05

use burn::tensor::{Device, Tensor, activation::softmax};


const INPUT: [[f32; 2]; 2] = [[1.0, 0.0], [0.0, 1.0]];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let input = Tensor::<2>::from_data(INPUT, &Device::flex());
    print_matrix("Eingabe X = Query = Key = Value", input.clone())?;
    let (scores, weights, context) = attention(input);
    print_matrix("Scores", scores)?;
    print_matrix("Attention-Gewichte", weights)?;
    print_matrix("Kontext-Vektoren", context)?;
    Ok(())
}

// ANCHOR: attention
fn attention(input: Tensor<2>) -> (Tensor<2>, Tensor<2>, Tensor<2>) {
    let query = input.clone();
    let key = input.clone();
    let value = input;
    let scores = query.matmul(key.transpose());
    let weights = softmax(scores.clone(), 1);
    let context = weights.clone().matmul(value);
    (scores, weights, context)
}
// ANCHOR_END: attention

fn print_matrix(label: &str, tensor: Tensor<2>) -> Result<(), Box<dyn std::error::Error>> {
    let [rows, columns] = tensor.dims();
    let values = tensor.into_data().try_to_vec::<f32>()?;
    println!("{label}, Form [{rows}, {columns}]:");
    for (row, vector) in values.chunks(columns).enumerate() {
        let numbers: Vec<_> = vector.iter().map(|value| format!("{value:.3}")).collect();
        println!("  Token {row}: [{}]", numbers.join(", "));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: f32 = 1e-6;
    const SELF_WEIGHT: f32 = 0.731_058_6;
    const OTHER_WEIGHT: f32 = 0.268_941_4;

    fn assert_values(tensor: Tensor<2>, expected: [f32; 4]) {
        let actual = tensor.into_data().try_to_vec::<f32>().expect("Tensor lesbar");
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!(
                (actual - expected).abs() < TOLERANCE,
                "{actual} != {expected}"
            );
        }
    }

    #[test]
    fn reproduces_hand_calculated_scores_weights_and_context() {
        let input = Tensor::<2>::from_data(INPUT, &Device::flex());
        let (scores, weights, context) = attention(input);
        assert_eq!(context.dims(), [2, 2]);
        assert_values(scores, [1.0, 0.0, 0.0, 1.0]);
        let expected = [SELF_WEIGHT, OTHER_WEIGHT, OTHER_WEIGHT, SELF_WEIGHT];
        assert_values(weights, expected);
        assert_values(context, expected);
    }

    #[test]
    fn normalizes_each_query_row_over_keys() {
        let input = Tensor::<2>::from_data(INPUT, &Device::flex());
        let (_, weights, _) = attention(input);
        let values = weights
            .into_data()
            .try_to_vec::<f32>()
            .expect("Gewichte lesbar");
        for row in values.chunks(2) {
            assert!(row.iter().all(|weight| *weight > 0.0));
            assert!((row.iter().sum::<f32>() - 1.0).abs() < TOLERANCE);
        }
    }

    #[test]
    fn equal_scores_mix_values_equally() {
        let input = Tensor::<2>::from_data([[0.0, 0.0], [0.0, 2.0]], &Device::flex());
        let (_, weights, context) = attention(input);
        let weights = weights
            .into_data()
            .try_to_vec::<f32>()
            .expect("Gewichte lesbar");
        let context = context.into_data().try_to_vec::<f32>().expect("Kontext lesbar");
        assert!((weights[0] - 0.5).abs() < TOLERANCE);
        assert!((weights[1] - 0.5).abs() < TOLERANCE);
        assert!((context[0] - 0.0).abs() < TOLERANCE);
        assert!((context[1] - 1.0).abs() < TOLERANCE);
    }
}
