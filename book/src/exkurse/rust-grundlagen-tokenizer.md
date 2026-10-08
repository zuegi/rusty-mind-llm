# Exkurs: Rust-Grundlagen für den Tokenizer

Dieser Exkurs stellt die Rust-Konzepte vor, die dir in Kapitel 1 begegnen. Mehr
brauchst du dafür nicht. Weitere Konzepte folgen in späteren Exkursen, sobald wir sie
wirklich verwenden.

## `String` und `&str`

```rust
let besitzt: String = String::from("Hallo"); // eigener, veränderbarer Text
let leiht: &str = "Hallo";                   // nur eine Sicht auf Text
```

- `String` besitzt seinen Text im Speicher, ähnlich wie ein `StringBuilder` in Java.
- `&str` ist eine Leihgabe: lesen ja, besitzen nein. Das `&` steht für *Referenz*.
- Faustregel: Funktionen nehmen `&str` als Parameter und geben `String` zurück.

## `Vec<u32>` und Slices

```rust
let ids: Vec<u32> = vec![15496, 995]; // wachsende Liste, wie ArrayList<Int>
let sicht: &[u32] = &ids;             // Slice: Leihgabe auf eine Folge
```

- `u32` ist eine vorzeichenlose 32-Bit-Zahl. Token-IDs sind nie negativ, und
  50 257 passt locker hinein.
- `Vec<u32>` ist die Liste, `&[u32]` die Sicht darauf, analog zu `String` und `&str`.

## Ownership

```rust,compile_fail
let a = String::from("Hallo");
let b = a;          // der Besitz wandert von a nach b
println!("{a}");    // Compilerfehler: a ist nicht mehr gültig
```

Jeder Wert hat genau einen Besitzer. Das ersetzt den Garbage Collector, denn der
Compiler prüft es vor dem Start und nicht erst zur Laufzeit. Mit `&` leihen wir einen
Wert aus, ohne den Besitz abzugeben.

## `Result` und `?`

```rust
fn lade(pfad: &str) -> Result<String, std::io::Error> {
    let text = std::fs::read_to_string(pfad)?; // bei Fehler: sofort zurück
    Ok(text)
}
```

- Rust hat keine Exceptions. Fehler sind normale Rückgabewerte: `Ok(wert)` oder
  `Err(fehler)`.
- `?` bedeutet: Bei `Err` gib den Fehler an den Aufrufer weiter, sonst nimm den Wert.
- Der Compiler zwingt uns, Fehler zu behandeln. Nichts wird stillschweigend geschluckt.

## `struct` und `impl`

```rust,ignore
struct Tokenizer { /* … */ }

impl Tokenizer {
    fn new() -> Self { /* … */ }
    fn encode(&self, text: &str) -> Vec<u32> { /* … */ }
}
```

- `struct` bündelt Daten, `impl` hängt Methoden daran.
- `&self` heisst: die Methode liest nur. `&mut self` hiesse: sie verändert.

## Module und `use`

`use` holt Namen in den aktuellen Bereich, damit man nicht jedes Mal den vollen Pfad
schreiben muss. Es entspricht `import` in Kotlin und Java.

```rust,ignore
use training::tokenizer::{Gpt2Tokenizer, VOCAB_SIZE};
```

- `training` ist das Crate, also der Name aus der `Cargo.toml`. Ein Beispiel wie
  `ch01.rs` liegt ausserhalb des Crates und behandelt es wie eine fremde Bibliothek.
- `tokenizer` ist das Modul, hier die Datei `tokenizer.rs`. In `lib.rs` wird es mit
  `pub mod tokenizer;` freigegeben.
- `{Gpt2Tokenizer, VOCAB_SIZE}` importiert zwei Namen auf einmal. Das ist die Kurzform
  für zwei getrennte `use`-Zeilen.
- `::` trennt die Teile eines Pfads, wo Kotlin einen Punkt verwendet.

Der Import funktioniert nur, weil beide Namen `pub` sind. Ohne `pub` meldet der
Compiler einen Fehler. Ohne `use` müsste man jedes Mal
`training::tokenizer::Gpt2Tokenizer::new()` ausschreiben.

## Grenzen

Ownership wird hier nur angerissen. Einen eigenen Exkurs bekommt das Thema, sobald uns
der Borrow-Checker im Code wirklich begegnet.
