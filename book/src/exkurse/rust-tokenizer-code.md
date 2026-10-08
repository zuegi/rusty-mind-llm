# Exkurs: Der Tokenizer-Code im Detail

Ein Durchgang durch `crates/training/src/tokenizer.rs`. Er erklärt, wie Rust-Code
aufgebaut ist, am Beispiel des `Gpt2Tokenizer`.

## Was ist `tokenizer.rs`?

Die Datei ist ein **Modul** mit dem Namen `tokenizer`. Über

```rust,ignore
pub mod tokenizer;
```

in `lib.rs` wird sie Teil des Crates `training`. Von aussen heisst der Tokenizer
`training::tokenizer::Gpt2Tokenizer`.

Eine Klasse im Sinne von Java oder Kotlin gibt es in Rust nicht. `Gpt2Tokenizer` ist ein
`struct` (die Daten), seine Methoden stehen in einem separaten `impl`-Block.

## 1. Konstanten

```rust,ignore
pub const VOCAB_SIZE: usize = 50_257;
pub const END_OF_TEXT_ID: u32 = 50_256;
```

- `const` ist ein Wert, der zur Compilezeit feststeht. Der Typ ist Pflicht.
- `pub` macht den Namen ausserhalb des Moduls sichtbar. Ohne `pub` ist er privat.
- Der Unterstrich in Zahlen dient nur der Lesbarkeit.
- `usize` ist der Typ für Grössen und Indizes, `u32` der Typ der Token-IDs.

## 2. Eigener Fehlertyp

```rust,ignore
#[derive(Debug)]
pub struct TokenizerError(String);

impl fmt::Display for TokenizerError { /* … */ }
impl std::error::Error for TokenizerError {}
```

- `struct X(String)` ist ein *Tuple-Struct*: ein Feld ohne Namen.
- `#[derive(Debug)]` lässt den Compiler die Debug-Ausgabe (`{:?}`) erzeugen.
- `impl Display` legt fest, wie der Fehler als Text aussieht.
- `impl std::error::Error` macht ihn zu einem Standardfehler. `Display` und `Error`
  sind *Traits*, vergleichbar mit Interfaces.

## 3. Das `struct`

```rust,ignore
pub struct Gpt2Tokenizer {
    bpe: &'static CoreBPE,
}
```

- Das Feld `bpe` ist privat. Von aussen sind nur die Methoden sichtbar. Das ist die
  Kapselung der Bibliothek `tiktoken-rs`.
- `&'static` ist eine Referenz, die das ganze Programm lang gültig ist. Sie zeigt auf die
  einmal geladene Instanz der Bibliothek. Wir besitzen sie nicht.

## 4. Konstruktor

```rust,ignore
pub fn new() -> Self {
    Self { bpe: tiktoken_rs::r50k_base_singleton() }
}
```

- `new` ist nur eine Konvention, kein Schlüsselwort.
- `Self` steht für den Typ, in dessen `impl` wir sind, hier `Gpt2Tokenizer`.
- Der Singleton lädt das Vokabular beim ersten Aufruf und gibt danach dieselbe Instanz
  zurück.

## 5. `encode` und `decode`

```rust,ignore
pub fn encode(&self, text: &str) -> Vec<u32> { /* … */ }
pub fn decode(&self, ids: &[u32]) -> Result<String, TokenizerError> { /* … */ }
```

- Parameter sind Leihgaben (`&str`, `&[u32]`), Rückgaben sind Besitz (`Vec`, `String`).
  Das ist die Faustregel aus dem Exkurs "Rust-Grundlagen für den Tokenizer".
- `encode` kann nicht scheitern und gibt direkt `Vec<u32>` zurück. `decode` kann
  scheitern, wenn die Bytes kein gültiges UTF-8 ergeben, und gibt deshalb ein `Result`
  zurück.
- `.map_err(|e| TokenizerError(e.to_string()))` wandelt den Fehler der Bibliothek in
  unseren eigenen um. `|e| …` ist ein Lambda. So kennt der Rest des Codes `tiktoken-rs`
  nicht.

## 6. `pieces` und `piece`

```rust,ignore
self.encode(text).into_iter().map(|id| self.piece(id)).collect()
```

Eine Iterator-Kette: IDs durchgehen, jede in Text umwandeln, zu einem `Vec<String>`
sammeln. In Kotlin wäre das `map { … }` und `toList()`.

`piece` ist privat und ein reiner Helfer. `String::from_utf8_lossy` ersetzt ungültige
Bytes durch `�`. Deshalb zeigt die Demo bei Umlauten halbe Zeichen.

## 7. `Default`

```rust,ignore
impl Default for Gpt2Tokenizer {
    fn default() -> Self { Self::new() }
}
```

`Default` ist der Standard-Trait für einen Standardwert. Er ist nicht zwingend, aber
eine Rust-Konvention bei einem `new()` ohne Parameter. Das Werkzeug `clippy` fordert ihn.

## 8. Tests im selben Modul

```rust,ignore
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_erhaelt_text() { /* … */ }
}
```

- `#[cfg(test)]` bedeutet: Der Block wird **nur bei `cargo test`** kompiliert. Beim
  normalen Build, auch mit `--release`, existiert er nicht und landet nicht in der
  ausgelieferten Binary.
- `use super::*;` holt alles aus dem umgebenden Modul in den Test.
- Als Kind-Modul sehen die Tests auch private Funktionen.
- Jede Funktion mit `#[test]` ist ein Test. Er schlägt fehl, sobald ein `assert!` oder
  `assert_eq!` nicht zutrifft oder die Funktion mit `panic!` abbricht.

Rust kennt zwei Arten von Tests:

| Art            | Ort                      | Sieht                   | Einsatz                    |
| -------------- | ------------------------ | ----------------------- | -------------------------- |
| Unit-Test      | im Modul, `#[cfg(test)]` | auch Privates           | einzelne Funktionen        |
| Integrationstest | Ordner `tests/`        | nur die öffentliche API | Zusammenspiel von aussen   |

Für den Tokenizer reichen Unit-Tests. Beim Modell (Kapitel 6) nutzen wir
Integrationstests.

## Zwei offene Punkte

- `unwrap_or_default()` in `piece` schluckt einen Fehler still und liefert dann leeren
  Text. Das ist für die Anzeige gedacht und vertretbar, weil `pieces` nur eigene,
  gültige IDs verwendet.
- `new()` kann abstürzen, falls das eingebettete Vokabular defekt wäre (`unwrap` in
  `tiktoken-rs`). Bei einem festen, eingebetteten Vokabular ist das praktisch
  ausgeschlossen.
