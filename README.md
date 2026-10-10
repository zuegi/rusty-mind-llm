# rusty-mind-llm

Ein persönliches Lernprojekt: Schritt für Schritt ein GPT-2-artiges Sprachmodell
in **Rust mit Burn** aufbauen und verstehen. Die deutsche Dokumentation erklärt
die Konzepte, den Rust-Code und kleine, nachvollziehbare Demos.

## Ziel und aktueller Stand

Im Mittelpunkt steht das Lernen, nicht ein fertiges oder produktionsreifes LLM.
Das Projekt richtet sich auch an Rust-Einsteiger. Exkurse erläutern Rust,
Burn und Unterschiede zu Kotlin und Java.

Bisher umgesetzt:

- **Tokenisierung:** Text mit dem vorhandenen GPT-2-Tokenizer in Token-IDs
  umwandeln und wieder zurück.
- **DataLoader:** Den Token-Strom in Training und Validierung teilen, Fenster
  mit verschobenen Zielen bilden und zu Batches zusammenfassen.
- **Embedding:** Lernbare Token- und absolute Positions-Embeddings mit Burn
  nachschlagen und addieren. Eine Mini-Demo mit festen Gewichten macht die
  Berechnung nachprüfbar.

Attention, die vollständige GPT-Architektur und eine Trainingsschleife sind
noch nicht umgesetzt. Die Embedding-Demo ist bewusst noch nicht mit dem
echten Tokenizer und DataLoader verbunden.

## Projektstruktur

```text
crates/
  training/   Tokenizer, Datenaufbereitung und Demos für Kapitel 1 und 2
  model/      Burn-Modellbausteine und Embedding-Demo für Kapitel 3
book/         Deutsches mdBook mit Kapiteln und Exkursen
data/         Beispieltext und Herkunftshinweise
```

Modellberechnung und Training bleiben getrennt: `model` hängt nicht von
`training` ab. Die spätere Trainingslogik gehört in den Trainingsbereich.
Das Projekt ist ein Cargo-Workspace; Demo und Tests des Modells verwenden
Burns CPU-Backend `NdArray<f32>` und benötigen keine GPU.

## Demos starten

Voraussetzung ist eine aktuelle Rust-Toolchain mit Cargo und Unterstützung
für die Rust-Edition 2024. Cargo lädt die benötigten Abhängigkeiten.

```bash
# Kapitel 1: Tokenisierung
cargo run -p training --example ch01

# Kapitel 2: Fenster und Batches
cargo run -p training --example ch02

# Kapitel 3: Token- und Positions-Embedding
cargo run -p model --example ch03
```

Der DataLoader kann auch den enthaltenen Beispieltext verwenden:

```bash
cargo run -p training --example ch02 -- data/the-verdict.txt
```

Herkunft und Hinweise zum Text stehen in [data/README.md](data/README.md).

## Dokumentation lesen

Die Dokumentation liegt in [book/src](book/src). Einstieg:
[Einleitung](book/src/einleitung.md) und
[Inhaltsverzeichnis](book/src/SUMMARY.md).

Für die lokale HTML-Ausgabe wird zusätzlich
[mdBook](https://rust-lang.github.io/mdBook/) benötigt:

```bash
cargo install mdbook --locked
mdbook serve book --port 3000
```

Danach ist das Buch unter <http://localhost:3000> erreichbar.
Ohne lokalen Server lässt es sich mit `mdbook build book` nach `target/book`
bauen.

## Tests

```bash
cargo test --workspace
```

Die Tests prüfen unter anderem Datenfenster, Token-/Positions-Nachschlagen,
Tensorformen und die Lernbarkeit der Embedding-Gewichte. Sie ersetzen kein
vollständiges Modelltraining.

## Architektur und Quellen

Der Lernpfad orientiert sich am verwandten Projekt
[learn-neural-networks](https://github.com/zuegi/learn-neural-networks)
und an Sebastian Raschkas Buch **„Large Language Models selbst programmieren“**
(dpunkt.verlag), im englischen Original
**„Build a Large Language Model (From Scratch)“** (Manning Publications).

Die Konzepte werden hier in Rust mit Burn erarbeitet statt in Python/PyTorch
oder Kotlin. Dieses Repository ist ein eigenständiges Lernprojekt, keine
offizielle Buchimplementierung und keine vollständige GPT-2-Reproduktion.

## Disclaimer & AI Notice

### Deutsch

Code und Teile der Dokumentation entstehen mit Unterstützung von KI/LLMs.
Sie werden zu Lernzwecken erarbeitet und zusammengeführt. Es besteht kein
Anspruch auf wissenschaftliche Korrektheit, Vollständigkeit, optimale
Performance oder Eignung für produktiven Einsatz. Fehler sind möglich;
die Nutzung erfolgt auf eigene Verantwortung.

### English

This is a personal learning project exploring a GPT-2-style language model
in Rust with Burn. Code and parts of the documentation are developed with
AI/LLM assistance for educational purposes. There is no guarantee of
correctness, completeness, optimal performance, or production suitability.
Use at your own risk.

## Lizenz

Der Projektcode steht unter der [MIT-Lizenz](LICENSE).
Hinweise zu externen Daten stehen in [data/README.md](data/README.md).
