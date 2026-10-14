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
- **Autograd:** Gradienten mit Burns Autodiff berechnen, auch für die
  Embedding-Tabellen.
- **Attention:** Kausale Self-Attention mit Q/K/V-Projektionen, skalierten
  Scores und Multi-Head Attention.
- **Transformer-Block:** Pre-LayerNorm, Feed-Forward-Netz und
  Residual-Verbindungen zu mehreren Blöcken zusammensetzen.
- **GPT-Sprachmodell:** `GptEmbedding`, Transformer-Blöcke und separate
  Ausgabeprojektion liefern Logits `[Batch, Position, Vokabular]`.
- **Loss und erster Trainingsschritt:** Cross-Entropy über die verschobenen
  DataLoader-Ziele, Burn-Autograd und genau ein SGD-Update für das GPT-Modell.

- **Trainingsschleife und Validierung:** Mehrere Epochen mit erhaltenem
  Optimizer; Loss-Messung auf getrennten Daten ohne Gewichtsupdates.
- **Checkpoints:** Modellgewichte mit Burns Burnpack-API speichern und
  mit expliziter Modellkonfiguration laden. Der Optimizer-Zustand wird noch
  nicht gespeichert.

Als Nächstes folgt Textgenerierung. `model` enthält bewusst nur
Modellberechnung; Trainingsinfrastruktur bleibt im Crate `training`.

## Projektstruktur

```text
crates/
  training/   Tokenizer, Daten, Loss, Training; Demos für Kapitel 1, 2, 8, 9 und 10
  model/      Burn-Modellbausteine und Demos für Kapitel 3 bis 7
book/         Deutsches mdBook mit Kapiteln und Exkursen
data/         Beispieltext und Herkunftshinweise
```

Modellberechnung und Training bleiben getrennt: `model` hängt nicht von
`training` ab. Die Trainingslogik gehört in den Trainingsbereich.
Das Projekt ist ein Cargo-Workspace; Demo und Tests des Modells verwenden
Burn 0.22 mit dem CPU-Device `Device::flex()` und benötigen keine GPU.
Backend und Autodiff werden über `Device` gewählt, nicht über Modell-Typparameter.
Flex ist Burns reines Rust-CPU-Backend und ersetzt das seit Burn 0.22
deprecated NdArray-Backend.

Checkpoints verwenden jetzt Burnpack (`.bpk`) statt des früheren
`DefaultRecorder`-Formats (`.mpk`). Alte Dateien werden nicht automatisch konvertiert.

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

# Kapitel 4: Gradienten mit Burn-Autodiff
cargo run -p model --example ch04

# Kapitel 5: Ein Attention-Kopf mit festen Vektoren
cargo run -p model --example ch05

# Kapitel 5, Schritt 2: Q/K/V-Projektionen und skalierte Scores
cargo run -p model --example ch05_projections

# Kapitel 5, Schritt 4: Multi-Head Attention mit Burn
cargo run -p model --example ch05_multi_head

# Kapitel 6, Schritt 1: LayerNorm mit festen Zahlen
cargo run -p model --example ch06_layer_norm

# Kapitel 6, Schritt 2: GELU und Feed-Forward-Netz
cargo run -p model --example ch06_gelu
cargo run -p model --example ch06_feed_forward

# Kapitel 6, Schritt 3: Gradienten mit und ohne Residual-Verbindung
cargo run -p model --example ch06_residual

# Kapitel 6, Schritt 6: Mehrere Blöcke hinter dem Embedding
cargo run -p model --example ch06

# Kapitel 7: GPT-Modell mit Ausgabe-Logits
cargo run -p model --example ch07

# Kapitel 8: Cross-Entropy mit festen Logits
cargo run -p training --example ch08_loss

# Kapitel 8: Genau ein SGD-Schritt auf einem Mini-Batch
cargo run -p training --example ch08

# Kapitel 9: Trainingsschleife und getrennte Validierung
cargo run -p training --example ch09

# Kapitel 10: Modellgewichte speichern und laden
cargo run -p training --example ch10
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
[mdBook](https://rust-lang.github.io/mdBook/) und das Plugin
[mdbook-mermaid](https://github.com/badboy/mdbook-mermaid) (für Diagramme) benötigt:

```bash
cargo install mdbook --version 0.5.4 --locked
cargo install mdbook-mermaid --version 0.17.1
mdbook serve book --port 3000
```

Danach ist das Buch unter <http://localhost:3000> erreichbar.
Ohne lokalen Server lässt es sich mit `mdbook build book` nach `target/book`
bauen.

`mdbook-mermaid` wird bewusst ohne `--locked` gebaut: Sein veröffentlichtes
Lockfile bindet die mdBook-API an 0.5.0 und erzeugt mit mdBook 0.5.4 einen
Versionshinweis. Ohne dieses Lockfile wird die kompatible 0.5.4-API verwendet.
Bei einer bereits installierten Plugin-Version erzwingt `--force` den Neuaufbau:

```bash
cargo install mdbook-mermaid --version 0.17.1 --force
```

## Tests

```bash
cargo test --workspace

# Tests der Attention-Demo
cargo test -p model --example ch05

# Tests des trainierbaren Attention-Kopfes
cargo test -p model --test attention

# GPT-Modell: Formen, Parameter und Gradienten
cargo test -p model --test gpt

# Cross-Entropy, Batch-Tensoren und genau ein SGD-Update
cargo test -p training --test ch08

# Trainingsschleife und Auswertung ohne Gewichtsupdates
cargo test -p training --test ch09

# Modellgewichte speichern und laden
cargo test -p training --test ch10
```

Die Tests prüfen unter anderem Datenfenster, Token-/Positions-Nachschlagen,
kausale Attention, Tensorformen, Parameterzahlen, Gradientenfluss und ein
SGD-Update anhand der Cross-Entropy. Sie
ersetzen kein vollständiges Modelltraining.

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
