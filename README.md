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
- **Textgenerierung:** Mit `argmax` schrittweise Token erzeugen, den Kontext
  begrenzen und optional gespeicherte Modellgewichte laden. Die Demo nutzt
  weiterhin ein künstliches Vokabular mit fünf Einträgen.
- **Echter Text:** Kapitel 12 verbindet den GPT-2-Tokenizer mit einem
  begrenzten CPU-Trainingsversuch auf `data/the-verdict.txt`: getrennte
  Datenbereiche, wenige Fenster, höchstens drei SGD-Updates und
  Loss-Messungen vor und nach dem Training. Ein eigener Aufruf ergänzt
  ganze Epochen, Checkpoints und Generierung mit dem GPT-2-Tokenizer.

- **Auswertung und Overfitting:** Kapitel 13 misst Trainings- und
  Validierungs-Loss je Epoche, berechnet die Perplexity (`exp(loss)`),
  meldet ein Overfitting-Warnsignal und speichert nur den Stand mit dem
  besten Validierungs-Loss (einschliesslich der Ausgangsmessung).
  Das Buch zeigt echte Vergleichsläufe, Loss-Kurven und deren Einordnung.

Kapitel 11 bis 15 sind in `main`. Kapitel 14 ergänzt
[bessere Optimierung](book/src/kapitel/14-bessere-optimierung.md):
SGD, Momentum, Adam, Warmup mit linearer Absenkung und Norm-Clipping
je Parametergradiententensor. Fünf begrenzte Echttext-Läufe samt
Messausgaben sind enthalten. Kapitel 15 ergänzt eine separate
[Resume-Demo](book/src/kapitel/15-training-fortsetzen.md) mit Adam-Zustand,
Lernratenplan und Wiederaufnahme innerhalb einer Epoche.
Den Lernstatus beschreibt [NächsteKapitel.md](NächsteKapitel.md).
`model` enthält bewusst nur Modellberechnung; Trainingsinfrastruktur bleibt
im Crate `training`.

## Projektstruktur

```text
crates/
  training/   Tokenizer, Daten, Loss, Training; Demos für Kapitel 1, 2 und 8 bis 14
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

# Kapitel 5.3: Ein Attention-Kopf mit festen Vektoren
cargo run -p model --example ch05

# Kapitel 5.8: Q/K/V-Projektionen, skalierte Scores und kausale Maske
cargo run -p model --example ch05_projections

# Kapitel 5.9: Multi-Head Attention mit Burn
cargo run -p model --example ch05_multi_head

# Kapitel 6.1: LayerNorm mit festen Zahlen
cargo run -p model --example ch06_layer_norm

# Kapitel 6.2: GELU und Feed-Forward-Netz
cargo run -p model --example ch06_gelu
cargo run -p model --example ch06_feed_forward

# Kapitel 6.3: Gradienten mit und ohne Residual-Verbindung
cargo run -p model --example ch06_residual

# Kapitel 6.6: Mehrere Blöcke hinter dem Embedding
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

# Kapitel 11: Token schrittweise generieren (ohne Checkpoint: Zufallsgewichte)
cargo run -p training --example ch11

# Kapitel 12: Echter Text, höchstens drei Updates mit GPT-2-Vokabular
cargo run -p training --example ch12

# Kapitel 14: Fünf Optimizer-Vergleiche, je drei Epochen auf acht Trainingsfenstern
cargo run --release -p training --example ch14

# Kapitel 15: Nach Update 5 speichern; in neuem Prozess bis Update 12 fortsetzen
cargo run --release -p training --example ch15 -- train checkpoints/ch15-teil 5
cargo run --release -p training --example ch15 -- resume checkpoints/ch15-teil checkpoints/ch15-fortgesetzt 12
```

`ch15` speichert Modell- und Adam-Records sowie Konfiguration, verwendete
Tokens, Update-Stand und Prüfsummen in einem neuen Checkpoint-Ordner.
Vorhandene Zielordner werden nicht überschrieben. Die Ziel-Update-Nummer
ist absolut; der Lernratenplan umfasst von Anfang an zwölf Updates.
`ch12_full`, `ch13` und `ch14` behalten ihr bisheriges Verhalten.

Kapitel 12 prüft ein Trainingszeitbudget von 30 Sekunden zwischen Updates.
Ein laufendes Update und die Loss-Messungen werden nicht unterbrochen;
das ist keine harte Prozess-Laufzeitgrenze. Die Demo speichert keinen
Checkpoint. Das kleine Modell ist kein vortrainiertes GPT-2.

Der vollständige Ablauf startet nur ausdrücklich, mit einer bis drei
Epochen. Er hat keinen Zeitabbruch und speichert nach jeder Epoche die
Modellgewichte; ein erneuter Trainingsaufruf startet neu, nicht als Resume:

```bash
# Längerer Lauf: eine volle Epoche auf dem enthaltenen Text
cargo run --release -p training --example ch12_full -- train checkpoints/ch12-model.bpk 1

# Passende Gewichte laden und 16 neue Tokens erzeugen
cargo run --release -p training --example ch12_full -- generate checkpoints/ch12-model.bpk "I HAD always" 16
```

Die Generierung verwendet `argmax` und denselben GPT-2-Tokenizer.
Sinnvolle Sätze sind nicht garantiert; ungültiges UTF-8 wird als Fehler
gemeldet. Details und Grenzen stehen in
[Kapitel 12](book/src/kapitel/12-echter-text-und-tokenizer.md).

Kapitel 13 startet nur ausdrücklich, mit 1 bis 20 Epochen, und speichert den
Stand mit dem besten Validierungs-Loss:

```bash
cargo run --release -p training --example ch13 -- checkpoints/ch13-best.bpk 20

# Gleiche Daten und Initialisierung, kleinere Lernrate
cargo run --release -p training --example ch13 -- checkpoints/ch13-lr002.bpk 20 --learning-rate 0.02
```

Jeder Aufruf startet neu und ersetzt einen vorhandenen Checkpoint am
gewählten Pfad zuerst durch den Ausgangsstand. Unterschiedliche Pfade
bewahren die Modelle verschiedener Experimente.

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

### GitHub Pages

Jeder Push auf `main` baut das Buch automatisch und veröffentlicht es über
GitHub Pages. Die Seite ist anschließend unter
<https://zuegi.github.io/rusty-mind-llm/> erreichbar.

Falls GitHub Pages für das Repository noch nicht aktiviert ist, einmalig unter
**Settings → Pages → Build and deployment → Source** die Option **GitHub
Actions** auswählen. Danach kann der Workflow unter **Actions → Deploy mdBook
site to Pages** manuell gestartet werden; weitere Pushes auf `main` lösen ihn
automatisch aus.

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

# Echter Text, Tensorformen und begrenzter Trainingsversuch
cargo test -p training --example ch12

# Ganze Epochen auf kurzem Testtext, Checkpoint und Generierung
cargo test -p training --example ch12_full

# Perplexity, Bestmodell-Regel und Overfitting-Signal
cargo test -p training --example ch13

# Trainings-Resume: alle Gewichte gegen einen ununterbrochenen Lauf vergleichen
cargo test -p training --example ch15
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
