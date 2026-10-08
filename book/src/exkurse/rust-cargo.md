# Exkurs: Cargo und Workspace

Dieser Exkurs erklärt, was du im Projektordner siehst und welche Befehle du brauchst.
Vorwissen ist nicht nötig.

## Cargo in einem Satz

Cargo ist Build-Tool, Paketmanager und Test-Runner in einem. Es ersetzt, was du aus der
JVM-Welt von Maven oder Gradle kennst, und kommt direkt mit Rust.

## Crate

Ein *Crate* ist die kleinste Einheit, die Rust kompiliert. Es gibt zwei Arten:

- **Bibliothek (`lib`)**: Einstiegspunkt `src/lib.rs`. Anderer Code kann sie verwenden.
- **Programm (`bin`)**: Einstiegspunkt `src/main.rs` mit einer `main`-Funktion.

In diesem Buch gibt es zwei Crates:

| Crate      | Aufgabe                                                |
| ---------- | ------------------------------------------------------ |
| `model`    | Die Architektur des Sprachmodells (reine Modelllogik)  |
| `training` | Daten, Tokenizer, Trainingsschleife, Checkpoints, CLI  |

Die Abhängigkeit geht in genau eine Richtung: `training` darf `model` verwenden,
`model` kennt `training` nicht. So bleibt die Modelllogik von der Infrastruktur getrennt.

## `Cargo.toml` lesen

Jedes Crate hat eine `Cargo.toml`. Beispiel:

```toml
[package]
name = "training"
version = "0.1.0"
edition = "2024"

[dependencies]
model = { path = "../model" }
```

- `[package]` beschreibt das Crate. `edition` ist die Sprachversion von Rust.
- `[dependencies]` listet Abhängigkeiten. Entweder aus dem Paketregister
  [crates.io](https://crates.io) (`rand = "0.9"`) oder lokal per `path`.

## Workspace

Ein *Workspace* fasst mehrere Crates zusammen. Die `Cargo.toml` im Projektstamm
listet sie auf:

```toml
[workspace]
resolver = "2"
members = ["crates/model", "crates/training"]
```

Alle Crates im Workspace teilen sich eine `Cargo.lock` (exakte Versionen aller
Abhängigkeiten) und ein `target/`-Verzeichnis. Vergleichbar mit einem
Maven-Multi-Module-Projekt.

## Verzeichnisse

| Pfad              | Bedeutung                                                  |
| ----------------- | ---------------------------------------------------------- |
| `src/lib.rs`      | Einstieg einer Bibliothek                                  |
| `src/main.rs`     | Einstieg eines Programms                                   |
| `examples/`       | Kleine ausführbare Beispiele, je Kapitel eines             |
| `tests/`          | Integrationstests                                          |
| `target/`         | Build-Ausgabe, wird erzeugt und nicht ins Git eingecheckt  |
| `Cargo.lock`      | Exakte Versionen, bleibt unverändert im Repository         |

## Die wichtigsten Befehle

```bash
cargo build                       # kompilieren
cargo test                        # alle Tests ausführen
cargo test -p model               # nur Tests eines Crates (-p = package)
cargo run --example ch01          # ein Beispiel ausführen (ab Kapitel 1)
cargo add rand                    # Abhängigkeit hinzufügen
```

Beim ersten Build lädt Cargo alle Abhängigkeiten herunter und kompiliert sie.
Das kann einige Minuten dauern, danach geht es deutlich schneller.

## Debug oder Release

Standardmäßig baut Cargo im *Debug*-Modus: schnell kompiliert, aber der erzeugte
Code ist ohne Optimierungen langsam. Mit `--release` wird optimiert:

```bash
cargo run --release --example ch01
```

Für Training ist der Unterschied entscheidend. Release ist bei Tensor-Berechnungen
oft um ein Vielfaches schneller. Tests und kleine Experimente laufen auch ohne.

## Vergleich mit Kotlin und Maven

| Kotlin / Maven                  | Rust / Cargo                       |
| ------------------------------- | ---------------------------------- |
| `pom.xml`                       | `Cargo.toml`                       |
| Maven-Multi-Module-Projekt      | Workspace                          |
| Modul                           | Crate                              |
| Maven Central                   | crates.io                          |
| `mvn test`                      | `cargo test`                       |
| `target/`                       | `target/`                          |
| Dependency-Lockfile (Gradle)    | `Cargo.lock`                       |

## Grenzen

Dieser Exkurs lässt bewusst weg: Features (optionale Funktionen eines Crates),
Build-Skripte und das Veröffentlichen von Crates. Das brauchen wir im Buch nicht.
Wer mehr will, findet es im [Cargo Book](https://doc.rust-lang.org/cargo/).
