# 2. DataLoader

## Das Problem

Ein Sprachmodell lernt eine einzige Aufgabe: das nächste Token vorherzusagen. Aus dem
langen Token-Strom von Kapitel 1 brauchen wir deshalb viele kleine Paare aus
*Eingabe* und *Ziel*. Diese Aufbereitung übernimmt der DataLoader.

## Fenster mit verschobenem Ziel

Wir schneiden ein Fenster der Länge `context_length` aus dem Strom. Das ist die
Eingabe. Das Ziel ist dasselbe Fenster, um ein Token nach rechts verschoben.

Beispiel mit der Länge 4 und dem Strom `10 11 12 13 14 15 16 17 18`:

| Fenster | Eingabe          | Ziel             |
| ------- | ---------------- | ---------------- |
| 1       | `10 11 12 13`    | `11 12 13 14`    |
| 2       | `14 15 16 17`    | `15 16 17 18`    |

Jede Position der Eingabe lernt ihr eigenes nächstes Token: Nach `10` kommt `11`,
nach `10 11` kommt `12`, und so weiter. Aus einem Fenster entstehen also
`context_length` Lernaufgaben auf einmal.

Das letzte Ziel (`18`) braucht ein Token hinter dem Fenster. Ein Fenster passt deshalb
nur, wenn `context_length + 1` Token übrig sind.

## Lückenlos statt überlappend

Wir schneiden die Fenster lückenlos: Das nächste Fenster beginnt direkt hinter dem
vorigen. Überlappende Fenster (Schrittweite kleiner als die Fensterlänge) würden aus
demselben Text mehr Beispiele machen. Der Gewinn ist gering, denn das Modell lernt an
jeder Position ohnehin etwas, und die Rechenzeit steigt stark. Wir verzichten darum auf
einen Parameter dafür.

Wenn die Länge des Stroms nicht aufgeht, fallen die letzten Token weg. Bei einem
Textstrom ist das harmlos.

## Erst teilen, dann fenstern

Wir brauchen zwei Datensätze:

- **Training** zum Lernen.
- **Validierung** zum Prüfen mit Text, den das Modell nie gesehen hat.

Die Reihenfolge ist wichtig: **zuerst den Token-Strom teilen, dann fenstern.** Würden wir
zuerst fenstern und danach die Fenster verteilen, könnten sich Fenster an der Grenze
Token teilen. Das Modell hätte dann Teile des Validierungstextes schon beim Training
gesehen, und die Validierung wäre zu optimistisch (*Leakage*). Mit der Reihenfolge
"teilen, dann fenstern" ist sie ausgeschlossen, weil beide Hälften nie dieselben Token
enthalten.

Zur Orientierung: Üblich sind 90 % Training und 10 % Validierung.

## Batches

Das Modell verarbeitet mehrere Fenster gleichzeitig. Ein *Batch* bündelt `batch_size`
Fenster:

```text
Eingabe: [batch_size, context_length]
Ziel:    [batch_size, context_length]
```

Das ist effizienter als einzelne Fenster, weil Grafikchip und Prozessor viele Zahlen
gleichzeitig rechnen.

## Mischen

In jeder *Epoche* (ein Durchgang durch alle Fenster) mischen wir die Reihenfolge der
Fenster. Sonst sieht das Modell die Beispiele immer in derselben Folge und kann sich
daran gewöhnen. Das Mischen verwendet einen festen Startwert (*Seed*). Damit ist ein
Trainingslauf bei gleichem Seed wiederholbar.

## Der DataLoader in Rust

Der Code liegt in `crates/training/src/dataset.rs`. Er arbeitet nur mit `Vec<u32>` und
kennt Burn noch nicht. Die Umwandlung in Tensoren kommt in Kapitel 3. Rust-Hintergrund
steht im Exkurs [Rust-Grundlagen für den DataLoader](../exkurse/rust-grundlagen-dataloader.md).

### Typen

`Window` ist ein Trainingsbeispiel, `Batch` bündelt mehrere. Ein eigener Fehlertyp meldet,
wenn der Text zu kurz ist. Das ist ein erwartbarer Fehler und kein Programmierfehler,
deshalb ein `Result` und kein Abbruch:

```rust,ignore
{{#include ../../../crates/training/src/dataset.rs:types}}
```

### Erst teilen

`split_tokens` gibt zwei Slices auf denselben Strom zurück, ohne zu kopieren. Sie teilen
keine Token:

```rust,ignore
{{#include ../../../crates/training/src/dataset.rs:split}}
```

### Lückenlose Fenster

Eingabe und Ziel entstehen aus demselben Strom. Das Ziel beginnt ein Token später.
`chunks_exact` schneidet lückenlos und lässt einen Rest weg, `zip` führt beide Seiten
Fenster für Fenster zusammen:

```rust,ignore
{{#include ../../../crates/training/src/dataset.rs:new}}
```

### Eine Epoche

`epoch` mischt die Fensternummern mit festem Seed und bündelt sie zu Batches. Ein
unvollständiger letzter Batch wird verworfen, damit alle Batches dieselbe Form haben:

```rust,ignore
{{#include ../../../crates/training/src/dataset.rs:epoch}}
```

## Demo

Die Demo `ch02` tokenisiert einen Text, teilt ihn und zeigt Fenster und Batches. Ohne
Argument verwendet sie einen kurzen Beispieltext. Mit einer eigenen Textdatei:

```bash
cargo run -p training --example ch02 -- pfad/zum/text.txt
```

```rust,ignore
{{#include ../../../crates/training/examples/ch02.rs}}
```

Ausgabe mit dem eingebauten Beispieltext:

```text
{{#include ../ausgaben/ch02.txt}}
```

### Was die Ausgabe zeigt

- **Das Ziel ist die Eingabe, um ein Token verschoben.** Die erste Eingabe endet mit
  `d`, das Ziel beginnt mit ` Hund` und endet mit `durch`. Das Ziel enthält also jeweils
  das nächste Token.
- **Die Aufteilung stimmt.** Von 318 Token gehen 286 ins Training und 32 in die
  Validierung (rund 10 %).
- **Fenster und Batches.** 286 Token ergeben bei Länge 8 genau 35 Fenster. Bei Batchgrösse
  2 sind das 17 Batches, das 35. Fenster fällt weg.
- **Batchform.** Jeder Batch hat 2 Zeilen mit je 8 Token, also `[2, 8]`.

## Tests

Die beiden wichtigsten Tests prüfen die Verschiebung um eins und die Trennung von Training
und Validierung:

```rust,ignore
{{#include ../../../crates/training/src/dataset.rs:test_shift}}
```

```rust,ignore
{{#include ../../../crates/training/src/dataset.rs:test_split}}
```

Weitere Tests decken ab: abgeschnittener Rest, Fehler bei zu kurzem Strom, vollständige
Batches mit richtiger Form, jedes Fenster höchstens einmal pro Epoche und dieselbe
Reihenfolge bei gleichem Seed. Ausführen:

```bash
cargo test -p training
```

## Grenzen

- Der Beispieltext in der Demo ist sehr kurz und nur zum Anschauen gedacht. Für echtes
  Training braucht es einen längeren Text, den man per Pfad übergibt. Wir packen keinen
  Text ins Repository.
- Wenige Token bedeuten wenige Fenster. Bei einem kleinen Korpus lernt das Modell vor allem
  auswendig und verallgemeinert nicht.
- Die Ausgabe in `ausgaben/ch02.txt` ist ein Schnappschuss. Neu erzeugen mit
  `cargo run -q -p training --example ch02 > book/src/ausgaben/ch02.txt`.
- Alle Batches liegen für eine Epoche im Speicher. Für unsere Textgrössen genügt das.

## Ausblick

Im nächsten Kapitel wandeln wir die Token-IDs in Vektoren um (Embeddings) und lernen
dabei, wie Burn Tensoren darstellt. Dort werden aus `Vec<u32>` erstmals Tensoren.
