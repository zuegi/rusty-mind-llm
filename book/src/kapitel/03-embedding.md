# 3. Embedding

## Das Problem

Der DataLoader aus Kapitel 2 liefert Fenster aus Token-IDs. Diese IDs sind
Kennnummern, keine Beschreibung der Tokens: Token `2` ist nicht doppelt so
wichtig wie Token `1`, und benachbarte IDs bedeuten keine ähnliche Bedeutung.

GPT-2 verarbeitet stattdessen für jedes Token einen *Vektor*: eine Liste von
Zahlen. Die Übersetzung von einer Token-ID in diesen Vektor übernimmt das
**Token-Embedding**.

## Eine Tabelle für die Tokens

Stellen wir uns eine Tabelle vor: Jedes Token hat eine Zeile. Jede Zeile enthält
gleich viele Zahlen. Diese Anzahl heisst *Embedding-Dimension*.

Für unser Lernbeispiel verwenden wir fünf Tokens und die Dimension vier:

| Token-ID | Token | Embedding-Vektor |
| -------- | ----- | ---------------- |
| 0 | Der | `[0.1, 0.0, 0.2, -0.1]` |
| 1 | Hund | `[0.2, -0.1, 0.4, 0.3]` |
| 2 | Katze | `[-0.2, 0.3, 0.1, 0.0]` |
| 3 | rennt | `[0.0, 0.2, -0.3, 0.4]` |
| 4 | . | `[0.1, -0.2, 0.0, 0.2]` |

Die Zahlen sind für dieses Beispiel bewusst vorgegeben. Sie sind nicht gelernt
und stellen keine bestimmten Eigenschaften wie "Tier" oder "Geschwindigkeit" dar.
Auch die Token-IDs gehören nur zu unserem Mini-Vokabular, nicht zum GPT-2-Tokenizer.

Die Tabelle hat die Form `[5, 4]`: fünf Zeilen mit jeweils vier Zahlen.
Das echte GPT-2-Vokabular hat 50 257 Tokens; seine Tabelle braucht entsprechend
50 257 Zeilen. Die Embedding-Dimension ist eine eigene Entscheidung über die
Modellgrösse, unabhängig von der Anzahl der Tokens.

## Nachschlagen statt Rechnen mit IDs

Die Eingabe `Der Hund rennt .` hat in unserem Beispiel die Token-IDs:

```text
[0, 1, 3, 4]
```

Das Token-Embedding schlägt für jede ID die entsprechende Tabellenzeile nach:

```text
0 -> [0.1,  0.0,  0.2, -0.1]
1 -> [0.2, -0.1,  0.4,  0.3]
3 -> [0.0,  0.2, -0.3,  0.4]
4 -> [0.1, -0.2,  0.0,  0.2]
```

Aus vier IDs werden vier Vektoren mit jeweils vier Zahlen. Dieselbe Token-ID
liefert bei unveränderter Tabelle immer denselben Vektor, unabhängig von ihrer
Position oder den Nachbartokens. Kontext entsteht erst später durch Attention.

## Was daran lernbar ist

Die Zahlen in der Tabelle sind Modellparameter. Bei einem neuen Modell werden
sie zufällig initialisiert. Erst das Training passt sie an die Aufgabe an:
das nächste Token vorherzusagen.

Dabei können die Vektoren nützliche Gemeinsamkeiten zwischen Tokens abbilden.
Benachbarte Token-IDs garantieren solche Gemeinsamkeiten nicht. Auch einzelne
Vektoreinträge bekommen keine fest vorgegebene Bedeutung.

Das Embedding selbst schlägt nur Vektoren nach. Wie die Tabellenwerte gelernt
werden, gehört in den separaten Trainingsbereich.

## Warum die Position fehlt

In "Hund jagt Katze" und "Katze jagt Hund" stehen dieselben Tokens, aber die
Sätze bedeuten etwas anderes. Das Token-Embedding liefert für "Hund" in beiden
Sätzen denselben Vektor. Es beschreibt nicht, an welcher Stelle das Token steht.

GPT-2 ergänzt deshalb ein **Positions-Embedding**: eine zweite lernbare Tabelle
mit einem Vektor für jede Position im Eingabefenster. Die Positionen beginnen
bei `0`. Jedes Fenster beginnt wieder bei Position `0`.

Diese Tabelle hat die Form `[max_context_length, embedding_dim]`. Die maximale
Kontextlänge bestimmt, wie viele Positionen das Modell unterstützt. Ein längeres
Fenster passt nicht in die Tabelle. Beide Embedding-Tabellen verwenden dieselbe
Embedding-Dimension.

## Token und Position addieren

Für jede Position schlägt GPT-2 zwei Vektoren nach: einen über die Token-ID,
den anderen über die Position. Dann addiert es die entsprechenden Zahlen.
Die Vektoren werden nicht aneinandergehängt; die Dimension bleibt unverändert.

Nehmen wir zweimal "Hund", also die Eingabe `[1, 1]`. Mit bewusst vorgegebenen
Positions-Vektoren ergibt sich:

```text
Token-Vektor "Hund": [0.2, -0.1,  0.4, 0.3]

Position 0:         [0.0,  0.2, -0.1, 0.1]
Summe:              [0.2,  0.1,  0.3, 0.4]

Position 1:         [0.1,  0.0,  0.2, 0.0]
Summe:              [0.3, -0.1,  0.6, 0.3]
```

Vier Zahlen plus vier Zahlen ergeben wieder vier Zahlen. Obwohl beide Tokens
dieselbe ID haben, sind die Summen hier verschieden: Die Positions-Vektoren
unterscheiden sich.

Diese Summen gehen als Eingabe an die folgenden Modellschichten. Sie verbinden
Token und Position, enthalten aber noch keine durch Attention berechneten
Informationen über die Nachbartokens.

GPT-2 verwendet gelernte absolute Positions-Embeddings. Auch diese Tabellenwerte
werden bei einem neuen Modell zufällig initialisiert und erst im Training
angepasst. Andere Modelle verwenden andere Positionsverfahren; wir bleiben hier
beim GPT-2-Prinzip.

## Vom Batch zum Tensor

Ein *Tensor* enthält Zahlen in einer festgelegten Form. Für unsere Eingabe können
wir ihn zunächst als Tabelle betrachten: Jede Zeile ist ein Fenster, jede Spalte
eine Tokenposition.

Ein Batch mit zwei Fenstern und jeweils drei Tokens sieht beispielsweise so aus:

```text
[
  [0, 1, 3],
  [0, 2, 4]
]
```

Die Form ist `[2, 3]`, allgemein `[batch_size, context_length]`. Die Einträge sind
Token-IDs, deshalb verwenden wir in Burn einen Ganzzahl-Tensor.

Das Token-Embedding ersetzt jede ID durch einen Vektor mit vier Zahlen:

```text
[
  [
    [0.1,  0.0,  0.2, -0.1],
    [0.2, -0.1,  0.4,  0.3],
    [0.0,  0.2, -0.3,  0.4]
  ],
  [
    [ 0.1,  0.0, 0.2, -0.1],
    [-0.2,  0.3, 0.1,  0.0],
    [ 0.1, -0.2, 0.0,  0.2]
  ]
]
```

Die Ausgabe hat eine zusätzliche Dimension: `[2, 3, 4]`, allgemein
`[batch_size, context_length, embedding_dim]`. Sie enthält
Gleitkommazahlen statt Token-IDs. Es sind weiterhin zwei Fenster mit drei
Tokenpositionen; an jeder Position stehen jetzt vier Zahlen.

## Positionen im Batch

Beide Fenster verwenden die Positions-IDs `[0, 1, 2]`. Das zweite Fenster setzt
die Zählung nicht bei `3` fort: Jedes Fenster startet bei `0`.

Für diese drei Positionen liefert die Positions-Tabelle drei Vektoren mit jeweils
vier Zahlen. Beim Addieren werden dieselben Positions-Vektoren für alle Fenster
im Batch verwendet. In Burn lässt sich dafür eine Form `[1, 3, 4]` verwenden:
Die erste Dimension wird bei der Addition auf beide Fenster erweitert. Dieses
Verhalten heisst *Broadcasting*.

Die gemeinsame Embedding-Ausgabe bleibt `[2, 3, 4]`. Das Zusammenfassen zu einem
Batch macht die Fenster nicht zu einem einzigen langen Text.

## Tensorformen in Rust

Burn beschreibt unsere Eingabe mit dem Typ `Tensor<B, 2, Int>`:

- `B` ist das Backend: Es legt fest, wie und wo gerechnet wird.
- `2` ist die Anzahl der Dimensionen, nicht die Fensterlänge.
- `Int` steht für Ganzzahlen, also unsere Token-IDs.

Sowohl die Form `[2, 3]` als auch `[8, 16]` hat zwei Dimensionen und damit denselben
Rust-Typ. Die konkreten Grössen stehen im Tensor.

Die Embedding-Ausgabe hat den Typ `Tensor<B, 3>`. Ohne dritte Typangabe verwendet
Burn Gleitkommazahlen. Die `3` zählt die drei Achsen: Batch, Position und
Embedding-Dimension. Auch eine Ausgabe mit der Form `[2, 7, 4]` hat diesen Typ,
obwohl jedes Fenster sieben Tokens enthält.

**Merksatz: Der Rust-Typ zählt die Achsen; die Tensorform nennt deren Grössen.**

## Das Embedding-Modul in Rust

Burn liefert Tensoren, Rechenoperationen und Modellschichten für neuronale Netze
in Rust. Eine kurze Einführung steht im Exkurs [Burn: Die Bausteine](../exkurse/burn-grundlagen.md).

Modellcode liegt in der eigenen Crate `crates/model`. Der DataLoader bleibt in
`crates/training`; das Modell hängt nicht vom Trainingsbereich ab. Wir verwenden
Burn 0.21 mit seiner hier erklärten backend-generischen Tensor-API.
Für Demo und Tests genügt das reine Rust-CPU-Backend `NdArray<f32>`, ohne GPU.

Rust-Hintergrund zu den folgenden Typen und Methoden steht im Exkurs
[Rust-Grundlagen für das Embedding](../exkurse/rust-grundlagen-embedding.md).

### Zwei Tabellen und ihre Grössen

Die Konfiguration enthält Vokabulargrösse, maximale Kontextlänge und
Embedding-Dimension. Das Modul enthält zwei Burn-`Embedding`-Tabellen:

```rust,ignore
{{#include ../../../crates/model/src/embedding.rs:config}}
```

`#[derive(Module)]` macht die Struktur zu einem Burn-Modul. Burn kann dadurch
auch die Parameter der beiden enthaltenen Tabellen verwalten. Diese Parameter
sind nicht gewöhnliche unveränderliche Zahlen, sondern lernbare Gewichte.

### Initialisieren

`new` erstellt die Konfiguration. `init` erstellt daraus die beiden Tabellen
auf dem gewählten Gerät. Burn initialisiert sie standardmässig aus einer
Normalverteilung mit Mittelwert `0` und Standardabweichung `1`.
Das ist ein Lernbeispiel für das GPT-2-Embedding-Prinzip, keine vollständige
Reproduktion von GPT-2 samt dessen Initialisierung und Dropout.

```rust,ignore
{{#include ../../../crates/model/src/embedding.rs:init}}
```

Alle drei Grössen müssen positiv sein. Ungültige Modellkonfigurationen sind
Programmierfehler und werden mit `assert!` ausdrücklich gemeldet.

### Die Vorwärtsberechnung

`forward` nimmt einen Batch aus Token-IDs entgegen und liefert die Summe aus
Token- und Positions-Vektoren:

```rust,ignore
{{#include ../../../crates/model/src/embedding.rs:forward}}
```

Schritt für Schritt:

1. `dims()` liefert die konkreten Grössen des Eingabetensors.
2. Die Prüfungen schliessen leere Batches und zu lange Fenster aus.
3. `arange` erzeugt Positionen von `0` bis zur Fensterlänge, ohne Endwert.
4. `reshape` macht daraus eine Zeile mit der Form `[1, context_length]`.
5. Beide `forward`-Aufrufe schlagen Tabellenzeilen nach.
6. `+` addiert die Vektoren und erweitert die Positions-Vektoren per Broadcasting
   auf alle Fenster im Batch.

Alle Token-IDs müssen im Bereich `0..vocab_size` liegen. Sie sind Tabellenindizes,
keine beliebigen Zahlen. Wie bei Burns `Embedding::forward` ist dies eine
Voraussetzung des Aufrufs; ungültige Indizes sind Programmierfehler. Unser
GPT-2-Tokenizer erfüllt sie bei passender Vokabulargrösse automatisch.

Die Methode trainiert nichts. Die Möglichkeit, Gradienten für die Tabellen zu
berechnen, entsteht mit einem Autodiff-Backend; eine Trainingsschleife bleibt
Aufgabe der separaten Trainings-Crate.

## Demo

```bash
cargo run -p model --example ch03
```

Die Demo verwendet unsere fünf Tokens und vier Zahlen pro Vektor. Sie ersetzt
die zufälligen Gewichte ausdrücklich durch die vorgegebenen Werte aus diesem
Kapitel. Das geschieht nur in der Demo, nicht in der regulären Initialisierung.

Der erste Eingabebatch ist `[[1, 1, 3], [0, 2, 4]]`: Das erste Fenster enthält
zweimal "Hund". So können wir dieselben Token-Vektoren mit unterschiedlichen
Positions-Vektoren direkt vergleichen. Die Demo bleibt bewusst unabhängig vom
echten Tokenizer: Dessen IDs passen nicht in eine Tabelle mit nur fünf Zeilen.

```rust,ignore
{{#include ../../../crates/model/examples/ch03.rs}}
```

Ausgabe:

```text
{{#include ../ausgaben/ch03.txt}}
```

### Was die Ausgabe zeigt

Die Eingabe-IDs werden für die Ausgabe zu einer flachen Liste zusammengefasst.
Die angezeigte Form `[2, 3]` zeigt, dass sie weiterhin zwei Fenster bilden.

Die ersten beiden Token-Vektoren sind gleich. Nach der Addition unterscheiden
sich ihre Summen genau wie in unserem Zahlenbeispiel. Das zweite Fenster
verwendet dieselben drei Positions-Vektoren wie das erste.

Die Zahlen werden mit einer Nachkommastelle ausgegeben. Intern arbeitet die
Demo mit `f32`; kleine Rundungsabweichungen bei der Addition sind normal.

## Tests

```bash
cargo test -p model
```

Die Tests in `crates/model/tests/embedding.rs` verwenden kleine, vorgegebene
Tabellen mit ganzen Zahlen als Gleitkommawerte. Dadurch lässt sich das Ergebnis
exakt prüfen: Nachschlagen, Addition, Positionen pro Fenster und kürzere Fenster.
Weitere Tests prüfen ungültige Grössen und zu lange oder leere Fenster.

Ein Test verwendet `Autodiff<NdArray<f32>>`. Er prüft, dass beide Tabellen die
richtige Form haben und für beide Gewichte Gradienten entstehen. Das bestätigt
ihre Lernbarkeit, ohne bereits eine Trainingsschleife einzuführen.
