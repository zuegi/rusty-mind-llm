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

**Wichtig: `context_length` zählt die Tokenpositionen pro Fenster.
`embedding_dim` zählt die Zahlen pro Position, nicht die Positionen.**

Für die Form `[2, 3, 4]` bedeutet das:

| Achse | Grösse | Bedeutung |
| ----- | ------ | --------- |
| `batch_size` | 2 | Zwei Fenster im Batch |
| `context_length` | 3 | Drei Tokenpositionen pro Fenster |
| `embedding_dim` | 4 | Vier Zahlen im Vektor jeder Position |

Das erste Fenster lässt sich deshalb so lesen:

```text
Position 0 -> [0.1,  0.0,  0.2, -0.1]
Position 1 -> [0.2, -0.1,  0.4,  0.3]
Position 2 -> [0.0,  0.2, -0.3,  0.4]
```

**Vor dem Embedding: `[2, 3]` Token-IDs. Nach dem Embedding: `[2, 3, 4]`
Zahlenwerte.** Die Anzahl der Fenster und Tokenpositionen bleibt gleich;
jede einzelne ID wird durch einen Vektor ersetzt.

## Positionen im Batch

Beide Fenster verwenden die Positions-IDs `[0, 1, 2]`. Das zweite Fenster setzt
die Zählung nicht bei `3` fort: Jedes Fenster startet bei `0`.

Für diese drei Positionen liefert die Positions-Tabelle drei Vektoren mit jeweils
vier Zahlen. Nennen wir sie `P0`, `P1` und `P2`. Jedes Fenster braucht dieselben
Positions-Vektoren: Position `0` bekommt immer `P0`, unabhängig davon, welches
Token dort steht.

Die Token-Vektoren liegen bereits für beide Fenster vor, also in der Form
`[2, 3, 4]`. Die Positions-Vektoren brauchen wir dagegen nur als einen gemeinsamen
Satz für ein Fenster. Wir stellen sie als Tensor der Form `[1, 3, 4]` bereit:

```text
Token-Vektoren [2, 3, 4]       Positions-Vektoren [1, 3, 4]

Fenster 0: T00 T01 T02         Gemeinsamer Satz: P0 P1 P2
Fenster 1: T10 T11 T12
```

Jedes `T` und jedes `P` steht hier für einen Vektor aus vier Zahlen.
Die führende `1` bedeutet: Es gibt nur einen Satz von Positions-Vektoren,
nicht nur eine Tokenposition.

Bei der Addition verwendet Burn diesen einen Satz automatisch für beide Fenster.
Dieses Verhalten heisst **Broadcasting**. Gedanklich wird der Positions-Tensor
entlang der Batch-Achse wiederholt:

```text
Fenster 0: T00 + P0   T01 + P1   T02 + P2
Fenster 1: T10 + P0   T11 + P1   T12 + P2
```

Jede dieser Additionen erfolgt Zahl für Zahl innerhalb der beiden Vektoren.
Die Positions-Vektoren werden weder an die Token-Vektoren angehängt noch zwischen
den Fenstern weitergezählt. Eine manuelle Kopie für jedes Fenster ist nicht nötig.

**Warum passen die Formen zusammen?** Für die elementweise Addition müssen die
Grössen jeder Achse gleich sein, oder eine davon muss `1` sein. Eine Achse mit
Grösse `1` kann für die Operation auf die andere Grösse erweitert werden:

```text
Token:       [2, 3, 4]
Position:    [1, 3, 4]
             ↑
             1 wird für die Addition wie 2 verwendet
Ergebnis:    [2, 3, 4]
```

Die anderen Achsen stimmen bereits überein: drei Positionen, vier Zahlen je
Position. Eine Positionsform `[1, 2, 4]` wäre hier nicht passend, weil zwei
Positionen nicht mit drei Positionen übereinstimmen und keine dieser Grössen `1` ist.

Die gemeinsame Embedding-Ausgabe bleibt `[2, 3, 4]`. Das Zusammenfassen zu einem
Batch macht die Fenster nicht zu einem einzigen langen Text.

### Broadcasting mit konkreten Zahlen

Broadcasting bedeutet **Werte für eine Rechenoperation wiederverwenden**.
Es erzeugt keine neuen gelernten Werte. Die Wiederholung ist ein gedankliches
Modell; wir müssen den kleineren Tensor nicht selbst mit Kopien auffüllen.

Beginnen wir mit nur einer Achse:

```text
Form [4]:       [10, 20, 30, 40]
Form [1]:       [ 2]

Gedanklich:     [10, 20, 30, 40]
              + [ 2,  2,  2,  2]
Ergebnis [4]:   [12, 22, 32, 42]
```

Die Form `[1]` bedeutet hier: ein einziger Zahlenwert. Er wird für jede der vier
Additionen verwendet. Es wird weder eine neue Zahl angehängt noch die `2` auf
vier Einträge aufgeteilt.

### Welche Werte werden wiederverwendet?

**Entscheidend ist, an welcher Achse die Grösse `1` steht.** Im Embedding-Beispiel
stehen die Achsen für Fenster, Tokenposition und Vektoreintrag.

Bei `[1, 3, 4]` gibt es einen gemeinsamen Satz von drei vollständigen
Positions-Vektoren. Jeder enthält vier Zahlen. Nehmen wir an, der Vektor für
Position `0` lautet `[1, 2, 3, 4]`:

```text
Fenster 0, Position 0: [10, 20, 30, 40] + [1, 2, 3, 4]
                    = [11, 22, 33, 44]

Fenster 1, Position 0: [50, 60, 70, 80] + [1, 2, 3, 4]
                    = [51, 62, 73, 84]
```

Nur die Fenster-Achse wird erweitert. Der vollständige Positions-Vektor wird
für beide Fenster verwendet; seine vier Einträge bleiben verschieden.
An Position `1` wird entsprechend der Vektor für Position `1` verwendet.

Bei `[1, 3, 1]` steht dagegen **pro Position nur eine Zahl** bereit:

```text
[
  [[2], [5], [9]]
]
```

Hier muss Broadcasting zwei Achsen erweitern: den einen Satz auf beide Fenster
und die einzelne Zahl jeder Position auf alle vier Vektoreinträge:

```text
Position 0: [2] wird wie [2, 2, 2, 2] verwendet
Position 1: [5] wird wie [5, 5, 5, 5] verwendet
Position 2: [9] wird wie [9, 9, 9, 9] verwendet
```

Diese Zuordnung gilt für beide Fenster. Auch `[2, 3, 4] + [1, 3, 1]` ist deshalb
eine gültige Addition mit Ergebnisform `[2, 3, 4]`. **Das ist nur ein Beispiel für
die Rechenregel, nicht unser Positions-Embedding:** Unser Positions-Embedding
liefert vier einzeln lernbare Zahlen pro Position, also `[1, 3, 4]`.

### Formen Achse für Achse prüfen

Für unsere Tensoren mit jeweils drei Achsen prüfen wir die zusammengehörigen
Achsengrössen einzeln. Es genügt nicht, dass beide Tensoren drei Achsen haben:

| Addition | Fenster-Achse | Positions-Achse | Vektoreintrag-Achse | Ergebnis |
| -------- | ------------- | --------------- | ------------------ | -------- |
| `[2, 3, 4] + [1, 3, 4]` | `2`/`1`: wiederverwenden | `3`/`3`: gleich | `4`/`4`: gleich | `[2, 3, 4]` |
| `[2, 3, 4] + [1, 3, 1]` | `2`/`1`: wiederverwenden | `3`/`3`: gleich | `4`/`1`: wiederverwenden | `[2, 3, 4]` |
| `[2, 3, 4] + [1, 2, 4]` | `2`/`1`: wiederverwenden | `3`/`2`: unvereinbar | `4`/`4`: gleich | Nicht möglich |

Warum ist `3` gegenüber `2` nicht erlaubt? Für drei Positionen stehen nur zwei
Einträge bereit. Es gibt keinen einzelnen gemeinsamen Eintrag, der eindeutig
für alle Positionen gilt. Broadcasting wiederholt nicht beliebige Listen
zyklisch und füllt auch keine fehlenden Werte mit Nullen auf.

**Merksatz: Gleiche Achsengrössen passen direkt. Grösse `1` erlaubt
Wiederverwendung entlang dieser Achse. Zwei unterschiedliche Grössen grösser
als `1` passen nicht.**

## Tensorformen in Rust

Ein Tensor enthält Zahlen mit einer bestimmten Anordnung. Zum Verstehen helfen
verschachtelte Rust-Arrays: Wir betrachten zuerst eine Liste, dann eine Tabelle
und schliesslich mehrere Tabellen zusammen. Ein Burn-Tensor ist kein Rust-Array,
aber beim Erstellen können wir diese Arrays als Daten übergeben.

### Eine Liste: eine Achse
- Das Device des Tensors legt fest, wie und wo gerechnet wird; es ist kein Typparameter.
- `2` zählt die **Achsen** des Tensors: eine für die Fenster im Batch und eine
  für die Tokenpositionen innerhalb jedes Fensters. Es sagt nicht, wie gross
  diese Achsen sind.
- `Int` steht für Ganzzahlen, also unsere Token-IDs.

Beginnen wir mit einem Fenster aus drei Token-IDs:
Eine Achse ist eine Richtung, entlang der wir Einträge auswählen können. Bei
unserer Tabelle wählen wir erst das Fenster (Zeile), dann die Tokenposition
(Spalte). Zwei Angaben führen zu einer einzelnen Token-ID:

```text
                   Position 0   Position 1   Position 2
Fenster 0               0            1            3
Fenster 1               0            2            4

Auswahl: Fenster 1, Position 2 -> Token-ID 4
```

Dieser Tensor hat **zwei Achsen**, aber die **Form `[2, 3]`**: Die Fenster-Achse
hat Grösse `2`, die Positions-Achse Grösse `3`. Die `2` in `Tensor<2, Int>`
und die erste `2` in `[2, 3]` sind hier nur zufällig gleich.

Mit acht Fenstern und je sechzehn Positionen wäre die Form `[8, 16]`, der Typ
aber weiterhin `Tensor<2, Int>`. Es bleiben zwei Achsen. Die konkreten
Achsengrössen stehen im Tensor, nicht in dieser Typangabe.

```rust
let token_ids = [0, 1, 3];
```

Um eine Zahl auszuwählen, brauchen wir einen Index, etwa `token_ids[1]`.
Dieser Zugriff liefert die Token-ID `1`. Der Index bezeichnet die Position
in der Liste, nicht die Token-ID. Beispielsweise liefert `token_ids[2]` die
Token-ID `3`.

Diese Anordnung hat **eine Achse** mit **Grösse drei**. Ihre Form schreiben wir
als `[3]`. Die gültigen Indizes sind `0`, `1` und `2`: Indizes beginnen bei
null, die Grösse zählt dagegen alle Einträge.

### Eine Tabelle: zwei Achsen

Fassen wir zwei Fenster zu einem Batch zusammen:

```rust
let batch = [
    [0, 1, 3],
    [0, 2, 4],
];
```

Betrachten wir das zunächst als gewöhnliche Tabelle:

```text
           erste    zweite    dritte Spalte
erste Zeile    0         1         3
zweite Zeile   0         2         4
```

Jede Zeile ist ein Fenster. In jeder Zeile stehen drei Token-IDs.
Es gibt **zwei Zeilen mit je drei Einträgen**. Genau das beschreibt
die Form **`[2, 3]`**: zuerst die Anzahl der Zeilen, dann die Anzahl
der Einträge pro Zeile.

Um eine einzelne Zahl zu finden, wählen wir erst eine Zeile und darin
einen Eintrag. In Rust können wir diese beiden Schritte getrennt schreiben:

```rust
let fenster = batch[1];     // zweite Zeile: [0, 2, 4]
let token_id = fenster[2];  // dritter Eintrag: 4
```

Rust zählt ab null: `[0]` wählt den ersten Eintrag, `[1]` den zweiten
und `[2]` den dritten. Die Zahl in diesen Zugriffsklammern heisst *Index*.
Beide Schritte lassen sich auch als `batch[1][2]` schreiben.

Die Tabelle hat zwei Auswahlrichtungen: Zeile auswählen und Eintrag
innerhalb dieser Zeile auswählen. Diese nennen wir die beiden *Achsen*.
Fürs Erste genügt: **zwei Achsen, Form `[2, 3]`**.

### Vektoren pro Position: drei Achsen

Das Embedding ersetzt jede Token-ID durch vier Gleitkommazahlen.
Vorher stand an jeder Position eine einzelne ID. Jetzt steht dort eine
ganze Liste mit vier Zahlen, der Embedding-Vektor. Die zwei Fenster und
ihre jeweils drei Positionen bleiben erhalten:

```text
2 Fenster
  je 3 Tokenpositionen
    je 4 Zahlen im Embedding-Vektor
```

Als verschachteltes Array sieht diese Anordnung so aus:

```rust
let embeddings = [
    [
        [0.1,  0.0,  0.2, -0.1],
        [0.2, -0.1,  0.4,  0.3],
        [0.0,  0.2, -0.3,  0.4],
    ],
    [
        [ 0.1,  0.0, 0.2, -0.1],
        [-0.2,  0.3, 0.1,  0.0],
        [ 0.1, -0.2, 0.0,  0.2],
    ],
];
```

Dieses Rust-Array zeigt zunächst nur die Anordnung der Zahlen.
Zum Burn-Tensor wird es durch `from_floats`:

```rust
use burn::tensor::{Device, Tensor};

let device = Device::flex();
let tensor = Tensor::<3>::from_floats(embeddings, &device);

println!("{:?}", tensor.dims()); // [2, 3, 4]
```

Der Tensor enthält dieselben 24 Zahlen wie das Array, geordnet als
**zwei Fenster mit je drei Vektoren aus je vier Zahlen**.
Er ist kein einzelner Vektor und auch keine gewöhnliche Tabelle mehr:
Für eine einzelne Zahl müssen wir drei Entscheidungen treffen.

Am Array können wir diese Entscheidungen einzeln nachvollziehen:

```rust
let fenster = embeddings[1]; // zweites Fenster: drei Vektoren
let vektor = fenster[2];     // dritte Position: [0.1, -0.2, 0.0, 0.2]
let zahl = vektor[0];        // erste Zahl dieses Vektors: 0.1
```

Diese drei Auswahlebenen sind die drei Achsen des Tensors.
`Tensor::<3>` sagt Burn, dass es drei Achsen gibt.
`dims()` liefert deren konkrete Grössen: `[2, 3, 4]`.
Die Array-Zugriffe zeigen nur das Prinzip; sie sind hier keine Anleitung
zum Auslesen eines Burn-Tensors.

Beim echten Embedding müssen wir das Array nicht von Hand bauen.
Das Embedding-Modul erzeugt diesen dreidimensionalen Tensor aus dem
zweidimensionalen Eingabetensor mit den Token-IDs:

```text
Eingabetensor                 Embedding-Ausgabetensor
Form [2, 3]                  Form [2, 3, 4]
2 Fenster mit je 3 IDs   ->   2 Fenster mit je 3 Vektoren aus je 4 Zahlen
Tensor<2, Int>               Tensor<3>
```

Die Bedeutung der Achsen ergibt sich aus unseren Daten, nicht aus Burn.
Hier stehen die vier Zahlen für Embedding-Komponenten.

### Auch ein einzelnes Fenster bleibt ein Batch

Unser Modell erwartet Eingabe-IDs als `[Batch, Position]`. Bei nur einem Fenster
bleibt die äussere Ebene erhalten:

```rust
let ohne_batch = [0, 1, 3];   // Form [3]
let mit_batch = [[0, 1, 3]]; // Form [1, 3]
```

Beide enthalten drei IDs, aber ihre Formen sind verschieden.
Die zusätzliche Klammer erzeugt die Batch-Achse mit Grösse `1`.
Nach dem Embedding hat dieser einzelne Batch die Form `[1, 3, 4]`:
ein Fenster, drei Positionen, vier Zahlen pro Position.

**Eine Achse mit Grösse `1` zählt weiterhin als Achse.**
Die Form `[1, 3, 4]` hat genauso drei Achsen wie `[2, 3, 4]`.

### Rust-Typ und konkrete Form unterscheiden

Burn hält die **Anzahl der Achsen** im Rust-Typ fest. Diese Anzahl heisst auch
*Rang*. Die **Grösse jeder Achse** steht dagegen im Tensor und lässt sich mit
`dims()` auslesen.

| Burn-Typ | Beispiel einer Form | Bedeutung hier |
| -------- | ------------------- | -------------- |
| `Tensor<1, Int>` | `[3]` | ein Fenster ohne Batch-Achse |
| `Tensor<2, Int>` | `[2, 3]` | zwei Fenster mit je drei Token-IDs |
| `Tensor<3>` | `[2, 3, 4]` | Embedding-Vektoren für diesen Batch |

In `Tensor<2, Int>` bedeutet `2` also **zwei Achsen**, nicht zwei Zahlen oder
zwei Tokens. `Int` steht für Ganzzahlen, passend zu Token-IDs.
Ohne zweite Typangabe verwendet Burn Gleitkommazahlen, etwa bei `Tensor<3>`
für Embeddings. Das Device legt fest, wie und wo gerechnet wird;
es ist kein Typparameter.

Sowohl `[2, 3]` als auch `[8, 16]` passen zum Typ `Tensor<2, Int>`.
Beide Formen haben zwei Achsen, aber verschieden grosse Batches und Fenster.
Der Typ allein verrät auch nicht, wofür die Achsen stehen.

Ein kleines, eigenständig ausführbares Beispiel:

```rust
use burn::tensor::{Device, Int, Tensor};

fn main() {
    let device = Device::flex();
    let inputs = Tensor::<2, Int>::from_ints(
        [[0, 1, 3], [0, 2, 4]],
        &device,
    );

    println!("{:?}", inputs.dims()); // [2, 3]
}
```

`from_ints` übernimmt die Werte und ihre Form aus dem verschachtelten Array.
`::<2, Int>` gibt beim Erstellen Rang und Zahlenart an. `dims()` liefert
hier ein Rust-Array mit zwei Grössen, nicht die sechs gespeicherten Token-IDs.

Nach dem Embedding brauchen wir drei Angaben, um eine einzelne Zahl auszuwählen:
Fenster, Tokenposition und Eintrag im Embedding-Vektor. Deshalb hat die Ausgabe
drei Achsen und den Typ `Tensor<3>`. Die `3` bedeutet nicht drei Positionen
oder drei Zahlen pro Vektor.

**Merksatz: Der Rust-Typ zählt die Achsen; die Tensorform nennt deren Grössen.**

### Kleine Verständnisaufgaben

Versuche die Antworten zuerst ohne Ausführen herzuleiten:

1. Welche Form hat `[[0, 1, 3]]`? Wie viele Achsen sind das?
2. Ein Embedding-Tensor hat die Form `[2, 3, 4]`. Wie viele Zahlen enthält er
   insgesamt, und welche Grösse hat Achse `1`?
3. Ein Batch enthält ein Fenster mit fünf Tokens. Pro Token liefert das
   Embedding acht Zahlen. Welche Form und welchen Burn-Typ hat die Ausgabe?
4. Ändere im ausführbaren Beispiel nur das Datenarray zu `[[0, 1, 3]]`.
   Sage die Ausgabe von `dims()` voraus. Muss sich `Tensor::<2, Int>` ändern?

## Das Embedding-Modul in Rust

Burn liefert Tensoren, Rechenoperationen und Modellschichten für neuronale Netze
in Rust. Eine kurze Einführung steht im Exkurs [Burn: Die Bausteine](../exkurse/burn-grundlagen.md).

Modellcode liegt in der eigenen Crate `crates/model`. Der DataLoader bleibt in
`crates/training`; das Modell hängt nicht vom Trainingsbereich ab. Wir verwenden
Burn 0.22 mit backendfreien Tensor- und Modelltypen.
Für Demo und Tests genügt `Device::flex()`, ohne GPU.

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

**Warum brauchen wir diesen Schritt?** Die folgenden Modellschichten brauchen
für jede Tokenposition einen Zahlenvektor, nicht nur eine Token-ID. Das
Token-Embedding liefert einen lernbaren Vektor für das jeweilige Token. Weil
dieser allein nichts über seine Stelle im Fenster verrät, ergänzen wir den
Positions-Vektor. Die Addition verbindet beide Informationen, ohne die
Vektorlänge zu ändern.

*Vorwärtsberechnung* bedeutet: Aus einer Eingabe mit den aktuellen Gewichten
eine Ausgabe berechnen. Hier ist das noch keine Vorhersage des nächsten Tokens,
sondern die Vektordarstellung als Eingabe für die weiteren Modellschichten.
Die Gewichte werden dabei nicht verändert.

`forward` nimmt einen Batch aus Token-IDs entgegen und liefert die Summe aus
Token- und Positions-Vektoren:

```rust,ignore
{{#include ../../../crates/model/src/embedding.rs:forward}}
```

Schritt für Schritt:

1. `dims()` liest Batchgrösse und tatsächliche Fensterlänge aus der Eingabe sowie
   die maximale Fensterlänge aus der Positions-Tabelle. So wissen wir, wie viele
   Positionen benötigt werden und ob sie in die Tabelle passen.
2. Die Prüfungen weisen leere Batches und zu lange Fenster zurück: Wir benötigen
   mindestens eine Tokenposition und für jede Position einen vorhandenen Tabelleneintrag.
3. `arange` erzeugt Positions-IDs von `0` bis zur Fensterlänge, ohne Endwert.
   Bei Länge `3` entsteht `[0, 1, 2]`. Diese IDs wählen die Positions-Vektoren aus.
4. `reshape` macht daraus die Form `[1, context_length]`, weil Burns
   `Embedding::forward` einen Tensor mit zwei Achsen erwartet. Ein gemeinsamer Satz
   Positions-IDs genügt, weil jedes Fenster wieder bei Position `0` beginnt.
5. `self.token.forward` schlägt für jede Token-ID einen Vektor nach und liefert
   `[batch_size, context_length, embedding_dim]`. `self.position.forward` tut
   dasselbe für die Positions-IDs und liefert `[1, context_length, embedding_dim]`.
6. `+` addiert die Vektoren Zahl für Zahl. Broadcasting verwendet die gemeinsamen
   Positions-Vektoren für jedes Fenster. Das Ergebnis enthält Token- und
   Positionsinformation und behält die Form
   `[batch_size, context_length, embedding_dim]`.

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

Ein Test verwendet `Device::flex().autodiff()`. Er prüft, dass beide Tabellen die
richtige Form haben und für beide Gewichte Gradienten entstehen. Das bestätigt
ihre Lernbarkeit, ohne bereits eine Trainingsschleife einzuführen.
