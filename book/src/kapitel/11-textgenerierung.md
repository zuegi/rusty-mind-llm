# 11. Textgenerierung

## Lernziel

In Kapitel 9 haben wir ein GPT-Modell trainiert, damit es zu einer Folge von
Token-IDs das nächste Token vorhersagt. In Kapitel 10 haben wir die gelernten
Gewichte gespeichert und wieder geladen. Jetzt verwenden wir das Modell:
Aus einer vorhandenen Tokenfolge erzeugen wir Schritt für Schritt weitere
Token.

Das Modell erzeugt dabei nicht direkt Wörter. Es gibt für jede Position eine
Liste von Zahlen zurück, die **Logits**. Jede Zahl gehört zu einer möglichen
Token-ID. Die grösste Zahl wählen wir zunächst mit `argmax` aus.

## Vorhersagen statt ganzer Text

Ein Aufruf von `forward` verarbeitet einen Kontext und liefert für jede
Position eine Vorhersage:

```text
Token-IDs:              [Der, Hund, rennt]
                               │
                               ▼
GPT:                    Logits für jede Position
                               │
                               ▼
letzte Position:        Logits für das nächste Token
                               │
                               ▼
argmax:                 wahrscheinlichste Token-ID
```

Das Modell erwartet Token-IDs nicht als einfache Liste, sondern als eine
Sammlung von Fenstern. Diese äussere Ebene ist die **Batch-Achse**. Auch wenn
wir nur ein einziges Fenster generieren, legen wir es deshalb in eine Sammlung
mit genau einem Fenster. Für einen Kontext mit drei Token und einem Vokabular
mit fünf Einträgen sieht der Ablauf so aus:

```rust
let context = [0, 1, 3];              // ein Fenster: drei Token
let context_tensor = Tensor::<1, Int>::from_ints(context, &device);
let ids = context_tensor.reshape([1, context.len()]); // ein Batch mit einem Fenster
let logits = model.forward(ids);
```

Hier beschreiben `[0, 1, 3]` und `[1, 3]` zwei verschiedene Dinge:

```text
[0, 1, 3]       die drei gespeicherten Werte
                → eine Achse mit drei Tokenpositionen

[[0, 1, 3]]     dieselben Werte, zusätzlich in ein Fenster eingeordnet
                → zwei Achsen: ein Fenster mit drei Tokenpositionen
```

`context` ist zunächst ein gewöhnliches Rust-Array mit der Form `[3]`. Es
enthält drei Werte, aber noch keine Batch-Achse. `from_ints` übernimmt diese
Werte in einen Tensor mit einer Achse. `reshape([1, context.len()])` fügt
keine Token hinzu und verändert auch keine Werte. Es ordnet die drei Werte nur
als ein Fenster in einem Batch an. Der Eingabetensor hat dadurch die Form
`[1, 3]`: **ein Fenster** mit drei Positionen. Die erste `1` steht für ein
Fenster im Batch, die `3` für die Tokenpositionen. `forward` liefert Logits mit
der Form `[1, 3, 5]` zurück:
Batch, Position und mögliche Token-ID. An jeder der drei Positionen stehen
also fünf Zahlen, eine pro Vokabular-Eintrag.

Für das nächste Token schneiden wir die letzte Position heraus. Bei einer
Position mit Index `2` bleiben die Formen `[1, 1, 5]`:

```rust
let [batch_size, position_count, vocab_size] = logits.dims();
let last_logits = logits.slice([
    0..batch_size,
    (position_count - 1)..position_count,
    0..vocab_size,
]);
let next_token = last_logits.argmax(2);
let next_id = next_token.into_scalar::<i32>();
```

`argmax(2)` vergleicht die fünf Logits entlang der Vokabular-Achse. Der
höchste Wert bestimmt die nächste Token-ID. Die Batch- und Positions-Achse
bleiben zunächst erhalten; `into_scalar` liest den einzelnen Wert daraus aus.
Diese ID hängen wir anschliessend an die bisherige Tokenfolge an.

Für die Generierung interessiert uns nur die letzte Position. Die früheren
Positionen helfen dem Modell beim Berechnen des Kontexts, enthalten aber keine
neue Vorhersage, die wir jetzt anhängen würden.

## Ein Token nach dem anderen

Textgenerierung ist eine Schleife:

1. aktuellen Kontext auswählen,
2. Token-IDs in einen Tensor mit Batch-Achse umwandeln,
3. Modell ausführen,
4. Logits der letzten Position auswählen,
5. mit `argmax` nächste Token-ID bestimmen,
6. Token-ID an die Folge anhängen.

```text
[Der, Hund, rennt]
        ↓
[Der, Hund, rennt, .]
        ↓
[Der, Hund, rennt, ., .]
        ↓
...
```

Jeder neue Schritt sieht auch die Token, die in vorherigen Schritten erzeugt
wurden. Deshalb wird die Eingabe nach jedem Schritt länger.

## Kontextlänge begrenzen

Das Modell wurde mit einer maximalen Kontextlänge gebaut. Nach einigen
Generierungsschritten kann die vollständige Tokenfolge länger sein als dieser
Wert. Dann verwenden wir nur den hinteren Teil:

```rust
let start = token_ids.len().saturating_sub(context_length);
let context = &token_ids[start..];
```

Bei einer Kontextlänge von `3` ergibt sich:

```text
gesamte Folge: [0, 1, 3, 4, 4]
verwendeter Kontext:       [3, 4, 4]
```

Das Modell sieht weiterhin höchstens drei Token. Bereits erzeugte Token bleiben
in der vollständigen Folge erhalten, damit wir am Ende den gesamten Text
ausgeben können.

## Erste Demo

Die Datei `crates/training/examples/ch11.rs` verwendet ein kleines Modell,
beschränkt den Kontext und erzeugt fünf neue Token:

```bash
cargo run -p training --example ch11
```

Die Ausgabe zeigt pro Schritt den verwendeten Kontext und die wachsende
Tokenfolge. Am Ende werden die Token-IDs über ein kleines Vokabular in Wörter
umgewandelt.

Das Modell ist in dieser Demo nur zufällig initialisiert. Die Ausgabe ist
deshalb noch kein sinnvoller deutscher Satz. Sinnvolle Vorhersagen entstehen
erst mit einem trainierten Checkpoint. Die Generierungslogik ist davon
unabhängig.

## Warum zunächst `argmax`?

`argmax` ist deterministisch: Für dieselben Gewichte und denselben Kontext
wird immer dieselbe Token-ID gewählt. Dadurch können wir den Ablauf prüfen,
ohne zusätzlich Zufallszahlen oder Sampling-Regeln zu erklären.

Später können wir die Auswahl erweitern:

```text
Logits
  → Temperatur
  → Top-k oder Top-p
  → Zufallsstichprobe
  → nächste Token-ID
```

Diese Verfahren können abwechslungsreicheren Text erzeugen, ändern aber nicht
den grundlegenden Ablauf: Das Modell sagt immer eine nächste Token-Verteilung
für den aktuellen Kontext voraus.

## Zusammenfassung

- `forward` liefert Logits für jede Batch-Position.
- Für den nächsten Schritt verwenden wir nur die letzte Position.
- `argmax` wählt zunächst deterministisch die nächste Token-ID.
- Die neue ID wird an die vollständige Folge angehängt.
- Für den nächsten Modellaufruf wird der Kontext auf die Modellgrösse gekürzt.
- Ein zufällig initialisiertes Modell erzeugt noch keinen sinnvollen Text.
