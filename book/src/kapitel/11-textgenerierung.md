# 11. Textgenerierung

## Lernziel

In Kapitel 9 haben wir ein GPT-Modell trainiert, damit es zu einer Folge von
Token-IDs das nächste Token vorhersagt. In Kapitel 10 haben wir die gelernten
Gewichte gespeichert und wieder geladen. Jetzt verwenden wir das Modell:
Aus einer vorhandenen Tokenfolge erzeugen wir Schritt für Schritt weitere
Token.

Das Modell erzeugt dabei nicht direkt Wörter. Es gibt für jede Position einen
**Logit-Vektor** zurück: eine Liste von Zahlen, eine pro mögliche Token-ID.
Ein einzelner Wert darin heisst **Logit**. Die Bedeutung von Logit und der
Schreibweise `[B, T, V]` ist in [Kapitel 7: GPT – Das Sprachmodell](07-gpt.md)
ausführlich erklärt. Die grösste Zahl wählen wir zunächst mit `argmax` aus.

## Vorhersagen statt ganzer Text

Ein Aufruf von `forward` erzeugt **nicht direkt einen fertigen Satz**. GPT
verarbeitet die vorhandenen Token als Kontext und berechnet für jede Position
eine Vorhersage für das jeweils nächste Token:

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

Für die Generierung brauchen wir nur die Vorhersage nach dem vollständigen
aktuellen Kontext. Das ist die letzte Position. Frühere Positionen sagen
vorher, welche Tokens danach kämen; diese Tokens liegen aber bereits im
Kontext. Die Bedeutung der Positionen ist in [Kapitel 7: GPT – Das
Sprachmodell](07-gpt.md) erklärt.

Unsere Ausgabe hat die Form `[1, 3, 5]`: ein Fenster, drei Positionen und fünf
Logits je Position. Wir schneiden die letzte Position heraus. Rust zählt
Positionen ab null, deshalb hat sie bei drei Positionen den Index `2`. Die
beiden anderen Achsen bleiben unverändert:

```text
vorher: [1, 3, 5]  = 1 Fenster, 3 Positionen, 5 Logits
nachher: [1, 1, 5] = 1 Fenster, 1 ausgewählte Position, 5 Logits
```

Die `1` in der Mitte bedeutet nicht, dass ein Logit fehlt. Sie sagt, dass von
den drei Positionen genau eine übrig bleibt. Die Positionsachse bleibt also
erhalten, statt vollständig entfernt zu werden:

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

## Ein Token nach dem anderen

Textgenerierung ist eine Schleife:

1. aktuellen Kontext auswählen,
2. Token-IDs in einen Tensor mit Batch-Achse umwandeln,
3. Modell ausführen,
4. Logits der letzten Position auswählen,
5. mit `argmax` nächste Token-ID bestimmen,
6. Token-ID an die Folge anhängen.

Im Code wird daraus eine Schleife. `token_ids` enthält die vollständige Folge;
`context_window` wählt daraus den Teil, den das Modell im aktuellen Schritt
sieht:

```rust
let mut token_ids = vec![0, 1, 3];

for _ in 0..new_tokens {
    let context = context_window(&token_ids, context_length);
    let next_id = next_token(&model, context, &device);
    token_ids.push(next_id);
}
```

`next_token` bündelt dabei die vorherigen Schritte: Tensor erstellen, `forward`
ausführen, letzte Position auswählen und mit `argmax` eine Token-ID bestimmen.
Nach `push` enthält `token_ids` die neue ID und wird beim nächsten
Schleifendurchlauf wieder als Grundlage verwendet.

Am Anfang gibt es bereits einen Startkontext. Angenommen, die Tokenfolge
lautet `[0, 1, 3]` und `argmax` wählt als nächste ID `4`:

```text
Schritt 1:
Kontext:       [0, 1, 3]
Vorhersage:    4
Tokenfolge:    [0, 1, 3, 4]
```

Für den nächsten Aufruf wird nicht wieder nur der ursprüngliche Kontext
verwendet. Die neu erzeugte ID `4` gehört jetzt selbst zum Kontext:

```text
Schritt 2:
Kontext:       [0, 1, 3, 4]
Vorhersage:    2
Tokenfolge:    [0, 1, 3, 4, 2]
```

Die Schleife verändert also die Tokenfolge nach jedem Durchlauf. Das Modell
erzeugt nicht mehrere neue Tokens in einem einzigen Aufruf. Jeder neue Token
entsteht aus einem eigenen Modellaufruf und kann dadurch die folgenden
Vorhersagen beeinflussen:

```text
vorhandene Token
    → nächste ID vorhersagen
    → ID anhängen
    → erweiterten Kontext erneut verarbeiten
    → nächste ID vorhersagen
    → ...
```

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
wurden. Deshalb wird die Eingabe nach jedem Schritt länger. Die vollständige
Folge wird für die spätere Ausgabe behalten; welche Token tatsächlich an das
Modell übergeben werden, begrenzt die maximale Kontextlänge.

## Kontextlänge begrenzen

Die vollständige Tokenfolge wächst bei jedem Generierungsschritt. Das Modell
hat aber eine feste maximale Kontextlänge: In unserer Demo sind das drei
Token. Seine Positions-Embedding-Tabelle hat deshalb nur Einträge für die
Positionen `0`, `1` und `2`. Wir dürfen ihm nicht einfach vier oder fünf Token
auf einmal übergeben.

Wir behalten daher die vollständige Folge für die Ausgabe, verwenden für
jeden Modellaufruf aber nur die letzten höchstens drei Token. Die Hilfsfunktion
aus der Demo berechnet diesen Ausschnitt:

```rust
fn context_window(token_ids: &[i32], context_length: usize) -> &[i32] {
    assert!(context_length > 0, "context_length muss größer als 0 sein");
    let start = token_ids.len().saturating_sub(context_length);
    &token_ids[start..]
}
```

### Startposition mit `saturating_sub` berechnen

`token_ids.len()` zählt die vorhandenen Token. Davon ziehen wir die erlaubte
Kontextlänge ab. Bei fünf Token und einer Kontextlänge von drei ergibt sich
`5 - 3 = 2`: Der Ausschnitt beginnt am Index `2`.

Wenn die Folge kürzer als die Kontextlänge ist, behalten wir alle Token.
`saturating_sub` liefert dafür mindestens `0`. Bei zwei Token ergibt
`2usize.saturating_sub(3)` also `0`, statt eine negative Zahl zu erzeugen.
Das ist wichtig, weil Längen und Indizes vom Typ `usize` nicht negativ sein
können.

```text
Anzahl Token    Kontextlänge    Startindex    verwendeter Kontext
2               3               0             alle 2 Token
3               3               0             alle 3 Token
4               3               1             letzte 3 Token
5               3               2             letzte 3 Token
```

### Ausschnitt mit `&token_ids[start..]` auswählen

`start..` bedeutet: vom Index `start` bis zum Ende. Das `&` leiht diesen
Ausschnitt als Slice aus. Die Funktion kopiert oder löscht keine Token.

Bei fünf Token und einer Kontextlänge von `3` sieht das so aus:

```text
Index:          0  1  2  3  4
gesamte Folge: [0, 1, 3, 4, 4]
                     └─────┘
Kontext:            [3, 4, 4]
```

Das Modell sieht hier nur `[3, 4, 4]`; `token_ids` enthält weiterhin
`[0, 1, 3, 4, 4]`. Für den Modellaufruf werden die drei ausgewählten Token
wieder an den lokalen Positionen `0`, `1` und `2` eingeordnet.

Beim nächsten angehängten Token verschiebt sich der Ausschnitt um eine
Position nach rechts. Die vollständige Folge wächst, die Länge des
Modellkontexts bleibt höchstens drei:

```text
gesamte Folge: [0, 1, 3, 4]       → Kontext: [1, 3, 4]
gesamte Folge: [0, 1, 3, 4, 2]    → Kontext: [3, 4, 2]
gesamte Folge: [0, 1, 3, 4, 2, 0] → Kontext: [4, 2, 0]
```

Token ausserhalb dieses Ausschnitts gehen nicht mehr direkt in die nächste
Vorhersage ein. Kontextbegrenzung ist also nicht nur eine Formanpassung:
Das Modell kann ältere Teile der Folge nicht mehr sehen.

## Erste Demo

Die Datei `crates/training/examples/ch11.rs` verwendet ein kleines Modell,
beschränkt den Kontext und erzeugt fünf neue Token:

```bash
cargo run -p training --example ch11
```

Die Ausgabe zeigt pro Schritt den verwendeten Kontext und die wachsende
Tokenfolge. Am Ende werden die Token-IDs über ein kleines Vokabular in Wörter
umgewandelt.

### Token-IDs wieder als Text ausgeben

Das Modell und die Generierungsschleife arbeiten nur mit Zahlen. Um die
Ausgabe lesen zu können, brauchen wir dieselbe Zuordnung zwischen Token-ID
und Text wie bei der Eingabe. In unserer Demo übernimmt das ein kleines Array:

```rust
const VOCABULARY: [&str; 5] = ["Der", "Hund", "Katze", "rennt", "."];
```

Der Array-Index entspricht der Token-ID: `VOCABULARY[0]` liefert `"Der"`,
`VOCABULARY[3]` liefert `"rennt"`. Wir gehen die vollständige Tokenfolge durch
und schlagen jeden Eintrag nach:

```rust
for token_id in &token_ids {
    print!("{} ", VOCABULARY[*token_id as usize]);
}
println!();
```

`&token_ids` leiht die Folge aus, ohne sie zu verbrauchen. Deshalb ist
`token_id` eine Referenz auf eine Zahl. `*token_id` liest diese Zahl;
`as usize` wandelt sie in den Typ um, den Rust für Array-Indizes erwartet.
In dieser Demo sind die IDs gültige Indizes: Die Start-IDs liegen im
Vokabular, und `argmax` liefert einen Index innerhalb der fünf Logits.

```text
Tokenfolge: [0, 1, 3, 4]
Ausgabe:    Der Hund rennt .
```

Das Leerzeichen nach jedem Eintrag stammt aus `print!("{} ", ...)`, nicht
aus dem Modell. Deshalb steht hier auch vor dem Punkt ein Leerzeichen.
Diese einfache Worttabelle genügt für die Demo. Bei einem echten Tokenizer
würden wir dessen Decodierung verwenden, weil Tokens auch Wortteile,
Leerzeichen oder Satzzeichen darstellen können.

Wir geben die gesamte Folge aus, also Startkontext und neu erzeugte Token.
Der begrenzte Kontext beeinflusst nur den Modellaufruf, nicht diese Ausgabe.

### Was zeigt die Demo?

Das Modell ist in dieser Demo nur zufällig initialisiert. Die Ausgabe ist
deshalb noch kein sinnvoller deutscher Satz. Sinnvolle Vorhersagen entstehen
erst mit einem trainierten Checkpoint. Die Generierungslogik ist davon
unabhängig.

## Mit geladenen Gewichten generieren

Jetzt verbinden wir Training, Speichern und Generierung. Die Demo aus
Kapitel 10 nimmt optional einen Dateipfad entgegen und speichert dort nach dem
Training ihre Modellgewichte. Die Demo aus Kapitel 11 lädt diese Datei:

```bash
cargo run -p training --example ch10 -- checkpoints/ch11-model.bpk
cargo run -p training --example ch11 -- checkpoints/ch11-model.bpk
```

Beide Demos verwenden dieselbe Modellkonfiguration: fünf Token-IDs, maximal
drei Kontextpositionen, acht Embedding-Komponenten, zwei Attention-Köpfe und
einen Transformer-Block. Das ist wichtig: Der Checkpoint enthält Gewichte,
nicht die Konfiguration oder die Zuordnung zwischen IDs und Text.

Im Generierungsprogramm ersetzen wir die zufällige Initialisierung durch die
Ladefunktion aus Kapitel 10:

```rust
let model = load_model(&config, path, &device)?;
let token_ids = generate(
    &model,
    vec![0, 1, 3],
    NEW_TOKENS,
    CONTEXT_LENGTH,
    &device,
);
```

`generate` kapselt die bereits erklärte Schleife. Nur die Herkunft der
Gewichte ändert sich; Kontextbegrenzung, letzte Position und `argmax` bleiben
gleich. Die Generierung berechnet weder einen Loss noch Gradienten und führt
keine Optimizer-Updates aus.

Ein angegebener, aber fehlender oder inkompatibler Checkpoint führt zu einem
Fehler. Das Programm ersetzt ihn nicht stillschweigend durch zufällige
Gewichte. Ohne Pfad bleibt die bisherige Zufallsdemo verfügbar und weist
darauf in ihrer Ausgabe hin.

**Grenze dieses Lernbeispiels:** Die Checkpoint-Demo trainiert wie Kapitel 9 auf dem künstlichen
ID-Muster `0, 1, 2, 3, 4, 0, ...`, nicht auf deutschen Sätzen. Die geladenen
Gewichte enthalten deshalb ein gelerntes Zahlenmuster, kein deutsches
Sprachverständnis. Unsere Worttabelle macht diese IDs nur lesbar; sie
verwandelt das Training nicht nachträglich in Sprachtraining. Der
Startkontext `[0, 1, 3]` entspricht zudem nicht genau dem Trainingsmuster.
Echter Text und ein passender Tokenizer folgen in Kapitel 12.

Die Tests prüfen die Auswahl anhand fester Logits, die Anzahl neuer Token,
die Erhaltung des Startkontexts und die Generierung über die Kontextgrenze
hinaus. Ein weiterer Test vergleicht die erzeugten Token vor und nach dem
Speichern und Laden:

```bash
cargo test -p training --example ch11 --test ch10
```

## Zusammenfassung

- `forward` liefert Logits für jede Batch-Position.
- Für den nächsten Schritt verwenden wir nur die letzte Position.
- `argmax` wählt zunächst deterministisch die nächste Token-ID.
- Die neue ID wird an die vollständige Folge angehängt.
- Für den nächsten Modellaufruf wird der Kontext auf die Modellgrösse gekürzt.
- Ein zufällig initialisiertes Modell erzeugt noch keinen sinnvollen Text.
- Geladene Gewichte werden mit derselben Generierungsschleife verwendet.
