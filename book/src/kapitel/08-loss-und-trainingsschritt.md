# 8. Cross-Entropy und der erste Trainingsschritt

## Lernziel

Kapitel 7 liefert für jede Eingabeposition einen Logit pro möglichem nächsten
Token. Doch das Modell weiss noch nicht, welche Vorhersage richtig ist.
Jetzt verbinden wir seine Ausgabe mit den Zielen des DataLoaders:

```text
Eingaben [B, T] → GPT → Logits [B, T, V]
                             + Ziele [B, T]
                             → Cross-Entropy [1]
                             → Burn-Autograd
                             → Gradienten der Modellparameter
                             → genau ein SGD-Update
```

Wir wollen drei Fragen beantworten:

1. Wie wird aus einer Vorhersage eine einzelne Fehlerzahl, der **Loss**?
2. Wie passen die verschobenen Token-Ziele zu den Logits?
3. Wie verändert ein Optimizer die Gewichte anhand der Gradienten?

Dieses Kapitel verarbeitet **einen Batch und verändert die Gewichte einmal**.
Es gibt noch keine Trainingsschleife, keine Validierung, keine Checkpoints und
keine Textgenerierung. Die Beispiele laufen auf der CPU.

## Von Logits zu Wahrscheinlichkeiten

Ein Logit ist ein unbeschränkter Zahlenwert, keine Wahrscheinlichkeit. Erst
**Softmax** macht aus einem Logit-Vektor positive Werte, deren Summe `1` ist.
Die Wahrscheinlichkeit für einen Eintrag ist seine Exponentialfunktion,
geteilt durch die Summe aller Exponentialfunktionen.

Für ein nachrechenbares Beispiel wählen wir drei Logits:

```text
Token-ID:           0       1       2
Logits:          ln(2)   ln(3)   ln(5)
exp(Logits):        2       3       5
Summe:              2 + 3 + 5 = 10
Wahrscheinlichkeit: 0.2     0.3     0.5
```

`ln` ist der natürliche Logarithmus; `exp(ln(x)) = x`. Diese Logits sind
absichtlich so gewählt, dass wir Softmax ohne Taschenrechner nachvollziehen
können. Ein GPT-Modell berechnet seine Logits aus den Gewichten.

Angenommen, das richtige nächste Token hat die ID `2`. Das Modell gibt ihm
Wahrscheinlichkeit `0.5`. Wir brauchen eine Fehlerzahl, die kleiner wird,
wenn genau diese Wahrscheinlichkeit wächst.

## Cross-Entropy: Die Wahrscheinlichkeit des richtigen Tokens

Für eine Position mit genau einer richtigen Token-ID ist Cross-Entropy:

```text
Loss einer Position = -ln(Wahrscheinlichkeit des richtigen Tokens)

p = 1.0 → Loss = 0
p = 0.5 → Loss ≈ 0.693147
p = 0.2 → Loss ≈ 1.609438
```

Eine sichere richtige Vorhersage hat kleinen Loss. Eine sichere falsche
Vorhersage wird stark bestraft: Nähert sich die Wahrscheinlichkeit des
richtigen Tokens `0`, wächst der Loss ohne obere Grenze.

Wir verwenden nicht nur das wahrscheinlichste Token. Beispielsweise wählen
`[0.1, 0.2, 0.7]` und `[0.01, 0.02, 0.97]` beide ID `2`, haben aber
unterschiedlichen Loss. Die ganzen Logits bleiben Teil des Rechengraphen.
Ein `argmax` würde diese Information für das Lernen verlieren.

Für mehrere Positionen bilden wir den **Mittelwert** ihrer Loss-Werte.
Bei zwei richtigen Tokens mit Wahrscheinlichkeiten `0.5` und `0.2`:

```text
mittlerer Loss = (0.693147 + 1.609438) / 2 ≈ 1.151293
```

Damit bleibt die Fehlerzahl bei identischen Vorhersagen vergleichbar, wenn
wir die Anzahl Fenster oder Positionen ändern. Hier zählen alle Positionen
gleich; wir verwenden weder Padding noch zusätzliche Klassengewichte.

### Eine ausführbare Mini-Demo

Die Demo verwendet zwei Positionen mit vertauschten Logits. Erst setzen wir
die jeweils wahrscheinliche Token-ID als Ziel, danach die unwahrscheinliche:

```rust,ignore
{{#include ../../../crates/training/examples/ch08_loss.rs}}
```

```bash
cargo run -p training --example ch08_loss
```

Die flach ausgegebene Wahrscheinlichkeitsliste enthält ungefähr
`[0.2, 0.3, 0.5, 0.5, 0.3, 0.2]`. Die kleinen Rundungsabweichungen entstehen
durch `f32`. Die beiden letzten Ausgabezeilen sind:

```text
Loss für Ziele [2, 0]: 0.693147
Loss für Ziele [0, 2]: 1.609438
```

Die Vorhersagen sind gleich geblieben. Nur die richtigen Token-IDs wurden
geändert. Der Loss hängt also davon ab, **was wir vorhersagen sollten**.

## Warum rohe Logits in die Verlustfunktion gehen

Die Mini-Demo zeigt Softmax nur, um die Wahrscheinlichkeiten sichtbar zu
machen. An `cross_entropy` übergeben wir weiterhin **Logits**, nicht das
Ergebnis von Softmax.

Burn berechnet intern `log_softmax`: den Logarithmus der
Softmax-Wahrscheinlichkeiten auf numerisch stabile Weise. Ein separat
berechnetes Softmax könnte sehr kleine Wahrscheinlichkeiten zu `0` abrunden;
`ln(0)` wäre dann nicht endlich. Auch sehr grosse Logits könnten bei einer
naiven Exponentialfunktion überlaufen.

`CrossEntropyLossConfig::new()` erwartet standardmässig Logits. Softmax
vorher aufzurufen, würde die Wahrscheinlichkeiten erneut wie Logits behandeln
und damit **einen anderen, falschen Loss** berechnen.

## Die Ziele sind bereits verschoben

Wir benutzen `Dataset` und `Batch` aus Kapitel 2. Der kleine Token-Strom der
Trainingsdemo enthält nur IDs aus einem künstlichen Vokabular mit fünf
Einträgen. Das ist **nicht** der GPT-2-Tokenizer:

```text
Token-Strom: [0, 1, 2, 3, 4, 0, 1]
Kontextlänge: 3

Fenster 1:
  Eingabe: [0, 1, 2]
  Ziel:    [1, 2, 3]

Fenster 2:
  Eingabe: [3, 4, 0]
  Ziel:    [4, 0, 1]
```

Das Modell sieht zum Beispiel an Position `0` des ersten Fensters nur ID `0`.
Der Logit-Vektor dieser Position soll ID `1` vorhersagen. An Position `2`
kennt es die IDs `0, 1, 2` und soll ID `3` vorhersagen.

**Nicht noch einmal verschieben:** Der DataLoader hat Eingaben und Ziele
bereits passend angeordnet. Die kausale Attention verhindert weiterhin, dass
eine Position spätere Eingabetokens sieht.

Die Demo nimmt genau einen vollständigen Batch mit zwei Fenstern. `epoch`
ist die bestehende Batch-Erzeugung aus Kapitel 2, keine neue Trainingsschleife:

```rust,ignore
{{#include ../../../crates/training/examples/ch08.rs:batch}}
```

### Von Rust-Vektoren zu Burn-Tensoren

`Batch::to_tensors` erhält die Fensterreihenfolge und erzeugt zwei
Integer-Tensoren der Form `[B, T]`. Token-IDs sind Indizes, keine lernbaren
Gleitkommazahlen.

```rust,ignore
{{#include ../../../crates/training/src/dataset.rs:batch_tensors}}
```

`flatten()` reiht erst alle IDs des ersten Fensters, dann des zweiten
aneinander. `TensorData::new` gibt diesen Daten wieder die zweidimensionale
Form. Die Formprüfung lehnt leere Batches, leere Fenster und ungleiche
Eingabe-/Zielformen ab, statt sie stillschweigend umzudeuten.

Alle IDs müssen in `0..vocab_size` liegen. Die Trainingsdemo wählt deshalb
auch für das Modell `vocab_size = 5`; echte GPT-2-Token-IDs dürfen nicht
ungeprüft in dieses Mini-Modell eingesetzt werden.

## Logits und Ziele in dieselbe Reihenfolge bringen

Burns Cross-Entropy erwartet einen zweidimensionalen Logit-Tensor und eine
eindimensionale Liste richtiger Klassen-IDs. Wir fassen Batch und Position
zu einer Achse zusammen:

```text
GPT-Logits:       [B, T, V] → [B*T, V]
DataLoader-Ziele: [B, T]    → [B*T]

Demo:
Logits: [2, 3, 5] → [6, 5]
Ziele:  [2, 3]    → [6]
Loss:                     [1]
```

`reshape` ändert die Form, nicht die Reihenfolge. Jede Zeile der flachen
Logits gehört weiterhin zur richtigen Ziel-ID. **Die Vokabularachse bleibt
erhalten:** Wir mitteln nicht erst über die Logits und berechnen auch nicht
nur für die letzte Position einen Loss.

Unsere Funktion ist ein kleiner Adapter für Burns Verlustfunktion:

```rust,ignore
{{#include ../../../crates/training/src/loss.rs}}
```

Das Ergebnis ist ein Tensor mit einer Zahl, nicht sofort ein Rust-`f32`.
So bleibt der Rechengraph erhalten, und wir können `backward()` aufrufen.
`into_scalar()` ist erst für die Ausgabe gedacht.

## SGD: Aus Gradienten werden neue Gewichte

Kapitel 4 hat gezeigt, wie Burn automatisch Gradienten berechnet. Jetzt
verwenden wir sie. **SGD** bedeutet *Stochastic Gradient Descent*: Wir
schätzen die Richtung anhand eines Batches statt des ganzen Datensatzes.

Ohne Momentum oder Weight Decay lautet die Update-Regel für jeden Parameter:

```text
neues Gewicht = altes Gewicht - Lernrate * Gradient

altes Gewicht: 0.5
Gradient:     0.2
Lernrate:     0.01
neues Gewicht = 0.5 - 0.01 * 0.2 = 0.498
```

Ein positiver Gradient bedeutet: Eine kleine Erhöhung des Gewichts erhöht
den Loss. Deshalb gehen wir in die Gegenrichtung. Bei negativem Gradient
wird das Gewicht grösser.

Die **Lernrate** bestimmt die Schrittweite. Ein zu grosser Schritt kann über
ein günstiges Gebiet hinausspringen und den Loss erhöhen. Ein positiver
Wert allein garantiert daher noch keine Verbesserung.

### Der vollständige Trainingsschritt

```rust,ignore
{{#include ../../../crates/training/src/step.rs}}
```

Die Funktion führt genau diese Schritte aus:

1. `model.forward(inputs)` berechnet Logits aus den aktuellen Gewichten.
2. `cross_entropy` vergleicht alle Positionen mit den richtigen Token-IDs.
3. `backward()` berechnet Gradienten durch Loss, Ausgabeprojektion,
   Transformer-Blöcke und Embedding. **Burn übernimmt Autograd vollständig.**
4. `GradientsParams::from_grads` ordnet die Gradienten den Modellparametern zu.
5. `optimizer.step` wendet Burns SGD an und liefert das Modell mit neuen
   Gewichten zurück.

Burn markiert Modellparameter automatisch für Autograd. Wir brauchen weder
eine eigene Ableitung noch manuelle Updates einzelner Tabellen.

`loss.clone()` hält den ursprünglichen Loss für die Rückgabe verfügbar,
während `backward()` seinen Tensor verbraucht. `loss.inner()` entfernt den
Autodiff-Wrapper vom Ergebnis: Die Rückgabe enthält den **Loss vor dem
Update**, ohne einen weiteren Rechengraphen dafür aufzubauen.

`sgd_step` nimmt das Modell als Wert entgegen. Der Aufrufer muss mit dem
zurückgegebenen Modell weiterarbeiten; die alten Gewichte werden nicht
heimlich verändert.

Wir verwenden die Standardkonfiguration ohne Momentum, Weight Decay oder
Clipping. Für dieses zustandslose SGD-Update genügt ein neu erzeugter
Optimizer. Ein späterer Optimizer mit Zustand muss über Trainingsschritte
hinweg erhalten bleiben.

## Ein Batch durch das GPT-Modell

Die Modellarchitektur bleibt dieselbe wie in Kapitel 7. Für die Demo genügen
fünf Token-IDs, drei Positionen, vier Embedding-Dimensionen, zwei
Attention-Köpfe und ein Transformer-Block.

```rust,ignore
{{#include ../../../crates/training/examples/ch08.rs:setup}}
```

```rust,ignore
{{#include ../../../crates/training/examples/ch08.rs:model}}
```

`Ad = Autodiff<NdArray<f32>>` zeichnet den Trainingsgraphen auf.
Der feste Seed macht Initialisierung und Batch-Reihenfolge im CPU-Beispiel
wiederholbar; die beiden Zufallsgeneratoren werden getrennt gesetzt.

```rust,ignore
{{#include ../../../crates/training/examples/ch08.rs:main}}
```

Nach dem Update messen wir den Loss noch einmal auf **demselben Batch**.
`updated.valid()` liefert ein Modell mit denselben Gewichten auf dem
inneren Backend `NdArray<f32>`, ohne Autodiff-Aufzeichnung. Der Name der
Burn-Methode bedeutet hier **nicht**, dass wir einen Validierungsdatensatz
auswerten. Eine Validierung mit getrennten Daten folgt erst später.

```bash
cargo run -p training --example ch08
```

Beispielausgabe mit Burn 0.21 und dem CPU-Backend:

```text
{{#include ../ausgaben/ch08.txt}}
```

In diesem Beispiel sinkt der Loss nach einem Update. Das ist noch kein
trainiertes Sprachmodell und kein Beleg für bessere Vorhersagen auf fremden
Texten. Andere Initialisierungen oder Lernraten können andere Werte und auch
einen steigenden Loss ergeben.

## Trennung zwischen Modell und Training

Die Abhängigkeiten zeigen weiterhin nur in eine Richtung:

```text
training → model → Burn
training → Burn (Tensor-Konvertierung, Loss, Optimizer)
```

`model` liefert Logits. Dort gibt es weder Cross-Entropy noch SGD oder
Datensätze. `training` verbindet Daten, Modell, Verlustfunktion und Update.
Die CPU- und Autodiff-Backends sind wie im Modell-Crate nur für Demos und
Tests aktiviert.

## Tests und Grenzen

```bash
cargo test -p training --test ch08
```

Die Tests prüfen die Zielreihenfolge, den Mittelwert über mehrere Fenster,
bekannte Cross-Entropy-Werte und stabile Berechnung bei grossen Logits.
Sie vergleichen die Loss-Gradienten mit der bekannten
Softmax-minus-Ziel-Regel und prüfen für Embedding, Attention-Projektion und
Ausgabeprojektion das tatsächliche SGD-Update. Ungültige Batch-Formen und
Lernraten werden explizit abgelehnt.

**Nächster Lernschritt:** Mehrere Batches und Epochen durchlaufen, den
Optimizer erhalten und Training von Validierung unterscheiden.
Checkpoints und Textgenerierung bleiben ein späteres Kapitel.
