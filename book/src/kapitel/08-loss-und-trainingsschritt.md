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

Ein Logit ist ein unbeschränkter Zahlenwert, der nicht auf den Bereich von 0 bis 1 beschränkt und keine Wahrscheinlichkeit ist (Wahrscheinlichkeit => 0 - 1). Erst
**Softmax** macht aus einem Logit-Vektor positive Werte, deren Summe `1` ist.
Für jeden Logit `z` berechnen wir den Exponentialwert `exp(z)`. Diesen Wert
teilen wir durch die Summe der Exponentialwerte aller Logits. Das Ergebnis
ist die Wahrscheinlichkeit für die zugehörige Token-ID.

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

## Cross-Entropy: Wie gut passt die Vorhersage zum richtigen Token?

**Cross-Entropy ist keine Wahrscheinlichkeit, sondern eine Fehlerzahl.**
Sie misst, wie wenig Wahrscheinlichkeit das Modell dem richtigen nächsten
Token gibt. Weniger Loss bedeutet: Die Vorhersage passt besser zum Ziel.

### Woher wissen wir, welches Token richtig ist?

Das richtige Token kommt aus dem **Trainingstext**, nicht aus der Vorhersage
des Modells. Nehmen wir zur Vereinfachung an, jedes Wort sei ein Token:

```text
Trainingstext: „Der Hund bellt“
Eingabe:      „Der Hund“
Ziel:         „bellt“
```

„Richtig“ bedeutet hier: Dieses Token folgt tatsächlich im Beispieltext.
Es bedeutet nicht, dass nach „Der Hund“ niemals ein anderes Wort stehen darf.

Das Modell könnte folgende Wahrscheinlichkeiten vorhersagen:

```text
Token:              „bellt“  „rennt“  „schläft“
Wahrscheinlichkeit:   0.5      0.3       0.2
```

Für den Loss dieser Position nehmen wir die Wahrscheinlichkeit von
**„bellt“**, weil „bellt“ das Ziel ist. Die Wahrscheinlichkeiten der anderen
Tokens bestimmen wir nicht als zusätzliche Ziele.

### Aus der Ziel-Wahrscheinlichkeit wird eine Fehlerzahl

Wir nennen die Wahrscheinlichkeit des richtigen Tokens `p`. Cross-Entropy
berechnet daraus:

```text
Loss einer Position = -ln(p)
```

`ln` ist der natürliche Logarithmus. Für Werte zwischen `0` und `1` ist er
negativ; das Minuszeichen davor macht daraus eine positive Fehlerzahl.
Bei `p = 1` ist der Loss `0`.

Für unser Ziel „bellt“ bedeutet das:

```text
p = 0.1 → Loss ≈ 2.302585   wenig Wahrscheinlichkeit für das Ziel
p = 0.5 → Loss ≈ 0.693147   Vorhersage aus dem Beispiel
p = 0.8 → Loss ≈ 0.223144   mehr Wahrscheinlichkeit für das Ziel
p = 1.0 → Loss = 0          gesamte Wahrscheinlichkeit für das Ziel
```

**Das Training soll den Loss senken und dadurch die Wahrscheinlichkeit des
vorgegebenen nächsten Tokens erhöhen.** Nähert sich `p` der `0`, wächst der
Loss ohne obere Grenze: Das Modell hält das tatsächliche nächste Token dann
für sehr unwahrscheinlich.

### Warum nicht nur das wahrscheinlichste Token prüfen?

In unserem Beispiel steht „bellt“ mit `0.5` bereits auf Platz 1. Trotzdem
kann das Modell ihm noch mehr Wahrscheinlichkeit geben. Auch eine Vorhersage
mit `0.8` für „bellt“ wählt dasselbe Token, bekommt aber einen kleineren Loss.

Nur die Token-ID mit der höchsten Wahrscheinlichkeit auszuwählen, heisst
`argmax`. Damit würden beide Vorhersagen gleich aussehen. Cross-Entropy
berücksichtigt dagegen, **wie viel Wahrscheinlichkeit das Ziel bekommt**.
Für das Training behalten wir deshalb die ganzen Logits, statt zuerst mit
`argmax` eine einzelne Token-ID auszuwählen.

### Ein Loss für den ganzen Batch

Bis hier haben wir eine Position betrachtet. Ein Batch enthält mehrere
Fenster mit mehreren Positionen. Jede Position hat ihr eigenes nächstes
Token als Ziel und erhält einen eigenen Loss.

Wir bilden den **Mittelwert** dieser Loss-Werte. Bei zwei Positionen mit
Ziel-Wahrscheinlichkeiten `0.5` und `0.2`:

```text
Position 1: -ln(0.5) ≈ 0.693147
Position 2: -ln(0.2) ≈ 1.609438

Batch-Loss = (0.693147 + 1.609438) / 2 ≈ 1.151293
```

Wir mitteln also die **Loss-Werte**, nicht zuerst die Wahrscheinlichkeiten.
Der Mittelwert verhindert, dass der Loss allein deshalb grösser wird, weil
wir mehr Positionen mit denselben Fehlerwerten auswerten. Hier zählen alle
Positionen gleich; wir verwenden weder Padding noch zusätzliche
Klassengewichte.

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

Im vorherigen Abschnitt haben wir den Loss aus einer Wahrscheinlichkeit
berechnet. Im Code übergeben wir aber **Logits** an `cross_entropy`. Das ist
kein Widerspruch: Burn übernimmt die Umrechnung innerhalb der Verlustfunktion.
„Roh“ bedeutet hier nur: Auf die Modellausgabe wurde noch kein Softmax angewendet.

### Die Verlustfunktion übernimmt die ganze Rechnung

Zum Verstehen haben wir die Berechnung in einzelne Schritte zerlegt:

```text
Logits → Softmax → Wahrscheinlichkeit des Ziels → -ln → Loss
```

Burn berechnet dasselbe Ergebnis, verbindet aber Softmax und Logarithmus
zu einer Operation namens **`log_softmax`**:

```text
Logits → log_softmax → logarithmierter Wert des Ziels → Minuszeichen → Loss
```

`log_softmax` liefert für jede Token-ID direkt den Logarithmus ihrer
Wahrscheinlichkeit. Bei unserem Beispiel:

```text
Logits:                 [ln(2),   ln(3),   ln(5)]
Softmax zum Vergleich:  [0.2,     0.3,     0.5]
log_softmax:            [ln(0.2), ln(0.3), ln(0.5)]
                       ≈ [-1.609438, -1.203973, -0.693147]

Ziel-ID: 2
Loss:    -(-0.693147) = 0.693147
```

Burn wählt anhand der Ziel-ID den passenden Wert aus und bildet anschliessend
den mittleren Loss über alle Positionen. Die Wahrscheinlichkeiten müssen
dabei nicht als Zwischenergebnis gespeichert werden.

### Warum diese Verbindung numerisch stabiler ist

Computer rechnen nicht mit beliebig genauen Zahlen. `f32` kann sehr grosse
und sehr kleine Werte nur begrenzt darstellen. Das kann die getrennte
Rechnung `Softmax → ln` an zwei Stellen stören.

**Sehr grosse Logits:** Bei Logits wie `[1000, 1001, 1002]` würden die
Exponentialwerte einer naiven Softmax-Berechnung die Grenzen von `f32`
überschreiten. Wir können aber von allen Logits denselben Wert abziehen,
ohne die Softmax-Wahrscheinlichkeiten zu verändern:

```text
Logits:                       [1000, 1001, 1002]
Grössten Logit abziehen:       [-2,   -1,   0]
Exponentialwerte danach:       [0.135335, 0.367879, 1]
```

Der gemeinsame Faktor kürzt sich beim Teilen durch die Summe heraus.
Deshalb arbeitet eine stabile Berechnung mit diesen kleineren Zahlen.

**Sehr kleine Ziel-Wahrscheinlichkeit:** Selbst ein stabiles Softmax kann
eine extrem kleine Wahrscheinlichkeit auf `0` abrunden. Wenn wir danach
den Logarithmus nehmen, erhalten wir `ln(0)`, also keinen endlichen Wert.

```text
Logits: [0, -1000], Ziel-ID: 1

Getrennte Rechnung:
  Wahrscheinlichkeit des Ziels ≈ exp(-1000)
  In f32 auf 0 abgerundet → -ln(0) = unendlich

Direkte Rechnung mit log_softmax:
  logarithmierter Wert des Ziels ≈ -1000
  Loss ≈ 1000
```

`log_softmax` muss den winzigen Wert `exp(-1000)` nicht erst als
Ziel-Wahrscheinlichkeit darstellen. Es berechnet direkt im logarithmischen
Bereich. Dazu zieht es zunächst den grössten Logit ab und verwendet dann:

```text
verschobener Logit = Logit - grösster Logit
log_softmax eines Eintrags
  = verschobener Logit - ln(Summe der exp-Werte aller verschobenen Logits)
```

Im letzten Beispiel ist die Summe ungefähr `1`, ihr Logarithmus ungefähr
`0`. Für das Ziel ergibt sich deshalb direkt `-1000 - 0 = -1000`.
Das vermeidet hier den unendlichen Loss, obwohl die separat ausgegebene
Wahrscheinlichkeit auf `0` gerundet würde.

### Warum wir nicht vorher selbst Softmax aufrufen

`CrossEntropyLossConfig::new()` erwartet standardmässig **Logits**.
Wenn wir stattdessen bereits berechnete Wahrscheinlichkeiten übergeben,
behandelt Burn diese Zahlen wieder wie Logits. Es rechnet damit effektiv
ein zweites Softmax:

```text
Richtig:
  Logits [ln(2), ln(3), ln(5)]
  → Ziel-Wahrscheinlichkeit für ID 2: 0.5
  → Loss ≈ 0.693147

Falsch bei der Standardkonfiguration:
  vorher selbst Softmax: [0.2, 0.3, 0.5]
  → Burn behandelt [0.2, 0.3, 0.5] als Logits
  → neue Ziel-Wahrscheinlichkeit für ID 2:
      exp(0.5) / (exp(0.2) + exp(0.3) + exp(0.5)) ≈ 0.391
  → Loss ≈ 0.94
```

Das ist ein anderer Loss für dieselbe Modellausgabe.

**Für unser Training gilt deshalb:** Die GPT-Ausgabe unverändert an
`cross_entropy` übergeben. Softmax in der Mini-Demo dient nur dazu,
Wahrscheinlichkeiten anzuzeigen; sein Ergebnis geht nicht in den Loss ein.
Burn berechnet den Loss und seine Gradienten aus den ursprünglichen Logits.

## Die Ziele sind bereits verschoben

**„Verschobene Token-Ziele“ bedeutet: Zu jeder Eingabeposition gehört das
nächste Token im Text als Ziel.** Die Zielfolge beginnt ein Token später
als die Eingabefolge.

Nehmen wir zur Vereinfachung wieder an, jedes Wort sei ein Token:

```text
Text:     „Der Hund bellt laut“
Eingabe:  [Der,  Hund,  bellt]
Ziele:    [Hund, bellt, laut ]
```

Das Modell soll drei Vorhersagen lernen:

- Nach „Der“ soll es „Hund“ vorhersagen.
- Nach „Der Hund“ soll es „bellt“ vorhersagen.
- Nach „Der Hund bellt“ soll es „laut“ vorhersagen.

Eingaben und Ziele sind gleich lang. Das Ziel der letzten Position kommt
aus dem Text direkt nach dem Eingabefenster. Deshalb braucht der DataLoader
für ein Fenster mit drei Eingabetokens mindestens vier Tokens im Text.

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
**`cross_entropy` berechnet hier nur den Loss; sie ruft noch kein
`backward()` auf und verändert keine Gewichte.**

Beim Training mit dem Autodiff-Backend bleibt der Loss-Tensor mit dem
Rechengraphen verbunden. Erst die Funktion `sgd_step` im folgenden Abschnitt
[Der vollständige Trainingsschritt](#der-vollständige-trainingsschritt)
ruft darauf `backward()` auf und verwendet die Gradienten für das Update.
`into_scalar::<f32>()` liest dagegen nur den Zahlenwert für die Ausgabe aus;
dieser Rust-Zahlenwert hat keinen Rechengraphen mehr.

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
während `backward()` seinen Tensor verbraucht. `loss.without_autodiff()` entfernt den
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

`Device::flex().autodiff()` aktiviert die Aufzeichnung des Trainingsgraphen.
Der feste Seed macht Initialisierung und Batch-Reihenfolge im CPU-Beispiel
wiederholbar; die beiden Zufallsgeneratoren werden getrennt gesetzt.

```rust,ignore
{{#include ../../../crates/training/examples/ch08.rs:main}}
```

Nach dem Update messen wir den Loss noch einmal auf **demselben Batch**.
`updated.valid()` liefert ein Modell mit denselben Gewichten auf dem
Flex-Device ohne Autodiff-Aufzeichnung. Auch die neuen Eingabetensoren
erzeugen wir auf `device.without_autodiff()`. Der Name der
Burn-Methode bedeutet hier **nicht**, dass wir einen Validierungsdatensatz
auswerten. Eine Validierung mit getrennten Daten folgt erst später.

```bash
cargo run -p training --example ch08
```

Beispielausgabe mit Burn 0.22 und dem CPU-Backend:

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
Das CPU-Backend ist wie im Modell-Crate nur für Demos und Tests aktiviert.
`training` aktiviert das Burn-Feature `optim`, das auch `autodiff` einschliesst.
Der Optimizer-Typ für ganze Modelle heisst in Burn 0.22 `ModuleOptimizer`.

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

**Nächster Lernschritt:** [Kapitel 9: Trainingsschleife und
Validierung](09-trainingsschleife-und-validierung.md). Mehrere Batches und Epochen durchlaufen, den
Optimizer erhalten und Training von Validierung unterscheiden.
Checkpoints und Textgenerierung bleiben ein späteres Kapitel.
