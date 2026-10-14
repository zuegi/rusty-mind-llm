# 9. Trainingsschleife und Validierung

## Lernziel

Kapitel 8 hat die Gewichte des Modells genau einmal verändert. In diesem
Kapitel wiederholen wir diesen Schritt. Danach prüfen wir, ob die
Vorhersagen besser geworden sind.

Wir gehen in drei Etappen vor:

1. Drei konkrete Fenster durch eine Epoche verfolgen.
2. Lernen und reine Loss-Messung unterscheiden.
3. Diese Abläufe im Rust/Burn-Code wiederfinden.

Du musst dafür keine neue Modellarchitektur verstehen. Embedding,
Attention und Ausgabeprojektion bleiben dieselben. Checkpoints und
Textgenerierung folgen später.

## 1. Was aus Kapitel 8 gleich bleibt

Ein **Fenster** enthält eine kurze Token-Folge als Eingabe und die um ein
Token verschobene Folge als Ziel. Das Ziel jeder Position ist das nächste
Token im ursprünglichen Text.

Ein **Batch** bündelt Fenster. Bei Batch-Grösse 1 besteht ein Batch aus
genau einem Fenster; bei Batch-Grösse 4 aus vier Fenstern.

Ein **Trainingsschritt** verarbeitet einen Batch:

```text
1. Modell berechnet Logits aus den Eingaben.
2. Cross-Entropy vergleicht Logits mit den Zielen → Loss.
3. backward() berechnet Gradienten.
4. SGD verändert die Gewichte anhand der Gradienten.
```

Der **Loss** ist die Fehlerzahl aus Kapitel 8. Kleinere Werte bedeuten,
dass das Modell den richtigen nächsten Tokens mehr Wahrscheinlichkeit gibt.
Die **Gewichte** sind die lernbaren Zahlen des Modells. Ein Update verändert
diese Zahlen, nicht die Eingabe- oder Ziel-IDs.

Neu ist zunächst nur: Wir führen diesen Trainingsschritt mehrmals aus.

## 2. Drei Fenster von Hand durchgehen

Wir beginnen mit einem kleineren Beispiel als die spätere Demo. Der
Token-Strom enthält zehn IDs. Jede ID steht hier für ein künstliches Token,
nicht für ein Wort des GPT-2-Tokenizers:

```text
Token-Strom: [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
Kontextlänge: 3

Fenster A: Eingabe [0, 1, 2], Ziel [1, 2, 3]
Fenster B: Eingabe [3, 4, 0], Ziel [4, 0, 1]
Fenster C: Eingabe [1, 2, 3], Ziel [2, 3, 4]
```

Der DataLoader bildet die Eingabefenster ohne Überlappung. Die Ziele
beginnen jeweils ein Token später.

### Eine Epoche mit Batch-Grösse 1

Wir verwenden für dieses Gedankenbeispiel zuerst die Reihenfolge A, B, C.
Ein Fenster pro Batch bedeutet drei Batches und damit drei Updates:

```text
Startgewichte
  → Batch A → Trainingsschritt → Gewichte nach Update 1
  → Batch B → Trainingsschritt → Gewichte nach Update 2
  → Batch C → Trainingsschritt → Gewichte nach Update 3
```

**Batch B verwendet bereits die durch Batch A veränderten Gewichte.**
Batch C verwendet wiederum die Gewichte nach dem zweiten Update.
Wir starten das Modell nicht für jedes Fenster neu.

Ein solcher Durchlauf durch die Trainingsdaten heisst **Epoche**.
Hier gilt: eine Epoche, drei Batches, drei Gewichtsupdates.
Ein Batch kann mehrere Ziele enthalten, führt aber trotzdem nur zu
**einem gemeinsamen Update**.

### Was passiert in der zweiten Epoche?

Wir verwenden dieselben Trainingsfenster erneut. Das Modell behält seine
bisher gelernten Gewichte. Nur die Reihenfolge der Fenster wird gemischt:

```text
Epoche 1: A → B → C → insgesamt 3 Updates
Epoche 2: C → A → B → insgesamt 6 Updates
```

Mischen bedeutet: Fenster umordnen. **Die Token-Reihenfolge innerhalb eines
Fensters und die Zuordnung seiner Ziele bleiben unverändert.**

Die neue Reihenfolge verhindert, dass das Modell die Batches immer in
derselben Abfolge verarbeitet. Sie garantiert keinen sinkenden Loss.

### Was ändert Batch-Grösse 2?

Jetzt bündeln wir jeweils zwei Fenster:

```text
Batch 1: [A, B] → ein Trainingsschritt → ein Update
Rest:    [C]    → nur ein Fenster statt zwei
```

Unsere bestehende Trainingsmethode `Dataset::epoch` verwirft diesen
unvollständigen letzten Batch. In diesem Beispiel hätte eine Epoche
deshalb nur **ein Update**. Nach erneutem Mischen könnte ein anderes
Fenster übrig bleiben.

Diese Regel stammt aus Kapitel 2. „Eine Epoche“ bedeutet in unserer
Implementierung deshalb genauer: alle vollständigen Trainingsbatches
einmal verarbeiten.

## 3. Lernen und Messen sind zwei verschiedene Abläufe

Nach dem Training möchten wir wissen: Wie gut sind die Vorhersagen mit
den **jetzt aktuellen Gewichten**?

Dafür berechnen wir erneut Logits und Loss, aber ändern nichts:

```text
Training:
  Eingaben → Logits → Loss → backward() → SGD-Update

Messung:
  Eingaben → Logits → Loss → Zahlenwert anzeigen
```

**Bei der Messung fehlen `backward()` und SGD-Update.** Die Ziele brauchen
wir weiterhin, um den Loss zu berechnen. Das Modell bekommt aber nur die
Eingaben, nicht die richtigen Antworten als zusätzliche Eingabe.

Warum messen wir nach der Epoche noch einmal? Die Loss-Werte während des
Trainings gehören zu verschiedenen Gewichtsständen: vor Update 1, vor
Update 2 und so weiter. Eine neue Messung verwendet für alle Fenster
denselben aktuellen Gewichtsstand.

## 4. Was bedeutet Validierung?

Ein Modell kann die Beispiele, mit denen es geübt hat, immer besser
vorhersagen. Damit wissen wir noch nicht, ob es auch auf anderen Beispielen
bessere Vorhersagen macht.

Deshalb reservieren wir einen Teil der Daten zum **Prüfen statt Lernen**.
Dieser zurückgehaltene Teil heisst **Validierungsdaten**:

```text
Trainingsdaten:
  zum Lernen verwenden
  danach auch ihren Loss messen

Validierungsdaten:
  nicht zum Lernen verwenden
  nur ihren Loss messen
```

Wir erhalten zwei Fehlerzahlen:

- **Trainings-Loss:** Wie gut passen die aktuellen Vorhersagen zu den
  Trainingszielen?
- **Validierungs-Loss:** Wie gut passen sie zu den zurückgehaltenen Zielen?

Beide Werte entstehen durch dieselbe Loss-Berechnung. Der Unterschied
liegt in den Daten, nicht in einer anderen Verlustfunktion.

Sinkt der Trainings-Loss, während der Validierungs-Loss steigt, kann das
auf **Overfitting** hindeuten. Damit ist gemeint: Das Modell passt sich an
die Trainingsbeispiele an, ohne auf getrennten Daten besser zu werden.
Einzelne Schwankungen reichen für diese Schlussfolgerung noch nicht.

Wir trainieren in diesem Kapitel eine feste Anzahl Epochen. Die
Validierungswerte ändern weder die Lernrate noch die Gewichte.

### Erst den Token-Strom teilen, dann Fenster bilden

Wir teilen den ursprünglichen Token-Strom in zwei getrennte Bereiche.
Aus jedem Bereich erzeugen wir anschliessend eigene Fenster:

```text
Token-Strom → Trainingsbereich   → Trainingsfenster
            → Validierungsbereich → Validierungsfenster
```

Kein Fenster darf über die Grenze reichen. Sonst könnte ein
Trainingsfenster bereits Tokens aus dem Validierungsbereich verwenden.

Beide Bereiche dürfen dieselben Token-IDs enthalten. Beispielsweise darf
ID `2` in beiden vorkommen. Getrennt sein müssen die **ursprünglichen
Positionen im Token-Strom**, nicht die verwendeten Token-Arten.

## 5. Jetzt die Zahlen unserer ausführbaren Demo

Das Beispiel mit drei Fenstern hat den Ablauf gezeigt. Die ausführbare
Demo verwendet mehr Daten, aber dieselben Regeln:

```text
Token-Strom:    101 IDs mit dem Muster 0, 1, 2, 3, 4, 0, 1, ...
Kontextlänge:   3
Batch-Grösse:   4 Fenster
Epochen:        10
Lernrate:       0.05
```

Die **Lernrate** ist die Schrittweite von SGD aus Kapitel 8. Wir lassen
sie während der zehn Epochen unverändert.

### Die Daten teilen

20 Prozent sind für die Validierung vorgesehen. `split_tokens` rundet
die Trainingslänge nach unten ab; der Rest gehört zur Validierung:

```text
Training:     abrunden(101 * 0.8) = 80 Tokens
Validierung:  101 - 80            = 21 Tokens
```

### Aus Tokens werden Fenster

Das letzte Token eines Bereichs hat darin keinen Nachfolger. Es kann
deshalb Ziel, aber keine Eingabe mit passendem Ziel sein.
Bei `N` Tokens haben wir `N - 1` solche Eingabepositionen.

Je drei Positionen ergeben ein vollständiges Eingabefenster:

```text
Training:     (80 - 1) / 3 = 26 Fenster, Rest 1 Eingabeposition
Validierung:  (21 - 1) / 3 =  6 Fenster, Rest 2 Eingabepositionen
```

Die Restpositionen bilden kein vollständiges Fenster und werden nicht
verwendet. Jedes fertige Fenster hat drei Eingabe- und drei Ziel-IDs.

### Aus Trainingsfenstern werden Batches

```text
26 Trainingsfenster / 4 = 6 vollständige Batches, Rest 2 Fenster
6 Updates pro Epoche * 10 Epochen = 60 Updates
```

Die zwei übrigen **fertigen Fenster** verwirft die Trainingsmethode für
diese Epoche. Das ist ein anderer Rest als die zuvor verworfenen
Eingabepositionen.

Bei der **Messung** behalten wir dagegen auch Teil-Batches. So gehen alle
26 Trainingsfenster und alle 6 Validierungsfenster in ihre Loss-Messung ein.
Messen verlangt keine anschliessenden Gewichtsupdates mit voller Batch-Grösse.

Das Muster wiederholt sich absichtlich auch im Validierungsbereich.
Sinkender Validierungs-Loss zeigt hier, dass das Modell dieses einfache
Muster auch dort besser vorhersagt. Das ist kein realistischer Nachweis
für gute Vorhersagen auf unbekannten Sprachtexten.

## 6. Den Ablauf im Rust/Burn-Code wiederfinden

Ab hier übersetzen wir die bekannten Schritte in Code. Die Ausschnitte
stammen aus ausführbaren Dateien; die vollständige Demo steht weiter unten.

### Zwei Datensätze vorbereiten

```rust,ignore
{{#include ../../../crates/training/examples/ch09.rs:datasets}}
```

`tokens` enthält die 101 IDs. `split_tokens` liefert die beiden Bereiche.
`Dataset::new` bildet für jeden Bereich eigene Fenster. Die Rückgabe ist
ein Paar: zuerst Trainingsdatensatz, dann Validierungsdatensatz.

Das `?` gibt einen Fehler weiter, falls ein Bereich zu kurz für ein
Fenster ist. Es ersetzt ihn nicht durch einen leeren Datensatz.

### Ein Trainingsschritt mit erhaltenem Optimizer

Der **Optimizer** ist hier Burns SGD-Objekt: Es wendet die Update-Regel
auf die Modellgewichte an.

Kapitel 8 erzeugt seinen Optimizer für einen einzelnen Schritt.
Jetzt erzeugen wir ihn einmal und verwenden ihn in allen Schritten:

```rust,ignore
{{#include ../../../crates/training/src/step.rs:train_step}}
```

Die Berechnung bleibt dieselbe: Vorhersage, Loss, Gradienten, Update.
Neu ist der Parameter `optimizer: &mut ModuleOptimizer`.

`&mut` bedeutet in Rust: Die Funktion darf das bestehende Objekt über
eine veränderbare Referenz verwenden. Sie muss keinen neuen Optimizer
erzeugen. `ModuleOptimizer` verwaltet Updates und Zustand für ganze Modelle;
`SgdConfig::new().init()` konfiguriert ihn hier für SGD.

Unser SGD ohne Momentum benötigt keine Verlaufshistorie für seine Regel.
Wir erhalten das Objekt trotzdem über alle Schritte hinweg. Andere
Optimizer können frühere Schritte speichern und benötigen genau diese
Lebensdauer. Ihre Verfahren müssen wir für dieses Kapitel nicht verstehen.

`train_step` gibt das aktualisierte Modell zurück. `sgd_step` aus Kapitel 8
ruft dieselbe Funktion auf, mit einem selbst erzeugten SGD-Optimizer.

### Eine Epoche ist die innere Schleife

```rust,ignore
{{#include ../../../crates/training/src/runner.rs:train_epoch}}
```

`batches` ist die bereits vorbereitete Liste dieser Epoche. Die
`for batch in batches`-Schleife verarbeitet jeden Batch einmal.

`batch.to_tensors(device)` wandelt die Eingabe- und Ziel-Listen in
Burn-Tensoren um. `device` bezeichnet den Rechenort, hier die CPU.

Bei `(model, _) = train_step(...)` ersetzt das neue Modell das alte.
Der Unterstrich bedeutet: Den ebenfalls zurückgegebenen Loss dieses
einzelnen Schritts verwenden wir hier nicht.

Die Funktion liefert zuletzt Modell und Update-Anzahl zurück.
Eine leere Batch-Liste wird explizit abgelehnt: Ohne Updates hat
keine Trainingsepoche stattgefunden.

### Messen ohne Autograd-Aufzeichnung

Zum Trainieren verwenden wir `Device::flex().autodiff()`:
Flex rechnet auf der CPU; `.autodiff()` aktiviert die Aufzeichnung,
damit `backward()` daraus Gradienten berechnen kann.

Zur Messung reicht `Device::flex()` ohne diese Aufzeichnung.
`model.valid()` erzeugt in Burn eine Modellversion mit denselben
initialisierten Gewichten ohne Autodiff. Training und Messung haben dabei
denselben Modelltyp; die Autodiff-Einstellung steckt jetzt im Device.
`evaluate` lehnt deshalb sowohl ein Autodiff-Eingabe-Device als auch
Autodiff-Logits explizit ab, statt diese Grenze nur über den Typ abzusichern.

**`valid()` wählt keine Daten aus und führt noch keine Validierung aus.**
Erst die Übergabe des Validierungsdatensatzes an `evaluate` macht
die folgende Loss-Messung zur Validierung.

Für die Messung bildet `Dataset::batches` die Batches:

```rust,ignore
{{#include ../../../crates/training/src/dataset.rs:evaluation_batches}}
```

`chunks` behält den letzten Teil-Batch. `chunks_exact` im
Trainings-DataLoader verwirft ihn dagegen. Die Messung verarbeitet
alle fertig gebildeten Fenster in Originalreihenfolge.

### Warum der gemeinsame Loss gewichtet wird

Cross-Entropy liefert bereits den **mittleren Loss eines Batches**.
Bei unterschiedlich grossen Batches dürfen diese Mittelwerte nicht
gleich gewichtet werden:

```text
Batch 1: 12 Positionen, mittlerer Loss 1
Batch 2:  6 Positionen, mittlerer Loss 3

Falsch:  (1 + 3) / 2 = 2
Richtig: (12 * 1 + 6 * 3) / (12 + 6) ≈ 1.666667
```

Der grössere Batch enthält mehr Vorhersagen und trägt deshalb mehr
zum Gesamtwert bei. `evaluate` sammelt entsprechend:

```rust,ignore
{{#include ../../../crates/training/src/runner.rs:evaluate}}
```

`count` ist die Anzahl Positionen im aktuellen Batch.
`total_loss` sammelt Batch-Loss mal Positionsanzahl.
`positions` zählt alle ausgewerteten Positionen. Am Ende teilen
wir die Summe durch diese Gesamtzahl.

`into_scalar::<f32>()` liest den Loss als Zahl aus; `f64::from(...)`
wandelt ihn für die Summierung in den Rust-Zahlentyp `f64` um.
Ein nicht endlicher Loss wird als Fehler sichtbar.

### Zwei Messungen mit denselben Gewichten

```rust,ignore
{{#include ../../../crates/training/examples/ch09.rs:report}}
```

`fixed` ist die Modellversion ohne Autodiff-Aufzeichnung.
Beide `evaluate`-Aufrufe verwenden sie unverändert: einmal auf
Trainingsdaten, einmal auf Validierungsdaten.

Die Ausgabe nennt auch die Epochenzahl. **Epoche 0** bezeichnet
die Messung vor dem ersten Gewichtsupdate.

### Mehrere Epochen sind die äussere Schleife

```rust,ignore
{{#include ../../../crates/training/examples/ch09.rs:loop}}
```

Die Reihenfolge ist:

1. Daten, Modell und Optimizer einmal vorbereiten.
2. Vor dem Training beide Loss-Werte messen.
3. Für jede Epoche Trainingsfenster mischen und Batches bilden.
4. `train_epoch` ausführen, neues Modell übernehmen.
5. Mit diesen Gewichten beide Loss-Werte erneut messen.

`updates += steps` zählt alle Gewichtsupdates.
`1..=EPOCHS` durchläuft die Zahlen 1 bis einschliesslich 10.

Ein **Seed** ist ein Startwert für einen Zufallsgenerator.
`SEED + epoch as u64` gibt jeder Epoche einen anderen, aber wiederholbaren
Startwert zum Mischen. `as u64` wandelt die Epochenzahl in den dafür
benötigten Integer-Typ um.

Der Seed für die Modellinitialisierung wird nur einmal gesetzt.
Wir setzen die Gewichte nicht zu Beginn einer neuen Epoche zurück.

## 7. Demo ausführen und Ausgabe lesen

```bash
cargo run -p training --example ch09
```

Beispielausgabe mit Burn 0.22 und Flex auf der CPU
(Rundungsabweichungen gegenüber anderen Backends sind möglich):

```text
{{#include ../ausgaben/ch09.txt}}
```

Vergleiche zuerst Epoche 0 mit Epoche 10: Beide Loss-Werte sind
gesunken. Das Modell gibt den richtigen Tokens unseres Musters
nach den 60 Updates mehr Wahrscheinlichkeit.

Die Messungen selbst zählen nicht zu diesen 60 Updates.
Andere Daten, Startgewichte oder Lernraten können einen anderen
Verlauf ergeben. Ein hoher Validierungs-Loss ist kein Anlass,
heimlich auf den Validierungsdaten weiterzutrainieren.

### Vollständige ausführbare Demo

Die Demo verwendet Vokabulargrösse 5, Kontextlänge 3,
Embedding-Dimension 8, zwei Attention-Köpfe und einen Transformer-Block.
Diese Modellgrössen kennst du aus Kapitel 7; sie ändern nichts am
Prinzip der Trainingsschleife.

Burn kann Gewichte verzögert initialisieren. Die Funktion `model`
führt deshalb vor der ersten Messung einen Vorwärtsdurchlauf mit
Token-ID `0` aus. So verwenden Trainingsmodell und Messversion
dieselben bereits initialisierten Gewichte. Dieser Durchlauf
berechnet keine Gradienten und führt kein Update aus.

```rust,ignore
{{#include ../../../crates/training/examples/ch09.rs}}
```

## 8. Wo der Code liegt

Das Repository enthält zwei Rust-Crates, also zwei Codepakete:

**`model` berechnet Vorhersagen:** Token-IDs hinein, Logits heraus.
Es enthält Embedding, Transformer-Blöcke, Ausgabeprojektion und Gewichte.

**`training` organisiert Lernen und Messen:** Daten teilen, Batches bilden,
Modell aufrufen, Loss berechnen und beim Training Gewichte aktualisieren.

```text
training verwendet model
model verwendet keinen Code aus training
beide verwenden Burn
```

Diese Aufgabenteilung ist etwas anderes als die Trennung zwischen
Trainings- und Validierungsdaten. Das gleiche Modell kann beide
Datensätze auswerten. Nur der Trainingspfad verändert seine Gewichte.

## Tests und nächster Schritt

```bash
cargo test -p training --test ch09
```

Die Tests prüfen Update-Anzahl und Reihenfolge, Teil-Batches bei der
Messung, den gewichteten Loss und unveränderte Gewichte beim Messen.
Eine kleine Muster-Demo prüft ausserdem sinkende Trainings- und
Validierungswerte nach mehreren Epochen.

Als Nächstes folgen Checkpoints und Textgenerierung. Dieses Kapitel
speichert keine Gewichte und erzeugt noch keine neuen Token-Folgen.
