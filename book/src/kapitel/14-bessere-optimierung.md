# 14. Bessere Optimierung

## Lernziel

In Kapitel 13 haben wir untersucht, **welcher Gewichtsstand am besten
generalisiert**. Jetzt fragen wir: **Wie kommen die Gewichte zu ihrem
nächsten Stand?** Dafür ist der Optimizer zuständig.

Wir verstehen zuerst die Unterschiede zwischen einfachem SGD, Momentum
und Adam. Danach betrachten wir Lernratenänderungen, Warmup und Gradient
Clipping. Zum Schluss wird klar, warum gespeicherte Modellgewichte allein
für eine nahtlose Trainingswiederaufnahme nicht reichen.

Die ausführbare Demo vergleicht fünf Varianten auf echtem Text:

```bash
cargo run --release -p training --example ch14
```

Ohne Argument liest sie `data/the-verdict.txt`; ein optionales Argument
wählt eine andere UTF-8-Textdatei. Alle fünf Läufe starten neu mit
**identischen Modellgewichten** und verwenden dieselben Batches.
Die echten Messausgaben und ihre Grenzen stehen in Abschnitt 7.

## 1. Was der Optimizer verändert

Der Ablauf aus Kapitel 8 bleibt bestehen:

```text
Eingaben → Modell → Logits → Loss
                             ↓
                         backward()
                             ↓
                         Gradienten
                             ↓
                 Optimizer + Lernrate
                             ↓
                       neue Gewichte
```

Ein Gradient beschreibt, wie sich der Loss lokal verändert, wenn wir ein
Gewicht verändern. Der Optimizer verwendet diese Information für ein
Update. Er verändert die Gewichte, nicht die Token-IDs oder die Ziele.

Bei einfachem SGD wird für jedes Gewicht gerechnet:

```text
neues Gewicht = altes Gewicht - Lernrate * aktueller Gradient
```

Ein erfundenes Zahlenbeispiel: Gewicht 2,0, Gradient 0,5 und Lernrate 0,1
ergeben das neue Gewicht 1,95. Bei einem negativen Gradienten steigt das
Gewicht entsprechend. Die Richtung folgt also dem Vorzeichen des
Gradienten; die Lernrate bestimmt die Schrittweite mit.

Das ist eine lokale Entscheidung anhand des aktuellen Batches. Der
nächste Batch kann einen anderen Gradienten liefern. Ein einzelnes
Update muss deshalb weder den Loss auf allen Trainingsdaten noch den
Validierungs-Loss senken.

## 2. Momentum: bisherige Richtung mitnehmen

Einfaches SGD ohne Momentum hat kein Gedächtnis für frühere Gradienten.
Jeder Schritt folgt dem aktuellen Gradienten, auch wenn dieser stark
von den vorherigen abweicht.

**Momentum** ergänzt eine gespeicherte Bewegungsrichtung. Stell dir
einen Ball mit Schwung vor: Eine kleine Gegenbewegung kehrt seine
Bewegung nicht sofort um. Beim Optimizer ist dieser Schwung eine
zusätzliche Zahl für jedes Gewicht.

Eine gebräuchliche Variante lässt sich so lesen:

```text
neuer Schwung = Momentum-Faktor * alter Schwung + aktueller Gradient
neues Gewicht = altes Gewicht - Lernrate * neuer Schwung
```

Der Momentum-Faktor bestimmt, wie viel vom vorherigen Schwung bleibt.
Bei 0,9 bleiben davon 90 Prozent erhalten; der neue Gradient kommt hinzu.
Die Demo wählt dafür in Burn `MomentumConfig::new().with_momentum(0.9)`
mit `with_dampening(0.0)`: Der neue Gradient wird nicht abgeschwächt.
Ohne diese ausdrückliche Wahl wäre Burns Dampening standardmässig 0,1.
Nesterov bleibt ausgeschaltet. Andere Varianten gewichten auch den neuen
Gradienten oder verwenden eine zusätzliche Vorausschau.

Zeigen mehrere Gradienten in dieselbe Richtung, baut sich Schwung auf.
Wechseln ihre Vorzeichen häufig, können sich ihre Beiträge teilweise
aufheben. Das kann Zickzackbewegungen verringern. Zu viel Schwung oder
eine zu grosse Lernrate kann aber auch über günstige Gewichte
hinausschiessen. Momentum garantiert kein besseres Ergebnis.

### Wo liegt dieses Gedächtnis?

Die **Gewichte liegen im Modell**, der **Schwung liegt im Optimizer**.
Ein neuer Optimizer behält deshalb nicht automatisch die bisherige
Bewegungsrichtung, selbst wenn wir dasselbe Modell weiterverwenden.

Unser Code ist dafür bereits vorbereitet:

```rust
{{#include ../../../crates/training/src/step.rs:train_step}}
```

`optimizer: &mut ModuleOptimizer` bedeutet: Der Aufrufer übergibt einen
vorhandenen Optimizer, dessen Zustand verändert werden darf.
`train_step` erzeugt ihn nicht bei jedem Update neu. Auch `train_epoch`
verwendet den übergebenen Optimizer weiter.

Die bisherige Demo `ch13` initialisiert vor der Epochenschleife einmal
`SgdConfig::new().init()`. Das ist weiterhin **SGD ohne Momentum**.
Ein erhaltenes Optimizer-Objekt allein schaltet Momentum nicht ein.
Die Hilfsfunktion `sgd_step` erzeugt dagegen für ihren einzelnen
SGD-Schritt einen neuen Optimizer; sie ist nicht die Vorlage für eine
Trainingsschleife mit Gedächtnis.

## 3. Adam: Richtung und Grössenordnung merken

Adam verwendet für jedes Gewicht zwei geglättete Verläufe:

- den Verlauf der Gradienten, also eine gemittelte Richtung,
- den Verlauf der quadrierten Gradienten, also deren Grössenordnung
  ohne Vorzeichen.

Ein geglätteter Verlauf mischt den bisherigen Wert mit einer kleineren
Portion des neuen Werts. Einzelne Batches ersetzen das Gedächtnis
dadurch nicht vollständig. Die quadrierten Gradienten sind kein direktes
Mass für Unsicherheit, sondern Teil der Skalierung des Updates.

Adam teilt die geglättete Richtung durch eine aus dem zweiten Verlauf
berechnete Skala. Bei sonst gleicher Richtung dämpft eine grössere
Skala das Update stärker. So unterscheidet sich die Anpassung von
Gewicht zu Gewicht, statt überall nur denselben Lernratenfaktor auf
den aktuellen Gradienten anzuwenden.

Am Anfang starten beide Verläufe bei null. Adam korrigiert die dadurch
entstehende Anfangsverzerrung anhand der bisherigen Schrittzahl.
Ausserdem steht eine kleine positive Zahl im Nenner, um eine Division
durch null zu vermeiden. Diese Details gehören zur Methode; Adam ist
nicht bloss SGD mit einem anderen Namen.

### Braucht Adam noch eine Lernrate?

Ja. Die **gemeinsame Lernrate** skaliert weiterhin die Updates.
Adam ergänzt eine **Anpassung je Gewicht**. Die beiden Aufgaben
ersetzen einander nicht.

Die SGD-Lernrate 0,05 aus Kapitel 13 ist deshalb kein automatisch
passender Adam-Wert. Ein Vergleich mit ungeprüft gleicher Lernrate
zeigt nicht zuverlässig, welcher Optimizer besser geeignet ist.
Wir müssen die Lernrate zum Verfahren passend wählen und weiterhin
den Validierungs-Loss beobachten.

### Optimizer in Rust auswählen

```rust
{{#include ../../../crates/training/examples/ch14.rs:optimizer}}
```

Alle Zweige liefern denselben Typ `ModuleOptimizer`. Deshalb kann
`train_step` SGD, Momentum oder Adam verwenden, ohne Modellberechnung
oder Autograd zu ändern. `run` erzeugt den Optimizer einmal pro
Vergleichslauf, **nicht** einmal pro Batch oder Epoche.

Die Demo verwendet die Adam-Standardwerte von Burn 0.22:
`beta_1 = 0.9`, `beta_2 = 0.999`, `epsilon = 1e-5`, ohne AMSGrad
oder Weight Decay. Die Lernraten sind 0,05 für SGD, 0,01 für Momentum
und 0,001 für Adam. Das sind ausdrücklich Versuchswerte, keine
abgestimmten Optima.

Adam kann die Optimierung erleichtern, verhindert aber kein Overfitting.
Ein schneller sinkender Trainings-Loss beweist weder bessere
Generalisierung noch bessere Textgenerierung.

## 4. Lernratenänderung: Schrittweite über die Zeit

In Kapitel 13 konnten wir über `--learning-rate` eine Lernrate wählen.
Innerhalb eines Laufs blieb sie konstant. Ein **Lernratenplan**, auch
Scheduler genannt, ändert sie dagegen während des Trainings.

Eine spätere Absenkung soll feinere Anpassungen ermöglichen. Kleinere
Schritte garantieren jedoch weder ein besseres Minimum noch bessere
Validierungswerte. Sinkt die Lernrate zu früh oder zu stark, kann das
Lernen nahezu stehen bleiben.

Ein Plan braucht eine klare Zeiteinheit. Für unseren Lernpfad verwenden
wir als Bezug **Optimizer-Updates**, nicht Loss-Messungen oder erzeugte
Tokens. Eine Epoche kann viele Updates enthalten.

### Warmup

Warmup ist die Anfangsphase eines Lernratenplans: Wir beginnen klein
und erhöhen die Lernrate über einige Updates bis zum geplanten Wert.
Das kann sehr grosse Änderungen zu Trainingsbeginn begrenzen, während
Gewichte und Optimizer-Verläufe sich erst entwickeln.

Ein rein illustrativer Plan, keine gemessene Empfehlung:

```text
Update 1:          Lernrate 0,00025
Update 2:          Lernrate 0,00050
Update 3:          Lernrate 0,00075
Update 4:          Lernrate 0,00100
spätere Updates:   Lernrate schrittweise absenken
```

Die Zahlen bezeichnen die Lernrate für das jeweilige Update.
Warmup ist weder eine zusätzliche Epoche ohne Lernen noch eine
Adam-Eigenschaft, die automatisch eingeschaltet wäre. Auch während
des Warmups werden Gewichte verändert.

**Adam passt je Gewicht an; der Lernratenplan passt über die Zeit an.**
Beides kann gemeinsam verwendet werden. Warmup bleibt eine zu prüfende
Entscheidung, keine Pflicht für jeden kleinen Trainingsversuch.

### Wo passt der Plan in unseren Code?

`train_step` erhält die Lernrate bereits als Argument. Dort müssen wir
die Modellberechnung nicht für einen Scheduler neu schreiben.
`train_epoch` reicht momentan aber dieselbe Lernrate an alle Batches
weiter. Die Demo verwendet deshalb eine eigene kleine Batch-Schleife
und berechnet die globale Update-Nummer über die Epochengrenzen hinweg.

Nur vor jeder Epoche einen neuen Wert zu übergeben wäre ein Plan
pro Epoche, nicht der hier beschriebene Warmup pro Update.

### Warmup und Absenkung in Rust

```rust
{{#include ../../../crates/training/examples/ch14.rs:schedule}}
```

`update` zählt ab eins und bezeichnet das **bevorstehende** Update.
Die ersten vier Updates steigen linear auf 0,001. Danach sinkt die
Lernrate linear auf ein Zehntel dieses Werts beim letzten Update.
Bei einem Lauf mit weniger als vier Updates wird der Warmup verkürzt;
für die spätere Absenkung bleiben dann keine Updates übrig.
Keine Lernrate wird null.

```rust
{{#include ../../../crates/training/examples/ch14.rs:run}}
```

Bei vier Batches pro Epoche beginnen die globalen Nummern mit 1, 5 und 9.
Die zweite Epoche startet **nicht** erneut mit Warmup. Die Ausgabe zeigt
die tatsächlich verwendete Lernrate jedes Updates.

```rust
{{#include ../../../crates/training/examples/ch14.rs:epoch}}
```

Jeder Batch bekommt genau einen Wert aus dem Plan; derselbe Optimizer
behält sein Gedächtnis. Die Messung nach einer Epoche verwendet feste
Gewichte und verändert weder Optimizer noch Update-Nummer.

## 5. Gradient Clipping: grosse Gradienten begrenzen

Ein Gradient kann ungewöhnlich gross werden. Dann kann selbst eine
sonst brauchbare Lernrate ein heftiges Update bewirken.
**Gradient Clipping** begrenzt Gradienten, bevor der Optimizer sie
für sein Update verarbeitet.

Bei Clipping nach globaler Norm betrachten wir alle Parametergradienten
gemeinsam als einen langen Vektor. Überschreitet dessen Länge einen
Grenzwert, skalieren wir alle Beiträge mit demselben Faktor herunter.
Die gemeinsame Richtung bleibt dabei erhalten.

Ein erfundenes Beispiel: Hat dieser Vektor die Länge 10 und ist der
Grenzwert 2, werden alle Beiträge mit 0,2 multipliziert. Bei einer
Länge von 1 bleibt der Vektor unverändert.

Davon zu unterscheiden sind Clipping je Parametertensor und das
Begrenzen einzelner Werte auf ein Intervall. Diese Varianten haben
nicht dieselbe Wirkung.

Die Demo wählt `GradientClippingConfig::Norm(0.1)` über
`AdamConfig::with_grad_clipping`. In Burn 0.22 begrenzt diese API die
L2-Norm **jedes einzelnen Parametergradiententensors**, nicht die globale
Norm aller Gradienten zusammen. Jeder Tensor erhält seinen eigenen
Skalierungsfaktor; die gemeinsame Richtung aller Parametergradienten
muss deshalb nicht erhalten bleiben. Das globale Beispiel oben erklärt
das Konzept, nicht die genaue Wirkung dieser Burn-Konfiguration.

Clipping findet innerhalb von `ModuleOptimizer::step` vor der
Adam-Verarbeitung statt. `train_step` braucht dafür keine zweite
Clipping-Implementierung. Der kleine Grenzwert 0,1 dient dem Vergleich,
nicht als allgemeine Empfehlung.

```text
backward() → Gradienten → Clipping → Optimizer-Update
```

Clipping begrenzt nicht direkt den Loss oder die Modellgewichte.
Bei Adam mit seinem gespeicherten Zustand ist ein Grenzwert für
Gradienten auch kein identischer Grenzwert für Gewichtsänderungen.
Clipping ersetzt weder eine passende Lernrate noch die Untersuchung
nicht endlicher Werte wie `NaN`.

## 6. Warum der Optimizer-Zustand in den Checkpoint gehört

Nach Kapitel 10 speichern `save_model` und `load_model` nur Gewichte.
Für Auswertung und Generierung reicht das zusammen mit der passenden
Modellkonfiguration. Der Optimizer wird dabei nicht gebraucht.

Für dasselbe nächste Trainingsupdate reicht es bei Adam nicht:

```text
gleiche Gewichte + gleicher nächster Batch + gleiche Lernrate
    + bisheriger Adam-Zustand → fortgesetztes Update
    + neuer Adam-Zustand      → möglicherweise anderes Update
```

Mit einem neuen Adam gehen beide geglätteten Verläufe und sein bisheriger
Schrittstand verloren. Mit neuem Momentum-SGD geht der Schwung verloren.
Das Modell verliert dadurch nicht seine gelernten Gewichte; es lernt
aber unter veränderten Update-Bedingungen weiter.

Für eine vollständige Wiederaufnahme brauchen wir zusammengehörige Daten:

- Modellgewichte und Modellkonfiguration,
- Optimizer-Zustand und Optimizer-Konfiguration,
- globalen Update-Stand und Lernratenplan samt nötigem Zustand,
- Trainingsposition, Datenaufteilung und reproduzierbare Batch-Reihenfolge,
- nötige Zufallszustände, falls der Lauf Zufall verwendet.

Ein fixer Seed allein ersetzt nicht in jedem Fall einen bereits
fortgeschrittenen Zufallszustand. Beim Fortsetzen mitten in einer
Epoche muss ausserdem bekannt sein, welche Batches schon verarbeitet
wurden. Für die Bestmodell-Auswahl aus Kapitel 13 gehört auch der
bisher beste Validierungs-Loss zum fortzusetzenden Ablauf.

Alle Teile müssen denselben Trainingszeitpunkt beschreiben. Gewichte
aus Epoche 5 mit Optimizer-Zustand aus Epoche 8 sind kein konsistenter
Checkpoint. Auch mit vollständigem Zustand hängt bitgenaue
Reproduzierbarkeit von Backend und deterministischen Operationen ab.

**Aktueller Stand:** Vollständiger Trainings-Resume ist nicht umgesetzt.
`ch13` startet bei jedem Aufruf ein neu initialisiertes Modell und einen
neuen SGD-Optimizer. Die vorhandenen Ladefunktionen erlauben die Nutzung
gespeicherter Gewichte, stellen aber keinen gesamten Trainingslauf wieder
her. Auch `ch14` startet neu und speichert keinen Trainingszustand.

## 7. Fünf echte Trainingsläufe

Wir müssen dafür keine Attention oder Transformer-Blöcke neu bauen.
Die vorhandene Trennung bleibt: `model` berechnet Vorhersagen,
`training` steuert Updates und Messungen.

### Trainingsläufe starten

Starte im Projektverzeichnis alle fünf Varianten nacheinander:

```bash
cargo run --release -p training --example ch14
```

Ohne Argument verwendet die Demo `data/the-verdict.txt`. Für eine eigene
UTF-8-Textdatei übergibst du deren Pfad:

```bash
cargo run --release -p training --example ch14 -- /pfad/zum/text.txt
```

Beide Aufrufe starten SGD, Momentum, Adam, Adam mit Lernratenplan und
Adam mit Lernratenplan plus Clipping. Einzelne Varianten lassen sich
nicht per CLI auswählen. Jeder Aufruf beginnt mit neuen Ausgangsgewichten;
es wird kein Checkpoint geladen oder gespeichert.

### Daten und Umfang

Die Demo nutzt dieselbe Modellkonfiguration wie Kapitel 13:
50'257 Token-IDs, Kontext acht, Embedding acht, zwei Köpfe und einen
Transformer-Block. Nach der Aufteilung des ganzen Token-Stroms
(90 Prozent Training, 10 Prozent Validierung) verwendet sie nur die
**ersten acht Trainingsfenster und zwei Validierungsfenster** der
jeweiligen Bereiche. Kein Fenster überschreitet die Trennstelle.
Zu kurzer Text führt zu einer Fehlermeldung.

Mit Batch-Grösse zwei sind das vier Updates pro Epoche, drei Epochen
und **zwölf Updates je Variante**, insgesamt 60 Updates. Ausgewertet
werden die Ausgangsgewichte und jeder Epochenstand auf allen ausgewählten
Fenstern. Die Reihenfolge wird je Epoche mit `SEED + epoch` gemischt;
alle Varianten verwenden dieselbe Reihenfolge. Der Lauf hat keinen
Zeitabbruch, seine Dauer hängt von Hardware und Build ab.

### Ausgabe lesen

Für den Standardtext `the-verdict.txt` mit Seed 42 zeigt die Demo folgende
Messwerte. Sie verwendet Burn 0.22 mit dem CPU-Backend
`Device::flex().autodiff()`. Je nach Plattform können die Werte leicht
abweichen:

```text
{{#include ../ausgaben/ch14-vergleich.txt}}
```

### Was zeigen die Messwerte?

Alle fünf Ausgangsmessungen sind gleich: Training 11,110244 und
Validierung 10,768029. Das passt zum Kopieren derselben Ausgangsgewichte.
Alle fünf Varianten senken beide Loss-Werte in diesem kurzen Versuch.
SGD erreicht nach zwölf Updates den kleinsten Validierungs-Loss
(10,722695), Adam mit konstanter Lernrate 10,731487.
Das beweist keine Überlegenheit von SGD: Lernraten unterscheiden sich,
wurden nicht optimiert und es gibt nur zwei Validierungsfenster.

Beim Momentum-Lauf wird der Rückgang des Trainings-Loss von Epoche zu
Epoche grösser. Das ist mit aufgebautem Schwung vereinbar, isoliert
aber dessen Wirkung nicht von den wechselnden Batches.

Der geplante Adam-Lauf erreicht 10,748151 statt 10,731487 bei konstanter
Lernrate. Warmup und Absenkung reduzieren hier zusammen die Summe der
Lernraten von 0,012 auf 0,00645. Ein kleinerer Fortschritt nach denselben
zwölf Updates ist deshalb kein überraschendes Ergebnis.
Dieser Vergleich untersucht **den ganzen Plan**; er trennt Warmup
nicht von der späteren Absenkung.

Mit zusätzlichem Clipping liegt der Validierungs-Loss bei 10,754271.
Der Eingriff verändert also den Verlauf, verbessert ihn hier aber
nicht. Der Lauf zeigt keine explodierenden Gradienten und beweist
weder die Notwendigkeit von Clipping noch dessen Nutzen bei instabilem
Training. Ein Schutzmechanismus ist kein automatischer Qualitätsgewinn.

Die Werte sind nicht direkt mit Kapitel 13 vergleichbar: Dort wurden
alle Trainings- und Validierungsfenster ausgewertet, hier nur ein kleiner
Ausschnitt. Für belastbare Optimizer-Vergleiche brauchen wir längere
Läufe, mehrere Seeds, abgestimmte Lernraten und mehr zurückgehaltene Daten.
Vollständiges Resume braucht zusätzlich einen eigenen Speichern-/Laden-
Vergleich gegen einen ununterbrochenen Lauf und bleibt offen.

## Das nehmen wir mit

SGD ohne Momentum reagiert nur auf den aktuellen Gradienten.
Momentum bewahrt Bewegungsrichtung; Adam bewahrt zusätzlich einen
Verlauf quadrierter Gradienten und skaliert Updates je Gewicht.
Die Lernrate bleibt wichtig, ein Lernratenplan verändert sie über die
Zeit, Warmup gestaltet ihren Start und Clipping begrenzt Gradienten.

Gewichte enthalten das Gelernte. Optimizer-Zustand enthält das
Gedächtnis des Update-Verfahrens. Für eine nahtlose Wiederaufnahme
brauchen wir beides und die passende Trainingsposition.
