# 12. Echter Text und Tokenizer

## Lernziel

Bis Kapitel 11 haben wir Training, Checkpoints und Generierung mit fünf
künstlichen Token-IDs erarbeitet. Jetzt ersetzen wir das Zahlenmuster durch
echten Text: `data/the-verdict.txt`.

Dabei bauen wir keinen neuen Tokenizer und keine neue Modellarchitektur.
Wir verbinden den GPT-2-Tokenizer aus Kapitel 1 mit der Trainingsmechanik
aus Kapitel 9. Der erste Versuch bleibt absichtlich klein: wenige Fenster,
höchstens drei SGD-Updates, danach eine Loss-Messung auf getrennten Daten.

Danach erweitern wir den Versuch auf ganze Epochen, speichern die gelernten
Gewichte und laden sie in einem eigenen Programmaufruf zum Generieren.
Die zentrale Unterscheidung lautet: **Training verändert Gewichte;
Generierung verwendet Gewichte.**

## Wegweiser: Was führe ich wann aus?

Alle Befehle werden im Repository-Verzeichnis ausgeführt. Arbeite die
Schritte nacheinander durch, nicht alle auf einmal:

| Schritt | Programm | Was passiert? | Was soll ich erkennen? |
|---|---|---|---|
| 1 | `ch02` mit Textpfad | Text tokenisieren und Fenster bilden; kein Training | Eingabe und Ziel sind um ein Token verschoben. |
| 2 | `ch12` | Höchstens drei Updates; kein Speichern, keine Generierung | Die bekannte Trainingsmechanik funktioniert mit echten Text-IDs. |
| 3 | `ch12_full train` | Ganze Epochen, Loss-Messungen und Gewichte speichern | Lernen verändert die Vorhersagen; Validierung prüft getrennten Text. |
| 4 | `ch12_full generate` | Gespeicherte Gewichte laden und Text fortsetzen | Besserer Loss garantiert noch keine sinnvolle freie Fortsetzung. |

Beginne mit Schritt 1 in Abschnitt 1. Schritt 2 folgt in Abschnitt 4.
Führe den längeren Schritt 3 erst aus, wenn du die beiden Loss-Werte des
kurzen Versuchs einordnen kannst. Schritt 4 setzt einen erfolgreich
gespeicherten Checkpoint aus Schritt 3 voraus.

Ohne eigenen Textpfad verwenden beide Kapitel-12-Demos
`data/the-verdict.txt`. Ein optionaler Textpfad wählt eine andere Datei,
nicht eine andere Trainingsart. Ein erneuter `train`-Aufruf startet neu;
er setzt das bisherige Training nicht fort.

## 1. Erst den Datenweg ansehen

Führe im Repository-Verzeichnis die vorhandene Demo aus:

```bash
cargo run -p training --example ch02 -- data/the-verdict.txt
```

Mit der enthaltenen Datei zeigt sie:

```text
Token gesamt: 5145, Training: 4630, Validierung: 515
Fenster (Länge 8): 578

Fenster 0, Eingabe: "I HAD always thought Jack Gis"
Fenster 0, Ziel:    " HAD always thought Jack Gisburn"

Batches pro Epoche (Grösse 2): 289
```

Ein Token ist nicht zwingend ein Wort. Der Tokenizer kann beispielsweise
einen Namen in mehrere Wortteile zerlegen. Deshalb endet die Eingabe mit
`Gis`, während das um ein Token verschobene Ziel mit `Gisburn` endet.
Leerzeichen und Satzzeichen gehören ebenfalls zum tokenisierten Text.

Acht Eingabe-Tokens brauchen acht Nachfolger. Dafür benötigen wir neun
aufeinanderfolgende Tokens aus dem ursprünglichen Text:

```text
Textstrom: [a, b, c, d, e, f, g, h, i]
Eingabe:   [a, b, c, d, e, f, g, h]
Ziel:      [b, c, d, e, f, g, h, i]
```

Das neunte Token ist nur das letzte Ziel, kein zusätzliches Eingabe-Token.
Das Modell erhält die Eingabe. Die Ziele werden danach für die
Cross-Entropy verwendet. Beim Training liefert jede Position eine
Vorhersage, nicht nur die letzte Position wie bei der Generierung.

Die Batch-Reihenfolge wird gemischt. Der erste ausgegebene Batch muss
deshalb nicht mit Fenster 0 beginnen. Innerhalb eines Fensters bleiben
Token-Reihenfolge und Zuordnung zum Ziel unverändert.

## 2. Textlänge ist nicht Vokabular-Grösse

`Gpt2Tokenizer` verwendet bereits `r50k_base` aus `tiktoken-rs`:

```rust
use training::tokenizer::{Gpt2Tokenizer, VOCAB_SIZE};

let tokenizer = Gpt2Tokenizer::new();
let tokens = tokenizer.encode(&text);
```

Die Datei enthält **5'145 Tokens**, einschliesslich Wiederholungen.
Das GPT-2-Vokabular enthält dagegen **50'257 Einträge**. Jede ID bezeichnet
einen Eintrag dieser festen Zuordnung. Wir nummerieren die vorkommenden
IDs nicht neu und reduzieren sie auch nicht mit `% 5`.

Das Modell muss für jede Eingabeposition alle möglichen nächsten IDs
bewerten. Die Logits haben die Form `[B, T, V]`:

```text
B = 2 Fenster pro Batch
T = 8 Positionen pro Fenster
V = 50'257 mögliche nächste Token-IDs

Eingaben: [2, 8]
Ziele:    [2, 8]
Logits:   [2, 8, 50257]
```

Die Ziele enthalten pro Position genau eine richtige ID. Die Logits
enthalten pro Position eine Bewertung für jeden Vokabular-Eintrag.
Auch IDs, die nicht in diesem kurzen Text vorkommen, gehören zur Ausgabe.

**GPT-2-Tokenizer bedeutet nicht vortrainiertes GPT-2.** Wir verwenden
seine Text-ID-Zuordnung, aber unser kleines Modell startet mit zufälligen
Gewichten und lernt selbst.

## 3. Trennen, dann begrenzen, dann Fenster bilden

Die neue Demo liest standardmässig die enthaltene Textdatei. Ein optionaler
Pfad erlaubt einen anderen Text. Eine fehlende Datei führt zu einem Fehler,
nicht zu einem versteckten Ersatztext.

Der entscheidende Datenweg:

```text
Datei → Text → GPT-2-IDs → Training / Validierung
                             ↓              ↓
                       kurzer Anfang   kurzer Anfang
                             ↓              ↓
                       Trainingsfenster Validierungsfenster
```

Wir teilen zuerst den vollständigen Token-Strom: 90 Prozent Training,
der Rest Validierung. Erst danach nehmen wir aus beiden Bereichen jeweils
einen kleinen Anfang und bilden eigene Fenster:

```rust
{{#include ../../../crates/training/examples/ch12.rs:data}}
```

Die Trainingsdemo verwendet höchstens acht Trainingsfenster und zwei
Validierungsfenster. Bei Kontextlänge acht reichen dafür 65 Tokens aus
dem Trainingsbereich und 17 aus dem Validierungsbereich.

Die Validierung beginnt also nicht nach den 65 verwendeten Trainingstokens,
sondern am zuvor bestimmten Split des gesamten Textes. Kein Fenster
überschreitet diese Grenze. Dieselbe ID darf in beiden Bereichen vorkommen;
getrennt bleiben die ursprünglichen Textpositionen.

Sind die Bereiche kürzer, entstehen weniger Fenster. Reicht ein Bereich
nicht einmal für ein vollständiges Fenster mit Ziel, meldet `Dataset::new`
einen Fehler. Ohne vollständigen Trainingsbatch startet kein Training.

## 4. Einen begrenzten Trainingsversuch ausführen

```bash
cargo run -p training --example ch12

# Alternativ: eigener UTF-8-Text
cargo run -p training --example ch12 -- data/mein-text.txt
```

Ohne Textpfad verwendet die Demo automatisch `data/the-verdict.txt`.
`data/mein-text.txt` ist ein Beispielpfad für eine eigene, zuvor angelegte
UTF-8-Textdatei; diese Datei ist nicht im Repository enthalten.

Die Modellkonfiguration bleibt klein:

```text
Vokabular:         50'257
Kontextlänge:      8
Embedding-Grösse:  8
Attention-Köpfe:   2
Transformer-Blöcke: 1
Batch-Grösse:      2
Lernrate:          0.05
Zufallsseed:       42
```

Das CPU-Device ist `Device::flex().autodiff()`. Der bestehende
`train_step` berechnet Logits und Cross-Entropy, ruft `backward()` auf und
verändert die Gewichte mit SGD. Derselbe Optimizer bleibt während des
Versuchs erhalten:

```rust
{{#include ../../../crates/training/examples/ch12.rs:bounded_training}}
```

Acht Trainingsfenster ergeben vier vollständige Batches. Wir verwenden
höchstens die ersten drei gemischten Batches, jeweils einmal. Das ist
bewusst **keine vollständige Epoche** über den Beispieltext.

Vor jedem Update wird ein Zeitbudget von 30 Sekunden geprüft. Sobald es
verbraucht ist, startet kein weiteres Update; die Ausgabe nennt den
Abbruch. Ein laufendes Update wird nicht unterbrochen und darf die Grenze
überschreiten. **Das ist keine harte 30-Sekunden-Grenze für den Prozess.**
Dateilesen, Tokenisierung, Initialisierung und die Loss-Messungen liegen
ausserhalb dieses Trainingszeitbudgets. Auch Cargo-Kompilierung zählt nicht.

Nicht endlicher Trainings-Loss führt zu einem Fehler. Die Demo speichert
keinen Checkpoint und startet keine Generierung.

## 5. Loss vor und nach den Updates verstehen

Beide Messungen verwenden dieselben begrenzten Datenbereiche:

```rust
{{#include ../../../crates/training/examples/ch12.rs:report}}
```

`model.valid()` liefert die Variante ohne Autodiff. `evaluate` berechnet
den Loss ohne Gradienten und ohne Gewichtsupdates.

Die Ausgabe enthält:

```text
Vor Training: Trainings-Loss ..., Validierungs-Loss ...
Update 1: Loss vor Update ..., Trainingszeit ...s
Update 2: Loss vor Update ..., Trainingszeit ...s
Update 3: Loss vor Update ..., Trainingszeit ...s
Ausgeführte SGD-Updates: 3
Nach Training: Trainings-Loss ..., Validierungs-Loss ...
```

Die Update-Loss-Werte stammen aus unterschiedlichen Batches und
Gewichtsständen. Sie müssen nicht monoton sinken. Vergleiche zuerst
die erneuten Messungen vor und nach dem Training auf denselben Fenstern.
Die Trainingsmessung umfasst alle acht ausgewählten Fenster, auch die
Fenster im vierten, nicht trainierten Batch.

Ein sinkender Trainings-Loss zeigt bessere Vorhersagen auf diesen
Beispielen. Zwei Validierungsfenster sind aber viel zu wenig für eine
belastbare Aussage über allgemeine Sprachqualität. Drei Updates
garantieren weder sinkenden Validierungs-Loss noch sinnvolle Sätze.

## 6. Speicher und Rechenzeit

Ein voller Batch erzeugt:

```text
2 × 8 × 50'257 = 804'112 Logit-Werte
804'112 × 4 Bytes pro f32 = 3'216'448 Bytes ≈ 3,1 MiB
```

Das ist **nur der Logit-Tensor**, nicht der gesamte Speicherbedarf.
Token-Embedding und separate Ausgabeprojektion enthalten hier jeweils
50'257 × 8 Gewichte. Zusammen sind das weitere 804'112 Gewichte; dazu
kommen Positions-Embeddings, Transformer-Parameter, Zwischenwerte und
Gradienten. Backend und temporäre Berechnungen benötigen ebenfalls Speicher.

Mehr Fenster pro Batch und längerer Kontext vergrössern die Logits.
Längerer Kontext vergrössert zusätzlich die Attention-Scores; deren Anzahl
wächst quadratisch mit der Kontextlänge. Grössere Embeddings und mehr
Blöcke vergrössern das Modell und den Rechenaufwand.

Die Demo zeigt die tatsächlich verstrichene Trainingszeit nach jedem
Update. Konkreter RAM-Bedarf und Laufzeit hängen von Hardware und Build ab.
Der erste Cargo-Aufruf kann durch die Kompilierung deutlich länger dauern
als das eigentliche Training. Für Laufzeitvergleiche kann ein Release-Build
verwendet werden:

```bash
cargo run --release -p training --example ch12
```

Erhöhe nicht gleichzeitig Kontext, Batch-Grösse und Update-Anzahl.
Ändere zuerst nur eine dieser Konstanten und beobachte den Unterschied.

## 7. Was dieses Modell noch nicht kann

Ein einzelner kurzer Text und ein sehr kleines, neu initialisiertes Modell
ergeben kein allgemein brauchbares Sprachmodell. Es kann lokale Muster
lernen oder Beispiele auswendig lernen, hat aber wenig Daten und einen
Kontext von nur acht Tokens. Lesbare Ausgabe wäre noch kein Nachweis von
Sprachverständnis.

Die Checkpoints aus Kapitel 10 und 11 passen nicht zu dieser Demo: Sie
wurden für fünf IDs statt 50'257 erstellt. Gewichtsformen und Token-Zuordnung
müssen beim späteren Speichern und Generieren übereinstimmen. Für echtes
Text-Decoding brauchen wir wieder `Gpt2Tokenizer::decode`, nicht die
Fünf-Wörter-Tabelle aus Kapitel 11.

## Prüfen und weiterlernen

```bash
cargo test -p training --example ch12
```

Die Tests prüfen echte Text-IDs und Split-Grenzen, verschobene Ziele,
Tensorformen, eine Gewichtsänderung durch SGD, Auswertung ohne Updates,
Update- und Zeitbudget sowie Fehler bei zu wenig Text.

## 8. Jetzt den gesamten Text in Epochen verwenden

Der Drei-Update-Versuch bleibt der schnelle Einstieg. Die zweite Demo
`ch12_full` ergänzt den vollständigen Datenweg und startet nur über einen
ausdrücklichen Befehl. Ohne Argumente zeigt sie einen Aufruffehler, statt
unbemerkt ein längeres Training auszuführen.

Starte zunächst **eine** Epoche, wenn du diesen längeren Lauf durchführen
möchtest:

```bash
cargo run --release -p training --example ch12_full -- train checkpoints/ch12-model.bpk 1
```

Der letzte Wert ist die Epochenzahl. Erlaubt sind eins bis drei; null,
grössere Werte und ungültige Zahlen führen zu einem Fehler. Ohne Textpfad
wird automatisch `data/the-verdict.txt` verwendet. Ein optionaler Pfad
wählt stattdessen eine eigene, zuvor angelegte UTF-8-Textdatei:

```bash
cargo run --release -p training --example ch12_full -- train checkpoints/ch12-model.bpk 1 data/mein-text.txt
```

`data/mein-text.txt` ist nur ein Beispielpfad und nicht im Repository
enthalten. Explizite relative Pfade beziehen sich auf das aktuelle
Arbeitsverzeichnis. Die Standarddatei wird dagegen über den Projektpfad
gefunden.

Die Modellkonfiguration entspricht dem schnellen Versuch, aber wir
begrenzen die Daten nicht mehr auf die ersten Fenster:

```rust
{{#include ../../../crates/training/examples/ch12_full.rs:full_data}}
```

Mit der enthaltenen Datei ergeben sich:

```text
Training:     4'630 Tokens → 578 Fenster → 289 vollständige Batches
Validierung:    515 Tokens →  64 Fenster →  32 vollständige Batches
```

Die Formel bleibt `(Token-Anzahl - 1) / Kontextlänge`, ganzzahlig abgerundet.
Unvollständige Fenster am Ende fallen weg. Bei einem anderen Text kann auch
ein unvollständiger Trainingsbatch entstehen; `Dataset::epoch` verwirft
ihn wie in Kapitel 9. Die Auswertung berücksichtigt dagegen auch einen
letzten Teil-Batch. „Volle Epoche“ meint alle vollständigen Trainingsbatches,
nicht jede einzelne Textposition.

Eine Epoche führt hier 289 Updates durch, drei Epochen 867. Vor dem
Training und nach jeder Epoche werden Trainings- und Validierungs-Loss
auf **allen Fenstern der jeweiligen Bereiche** gemessen:

```rust
{{#include ../../../crates/training/examples/ch12_full.rs:full_training}}
```

Die Gewichte und derselbe SGD-Optimizer bleiben zwischen den Epochen
erhalten. Nur die Reihenfolge der Trainingsfenster wird mit einem anderen
Seed pro Epoche gemischt. Validierung verändert keine Gewichte.

**Dieser Lauf hat keinen 30-Sekunden-Zeitabbruch.** Die begrenzte
Epochenzahl begrenzt die Anzahl Updates, nicht die verstrichene Zeit.
Initialisierung, vollständige Loss-Messungen und Checkpoint-Schreiben
kommen zur Trainingszeit hinzu. Die Ausgabe zeigt nach jeder Epoche die
verstrichene Zeit seit Beginn der Epochenschleife. Eine Laufzeit für deine
Hardware ist nicht vorgegeben. Der Drei-Update-Versuch bleibt die
ressourcenschonende Alternative.

### Eine vollständige Trainingsausgabe lesen

Das folgende Beispiel stammt aus einem ausgeführten Lernlauf mit **drei**
Epochen. Dafür wurde als letzter Pflichtwert `3` statt `1` gewählt:

```bash
cargo run --release -p training --example ch12_full -- train checkpoints/ch12-model.bpk 3
```

```text
Fenster: Training 578, Validierung 64
3 Epochen, 289 Updates pro Epoche, 867 Updates insgesamt
Volle Epochen: kein Zeitabbruch. Abbrechen mit Ctrl+C; letzte abgeschlossene Epoche bleibt gespeichert.
Epoche 0: Trainings-Loss 11.186024, Validierungs-Loss 11.192645
Epoche 1: Trainings-Loss 10.267453, Validierungs-Loss 10.402124
Modellgewichte gespeichert: checkpoints/ch12-model.bpk
Epoche 1 abgeschlossen: 289 Updates, Gesamtzeit 3.04s
Epoche 2: Trainings-Loss 8.742053, Validierungs-Loss 8.892983
Modellgewichte gespeichert: checkpoints/ch12-model.bpk
Epoche 2 abgeschlossen: 289 Updates, Gesamtzeit 5.99s
Epoche 3: Trainings-Loss 8.098851, Validierungs-Loss 8.235974
Modellgewichte gespeichert: checkpoints/ch12-model.bpk
Epoche 3 abgeschlossen: 289 Updates, Gesamtzeit 8.95s
```

Deine Zahlen und Zeiten können abweichen. Die Bedeutung der Zeilen bleibt:

- **Fenster:** 578 Trainingsbeispiele und 64 zurückgehaltene Beispiele.
  Jedes Fenster enthält acht Eingabe-Tokens und acht Ziel-Tokens.
- **Updates:** Zwei Trainingsfenster bilden einen Batch. Deshalb ergeben
  578 Fenster genau 289 Batches. Jeder Batch führt zu einem gemeinsamen
  Gewichtupdate; drei Durchläufe ergeben 867 Updates.
- **Epoche 0:** Nur die Ausgangsmessung mit zufälligen Gewichten.
  Es wurde noch kein Update durchgeführt.
- **Epoche 1 bis 3:** Nach jedem Trainingsdurchlauf werden beide Loss-Werte
  mit dem jetzt aktuellen Gewichtsstand neu gemessen.
- **Gespeichert:** Derselbe Dateipfad wird nach jeder Epoche ersetzt.
  Nach diesem Lauf enthält er die Gewichte nach Epoche 3, nicht drei Modelle.
- **Gesamtzeit:** 3,04, 5,99 und 8,95 Sekunden sind kumulierte Zeiten seit
  Beginn der Epochenschleife, einschliesslich Auswertung und Speicherung.
  Es sind nicht die einzelnen Epochendauern.

**Loss ist keine Prozentzahl und keine Anzahl falscher Tokens.**
Cross-Entropy bewertet, wie viel Wahrscheinlichkeit das Modell den
richtigen nächsten IDs zuweist. Kleinere Werte bedeuten bessere
Vorhersagen nach dieser Messung.

Hier sinkt der Trainings-Loss von ungefähr 11,19 auf 8,10 und der
Validierungs-Loss von ungefähr 11,19 auf 8,24. Die Gewichte passen also
nicht nur besser zu den Trainingsfenstern, sondern verbessern die
Vorhersagen auch auf dem zurückgehaltenen Bereich dieser Geschichte.
Der Verlauf zeigt noch nicht das Warnsignal „Training besser,
Validierung schlechter“.

**Erkenntnis:** Das Modell hat seine Vorhersagen messbar verbessert.
Damit ist weder allgemeines Sprachverständnis noch gute Textgenerierung
nachgewiesen. Die Validierung misst nächste Tokens bei vorgegebenen echten
Textfenstern; bei freier Generierung muss das Modell mit seinen eigenen
vorherigen Vorhersagen weiterarbeiten.

## 9. Passende Gewichte speichern und wieder laden

Nach jeder abgeschlossenen und ausgewerteten Epoche schreibt die Demo
die aktuellen Gewichte in die angegebene Burnpack-Datei. Der Ordner wird
bei Bedarf erstellt. Dieselbe Datei wird nach jeder Epoche ersetzt:
Sie enthält den **letzten**, nicht automatisch den besten Gewichtsstand.

Zum Schreiben wird zunächst eine temporäre Datei neben dem Ziel angelegt
und danach umbenannt. Dadurch bleibt ein vorheriger Checkpoint während des
Schreibens erhalten. Ein Abbruch mit `Ctrl+C` mitten in einer Epoche
verliert deren noch nicht gespeicherte Updates. Gibt es noch keinen
erfolgreich gespeicherten Checkpoint, gibt es auch nichts zu laden.
Bei Abbruch während des Speicherns kann eine Datei mit Endung
`.pending.bpk` liegen bleiben; sie ist nicht der reguläre Checkpoint.

**Gespeichert werden nur Modellgewichte.** Weder Optimizer-Zustand noch
Epochenzähler, Modellkonfiguration oder Tokenizer-Zuordnung stehen darin.
Ein erneuter `train`-Aufruf startet mit neuen Gewichten, nicht als
Wiederaufnahme. Diese Demo implementiert kein Training-Resume.

Training und Generierung verwenden dieselbe `config()`:
50'257 IDs, Kontext acht, Embedding-Grösse acht, zwei Köpfe, ein Block.
Der `generate`-Aufruf ist für Checkpoints dieser Demo gedacht.
Ein fehlender oder in den Gewichtsformen inkompatibler Checkpoint führt
zu einem Fehler. Eine passende Gewichtsform allein kann nicht beweisen,
dass extern erstellte Gewichte dieselbe Token-Zuordnung verwenden.

## 10. Aus den geladenen Gewichten Text generieren

Nach dem Training kann ein eigener Programmaufruf die Gewichte laden:

```bash
cargo run --release -p training --example ch12_full -- generate checkpoints/ch12-model.bpk "I HAD always" 16
```

Der letzte Wert ist die Anzahl neuer Tokens, nicht Wörter. Erlaubt sind
eins bis 64. Der Starttext muss mindestens ein Token ergeben.
Die Demo läuft ohne Autodiff und trainiert bei der Generierung nicht.

```rust
{{#include ../../../crates/training/examples/ch12_full.rs:text_generation}}
```

Der Ablauf verbindet jetzt Kapitel 1 und 11:

```text
Starttext → Gpt2Tokenizer::encode → Start-IDs
       → letzte höchstens acht IDs als Kontext
       → forward → Logits der letzten Position → argmax
       → neue ID anhängen → wiederholen
       → Gpt2Tokenizer::decode → Text
```

Die gemeinsame Generierungslogik liegt in `training::generation`.
Kapitel 11 verwendet dieselben Hilfsfunktionen für Kontext und
Token-Auswahl; seine bisherige Ausgabe bleibt erhalten.

Für den Modellaufruf werden IDs von `u32` nach `i32` konvertiert; die
50'257 GPT-2-IDs passen in diesen Bereich. Für den Tokenizer werden die
erzeugten IDs wieder nach `u32` konvertiert. Beide Konvertierungen sind
geprüft. Die vollständige Ausgabe enthält den Starttext und die Fortsetzung;
nur der Modellkontext wird abgeschnitten.

Einzelne GPT-2-Tokens können unvollständige UTF-8-Bytefolgen enthalten.
Wenn die gesamte generierte Folge kein gültiges UTF-8 ergibt, meldet
`decode` einen Fehler. Die Demo zeigt davor die IDs; sie ersetzt ungültige
Bytes nicht stillschweigend durch scheinbar korrekten Text.

Auch nach ganzen Epochen kann die Ausgabe unsinnig sein oder sich
wiederholen. `argmax` wählt immer den höchsten Logit und erzeugt keine
abwechslungsreiche Zufallsstichprobe. Ein längerer Trainingslauf auf einem
kurzen Text hebt die Grenzen dieses winzigen Modells nicht auf.

### Die Generierungsausgabe einordnen

Die erste Ausgabezeile zeigt die IDs des Starttexts **und** der neu
erzeugten Tokens. Die nächste zeigt dieselbe Folge decodiert als Text.
`"I HAD always"` wird von dir vorgegeben; diesen Anfang hat das Modell
nicht selbst erfunden.

Ein **nur zur Erklärung erfundenes** Beispiel einer repetitiven Ausgabe
wäre:

```text
Text (keine Qualitätsgarantie):
I HAD always the the the the
```

Das ist keine gemessene Ausgabe und kein erwartetes Ergebnis des
16-Token-Aufrufs. Es illustriert die Aufteilung:

```text
Vorgegebener Starttext: I HAD always
Erzeugte Fortsetzung:   the the the the
```

Eine solche Wiederholung bedeutet nicht, dass das Laden gescheitert ist.
Ein winziges Modell kann einem häufigen Token immer wieder den höchsten
Logit geben. `argmax` nimmt es jedes Mal erneut. Sobald eigene Vorhersagen
im Kontext stehen, sieht das Modell zudem andere Folgen als die echten
Textfenster bei der Loss-Messung.

Prüfe an deiner eigenen Ausgabe zuerst: Ist der Starttext erhalten?
Stehen dahinter neue Tokens? Ist die Fortsetzung sprachlich plausibel
oder repetitiv? **Ein funktionierender Generierungsablauf und gute
Sprachqualität sind zwei verschiedene Ergebnisse.**

Die `16` im Befehl zählt neue Token-IDs. Wortteile und Leerzeichen können
eigene Tokens sein, deshalb müssen daraus nicht 16 Wörter entstehen.
Bei einem Decodierungsfehler stehen die IDs trotzdem in der Ausgabe;
ein Textblock fehlt dann, weil die gesamte Bytefolge kein gültiges UTF-8
ergibt.

## 11. Den vollständigen Ablauf prüfen

```bash
cargo test -p training --lib --example ch11 --example ch12 --example ch12_full --test ch10
```

Die zusätzliche Demo prüft die Fensterzahlen des gesamten Beispieltextes,
Epochen- und Generierungsgrenzen, inkompatible Checkpoints sowie
Training, Speichern, Laden und identische Generierung vor/nach dem Laden.
Der Trainingstest verwendet einen kurzen eigenen Testtext, nicht einen
langen Trainingslauf auf der ganzen Geschichte.

Der Übergang ist jetzt hergestellt: **echter Text → echte IDs →
Epochen und Validierung → gespeicherte Gewichte → generierter Text**.
Im nächsten Kapitel geht es um Auswertung und Overfitting, bevor wir aus
den Loss-Verläufen grössere Schlüsse ziehen.
