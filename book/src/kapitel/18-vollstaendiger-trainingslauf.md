# 18. Ein vollständiger Trainingslauf

## Lernziel

Bisher haben wir die Bausteine einzeln gebaut:
Validierung und Bestmodell ([Kapitel 13](13-auswertung-und-overfitting.md)),
Adam und Lernratenplan ([Kapitel 14](14-bessere-optimierung.md)),
Resume ([Kapitel 15](15-training-fortsetzen.md)) sowie AdamW und Dropout
([Kapitel 17](17-regularisierung.md)). Die Demo
`crates/training/examples/ch18.rs` setzt sie zu **einem** Lauf zusammen:

```text
Update (AdamW, Lernratenplan, Dropout)
  → am Epochenende Validierung mit festen Gewichten
  → bei neuem Bestwert: Bestmodell ersetzen
  → Checkpoint: letzter Stand und Bestmodell getrennt
  → Resume: Stand, Datenposition, Dropout-Maske und Bestwert fortsetzen
```

Kapitel 16 (Temperatur und Sampling) gehört zur Inferenz und kommt hier
nicht vor. Das Modell bleibt klein; die Demo zeigt den **Ablauf**, keinen
Qualitätsgewinn.

## 1. Aufruf

```bash
# Zehn Epochen = 40 Updates; nach Update 20 speichern.
cargo run --release -p training --example ch18 -- train checkpoints/ch18-teil 20

# Neuer Prozess: bis Update 40 fortsetzen.
cargo run --release -p training --example ch18 -- resume checkpoints/ch18-teil checkpoints/ch18-fertig 40
```

Die Zahl ist wie in Kapitel 15 die **absolute** Ziel-Update-Nummer; der
Lernratenplan umfasst von Beginn an alle 40 Updates. Zielordner müssen
neu sein; vorhandene Checkpoints werden nie überschrieben. Den Quellordner
bis zum Ende der Wiederaufnahme behalten: Burn liest Tensorinhalte erst
bei Verwendung aus den Dateien.

Daten: Wie in Kapitel 17 wird zuerst getrennt (`split_tokens`, 10 %
Validierung), dann gefenstert. Die Demo nutzt acht Trainings- und
zwei Validierungsfenster von `data/the-verdict.txt`, Batch-Grösse zwei,
AdamW mit `weight_decay = 0.1`, Dropout `0.1`, Lernrate 0.005 mit vier
Warmup-Updates und Absenkung auf 10 %.

## 2. Der Trainingszustand

```rust
{{#include ../../../crates/training/examples/ch18.rs:state}}
```

`Snapshot` ergänzt Kapitel 15 um beide Datensätze (Training und Validierung),
die AdamW-Konfiguration, das Dropout in den Modelleinstellungen und
`best`: den bisher besten Validierungs-Loss samt Update-Nummer. Die
Token-IDs stehen im Snapshot, die Textdatei wird beim Resume nicht
erneut gelesen.

## 3. Update, Validierung und Bestmodell

```rust
{{#include ../../../crates/training/examples/ch18.rs:update}}
```

Der Zähler wächst erst nach einem erfolgreichen Update. Ein nicht
endlicher Loss bricht ab, bevor irgendetwas gespeichert wird.

**Dropout und Resume.** Kapitel 17 hat offen gelassen, wie die
Zufallsmasken fortgesetzt werden. Hier gilt die dort genannte Alternative:
Vor jedem Update setzen wir den Zufallsgenerator auf `seed + Update-Nummer`.
Die Masken eines Updates hängen damit nur von dieser Nummer ab, nicht von
der Vorgeschichte. Ein neuer Prozess erzeugt bei Update 21 dieselben Masken
wie der durchgehende Lauf. Der Generatorzustand muss deshalb nicht
gespeichert werden. Zusätzlich erzeugt `Training::new` die Modellparameter
sofort; Burn würde sie sonst erst im ersten Update initialisieren und dabei
Zufallszahlen verbrauchen, die ein Resume nicht kennt.

```rust
{{#include ../../../crates/training/examples/ch18.rs:validate}}
```

Validiert wird **am Ende jeder Epoche** mit `model.valid()`:
Dropout ist aus, und es entsteht kein Autodiff-Graph. Das Bestmodell
wird nur bei einem **echt kleineren** Validierungs-Loss ersetzt; bei
Gleichstand bleibt das frühere. Zwischen den Epochengrenzen gibt es
keine Validierung; wer mitten in einer Epoche stoppt, setzt am nächsten
Batch fort und validiert am Epochenende.

## 4. Zwei Checkpoints in einem Ordner

```text
ch18-teil/
  last/   model.bpk  optimizer.bpk  state.json   ← zum Weitertrainieren
  best/   model.bpk  best.json                   ← bester Validierungsstand
```

```rust
{{#include ../../../crates/training/examples/ch18.rs:save}}
```

- **`last/`** ist der Trainingsstand nach dem letzten Update: Gewichte,
  AdamW-Zustand, Datenposition. Nur damit lässt sich exakt fortsetzen.
- **`best/`** enthält ausschliesslich Gewichte des besten Validierungsstands
  (ohne Dropout-Zustand, ohne Optimizer) plus `best.json` mit
  Validierungs-Loss, Update-Nummer, Modelleinstellungen und Prüfsumme. Es
  ist das Modell zum Anwenden und liegt in der Regel **hinter** dem letzten
  Stand zurück.
- Gibt es noch keine Epochengrenze, existiert kein `best/`.

Beim Laden prüfen wir wie in Kapitel 15 die SHA-256-Summen von `last/`
und zusätzlich, dass `best/best.json` mit dem Eintrag `best` in
`state.json` übereinstimmt und zur Prüfsumme der `best/model.bpk` passt.
Ein fehlendes, verändertes oder zu einem anderen Lauf gehörendes
Bestmodell ist ein Fehler, kein stiller Neustart. Wie in Kapitel 15 sind
Prüfsummen keine Signatur und kein Schutz vor Stromausfällen; nur eigene,
vertrauenswürdige Checkpoints laden.

```rust
{{#include ../../../crates/training/examples/ch18.rs:load}}
```

**Der Bestwert bleibt beim Resume erhalten.** `load` stellt `best` und das
Bestmodell wieder her. Ein späterer Lauf ersetzt sie nur, wenn die
Validierung den *gespeicherten* Bestwert unterbietet. Würde der Resume
dagegen mit leerem Bestwert starten, überschriebe die erste schlechtere
Epoche das bessere Modell.

## 5. Beispielausgabe

Ein durchgehender Lauf und die Folge „bis 20, neuer Prozess, bis 40“ liefern
dieselben angezeigten Lernraten, Loss-Werte und Epochenzeilen, zum Beispiel:

```text
Epoche 1: Training 10.9853, Validierung 10.7335  <- neues Bestmodell
Epoche 5: Training 10.3954, Validierung 10.5870  <- neues Bestmodell
Epoche 10: Training 9.9914, Validierung 10.4521  <- neues Bestmodell
```

In diesem kurzen Lauf mit kleinem Modell und kleiner Lernrate fällt auch
der Validierungs-Loss in jeder Epoche; die Demo zeigt deshalb **keinen**
Fall, in dem das Bestmodell hinter dem letzten Stand zurückbleibt. Dieses
Verhalten prüft ein Test (Abschnitt 6). Den Loss-Bereich um 10 teilt das
Modell mit zufälligen Gewichten (ln 50257 ≈ 10.8): Es hat erst wenig gelernt.

## 6. Was prüfen wir?

```bash
cargo test -p training --example ch18
```

Die Tests verwenden ein Fünf-Token-Vokabular mit Dropout `0.2`:

- Unterbrechung nach Update 3, 4, 5 und 8 und Fortsetzung bis 12 ergibt
  **exakt dieselben Modellgewichte, Bestwerte und Bestmodell-Gewichte**
  wie der ununterbrochene Lauf. Das gilt für die deterministische
  CPU-Ausführung in derselben Umgebung, nicht für beliebige Backends.
- Ein künstlich unschlagbarer Bestwert vor dem Speichern bleibt nach dem
  Resume samt Gewichten erhalten.
- `last/` und `best/` sind getrennt; das gespeicherte Bestmodell liefert
  beim Laden denselben Validierungs-Loss wie gemeldet.
- Beschädigte, fehlende oder nicht eingetragene Bestmodelle und ein
  beschädigter Optimizer-Zustand sind Fehler; ungültige Ziele und
  Überschreiben ebenso.
- Weight Decay verändert die Gewichte und steht nach dem Laden unverändert
  in der gespeicherten Optimizer-Konfiguration.
- Mit `valid()` liefert das Modell trotz Dropout wiederholbare Ausgaben.

## Grenzen

- Ein Resume mit Dropout gilt hier nur, weil die Maske aus Seed und
  Update-Nummer abgeleitet wird; andere Zufallsquellen müssten ähnlich
  behandelt werden.
- Die Validierung ist winzig (zwei Fenster). Bestmodell-Auswahl auf so
  wenigen Daten ist Mechanik, kein Generalisierungsnachweis. Das
  Bestmodell ist zudem am Validierungsdatensatz „ausgesucht“; für eine
  ehrliche Schlussbewertung bräuchte man einen unberührten Testbereich.
- Kein Early Stopping, kein Gradient Clipping, kein Zeitbudget.

## Das nehmen wir mit

Ein vollständiger Lauf trennt zwei Dinge: den **letzten Stand** zum
Weitertrainieren und den **besten Validierungsstand** zum Anwenden.
Beim Resume gehört der bisherige Bestwert zum Trainingszustand, und
Dropout bleibt reproduzierbar, wenn die Zufallsmaske aus der
Update-Nummer abgeleitet wird.
