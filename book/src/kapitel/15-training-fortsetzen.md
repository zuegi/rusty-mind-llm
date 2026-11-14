# 15. Training unterbrechen und fortsetzen

## Lernziel

Kapitel 10 speichert Modellgewichte. Kapitel 14 erklärt, warum Adam
zusätzlich ein Gedächtnis braucht. Jetzt setzen wir einen Trainingslauf
nach dem Beenden des Programms am nächsten Update fort.

**Resume** bedeutet hier Wiederaufnahme desselben Laufs: gleiche Gewichte,
gleicher Adam-Zustand, gleicher nächster Batch und gleiche nächste Lernrate.
Ein Neustart mit geladenen Gewichten und einem neuen Adam wäre dagegen
ein anderer Trainingsverlauf.

Die separate Demo `crates/training/examples/ch15.rs` verwendet wieder
ein kleines GPT-Modell und echte GPT-2-Token-IDs. `ch13` und `ch14`
behalten ihr bisheriges Verhalten; sie bekommen keinen Resume-Modus.

## 1. Einen Lauf über zwei Prozesse fortsetzen

Im Projektverzeichnis:

```bash
# Neu starten, nach Update 5 speichern und Programm beenden.
cargo run --release -p training --example ch15 -- train checkpoints/ch15-teil 5

# In einem neuen Prozess laden und Updates 6 bis 12 ausführen.
cargo run --release -p training --example ch15 -- resume checkpoints/ch15-teil checkpoints/ch15-fortgesetzt 12
```

Die letzte Zahl ist immer die **absolute Ziel-Update-Nummer**, nicht die
Anzahl zusätzlicher Updates. Der gesamte Lernratenplan steht schon beim
Start fest: drei Epochen mit vier Batches ergeben zwölf Updates.
Update 5 ist deshalb kein neues Planende; die Lernrate wird nicht für
den ersten Teil auf fünf Updates zusammengedrückt.

Jeder Zielordner muss neu sein. Ein vorhandener Checkpoint wird weder
beim Neustart noch beim Resume überschrieben. Ungültige Ziele, etwa
Update 3 nach einem Checkpoint bei Update 5 oder ein Ziel über zwölf,
führen zu einem Fehler.
Den Quellordner bis zum Ende der Wiederaufnahme unverändert behalten:
Burn kann Tensorinhalte erst bei ihrer Verwendung aus den Dateien lesen.

Die Demo nutzt die ersten 65 Tokens von `data/the-verdict.txt`:
acht Trainingsfenster mit Kontextlänge acht und Batch-Grösse zwei.
Embedding-Dimension acht, zwei Köpfe und ein Transformer-Block halten
das Modell klein. Das ist ein begrenzter Mechanikversuch, kein
Qualitätsnachweis. Eine Validierung und Bestmodell-Auswahl wie in
Kapitel 13 sind hier bewusst nicht Teil des Trainingsablaufs.

## 2. Was im Checkpoint liegt

Ein Checkpoint ist hier ein Verzeichnis mit drei zusammengehörigen Dateien:

```text
ch15-teil/
  model.bpk       Modellgewichte einschliesslich Parameter-IDs
  optimizer.bpk   Adams gespeicherte Verläufe und Schrittzähler
  state.json      Konfiguration, Trainingsposition, Daten und Prüfsummen
```

Die beiden `.bpk`-Dateien verwenden Burns Burnpack-Format.
`state.json` ist eine lesbare Beschreibung des Laufs. Sie speichert
Modellkonfiguration, Adam-Konfiguration, Batch-Grösse, Epochenzahl,
Seed, Lernratenplan, verarbeitete Update-Anzahl und die tatsächlich
verwendeten Token-IDs. Beim Resume wird die ursprüngliche Textdatei
deshalb nicht erneut gelesen oder tokenisiert.

Das ist für unsere wenigen Tokens praktisch. Grosse Datensätze würde
man nicht vollständig in diese Datei kopieren: Dort bräuchte man
eine überprüfbare Datenversion und eine passende Positionsangabe.

```rust
{{#include ../../../crates/training/examples/ch15.rs:state}}
```

`Training` enthält die laufenden Rust-Objekte. `Snapshot` beschreibt
die zusätzlich zu den beiden Burnpack-Dateien gespeicherten Angaben.
`Serialize` und `Deserialize` stammen aus Serde und erlauben die
Umwandlung zwischen diesen Datenstrukturen und JSON.

**Parameter-IDs** verbinden Adams gespeicherte Verläufe mit den
zugehörigen Modellparametern. Gleiche Tensorformen allein reichen
nicht. Deshalb müssen Gewichte und Optimizer-Zustand aus demselben
Speichervorgang geladen werden; Burns Modell-Records erhalten diese IDs.

## 3. Wo geht es innerhalb einer Epoche weiter?

`completed_updates` zählt bereits abgeschlossene Updates. Die Zahl der
vollständigen Batches je Epoche ist aus Daten und Batch-Grösse bekannt.
Division ergibt die bisher abgeschlossenen Epochen; der Rest ergibt
die Position innerhalb der nächsten beziehungsweise laufenden Epoche.

Bei vier Batches je Epoche und fünf abgeschlossenen Updates gilt:
Epoche 1 ist abgeschlossen, der erste Batch von Epoche 2 ebenfalls.
Als Nächstes folgt der zweite Batch von Epoche 2 — nicht erneut deren
erster Batch.

```rust
{{#include ../../../crates/training/examples/ch15.rs:resume_batches}}
```

`Dataset::epoch` mischt mit einem eigenen, frisch aus dem Epochen-Seed
erzeugten Zufallsgenerator. Mit denselben Tokens, derselben Batch-Grösse
und `seed + epoch` entsteht wieder dieselbe Reihenfolge. Bereits
verarbeitete Batches werden nicht noch einmal trainiert.

Für diese kleine Demo wird die Reihenfolge bei jedem Update erneut
erzeugt. Das spart zusätzlichen gespeicherten Laufzeitzustand, wäre für
grosse Datensätze aber unnötig aufwendig.

**Warum reicht hier ein Seed?** Das Modell verwendet kein Dropout und
zieht während seiner Trainingsupdates keine weiteren Zufallszahlen.
Die Batch-Mischung beginnt je Epoche mit einem lokalen Generator.
Wir müssen deshalb keinen fortgeschrittenen globalen Zufallszustand
wiederherstellen. Mit Dropout, zufälligen Datenänderungen oder anderen
zufallsabhängigen Operationen würde diese Vereinfachung nicht reichen.

## 4. Adam und Lernratenplan gemeinsam fortsetzen

Der Checkpoint enthält Adams beide Verläufe und den Schrittzähler für
jeden trainierten Parametertensor. Beim Laden erzeugen wir Adam mit
der gespeicherten Konfiguration und laden anschliessend seinen Zustand.

```rust
{{#include ../../../crates/training/examples/ch15.rs:load}}
```

Die zusätzliche Prüfung in `load_optimizer` erwartet insgesamt doppelt
so viele Moment-Tensoren wie Modellparameter. Sie erkennt damit etwa
einen leeren oder teilweise rekonstruierten Adam-Zustand, prüft aber
nicht die Zuordnung der Parameter-IDs. Ein eigens zusammengestellter
Checkpoint mit passenden Prüfsummen und gleich vielen Moment-Tensoren
für fremde IDs kann diese Prüfung bestehen. Beim nächsten Update würde
Adam für die betroffenen Modellparameter unbemerkt neue Verläufe anlegen.
Die Prüfung ist deshalb keine vollständige Garantie für einen
konsistenten Optimizer-Zustand; Modell und Optimizer müssen weiterhin
aus demselben Speichervorgang stammen.
Die Demo verwendet normales Adam, ohne AMSGrad, Weight Decay oder Clipping.

Der Lernratenplan verwendet dieselbe gemeinsame Implementierung
`WarmupDecay` wie Kapitel 14:

```rust
{{#include ../../../crates/training/src/schedule.rs}}
```

Der Plan selbst hat kein veränderliches Gedächtnis. Seine Konfiguration,
die Gesamtzahl der Updates und `completed_updates + 1` bestimmen die
nächste Lernrate vollständig. Warmup beginnt beim Resume nicht erneut.

```rust
{{#include ../../../crates/training/examples/ch15.rs:resume_update}}
```

Erst nach einem erfolgreichen Update erhöhen wir den Zähler. Ein Checkpoint
liegt immer **zwischen** zwei Updates; einen halb ausgeführten
Backward-Durchlauf oder Autodiff-Graphen speichern wir nicht.

## 5. Zusammengehörig speichern und geprüft laden

```rust
{{#include ../../../crates/training/examples/ch15.rs:save}}
```

Der neue Ordner wird exklusiv angelegt. Zuerst werden beide Burnpack-Dateien
geschrieben und mit `sync_all()` an das Betriebssystem zum dauerhaften
Speichern übergeben. Danach berechnen wir je Datei eine SHA-256-Prüfsumme,
also einen aus ihrem Inhalt berechneten Fingerabdruck. `state.json` wird
zuletzt geschrieben und ebenfalls synchronisiert.

Beim Laden müssen alle drei Dateien vorhanden und lesbar sein.
`read_snapshot` prüft Format-/Kompatibilitätskennung, Konfiguration,
Update-Zähler und Prüfsummen, bevor die Records geladen werden.
Fehlende, beschädigte oder versehentlich vertauschte Record-Dateien führen
zu einem Fehler. Es gibt keinen automatischen Neustart als Ersatz.

Bricht das Schreiben vorher ab, kann ein unvollständiger neuer Ordner
zurückbleiben. Dieser wird nicht als gültiger Checkpoint geladen.
Der vorige Checkpoint bleibt unverändert. Das Verfahren ist keine
allgemeine Garantie gegen Stromausfälle oder Dateisystemfehler.
Prüfsummen sind auch keine Signatur: Nur eigene, vertrauenswürdige
Checkpoints laden und `state.json` nicht von Hand verändern.

Ein unerwarteter Abbruch während des Trainings setzt beim **letzten
gespeicherten Checkpoint** an. Später ausgeführte, aber nicht gespeicherte
Updates gehen verloren und werden bei der Wiederaufnahme erneut gerechnet.

## 6. Ist das wirklich derselbe Trainingslauf?

Ein ununterbrochener Vergleichslauf:

```bash
cargo run --release -p training --example ch15 -- train checkpoints/ch15-durchgehend 12
```

Die tatsächlichen Prozessläufe lieferten nach der Unterbrechung bei
Update 5 dieselben angezeigten Lernraten und Loss-Werte wie der
ununterbrochene Lauf. Beispielsweise:

```text
Update 6: Lernrate 0.00077500, Loss 11.448120
Update 12: Lernrate 0.00010000, Loss 11.520178
```

Gleiche gerundete Loss-Ausgaben allein wären aber kein ausreichender
Nachweis. Die automatisierten Tests verwenden deshalb ein winziges
Fünf-Token-Vokabular und vergleichen **alle Modellparametertensoren
exakt**, nicht nur den Loss:

```bash
cargo test -p training --example ch15
```

Unterbrechungen nach Update 3, 4 und 5 prüfen den Warmup sowie Positionen
vor, auf und nach einer Epochengrenze. Der nächste Batch und die Gewichte
nach insgesamt zwölf Updates stimmen mit dem ununterbrochenen Lauf
überein. Ein Gegenversuch mit neu initialisiertem Adam verändert dagegen
schon das nächste Update. Weitere Tests prüfen beschädigte und
unvollständige Checkpoints, inkompatible Formate, ungültige Update-Ziele
und den Schutz vor Überschreiben.

Diese Aussage gilt für die hier getestete deterministische CPU-Ausführung
mit derselben Softwareumgebung. Die Kompatibilitätskennung verhindert
das versehentliche Laden anderer Checkpoint-Formate; sie verspricht keine
Portabilität über beliebige Burn-/Rand-Versionen, Hardware oder Backends.

## Das nehmen wir mit

Ein Gewichte-Checkpoint reicht zum Anwenden eines Modells.
Ein Trainings-Checkpoint beschreibt zusätzlich, **wie und wo weitergelernt
wird**. Die kleine Resume-Demo führt diese Teile zusammen, ohne den
Optimizer-Vergleich aus Kapitel 14 oder die Bestmodell-Auswahl aus
Kapitel 13 umzubauen.
