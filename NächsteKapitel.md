Nach Kapitel 9 sind **Checkpoints und Textgenerierung** logisch. Sinnvolle Reihenfolge:

## Kapitel 10: Checkpoints speichern und laden

**Lernziel:** Modellzustand dauerhaft sichern.

Ablauf:

```text
Training
  → Modellgewichte + Optimizer-Zustand speichern
  → Programm beenden
  → Checkpoint laden
  → Training oder Auswertung fortsetzen
```

Wichtige Inhalte:

- Warum Modellgewichte allein nicht immer reichen
- Unterschied zwischen Modell-Checkpoint und Optimizer-Zustand
- Burn-Records zum Speichern/Laden
- Checkpoint nach jeder Epoche oder bei bestem Validierungs-Loss
- Wiederaufnahme nach Programmabbruch
- Prüfen, dass geladene Gewichte dieselben Ergebnisse liefern

**Warum zuerst?** Textgenerierung sollte auf einem gespeicherten, reproduzierbaren Modell aufbauen.

## Kapitel 11: Textgenerierung

**Status (07.11.2026): abgeschlossen und in `main`.** Kapiteltext und Demo
`crates/training/examples/ch11.rs` sind mit PR #12 integriert. Die Demo
generiert mit `argmax`, begrenzt den Kontext und kann gespeicherte Gewichte
laden. Das Training verwendet weiterhin künstliche Token-IDs; die kleine
Worttabelle ist kein GPT-2-Tokenizer.

**Lernziel:** Aus bisherigem Text neue Token vorhersagen.

Ablauf:

```text
Start-Tokenfolge
  → GPT berechnet Logits
  → letzte Position auswählen
  → nächstes Token wählen
  → Tokenfolge verlängern
  → wiederholen
```

Schritte:

1. Logits der letzten Position auswählen
2. `argmax`: wahrscheinlichstes Token wählen
3. Token an Eingabe anhängen
4. Kontextlänge begrenzen
5. Mehrfach wiederholen
6. Token-IDs wieder in Text umwandeln

Danach optional:

- Temperatur
- Top-k-Sampling
- Zufallsseed
- Unterschied zwischen deterministischer und zufälliger Generierung
- Stopp-Token

Zuerst nur `argmax`; Sampling erst als Erweiterung. So bleibt der Grundablauf verständlich.

## Kapitel 12: Echter Text und Tokenizer

**Status (07.11.2026): abgeschlossen; zur Integration in `main` bereit.**
`book/src/kapitel/12-echter-text-und-tokenizer.md` und
`crates/training/examples/ch12.rs` verbinden den vorhandenen GPT-2-Tokenizer
mit getrennten echten Trainings- und Validierungsdaten. Die Demo verwendet
höchstens acht Trainingsfenster, zwei Validierungsfenster und drei Updates.
`crates/training/examples/ch12_full.rs` ergänzt ausdrücklich gestartete
volle Epochen (eins bis drei), Auswertung aller Fenster der getrennten
Bereiche, Speichern/Laden passender Modellgewichte und Textgenerierung.
Ein längerer Trainingslauf auf der gesamten Geschichte wurde nicht
automatisch ausgeführt. Tests verwenden dafür einen kurzen Testtext.
Die Integration in `main` erfolgt über einen Pull Request.

Kapitel 9 nutzt künstliche IDs. Kapitel 12 ergänzt:

- `data/the-verdict.txt` tokenisieren
- echtes Vokabular verwenden
- Training und Validierung mit echten Token-Daten
- Speicher- und Laufzeitgrenzen erklären
- Unterschied zwischen Lernbeispiel und brauchbarem Sprachmodell zeigen

Dieses Kapitel sollte **nach** Checkpoints und Generierung kommen: Erst Mechanik beweisen, dann echten Text einsetzen.

**Einstieg durchgeführt:** Die vorhandene Demo aus Kapitel 2 mit
`data/the-verdict.txt` wurde ausgeführt; der Weg vom Text zu Token-IDs,
verschobenen Zielen und der Logit-Form wurde gemeinsam besprochen.
`Gpt2Tokenizer` verwendet bereits GPT-2 (`r50k_base`) mit 50'257
Vokabular-Einträgen. Diese Grösse ist nicht die Anzahl Tokens im Text.
Der Token-Strom wird vor der Fensterbildung in Training und Validierung
geteilt; kein Fenster darf über diese Grenze reichen.

Der freigegebene kleine Trainingsversuch verwendet Kontextlänge acht,
Batch-Grösse zwei und höchstens drei Updates. Ein Zeitbudget von 30 Sekunden
wird zwischen Updates geprüft; laufende Updates und Auswertung dürfen
länger dauern. Das ist keine harte Prozess-Laufzeitgrenze. Das grössere
Vokabular vergrössert Embedding, Ausgabeprojektion und Logits gegenüber den
fünf IDs aus Kapitel 9. Die Demo zeigt verstrichene Trainingszeit und die
Grösse des Logit-Tensors, nicht den gesamten RAM-Bedarf. Allgemeingültige
Speicher- oder Laufzeitwerte lassen sich daraus nicht ableiten.
Ein kleines Modell mit einem einzelnen kurzen Text dient dem Lernen,
nicht als allgemein brauchbares Sprachmodell.

Der vollständige Epochenlauf hat keinen Zeitabbruch. Seine Checkpoints
enthalten nur Modellgewichte, keinen Optimizer-Zustand oder Epochenzähler.
Erneutes Training startet neu; Training-Resume ist nicht umgesetzt.
Generierung lädt die feste passende Konfiguration und decodiert GPT-2-IDs.

## Kapitel 13: Auswertung und Overfitting

**Status (10.11.2026): abgeschlossen und in `main` (PR #14).**
`book/src/kapitel/13-auswertung-und-overfitting.md` und
`crates/training/examples/ch13.rs` trainieren bis zu 20 Epochen auf echtem
Text, messen je Epoche Trainings- und Validierungs-Loss samt Perplexity
(`exp(loss)`), speichern nur bei strikt kleinerem Validierungs-Loss und
melden ein Overfitting-Warnsignal. Epoche 0 zählt als Ausgangskandidat
für den besten Checkpoint. Das Buch enthält echte Ausgaben, beide
Loss-Kurven sowie ausgewertete Vergleiche von 5/20 Epochen bei Lernrate
0,05 und 20 Epochen bei 0,02. Lernraten lassen sich über
`--learning-rate` vergleichen. In diesen Läufen war Epoche 16 bei 0,05
der beste Stand (Validierungs-Loss 6,6642, Perplexity 783,9).
Tests verwenden einen kurzen Text; die drei dokumentierten Echttext-Läufe
wurden für den Kapitelabschluss ausgeführt. Grenzen, unabhängige
Testdaten und der Übergang zu Kapitel 14 sind erklärt. Laden und
Generierung mit dem besten Checkpoint sind anhand einer echten Ausgabe
beschrieben. Der Lernpfad führt vom Trainingslauf über Auswertung und
Bestmodell-Auswahl bis zur Verwendung der gespeicherten Gewichte.
Kapitel 12 blieb unverändert. Der Abschluss wurde mit PR #14 integriert.

Behandelte Themen:

- Trainings-Loss und Validierungs-Loss visualisieren
- Perplexity aus dem Loss erklären
- Overfitting erkennen
- Epochenzahl und Lernrate vergleichen
- Bestes Modell anhand Validierungs-Loss speichern

## Kapitel 14: Bessere Optimierung

**Status (14.11.2026): abgeschlossen und in `main` (PR #15), Review durchgeführt.**
`book/src/kapitel/14-bessere-optimierung.md` erklärt SGD, Momentum, Adam,
Lernratenpläne pro Update, Warmup, Clipping-Varianten und den nötigen
Zustand für eine vollständige Trainingswiederaufnahme.
`crates/training/examples/ch14.rs` verwendet die vorhandenen Trainingsfunktionen
für fünf Vergleiche: SGD, Momentum, Adam, Adam mit Warmup/linearer Absenkung
und zusätzlich Norm-Clipping je Parametergradiententensor. Gleiche
Ausgangsgewichte, gleiche Batches und zwölf Updates je Variante machen
den begrenzten Vergleich nachvollziehbar. Alle fünf Echttext-Läufe wurden
ausgeführt; `book/src/ausgaben/ch14-vergleich.txt` enthält ihre echten
Messwerte, das Kapitel die Rust-Beispiele und deren Einordnung.
Die Demo nutzt nur acht Trainings- und zwei Validierungsfenster und
speichert keine Checkpoints. Die separate Resume-Demo folgt in Kapitel 15.

Themen:

- Adam statt SGD
- Momentum
- Lernratenänderung
- Gradient Clipping
- Warmup
- Warum Optimizer-Zustand gespeichert werden muss

Nicht vor Checkpoints behandeln. Kapitel 9 soll erst den grundlegenden Ablauf festigen.

## Kapitel 15: Training unterbrechen und fortsetzen

**Status (14.11.2026): abgeschlossen und in `main` (PR #15), Review durchgeführt.**
`crates/training/examples/ch15.rs` speichert Modell- und Adam-Zustand
als Burnpack-Records sowie Konfiguration, Token-IDs, Lernratenplan,
Seed und Update-Zähler als JSON. Neue Checkpoint-Ordner werden nicht
überschrieben; Prüfsummen prüfen die zusammengehörigen Record-Dateien.
Ein neuer Prozess setzt auch innerhalb einer Epoche beim nächsten Batch
fort. Die kleine Demo verwendet acht Echttext-Fenster und zwölf Updates,
ohne Validierung oder Bestmodell-Auswahl. `ch13` und `ch14` bleiben
eigenständige Demos ohne Resume-Modus.

Tests vergleichen alle Modellgewichte nach Unterbrechungen vor, auf und
nach der Epochengrenze exakt mit einem ununterbrochenen CPU-Lauf.
Ein neuer Adam ohne gespeicherten Zustand liefert dagegen ein anderes
nächstes Update. Zwei separate Prozessläufe auf echten Tokens zeigen
dieselben angezeigten Lernraten und Loss-Werte wie der durchgehende Lauf.
Lernratenplan und Berechnung bleiben mit Kapitel 14 gemeinsam.

## Ausblick: Vom eigenen Training zum lokalen Chatbot

**Kapitel 16 bis 25 sind geplant, noch nicht umgesetzt.**
Der Lernpfad behält die Vertiefung des eigenen Trainings bei und ergänzt
danach den Import vortrainierter Gewichte sowie einen lokal ansprechbaren
Chatbot. Ziel ist ein vollständiger, nachvollziehbarer Weg, nicht das
Versprechen guter Antworten.

## Kapitel 16: Temperatur und Sampling

Temperatur, Top-k, Top-p, Zufallsseed und Stoppbedingungen erweitern die
bisherige Generierung mit `argmax`. Gleicher Checkpoint und gleicher Prompt
machen die Verfahren vergleichbar. Sampling verändert die Auswahl und
Vielfalt der Antworten, nicht die gelernte Modellqualität.

## Kapitel 17: Regularisierung gegen Overfitting

AdamW, Weight Decay und Dropout werden anhand von Trainings- und
Validierungs-Loss untersucht. Dropout braucht einen expliziten
Trainings-/Auswertungsmodus; seine Auswirkungen auf reproduzierbares
Resume werden erklärt.

## Kapitel 18: Ein vollständiger Trainingslauf

Die Bausteine aus Kapitel 13 bis 15 zusammenführen: Adam, Lernratenplan,
Validierung, Bestmodell-Auswahl und Resume. Letzten Trainingsstand und
besten Validierungsstand getrennt speichern; beim Resume auch den
bisherigen Bestwert erhalten.

## Kapitel 19: Trainingsdaten und Datenqualität

Mehrere Dokumente, Dokumentgrenzen, Duplikate und Datenlecks behandeln.
Training, Validierung und unabhängigen Testdatensatz trennen.
Eine überprüfbare Datenversion und Trainingsposition ersetzen bei
grösseren Datensätzen die vollständige Tokenliste im Checkpoint.

## Kapitel 20: Laufzeit und Speicher verstehen

Engpässe messen, Batch- und Kontextgrösse vergleichen und Gradient
Accumulation einführen. GPU und reduzierte Präzision erst darauf aufbauend
betrachten; Grenzen deterministischer Ergebnisse erläutern.

## Kapitel 21: Unser Modell mit GPT-2 abgleichen

Architekturunterschiede sichtbar machen und gezielt schliessen:
finale LayerNorm, Aktivierungsfunktion, Biases, Weight Tying und
Modellkonfiguration. Das aktuelle Modell ist GPT-2-artig, aber noch
kein direkt kompatibler GPT-2-Nachbau. Noch keine fremden Gewichte laden.

## Kapitel 22: Vortrainierte GPT-2-Gewichte laden

Die kleinste GPT-2-Variante importieren: Tensoren zuordnen, Formen und
nötige Transpositionen prüfen. Tokenizer und Logits gegen eine
Referenzimplementierung vergleichen. Plausibler Text allein beweist
keinen korrekten Gewichtsimport.

## Kapitel 23: Mit dem Modell im Terminal chatten

Eine Eingabeschleife mit Rollen als Textvorlage, Gesprächsverlauf,
Kontextbudget und begrenzten Antworten bauen. Zunächst ohne
Weboberfläche bleibt das Modellverhalten sichtbar. Eigene Mini-Gewichte
und vortrainiertes GPT-2 über denselben Chatpfad vergleichen:
korrekte Mechanik gegenüber durch Vortraining erworbener Fähigkeit.

## Kapitel 24: Lokaler Modelldienst und Chatoberfläche

Inferenz hinter eine lokale HTTP-API stellen und eine einfache
Chatoberfläche anbinden. Generierung begrenzen sowie Abbruch und
Fehler behandeln; Streaming anschliessend ergänzen.

## Kapitel 25: Vom Textfortsetzer zum Assistenten

Basismodell und Instruction-/Chat-Modell unterscheiden, Halluzinationen
und fehlende zuverlässige Rollentrennung erklären. Ein kleines
Instruction-Fine-Tuning als Experiment durchführen und Vorher/Nachher
an festen Dialogen vergleichen.

GPT-2 ist ein Textfortsetzungsmodell, kein trainierter Chatassistent.
Eine Chatoberfläche ändert das nicht. Abschweifende Antworten, erfundene
Rollen und unsinnige Texte sind mögliche Ergebnisse; gerade diese Grenzen
sollen nachvollziehbar werden.

## Empfohlener Lernpfad

```text
Kapitel 9  Trainingsschleife + Validierung
    ↓
Kapitel 10 Checkpoints speichern/laden
    ↓
Kapitel 11 Textgenerierung mit argmax
    ↓
Kapitel 12 Echter Text + echter Tokenizer
    ↓
Kapitel 13 Perplexity + Overfitting
    ↓
Kapitel 14 Adam und bessere Optimierung
    ↓
Kapitel 15 Training unterbrechen und fortsetzen
    ↓
Kapitel 16 Temperatur und Sampling
    ↓
Kapitel 17 Regularisierung gegen Overfitting
    ↓
Kapitel 18 Vollständiger Trainingslauf
    ↓
Kapitel 19 Trainingsdaten und Datenqualität
    ↓
Kapitel 20 Laufzeit und Speicher
    ↓
Kapitel 21 Architektur mit GPT-2 abgleichen
    ↓
Kapitel 22 Vortrainierte Gewichte importieren und prüfen
    ↓
Kapitel 23 Terminal-Chat
    ↓
Kapitel 24 Lokaler HTTP-Dienst und Chatoberfläche
    ↓
Kapitel 25 Instruction-Tuning und Grenzen
```

Kapitel 17 bis 20 sind keine zwingenden Voraussetzungen für den
Gewichtsimport, vertiefen aber das Verständnis des eigenen Trainings.

**Wichtig:** Modellberechnung und Parameter bleiben im `model`-Crate;
Trainingssteuerung gehört in `training`, Generierung und Chat-Anbindung
in den passenden Inference-Bereich. Dropout und der GPT-2-Abgleich
erfordern gezielte Änderungen an der Modellimplementierung.