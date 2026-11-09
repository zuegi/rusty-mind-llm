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

**Status (09.11.2026): inhaltlich abgeschlossen; noch nicht in `main`.**
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
Kapitel 12 blieb unverändert. Kein Commit, Push oder PR erstellt.

Behandelte Themen:

- Trainings-Loss und Validierungs-Loss visualisieren
- Perplexity aus dem Loss erklären
- Overfitting erkennen
- Epochenzahl und Lernrate vergleichen
- Bestes Modell anhand Validierungs-Loss speichern

## Kapitel 14: Bessere Optimierung

**Status: nächster geplanter Schritt, noch nicht begonnen.**

Optional:

- Adam statt SGD
- Momentum
- Lernratenänderung
- Gradient Clipping
- Warmup
- Warum Optimizer-Zustand gespeichert werden muss

Nicht vor Checkpoints behandeln. Kapitel 9 soll erst den grundlegenden Ablauf festigen.

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
```

**Wichtig:** Kein Kapitel muss mehr Architektur im `model`-Crate hinzufügen. Die nächsten Kapitel gehören fast vollständig in `training` oder einen separaten Inference-Bereich. `model` bleibt für Vorwärtsberechnung und Parameter zuständig.