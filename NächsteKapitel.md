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

**Status (18.10.2026): pausiert, nicht abgeschlossen.** Die vorhandenen Commits
und die Demo `crates/training/examples/ch11.rs` sind ein Lern-Zwischenstand.
Der Zwischenstand auf Branch `rene-weishaupt-kapitel-11-generieren` ist
committet und nach `origin` gepusht. Zuerst werden die Grundlagen in der
Lernsession „Kapitel selbst programmieren“ schrittweise aufgearbeitet.

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

Bisher nutzt Kapitel 9 künstliche IDs. Danach sinnvoll:

- `data/the-verdict.txt` tokenisieren
- echtes Vokabular verwenden
- Training und Validierung mit echten Token-Daten
- Speicher- und Laufzeitgrenzen erklären
- Unterschied zwischen Lernbeispiel und brauchbarem Sprachmodell zeigen

Dieses Kapitel sollte **nach** Checkpoints und Generierung kommen: Erst Mechanik beweisen, dann echten Text einsetzen.

## Kapitel 13: Auswertung und Overfitting

Optional, aber didaktisch wertvoll:

- Trainings-Loss und Validierungs-Loss visualisieren
- Perplexity aus dem Loss erklären
- Overfitting erkennen
- Epochenzahl und Lernrate vergleichen
- Bestes Modell anhand Validierungs-Loss speichern

Teile davon könnten auch in Kapitel 10 integriert werden.

## Kapitel 14: Bessere Optimierung

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