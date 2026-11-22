---
theme: default
title: Vom Text zum kleinen Sprachmodell
info: |
  Ein Lernprojekt in Rust mit Burn, basierend auf Kapitel 1 bis 18.
  18 Minuten Vortrag und 2 Minuten Fragen.
lang: de
colorSchema: light
transition: slide-left
fonts:
  sans: Arial
  mono: monospace
  provider: none
---

# Vom Text zum kleinen Sprachmodell

Wie ein GPT-artiges Modell Text fortsetzt

**Wie geht „Der Hund …“ weiter?**

Rust · Burn · Ein nachvollziehbares Lernprojekt

Kann ich hier direkt

<!--
0:00–1:00.

Etwa 20 Sekunden persönliche Entstehungsgeschichte:
„Ich habe ursprünglich mit Kotlin angefangen. Für meinen Ansatz fehlten mir
passende Libraries; der eigene Implementierungsaufwand wurde zu groß.
Deshalb bin ich zu Rust mit Burn gewechselt und habe dort einen strukturierten
Lernpfad aufgebaut.“ Kein allgemeiner Vergleich der Sprachen.

Danach das Publikum mögliche Fortsetzungen nennen lassen.
Lernziel: nächste Tokenvorhersage, Lernen und Grenzen der Auswertung verstehen.
-->

---

# Text wird Zahlen

```text
Text → Tokens → IDs → lernbare Vektoren + Positionen
```

- Ein Token kann ein Wort, ein Wortteil oder ein Satzzeichen sein.
- IDs sind Kennungen, keine Bedeutungen oder Wahrscheinlichkeiten.
- Das Modell verarbeitet Zahlen; der Tokenizer übersetzt zurück in Text.

**Tokenizer und Sprachmodell sind unterschiedliche Bausteine.**

<!--
1:00–3:00. Grundlage: Kapitel 1–3.

Textkarten und IDs gedanklich nebeneinanderlegen. Die Worttabelle auf der
nächsten Beispiel-Folie ist künstlich; der echte GPT-2-Tokenizer hat 50.257 IDs.
Das vorhandene Tokenizer-Vokabular wird verwendet, nicht hier neu trainiert.
Keine Tensorformen oder Rust-Syntax erklären.
-->

---

# Der bisherige Kontext zählt

```text
Bisherige Tokens → Attention → Transformer-Blöcke
                                           ↓
                           Bewertung nächster Tokens
```

- Attention gewichtet Informationen aus dem bisherigen Kontext.
- Die Positionen der Tokens spielen eine Rolle.
- Die kausale Maske verhindert den Blick in die Zukunft.

**Das nächste Token wird nicht isoliert bewertet.**

<!--
3:00–5:00. Grundlage: Kapitel 5–7.

Attention als gewichtetes Verbinden von Informationen erklären, nicht als
menschliche Aufmerksamkeit. Spätere Tokens bleiben für eine frühere
Vorhersage unsichtbar. Q/K/V, Matrizen und Ableitungen auslassen.
-->

---

# Ein nächstes Token entsteht

**Illustrativ — keine gemessene Modellvorhersage**

Kontext: **Der Hund**

| ID | 0 | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- | --- |
| Token | Der | Hund | Katze | **rennt** | . |
| Logit | 1.2 | -0.4 | 0.8 | **2.1** | 0.3 |

`argmax` → ID `3` → **Der Hund rennt**

Token anhängen → **neu berechnen** → nächstes Token auswählen

Logits = Bewertungen, keine Prozentwerte. Softmax liefert Wahrscheinlichkeiten.

<!--
5:00–7:00.

Quellen: book/src/kapitel/07-gpt.md, Abschnitt „Was ist ein Logit?“;
book/src/kapitel/11-textgenerierung.md, „Token-IDs wieder als Text ausgeben“.
Die Verbindung dieser Logits mit dem Kontext „Der Hund“ ist eine Illustration.
Die IDs sind künstliche Worttabellen-IDs, nicht echte GPT-2-IDs.

Greedy kann direkt den höchsten Logit auswählen. Danach nicht die übrigen
Kandidaten abarbeiten, sondern mit dem erweiterten Kontext erneut vorhersagen.
Kapitel 11 nutzt ohne Checkpoint Zufallsgewichte und beweist keine Sprachleistung.
-->

---

# Wie das Modell lernt

```text
Trainingstext liefert das nächste Zieltoken
                    ↓
Vorhersage → Loss → Gradienten → Gewichtsupdate
                    ↑                 |
                    └── wiederholen ──┘
```

- Loss bewertet die Wahrscheinlichkeit des Zieltokens.
- „Richtig“ bedeutet: Dieses Token folgt im Trainingstext.
- Beim Generieren bleiben die Gewichte fest.

**Training verändert Gewichte. Generierung verlängert Text.**

<!--
7:00–10:00. Grundlage: Kapitel 2, 4, 8 und 9.

Beispieltext „Der Hund rennt“: Kontext „Der Hund“, Ziel „rennt“.
Andere Fortsetzungen können ebenfalls plausibel sein.
Ein Optimizer verändert Gewichte anhand der Gradienten; viele Updates sollen
die durchschnittlichen Vorhersagen verbessern, nicht jeder Schritt garantiert
eine Verbesserung auf unbekannten Daten.
-->

---

# Lernen oder Auswendiglernen?

**Dokumentierter Lauf aus Kapitel 13**

| Epoche | Trainings-Loss | Validierungs-Loss |
| --- | --- | --- |
| 0 | 11.1860 | 11.1926 |
| **16** | **6.0976** | **6.6642 — bester Stand** |
| 20 | 5.7849 | 6.7019 |

Training verbessert sich weiter; Validierung nicht.

**Besten Validierungsstand speichern, nicht einfach letzten Stand.**

Kleiner ist besser. Kleine Validierung → Warnsignal, kein allgemeiner Beweis.

<!--
10:00–13:00.

Quelle: book/src/kapitel/13-auswertung-und-overfitting.md,
dokumentierter Lauf mit Lernrate 0.05 und Loss-Kurven.
Dies sind dokumentierte Messwerte, keine neu ausgeführte Präsentationsdemo.
Die Tabelle ist die Basis; die vollständigen Kurven können später ergänzt werden.

Training und Validierung sind getrennte Datenbereiche derselben kurzen
Geschichte. Ein einzelner Anstieg beweist kein Overfitting. Die Auswahl auf
Validierungsdaten ersetzt keinen unberührten Testdatensatz.
-->

---

# Funktionierende Generierung ≠ gute Sprache

**Dokumentierte Ausgabe mit trainiertem Checkpoint — Kapitel 16**

Vorgegebener Prompt: `I HAD always`

Neu erzeugte Fortsetzung:

```text
.

""I to-- to I to I to I to I
```

Greedy wählt jeweils den höchsten Logit. Wiederholungen bleiben möglich.

**Sampling verändert die Auswahl, nicht das Wissen oder die Gewichte.**

<!--
13:00–15:00.

Quelle: book/src/kapitel/16-temperatur-und-sampling.md,
Abschnitt „Gleicher Checkpoint, gleicher Prompt“.
Die dokumentierte argmax-Ausgabe nutzt den Kapitel-13-Checkpoint,
Prompt „I HAD always“ und 16 neue Tokens. Der Punkt, zwei Zeilenumbrüche
und der folgende Text gehören zur Fortsetzung, nicht zum Prompt.
Andere Checkpoints können andere Ausgaben liefern.

Noch kein eigener Mitschnitt vorhanden. Die dokumentierte Ausgabe genügt
für die Basis. Nicht die erfundene „the the“-Illustration aus Kapitel 12
als gemessene Generierung zeigen.
-->

---

# Der vollständige Ablauf

```text
Daten → Training → Validierung → gespeichertes Modell → Generierung
                         ↓
              letzter Stand / bester Stand
```

- **Letzter Stand:** Training mit Optimizer-Zustand fortsetzen.
- **Bester Validierungsstand:** Modell zum Anwenden auswählen.
- Optimizer und Regularisierung beeinflussen das Lernen.

Kapitel 18: **8 Trainingsfenster, 2 Validierungsfenster**

**Vollständige Mechanik — kein Generalisierungsnachweis.**

<!--
15:00–17:00. Grundlage: Kapitel 10, 14, 15, 17 und 18.

Speichern nur der Modellgewichte reicht nicht für exaktes Trainings-Resume.
Kapitel 18 unterscheidet last/ und best/. In seinem kurzen dokumentierten Lauf
fällt die Validierung durchgehend; dort fällt best/ nicht hinter last/ zurück.
Den Unterschied der Auswahl zeigt der Lauf aus Kapitel 13.
Keine Details zu AdamW, Dropout-Seeds oder Checkpoint-Dateiformaten vertiefen.
-->

---
layout: center
---

# Was bleibt?

**Ein Sprachmodell lernt, Text fortzusetzen.**

Vorhersage → Lernen → getrennte Bewertung → Generierung

Besserer Loss garantiert keine:

- korrekten Fakten,
- allgemeinen Sprachfähigkeiten,
- Produktionstauglichkeit.

**Verlässlichkeit für einen Einsatz muss zusätzlich geprüft werden.**

<!--
17:00–18:00.

Das kleine Lernprojekt macht die Mechanik nachvollziehbar. Begrenzte Daten
und winzige Modelle erlauben keine allgemeinen Aussagen über große LLMs.
Für eine unabhängige Schlussbewertung wäre ein unberührter Testbereich nötig.
-->

---
layout: center
---

# Fragen

Was verändert sich beim **Training**?

Was verändert sich beim **Generieren**?

18 Minuten Vortrag · 2 Minuten Fragen

<!--
18:00–20:00.

Verständnisfrage und Rückfragen. Gewichte versus wachsende Tokenfolge
als zentrale Unterscheidung aufgreifen.
-->
