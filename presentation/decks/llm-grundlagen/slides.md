---
theme: default
layout: default
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

<div class="grid grid-cols-[1.1fr_0.9fr] gap-6 items-center mt-2">
<div>

Wie ein GPT-artiges Modell Text fortsetzt

<div class="mt-5 p-4 rounded-lg bg-slate-100 border border-slate-300">
  <div class="text-sm uppercase tracking-wide text-slate-500">Einstiegsfrage</div>
  <div class="text-xl font-bold mt-1">Wie geht „Der Hund …“ weiter?</div>
</div>

- Modell bewertet mögliche Fortsetzungen — kein Nachschlagen fertiger Sätze.
- Offenes Lernprojekt & Buch in Rust + Burn (Kapitel 1–18):
  - Code: `github.com/zuegi/rusty-mind-llm`
  - Buch: `zuegi.github.io/rusty-mind-llm/`

</div>
<div>
  <LlmArchitecture :step="1" />
</div>
</div>

<!--
0:00–1:00.

Etwa 20 Sekunden persönliche Entstehungsgeschichte:
„Ich habe ursprünglich mit Kotlin angefangen. Für meinen Ansatz fehlten mir
passende Libraries; der eigene Implementierungsaufwand wurde zu groß.
Deshalb bin ich zu Rust mit Burn gewechselt und habe dort einen strukturierten
Lernpfad aufgebaut.“ Kein allgemeiner Vergleich der Sprachen.

Hinweis auf das offene Repo und das begleitende deutsche Buch (aktueller Stand:
Kapitel 1–18).

Danach das Publikum mögliche Fortsetzungen für „Der Hund …“ nennen lassen.
Lernziel: nächste Tokenvorhersage, Lernen und Grenzen der Auswertung verstehen.
-->

---

# Text wird Zahlen

<div class="grid grid-cols-[1.1fr_0.9fr] gap-6 items-center mt-2">
<div>

```text
„Der Hund“ → Tokens → IDs → Embeddings + Positionen
```

- **Tokenizer (außerhalb):** zerlegt Text in Wörter, Wortteile oder Satzzeichen → Token-IDs.
- **IDs** sind Kennungen, keine Bedeutungen oder Wahrscheinlichkeiten.
- **Token-Embedding:** lernbarer Zahlenvektor pro ID.
- **Positions-Embedding:** ergänzt Reihenfolge (im Projekt ebenfalls lernbar).

**Welche Beziehungen Embeddings abbilden, entsteht erst durch Training.**

</div>
<div>
  <LlmArchitecture :step="2" />
</div>
</div>

<!--
1:00–3:00. Grundlage: Kapitel 1–3.

Drei Schritte nacheinander betonen: Textkarten → Token-IDs → lernbare Vektoren,
danach Positionen ergänzen. Embeddings sind kein fest vergebener „Bedeutungscode“.
Die Worttabelle auf Folie 4 ist künstlich; der echte GPT-2-Tokenizer hat 50.257 IDs.
Keine Tensorformen oder Rust-Syntax erklären.
-->

---

# Warum der bisherige Kontext zählt

<div class="grid grid-cols-[0.95fr_1.05fr] gap-5 items-center mt-2">
<div>

- Ein Token wird nicht isoliert vorhergesagt: der bisherige Kontext („Der Hund …“) zählt.
- **Attention** gewichtet und verbindet Informationen aus dem Kontext.
- **Kausale Maske** verhindert den Blick auf zukünftige Tokens.
- **N× Transformer-Blöcke** verfeinern die Darstellung schrittweise (Pre-LayerNorm + Residual `+`).

**Keine feste Rollenverteilung wie „Schicht 1 = Grammatik“ — was wichtig ist, hängt von Eingabe und Schicht ab.**

</div>
<div>
  <LlmArchitecture :step="3" />
</div>
</div>

<!--
3:00–5:00. Grundlage: Kapitel 5–7.

Grafik bewusst größer: Aufbau eines Transformer-Blocks im Stil von „Attention Is
All You Need“, angepasst an rusty-mind-llm (Decoder-only, Pre-LayerNorm vor
Attention/Feed-Forward, Residual-Verbindungen danach).
Kurze Wörter oder Artikel sind nicht grundsätzlich unwichtiger.
Keine Q/K/V-Formeln oder Tensorformen im Hauptteil.
-->

---

# Ein nächstes Token entsteht

<div class="grid grid-cols-[1.15fr_0.85fr] gap-5 items-center mt-1">
<div>

<div class="text-xs font-semibold uppercase tracking-wide text-amber-700 mb-1">
  Illustratives Beispiel — keine gemessene Modellvorhersage
</div>

Kontext: **Der Hund**

| ID | 0 | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- | --- |
| Token | Der | Hund | Katze | **rennt** | . |
| Logit | 1.2 | -0.4 | 0.8 | **2.1** | 0.3 |
| Softmax | 21.10% | 4.26% | 14.15% | **51.91%** | 8.58% |

- `argmax` → ID `3` → **„rennt“** (Greedy braucht kein Softmax).
- Neuer Kontext **„Der Hund rennt“** → erneut berechnen.
- Anwenden mit festen Gewichten heißt **Inferenz**.

</div>
<div>
  <LlmArchitecture :step="4" />
</div>
</div>

<!--
5:00–7:00. Grundlage: Kapitel 7, 11 und 16.

Quellen: Kapitel 7 (Logits), Kapitel 11 (künstliche Worttabelle). Zuordnung zu
„Der Hund“ dient nur der Erklärung; IDs sind keine echten GPT-2-IDs.
Softmax bei Temperatur 1 aus den illustrativen Logits berechnet — kein Maß für
Faktentreue.

Nach dem Anhängen von „rennt“ werden nicht die übrigen Kandidaten abgearbeitet,
sondern der neue Kontext erneut bewertet.
Praktische Systeme können frühere Berechnungen zwischenspeichern (KV-Cache);
keine pauschale Tokens/Sekunde-Zahl nennen.
-->

---

# Wie das Modell lernt

<div class="grid grid-cols-[1.1fr_0.9fr] gap-5 items-center mt-2">
<div>

Beispieltext **„Der Hund rennt“**:
Kontext **„Der Hund“** → Zieltoken **„rennt“**

- **Cross-Entropy Loss** bewertet, wie viel Wahrscheinlichkeit das Modell dem Zieltoken gibt.
- **Gradienten & Optimizer** passen die Modellgewichte an.
- „Richtig“ bedeutet: dieses Token steht im Trainingstext — nicht, dass andere Fortsetzungen falsch sind.

**Training ändert Gewichte. Inferenz hält Gewichte fest und verlängert Text.**

</div>
<div>
  <LlmArchitecture :step="5" />
</div>
</div>

<!--
7:00–10:00. Grundlage: Kapitel 2, 4, 8 und 9.

Dasselbe Beispiel wird jetzt zum Trainingsfall. Die hohen Werte von Folie 4 sind
keine Ausgangsmessung vor einem Update. Viele Schritte sollen Vorhersagen
verbessern, garantieren aber nicht bei jedem Schritt eine bessere Generalisierung.
-->

---

# Lernen oder Auswendiglernen?

<div class="grid grid-cols-[1.1fr_0.9fr] gap-5 items-center mt-2">
<div>

<div class="text-xs font-semibold uppercase tracking-wide text-cyan-800 mb-1">
  Dokumentierte Messwerte — Kapitel 13 (Lernrate 0.05)
</div>

| Epoche | Trainings-Loss | Validierungs-Loss |
| --- | --- | --- |
| 0 | 11.1860 | 11.1926 |
| **16** | **6.0976** | **6.6642 — bester Stand** |
| 20 | 5.7849 | 6.7019 |

- **Validierung** misst auf getrennten Daten bei **festen Gewichten**.
- Nach Epoche 16 sinkt Training weiter, Validierung steigt wieder → **Overfitting-Warnsignal**.

**Besten Validierungsstand speichern — nicht einfach den letzten Stand.**

</div>
<div>
  <LlmArchitecture :step="6" />
</div>
</div>

<!--
10:00–13:00. Grundlage: Kapitel 9 und 13.

Dokumentierte Messwerte aus Kapitel 13 (kein neu ausgeführter Lauf).
Kleinerer Loss ist besser. Kleine Validierung ist ein Warnsignal, kein Beweis
für allgemeines Sprachverständnis — und ersetzt keinen unberührten Testdatensatz.
-->

---

# Was unser Modell tatsächlich erzeugt

<div class="grid grid-cols-[1.1fr_0.9fr] gap-5 items-center mt-2">
<div>

<div class="text-xs font-semibold uppercase tracking-wide text-emerald-800 mb-1">
  Dokumentierte Ausgabe mit trainiertem Checkpoint — Kapitel 16
</div>

Vorgegebener Prompt: `I HAD always`

Neu erzeugte Fortsetzung (Greedy / `argmax`):

```text
.

""I to-- to I to I to I to I
```

- Generierung funktioniert technisch — liefert aber noch keine gute Sprache.
- **Greedy, Sampling & Temperatur** gehören zur **Inferenz**: sie ändern die Tokenauswahl, **nicht** die Gewichte.

**Andere Auswahlverfahren garantieren weder bessere Texte noch korrekte Fakten.**

</div>
<div>
  <LlmArchitecture :step="7" />
</div>
</div>

<!--
13:00–15:00. Grundlage: Kapitel 11, 12 und 16.

Quelle: Kapitel 16 („Gleicher Checkpoint, gleicher Prompt“), 16 neue Tokens.
Prompt und Fortsetzung klar trennen. Niedrigere Temperatur konzentriert die
Verteilung stärker, höhere macht sie flacher — sie steuert nicht direkt „sicher“
vs. „kreativ“. Die Zufalls-Demo aus Kapitel 11 wird nicht als Sprachleistung
präsentiert.
-->

---

# Der vollständige Lern- und Trainingsablauf

<div class="grid grid-cols-[1.02fr_0.98fr] gap-4 items-start mt-1">
<div class="text-sm">

<table class="w-full text-xs border-collapse mb-3">
  <thead>
    <tr class="border-b border-slate-300 text-left">
      <th class="py-1 pr-2">Begriff</th>
      <th class="py-1 pr-2">Gewichte</th>
      <th class="py-1 pr-2">Ziel?</th>
      <th class="py-1">Zweck</th>
    </tr>
  </thead>
  <tbody class="divide-y divide-slate-200">
    <tr><td class="py-1 pr-2 font-bold">Modell</td><td class="py-1 pr-2">enthält sie</td><td class="py-1 pr-2">nein</td><td class="py-1">Logits aus Token-IDs</td></tr>
    <tr><td class="py-1 pr-2 font-bold">Training</td><td class="py-1 pr-2">geändert</td><td class="py-1 pr-2">ja</td><td class="py-1">Loss senken</td></tr>
    <tr><td class="py-1 pr-2 font-bold">Validierung</td><td class="py-1 pr-2">fest</td><td class="py-1 pr-2">ja</td><td class="py-1">auf neuen Daten messen</td></tr>
    <tr><td class="py-1 pr-2 font-bold">Inferenz</td><td class="py-1 pr-2">fest</td><td class="py-1 pr-2">nein</td><td class="py-1">Vorhersage anwenden</td></tr>
  </tbody>
</table>

- `last/`: Modell + Optimizer zum **Weitertrainieren**.
- `best/`: bester Validierungsstand zum **Anwenden**.
- Kapitel 18: **8 Trainings-, 2 Validierungsfenster** → vollständige Mechanik, kein Generalisierungsnachweis.

</div>
<div>
  <LlmArchitecture :step="8" />
</div>
</div>

<!--
15:00–17:00. Grundlage: Kapitel 10, 14, 15, 17 und 18.

Vier zentrale Begriffe klar trennen: Modell, Training, Validierung, Inferenz.
Nur Modellgewichte reichen nicht für exaktes Trainings-Resume — daher last/
mit Optimizer-Zustand vs. best/ für die Anwendung.
-->

---

# Was die Experimente zeigen — und was nicht

<div class="grid grid-cols-[1.1fr_0.9fr] gap-5 items-center mt-1">
<div>

**Kernaussage:** Ein Sprachmodell lernt Textfortsetzung. Verlässlichkeit im Einsatz ist eine zusätzliche Bewertungsaufgabe.

- Besserer Loss garantiert **keine** Faktentreue, allgemeines Verständnis oder Produktionstauglichkeit (unberührter Testdatensatz nötig).

<div class="mt-3 text-sm">
  <div class="font-bold text-slate-700 mb-1">Ausblick — geplante Kapitel 19–28:</div>
  <ul class="space-y-1 text-xs leading-snug">
    <li><strong>Kap. 19–20:</strong> Datenqualität, Testdatensatz, Effizienz</li>
    <li><strong>Kap. 21–22:</strong> GPT-2-Abgleich (u. a. finale LayerNorm) & Gewichtsimport</li>
    <li><strong>Kap. 23–24:</strong> Lokaler Terminal- & Web-Chat</li>
    <li><strong>Kap. 25–28:</strong> RAG, Fine-Tuning, Evaluation & Betrieb</li>
  </ul>
</div>

</div>
<div>
  <LlmArchitecture :step="9" />
</div>
</div>

<!--
17:00–18:00.

Das Projekt macht Vorhersage, Gewichtsupdates, Validierung und Generierung
nachvollziehbar. Aus dem Next-Token-Prinzip allein folgt weder ein Beweis für
menschliches Denken noch der Beweis, dass komplexe Fähigkeiten unmöglich sind —
unsere kleinen Experimente beantworten diese allgemeinen Fragen nicht.

Ausblick basiert auf dem geplanten Lernpfad in NächsteKapitel.md (noch nicht
umgesetzt).
-->

---
layout: center
---

# Fragen & Diskussion

Was verändert sich beim **Training** — was bei der **Inferenz**?

<div class="mt-6 text-sm text-slate-600 space-y-1">
  <div>Repository: <code>https://github.com/zuegi/rusty-mind-llm</code></div>
  <div>Buch (Kapitel 1–18): <code>https://zuegi.github.io/rusty-mind-llm/</code></div>
</div>

<div class="mt-4 text-xs text-slate-500">18 Minuten Vortrag · 2 Minuten Fragen</div>

<!--
18:00–20:00.

Verständnisfrage und Rückfragen. Gewichte (Training) versus wachsende
Tokenfolge bei festen Gewichten (Inferenz) als zentrale Unterscheidung
aufgreifen.
-->
