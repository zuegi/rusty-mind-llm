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

Wie ein Sprachmodell Schritt für Schritt Text fortsetzt

<div class="mt-5 p-4 rounded-lg bg-slate-100 border border-slate-300">
  <div class="text-sm uppercase tracking-wide text-slate-500">Einstiegsfrage</div>
  <div class="text-xl font-bold mt-1">Wie geht „Der Hund …“ weiter?</div>
</div>

- Es gibt mehrere passende Antworten — das Modell **schätzt**, was am ehesten folgt.
- Es schlägt keine fertigen Sätze aus einer Datenbank nach.

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
Lernziel: nächste Wort-/Tokenvorhersage, Lernen und Grenzen der Auswertung verstehen.
-->

---

# Schritt 1: Text in Stücke und Nummern

<div class="grid grid-cols-[1.1fr_0.9fr] gap-6 items-center mt-2">
<div>

Computer können nur rechnen — nicht lesen.

<div class="my-5 flex items-center gap-2 text-sm font-mono">
  <span class="px-3 py-1.5 rounded bg-slate-100 border border-slate-300">„Der Hund“</span>
  <span class="text-slate-400">→</span>
  <span class="px-2.5 py-1.5 rounded bg-blue-50 border border-blue-300">Der</span>
  <span class="px-2.5 py-1.5 rounded bg-blue-50 border border-blue-300">Hund</span>
  <span class="text-slate-400">→</span>
  <span class="px-2.5 py-1.5 rounded bg-amber-50 border border-amber-300 font-bold">0</span>
  <span class="px-2.5 py-1.5 rounded bg-amber-50 border border-amber-300 font-bold">1</span>
</div>

- Der **Tokenizer** (außerhalb des Modells) zerlegt Text in **Tokens**: Wörter, Wortteile oder Satzzeichen.
- Jedes Stück bekommt eine **Nummer (ID)** — wie eine Artikelnummer, noch ohne Bedeutung.

</div>
<div>
  <LlmArchitecture :step="2" />
</div>
</div>

<!--
1:00–2:00. Grundlage: Kapitel 1–2.

Die IDs 0 und 1 gehören zu unserer künstlichen Beispieltabelle auf Folie 6.
Der echte GPT-2-Tokenizer im Projekt kennt 50.257 verschiedene Tokens.
Wichtig: Tokenizer und Sprachmodell sind getrennte Bausteine.
-->

---

# Schritt 2: Nummern werden zu Zahlenlisten

<div class="grid grid-cols-[1.1fr_0.9fr] gap-6 items-center mt-2">
<div>

Aus jeder ID schlägt das Modell eine **Zahlenliste** nach:

<div class="my-4 p-3 rounded-lg bg-pink-50 border border-pink-200 text-xs space-y-2 font-mono">
  <div><strong>Der</strong> (ID 0) → [ 0.12, -0.45, 0.88, … ] + <strong>Position 1</strong></div>
  <div><strong>Hund</strong> (ID 1) → [ 0.74,  0.31, -0.19, … ] + <strong>Position 2</strong></div>
</div>

- **Token-Embedding:** lernbare Zahlenliste pro Textstück.
- **Positions-Embedding:** markiert die Reihenfolge („Der Hund beißt …“ ≠ „… beißt den Hund“).

**Was ähnliche Wörter verbindet, lernt das Modell erst beim Training.**

</div>
<div>
  <LlmArchitecture :step="2" />
</div>
</div>

<!--
2:00–3:00. Grundlage: Kapitel 3.

Die gezeigten Zahlen sind reine Platzhalter zur Anschauung.
Embeddings sind kein vorab festgelegter „Bedeutungscode“ — sie starten als
Zufallszahlen und werden erst durch Training sinnvoll.
Im Projekt sind sowohl Token- als auch Positions-Embeddings lernbar.
-->

---

# Masked Attention: Wer gehört zu wem?

<div class="grid grid-cols-[1.05fr_0.95fr] gap-5 items-center mt-1">
<div>

<div class="text-xs font-semibold uppercase tracking-wide text-indigo-700 mb-1.5">
  Anschauungsbeispiel: Worauf bezieht sich „er“?
</div>

<div class="p-3 rounded-lg bg-indigo-50 border border-indigo-200 mb-3 text-xs space-y-2">
  <div class="flex flex-wrap items-center gap-1 font-mono">
    <span class="px-1.5 py-0.5 rounded bg-white border border-slate-200 text-slate-500">Die</span>
    <span class="px-1.5 py-0.5 rounded bg-white border border-slate-200 text-slate-500">Katze</span>
    <span class="px-1.5 py-0.5 rounded bg-white border border-slate-200 text-slate-500">sieht</span>
    <span class="px-1.5 py-0.5 rounded bg-white border border-slate-200 text-slate-500">den</span>
    <span class="px-1.5 py-0.5 rounded bg-amber-100 border border-amber-400 font-bold text-amber-900">Hund</span>
    <span class="px-1.5 py-0.5 rounded bg-white border border-slate-200 text-slate-400">—</span>
    <span class="px-1.5 py-0.5 rounded bg-indigo-600 text-white font-bold">er</span>
    <span class="px-1.5 py-0.5 rounded bg-slate-200 border border-dashed border-slate-400 text-slate-500">???</span>
  </div>
  <div class="text-indigo-900">
    <strong>er</strong> schaut stark zurück auf <strong>Hund</strong> (kaum auf Katze) — <strong>???</strong> bleibt gesperrt.
  </div>
</div>

- **Attention:** jedes Wort schaut auf frühere Wörter und gewichtet, welche gerade wichtig sind.
- **Multi-Head:** mehrere Blickwinkel gleichzeitig (z. B. Satzbezug, Grammatik, Thema).
- **Masked:** spätere Wörter bleiben verdeckt — beim Lernen kein „Spicken“ in der Zukunft.

</div>
<div>
  <LlmArchitecture :step="3" />
</div>
</div>

<!--
3:00–4:15. Grundlage: Kapitel 5 und 6.

Tonspur-Leitfaden:
- Am Beispielsatz zeigen: Das Wort „er“ allein sagt fast nichts. Erst der
  Rückblick auf „Hund“ macht klar, wer gemeint ist.
- Multi-Head = das Modell macht diesen Rückblick nicht nur einmal, sondern mit
  mehreren parallelen „Blickwinkeln“ (keine fix vorprogrammierten Rollen!).
- Masked = alles rechts vom aktuellen Wort wird abgedeckt, damit das Modell
  wirklich das nächste Wort vorhersagen lernt statt zu spicken.

Hinweis: Das Satzbeispiel ist ein reines Anschauungsbeispiel (keine echte
Attention-Messung aus dem Modell). Keine Q/K/V-Formeln im Hauptteil.
-->

---

# Verarbeiten, stabil halten, wiederholen

<div class="grid grid-cols-[1.05fr_0.95fr] gap-5 items-center mt-1">
<div>

Nach dem Austausch zwischen den Wörtern folgen im Block noch drei Bausteine:

<div class="my-3 space-y-2 text-xs">
  <div class="p-2.5 rounded-lg bg-sky-50 border border-sky-200">
    <strong class="text-sky-900">Feed-Forward:</strong> jedes Wort verarbeitet seinen neuen Kontext <strong>einzeln</strong> weiter — wie eine eigene kleine Recheneinheit pro Position.
  </div>
  <div class="p-2.5 rounded-lg bg-amber-50 border border-amber-200">
    <strong class="text-amber-900">LayerNorm:</strong> hält alle Zahlenwerte in einem gesunden Bereich, damit nichts nach oben oder unten ausbricht.
  </div>
  <div class="p-2.5 rounded-lg bg-slate-50 border border-slate-200">
    <strong class="text-slate-900">+ (Abkürzung / Residual):</strong> addiert den vorherigen Zustand wieder dazu — ursprüngliche Information geht unterwegs nicht verloren.
  </div>
</div>

- **`N×` Durchläufe:** derselbe Block wird mehrfach gestapelt — das Ergebnis wird Schritt für Schritt feiner.

</div>
<div>
  <LlmArchitecture :step="3" />
</div>
</div>

<!--
4:15–5:00. Grundlage: Kapitel 6 und 7.

Tonspur-Leitfaden:
- Attention tauscht Infos zwischen Wörtern aus; Feed-Forward verarbeitet danach
  jedes Wort für sich weiter.
- LayerNorm & + sind das „Sicherheitsnetz“: Ohne sie werden tiefe Netze
  instabil oder vergessen den ursprünglichen Input.
- In der Grafik steht N×: Dieser ganze Block läuft mehrfach hintereinander
  (im Lernprojekt z. B. mehrere Schichten, in großen Modellen Dutzende).

Hinweis bei Rückfragen:
- Keine feste Rollenverteilung wie „Schicht 1 = Grammatik, letzte = Verständnis“.
- In rusty-mind-llm liegt LayerNorm jeweils vor Attention und Feed-Forward
  (Pre-LayerNorm), Residual + danach.
-->

---

# Ein nächstes Wort entsteht

<div class="grid grid-cols-[1.15fr_0.85fr] gap-5 items-center mt-1">
<div>

<div class="text-xs font-semibold uppercase tracking-wide text-amber-700 mb-1.5">
  Rechenbeispiel: Mini-Wörterbuch mit 5 Wörtern (real: 50.257)
</div>

<div class="flex flex-wrap items-center gap-1.5 text-xs mb-2.5 p-2 rounded bg-slate-50 border border-slate-200">
  <span class="font-semibold">„Der Hund“</span>
  <span class="text-slate-400">→</span>
  <span>Punkte pro Wort</span>
  <span class="text-slate-400">→</span>
  <span>Softmax (100 %)</span>
  <span class="text-slate-400">→</span>
  <span class="font-bold text-emerald-700">„rennt“</span>
</div>

<div class="my-2.5 space-y-1.5 text-xs">
  <div class="grid grid-cols-[54px_54px_1fr_44px] items-center gap-2">
    <span class="font-bold text-emerald-700">rennt</span>
    <span class="text-slate-500">+2.1 Pkt</span>
    <div class="h-4 bg-slate-100 rounded overflow-hidden"><div class="h-full bg-emerald-500 rounded" style="width: 51.9%"></div></div>
    <span class="font-bold text-emerald-700 text-right">51.9%</span>
  </div>
  <div class="grid grid-cols-[54px_54px_1fr_44px] items-center gap-2">
    <span>Der</span>
    <span class="text-slate-500">+1.2 Pkt</span>
    <div class="h-4 bg-slate-100 rounded overflow-hidden"><div class="h-full bg-slate-400 rounded" style="width: 21.1%"></div></div>
    <span class="text-slate-600 text-right">21.1%</span>
  </div>
  <div class="grid grid-cols-[54px_54px_1fr_44px] items-center gap-2">
    <span>Katze</span>
    <span class="text-slate-500">+0.8 Pkt</span>
    <div class="h-4 bg-slate-100 rounded overflow-hidden"><div class="h-full bg-slate-400 rounded" style="width: 14.2%"></div></div>
    <span class="text-slate-600 text-right">14.2%</span>
  </div>
  <div class="grid grid-cols-[54px_54px_1fr_44px] items-center gap-2">
    <span>.</span>
    <span class="text-slate-500">+0.3 Pkt</span>
    <div class="h-4 bg-slate-100 rounded overflow-hidden"><div class="h-full bg-slate-400 rounded" style="width: 8.6%"></div></div>
    <span class="text-slate-600 text-right">8.6%</span>
  </div>
  <div class="grid grid-cols-[54px_54px_1fr_44px] items-center gap-2">
    <span>Hund</span>
    <span class="text-slate-500">-0.4 Pkt</span>
    <div class="h-4 bg-slate-100 rounded overflow-hidden"><div class="h-full bg-slate-300 rounded" style="width: 4.3%"></div></div>
    <span class="text-slate-600 text-right">4.3%</span>
  </div>
</div>

- **Punkte (Logits):** hoch (`+2.1`) = passt gut, tief (`-0.4`) = passt schlecht.
- **Softmax:** rechnet Rohpunkte in **Prozentanteile (Summe 100 %)** um.

</div>
<div>
  <LlmArchitecture :step="4" />
</div>
</div>

<!--
5:00–6:15. Grundlage: Kapitel 7 und 11.

Quellen: Kapitel 7 (Logits: 1.2, -0.4, 0.8, 2.1, 0.3), Kapitel 11 (künstliche
Worttabelle IDs 0–4: Der, Hund, Katze, rennt, .). Softmax bei Temperatur 1:
21.10%, 4.26%, 14.15%, 51.91%, 8.58%. Keine echten GPT-2-IDs, kein Maß für
Faktentreue.
Output-Projektion bildet die letzte Zahlenliste auf genau einen Punktwert pro
bekanntem Wort ab. Bei Greedy-Auswahl (argmax) gewinnt immer der höchste Logit —
dafür wäre Softmax rechnerisch gar nicht nötig.
-->

---

# Wort anhängen — und wieder von vorn

<div class="grid grid-cols-[1.12fr_0.88fr] gap-5 items-center mt-1">
<div>

<div class="text-xs font-semibold uppercase tracking-wide text-emerald-700 mb-1.5">
  Fortsetzung des Beispiels — ein Token pro Durchlauf
</div>

<div class="my-2.5 p-3 rounded-lg bg-emerald-50 border border-emerald-200 space-y-2.5 text-xs">
  <div class="flex flex-wrap items-center gap-1 font-mono">
    <span class="font-sans font-bold text-slate-700 mr-1">Runde 1:</span>
    <span class="px-1.5 py-0.5 rounded bg-white border border-slate-300">Der</span>
    <span class="px-1.5 py-0.5 rounded bg-white border border-slate-300">Hund</span>
    <span class="text-slate-400">→</span>
    <span class="px-1.5 py-0.5 rounded bg-emerald-600 text-white font-bold">rennt</span>
    <span class="font-sans text-emerald-800 ml-1">(51.9 %)</span>
  </div>
  <div class="flex flex-wrap items-center gap-1 font-mono">
    <span class="font-sans font-bold text-slate-700 mr-1">Runde 2:</span>
    <span class="px-1.5 py-0.5 rounded bg-white border border-slate-300">Der</span>
    <span class="px-1.5 py-0.5 rounded bg-white border border-slate-300">Hund</span>
    <span class="px-1.5 py-0.5 rounded bg-emerald-100 border border-emerald-300 text-emerald-900">rennt</span>
    <span class="text-slate-400">→</span>
    <span class="px-1.5 py-0.5 rounded bg-emerald-600 text-white font-bold">.</span>
    <span class="font-sans text-slate-600 ml-1">(neu berechnet)</span>
  </div>
  <div class="font-sans text-slate-600 pt-1 border-t border-emerald-200">
    <strong>Stopp:</strong> bei Satzende oder Maximallänge — sonst Runde 3, 4, …
  </div>
</div>

- Es nimmt **nicht Platz 2** aus Runde 1 („Der“, 21.1 %) — jede Runde berechnet für den neuen Satz **neue Punkte**.
- **Inferenz:** Anwenden mit **eingefrorenen Einstellungen** — das Modell lernt hier nichts dazu.

</div>
<div>
  <LlmArchitecture :step="4" />
</div>
</div>

<!--
6:15–7:00. Grundlage: Kapitel 11 und 16.

Die grüne Rückführung links in der Grafik zeigt genau diese Schleife.
Hinweis bei Rückfragen: Praktische Systeme speichern Zwischenergebnisse
(KV-Cache), damit nicht alles von Grund auf neu gerechnet werden muss.
-->

---

# Wie das Modell lernt

<div class="grid grid-cols-[1.1fr_0.9fr] gap-5 items-center mt-2">
<div>

Übungssatz aus dem Trainingstext: **„Der Hund rennt“**

<div class="my-4 space-y-2 text-sm">
  <div class="p-2.5 rounded bg-slate-50 border border-slate-200"><strong>1. Schätzen:</strong> Eingabe „Der Hund“ → Modell vergibt Punkte.</div>
  <div class="p-2.5 rounded bg-red-50 border border-red-200"><strong>2. Fehler messen (Loss):</strong> Wie weit lag es vom Ziel <strong>„rennt“</strong> weg?</div>
  <div class="p-2.5 rounded bg-red-50 border border-red-200"><strong>3. Anpassen:</strong> Stellschrauben (Gewichte) minimal drehen → Fehler sinkt.</div>
</div>

**Training verändert das Modell. Anwenden (Inferenz) lässt es unverändert.**

</div>
<div>
  <LlmArchitecture :step="5" />
</div>
</div>

<!--
7:00–10:00. Grundlage: Kapitel 2, 4, 8 und 9.

Technisch: Cross-Entropy-Loss bewertet die Wahrscheinlichkeit des Zieltokens;
Gradienten zeigen die Richtung, der Optimizer führt das Gewichtsupdate aus.
„Richtig“ heißt nur: dieses Token steht im Trainingstext — andere Fortsetzungen
wären sprachlich oft genauso sinnvoll.
-->

---

# Lernen oder Auswendiglernen?

<div class="grid grid-cols-[1.12fr_0.88fr] gap-5 items-center mt-1">
<div>

<div class="text-xs font-semibold uppercase tracking-wide text-cyan-800 mb-1">
  Echte Messung aus unserem Lernprojekt (kleiner Wert = besser)
</div>

<LossCurveChart />

- **Prüfung (Validierung):** misst auf **fremdem Text** bei festem Modell.
- Ab Epoche 16 lernt das Modell die Übungsdaten nur noch **auswendig**.
- **Deshalb speichern wir Epoche 16 — nicht den letzten Stand.**

</div>
<div>
  <LlmArchitecture :step="6" />
</div>
</div>

<!--
10:00–13:00. Grundlage: Kapitel 9 und 13.

Alle 21 Punkte stammen direkt aus dem dokumentierten Lauf in Kapitel 13
(Lernrate 0.05):
- Epoche 0: 11.1860 / 11.1926
- Epoche 16: 6.0976 / 6.6642 (bester Validierungs-Loss)
- Epoche 20: 5.7849 / 6.7019

Fachbegriff: Overfitting. Kleiner Datensatz ist ein Warnsignal, kein allgemeiner
Beweis; für eine unabhängige Schlussbewertung bräuchte es zusätzlich unberührte
Testdaten.
-->

---

# Was unser kleines Modell wirklich schreibt

<div class="grid grid-cols-[1.1fr_0.9fr] gap-5 items-center mt-2">
<div>

<div class="text-xs font-semibold uppercase tracking-wide text-emerald-800 mb-1">
  Echte Ausgabe unseres trainierten Mini-Modells
</div>

Vorgegebener Starttext: `I HAD always`

Vom Modell erzeugte Fortsetzung:

```text
.

""I to-- to I to I to I to I
```

- Die Technik funktioniert — gute Sprache braucht aber **viel mehr Daten und Modellgröße**.
- **Auswahlstrategie (z. B. Temperatur):** entscheidet nur, ob immer das Top-Wort oder auch Alternativen gewählt werden.
- Das Modell selbst ändert sich dabei **nicht**.

</div>
<div>
  <LlmArchitecture :step="7" />
</div>
</div>

<!--
13:00–15:00. Grundlage: Kapitel 11, 12 und 16.

Quelle: Kapitel 16 („Gleicher Checkpoint, gleicher Prompt“), 16 neue Tokens mit
Greedy/argmax. Prompt und Fortsetzung klar trennen.
Temperatur gehört zur Inferenz: niedrige Temperatur konzentriert die Verteilung
stärker auf Top-Kandidaten, höhere macht sie flacher — sie garantiert aber weder
bessere Texte noch korrekte Fakten.
-->

---

# Vier Begriffe klar getrennt

<div class="grid grid-cols-[1.05fr_0.95fr] gap-4 items-center mt-2">
<div>

<table class="w-full text-xs border-collapse">
  <thead>
    <tr class="border-b-2 border-slate-300 text-left">
      <th class="py-2 pr-2">Begriff</th>
      <th class="py-2 pr-2">Modell ändern?</th>
      <th class="py-2">Wozu?</th>
    </tr>
  </thead>
  <tbody class="divide-y divide-slate-200">
    <tr><td class="py-2 pr-2 font-bold">Modell</td><td class="py-2 pr-2">enthält Wissen</td><td class="py-2">bewertet mögliche nächste Wörter</td></tr>
    <tr><td class="py-2 pr-2 font-bold text-red-700">Training</td><td class="py-2 pr-2">ja</td><td class="py-2">Fehler schrittweise senken</td></tr>
    <tr><td class="py-2 pr-2 font-bold text-cyan-700">Validierung</td><td class="py-2 pr-2">nein (fest)</td><td class="py-2">auf fremdem Text prüfen</td></tr>
    <tr><td class="py-2 pr-2 font-bold text-emerald-700">Inferenz</td><td class="py-2 pr-2">nein (fest)</td><td class="py-2">fertiges Modell anwenden</td></tr>
  </tbody>
</table>

<div class="mt-4 text-sm text-slate-700">
  Die vier Rollen sind im Ablauf sauber getrennt — und bleiben es auch in großen LLMs.
</div>

</div>
<div>
  <LlmArchitecture :step="8" />
</div>
</div>

<!--
15:00–16:00. Grundlage: Kapitel 10, 14, 15, 17 und 18.

Diese vier Begriffe werden in Videos und Artikeln oft vermischt.
Hier kurz auf die passenden Bereiche in der Grafik rechts zeigen.
-->

---

# Zwei Speicherstände — zwei Zwecke

<div class="grid grid-cols-[1.05fr_0.95fr] gap-5 items-center mt-2">
<div>

<div class="space-y-3 text-sm">
  <div class="p-3.5 rounded-lg bg-purple-50 border border-purple-200">
    <div class="font-bold text-purple-900"><code>last/</code> — Weitertrainieren</div>
    <div class="text-xs text-slate-700 mt-1">Speichert den letzten Trainingsstand inklusive Lernzustand, um später nahtlos weiterzumachen.</div>
  </div>
  <div class="p-3.5 rounded-lg bg-emerald-50 border border-emerald-200">
    <div class="font-bold text-emerald-900"><code>best/</code> — Anwenden</div>
    <div class="text-xs text-slate-700 mt-1">Speichert den Stand mit dem kleinsten Prüfungsfehler — für die Textgenerierung.</div>
  </div>
</div>

<div class="mt-4 text-sm font-semibold text-slate-800">
  Der komplette Ablauf funktioniert End-to-End — bleibt aber ein kleines Lernexperiment.
</div>

</div>
<div>
  <LlmArchitecture :step="8" />
</div>
</div>

<!--
16:00–17:00. Grundlage: Kapitel 17 und 18.

Technische Details bei Rückfragen:
- last/ speichert Modellgewichte + Optimizer-Zustand für exaktes Resume.
- best/ speichert den Stand mit bestem Validierungs-Loss.
- Kapitel 18 verbindet die gesamte Pipeline auf nur 8 Trainings- und
  2 Validierungsfenstern: vollständige Mechanik, kein Generalisierungsnachweis.
-->

---

# Was bleibt — und was nicht

<div class="grid grid-cols-[1.1fr_0.9fr] gap-5 items-center mt-2">
<div>

<div class="p-4 rounded-lg bg-slate-100 border border-slate-300 text-base font-bold text-slate-900 mb-4">
  Ein Sprachmodell lernt Textfortsetzung. Ob seine Antworten verlässlich sind, muss man zusätzlich prüfen.
</div>

- Ein besserer Messwert garantiert **keine** wahren Fakten.
- Ein funktionierender Prototyp beweist **keine** Produktionstauglichkeit.
- Die ganze Architektur dahinter ist kein Zauber — sondern ein nachvollziehbarer Ablauf.

</div>
<div>
  <LlmArchitecture :step="9" />
</div>
</div>

<!--
17:00–17:30.

Aus dem Next-Token-Prinzip allein folgt weder ein Beweis für menschliches Denken
noch der Beweis, dass komplexe Fähigkeiten unmöglich sind — unsere kleinen
Experimente beantworten diese allgemeinen Fragen nicht.
Für eine unabhängige Schlussbewertung wäre ein unberührter Testdatensatz nötig.
-->

---

# Wie es im Buch weitergeht

<div class="grid grid-cols-[1.1fr_0.9fr] gap-5 items-center mt-2">
<div>

Bisher umgesetzt: **Kapitel 1–18** (Basismodell bis vollständiger Trainingsablauf).

<div class="mt-4 space-y-2.5 text-sm">
  <div class="p-2.5 rounded bg-slate-50 border border-slate-200"><strong>Bessere Daten & Tempo:</strong> sauberere Texte, unabhängige Tests, Effizienz</div>
  <div class="p-2.5 rounded bg-slate-50 border border-slate-200"><strong>Fertiges Wissen laden:</strong> echtes vortrainiertes GPT-2 importieren</div>
  <div class="p-2.5 rounded bg-slate-50 border border-slate-200"><strong>Eigener Chat:</strong> lokal im Terminal und im Browser</div>
  <div class="p-2.5 rounded bg-slate-50 border border-slate-200"><strong>Praxis-Einsatz:</strong> eigenes Wissen einbinden (RAG), Feinschliff & Betrieb</div>
</div>

</div>
<div>
  <LlmArchitecture :step="9" />
</div>
</div>

<!--
17:30–18:00.

Ausblick entspricht dem geplanten Lernpfad in NächsteKapitel.md (Kapitel 19–28,
noch nicht umgesetzt):
- Kap. 19–20: Datenqualität, unabhängiger Testdatensatz, Laufzeit/Speicher
- Kap. 21–22: GPT-2-Architekturabgleich (u. a. finale LayerNorm) & Gewichtsimport
- Kap. 23–24: Terminal-Chat, HTTP-Dienst, Web-UI
- Kap. 25–28: RAG, Fine-Tuning/Instruction-Tuning, Evaluation, Sicherheit & Betrieb
-->

---
layout: center
---

# Fragen & Diskussion

Was verändert sich beim **Training** — und was bei der **Anwendung (Inferenz)**?

<div class="mt-6 text-sm text-slate-600 space-y-1">
  <div>Projekt & Code: <code>https://github.com/zuegi/rusty-mind-llm</code></div>
  <div>Begleitendes Buch: <code>https://zuegi.github.io/rusty-mind-llm/</code></div>
</div>

<div class="mt-4 text-xs text-slate-500">18 Minuten Vortrag · 2 Minuten Fragen</div>

<!--
18:00–20:00.

Verständnisfrage und Rückfragen. Stellschrauben/Gewichte ändern (Training)
versus wachsende Wortfolge bei festem Modell (Inferenz) als zentrale
Unterscheidung aufgreifen.
-->
