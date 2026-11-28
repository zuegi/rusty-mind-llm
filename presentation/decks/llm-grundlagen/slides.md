---
theme: default
layout: default
class: flex items-center text-left !bg-cover !bg-center
style: "background-image: url('/cover-bg.svg')"
title: rusty-mind-llm — Vom Text zum kleinen Sprachmodell
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

<div class="max-w-xl">
  <div class="inline-block px-3 py-1 rounded-full text-xs font-bold tracking-wider uppercase bg-orange-100 text-orange-800 border border-orange-300 mb-4">
    Lernprojekt in Rust · Burn · Kapitel 1–18
  </div>

  <h1 class="!text-5xl !font-bold !text-slate-900 !leading-tight !mb-3">
    rusty-mind-llm
  </h1>

  <p class="!text-xl !text-slate-600 !font-normal !leading-snug !mt-0 !mb-8">
    Vom Text zum kleinen Sprachmodell —<br />
    wie ein LLM lernt, bewertet wird und wo seine Grenzen liegen
  </p>

  <div class="text-sm text-slate-600 space-y-1">
    <div class="font-semibold text-slate-900">René Weishaupt</div>
    <div class="text-xs text-slate-500">20 Minuten Vortrag · <span class="font-mono text-orange-700">github.com/zuegi/rusty-mind-llm</span></div>
  </div>
</div>

<!--
Startfolie (vor dem eigentlichen Zeitblock / beim Ankommen des Publikums).
Titel: rusty-mind-llm — Ein eigenes kleines Sprachmodell von Grund auf in Rust mit Burn.
-->

---
layout: default
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

<PhaseBadge phase="inference" />
<SourceFooter chapter="Kap. 2" code="crates/training/src/tokenizer.rs" />

# Schritt 1: Text in Token und Nummern

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
- Jedes Token bekommt eine **Nummer (ID)** — wie eine Artikelnummer, noch ohne Bedeutung.

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

<PhaseBadge phase="inference" />
<SourceFooter chapter="Kap. 3" code="crates/model/src/embedding.rs" />

# Schritt 2: Jedes Token wird zum Vektor

<div class="grid grid-cols-[1.1fr_0.9fr] gap-6 items-center mt-1">
<div class="text-sm leading-snug space-y-2.5">

Aus jeder Token-ID wird ein **Embedding-Vektor** (feste Zahlenliste der Länge `embedding_dim`):

<div class="p-3 rounded-lg bg-pink-50 border border-pink-200 text-xs space-y-1.5 font-mono">
  <div class="text-[10px] font-sans font-semibold uppercase tracking-wide text-pink-800">Token-ID → Embedding-Vektor + Positions-Vektor</div>
  <div><strong>Der</strong> &nbsp;(ID 0) → [ 0.12, -0.45, 0.88, … ] + <strong>Pos. 1</strong></div>
  <div><strong>Hund</strong> (ID 1) → [ 0.74,  0.31, -0.19, … ] + <strong>Pos. 2</strong></div>
</div>

- **Embedding = Vektor:** Koordinaten statt bloßer Nummer — erst mit Vektoren kann das Modell rechnen und Ähnlichkeit ausdrücken (z. B. „Hund“ nah bei „Katze“).
- **Positions-Embedding:** eigener Vektor gleicher Länge — wird addiert, damit die Reihenfolge zählt („Der Hund beißt …“ ≠ „… beißt den Hund“).

**Wo welcher Vektor liegt, lernt das Modell erst beim Training.**

</div>
<div>
  <LlmArchitecture :step="3" />
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

<PhaseBadge phase="inference" />
<SourceFooter chapter="Kap. 4–5" code="crates/model/src/attention.rs" />

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
  <LlmArchitecture :step="4" />
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

<PhaseBadge phase="inference" />
<SourceFooter chapter="Kap. 5" code="crates/model/src/attention.rs" />

# Multi-Head Attention: Warum parallel?

<div class="grid grid-cols-[1.05fr_0.95fr] gap-5 items-center mt-1">
<div class="space-y-2 text-sm leading-snug">

- **Aufteilen statt verdoppeln:** `embedding_dim` wird durch `num_heads` geteilt (`head_dim = embedding_dim / num_heads`) — am Ende wieder zusammengefügt → **gleiche Länge**.
- **Zwei Ebenen Parallelität (der Durchbruch):**
  1. **Alle Heads gleichzeitig** auf ihren Teilvektoren.
  2. **Alle Tokens gleichzeitig** beim Training als große Matrixrechnung auf der GPU (statt Token-für-Token).

<div class="p-2.5 rounded-lg bg-orange-50 border border-orange-200 text-xs space-y-1">
  <div class="font-semibold text-orange-900">Zahlen: <code>embedding_dim = Heads × head_dim</code></div>
  <div class="font-mono text-[11px] text-slate-700 space-y-0.5">
    <div><strong>rusty-mind-llm</strong> (Demo): 8 = 2 Heads × 4</div>
    <div><strong>GPT-2</strong> (klein): &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 768 = 12 Heads × 64</div>
    <div><strong>Große LLMs</strong>: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 4.096+ = 32–96 Heads × 128</div>
  </div>
</div>

<div class="text-xs text-slate-700 pt-0.5">
  <strong>Merksatz:</strong> Innen (Heads &amp; Tokens) <strong>parallel</strong> → GPU-Tempo. Außen (<code>N×</code> Blöcke, nächste Folie) <strong>sequenziell</strong>.
</div>

</div>
<div>
  <MultiHeadParallel />
</div>
</div>

<!--
4:00–4:50. Grundlage: Kapitel 5 (crates/model/src/attention.rs).

Tonspur-Leitfaden:
- Warum waren Transformer 2017 („Attention Is All You Need“) der Durchbruch?
  Vorherige Netze (RNNs/LSTMs) mussten Text streng Wort für Wort lesen.
  Transformer verarbeiten im Training alle Wörter gleichzeitig als große
  Matrixrechnung — perfekt für GPUs.
- Und zusätzlich teilt Multi-Head Attention den Vektor eines Wortes in H gleich
  große Stücke (embedding_dim % num_heads == 0 im Code). Alle Heads laufen
  parallel, am Ende werden die Ergebnisse wieder zusammengeklebt.
- Konkrete Zahlen: Im Mini-Demo-Code 8 = 2 × 4, bei GPT-2 768 = 12 × 64, bei
  großen Modellen 32 bis 96 Heads.
- Überleitung zur nächsten Folie: Heads laufen parallel — die N Blöcke danach
  aber streng nacheinander.
-->

---

<PhaseBadge phase="inference" />
<SourceFooter chapter="Kap. 6–7" code="crates/model/src/transformer_block.rs" />

# Verarbeiten, stabil halten, wiederholen

<div class="grid grid-cols-[1.05fr_0.95fr] gap-5 items-center mt-1">
<div>

Im Block arbeiten neben Attention drei weitere Bausteine:

<div class="my-2.5 space-y-1.5 text-xs">
  <div class="p-2 rounded-lg bg-amber-50 border border-amber-200">
    <strong class="text-amber-900">LayerNorm:</strong> läuft <strong>vor</strong> Attention und Feed-Forward und hält alle Zahlenwerte in einem gesunden Bereich, damit nichts ausbricht.
  </div>
  <div class="p-2 rounded-lg bg-sky-50 border border-sky-200">
    <strong class="text-sky-900">Feed-Forward:</strong> jedes Token verarbeitet seinen neuen Kontext <strong>einzeln</strong> weiter — wie eine eigene kleine Recheneinheit pro Position.
  </div>
  <div class="p-2 rounded-lg bg-slate-50 border border-slate-200">
    <strong class="text-slate-900">+ (Abkürzung / Residual):</strong> addiert den vorherigen Zustand wieder dazu — ursprüngliche Information geht unterwegs nicht verloren.
  </div>
</div>

<div class="mt-2 space-y-1.5 text-xs leading-snug text-slate-800">
  <div>• <strong><code>N×</code> Schichten hintereinander (sequenziell):</strong> Output von Block 1 = Input von Block 2 (jeder mit eigenen Gewichten).</div>
  <div>• <strong>Wer bestimmt <code>N</code>?</strong> Wird vor dem Training als Einstellung (Hyperparameter: <code>num_blocks</code>, 1–3 im Lernprojekt, 12 bei GPT-2, 32–96+ bei großen LLMs).</div>
</div>

</div>
<div>
  <LlmArchitecture :step="5" />
</div>
</div>

<!--
4:15–5:00. Grundlage: Kapitel 6 und 7.

Tonspur-Leitfaden:
- Attention tauscht Infos zwischen Wörtern aus; Feed-Forward verarbeitet danach
  jedes Wort für sich weiter.
- LayerNorm & + sind das „Sicherheitsnetz“: Ohne sie werden tiefe Netze
  instabil oder vergessen den ursprünglichen Input.
- In der Grafik steht N×: N Blöcke laufen hintereinander (sequenziell, jeder mit
  eigenen Gewichten). Parallel laufen dagegen die Heads in Multi-Head Attention
  und die Token-Positionen innerhalb eines Blocks.
- N ist ein Hyperparameter (num_blocks in crates/model/src/gpt.rs), den der
  Entwickler vor dem Training festlegt — mehr Schichten = tieferes Modell, aber
  mehr Speicher, Rechenzeit und Datenbedarf.

Hinweis bei Rückfragen:
- Keine feste Rollenverteilung wie „Schicht 1 = Grammatik, letzte = Verständnis“.
- In rusty-mind-llm liegt LayerNorm jeweils vor Attention und Feed-Forward
  (Pre-LayerNorm), Residual + danach.
-->

---

<PhaseBadge phase="inference" />
<SourceFooter chapter="Kap. 7" code="crates/model/src/gpt.rs" />

# Das komplette GPT-Modell

<div class="grid grid-cols-[1.12fr_0.88fr] gap-5 items-center mt-1">
<div class="space-y-2.5 text-xs">

<div class="p-2.5 rounded-lg bg-slate-50 border border-slate-200 space-y-1">
  <div class="font-bold text-slate-900 text-sm mb-1">Wofür steht „GPT“?</div>
  <div>• <strong class="text-emerald-800">G</strong>enerative — erzeugt Text Schritt für Schritt, Token für Token.</div>
  <div>• <strong class="text-sky-800">P</strong>re-trained — Gewichte werden vorab auf viel Text trainiert.</div>
  <div>• <strong class="text-orange-800">T</strong>ransformer — genau diese Architektur rechts (<strong>Decoder-only</strong>: blickt nur zurück).</div>
</div>

<div class="p-2.5 rounded-lg bg-emerald-50/70 border border-emerald-200 space-y-1">
  <div class="font-bold text-emerald-950">Im Code sind es genau drei Bausteine (<code>GptModel</code>)</div>
  <div>1. <code>gpt_embedding</code> — Token-ID + Position → Vektor</div>
  <div>2. <code>blocks</code> — <code>N×</code> Transformer-Block hintereinander</div>
  <div>3. <code>output_projection</code> — Vektor → Punkte (<strong>Logits</strong>) für jedes Token im Vokabular</div>
</div>

<div class="p-2 rounded-lg bg-amber-50 border border-amber-200 leading-snug">
  <strong class="text-amber-950">Bauplan (<code>GptModelConfig</code>):</strong>
  <code>vocab_size</code> · <code>max_context_length</code> · <code>embedding_dim</code> · <code>num_heads</code> · <code>num_blocks</code> · <code>dropout</code>
</div>

</div>
<div>
  <LlmArchitecture :step="6" />
</div>
</div>

<!--
5:00–5:45. Grundlage: Kapitel 7 (crates/model/src/gpt.rs).

Tonspur-Leitfaden:
- Jetzt schließt sich der Rahmen rechts: Mit der Ausgabeschicht oben drauf
  haben wir das komplette GptModel — und genau das steckt hinter dem Namen GPT.
- Generative (erzeugt Text), Pre-trained (auf viel Text vortrainiert),
  Transformer (die Architektur, die wir gerade Baustein für Baustein aufgebaut
  haben — als Decoder-only-Variante mit kausaler Maske).
- Für Entwickler der Aha-Moment: In crates/model/src/gpt.rs hat GptModel genau
  drei Felder (gpt_embedding, blocks, output_projection) und sechs Stellschrauben
  in GptModelConfig. Große Modelle wie GPT-2 oder GPT-3 nutzen dasselbe Prinzip,
  nur mit deutlich größeren Zahlen in dieser Konfiguration.
-->

---

<PhaseBadge phase="inference" />
<SourceFooter chapter="Kap. 7" code="crates/model/src/gpt.rs" />

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
  <LlmArchitecture :step="6" />
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

<PhaseBadge phase="inference" />
<SourceFooter chapter="Kap. 11 & 16" code="crates/training/src/generation.rs" />

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
  <LlmArchitecture :step="7" />
</div>
</div>

<!--
6:15–7:00. Grundlage: Kapitel 11 und 16.

Die grüne Rückführung links in der Grafik zeigt genau diese Schleife.
Hinweis bei Rückfragen: Praktische Systeme speichern Zwischenergebnisse
(KV-Cache), damit nicht alles von Grund auf neu gerechnet werden muss.
-->

---

<PhaseBadge phase="training" />
<SourceFooter chapter="Kap. 8 & 12" code="crates/training/src/step.rs" />

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
  <LlmArchitecture :step="8" />
</div>
</div>

<!--
7:00–8:15. Grundlage: Kapitel 2, 4, 8 und 9.

Technisch: Cross-Entropy-Loss bewertet die Wahrscheinlichkeit des Zieltokens;
Gradienten zeigen die Richtung, der Optimizer führt das Gewichtsupdate aus.
„Richtig“ heißt nur: dieses Token steht im Trainingstext — andere Fortsetzungen
wären sprachlich oft genauso sinnvoll.
-->

---

<PhaseBadge phase="training" />
<SourceFooter chapter="Kap. 8 & Exkurs Ableitungen" code="crates/training/src/step.rs" />

# Backpropagation: Wer hatte Anteil am Fehler?

<div class="grid grid-cols-[1.12fr_0.88fr] gap-5 items-center mt-1">
<div class="text-sm leading-snug space-y-2.5">

Ein Modell hat Millionen Gewichte in vielen Schichten — am Ende steht aber nur **eine einzige Fehlerzahl (Loss)**.

<div class="space-y-2 text-xs">
  <div class="p-2.5 rounded-lg bg-slate-50 border border-slate-200">
    <strong class="text-slate-900">↑ Vorwärts (Forward):</strong> Tokens laufen von unten nach oben durch alle Schichten → Vorhersage → <strong>Fehlerzahl (Loss)</strong>.
  </div>
  <div class="p-2.5 rounded-lg bg-red-50 border border-red-200">
    <strong class="text-red-900">↓ Rückwärts (Backpropagation):</strong> Der Fehler wird <strong>Schicht für Schicht zurückgereicht</strong> (Kettenregel) — von der Ausgabeschicht bis zu den Embeddings.
  </div>
</div>

- **Ergebnis:** Jedes einzelne Gewicht bekommt seinen **Gradienten** (Anteil am Fehler + Richtung).
- **In der Praxis (Autodiff):** Niemand rechnet das von Hand — das Framework (`burn` / PyTorch) merkt sich die Rechenschritte und führt Backpropagation automatisch aus (`loss.backward()`).

</div>
<div>
  <LlmArchitecture :step="9" />
</div>
</div>

<!--
8:00–8:50. Grundlage: Kapitel 8 und Exkurs Ableitungen.

Tonspur-Leitfaden:
- Kernfrage beim Training: Am Ende kennen wir nur den Gesamtfehler (Loss).
  Woher weiß ein Gewicht ganz unten im Embedding oder in Block 1, ob es größer
  oder kleiner werden muss?
- Antwort: Backpropagation („Rückwärts-Weitergabe“). Wie eine Kette aus
  Zahnrädern: Wenn das letzte Zahnrad zu weit gedreht hat, rechnet man
  rückwärts durch die Kette (Kettenregel), welches Zahnrad wie viel Anteil
  daran hatte.
- In der Grafik rechts: Grau/Schwarz fließt vorwärts (unten → oben), Rot fließt
  rückwärts (oben → unten) bis in alle Schichten.
- Entwickler-Punkt: Dafür gibt es Autodiff-Frameworks (im Projekt: burn mit
  Autodiff-Backend) — ein einziger Aufruf loss.backward() berechnet alle
  Millionen Gradienten automatisch.
-->

---

<PhaseBadge phase="training" />
<SourceFooter chapter="Kap. 8 & Exkurs Ableitungen" code="crates/training/src/loss.rs" />

# Wie werden Gewichte angepasst?

<div class="grid grid-cols-[1.12fr_0.88fr] gap-5 items-center mt-1">
<div>

<div class="text-xs font-semibold uppercase tracking-wide text-red-700 mb-1.5">
  Fehlerfunktion Loss(Gewichte) — „Bergab im Nebel“
</div>

<GradientDescentDiagram />

<div class="mt-2.5 space-y-1.5 text-xs">
  <div><strong>Welche Funktion?</strong> Die <strong>Fehlerfunktion (Cross-Entropy-Loss)</strong>: <code>Gewichte → Modell → % für „rennt“ → Fehlerzahl</code> (90 % = klein, 5 % = groß).</div>
  <div><strong>1. Zufallsstart:</strong> Alle Gewichte in den Schichten starten zufällig.</div>
  <div><strong>2. Richtung (Gradient):</strong> Liefert <strong>Backpropagation</strong> (vorherige Folie) für jedes Gewicht: höher oder tiefer senkt den Fehler?</div>
  <div><strong>3. Schritt (Optimizer):</strong> <code class="text-red-800">neu = alt − Lernrate × Gradient</code> — tausendfach wiederholt.</div>
</div>

</div>
<div>
  <LlmArchitecture :step="9" />
</div>
</div>

<!--
8:15–10:00. Grundlage: Kapitel 8 und 9.

Tonspur-Leitfaden für die garantierte Rückfrage „Wie werden Gewichte berechnet /
auf welcher Funktion?“:
- Welche Funktion? Die Fehlerfunktion (Cross-Entropy-Loss) als Funktion aller
  Gewichte bei festem Übungssatz („Der Hund → rennt“): im Kern -ln(p) für das
  Zielwort (p = 90 % → Loss ≈ 0.11, p = 5 % → Loss ≈ 3.00).
- Man kennt die gesamte Fehlerlandschaft (Millionen Dimensionen) nicht global,
  aber weil jeder Rechenschritt im Modell differenzierbar ist, liefert
  Backpropagation (Kettenregel via Burn/Autodiff) automatisch die lokale
  Steigung (Gradient) am aktuellen Punkt.
- Bild „Bergab im Nebel“: Man sieht nicht das ganze Tal, spürt aber unter den
  Füßen, wo es bergab geht.
- Der Optimizer (z. B. SGD oder Adam) macht pro Schritt nur eine winzige
  Änderung (Lernrate): zu groß = springt übers Tal, zu klein = lernt ewig.
-->

---

<PhaseBadge phase="validation" />
<SourceFooter chapter="Kap. 9 & 13" code="crates/training/src/runner.rs" />

# Lernen oder Auswendiglernen?

<div class="grid grid-cols-[1.12fr_0.88fr] gap-5 items-center mt-1">
<div>

<div class="p-2 rounded bg-cyan-50 border border-cyan-200 text-xs leading-snug mb-1.5">
  <strong>Wie bei einer Schulprüfung:</strong> Wer nur Übungsaufgaben paukt, scheitert an neuen Aufgaben. Deshalb legen wir Prüfungstexte <strong>beiseite</strong>. <span class="text-cyan-900"><strong>1 Epoche</strong> = 1× alle Übungstexte komplett durchgelernt.</span>
</div>

<LossCurveChart />

<div class="text-xs leading-snug space-y-1 mt-1.5">
  <div>• <strong>Bis Durchlauf 16:</strong> Fehler sinkt bei <strong class="text-blue-700">Übungstext</strong> und <strong class="text-red-700">Prüfungstext</strong> → Modell lernt <strong>echte Sprachmuster</strong>.</div>
  <div>• <strong>Ab Durchlauf 16:</strong> nur <strong class="text-blue-700">Übungstext</strong> wird besser, <strong class="text-red-700">Prüfung</strong> schlechter → es lernt nur noch <strong>auswendig</strong>.</div>
  <div>• <strong>Folge:</strong> Wir behalten Stand <strong>16</strong> (beste Prüfung), nicht den letzten Stand 20.</div>
</div>

</div>
<div>
  <LlmArchitecture :step="10" />
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

<PhaseBadge phase="overview" label="Checkpoints · Speichern" />
<SourceFooter chapter="Kap. 17–18" code="crates/training/src/checkpoint.rs" />

# Zwei Speicherstände — zwei Zwecke

<div class="grid grid-cols-[1.05fr_0.95fr] gap-5 items-center mt-2">
<div>

Weil Training und Prüfung auseinanderlaufen (Epoche 16 vs. 20), speichern wir **zwei Stände**:

<div class="my-3 space-y-3 text-sm">
  <div class="p-3.5 rounded-lg bg-purple-50 border border-purple-200">
    <div class="font-bold text-purple-900"><code>last/</code> — Weitertrainieren (z. B. Epoche 20)</div>
    <div class="text-xs text-slate-700 mt-1">Speichert den letzten Stand inklusive Lernzustand, um später nahtlos weiterzumachen.</div>
  </div>
  <div class="p-3.5 rounded-lg bg-emerald-50 border border-emerald-200">
    <div class="font-bold text-emerald-900"><code>best/</code> — Anwenden (z. B. Epoche 16)</div>
    <div class="text-xs text-slate-700 mt-1">Speichert den Stand mit dem kleinsten Prüfungsfehler — diesen laden wir für die Textgenerierung.</div>
  </div>
</div>

</div>
<div>
  <LlmArchitecture :step="11" />
</div>
</div>

<!--
13:00–14:00. Grundlage: Kapitel 17 und 18.

Direkte Brücke von Folie 10 (Overfitting ab Epoche 16): Deshalb reicht ein
einziger Speicherstand nicht.
Technische Details bei Rückfragen:
- last/ speichert Modellgewichte + Optimizer-Zustand für exaktes Resume.
- best/ speichert den Stand mit bestem Validierungs-Loss.
- Kapitel 18 verbindet die gesamte Pipeline auf nur 8 Trainings- und
  2 Validierungsfenstern: vollständige Mechanik, kein Generalisierungsnachweis.
-->

---

<PhaseBadge phase="inference" label="Praxis · Echte Inferenz" />
<SourceFooter chapter="Kap. 12 & 16" code="crates/training/src/sampling.rs" />

# Was unser kleines Modell wirklich schreibt

<div class="grid grid-cols-[1.1fr_0.9fr] gap-5 items-center mt-1">
<div>

<div class="text-xs font-semibold uppercase tracking-wide text-emerald-800 mb-1">
  Echter Test: geladenen <code>best/</code>-Stand anwenden
</div>

<div class="text-xs mb-1">Vorgegebener Starttext: <code>I HAD always</code></div>

<div class="p-2 rounded bg-slate-100 border border-slate-200 font-mono text-xs mb-2">
  <div class="font-sans text-slate-500 text-[11px] mb-0.5">Vom Modell erzeugte Fortsetzung:</div>
  <div class="leading-snug">.<br /><br />""I to-- to I to I to I to I</div>
</div>

<div class="space-y-1 text-xs leading-snug">
  <div>• <strong>Was man sieht:</strong> echte englische Wörter statt Zufallssalat — aber schnelle Schleifen (<code>to I to I</code>).</div>
  <div>• <strong>Temperatur:</strong> steuert nur Abwechslung bei der Wortauswahl, <strong>nicht</strong> das Wissen.</div>
  <div>• <strong>Warum nicht produktiv deployen?</strong> Technisch startbar — aber winziges Modell, wenig Text, <strong>Laptop-Rechenleistung statt GPU-Cluster</strong> und <strong>kein Chat-/Anweisungs-Training</strong> (es setzt nur Text fort, beantwortet keine Fragen).</div>
</div>

</div>
<div>
  <LlmArchitecture :step="12" />
</div>
</div>

<!--
14:00–15:30. Grundlage: Kapitel 11, 12 und 16.

Quelle: Kapitel 16 („Gleicher Checkpoint, gleicher Prompt“), 16 neue Tokens mit
Greedy/argmax. Prompt und Fortsetzung klar trennen.
Wichtig für das Publikum: Das Ergebnis ist absichtlich ehrlich und ungeschönt.
Der Ablauf funktioniert technisch End-to-End, aber für gute Sprache fehlen
Größe und Daten.
Temperatur gehört zur Inferenz: niedrige Temperatur konzentriert die Verteilung
stärker auf Top-Kandidaten, höhere macht sie flacher — sie garantiert aber weder
bessere Texte noch korrekte Fakten.
-->

---

<PhaseBadge phase="overview" />
<SourceFooter chapter="Kap. 10–18" code="crates/model + crates/training" />

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
  <LlmArchitecture :step="13" />
</div>
</div>

<!--
15:30–17:00. Grundlage: Kapitel 10, 14, 15, 17 und 18.

Diese vier Begriffe werden in Videos und Artikeln oft vermischt.
Perfekte Klammer vor dem Fazit: Hier kurz auf das komplette Gesamtbild rechts
zeigen.
-->

---

<PhaseBadge phase="summary" />
<SourceFooter chapter="Kap. 1–18" />

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
  <LlmArchitecture :step="13" />
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

<PhaseBadge phase="summary" />
<SourceFooter chapter="Ausblick Kap. 19–28" />

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
  <RoadmapArchitecture />
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

<div class="mt-6 flex items-center justify-center gap-6">
  <div class="p-2 bg-white rounded-lg border border-slate-200 shadow-sm text-center">
    <img src="/book-qr.svg" alt="QR-Code zum begleitenden Buch" class="w-28 h-28 block mx-auto" />
    <div class="text-[10px] text-slate-500 mt-1">Buch öffnen</div>
  </div>
  <div class="text-left text-sm text-slate-600 space-y-1.5">
    <div><strong>Projekt & Code:</strong><br /><code>https://github.com/zuegi/rusty-mind-llm</code></div>
    <div><strong>Begleitendes Buch:</strong><br /><code>https://zuegi.github.io/rusty-mind-llm/</code></div>
  </div>
</div>

<div class="mt-5 text-xs text-slate-500">18 Minuten Vortrag · 2 Minuten Fragen</div>

<!--
18:00–20:00.

Verständnisfrage und Rückfragen. Stellschrauben/Gewichte ändern (Training)
versus wachsende Wortfolge bei festem Modell (Inferenz) als zentrale
Unterscheidung aufgreifen.
-->

---

<PhaseBadge phase="overview" label="Backup · Q&A Mathe-Detail" />
<SourceFooter chapter="Exkurs Ableitungen" code="crates/model + crates/training" />

# Backup: Gradientenabstieg konkret

<div class="grid grid-cols-[1.08fr_0.92fr] gap-5 items-start mt-2 text-xs">
<div class="space-y-2.5">

<div class="p-3 rounded-lg bg-slate-50 border border-slate-200 space-y-1">
  <div class="font-bold text-slate-800">Mini-Funktion (1 Gewicht <code>w</code>, Optimum bei 3)</div>
  <div>Fehler: <code>Loss(w) = (w − 3)²</code> · Ableitung: <code>Loss'(w) = 2 · (w − 3)</code></div>
  <div class="text-slate-600">Zufallsstart <strong>w = 1.0</strong> · Lernrate <strong>η = 0.2</strong></div>
</div>

<div class="p-3 rounded-lg bg-red-50 border border-red-200 space-y-1.5 font-mono">
  <div class="font-sans font-bold text-red-900">Schritt 1 (w = 1.0 → Loss = 4.00)</div>
  <div>Gradient = 2 · (1.0 − 3) = <strong>−4.0</strong></div>
  <div>w_neu = 1.0 − 0.2 · (−4.0) = <strong>1.80</strong> → Loss = <strong>1.44</strong></div>
</div>

<div class="p-3 rounded-lg bg-emerald-50 border border-emerald-200 space-y-1.5 font-mono">
  <div class="font-sans font-bold text-emerald-900">Schritt 2 (w = 1.80 → Loss = 1.44)</div>
  <div>Gradient = 2 · (1.80 − 3) = <strong>−2.4</strong></div>
  <div>w_neu = 1.80 − 0.2 · (−2.4) = <strong>2.28</strong> → Loss = <strong>0.52</strong></div>
</div>

<div class="text-slate-600">
  <strong>Gegenprobe (zu große Lernrate η = 1.2):</strong> <code>w = 1.0 → 5.80</code> → Loss steigt von <code>4.00</code> auf <strong>7.84</strong> (übers Tal gesprungen).
</div>

</div>
<div class="space-y-2.5">

<div class="p-3 rounded-lg bg-indigo-50 border border-indigo-200 space-y-1.5">
  <div class="font-bold text-indigo-900">Übertragung auf das Sprachmodell</div>
  <div>Statt 1 Gewicht <code>w</code> hat ein LLM tausende bis Milliarden Gewichte gleichzeitig.</div>
  <div>Statt <code>(w − 3)²</code> nutzt es <strong>Cross-Entropy-Loss</strong> auf der Wahrscheinlichkeit <code>p</code> des Zieltokens: <code>Loss = −ln(p)</code></div>
</div>

<div class="p-3 rounded-lg bg-white border border-slate-200 space-y-1">
  <div class="font-bold text-slate-800">Beispiele für <code>Loss = −ln(p)</code></div>
  <div>• <code>p = 90 %</code> → Loss <strong>0.11</strong> (sehr sicher & richtig)</div>
  <div>• <code>p = 52 %</code> („rennt“-Beispiel) → Loss <strong>0.66</strong></div>
  <div>• <code>p = 5 %</code> → Loss <strong>3.00</strong> (starke Korrektur nötig)</div>
</div>

<div class="p-3 rounded-lg bg-amber-50 border border-amber-200">
  <strong class="text-amber-900">Backpropagation (Kettenregel):</strong> Ein LLM verkettet viele Einzelfunktionen (<code>Linear</code>, <code>GELU</code>, <code>Softmax</code>, <code>LayerNorm</code>, <code>Loss</code>). Weil jede für sich differenzierbar ist, verknüpft <strong>Burn / Autodiff</strong> ihre Ableitungen rückwärts automatisch für jedes Gewicht.
</div>

<div class="text-slate-600 pt-0.5">
  <strong>Vertiefung im Buch (Exkurs Ableitungen):</strong><br />
  <a href="https://zuegi.github.io/rusty-mind-llm/exkurse/ableitungen.html" target="_blank" class="text-indigo-700 underline">https://zuegi.github.io/rusty-mind-llm/exkurse/ableitungen.html</a>
</div>

</div>
</div>

<!--
Backup-Folie für Q&A (nicht Teil der 18-Minuten-Kernzeit).
Nur öffnen, falls im Publikum nach der konkreten mathematischen Funktion oder
einem Zahlenbeispiel zum Gradientenabstieg gefragt wird.
-->

---

<PhaseBadge phase="overview" label="Backup · Q&A Architektur" />
<SourceFooter chapter="Einordnung · Dense vs. MoE" code="crates/model/src/transformer_block.rs" />

# Backup: Attention Head ≠ Experte (MoE)

<div class="grid grid-cols-2 gap-5 items-stretch mt-2 text-xs">
<div class="p-3.5 rounded-lg bg-slate-50 border border-slate-200 space-y-2.5">
  <div class="font-bold text-sm text-slate-900">Dense Transformer <span class="font-normal text-slate-500">(rusty-mind-llm, GPT-2)</span></div>

  <div class="p-2.5 rounded bg-orange-50 border border-orange-200 space-y-1">
    <div class="font-bold text-orange-900">1. Multi-Head Attention</div>
    <div><strong>Alle Heads</strong> rechnen bei jedem Token gleichzeitig mit (verschiedene Kontext-Blickwinkel).</div>
  </div>

  <div class="p-2.5 rounded bg-sky-50 border border-sky-200 space-y-1">
    <div class="font-bold text-sky-900">2. Ein Feed-Forward-Block</div>
    <div>Jedes Token durchläuft <strong>dasselbe</strong> Feed-Forward-Netz → <strong>100 %</strong> der Gewichte pro Token aktiv.</div>
  </div>

  <div class="text-slate-600 pt-0.5">
    <strong>Vorteil:</strong> einfach, stabil, überschaubar.<br />
    <strong>Grenze:</strong> doppeltes Wissen ≈ doppelte Rechenzeit.
  </div>
</div>

<div class="p-3.5 rounded-lg bg-violet-50/60 border border-violet-200 space-y-2.5">
  <div class="font-bold text-sm text-violet-950">Mixture of Experts / MoE <span class="font-normal text-violet-700">(z. B. Mixtral, DeepSeek)</span></div>

  <div class="p-2.5 rounded bg-orange-50 border border-orange-200 space-y-1">
    <div class="font-bold text-orange-900">1. Multi-Head Attention (unverändert)</div>
    <div>Heads sind <strong>keine</strong> Experten — Attention läuft weiterhin für alle Tokens.</div>
  </div>

  <div class="p-2.5 rounded bg-violet-100 border border-violet-300 space-y-1">
    <div class="font-bold text-violet-950">2. Router + viele Feed-Forward-„Experten“</div>
    <div>Statt 1 Block gibt es viele (z. B. 8–64) — ein <strong>Router</strong> aktiviert pro Token nur <strong>1–2 Experten</strong> (sparse).</div>
  </div>

  <div class="text-violet-900 pt-0.5">
    <strong>Vorteil:</strong> sehr viele Parameter (Wissen) bei schneller Inferenz.<br />
    <strong>Preis:</strong> hoher RAM-Bedarf &amp; komplexeres Training.
  </div>
</div>
</div>

<div class="mt-3 p-2.5 rounded-lg bg-emerald-50 border border-emerald-200 text-xs text-emerald-950">
  <strong>Merksatz:</strong> <strong>Attention Heads</strong> laufen <strong>immer alle</strong> (Kontext). <strong>MoE-Experten</strong> ersetzen den Feed-Forward-Teil — ein Router wählt wenige pro Token (Rechenzeit sparen). <code>rusty-mind-llm</code> ist ein klassisches <strong>Dense</strong>-Modell.
</div>

<!--
Backup-Folie für Q&A (nicht Teil der 18-Minuten-Kernzeit).
Nur öffnen, falls im Publikum nach „Mixture of Experts“ (MoE), Mixtral oder
DeepSeek gefragt wird oder Attention Heads mit Experten verwechselt werden.
-->
