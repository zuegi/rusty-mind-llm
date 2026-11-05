# 5.3 Erste Attention in Burn

**Lernziel:** Jede Zeile der kleinen Burn-Funktion einem bekannten Rechenschritt zuordnen.

## Anschluss: Bekannte Mathematik, neue Schreibweise

[5.2](05-02-anteile-berechnen.md) hat für A die Anteile
`[0.731..., 0.269...]` und für B `[0.269..., 0.731...]` geliefert.
Wir setzen genau diese Rechnung in Code um. Neue Attention-Schritte,
Batch-Achsen und lernbare Schichten kommen hier nicht hinzu.

## Die Rechnung mit Burn

Bis hier haben wir die erste, vereinfachte Attention-Rechnung von Hand
ausgeführt. Genau diese Rechnung setzt der folgende Burn-Code um. Er gehört
noch **nicht** zu den [Projektionen aus 5.4](05-04-projektionen.md):
Hier gilt weiterhin `Q = K = V = X`.

Die Variablen im Code entsprechen direkt den Formeln aus den vorherigen
Abschnitten:

```text
input              = X = [[1, 0], [0, 1]]
query, key, value   = X
scores              = Q @ K^T
weights             = softmax(scores)
context             = weights @ V
```

Damit wird in Burn nichts Neues gerechnet. Burn führt nur dieselben drei
Schritte als Tensoroperationen aus:

1. **Vergleichen**: `query.matmul(key.transpose())` vergleicht jede Query mit jedem Key und
   erzeugt die Score-Matrix `[[1, 0], [0, 1]]`.
2. **Gewichten**: `softmax(scores.clone(), 1)` macht aus jeder Score-Zeile die Anteile
   `[[0.731, 0.269], [0.269, 0.731]]`.
3. **Mischen**: `weights.clone().matmul(value)` mischt damit die Values und erzeugt die
   Kontext-Vektoren. Weil `V = [[1, 0], [0, 1]]` gilt, sehen sie hier zufällig
   genauso aus wie die Gewichte.

```rust,ignore
{{#include ../../../crates/model/examples/ch05.rs:attention}}
```

`Tensor<2>` ist ein zweidimensionaler Tensor. Die Demo wählt das CPU-Backend
mit `Device::flex()`. `clone()` erhält einen Tensor für weitere Verwendungen, weil
Burns Rechenoperationen ihre Eingaben in Rust verbrauchen.

`matmul` führt Matrixmultiplikation aus, `transpose` vertauscht die beiden
Dimensionen. `softmax(scores.clone(), 1)` normalisiert über Dimension `1`:
die Key-Spalten jeder Query-Zeile. Über Dimension `0` würden wir stattdessen
verschiedene Queries vermischen; das wäre hier falsch.

Die Funktion gibt Scores, Gewichte und Kontext zurück, damit wir jeden
Zwischenschritt ansehen können. Sie bleibt eine Lernhilfe im Example, keine
öffentliche Modell-API. Das Modell-Crate hängt weiterhin nicht vom
Training-Crate ab.

## Demo

```bash
cargo run -p model --example ch05
```

Ausgabe:

```text
{{#include ../ausgaben/ch05.txt}}
```

<details>
<summary>Vertiefung: Vollständiger ausführbarer Code und Tests</summary>

```rust,ignore
{{#include ../../../crates/model/examples/ch05.rs}}
```

Die Tests im Example prüfen unsere Rechnung, die zeilenweise Normalisierung
und einen zweiten Fall: Eine Query `[0, 0]` liefert für beide Keys den Score `0`.
Die Gewichte sind dann `[0.5, 0.5]`; Values `[0, 0]` und `[0, 2]` ergeben
den Kontext `[0, 1]`. Anders als im ersten Beispiel sind Gewichte und Kontext
hier verschieden.

```bash
cargo test -p model --example ch05
```

</details>

## Selbst versuchen

Welche Zeile macht aus Anteilen einen Inhaltsvektor? Warum kann die
Funktion nicht bereits nach `softmax` ihren Kontext zurückgeben?

<details>
<summary>Lösung anzeigen</summary>

`weights.clone().matmul(value)` mischt die Values. Softmax liefert nur
Anteile, nicht den Inhalt. Dass beide Tensoren in dieser Demo dieselben
Zahlen zeigen, liegt an den besonders einfachen Values.

</details>

## Was bleibt?

**Drei bekannte Schritte, drei Codezeilen:** `matmul` vergleicht, `softmax`
gewichtet, das zweite `matmul` mischt. `clone()` dient Rusts Verwendung der
Tensoren, nicht einem zusätzlichen mathematischen Schritt.

Offen bleibt, warum Query, Key und Value dieselben Zahlen haben sollen.
Diese Vereinfachung heben wir jetzt auf.

Zurück: [5.2 Anteile berechnen](05-02-anteile-berechnen.md).
Weiter: [5.4 Query, Key und Value erzeugen](05-04-projektionen.md).
