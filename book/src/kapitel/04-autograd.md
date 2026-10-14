# 4. Autograd

## Das Problem

Die Embedding-Tabellen aus Kapitel 3 enthalten Zahlen, die das Modell später
lernen soll. Dafür muss es wissen: In welche Richtung muss sich jede Zahl
ändern, damit das Ergebnis besser wird?

Diese Information heisst **Gradient**. Ein GPT-Modell hat sehr viele Parameter;
ihre Gradienten von Hand auszurechnen, wäre unmöglich. Diese Arbeit übernimmt
**Autograd** (automatisches Ableiten). Wir schreiben dafür keinen eigenen Code:
Burn bringt Autograd mit.

Dieses Kapitel berechnet nur Gradienten. Die Gewichte damit zu verändern, ist
Aufgabe eines Optimizers. Er folgt später im Kapitel zum Training.

## Was ein Gradient bedeutet

Nehmen wir die Funktion

```text
y = x²
```

Bei `x = 3` ist `y = 9`. Die Ableitung lautet `dy/dx = 2x`, also `6`.

Diese `6` sagt: Wird `x` ein kleines Stück grösser, wächst `y` etwa sechsmal
so stark. Das lässt sich ohne Ableitungsregel nachrechnen:

```text
x = 3     → y = 9
x = 3.01  → y = 9.0601

Änderung von y / Änderung von x = 0.0601 / 0.01 = 6.01
```

Je kleiner der Schritt `h`, desto näher liegt das Verhältnis an `6`:

| Schritt `h` | `(y(3 + h) - 9) / h` |
| ----------- | -------------------- |
| 0.1 | 6.1 |
| 0.01 | 6.01 |
| 0.001 | 6.001 |

Dieser Grenzwert ist die Ableitung. Die Regel `2x` ist eine Abkürzung für diese
Rechnung. Woher sie kommt, zeigt der Exkurs
[Ableitungen für Programmierer](../exkurse/ableitungen.md).

Das Vorzeichen zeigt die Richtung. Bei `x = -2` ist der Gradient `-4`: Wird `x`
etwas grösser, wird `y` kleiner.

## Autodiff in Burn

Für Gradienten aktivieren wir Autodiff auf dem Device, bevor wir Tensoren
und Modellparameter erzeugen:

```rust,ignore
let device = Device::flex().autodiff();
```

Flex rechnet auf der CPU. `.autodiff()` aktiviert die Aufzeichnung, wie jedes
Ergebnis entstanden ist. Diese Aufzeichnung heisst *Rechengraph*.

Danach folgen immer drei Schritte:

1. **Markieren:** `require_grad()` sagt Burn, für welchen Tensor wir einen
   Gradienten wollen.
2. **Rückwärts rechnen:** `backward()` läuft den Rechengraphen vom Ergebnis
   zurück und sammelt alle Gradienten.
3. **Abfragen:** `grad(&grads)` liefert den Gradienten eines markierten Tensors.

`backward()` braucht ein Ergebnis mit genau einer Zahl. Bei Tensoren mit
mehreren Werten bilden wir deshalb zuerst eine Summe mit `sum()`.

`grad` liefert eine `Option`: `None`, falls Burn für diesen Tensor keinen
Gradienten berechnet hat. Die Beispiele wandeln das mit `ok_or` in einen Fehler
mit verständlicher Meldung um.

## Beispiel 1: Eine Variable

```rust,ignore
{{#include ../../../crates/model/examples/ch04.rs:square}}
```

`x.clone() * x.clone()` berechnet `x²`. Die Kopien sind nötig, weil die
Multiplikation ihre Eingaben in Rust verbraucht; `x` selbst brauchen wir danach
noch für `grad`. Die Kopie eines Tensors teilt sich die Daten und ist günstig.

Burn liefert `dy/dx = 6`, wie von Hand berechnet.

## Beispiel 2: Die Kettenregel

Jetzt besteht die Rechnung aus zwei Schritten:

```text
u = 2x + 1
y = u²
```

Bei `x = 3` ist `u = 7` und `y = 49`. Der Gradient wird rückwärts durch beide
Schritte weitergegeben und unterwegs multipliziert:

```text
dy/du = 2u = 14
du/dx = 2
dy/dx = 14 · 2 = 28
```

Das ist die **Kettenregel**. Autograd wendet sie für jeden Rechenschritt im
Graphen an, egal wie lang die Kette ist.

```rust,ignore
{{#include ../../../crates/model/examples/ch04.rs:chain_rule}}
```

## Beispiel 3: Tensoren

Neuronale Netze rechnen nicht mit einzelnen Zahlen, sondern mit Tensoren. Hier
gewichten wir drei Eingaben mit drei Gewichten:

```text
w    = [0.5, -1, 2]
x    = [1,    2, 3]
loss = 0.5·1 + (-1)·2 + 2·3 = 4.5
```

Wird `w[0]` etwas grösser, wächst `loss` um das `1`-fache, bei `w[1]` um das
`2`-fache, bei `w[2]` um das `3`-fache. Der Gradient von `w` ist also genau
`x = [1, 2, 3]`.

Nur `w` ist mit `require_grad()` markiert; `x` sind feste Eingabedaten.

```rust,ignore
{{#include ../../../crates/model/examples/ch04.rs:weighted_sum}}
```

## Beispiel 4: Gradienten im Embedding

Nun verwenden wir das Embedding aus Kapitel 3 mit der Eingabe `[[1, 1, 3]]`,
also "Hund Hund rennt". Als einfaches Ergebnis summieren wir alle Zahlen der
Ausgabe:

```text
loss = sum(embedding([[1, 1, 3]]))
```

Die Tabellen müssen wir nicht mit `require_grad()` markieren: Burn macht das
für Modellparameter (`Param`) automatisch.

```rust,ignore
{{#include ../../../crates/model/examples/ch04.rs:embedding}}
```

### Der Token-Gradient

Jede Zahl der Ausgabe geht mit dem Faktor `1` in die Summe ein. Ihr Gradient
ist deshalb `1`. Burn gibt diese `1` an die Tabellenzeile zurück, aus der die
Zahl gelesen wurde. Wird eine Zeile mehrfach gelesen, werden ihre Beiträge
addiert:

```text
Zeile 0 (Der)    [0, 0, 0, 0]   nicht benutzt
Zeile 1 (Hund)   [2, 2, 2, 2]   zweimal benutzt
Zeile 2 (Katze)  [0, 0, 0, 0]   nicht benutzt
Zeile 3 (rennt)  [1, 1, 1, 1]   einmal benutzt
Zeile 4 (.)      [0, 0, 0, 0]   nicht benutzt
```

Nur Zeilen von Tokens, die im Text vorkommen, erhalten einen Gradienten. Ein
Training würde also nur diese Zeilen verändern.

### Der Positions-Gradient

Die Ausgabe des Embeddings ist eine Addition:

```text
Ausgabe = Token-Vektor + Positions-Vektor
```

Wird einer der beiden Summanden etwas grösser, wird die Ausgabe um genau so viel
grösser. Eine Addition gibt den Gradienten deshalb unverändert an beide
Summanden weiter. Auch die Positions-Vektoren erhalten so den Gradienten `1`.

In unserem Fenster kommt jede Position genau einmal vor:

```text
Token:     Hund  Hund  rennt
Position:   0     1     2
```

Daher hat jede Zeile der Positions-Tabelle den Gradienten `[1, 1, 1, 1]`.
Mit zwei Fenstern im Batch würde jede Position zweimal benutzt; jede Zeile
hätte dann den Gradienten `[2, 2, 2, 2]`.

### Warum die Gewichte zufällig sein dürfen

Anders als in Kapitel 3 verwendet die Demo die zufälligen Startgewichte. Die
Ableitung einer Summe hängt nicht von den summierten Werten ab, sondern nur
davon, wie oft jede Tabellenzeile benutzt wurde. Die Gradienten sind deshalb
bei jedem Lauf gleich.

## Demo

```bash
cargo run -p model --example ch04
```

```rust,ignore
{{#include ../../../crates/model/examples/ch04.rs}}
```

Ausgabe:

```text
{{#include ../ausgaben/ch04.txt}}
```

Die Werte stimmen mit den Rechnungen in diesem Kapitel überein.

## Was noch fehlt

Wir können jetzt für jeden Parameter berechnen, in welche Richtung er sich
ändern müsste. Die Summe als Ergebnis war nur ein Lernbeispiel. Im Training
messen wir stattdessen mit einer Verlustfunktion, wie gut die Vorhersage des
Modells ist. Ein Optimizer verändert dann die Parameter anhand ihrer
Gradienten. Beides folgt im Kapitel zum Training.
