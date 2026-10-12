# Ableitungen für Programmierer

Dieser Exkurs erklärt die wenigen Regeln, die für Kapitel 4 nötig sind. Burn
wendet sie automatisch an; wir müssen sie nur verstehen, nicht selbst
programmieren.

## Die Idee: Wie stark reagiert das Ergebnis?

Eine Funktion nimmt eine Zahl `x` und liefert eine Zahl `y`. Die Ableitung
beantwortet eine Frage: Wenn ich `x` um ein winziges Stück `h` verändere, um
wie viel ändert sich dann `y`?

Dafür teilen wir die Änderung von `y` durch die Änderung von `x`:

```text
(f(x + h) - f(x)) / h
```

Dieser Ausdruck heisst *Differenzenquotient*. Lassen wir `h` immer kleiner
werden, nähert er sich einer festen Zahl: der **Ableitung** `f'(x)`, auch
`dy/dx` geschrieben.

## Geometrisch: Die Steigung

Zeichnet man `y = f(x)` als Kurve, ist die Ableitung die Steigung der Kurve an
einer Stelle:

- positive Ableitung: Die Kurve steigt; grösseres `x` gibt grösseres `y`.
- negative Ableitung: Die Kurve fällt; grösseres `x` gibt kleineres `y`.
- Ableitung `0`: Die Kurve ist an dieser Stelle flach.

Im Training interessiert uns genau das: In welche Richtung muss sich ein
Parameter bewegen, damit der Fehler kleiner wird?

## Herleitung für `y = x²`

```text
(x + h)² - x² = x² + 2xh + h² - x² = 2xh + h²
```

Geteilt durch `h`:

```text
(2xh + h²) / h = 2x + h
```

Wird `h` winzig, bleibt `2x`. Also gilt `d(x²)/dx = 2x`. Bei `x = 3` ist das
`6`, bei `x = -2` ist es `-4`.

## Die Regeln aus Kapitel 4

| Funktion | Ableitung nach `x` | Bedeutung |
| -------- | ------------------ | --------- |
| `c` (Konstante) | `0` | hängt nicht von `x` ab |
| `c · x` | `c` | jede Änderung von `x` wirkt `c`-fach |
| `x²` | `2x` | Herleitung oben |
| `f + g` | `f' + g'` | Summen leitet man einzeln ab |

### Was bedeutet `loss` in Beispiel 3?

**Loss** bedeutet *Verlust*. Im Training ist das eine einzelne Zahl, die misst,
wie stark eine Modellvorhersage vom gewünschten Ergebnis abweicht. Ein kleinerer
Loss bedeutet eine bessere Vorhersage gemessen an dieser Verlustfunktion.

In Beispiel 3 verwenden wir bewusst eine einfachere Rechnung:

```text
w    = [0.5, -1, 2]   Gewichte
x    = [1,    2, 3]   feste Eingaben
loss = w₀·x₀ + w₁·x₁ + w₂·x₂
     = 0.5·1 + (-1)·2 + 2·3
     = 4.5
```

Der Index `0` bezeichnet das erste Element, `1` das zweite und `2` das dritte.
Wir multiplizieren jede Eingabe mit ihrem Gewicht und addieren die Ergebnisse.

Der Name `loss` bleibt hier derselbe wie im Training. Diese Summe ist aber
**noch kein Vorhersagefehler**: Es fehlt der Vergleich mit einem gewünschten
Ergebnis. Sie dient nur dazu, das Ableiten mit mehreren Gewichten zu verstehen.
Die eigentliche Verlustfunktion folgt im Training-Kapitel.

Für die Ableitung nach `w₀` halten wir die Eingaben und die anderen Gewichte
fest. Nur `w₀·x₀` hängt dann von `w₀` ab; die beiden anderen Summanden sind
Konstanten mit Ableitung `0`. Nach der Regel für `c · x` bleibt `x₀ = 1`.
Genauso ergibt die Ableitung nach `w₁` den Wert `x₁ = 2` und nach `w₂` den Wert
`x₂ = 3`. Der Gradient der Gewichte ist deshalb `[1, 2, 3]`.

### Die Summe beim Embedding

Auch das Embedding in Beispiel 4 folgt daraus. Die Summe aller Ausgabezahlen
enthält jede gelesene Tabellenzahl mit dem Faktor `1`. Wird eine Zeile zweimal
gelesen, erscheint sie zweimal in der Summe; ihre Ableitung ist dann `1 + 1 = 2`.

## Die Kettenregel

Oft ist eine Funktion aus Schritten zusammengesetzt:

```text
u = 2x + 1
y = u²
```

Die Kettenregel sagt: Die Reaktionen der einzelnen Schritte werden
multipliziert.

```text
dy/dx = dy/du · du/dx
```

Anschaulich: Ändert sich `x` um ein Stück, ändert sich `u` doppelt so stark
(`du/dx = 2`). Jede Änderung von `u` wirkt auf `y` mit dem Faktor `2u`. Bei
`x = 3` ist `u = 7`, also:

```text
dy/dx = 2u · 2 = 14 · 2 = 28
```

Ein neuronales Netz ist eine sehr lange Kette solcher Schritte. Autograd geht
sie vom Ergebnis aus rückwärts durch und multipliziert die Faktoren Schritt
für Schritt. Darum heisst die Methode auch *Backpropagation*.

## Mehrere Variablen

Hängt ein Ergebnis von mehreren Zahlen ab, leiten wir nach jeder Zahl einzeln
ab und halten dabei die anderen fest. Das nennt man *partielle Ableitung*. Alle
partiellen Ableitungen zusammen bilden den **Gradienten**. Er hat dieselbe Form
wie die Parameter selbst: Für die Token-Tabelle `[5, 4]` ist auch der Gradient
`[5, 4]`.
