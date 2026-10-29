# 6.3 Residual-Verbindungen

**Lernziel:** Die Addition `x + f(x)` als direkten Informations- und
Gradientenpfad erklären und ihre Grenzen kennen.

## Anschluss: Änderungen statt vollständiger Neuerzeugung

Attention aus Kapitel 5 und das FFN aus [6.2](06-02-feed-forward.md)
berechnen neue Token-Vektoren. Jetzt verbinden wir ihre Ergebnisse
mit den bisherigen Vektoren, statt ausschliesslich den neuen Zweig zu verwenden.

## Das Problem: Gradienten verschwinden

Kapitel 4 hat gezeigt: Beim Rückwärtsrechnen multipliziert Autograd die
Ableitungen aller Schichten. Wirkt jede Schicht nur schwach (Faktor `0.1`),
schrumpft der Gradient mit jeder Schicht um den Faktor 10. Nach zehn Schichten
kommt am Anfang fast nichts mehr an. Die ersten Schichten können dann nicht
mehr lernen.

## Die Lösung: Eingabe dazuaddieren

```text
ohne Residual:  y = Schicht(x)
mit  Residual:  y = x + Schicht(x)
```

Die Schicht muss den Vektor nicht komplett neu erzeugen. Wir interpretieren
ihre Ausgabe als **Änderung**, die auf `x` aufgerechnet wird.
Für ein skalares Beispiel ist die Ableitung von `y = x + f(x)`
gleich `1 + f'(x)`. Die `1` stammt aus der direkten Verbindung.

Bei einem Vektor wird jede Eingabekomponente direkt zu ihrer
entsprechenden Ausgabekomponente addiert:

```text
y[0] = x[0] + f(x)[0]
y[1] = x[1] + f(x)[1]
```

Betrachten wir nur diesen direkten Weg, ohne den Zweig `f(x)`:
Steigt `x[0]` um 1, steigt auch `y[0]` um 1; `y[1]` erhält dadurch
keinen direkten Beitrag. Für `x[1]` gilt dasselbe entsprechend.
Die direkte Verbindung gibt also Änderungen jeder Komponente
unverändert weiter. Der Zweig `f(x)` kann zusätzlich andere Änderungen
beitragen.

<details>
<summary>Mathematischer Begriff: Identitätsmatrix</summary>

Bei mehreren Ein- und Ausgabekomponenten beschreiben wir die Ableitungen
mit einer Matrix: Jede Zeile gehört zu einer Ausgabekomponente,
jede Spalte zu einer Eingabekomponente. Für den direkten Weg oben ist das:

```text
                         Eingabe
                        x[0]  x[1]
Ausgabe y[0]              1     0
Ausgabe y[1]              0     1
```

Die Einsen bedeuten: Die jeweilige Komponente wird unverändert
weitergegeben. Die Nullen bedeuten: Über diesen Weg gibt es keinen
Einfluss auf die andere Komponente. Diese Matrix heisst
**Identitätsmatrix**. Sie übernimmt bei Vektoren die Rolle der `1`
im skalaren Beispiel. Im Rückwärtsdurchlauf reicht dieser direkte
Weg auch den ankommenden Gradienten unverändert weiter; hinzu kommt
der Beitrag des Zweigs `f(x)`.

</details>

Dieser direkte Gradientenpfad hilft, auch tiefe Netze zu trainieren; er garantiert
aber weder stabile Gradienten noch den Erhalt aller Informationen.

## Wann kommt die Addition zum Zug?

Residual ist keine eigene Schicht, sondern eine Addition. Sie steht **direkt
nach** einer Schicht, im Block also zweimal:

```text
x
├─→ LayerNorm → Attention ─┐
└──────────────────────────┴→ ⊕  = x₁     Residual 1
```

Zahlenbeispiel mit `E = 2` (die Zahlen sind erfunden):

```text
x                    = [1.0,  2.0]
Attention-Ausgabe    = [0.1, -0.2]
x₁ = x + Ausgabe     = [1.1,  1.8]
```

In diesem Beispiel ist die Änderung klein. Liefert der Zweig `[0, 0]`,
bleibt `x₁ = x`: Er kann also nichts beitragen, ohne die Eingabe zu
verändern. Das ist keine allgemeine Erhaltungsgarantie: Bei einer Ausgabe
`-x` wäre die Summe null. Der zweite Teilschritt im Block (Feed-Forward)
folgt demselben Additionsmuster.

## Ausprobieren

Die Demo schaltet bis zu zehn Schichten hintereinander. Jede Schicht multipliziert
mit `0.1` (eine bewusst schwache Schicht). Wir fragen nach dem Gradienten am
Eingang:

```rust,ignore
{{#include ../../../crates/model/examples/ch06_residual.rs:stack}}
```

- `require_grad()` markiert `x`, damit Burn dessen Gradienten merkt.
- `backward()` rechnet rückwärts, `x.grad(&grads)` liest den Gradienten für `x`.

Die Variablen in der Schleife haben zwei unterschiedliche Aufgaben:
`hidden` enthält den bisherigen Wert, also die Eingabe der aktuellen
Schicht. `update` ist deren neu berechnete Ausgabe:
`hidden * 0.1`. Die Konstante `LAYER_SCALE` ist dafür auf `0.1` gesetzt.

Mit `residual` wählen wir, wie es danach weitergeht:

```text
Bei hidden = 1.0 berechnet die Schicht zuerst update = 1.0 * 0.1 = 0.1.

Mit Residual:  neues hidden = hidden + update = 1.0 + 0.1 = 1.1
Ohne Residual: neues hidden = update                    = 0.1
```

**Mit Residual addieren wir die Schichtausgabe zur bisherigen Eingabe.
Ohne Residual ersetzen wir die Eingabe durch die Schichtausgabe.**
Genau das entscheidet die Codezeile
`hidden = if residual { hidden + update } else { update };`.
Der neue Wert von `hidden` geht dann in den nächsten Schleifendurchlauf.
„Update“ bedeutet hier die berechnete Änderung am Zwischenwert,
nicht eine Aktualisierung der Modellparameter durch einen Optimizer.

```bash
cargo run -p model --example ch06_residual
```

```text
{{#include ../ausgaben/ch06_residual.txt}}
```

Ohne Residual wird der Gradient mit jeder Schicht zehnmal kleiner
(`1e-10` nach zehn Schichten). Mit Residual ist der Faktor pro Schicht
`1 + 0.1 = 1.1`; nach zehn Schichten beträgt der Gradient
`1.1^10 ≈ 2.594`. Er schrumpft hier also nicht exponentiell gegen null,
sondern wächst. Die Demo zeigt den zusätzlichen Gradientenpfad,
nicht das genaue Verhalten eines echten Transformer-Trainings.

## Formen müssen passen

Bei `x + Schicht(x)` addieren wir nicht zwei Tensoren als ganze Zahlen.
Die Addition erfolgt **komponentenweise**: Jede Zahl aus `x` wird zur
Zahl an derselben Stelle der Schichtausgabe addiert.

### Was muss übereinstimmen?

Die Tensorform (Shape) `[B, T, E]` beschreibt drei Achsen:

| Achse | Bedeutung | Voraussetzung für unsere Residual-Addition |
|---|---|---|
| `B` (Batch Size) | Anzahl der Fenster im Batch | dieselben Fenster in derselben Reihenfolge |
| `T` (Sequence Length) | Anzahl der Token-Positionen pro Fenster | dieselben Positionen in derselben Reihenfolge |
| `E` (Embedding Dimension) | Anzahl der Komponenten pro Token-Vektor | gleich viele Komponenten mit entsprechender Indexzuordnung |

Für unsere Residual-Verbindung haben Eingabe und Schichtausgabe deshalb
**exakt dieselbe Form**. Gleiche Anzahl von Zahlen allein genügt nicht:
`[1, 2, 4]` und `[1, 4, 2]` enthalten beide acht Zahlen, ordnen sie aber
unterschiedlichen Positionen und Komponenten zu.

### Beispiel: Zwei Token-Vektoren addieren

Wir betrachten einen Batch mit einem Fenster, zwei Tokens und jeweils
zwei Komponenten. Beide Tensoren haben die Form `[1, 2, 2]`:

```text
             Eingabe x       Schichtausgabe       Summe
Token 0:     [1.0, 2.0]   +  [ 0.1, -0.2]    =  [1.1, 1.8]
Token 1:     [3.0, 4.0]   +  [-0.5,  0.3]    =  [2.5, 4.3]
```

Die erste Komponente von Token 0 wird nur zur ersten Komponente von
Token 0 addiert, nicht zu einer Komponente von Token 1. Die Residual-Addition
mischt keine Token-Positionen und hängt auch keine Werte aneinander.
Anders als bei Concat bleibt die Breite `E = 2`, und die Zahlen werden
miteinander verrechnet.

Die Schichtausgabe muss dabei nicht denselben Inhalt wie `x` haben:
Sie liefert neue Werte, die wir an den entsprechenden Stellen addieren.
**Gleiche Form bedeutet gleiche Anordnung, nicht gleiche Zahlen.**

### Warum führt das FFN wieder auf `E` zurück?

Im FFN aus [6.2](06-02-feed-forward.md) ändert sich die Breite intern.
Für die Eingabe aus dem Beispiel sieht der Weg so aus:

```text
Eingabe x:          [1, 2, 2] ───────────────────────────┐
                                                       │
FFN-Zweig:         [1, 2, 2] → expand → [1, 2, 8]        │
                            → GELU   → [1, 2, 8]        │
                            → contract → [1, 2, 2]      │
                                                       ↓
Residual-Addition: [1, 2, 2]          +          [1, 2, 2]
Ergebnis:                            [1, 2, 2]
```

Nach `expand` hat jeder Token-Vektor acht Komponenten, die ursprüngliche
Eingabe aber nur zwei. Eine direkte Addition dieser Formen passt nicht.
`contract` berechnet deshalb wieder zwei Ausgabekomponenten pro Token.
Es schneidet nicht einfach sechs Zahlen ab und stellt auch nicht die
ursprünglichen Werte wieder her.

`MultiHeadAttention` liefert ebenfalls wieder `[B, T, E]`: Die Köpfe
werden zusammengefügt und ihre Ausgaben projiziert. So passen in
beiden Teilschritten des Blocks die Ein- und Ausgaben zur Residual-Addition.
Die LayerNorm vor Attention beziehungsweise FFN verändert die Form nicht.

<details>
<summary>Warum verwenden wir hier kein Broadcasting?</summary>

Broadcasting kann bei Tensor-Additionen Achsen der Länge 1 auf passende
grössere Achsen ausdehnen. Beispielsweise könnte ein Tensor `[1, 1, E]`
zu `[B, T, E]` addiert werden: Alle Fenster und Positionen erhielten
dann denselben Zusatzvektor.

Das ist etwa beim Bias einer linearen Schicht nützlich, entspricht aber
nicht unserer Residual-Verbindung. Dort soll jede Position ihre eigene,
von der Schicht berechnete Ausgabe erhalten. Deshalb verwenden wir
hier zwei Tensoren mit exakt gleicher Form, statt uns auf Broadcasting
zu verlassen.

</details>

## Selbst versuchen

### Was passiert bei einem Residual-Zweig mit Ausgabe null?

Attention liefert im ersten Teilschritt überall null. Welcher Vektor
geht dann in den zweiten Teilschritt? Ist damit auch die Ausgabe des
gesamten Blocks unverändert?

<details>
<summary>Lösung anzeigen</summary>

Es gilt `x₁ = x + 0 = x`. Der zweite Teilschritt erhält also die
ursprüngliche Eingabe. Sein FFN kann trotzdem eine Änderung beitragen:
Erst wenn auch dieser Zweig null liefert, ist `x₂ = x`.

</details>

## Was bleibt?

**Residual bedeutet: Eingabe und berechnete Änderung addieren.**
Der direkte Pfad unterstützt den Gradientenfluss, garantiert aber
keinen verlustfreien Informationserhalt. Beide Summanden haben bei uns
exakt die Form `[B, T, E]`.

Zurück: [6.2 Feed-Forward-Netz](06-02-feed-forward.md).
Weiter: [6.4 Den Transformer-Block zusammensetzen](06-04-block-zusammensetzen.md).
