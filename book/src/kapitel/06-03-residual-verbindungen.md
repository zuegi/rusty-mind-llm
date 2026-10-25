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
Bei Vektoren entspricht ihr eine Identitätsmatrix. Dieser direkte
Gradientenpfad hilft, auch tiefe Netze zu trainieren; er garantiert
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
- `hidden + update` ist die Residual-Addition, `update` allein ist die Variante
  ohne.
- `backward()` rechnet rückwärts, `x.grad(&grads)` liest den Gradienten für `x`.

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

Tensor-Additionen erlauben unter bestimmten Bedingungen auch Broadcasting.
Für unsere Residual-Verbindung wollen wir aber **exakt dieselbe Form**:
Jede Komponente soll zur entsprechenden Komponente desselben Tokens
addiert werden. Deshalb muss `Schicht(x)` wieder `[B, T, E]`
zurückgeben, wenn `x` diese Form hat.

`MultiHeadAttention` und `FeedForward` tun genau das. Hätte das FFN nach
`expand` aufgehört und `4·E` Zahlen geliefert, ginge die Addition nicht auf.
Deshalb gibt es `contract`.

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
