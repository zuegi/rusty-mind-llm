# Aktivierungsfunktionen und GELU

Dieser Exkurs gehört zu Kapitel 6, Schritt 2 (Feed-Forward-Netz). Er beantwortet
drei Fragen: Warum braucht ein Netz überhaupt eine Aktivierungsfunktion? Was ist
ReLU? Und was macht GELU anders?

## Warum zwei Linear-Schichten nicht genügen

Eine Linear-Schicht multipliziert mit einer Matrix (und addiert einen Vektor).
Nimm zur Vereinfachung einzelne Zahlen statt Matrizen:

```text
Schicht 1:  h = 2 · x
Schicht 2:  y = 3 · h
Zusammen:   y = 3 · (2 · x) = 6 · x
```

Zwei Schichten hintereinander sind wieder nur **eine** Multiplikation, hier
mit `6`. Bei Matrizen gilt dasselbe: `W₂ · (W₁ · x) = (W₂ · W₁) · x`. Die
Matrix `W₂ · W₁` ist wieder eine einzige Matrix. Beliebig viele Linear-Schichten
ohne etwas dazwischen können also nie mehr als eine einzige.

Eine einzelne Linear-Schicht kann nur **gerade** Zusammenhänge darstellen:
Verdoppelt sich die Eingabe, verdoppelt sich die Ausgabe. Sprache folgt keinem
solchen Muster. Dazu braucht das Netz eine **Nichtlinearität**, also eine
Funktion, deren Ausgabe nicht proportional zur Eingabe wächst. Sie steht
zwischen den Linear-Schichten und heisst **Aktivierungsfunktion**.

Anschaulich in Kotlin: Eine Kette von `map { it * a }` bleibt ein `map { it * c }`.
Erst ein `if` dazwischen (zum Beispiel "negative Werte abschneiden") macht daraus
etwas Neues.

## ReLU: der einfachste Knick

```text
ReLU(x) = max(0, x)
```

Negative Werte werden zu 0, positive bleiben unverändert. Das ist ein "Knick"
bei `x = 0` und reicht schon für Nichtlinearität. Zwei Nachteile:

- Bei `x = 0` hat die Funktion eine Ecke, also keine eindeutige Ableitung.
- Alles Negative wird exakt 0. Dort ist die Ableitung ebenfalls 0, und das
  Training kann für diese Werte keine Richtung mehr erkennen.

## GELU: ein glatter Knick

GELU steht für *Gaussian Error Linear Unit*. Die Idee: Wir multiplizieren `x`
mit einer Zahl zwischen 0 und 1, die angibt, wie "wahrscheinlich" `x` behalten
werden soll.

```text
GELU(x) = x · Φ(x)
```

`Φ(x)` (sprich: Phi) ist die Wahrscheinlichkeit, dass eine normalverteilte
Zufallszahl kleiner als `x` ist. Du musst die Formel nicht herleiten. Wichtig
sind die Eigenschaften:

- Für grosse `x` ist `Φ(x)` fast 1, also `GELU(x) ≈ x`.
- Für sehr negative `x` ist `Φ(x)` fast 0, also `GELU(x) ≈ 0`.
- Dazwischen verläuft die Kurve **glatt** und überall ableitbar.

## Die Zahlen

```rust,ignore
{{#include ../../../crates/model/examples/ch06_gelu.rs}}
```

```bash
cargo run -p model --example ch06_gelu
```

```text
{{#include ../ausgaben/ch06_gelu.txt}}
```

Zum Lesen:

- Ab `x = 2` sind ReLU und GELU fast gleich (`2.000` gegen `1.954`).
- Bei kleinen negativen Werten lässt GELU **etwas** durch (`-0.159` bei
  `x = -1`), ReLU nichts. Der Wert ist leicht negativ, danach geht er
  wieder gegen 0. Deshalb ist die Kurve bei `-1` kurz unter null.
- Bei `x = 0.5` liefert GELU `0.346` statt `0.5`: Es dämpft kleine positive Werte
  sanft statt sie voll durchzulassen.

## Warum GPT-2 GELU nimmt

GELU ist glatt, und Gradienten verschwinden für negative Werte nicht abrupt.
Praktisch bewährt hat es sich in GPT-2 und BERT, daher übernehmen wir es.
Burn bringt `burn::tensor::activation::gelu` fertig mit. Die ursprüngliche
GPT-2-Implementierung nutzte eine schnellere Näherung mit `tanh`; sie
unterscheidet sich nur in den hinteren Nachkommastellen.
