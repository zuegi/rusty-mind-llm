# 6.2 Feed-Forward-Netz

**Lernziel:** Du verstehst, wie das Feed-Forward-Netz (FFN) die durch
Attention aufgenommenen Informationen innerhalb jedes Token-Vektors
weiterverarbeitet, ohne selbst zwischen Positionen zu mischen. Du kannst
nachvollziehen, wie die erste lineare Schicht die Einbettungsbreite
(Embedding Dimension, `E`) auf `4·E` erweitert und die zweite sie wieder
auf `E` zurückführt. Dabei lernst du, warum dazwischen die nichtlineare
Aktivierungsfunktion GELU gebraucht wird und warum alle Tokens mit
denselben Parametern verarbeitet werden. Zum Schluss findest du diese
Schritte und die zugehörigen Tensorformen im Burn-Code wieder.

## Anschluss: Normieren ist noch keine Merkmalsverarbeitung

[6.1](06-01-layernorm.md) hat LayerNorm erklärt. Das Feed-Forward-Netz
(FFN) ist ein weiterer Baustein: Es berechnet aus den Komponenten eines
Tokens neue Merkmale. Wie beide im Block zusammenspielen, folgt in
[6.4](06-04-block-zusammensetzen.md).

## Warum noch eine Schicht?

Attention hat Informationen **zwischen** Positionen gemischt. Danach soll jeder
Token-Vektor für sich weiterverarbeitet werden: Was hat das Token aus seinem
Kontext gelernt, und was folgt daraus? Das übernimmt ein kleines Netz, das auf
jeden Token-Vektor **einzeln** angewendet wird, also ohne Blick auf andere
Positionen. Der Vektor enthält durch Attention aber bereits Kontext (der
ganze Datenweg steht in [6.4](06-04-block-zusammensetzen.md)). Man nennt es Feed-Forward-Netz
(FFN), weil Daten nur vorwärts durch zwei Schichten laufen.

## Aufbau

```text
E  →  Linear (expand)  →  4·E  →  GELU  →  Linear (contract)  →  E
```

- **expand** weitet den Vektor von `E` auf `4·E` Zahlen. Der Faktor 4 ist der
  GPT-2-Standard. Im weiten Raum hat das Netz mehr Platz, Merkmale zu trennen.
- **GELU** bringt die Nichtlinearität. Ohne sie wären zwei Linear-Schichten
  nur eine einzige Matrix (siehe
  [Exkurs: Aktivierungsfunktionen und GELU](../exkurse/aktivierung-gelu.md)).
- **contract** verengt wieder auf `E`. Nur so passt die Form zur
  Residual-Addition in [6.3](06-03-residual-verbindungen.md).

Für `E = 2` kann man die Breiten an einem einzelnen Token verfolgen:

```text
Eingabe:      [x0, x1]                         2 Komponenten
expand:      [h0, h1, h2, h3, h4, h5, h6, h7] 8 Zwischenwerte
GELU:        [g0, g1, g2, g3, g4, g5, g6, g7] 8 veränderte Werte
contract:    [y0, y1]                         2 Ausgabekomponenten
```

Die Buchstaben stehen für berechnete Zahlen, nicht für zusätzliche Tokens.
GELU verändert die Zwischenwerte abhängig von ihrem Wert nichtlinear.
Ohne diese Nichtlinearität liessen sich beide linearen Schichten samt
Bias zu einer einzigen affinen Abbildung zusammenfassen. Mit GELU kann
das Netz Zusammenhänge darstellen, die eine solche Abbildung nicht erfasst.
**Alle Tokens durchlaufen dieselbe Funktion mit denselben Parametern**;
es gibt kein eigenes FFN pro Position.

## Der Code

```rust,ignore
{{#include ../../../crates/model/src/feed_forward.rs:config}}
```

`expand` und `contract` sind zwei `Linear`-Schichten. Anders als bei den
Attention-Projektionen haben sie einen Bias (Standard von `LinearConfig`).

```rust,ignore
{{#include ../../../crates/model/src/feed_forward.rs:forward}}
```

Die Methode hat nur zwei Zeilen: erst `expand` und `gelu`, dann `contract`.
Der Eingang ist `Tensor<3>` mit Form `[B, T, E]`. `Linear` wirkt auf die
letzte Achse, Achse 2. Alle Fenster und Positionen werden so mit denselben
Gewichten behandelt, jede für sich.

## Ausführen

```rust,ignore
{{#include ../../../crates/model/examples/ch06_feed_forward.rs}}
```

```bash
cargo run -p model --example ch06_feed_forward
```

```text
{{#include ../ausgaben/ch06_feed_forward.txt}}
```

Zum Lesen:

- Die Form wird in der Mitte breiter (`8 = 4 · 2`) und am Ende wieder `[1, 3, 2]`.
- Token 0 und Token 2 haben denselben Eingabevektor und bekommen dieselbe
  Ausgabe. Das FFN weiss nichts von Positionen und Nachbarn. Sie kommen allein
  aus Attention und Positions-Embedding.
- Parameter bei `E = 2`: `(2·8 + 8) + (8·2 + 2) = 24 + 18 = 42`.

Bei `E = 8` liegen im FFN mehr Parameter als in der Attention (552 gegen 256,
siehe [6.6](06-06-bloecke-stapeln.md)). Ein grosser Teil des Wissens eines GPT steckt in diesen
Schichten.

```bash
cargo test -p model --test transformer_block feed_forward
```

## Selbst versuchen

### Wie kennt das FFN den Kontext?

Das FFN liest keine Nachbar-Tokens. Warum kann seine Ausgabe trotzdem
von vorherigen Tokens abhängen?

<details>
<summary>Lösung anzeigen</summary>

Seine Eingabe `x₁` enthält bereits Informationen, die Attention aus
erlaubten Positionen eingemischt hat. Das FFN verarbeitet diesen
kontextabhängigen Vektor einzeln weiter, ohne selbst zwischen Positionen
zu mischen. Dieselbe FFN-Funktion wird an jeder Position angewendet.

</details>

## Was bleibt?

**Attention mischt zwischen Positionen; das FFN verarbeitet jede Position
einzeln mit derselben Funktion.** Es erweitert die Merkmalsbreite intern
auf `4·E` und liefert wieder `E` Komponenten.

Zurück: [6.1 LayerNorm verstehen und bauen](06-01-layernorm.md).
Weiter: [6.3 Residual-Verbindungen](06-03-residual-verbindungen.md).
