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

**Informationen zusammenzutragen und aus ihrer Kombination neue Merkmale
zu berechnen sind zwei unterschiedliche Aufgaben.** Attention übernimmt
den Austausch zwischen Token-Positionen. Das FFN ergänzt eine lernbare,
nichtlineare Verarbeitung der Informationen, die an einer Position
zusammengekommen sind.

Betrachten wir zur Anschauung wieder:

```text
Die Bank am Fluss war nass.
Die Bank genehmigt den Kredit.
```

An der Position „Fluss“ kann die kausale Attention Informationen aus
„Bank“ und „am“ aufnehmen. Dort liegen nun Hinweise auf „Bank“ und auf
eine Flussumgebung gemeinsam in einem Vektor vor. Das FFN kann lernen,
auf diese Kombination anders zu reagieren als auf jeden Hinweis allein.
So kann eine für die weitere Vorhersage nützliche Darstellung entstehen.
Die Wörter dienen nur der Anschauung; echte Tokens und Vektorkomponenten
tragen keine von uns fest vergebenen Bedeutungen.

**Warum genügt die Mischung nicht?** Der Kern der Attention bildet
gewichtete Summen der Value-Vektoren. Auch die Output-Projektion `W_O`
verrechnet die Kopf-Ausgaben linear. Das FFN bietet zusätzlich die
Möglichkeit, mit GELU auf Kombinationen von Komponenten nichtlinear zu
reagieren. Es kann dadurch neue Merkmale berechnen, statt nur die
vorhandenen Beiträge anders zu gewichten. Attention selbst ist durch
ihre eingabeabhängigen Softmax-Gewichte bereits nichtlinear; das FFN
ergänzt sie um eine andere, gezielt auf die Komponenten gerichtete
Verarbeitung.

Es wäre möglich, ein Modell ohne FFN zu bauen. **Für die Ausführbarkeit
der Attention ist es nicht nötig.** Im GPT-Block verwenden wir es, um
die Ausdrucksfähigkeit des Modells zu erweitern: Zwischen Positionen
Informationen austauschen und diese innerhalb einer Position verarbeiten
wechseln sich von Block zu Block ab.

Das FFN liest dafür keine anderen Positionen direkt. Sein Eingabevektor
enthält aber bereits den von Attention aufgenommenen Kontext
(der ganze Datenweg steht in [6.4](06-04-block-zusammensetzen.md)).
Dieselbe FFN-Funktion wird auf jeden Token-Vektor **einzeln** angewendet.
Man nennt es Feed-Forward-Netz (FFN), weil bei der Vorwärtsberechnung
die Daten ohne Rückkopplung durch seine Schichten laufen.
Das bedeutet nicht, dass die Parameter unveränderlich sind: Beim Training
berechnet Autograd auch durch das FFN rückwärts die Gradienten.

Anders als LayerNorm berechnet das FFN keinen Mittelwert und keine Varianz.
Es lernt Kombinationen der vorhandenen Komponenten. Wir geben nicht vor,
welche Komponente etwa eine Wortart erkennen soll; welche Kombinationen
nützlich sind, ergibt sich erst durch das Training.

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

## Handrechnung

Wir verfolgen einen einzelnen Token-Vektor `x = [1, -1]` durch alle drei
Schritte. Damit bleiben wir bei `E = 2` und der Zwischenbreite
`hidden_dim = 4 * E = 8`. Die folgenden Gewichte und Bias-Werte sind
**für die Handrechnung vorgegeben**, nicht aus einem trainierten Modell
übernommen. Die Demo weiter unten verwendet dagegen zufällige Startgewichte.

### 1. Erweitern: Acht Kombinationen aus zwei Zahlen berechnen

Eine lineare Schicht berechnet `x @ W + b`: Matrixmultiplikation und
anschliessend komponentenweise Addition des Bias-Vektors `b`.
Der Bias ist ein lernbarer Zusatzwert pro Ausgabekomponente.

Für `expand` hat die Gewichtsmatrix `W_expand` die Form `[2, 8]`,
der Bias `b_expand` die Form `[8]`:

```text
             h0  h1  h2  h3  h4  h5  h6  h7
W_expand = [[ 1,  0,  1,  1,  0,  0,  0,  0],
            [ 0,  1,  1, -1,  0,  0,  0,  0]]
b_expand =  [ 0,  0,  0,  0,  1, -1,  0,  0]
```

Jede Spalte berechnet eine Zwischenkomponente aus den beiden Eingabezahlen:

```text
h0 = 1 * 1 + (-1) *  0 + 0 =  1
h1 = 1 * 0 + (-1) *  1 + 0 = -1
h2 = 1 * 1 + (-1) *  1 + 0 =  0
h3 = 1 * 1 + (-1) * -1 + 0 =  2
h4 = 1 * 0 + (-1) *  0 + 1 =  1
h5 = 1 * 0 + (-1) *  0 - 1 = -1
h6 = 1 * 0 + (-1) *  0 + 0 =  0
h7 = 1 * 0 + (-1) *  0 + 0 =  0

h = [1, -1, 0, 2, 1, -1, 0, 0]
```

**Erweitern heisst nicht, den Vektor mit Nullen aufzufüllen oder Tokens
hinzuzufügen.** Wir berechnen acht Zwischenwerte aus denselben zwei
Eingabekomponenten. Einige Gewichte sind hier nur für die Übersicht null;
im Modell sind alle Einträge der Matrix lernbar.

### 2. GELU: Jede Zwischenkomponente nichtlinear verändern

GELU (Gaussian Error Linear Unit) ist eine feste Aktivierungsfunktion.
Sie wird auf jede Zahl einzeln angewendet und hat selbst keine lernbaren
Parameter. Einige Werte, auf drei Nachkommastellen gerundet:

```text
GELU(-1) ≈ -0.159
GELU( 0) =  0
GELU( 1) ≈  0.841
GELU( 2) ≈  1.954

g = GELU(h)
  ≈ [0.841, -0.159, 0, 1.954, 0.841, -0.159, 0, 0]
```

Negative Werte werden also nicht grundsätzlich auf null gesetzt.
Die Form bleibt unverändert: acht Zahlen hinein, acht Zahlen heraus.
Durch diese wertabhängige Veränderung kann das gesamte FFN mehr darstellen
als eine einzige lineare Schicht. Die Herleitung und ein Vergleich mit
ReLU stehen im [GELU-Exkurs](../exkurse/aktivierung-gelu.md).

### 3. Zurückführen: Zwei neue Ausgabekomponenten berechnen

`contract` ist eine zweite, eigenständige lineare Schicht. Ihre Matrix
`W_contract` hat die Form `[8, 2]`, ihr Bias `b_contract` die Form `[2]`.
Für unser Beispiel wählen wir:

```text
                 y0  y1
W_contract = [[   1,  0],   Beitrag von g0
              [   1,  0],   Beitrag von g1
              [   0,  0],   Beitrag von g2
              [   0,  1],   Beitrag von g3
              [   0, -1],   Beitrag von g4
              [   0,  0],   Beitrag von g5
              [   0,  0],   Beitrag von g6
              [   0,  0]]   Beitrag von g7
b_contract = [0.5, -0.5]
```

Damit erhalten wir, mit gerundeten Zwischenwerten:

```text
y0 = g0 + g1 + 0.5 = 0.841 - 0.159 + 0.5 = 1.182
y1 = g3 - g4 - 0.5 = 1.954 - 0.841 - 0.5 = 0.613

Eingabe x = [1, -1]
Ausgabe y ≈ [1.182, 0.613]
```

`contract` macht die erste Schicht **nicht rückgängig**. Es berechnet
neue Kombinationen der aktivierten Zwischenwerte. Die ursprüngliche Breite
ist wiederhergestellt, nicht der ursprüngliche Inhalt. Die Residual-Addition
kommt erst im Block dazu; sie gehört nicht zu `FeedForward::forward`.

## Welche Achse?

Wie in [6.1](06-01-layernorm.md) verwenden wir `[B, T, E]`:
Batch-Grösse, Token-Positionen und Embedding-Komponenten.
`Linear` wirkt auf die letzte Achse, hier Achse `2`.
Anders als `mean_dim(2)` fasst es diese Achse nicht zu einem Mittelwert
zusammen, sondern berechnet eine neue Anzahl von Komponenten.

Für einen Batch mit drei Tokens und `E = 2`:

```text
Eingabe:   [1, 3, 2]  drei Token-Vektoren mit je zwei Zahlen
expand:    [1, 3, 8]  drei Token-Vektoren mit je acht Zwischenwerten
GELU:      [1, 3, 8]  gleiche Form, veränderte Zahlen
contract:  [1, 3, 2]  drei neue Token-Vektoren mit je zwei Zahlen
```

**Batch-Grösse und Tokenanzahl bleiben bei allen Schritten gleich.**
Jedes Token wird unabhängig verarbeitet. Erhalten zwei Positionen exakt
denselben Eingabevektor, liefert dieses FFN dort dieselbe Ausgabe.
Im Transformer können auch gleiche Token-IDs unterschiedliche Vektoren
haben: Position und bisheriger Kontext stecken bereits in ihrer Eingabe.

## Was ist daran lernbar?

Lernbar sind die beiden Gewichtsmatrizen und die beiden Bias-Vektoren.
Bei `E = 2` zählen wir:

```text
expand:    W_expand [2, 8] + b_expand [8] = 16 + 8 = 24 Parameter
contract:  W_contract [8, 2] + b_contract [2] = 16 + 2 = 18 Parameter
GELU:                                                   0 Parameter
Summe:                                                 42 Parameter
```

Diese Parameter werden für alle Tokens und Batches dieses FFN
wiederverwendet. Mehr Tokens bedeuten mehr Rechenarbeit, aber nicht mehr
FFN-Parameter. Der Faktor `4` ist eine fest gewählte Architekturentscheidung,
kein Wert, den das Training selbst verändert.

## Der Code

Die folgenden Ausschnitte stammen aus `crates/model/src/feed_forward.rs`.

### Die Struktur: Zwei lineare Schichten

```rust,ignore
{{#include ../../../crates/model/src/feed_forward.rs:config}}
```

`expand` und `contract` sind zwei `Linear`-Schichten. Anders als bei den
Attention-Projektionen haben sie einen Bias (Standard von `LinearConfig`).
Burns `Linear` verwaltet seine Gewichtsmatrix und seinen Bias bereits als
lernbar. Deshalb stehen hier keine eigenen `Param`-Felder wie bei LayerNorm.
`#[derive(Module, Debug)]` bindet die Teilschichten in Burns Modellverwaltung
ein und ermöglicht Debug-Ausgaben.

### In `init`: Die beiden Breiten festlegen

```rust,ignore
{{#include ../../../crates/model/src/feed_forward.rs:init}}
```

`FeedForwardConfig::init` prüft die Eingabebreite und berechnet die
Zwischenbreite `hidden_dim`. `EXPANSION_FACTOR` ist im Quellcode als `4`
festgelegt. `LinearConfig::new(eingabebreite, ausgabebreite)` erzeugt
zuerst die Schicht von `E` nach `4·E`, dann die Schicht von `4·E` nach `E`.
Die Gewichte werden initialisiert, nicht aus unserer Handrechnung übernommen.

### In `forward`: Erweitern und GELU anwenden

```rust,ignore
{{#include ../../../crates/model/src/feed_forward.rs:expand_activate}}
```

Dieser Ausschnitt aus `FeedForward::forward` fasst die ersten beiden
Rechenschritte zusammen. Zuerst berechnet `self.expand.forward(input)`
die Matrixmultiplikation samt Bias. Danach verändert `gelu(...)` jede
Zwischenkomponente einzeln. `hidden` hat die Form `[B, T, 4·E]`.
Es ist keine zusätzliche Tokenfolge, sondern dieselbe Folge mit mehr
Komponenten pro Position.

### In `forward`: Auf die Embedding-Breite zurückführen

```rust,ignore
{{#include ../../../crates/model/src/feed_forward.rs:contract}}
```

Die zweite Schicht berechnet aus den Zwischenwerten die Ausgabe mit Form
`[B, T, E]`. Der Ausdruck hat kein Semikolon, weil er als letzte Zeile
der Methode direkt deren Rückgabewert ist.

<details>
<summary>Vollständiger Code: FeedForward::forward</summary>

```rust,ignore
{{#include ../../../crates/model/src/feed_forward.rs:forward}}
```

</details>

## Ausführen

Die Demo in `crates/model/examples/ch06_feed_forward.rs` führt genau
die drei beschriebenen Schritte aus. Sie prüft Formen und die gleiche
Behandlung gleicher Eingabevektoren, nicht die festen Zahlen der Handrechnung.

<details>
<summary>Vollständiger Code der Demo</summary>

```rust,ignore
{{#include ../../../crates/model/examples/ch06_feed_forward.rs}}
```

</details>

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

Bei `E = 8` liegen in unserem FFN mehr Parameter als in der Attention
(552 gegen 256, siehe [6.6](06-06-bloecke-stapeln.md)).
Das FFN ist also trotz seiner kurzen `forward`-Methode ein erheblicher
Teil der lernbaren Modellberechnung.

## Formen mit einem Test prüfen

Der Test `feed_forward_keeps_shape` in `crates/model/tests/transformer_block.rs`
gibt dem FFN einen Tensor `[2, 3, 4]` und erwartet dieselbe Ausgabeform.
Er prüft damit die Voraussetzung für die spätere Residual-Addition,
nicht die numerische Richtigkeit aller Rechenschritte.

```bash
cargo test -p model --test transformer_block feed_forward
```

## Selbst versuchen

### Breiten und Parameter zählen

Ein FFN erhält `[2, 3, 4]`, also zwei Fenster mit je drei Tokens und
`E = 4`. Welche Form hat `hidden` nach GELU? Welche Form hat die Ausgabe?
Wie viele lernbare Parameter haben die beiden linearen Schichten
einschliesslich Bias? Ändert sich diese Zahl bei sechs statt drei Tokens?

<details>
<summary>Lösung anzeigen</summary>

Die Zwischenbreite ist `4 * E = 16`: `hidden` hat die Form `[2, 3, 16]`,
die Ausgabe wieder `[2, 3, 4]`.

```text
expand:   4 * 16 + 16 = 80
contract: 16 * 4 + 4  = 68
Summe:                148 Parameter
```

Bei sechs Tokens werden dieselben 148 Parameter häufiger angewendet.
Die Tokenanzahl verändert die Tensorformen und Rechenarbeit, nicht
die Parameterzahl dieses FFN.

</details>

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
