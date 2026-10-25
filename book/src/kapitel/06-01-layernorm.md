# 6.1 LayerNorm verstehen und bauen

**Lernziel:** Einen Token-Vektor normieren und die lernbare Anpassung mit
`γ` und `β` in der Handrechnung und im Burn-Code wiederfinden.

## Anschluss: Kontext-Vektoren weiterverarbeiten

In [5.9](05-09-mehrere-koepfe.md) haben wir Multi-Head Attention gebaut.
Der Transformer-Block ergänzt unter anderem LayerNorm. Sie verarbeitet
die Komponenten eines Tokens, ohne Informationen anderer Positionen einzumischen.

## Kurz erklärt: Achsen

Eine **Achse** ist eine Richtung im Tensor. Jede Zahl in der Form steht für
eine Achse. Gezählt wird ab 0. Das entspricht der Verschachtelungstiefe, etwa
in Kotlin:

```text
Vektor  [4]       1 Achse    List<Float>
Matrix  [3, 4]    2 Achsen   List<List<Float>>
Tensor  [2, 3, 4] 3 Achsen   List<List<List<Float>>>
```

Unser Tensor `[B, T, E]` hat drei Achsen: `B` steht für die Batch-Grösse
(Batch Size), `T` für die Sequenzlänge (Sequence Length) und `E` für die
Einbettungsbreite (Embedding Dimension, `embedding_dim`).

| Achse | Name | Bedeutung                       |
| ----- | ---- | ------------------------------- |
| 0     | B    | welches Fenster                 |
| 1     | T    | welches Token im Fenster        |
| 2     | E    | welche Komponente des Vektors   |

Bei `[2, 4, 8]` gibt es 2 Fenster mit je 4 Tokens und je 8 Zahlen.

**Über eine Achse rechnen** heisst: Diese Achse wird zusammengefasst, alle
anderen bleiben getrennt. `mean_dim(2)` mittelt für jedes (Fenster, Token)
die 8 Zahlen und liefert die Form `[2, 4, 1]`. Merkhilfe: Achse 2 ist die
innerste Liste.

## Das Problem

Nach mehreren Schichten können die Zahlen eines Vektors sehr gross oder sehr
klein werden und das Training erschweren. LayerNorm verarbeitet deshalb
jeden Token-Vektor einzeln in zwei Schritten:

1. **Normieren:** Mittelwert abziehen und durch die stabilisierte
   Standardabweichung teilen. Das ergibt rechnerisch Mittelwert 0.
   Bei einer Varianz deutlich grösser als `ε` liegt die neue Varianz nahe 1.
2. **Lernbar anpassen:** Jede Komponente mit einem eigenen Faktor `γ`
   skalieren und um einen eigenen Wert `β` verschieben.

Die **Varianz** ist der Mittelwert der quadrierten Abweichungen vom
Mittelwert. `[10, 10, 30, 30]` streut weit (Varianz 100),
`[0.1, 0.1, 0.3, 0.3]` eng (Varianz 0.01).
LayerNorm macht diese unterschiedlichen Skalen vergleichbarer.
Die endgültige Ausgabe muss wegen `γ` und `β` aber weder Mittelwert 0
noch Varianz 1 haben.

## Handrechnung

Der Token-Vektor `x = [1, 2, 3, 4]`:

```text
Mittelwert  μ  = (1 + 2 + 3 + 4) / 4              = 2.5
Abweichungen   = [-1.5, -0.5, 0.5, 1.5]
Varianz     σ² = (2.25 + 0.25 + 0.25 + 2.25) / 4  = 1.25
Konstante   ε  = 0.00001
Nenner        = √(1.25 + ε)                      ≈ 1.118
Normiert    x̂  = Abweichungen / Nenner            ≈ [-1.342, -0.447, 0.447, 1.342]
```

Die Varianz ist der Mittelwert der quadrierten Abweichungen. Sie ist ein Mass
dafür, wie weit die Zahlen streuen.

Danach folgt ein lernbarer Schritt:

```text
Ausgabe[i] = γ[i] * x̂[i] + β[i]
```

`γ` (scale) und `β` (shift) sind zwei Vektoren der Länge `E`. Sie starten mit
`γ = 1` und `β = 0`. Das Training darf sie ändern, falls das Modell andere
Bereiche braucht. Zu Beginn ändern sie nichts.
Die Rechnung erfolgt **komponentenweise**, nicht als Skalarprodukt:
Für jede Komponente `i` gibt es einen Faktor und einen Verschiebungswert.
Dieselben beiden Parametervektoren werden für alle Tokens und Fenster
dieser LayerNorm verwendet; es gibt keine eigenen Parameter pro Position.

Eine winzige Konstante `ε` (epsilon, hier `1e-5`) wird zur Varianz addiert.
Hätte ein Vektor lauter gleiche Zahlen, wäre die Varianz 0 und die Division
`0 / 0` ergäbe `NaN`. Mit `ε` bleibt das Ergebnis 0. `ε` ist fest und wird
nicht gelernt, deshalb ist es ein normales Feld und kein `Param`.
Bei gleicher Eingabe ist der normierte Vektor dann null; die endgültige
Ausgabe nach der lernbaren Anpassung ist `β`.

## Welche Achse?

Wir normieren über **Achse 2**, die Embedding-Achse. Für jedes Token werden
also dessen `E` Zahlen zusammen betrachtet: ein Mittelwert und eine Varianz pro
Token. Andere Tokens und andere Fenster fliessen nicht ein.

```text
Eingabe [2, 4, 8]  →  2 · 4 = 8 Token-Vektoren mit je 8 Zahlen
                      →  8 Mittelwerte und 8 Varianzen, Form [2, 4, 1]
```

Entscheidend ist die **Bedeutung der Achse**, nicht eine feste Nummer:

| Operation | Tensorform | Achse | Worüber wird gerechnet? |
| --------- | ---------- | ----- | ----------------------- |
| Softmax im einzelnen Kopf | `[B, T, T]` | 2 | Key-Positionen pro Query |
| Softmax in Multi-Head Attention aus 5.9 | `[B, Köpfe, T, T]` | 3 | Key-Positionen pro Query und Kopf |
| LayerNorm | `[B, T, E]` | 2 | Komponenten eines Token-Vektors |

## Der Code

```rust,ignore
{{#include ../../../crates/model/src/layer_norm.rs:config}}
```

`Param` markiert Tensoren, die Burn als lernbar behandelt. `scale` und `shift`
sind Vektoren (`Tensor<1>`).

### In `init`: Startwerte erzeugen

```rust,ignore
{{#include ../../../crates/model/src/layer_norm.rs:init}}
```

`LayerNormConfig::init` erzeugt `scale` (`γ`) mit Einsen und `shift`
(`β`) mit Nullen, jeweils mit `embedding_dim` Komponenten.
`epsilon` übernimmt den festen Wert aus der Konfiguration; der Standard
ist `1e-5`.

### In `forward`: Den Token-Vektor normieren

Der folgende Ausschnitt stammt aus `LayerNorm::forward` in
`crates/model/src/layer_norm.rs`, nach der Prüfung der Embedding-Breite:

```rust,ignore
{{#include ../../../crates/model/src/layer_norm.rs:normalize}}
```

- `mean_dim(2)` bildet den Mittelwert über die Embedding-Achse. Die Form wird
  `[B, T, 1]`: Die Achse bleibt mit Länge 1 erhalten.
- `input - mean` zieht von jedem Wert den Mittelwert seines Tokens ab.
  Burn streckt die Achse der Länge 1 automatisch (Broadcasting).
- `centered.clone()`: `centered` wird zweimal gebraucht, für die Varianz und
  für die Division. Tensor-Operationen verbrauchen ihre Eingabe (Move), also
  brauchen wir einen Clone. Er kopiert die Daten nicht.
- `powf_scalar(2.0).mean_dim(2)` berechnet die Varianz aus den quadrierten
  Abweichungen. Die Division durch `sqrt(variance + epsilon)` entspricht
  der Handrechnung.

### In `forward`: Lernbare Skalierung und Verschiebung

```rust,ignore
{{#include ../../../crates/model/src/layer_norm.rs:affine}}
```

`scale` und `shift` haben Form `[E]`, `normalized` hat `[B, T, E]`.
Für diese Tensoroperationen benötigt Burn gleichen Rang.
`reshape([1, 1, E])` ergänzt deshalb zwei Achsen der Länge 1.
Broadcasting verwendet dieselben Parameter an allen Fenstern und Positionen.
`normalized * scale + shift` ist die komponentenweise Anpassung.

<details>
<summary>Vollständiger Code: LayerNorm::forward</summary>

```rust,ignore
{{#include ../../../crates/model/src/layer_norm.rs:forward}}
```

</details>

## Ausführen

```rust,ignore
{{#include ../../../crates/model/examples/ch06_layer_norm.rs}}
```

```bash
cargo run -p model --example ch06_layer_norm
```

```text
{{#include ../ausgaben/ch06_layer_norm.txt}}
```

Token 0 entspricht der Handrechnung. Token 1 hat vor der Normierung die
Varianz 100. Bei den Startwerten `γ = 1` und `β = 0` haben beide Ausgaben
eine Varianz nahe 1. `ε` verhindert exakte Varianz 1; spätere Änderungen
von `γ` und `β` können die Ausgabeskala zusätzlich verändern.

## Gegenprobe mit Burn

Wir haben LayerNorm selbst gebaut. Stimmt die Rechnung? Burn bringt
`burn::nn::LayerNorm` mit. Wenn beide bei gleicher Eingabe und gleichen
Parametern dasselbe liefern, stützt das unsere Implementierung für die
geprüften Fälle. Es ergänzt die Handrechnung, ist aber kein Beweis für
jede mögliche Eingabe.

Was die Tests tun:

- **Erster Test:** Beide Module starten mit `γ = 1` und `β = 0`. Die Eingabe hat
  zwei Tokens. Die Ausgaben werden Zahl für Zahl verglichen (`zip` paart je
  eine Zahl von uns mit der von Burn).
- **Zweiter Test:** Wir setzen in beiden Modulen dieselben Werte für `γ` und
  `β`. So prüfen wir auch den lernbaren Teil `γ · x̂ + β`, der mit den
  Startwerten nichts verändert.
- **Toleranz statt Gleichheit:** Wir vergleichen mit `< 1e-4` statt mit `==`.
  Gleitkommazahlen runden. Burn rechnet intern in anderer Reihenfolge
  (eine fertig zusammengefasste Operation des Backends), daher können die
  letzten Nachkommastellen abweichen, ohne dass die Rechnung anders wäre.

<details>
<summary>Vollständiger Code der Vergleichstests mit Burn</summary>

```rust,ignore
{{#include ../../../crates/model/tests/layer_norm.rs:compare}}
```

</details>

Zwei Unterschiede im Aufbau fallen auf:

| Unser Modul | Burns Modul          |
| ----------- | -------------------- |
| `scale`     | `gamma`              |
| `shift`     | `beta` (optional)    |
| nur `[B, T, E]` (`Tensor<3>`) | beliebiger Rang |

Wir haben bewusst nur den Fall gebaut, den der Transformer-Block braucht.
Im Rest des Kapitels bleibt unser eigenes `LayerNorm` im Block. Du kannst es
später durch `burn::nn::LayerNorm` ersetzen, ohne das Verhalten zu ändern.

```bash
cargo test -p model --test layer_norm
```

Weitere Tests prüfen Mittelwert und Varianz je Token, Gradienten für `scale`
und `shift` und die Fehlermeldung bei falscher Embedding-Breite.

## Selbst versuchen

Ein Token-Vektor lautet `[5, 5, 5, 5]`. Was ergibt der Normierungsschritt
mit `ε`? Was liefert die gesamte LayerNorm bei `γ = 1` und
`β = [1, 2, 3, 4]`?

<details>
<summary>Lösung anzeigen</summary>

Der Mittelwert ist 5, alle Abweichungen und die Varianz sind null.
Durch `ε` bleibt der Nenner positiv: Der normierte Vektor ist
`[0, 0, 0, 0]`. Die lernbare Anpassung liefert anschliessend
`[1, 2, 3, 4]`, also `β`. Die endgültige Ausgabe muss deshalb nicht
Mittelwert 0 haben.

</details>

## Was bleibt?

**LayerNorm normiert je Token über die Embedding-Komponenten und passt
das Ergebnis komponentenweise lernbar an.** Die Form `[B, T, E]` bleibt erhalten.

Zurück: [6. Übersicht](06-transformer-block.md).
Weiter: [6.2 Feed-Forward-Netz](06-02-feed-forward.md).
