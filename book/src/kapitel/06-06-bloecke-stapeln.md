# 6.6 Mehrere Blöcke stapeln

**Lernziel:** Mehrere eigenständige Blöcke verketten und dabei
Tensorformen, Parameterzahl und Kausalität nachvollziehen.

## Anschluss: Ein Block ist fertig

In [6.5](06-05-block-in-rust.md) haben wir den Transformer-Block in Rust
zusammengesetzt. Jetzt verwenden wir mehrere Instanzen hinter dem Embedding.

Bisher haben wir einen Block betrachtet. Ein GPT stapelt viele davon
(GPT-2 small: 12). Das geht, weil jeder Block `[B, T, E]` annimmt und
`[B, T, E]` zurückgibt: Die Ausgabe des einen ist die Eingabe des nächsten.

```mermaid
flowchart LR
    ids["Token-IDs [B, T]"] --> emb["GptEmbedding"]
    emb --> b1["Block 1"]
    b1 --> b2["Block 2"]
    b2 --> b3["Block 3"]
    b3 --> out["Ausgabe [B, T, E]"]
```

Jeder Block hat **eigene** Parameter. Block 2 ist also nicht Block 1 ein
zweites Mal, sondern eine neue Instanz mit eigenen Gewichten.

## Blöcke hintereinander

```rust,ignore
{{#include ../../../crates/model/examples/ch06.rs:stack}}
```

`fold` reicht einen Wert durch eine Kette von Schritten. Start ist die
Embedding-Ausgabe. Jeder Block bekommt das bisherige `hidden` und liefert das
nächste. Das entspricht in Kotlin `blocks.fold(start) { hidden, block -> block(hidden) }`.

## Ausführen

```bash
cargo run -p model --example ch06
```

```text
{{#include ../ausgaben/ch06.txt}}
```

Die Ausgabe hat drei Teile:

1. **Vektoren pro Stufe.** Wir zeigen die ersten vier Zahlen von Token 0 nach
   dem Embedding und nach jedem Block. Die Werte ändern sich in jedem Block, die
   Form bleibt `[1, 4, 8]`. Der Block berechnet also neue Token-Vektoren;
   die Residual-Pfade beziehen die bisherigen Vektoren direkt ein.
2. **Parameter.** Drei Blöcke mit `E = 8` haben `3 · 840 = 2520` Parameter.
   Die Rechnung für einen Block steht unten.
3. **Kausalität.** Wir ändern das *letzte* Token (4 → 9). Die Ausgaben von
   Token 0 bis 2 bleiben exakt gleich, nur Token 3 ändert sich. Token blicken
   nur nach hinten, deshalb kann eine spätere Änderung frühere Positionen nicht
   beeinflussen. Das ist die kausale Maske aus Kapitel 5, die jetzt durch drei
   Blöcke hindurch hält.

Die konkreten Zahlen hängen von der zufälligen Startinitialisierung ab. Bei
einem erneuten Lauf sehen sie anders aus. Die Formen, die Parameterzahl und das
Ergebnis der Kausalitätsprüfung bleiben gleich.

Die Parameterzahl eines Blocks lässt sich nachrechnen (`E = 8`):

```text
Attention:     4 Matrizen à 8·8                 = 256
LayerNorm:     2 × (γ: 8 + β: 8)                =  32
Feed-Forward:  (8·32 + 32) + (32·8 + 8)         = 552
Summe                                           = 840
```

## Tests

```bash
cargo test -p model --test transformer_block
```

Die Tests prüfen:

- Die Form bleibt erhalten, auch hinter `GptEmbedding`.
- **Kausalität:** Ändert sich Token 1, bleibt die Ausgabe für Token 0
  unverändert.
- Gradienten kommen bei Attention, Feed-Forward und beiden LayerNorms an.
- Die Parameterzahl entspricht der Rechnung oben.

## Die Zusammenhänge wiederfinden

- **Formen:** Die Bausteine erhalten an ihren Ein- und Ausgängen
  `[B, T, E]`. Das ermöglicht die Residual-Addition und das
  Hintereinanderschalten. Ihre internen Tensoren dürfen andere Formen haben.
- **Achsen:** Softmax normiert über Keys, LayerNorm über Embedding-Komponenten,
  das Feed-Forward-Netz rechnet je Position.
- **Rust:** `clone()` erhält einen Tensor für die zweite Verwendung. Es kopiert
  die Daten nicht.
- **Pre-LayerNorm:** Der Residual-Pfad bleibt ohne LayerNorm. Nur der Weg durch
  die Schicht wird normiert.

## Selbst versuchen

Zwei Blöcke mit `E = 8` erhalten nach dem Embedding die Form `[2, 3, 8]`.
Welche Form liegt nach Block 2 vor? Wie viele Parameter haben die beiden
Blöcke zusammen, ohne Embedding? Darf eine Änderung an Token 2 die Ausgabe
für Token 0 beeinflussen?

<details>
<summary>Lösung anzeigen</summary>

Die Form bleibt `[2, 3, 8]`. Beide Blöcke besitzen eigene Parameter,
zusammen also `2 * 840 = 1680`. Die kausale Maske verhindert einen Einfluss
von Token 2 auf Token 0. LayerNorm und FFN verarbeiten jede Position
einzeln; die Residual-Additionen mischen ebenfalls keine Positionen.
Deshalb bleibt diese Eigenschaft auch über mehrere Blöcke erhalten.

</details>

## Was bleibt?

Ein Transformer-Block verbindet Multi-Head-Attention und Feed-Forward-Netz.
Beide sind mit LayerNorm davor und einer Residual-Verbindung darum versehen.
Die Form bleibt gleich, daher lassen sich Blöcke stapeln.

## Was als Nächstes kommt

Das Modell besteht noch aus Einzelteilen. Das nächste Kapitel stapelt mehrere
Blöcke hinter dem Embedding und ergänzt eine Ausgabeschicht, die für jede
Position Werte für alle Tokens des Vokabulars liefert.

Zurück: [6.5 Den Block in Rust bauen](06-05-block-in-rust.md).
Weiter: [7. GPT: Das Sprachmodell](07-gpt.md).
