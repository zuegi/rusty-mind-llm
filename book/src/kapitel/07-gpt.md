# 7. GPT: Das Sprachmodell

## Lernziel

Die vorherigen Kapitel haben Bausteine gebaut. Jetzt verbinden wir sie zu
einem kleinen GPT-artigen Sprachmodell:

```text
Token-IDs [B, T]
  → GptEmbedding (Token + Position) [B, T, E]
  → Transformer-Block 1             [B, T, E]
  → ...
  → Transformer-Block N             [B, T, E]
  → Ausgabeprojektion               [B, T, V]
  → Logits
```

`B` ist die Anzahl Fenster im Batch, `T` die Kontextlänge, `E` die
Embedding-Dimension und `V` die Vokabulargröße. Das Modell lebt weiter im
Crate `model`; Daten, Loss, Optimizer und Training bleiben im Crate
`training`.

## Von Vektoren zu nächsten Token

Ein Transformer-Block liefert für jede Position einen Vektor mit `E` Zahlen.
Zum Vorhersagen braucht das Modell aber einen Wert für **jede** Token-ID:
Welche IDs passen als Nächstes?

Eine Linear-Schicht mit Gewichten der Form `[E, V]` übersetzt daher jeden
Vektor in `V` Zahlen:

```text
versteckter Vektor [E]  →  Ausgabeprojektion [E, V]  →  Logits [V]
```

Diese Zahlen heißen **Logits**. Sie sind noch keine Wahrscheinlichkeiten.
Später wandelt Softmax sie in Wahrscheinlichkeiten um und Cross-Entropy
vergleicht sie mit dem richtigen nächsten Token.

Der Logit an Position `t` beschreibt die Vorhersage für das Token **nach**
Position `t`. Durch die kausale Maske in Attention hängt er nur von den
Token-IDs bis `t` ab, nie von späteren IDs.

## Modell konfigurieren

Die Konfiguration bündelt alle Architekturgrößen:

```rust,ignore
{{#include ../../../crates/model/src/gpt.rs:config}}
```

- `vocab_size`: Anzahl möglicher Token-IDs und damit Länge jedes Logit-Vektors.
- `max_context_length`: maximale Anzahl Eingabetokens. Das begrenzt die
  Positions-Embedding-Tabelle.
- `embedding_dim`: Länge eines Token-Vektors.
- `num_heads`: Köpfe in jedem Transformer-Block. `embedding_dim` muss durch
  diese Zahl teilbar sein.
- `num_blocks`: wie oft der Transformer-Block hintereinander vorkommt.

`GptModel` hält ein `gpt_embedding: GptEmbedding`, die Transformer-Blöcke
und eine neue `output_projection`. `GptEmbedding` ist der Baustein aus
Kapitel 3: Token- und Positions-Embedding mit Ausgabe `[B, T, E]`. Die
Projektion hat keine Bias-Werte, damit sie nur eine eigene Matrix `[E, V]`
beiträgt.

## Initialisierung

```rust,ignore
{{#include ../../../crates/model/src/gpt.rs:init}}
```

`GptEmbeddingConfig` erzeugt das `GptEmbedding` mit Token- und
Positions-Tabellen. Dieselbe
`TransformerBlockConfig` erzeugt `num_blocks` unabhängige Blöcke: Sie haben
denselben Aufbau, aber **keine** gemeinsamen Gewichte.

Die Ausgabeprojektion ist ebenfalls unabhängig von der Token-Tabelle. Diese
Variante heißt **kein Weight Tying**:

```text
Token-Embedding:   [V, E]  eigene Parameter
Ausgabeprojektion: [E, V]  eigene Parameter
```

Manche GPT-Modelle teilen beide Matrizen (Weight Tying); das spart Parameter.
Hier verwenden wir bewusst getrennte Matrizen, damit Eingabe-Nachschlagen und
Ausgabe-Vorhersage als zwei klar unterscheidbare Schritte sichtbar bleiben.

## Vorwärtsdurchlauf

```rust,ignore
{{#include ../../../crates/model/src/gpt.rs:forward}}
```

`fold` führt den versteckten Zustand nacheinander durch alle Blöcke. Die
Ausgabeprojektion rechnet auf der letzten Achse und erhält die Form:

```text
Token-IDs:      [B, T]
GptEmbedding:   [B, T, E]
nach N Blöcken: [B, T, E]
Logits:         [B, T, V]
```

Es gibt absichtlich kein Softmax im Modell. Training braucht rohe Logits für
eine numerisch stabile Cross-Entropy-Berechnung. Auch beim Generieren wird
Softmax erst nach Wahl der letzten Position angewendet.

## Ausführen

Das Beispiel baut ein Modell mit `V = 50`, `T = 4`, `E = 8`, zwei Köpfen und
zwei Blöcken. Es druckt die Tensorform, Parameterzahl und fünf Logits der
letzten Position:

```rust,ignore
{{#include ../../../crates/model/examples/ch07.rs}}
```

```bash
cargo run -p model --example ch07
```

Die konkreten Logits unterscheiden sich pro Lauf, weil Burn Gewichte zufällig
initialisiert. Stabil bleiben die Formen:

```text
Eingabe-Form: [1, 4]
GptEmbedding:  [1, 4, 8]
Logits-Form:   [1, 4, 50]
```

Das Modell kennt noch keine Wörter: Ohne Training sind seine Logits nur
Startwerte. Der sinnvolle nächste Schritt ist deshalb kein größeres Modell,
sondern Targets, Cross-Entropy und ein Optimizer.
Das folgt in [Kapitel 8: Cross-Entropy und der erste
Trainingsschritt](08-loss-und-trainingsschritt.md).

## Tests

```rust,ignore
{{#include ../../../crates/model/tests/gpt.rs}}
```

```bash
cargo test -p model --test gpt
```

Die Tests prüfen:

1. Jeder Eingabeposition liefert einen Logit für jede Vokabular-ID.
2. Die Ausgabeprojektion hat ihre eigene Form `[E, V]` und ihre Parameter
   zählen zusätzlich zu Embedding und Blöcken.
3. Eine Änderung eines späteren Tokens verändert keine früheren Logits.
4. Autodiff erreicht `GptEmbedding`, Transformer-Block und Ausgabeprojektion.
5. Ein Modell ohne Transformer-Block wird abgelehnt.
