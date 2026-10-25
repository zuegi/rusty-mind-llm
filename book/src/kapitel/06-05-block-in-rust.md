# 6.5 Den Block in Rust bauen

**Lernziel:** Den Datenfluss des Blocks in seinen Rust-Feldern und
den vier Rechenschritten von `forward` wiederfinden.

## Anschluss: Vom Diagramm zum Modell

Jetzt setzen wir das Diagramm aus [6.4](06-04-block-zusammensetzen.md)
in Code um. Der Block fasst LayerNorm, FFN und Residual-Verbindungen
aus den vorherigen Seiten und die `MultiHeadAttention` aus
Kapitel 5 zusammen.

## Der Code

```rust,ignore
{{#include ../../../crates/model/src/transformer_block.rs:config}}
```

Der Block hat vier Felder: zwei LayerNorms, die Attention und das FFN. Es gibt
zwei LayerNorms, weil jede ihre eigenen `γ` und `β` lernt. Burns
`#[derive(Module)]` sammelt die Parameter aller Felder automatisch ein.

```rust,ignore
{{#include ../../../crates/model/src/transformer_block.rs:forward}}
```

Die vier Zeilen entsprechen den Formeln:

| Zeile im Code | Formel |
| ------------- | ------ |
| `normed = attention_norm.forward(input.clone())` | `LayerNorm₁(x)` |
| `after_attention = input + attention.forward(normed)` | `x₁ = x + Attention(…)` |
| `normed = feed_forward_norm.forward(after_attention.clone())` | `LayerNorm₂(x₁)` |
| `after_attention + feed_forward.forward(normed)` | `x₂ = x₁ + FeedForward(…)` |

Zwei Rust-Punkte:

- **`clone()`:** `input` wird zweimal gebraucht, in der LayerNorm und in der
  Addition. Tensor-Operationen verbrauchen ihre Eingabe, also geben wir der
  LayerNorm einen Clone. Dasselbe gilt für `after_attention`. Der Clone kopiert
  keine Daten.
- **Shadowing:** `let normed` steht zweimal. Die zweite Zeile legt eine neue
  Variable gleichen Namens an und verdeckt die erste. Die erste wurde schon an
  die Attention übergeben und wird nicht mehr gebraucht.

Die kausale Maske steckt in `MultiHeadAttention`. Der Block erbt sie, ohne
selbst etwas zu tun.

## Selbst versuchen

### Warum zwei LayerNorms?

Könnten `attention_norm` und `feed_forward_norm` einfach dieselbe Instanz
sein? Was wäre dann anders?

<details>
<summary>Lösung anzeigen</summary>

Die Formen würden passen, aber beide Stellen müssten dann dieselben
Parameter `γ` und `β` verwenden. Unser Block hat zwei eigenständige
LayerNorms: Eine verarbeitet `x` vor Attention, die andere `x₁` vor dem
FFN. So kann jede ihre eigene Skalierung und Verschiebung lernen.

</details>

## Was bleibt?

**Vier Felder bilden den Block; `forward` setzt die zwei Residual-Schritte um.**
Die beiden LayerNorms besitzen eigene Parameter. `clone()` erhält die
jeweilige Eingabe für ihre spätere Addition.

Zurück: [6.4 Den Transformer-Block zusammensetzen](06-04-block-zusammensetzen.md).
Weiter: [6.6 Mehrere Blöcke stapeln](06-06-bloecke-stapeln.md).
