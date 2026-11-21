# 17. Regularisierung gegen Overfitting

## Lernziel

In [Kapitel 13](13-auswertung-und-overfitting.md) haben wir
Overfitting an der Lücke zwischen Trainings- und Validierungs-Loss
erkannt und den besten Stand gespeichert. In
[Kapitel 14](14-bessere-optimierung.md) haben wir Adam als Optimizer
kennengelernt. Jetzt fragen wir: **Wie begrenzen wir, wie stark sich das
Modell an die Trainingsdaten anpasst?**

Ohne Begrenzung übernimmt das Training jede Parameteränderung, die den
Trainings-Loss senkt, auch reines Auswendiglernen. Regularisierung bremst
solche Änderungen, damit das Modell Muster lernt, die auch auf unbekannten
Daten gelten.

Wir betrachten zwei Verfahren: **Weight Decay** (über den Optimizer AdamW)
und **Dropout** (im Transformer-Block). Beide sind unabhängig
voneinander: Weight Decay wirkt im Optimizer, Dropout im Modell. Wir
vergleichen sie einzeln, kombiniert und mit einem Lauf ohne
Regularisierung:

```bash
cargo run --release -p training --example ch17
```

Ohne Argument liest die Demo `data/the-verdict.txt`; ein optionales
Argument wählt eine andere UTF-8-Textdatei. Alle vier Varianten starten
mit **identischen Anfangsgewichten** und verwenden dieselben Batches.
Die echte Ausgabe und ihre Grenzen stehen in Abschnitt 5.

## 1. Was wir dagegen tun können

Overfitting zeigt sich als wachsende **Lücke**: Der Trainings-Loss sinkt
weiter, der Validierungs-Loss steigt oder verbessert sich nicht mehr.

```text
Training:    sinkt  ->  sinkt  ->  sinkt
Validierung: sinkt  ->  steigt  ->  steigt
                         ^ ab hier merkt sich das Modell Beispiele
```

Zu Beginn ist die Lücke nahe null, weil das Modell noch nichts gelernt
hat. Beim Auswendiglernen wächst sie. Zwei Verfahren bremsen das:

- **Weight Decay:** Die Gewichte werden bei jedem Update zusätzlich
  verkleinert.
- **Dropout:** Beim Training werden zufällig ausgewählte Aktivierungen
  ausgeblendet.

Beide **bremsen** Änderungen, die nur die Trainingsdaten bedienen. Sie
garantieren keinen besseren Validierungs-Loss.

## 2. Weight Decay: grosse Gewichte verteuern

Beim Auswendiglernen wachsen Gewichte oft stark, damit das Modell
einzelne Beispiele sehr sicher vorhersagen kann. **Weight Decay**
zieht deshalb jedes Gewicht bei jedem Update ein kleines Stück Richtung
null.

Der Optimizer dafür heisst **AdamW**. Er ist Adam aus Kapitel 14 mit
einem direkten Verkleinerungsschritt:

```text
Gewicht_neu = Gewicht_alt - Lernrate * Adam-Richtung
                          - Lernrate * Weight_Decay * Gewicht_alt
```

Ein Zahlenbeispiel mit den Werten unserer Demo (Lernrate `0.01`,
Weight Decay `0.1`) und einem Gewicht `2.0`. Nehmen wir an, die
Adam-Richtung wäre null:

```text
Verkleinerung = 0.01 * 0.1 * 2.0 = 0.002
Gewicht_neu   = 2.0 - 0.002      = 1.998
```

Pro Update schrumpft ein Gewicht um den Faktor `1 - 0.01 * 0.1 = 0.999`.
Nach 240 Updates wäre ohne jede andere Kraft nur noch
`0.999^240 ≈ 0.79` des Gewichts übrig. In Wirklichkeit wirken die
Gradienten dagegen. Weight Decay ist also ein stetiger Zug zu kleineren
Gewichten, kein Zwang.

**Weight Decay ist keine zusätzliche Loss-Komponente.** Die
Verkleinerung passiert direkt im Update und nicht im Gradienten.
Dadurch fliesst sie nicht in Adams Momentum und Skalierung ein. Das
unterscheidet AdamW von klassischer L2-Regularisierung.

### Optimizer in Rust auswählen

```rust
{{#include ../../../crates/training/examples/ch17.rs:optimizer}}
```

`with_weight_decay(0.0)` ist AdamW ohne Verkleinerung. So unterscheiden
sich die Varianten nur in einem Wert: "Keine Regularisierung" und
"Dropout" verwenden `0.0`, nur "Weight Decay" verwendet
`WEIGHT_DECAY = 0.1`.

## 3. Dropout: Aktivierungen zufällig ausblenden

**Dropout** setzt bei jedem Trainingsschritt einen Anteil `p` der
Aktivierungen auf null. Die übrigen Werte werden mit `1 / (1 - p)`
skaliert, damit der Erwartungswert gleich bleibt.

Ein Beispiel mit fünf Aktivierungen und `p = 0.4`:

```text
Eingabe:           1.0   2.0   3.0   4.0   5.0
Zufällige Maske:    1     0     1     1     0
Nach Maske:        1.0   0     3.0   4.0   0
Skalierung 1/0.6:  1.67  0     5.0   6.67  0
```

Jeder Wert bleibt mit Wahrscheinlichkeit `0.6` erhalten und wird dann mit
`1 / 0.6` multipliziert. Im Mittel kommt wieder der Originalwert heraus.
**Bei jedem Schritt entsteht eine neue Maske.** Das Modell kann sich
deshalb nicht darauf verlassen, dass eine bestimmte Aktivierung immer
da ist.

Unsere Demo verwendet `p = 0.2`: Im Mittel bleiben 80 Prozent erhalten,
die übrigen werden mit `1 / 0.8 = 1.25` skaliert.

### Wo sitzt Dropout im Transformer-Block?

Wir setzen Dropout auf die **Ausgabe** der beiden Teilschichten, vor dem
Addieren auf die Residual-Verbindung:

```rust
{{#include ../../../crates/model/src/transformer_block.rs:forward}}
```

Die Residual-Verbindungen selbst (`input + attention` und
`after_attention + feed_forward`) bleiben unberührt. Der ursprüngliche
Signalweg wird also nie ausgeblendet; nur das, was die Teilschichten
beitragen, kann ausfallen.

Die Bausteine liegen im Block:

```rust
{{#include ../../../crates/model/src/transformer_block.rs:config}}
```

`attention_dropout` und `residual_dropout` sind zwei getrennte
`Dropout`-Bausteine mit derselben Wahrscheinlichkeit. `Dropout` hat
keine lernbaren Parameter; es speichert nur `prob`.
`GptModelConfig::with_dropout(p)` reicht `p` an jeden Block weiter. Mit
dem Standardwert `0.0` verhält sich das Modell wie in allen früheren
Kapiteln.

## 4. Trainings- und Auswertungsmodus

**Dropout darf nur im Training aktiv sein.**

```text
Training    -> zufällige Masken
Validierung -> keine Masken
Inferenz    -> keine Masken
```

Wäre Dropout bei der Validierung aktiv, würde derselbe Text bei jeder
Messung einen anderen Loss liefern. Vergleiche wären dann durch Zufall
verfälscht.

Burn trennt beide Modi über das Backend: Auf einem Autodiff-Backend
(Training) wendet `Dropout` die Maske an. `model.valid()` liefert
dieselben Gewichte **ohne** Autodiff; dort reicht Dropout die Werte
unverändert durch. Unsere Messfunktion nutzt genau das:

```rust
{{#include ../../../crates/training/examples/ch17.rs:measure}}
```

`measure` wertet **beide** Datensätze mit festen Gewichten und
ausgeschaltetem Dropout aus. Der Trainings-Loss in der Ausgabe ist
deshalb nicht der verrauschte Loss während des Trainings, sondern der
saubere Loss des aktuellen Modells. Beide Zahlen sind direkt
vergleichbar.

## 5. Vier Varianten vergleichen

### Dropout setzen, ohne die Gewichte zu verändern

Alle Varianten müssen mit denselben Anfangsgewichten starten, sonst
vergleichen wir nur unterschiedliche Zufallsstarts. Wir erzeugen deshalb
pro Seed **ein** Modell und klonen es für jede Variante. Danach wird nur
die Dropout-Wahrscheinlichkeit gesetzt:

```rust
{{#include ../../../crates/training/examples/ch17.rs:model}}
```

### Der Trainingslauf

```rust
{{#include ../../../crates/training/examples/ch17.rs:run}}
```

- `regularization.model(initial)` und `.optimizer()` wählen Dropout und
  Weight Decay der Variante.
- `device.seed(seed)` legt den Zufallszustand fest; auch die
  Dropout-Masken sind damit pro Seed reproduzierbar.
- `train.epoch(BATCH_SIZE, seed + epoch)` mischt die Trainingsfenster
  mit festem Seed neu. Alle Varianten sehen dieselbe Reihenfolge.
- Alle `REPORT_EVERY = 10` Epochen misst `measure` Trainings- und
  Validierungs-Loss.

### Echte Messausgabe

Das Modell ist klein (`EMBEDDING_DIM = 8`, ein Block) und trainiert auf
nur **8 Trainingsfenstern** à 8 Token, bei Lernrate `0.01` über 60 Epochen
(240 Updates). Das ist absichtlich zu viel für so wenige Daten: So sehen
wir Overfitting innerhalb weniger Sekunden. Validiert wird auf zwei
Fenstern. Die Ausgabe mittelt über die Seeds 42, 43 und 44. Der
Ausschnitt zeigt nicht jede Zeile; die Demo gibt alle 40 Updates eine
Zeile je Variante aus.

```text
Keine Regularisierung
Updates    0: Training 11.0988, Validierung 11.0563, Lücke -0.0425
Updates   40: Training 3.2753, Validierung 10.0584, Lücke +6.7830
Updates   80: Training 0.8176, Validierung 15.1577, Lücke +14.3401
Updates  120: Training 0.1227, Validierung 21.5645, Lücke +21.4419
Updates  240: Training 0.0065, Validierung 28.6210, Lücke +28.6145

AdamW mit Weight Decay
Updates   40: Training 3.3628, Validierung 9.9755, Lücke +6.6128
Updates   80: Training 0.9486, Validierung 14.1900, Lücke +13.2415
Updates  240: Training 0.0198, Validierung 20.8016, Lücke +20.7818

Dropout 0.2
Updates   40: Training 3.6185, Validierung 9.8426, Lücke +6.2241
Updates   80: Training 1.2192, Validierung 11.5098, Lücke +10.2906
Updates  240: Training 0.0623, Validierung 15.7527, Lücke +15.6905

AdamW mit Weight Decay und Dropout 0.2
Updates   40: Training 3.7067, Validierung 9.7911, Lücke +6.0845
Updates   80: Training 1.3614, Validierung 11.1943, Lücke +9.8329
Updates  240: Training 0.0921, Validierung 14.5975, Lücke +14.5055
```

### Was die Zahlen zeigen

Alle vier Varianten **memorieren** die acht Trainingsfenster: Der
Trainings-Loss sinkt gegen null. Der Validierungs-Loss fällt anfangs
leicht (bis Update 40 von etwa 11.1 auf 9.8 bis 10.1) und steigt
danach wieder an. Das ist Overfitting.

Auffällig ist der absolute Wert: Ein Validierungs-Loss von 28 liegt weit
über dem Ausgangswert von etwa 11.1, der einer fast gleichverteilten
Vorhersage entspricht. Das Modell ist nach dem Auswendiglernen bei
falschen Tokens **sehr sicher**, und genau das bestraft der Loss stark.

| Variante | Training nach 240 Updates | Validierung nach 240 Updates |
|---|---|---|
| Keine Regularisierung | 0.007 | 28.6 |
| AdamW mit Weight Decay | 0.020 | 20.8 |
| Dropout 0.2 | 0.062 | 15.8 |
| Weight Decay und Dropout | 0.092 | 14.6 |

Mit Regularisierung lernt das Modell die Trainingsdaten langsamer
auswendig, und der Validierungs-Loss steigt weniger. **Beseitigt** wird
Overfitting nicht: Auch die Kombination liegt bei 14.6 weit über dem Startwert.

Die Kombination liegt hier leicht unter Dropout allein (14.6 gegen
15.8). Die Wirkungen sind unabhängig und können sich ergänzen, sie
müssen es aber nicht: Mit drei Seeds und nicht abgestimmten Werten ist
der Unterschied kein Beleg.

### Grenzen dieses Versuchs

- Drei Seeds, ein winziges Modell, nur zwei Validierungsfenster.
- Weight Decay `0.1` und Dropout `0.2` sind nicht abgestimmt. Andere
  Werte können die Reihenfolge verändern.
- Bei einem erneuten Lauf oder anderem Text können die Zahlen leicht
  abweichen; Tendenz und Grössenordnung sind das Ergebnis, nicht die
  dritte Nachkommastelle.
- Ein früheres Abbrechen bei Update 40 hätte hier den besten
  Validierungs-Loss geliefert. Das ist der Gedanke hinter dem besten
  Stand aus Kapitel 13.

Für belastbare Aussagen brauchen wir mehr Daten, längere Läufe, mehr
Seeds und abgestimmte Hyperparameter.

## 6. Resume: was Dropout für Kapitel 15 bedeutet

Kapitel 15 setzt Training aus einem Checkpoint fort. Zwei Dinge ändern
sich durch dieses Kapitel:

**AdamW** speichert denselben Momentum- und Skalierungszustand wie Adam
und keine separate Decay-Historie. Der Wert `weight_decay` gehört zur
Optimizer-Konfiguration und muss beim Resume **identisch** bleiben,
sonst läuft die Fortsetzung mit einer anderen Regel.

**Dropout** hängt von Zufallsmasken ab. Seed und Batchposition allein
reichen für ein exakt reproduzierbares Resume **nicht**: Auch der
Zustand des Dropout-Zufallszahlengenerators muss wiederhergestellt
werden. Alternativ muss eine deterministische Maskenfolge an derselben
Stelle fortgesetzt werden, etwa durch eine festgelegte Seed-Ableitung
pro Update. Nur den Startseed erneut zu setzen würde die Maskenfolge von
vorne beginnen lassen.

Kapitel 15 speichert derzeit einen Seed und die Trainingsposition, aber
kein Dropout; die Demo in diesem Kapitel implementiert kein Resume. Das
holen wir in [Kapitel 18](18-vollstaendiger-trainingslauf.md) beim
vollständigen Trainingslauf nach: Dort leiten wir die Dropout-Masken
pro Update aus Seed und Update-Nummer ab, statt einen Generatorzustand zu
speichern.

## 7. Was prüfen wir?

```bash
cargo test -p model
cargo test -p training --example ch17
```

Die Tests prüfen, dass Dropout im Trainingsmodus die Ausgabe verändert,
nach `valid()` aber nicht, und dass `GptModelConfig::with_dropout` die
Wahrscheinlichkeit an jeden Block weiterreicht. Ein weiterer Test in
`ch17.rs` stellt sicher, dass alle Varianten dieselben Startvorhersagen
liefern und das Training das gemeinsame Ausgangsmodell nicht verändert.
Der lange Lauf auf `data/the-verdict.txt` gehört nicht zu den Tests.

## Das nehmen wir mit

Overfitting zeigt sich als wachsende Lücke zwischen Trainings- und
Validierungs-Loss. Regularisierung begrenzt, wie stark sich das Modell
an die Trainingsdaten anpasst.

Weight Decay und Dropout sind unabhängige Verfahren und lassen sich kombinieren. Weight Decay verkleinert bei jedem Update die Gewichte direkt (AdamW).
Dropout blendet im Training zufällig Aktivierungen aus und skaliert den
Rest; bei Validierung und Inferenz ist es abgeschaltet. Beides bremst
Overfitting, garantiert aber weder besseren Validierungs-Loss noch
ersetzt es mehr Daten.

Für exakt reproduzierbares Resume mit Dropout gehört neben Gewichten,
Optimizer-Zustand und Datenposition auch der Zustand der Zufallsmasken
in den Checkpoint.
