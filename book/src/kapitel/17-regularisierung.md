# Kapitel 17: Regularisierung gegen Overfitting

Kapitel 16 entscheidet, welches nächste Token erzeugt wird. Kapitel 17
entscheidet, welche Parameteränderungen das Training zulässt.

## 1. Regularisierung begrenzen

Overfitting bedeutet: Der Trainings-Loss sinkt weiter, aber der
Validierungs-Loss steigt oder verbessert sich nicht mehr. Regularisierung
ändert die Optimierung oder die Aktivierungen, damit das Modell weniger
stark einzelne Trainingsbeispiele memoriert.

Dieses Kapitel betrachtet zwei Varianten:

- **Weight Decay:** Gewichte werden bei jedem Adam-Update zusätzlich
  verkleinert.
- **Dropout:** Zufällig ausgewählte Aktivierungen werden im Training
  ausgeblendet.

## 2. AdamW und Weight Decay

Bei AdamW wird die Gewichtsanpassung aus Adam mit einer direkten
Verkleinerung der Parameter gekoppelt:

```text
Parameter_neu = Parameter_alt - Lernrate * Adam-Richtung
                 - Lernrate * Weight_Decay * Parameter_alt
```

Weight Decay ist keine zusätzliche Loss-Komponente. Die Regel reduziert
Parameter vor jedem Update direkt. Dadurch bleibt AdamW von einer
klassischen L2-Regularisierung abgesetzt, deren Gradienten in einer
anderen Berechnungsreihenfolge in den Adam-Zustand eingehen können.

Die Demo verwendet Weight Decay 0.1 und eine konstante Lernrate von
0.001 für alle Varianten, keinen Warmup-/Decay-Lernratenplan. In der
Praxis müssen Lernrate, Decay und Trainingsdauer gemeinsam verglichen werden.

## 3. Dropout

Dropout setzt bei jedem Trainingsschritt einen Anteil der Aktivierungen
auf null und skaliert die übrigen Werte:

```text
beibehaltener Anteil = 1 - p
erwartete Ausgabe = originaler Wert
```

Die Parameter `p = 0.2` bezeichnet die Ausfallwahrscheinlichkeit. Mit
`p = 0.2` bleiben durchschnittlich 80 Prozent einer Aktivierung erhalten;
die restlichen Werte werden durch `1 / 0.8` skaliert.

Die erste Variante setzt Dropout nach Attention-Ausgabe und
Feed-Forward-Ausgabe. Residual-Verbindungen bleiben erhalten.

## 4. Trainings- und Auswertungsmodus

Dropout darf nur im Training aktiv sein:

```text
Training  -> zufällige Masken
Validierung -> keine Masken
Inferenz   -> keine Masken
```

Burn stellt dafür einen `Dropout`-Baustein und `Module::valid()` bereit.
Validierung erhält feste Gewichte und feste Aktivierungen. Dadurch sind
Validierungsvergleiche nicht durch zusätzliche Zufallsmasken verfälscht.

## 5. Demo

```bash
cargo run --release -p training --example ch17
```

Die Demo vergleicht:

1. AdamW ohne Weight Decay
2. AdamW mit Weight Decay
3. Dropout 0.2

Alle Varianten starten mit Kopien derselben Anfangsgewichte (Seed 42)
und verwenden dieselben Trainingsbatches und Validierungsdaten.
Nur Dropout-Wahrscheinlichkeit und Weight Decay unterscheiden sich.
Die Ausgabe zeigt Trainings- und Validierungs-Loss vor und nach den
Updates. Beide werden mit deaktiviertem Dropout gemessen; identische
Startwerte machen den gemeinsamen Ausgangspunkt sichtbar.

Mit dem Standardtext führt jede Variante nur vier Updates aus. Das
veranschaulicht den Ablauf, beweist aber weder Overfitting noch bessere
Generalisierung oder eine optimale Hyperparameterwahl. Dafür braucht es
längere Loss-Verläufe und wiederholte Versuche mit mehreren Seeds.

## 6. Resume

AdamW speichert denselben Momentum-Zustand wie Adam und zusätzlich keine
separate Decay-Historie. Der Weight-Decay-Wert gehört zur
Optimizer-Konfiguration und muss beim Resume identisch erhalten bleiben.

Dropout hängt von Zufallsmasken ab. Seed und Batchposition allein reichen
für ein exakt reproduzierbares Resume nicht: Auch der Zustand des
Dropout-Zufallszahlengenerators muss wiederhergestellt werden. Alternativ
muss eine deterministische Maskenfolge an derselben Stelle fortgesetzt
werden, etwa durch eine festgelegte Seed-Ableitung pro Update. Nur den
Startseed erneut zu setzen würde die Maskenfolge von vorne beginnen lassen.

Kapitel 15 speichert derzeit einen Seed und die Trainingsposition,
verwendet aber kein Dropout. Die Demo hier implementiert kein Resume.
Kapitel 17 grenzt diese Erweiterung ausdrücklich ein: Erst
Regularisierung verstehen, dann Resume-Zustand erweitern.
