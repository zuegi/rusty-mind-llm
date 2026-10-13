# Warum Attention durch die Quadratwurzel teilt

Dieser Exkurs ist optional. Für das Attention-Kapitel genügt die Regel:
Scores vor Softmax durch `sqrt(head_dim)` teilen. Hier erklären wir, warum
ausgerechnet die Quadratwurzel der Query-/Key-Breite verwendet wird.

## Streuung statt bloss Anzahl der Summanden

Ein Skalarprodukt addiert Produkte von Query- und Key-Komponenten:

```text
Score = q[0] * k[0] + q[1] * k[1] + ...
```

Bei grösserer Vektorbreite kommen mehr Produkte zusammen. Diese können positiv
oder negativ sein und sich teilweise aufheben. Daher wächst der typische
Betrag einer solchen Summe nicht einfach proportional zur Anzahl der Produkte.

Um die Streuung zu beschreiben, verwenden wir zwei Begriffe:

- **Varianz:** durchschnittlicher quadrierter Abstand zum Mittelwert.
- **Standardabweichung:** Quadratwurzel der Varianz; ein Mass für die typische
  Grössenordnung der Abweichung, in derselben Einheit wie die betrachteten Werte.

Die Varianz quadriert die Abstände, damit positive und negative Abweichungen
sich nicht gegenseitig aufheben.

## Eine vereinfachte Annahme

Für die Herleitung nehmen wir an, dass alle Query- und Key-Komponenten
unabhängig sind, Mittelwert `0` und Varianz `1` haben. Das ist eine
Modellannahme, keine Garantie für gelernte Attention-Vektoren.

Unter dieser Annahme hat auch jedes Produkt `q[i] * k[i]` Mittelwert `0` und
Varianz `1`. Bei einer Summe unabhängiger Produkte addieren sich die Varianzen:

```text
Query-/Key-Breite d = head_dim
Varianz des Scores = d
Standardabweichung = sqrt(d)
```

Beispielsweise ergibt Breite `64` eine Standardabweichung von `sqrt(64) = 8`.
Bei Breite `4` ist sie `sqrt(4) = 2`.

## Den Breiteneffekt ausgleichen

Teilen wir einen Wert durch eine positive Zahl, teilen wir seine
Standardabweichung durch dieselbe Zahl. Deshalb:

```text
Skalierter Score = Score / sqrt(d)
Neue Standardabweichung = sqrt(d) / sqrt(d) = 1
Neue Varianz = 1
```

Unter unseren Annahmen bleibt damit die Streuung der skalierten Scores gleich,
auch wenn die Vektoren breiter werden. Eine Division durch `d` würde stärker
verkleinern und die Standardabweichung auf `1 / sqrt(d)` reduzieren.

Softmax kann bei sehr grossen Score-Unterschieden fast das gesamte Gewicht
einer einzigen Quelle geben. Seine Reaktion auf kleine Änderungen wird dann
schwach. Die Skalierung wirkt diesem Effekt entgegen, soweit er durch die
Vektorbreite entsteht.

Sie verhindert nicht jede starke Konzentration: Das Training kann weiterhin
Projektionen mit grossen oder stark unterschiedlichen Scores erzeugen.

Zurück zum [Attention-Kapitel](../kapitel/05-attention.md).
