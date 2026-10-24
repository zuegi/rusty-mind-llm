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

Mit **Query-/Key-Breite** meinen wir die Anzahl der Zahlen in **einem**
Query- beziehungsweise Key-Vektor. Beide müssen gleich viele Zahlen enthalten,
damit wir die jeweils gleich positionierten Komponenten multiplizieren können.
Diese Anzahl nennen wir in der Herleitung `d`; im Code heisst sie `head_dim`.

```text
Query eines Tokens: [1, 0]   -> zwei Komponenten
Key einer Quelle:   [0, 1]   -> zwei Komponenten

d = head_dim = 2
Score = 1 * 0 + 0 * 1        -> zwei Produkte werden addiert
```

`d` zählt also **nicht die Tokens oder die Attention-Köpfe**, sondern die
Komponenten pro Query-/Key-Vektor innerhalb eines Kopfes.
Bei zwei Tokens gibt es zwei Queries und zwei Keys; die Breite jedes
einzelnen Vektors bleibt in unserem Beispiel `2`.
Die Vektoren sind Ausgaben der
[Projektionen aus 5.4](../kapitel/05-04-projektionen.md).
Ihre Breite muss nicht der Breite des ursprünglichen Token-Vektors entsprechen.

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

Zurück zu [5.6 Scores skalieren](../kapitel/05-06-scores-skalieren.md).
