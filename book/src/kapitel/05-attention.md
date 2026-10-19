# 5. Attention und Multi-Head Attention

## Lernziel und Lernpfad

Das Embedding liefert einen Vektor pro Token und Position. Noch hat kein Token
Informationen von seinen Nachbarn aufgenommen. Warum braucht es diese
Informationen überhaupt?

Betrachte diese beiden Sätze:

```text
Die Bank am Fluss war nass.
Die Bank genehmigt den Kredit.
```

Das Wort "Bank" hat je nach Umgebung eine andere Bedeutung. Das Token-Embedding
allein kann diesen Unterschied nicht darstellen: Dieselbe Token-ID schlägt
immer denselben Vektor nach. Auch das Positions-Embedding sagt nur, wo ein
Token steht, nicht welche anderen Tokens im Text vorkommen.

Für die Vorhersage des nächsten Tokens reicht es deshalb nicht, jedes Token
isoliert zu betrachten. Das Modell muss Informationen aus dem bisherigen Text
verbinden. Nach "Die Bank am" sind andere Fortsetzungen plausibel als nach
"Die Bank genehmigt den". Die Wörter helfen hier nur beim Veranschaulichen:
Unser Tokenizer kann ein Wort auch in mehrere Tokens zerlegen.

"Informationen aufnehmen" bedeutet keine Änderung der Token-ID oder der
Embedding-Tabelle. Stattdessen berechnen wir für jede Position einen **neuen
Vektor**, der Informationen anderer Positionen enthält. Dieser Vektor hängt
dann von der konkreten Eingabesequenz ab.

**Attention** berechnet solche kontextabhängigen Vektoren, indem sie
Informationen anderer Positionen gewichtet und mischt. "Nachbarn" meint dabei
nicht nur direkt angrenzende Tokens, sondern auch weiter entfernte Positionen
im Eingabefenster. In unserem späteren GPT-Baustein darf eine Position nur sich
selbst und vorherige Positionen verwenden, keine zukünftigen Tokens.

## Neun kleine Lernseiten

Du musst diesen Lernpfad nicht am Stück lesen. Jede Seite beantwortet eine
Frage und endet mit einer kurzen Aufgabe. Erst selbst überlegen, dann die
aufklappbare Lösung öffnen. Wenn du die Aufgabe erklären kannst, ist das
ein guter Haltepunkt für eine Pause.

| Seite | Die Frage dieser Seite |
|---|---|
| <span id="ein-kopf-zwei-tokens"></span><span id="schritt-1-ein-kopf-zwei-tokens"></span>[5.1 Informationen mischen](05-01-informationen-mischen.md) | Wie wird aus vorhandenen Inhalten ein neuer Vektor? |
| <span id="warum-query-key-und-value"></span><span id="die-drei-schritte-gehören-zusammen"></span>[5.2 Anteile berechnen](05-02-anteile-berechnen.md) | Wie viel übernimmt eine Position von jeder Quelle? |
| <span id="die-rechnung-mit-burn"></span>[5.3 Erste Attention in Burn](05-03-erste-attention-in-burn.md) | Wo steckt die Handrechnung im Code? |
| <span id="schritt-2-drei-lernbare-projektionen"></span>[5.4 Query, Key und Value erzeugen](05-04-projektionen.md) | Woher kommen die drei Vektoren eines Tokens? |
| [5.5 Was daran lernbar ist](05-05-lernbare-parameter.md) | Welche Zahlen verändert Training? |
| [5.6 Scores skalieren](05-06-scores-skalieren.md) | Warum teilen wir vor Softmax durch einen Faktor? |
| <span id="schritt-3-kausale-maskierung-causal-masking"></span>[5.7 Zukunft ausblenden](05-07-zukunft-ausblenden.md) | Wie verhindern wir Informationen aus der Zukunft? |
| [5.8 Einen Attention-Kopf in Burn bauen](05-08-attention-kopf-in-burn.md) | Wie werden Rechnung, Parameter und Batch zum Baustein? |
| <span id="schritt-4-multi-head-attention"></span>[5.9 Mehrere Köpfe verbinden](05-09-mehrere-koepfe.md) | Wie verbinden wir mehrere unabhängige Mischungen? |

Die Seiten gehören in diese Reihenfolge. Mathematik und Code haben getrennte
Lernziele: Wenn Rusts Ownership noch unklar ist, bedeutet das nicht, dass du
die Attention-Rechnung nicht verstanden hast. Details zu Rust, Burn und
optimierten Tensorformen stehen in ausdrücklich markierten Vertiefungen.

## Ein Beispiel, schrittweise erweitert

Bis einschliesslich 5.8 verwenden wir dieselben künstlichen Eingabevektoren:
Token A `[1, 0]` und Token B `[0, 1]`. Sie stehen für keine bestimmten Wörter.
Am Anfang geben wir Anteile vor. Danach berechnen wir sie, ergänzen
Projektionen, Skalierung und zuletzt die Maske.

| Stand | Was ist neu? | Kontext für A |
|---|---|---|
| 5.1 | Vorgegebene Anteile `[0.75, 0.25]` | `[0.75, 0.25]` |
| 5.2–5.3 | Berechnete Anteile; `Q = K = V = X` | ungefähr `[0.731, 0.269]` |
| 5.4–5.5 | Eigene Q/K/V-Projektionen | Wir bereiten die Vektoren vor; die vollständige Rechnung folgt in 5.6. |
| 5.6 | Projektionen und skalierte Scores | ungefähr `[0.660, 2.009]` |
| 5.7–5.8 | Zusätzlich kausale Maske | `[2, 0]` |
| 5.9 | Zwei Köpfe mit je Breite 2 | Neues Beispiel mit Eingabebreite 4, ausdrücklich angekündigt. |

Unterschiedliche Ergebnisse sind also kein Widerspruch: Jede Erweiterung
ändert einen benannten Teil der Rechnung. Die übrigen Teile bleiben erhalten.
Es findet in diesen Demos noch kein Training mit einem Optimizer statt.

## So lernst du damit

Lies zunächst nur eine Seite und schliesse danach die Erklärung. Beschreibe
in eigenen Worten Eingabe, neuen Rechenschritt und Ergebnis. Öffne die Seite
wieder, wenn dir ein Zusammenhang fehlt. Zu Beginn der nächsten Lernsitzung
versuche die letzte Aufgabe nochmals, bevor du weiterliest.

Mermaid zeigt Wege und Abhängigkeiten. Zahlen, Matrizen und Handrechnungen
bleiben daneben als Text lesbar. Hervorgehobene Knoten markieren den
Rechenschritt, den wir gerade ergänzen.

Start: [5.1 Informationen mischen](05-01-informationen-mischen.md).
