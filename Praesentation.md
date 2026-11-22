# Vom Text zum kleinen Sprachmodell

## Rahmen

20 Minuten: 18 Minuten Vortrag und 2 Minuten Fragen.
Zielgruppe: Kotlin-/Angular-Entwickler und Application Managers ohne LLM-Vorkenntnisse.
Grundlage: Kapitel 1 bis 18 von `rusty-mind-llm`; kein Kotlin-Code und kein Sprachvergleich.

**Lernziel:** Die Teilnehmenden verstehen, wie ein kleines GPT-artiges Modell
das nächste Token vorhersagt, wie Training diese Vorhersage verändert und
warum bessere Messwerte noch keine verlässlichen Antworten beweisen.

## 1. Wie geht dieser Text weiter? (0:00–1:00)

Kurzer persönlicher Einstieg: Das Projekt begann in Kotlin. Für den eigenen
Ansatz fehlten passende Libraries, und der Implementierungsaufwand wurde zu
groß; deshalb folgte der Wechsel zu Rust mit Burn und einem strukturierten
Lernpfad. Das ist persönliche Projekterfahrung, keine allgemeine Bewertung
von Kotlin.

Danach die Frage an das Publikum: Wie geht „Der Hund …“ weiter?
Mehrere plausible Antworten zeigen die Grundaufgabe: Das Modell bewertet
mögliche Fortsetzungen, statt einen fertigen Satz aus einer Tabelle zu holen.

## 2. Text wird Zahlen (1:00–3:00)

Ein Modell rechnet mit Zahlen. Der Tokenizer zerlegt Text in Tokens und ordnet
ihnen IDs zu; Tokens können Wörter, Wortteile oder Satzzeichen sein.
Lernbare Vektoren und Positionsinformationen machen diese IDs für das Modell
verarbeitbar.

**Visual:** Textkarten → Token-IDs → Vektoren mit Positionen.
Grundlage: Kapitel 1 bis 3.

## 3. Warum der bisherige Kontext zählt (3:00–5:00)

Ein Token wird nicht isoliert vorhergesagt: Die bisherige Textfolge beeinflusst
seine Bewertung. Attention verbindet Informationen aus diesem Kontext;
die kausale Maske verhindert den Blick auf zukünftige Tokens.
Transformer-Blöcke verarbeiten diese Informationen weiter.

**Visual:** Pfeile zwischen bisherigen Tokenkarten, Zukunft ausgeblendet.
Keine Q/K/V-Formeln oder Tensorformen im Hauptteil.
Grundlage: Kapitel 5 bis 7.

## 4. Ein nächstes Token entsteht (5:00–7:00)

Das Modell liefert für jedes mögliche nächste Token eine Bewertung, einen
Logit. Logits sind noch keine Wahrscheinlichkeiten; Softmax kann sie in eine
Wahrscheinlichkeitsverteilung umwandeln. Bei Greedy-Auswahl gewinnt der höchste
Logit; das gewählte Token wird angehängt und die Vorhersage erneut berechnet.

**Illustratives Beispiel, keine gemessene Modellvorhersage:**

```text
Kontext:  Der Hund

ID:       0      1       2       3      4
Token:    Der    Hund    Katze   rennt  .
Logit:    1.2   -0.4     0.8     2.1    0.3

argmax → ID 3 → „rennt“
Neuer Kontext: Der Hund rennt
```

Die Beispiel-Logits stammen aus Kapitel 7, die künstliche Worttabelle aus
Kapitel 11. Ihre Zuordnung zum Kontext „Der Hund“ dient nur der Erklärung.
Diese fünf IDs sind nicht die entsprechenden IDs des echten GPT-2-Tokenizers.

**Visual:** Kandidaten als Balken; Auswahl und erneute Berechnung animieren.
Grundlage: Kapitel 7, 11 und 16.

## 5. Wie das Modell lernt (7:00–10:00)

Der Trainingstext liefert das Ziel: das tatsächlich folgende Token.
Der Loss bewertet, wie viel Wahrscheinlichkeit das Modell diesem Ziel gibt;
Gradienten und Optimizer verändern anschließend die Gewichte.
Viele solcher Schritte sollen die Vorhersagen verbessern.

„Richtig“ heißt hier: Dieses Token steht im Trainingstext, nicht:
Alle anderen Fortsetzungen sind falsch. Beim Generieren bleiben die Gewichte
fest; Tokenauswahl ist kein erneutes Training.

**Visual:** Vorhersage → Vergleich mit Ziel → Loss → Gewichtsupdate.
Grundlage: Kapitel 2, 4, 8 und 9.

## 6. Lernen oder Auswendiglernen? (10:00–13:00)

Ein fallender Trainings-Loss allein reicht nicht. Getrennte Validierungsdaten
zeigen, wie gut das Modell Tokens außerhalb seiner Trainingsfenster vorhersagt.
Die dokumentierten Kurven aus Kapitel 13 zeigen, warum der beste
Validierungsstand gespeichert wird und nicht einfach der letzte Stand.

Im dortigen Lauf ist Epoche 16 mit Validierungs-Loss 6.6642 besser als
Epoche 20 mit 6.7019, obwohl der Trainings-Loss weiter sinkt.
Das ist ein Overfitting-Warnsignal, kein Beweis für allgemeines Sprachverständnis
oder dessen Fehlen.

**Visual:** Dokumentierte Trainings- und Validierungs-Loss-Kurven;
Epoche 16 markieren. Die Werte sind Messungen aus dem Buch, keine
illustrativen Zahlen und kein neu ausgeführter Präsentationslauf.
Grundlage: Kapitel 9 und 13.

## 7. Was unser Modell tatsächlich erzeugt (13:00–15:00)

Eine dokumentierte Fortsetzung mit trainiertem Checkpoint aus Kapitel 16
zeigt Wiederholungen und unverständlichen Text. Vorgegebener Prompt und neu
erzeugte Fortsetzung werden sichtbar getrennt.
Die Generierung funktioniert technisch, liefert aber noch keine gute Sprache.

Greedy und Sampling verändern die Tokenauswahl, nicht die Modellgewichte.
Andere Auswahlverfahren garantieren weder bessere Texte noch korrekte Fakten.
Die zufällig initialisierte Demo aus Kapitel 11 wird nicht als gelernte
Sprachleistung präsentiert.

**Visual/Demo:** Kurzer vorbereiteter Mitschnitt einer vorab geprüften
Generierung oder die dokumentierte Ausgabe; kein Live-Training.
Illustrationen und gemessene Ergebnisse klar kennzeichnen.
Grundlage: Kapitel 11, 12 und 16.

## 8. Der vollständige Lern- und Trainingsablauf (15:00–17:00)

Die Bausteine ergeben einen Ablauf: Daten vorbereiten, trainieren, validieren,
Gewichte speichern, Training fortsetzen und Text generieren.
Optimizer und Regularisierung beeinflussen das Lernen; letzter Trainingsstand
und bester Validierungsstand erfüllen unterschiedliche Zwecke.

Kapitel 18 verbindet diese Mechanik, verwendet aber nur acht Trainings-
und zwei Validierungsfenster. Ein vollständiger Ablauf ist damit gezeigt,
kein belastbarer Generalisierungsnachweis.

**Visual:** Ablaufgrafik mit zwei Speicherständen:
„Weitertrainieren“ und „bestes Validierungsmodell“.
Grundlage: Kapitel 10, 14, 15, 17 und 18.

## 9. Was die Experimente zeigen – und was nicht (17:00–18:00)

Das Projekt macht nachvollziehbar, wie Tokenvorhersage, Gewichtsupdates,
Validierung und Generierung zusammenarbeiten.
Bessere Messwerte auf einem kleinen Datenausschnitt garantieren keine
Faktentreue, allgemeines Sprachverständnis oder Produktionstauglichkeit.
Für eine unabhängige Schlussbewertung wäre zusätzlich ein unberührter
Testdatensatz nötig.

**Kernaussage:** Ein Sprachmodell lernt Textfortsetzung.
Ob seine Antworten für einen konkreten Einsatz verlässlich sind,
ist eine zusätzliche Bewertungsaufgabe.

## Fragen (18:00–20:00)

Zwei Minuten für Rückfragen. Als Verständnisfrage eignet sich:
„Was verändert sich beim Training, und was verändert sich beim Generieren?“
Damit wird das Lernziel aufgegriffen, ohne weitere technische Details
einzuführen.
