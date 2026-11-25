# Vom Text zum kleinen Sprachmodell

## Rahmen

20 Minuten: 18 Minuten Vortrag und 2 Minuten Fragen.
Zielgruppe: Kotlin-/Angular-Entwickler und Application Managers ohne LLM-Vorkenntnisse.
Grundlage: Kapitel 1 bis 18 von `rusty-mind-llm`; kein Kotlin-Code und kein Sprachvergleich.

**Lernziel:** Die Teilnehmenden verstehen, wie ein kleines GPT-artiges Modell
das nächste Token vorhersagt, wie Training diese Vorhersage verändert und
warum bessere Messwerte noch keine verlässlichen Antworten beweisen.

**Roter Faden:** „Der Hund …“ begleitet Einstieg, Tokenisierung,
Kontextverarbeitung, Tokenauswahl und Training. Für dieses didaktische
Beispiel verwenden wir eine künstliche Worttabelle. Echte Tokenisierung,
gemessene Loss-Werte und dokumentierte Modellfortsetzungen werden davon
sichtbar getrennt.

**Wachsende Architektur-Grafik:** Jede Folie zeigt eine positionsstabile
Übersichtsgrafik, die sich schrittweise zur Architektur von `rusty-mind-llm`
erweitert. Bisher eingeführte Komponenten bleiben sichtbar, die aktuelle
Komponente wird farbig hervorgehoben. Modellkern (`GptEmbedding`,
`TransformerBlock` 1..N, `output_projection`) und Infrastruktur (Tokenizer,
Generierungsschleife, Training, Validierung, Checkpoints) bleiben visuell
klar getrennt. Auf Folien 3 und 8 wird die Grafik größer gezeigt.

## 1. Wie geht dieser Text weiter? (0:00–1:00)

Kurzer persönlicher Einstieg: Das Projekt begann in Kotlin. Für den eigenen
Ansatz fehlten passende Libraries, und der Implementierungsaufwand wurde zu
groß; deshalb folgte der Wechsel zu Rust mit Burn und einem strukturierten
Lernpfad. Das ist persönliche Projekterfahrung, keine allgemeine Bewertung
von Kotlin.
Hinweis auf das offene Lernprojekt und begleitende deutsche Buch
(`https://github.com/zuegi/rusty-mind-llm`, Buch:
`https://zuegi.github.io/rusty-mind-llm/`, aktueller Stand: Kapitel 1–18).

Danach die Frage an das Publikum: Wie geht „Der Hund …“ weiter?
Mehrere plausible Antworten zeigen die Grundaufgabe: Das Modell bewertet
mögliche Fortsetzungen, statt einen fertigen Satz aus einer Tabelle zu holen.

**Architektur-Grafik:** Einstieg als Blackbox: Text rein → Fortsetzung raus.

## 2. Text wird Zahlen (1:00–3:00)

Ein Modell rechnet mit Zahlen. Drei Schritte werden nacheinander sichtbar:
Der Tokenizer zerlegt Text in Tokens, ordnet ihnen IDs zu, und das Modell
schlägt für jede ID einen lernbaren Vektor in seiner Embedding-Tabelle nach.
Tokens können Wörter, Wortteile oder Satzzeichen sein; IDs sind Kennungen,
keine Bedeutungen oder Wahrscheinlichkeiten.

Für „Der Hund“ zeigen wir vereinfachte Wortkarten und ihre künstlichen IDs.
Positionsinformationen ergänzen die Vektoren, damit Reihenfolge berücksichtigt
werden kann. Im Projekt sind auch diese Positionsvektoren lernbar.
Embeddings sind kein fest vergebener „Bedeutungscode“: Welche Beziehungen
die Vektoren abbilden, hängt vom Training ab.

**Visual:** Drei aufeinanderfolgende Einblendungen:
Textkarten → Token-IDs → lernbare Vektoren; anschließend Positionen ergänzen.
**Architektur-Grafik:** Erweitert um Tokenizer (außerhalb des Modells) →
Token-IDs → `GptEmbedding` (Token-Embedding + Positions-Embedding).
Grundlage: Kapitel 1 bis 3.

## 3. Warum der bisherige Kontext zählt (3:00–5:00)

Ein Token wird nicht isoliert vorhergesagt: Die bisherige Textfolge beeinflusst
seine Bewertung. Attention gewichtet und verbindet Informationen aus diesem
Kontext; die kausale Maske verhindert den Blick auf zukünftige Tokens.
Mehrere Transformer-Blöcke verändern die Repräsentationen schrittweise.
Wir behaupten keine feste Aufgabenverteilung wie „erste Schicht: Grammatik,
letzte Schicht: Verständnis“.

„Der Hund …“ bleibt das Beispiel. Welche Kontextbeziehungen wichtig sind,
hängt von Eingabe, Attention-Kopf und Schicht ab. Kurze Wörter oder Artikel
sind nicht grundsätzlich weniger wichtig. Pfeile illustrieren nur den
Informationsfluss, keine gemessenen Attention-Gewichte.

**Visual:** Bisherige Tokenkarten → wiederholte Transformer-Blöcke →
Bewertungen nächster Tokens; Zukunft ausgeblendet.
**Architektur-Grafik (größer):** Erweitert um `TransformerBlock` 1..N als
gestapelte Blöcke mit Detailfenster für einen Block (Pre-LayerNorm, kausale
Multi-Head Attention, Feed-Forward-Netz, Residual-Verbindungen).
Keine Q/K/V-Formeln oder Tensorformen im Hauptteil.
Grundlage: Kapitel 5 bis 7.

## 4. Ein nächstes Token entsteht (5:00–7:00)

Das Modell liefert für jedes mögliche nächste Token eine Bewertung, einen
Logit. Logits sind noch keine Wahrscheinlichkeiten; Softmax kann sie in eine
Wahrscheinlichkeitsverteilung umwandeln. Zuerst zeigen wir die Bewertungen,
dann die daraus berechneten Wahrscheinlichkeiten, schließlich die Auswahl.
Bei Greedy-Auswahl gewinnt der höchste Logit; dafür ist Softmax nicht nötig,
weil die Rangfolge gleich bleibt.

**Illustratives Beispiel, keine gemessene Modellvorhersage:**

```text
Kontext:             Der Hund

ID:                  0       1       2       3       4
Token:               Der     Hund    Katze   rennt   .
Logit:               1.2    -0.4      0.8     2.1     0.3
Softmax, gerundet:   21.10%   4.26%  14.15%  51.91%   8.58%

argmax → ID 3 → „rennt“
Neuer Kontext: Der Hund rennt
```

Die Beispiel-Logits stammen aus Kapitel 7, die künstliche Worttabelle aus
Kapitel 11. Ihre Zuordnung zum Kontext „Der Hund“ dient nur der Erklärung.
Diese fünf IDs sind nicht die entsprechenden IDs des echten GPT-2-Tokenizers.
Die Prozentwerte sind aus den illustrativen Logits bei Temperatur 1 berechnet,
keine gemessenen Vorhersagen und keine Wahrscheinlichkeit für Faktentreue.

Beim nächsten Klick wird „rennt“ angehängt: Der Kontext lautet nun
„Der Hund rennt“. Das Modell berechnet daraus erneut Bewertungen für das
nächste Token. Die vorherigen Kandidaten werden nicht der Reihe nach
abgearbeitet. Ohne weitere belegte Zahlen zeigen wir für diesen zweiten
Schritt nur den Ablauf, keine erfundene Ergebnisverteilung.
Dieses Anwenden des Modells mit festen Gewichten heißt **Inferenz**.

**Visual:** Logits → Wahrscheinlichkeitsbalken → Auswahl → erweiterter
Kontext → erneute Vorhersage. Eine spätere Button-Demo könnte diesen Ablauf
illustrieren; sie wäre keine Ausführung des trainierten Rust-Modells.
**Architektur-Grafik:** Erweitert um `output_projection` → Logits als
Abschluss des Modellkerns; außerhalb des Modells: Auswahl → Token anhängen →
Rückführung zum Eingang.

**Sprechernotiz:** Die autoregressive Schleife beschreibt das Prinzip.
Praktische Implementierungen können frühere Berechnungen zwischenspeichern;
nicht jedes System berechnet bei jedem Token alles von Grund auf neu.
Keine pauschale Anzahl Tokens pro Sekunde nennen.
Grundlage: Kapitel 7, 11 und 16.

## 5. Wie das Modell lernt (7:00–10:00)

Der Trainingstext liefert das Ziel: das tatsächlich folgende Token.
Das durchgehende Beispiel wird jetzt zum Trainingsfall:
Im Beispieltext „Der Hund rennt“ ist „Der Hund“ der Kontext und „rennt“ das Ziel.
Der Loss bewertet, wie viel Wahrscheinlichkeit das Modell diesem Ziel gibt;
Gradienten und Optimizer verändern anschließend die Gewichte.
Viele solcher Schritte sollen die Vorhersagen verbessern.

„Richtig“ heißt hier: Dieses Token steht im Trainingstext, nicht:
Alle anderen Fortsetzungen sind falsch. Beim Generieren bleiben die Gewichte
fest; Tokenauswahl ist kein erneutes Training. Die hohen Bewertungen unseres
illustrativen Beispiels sind keine Ausgangsmessung vor einem Trainingsupdate.

**Visual:** Derselbe Kontext und das Zieltoken → Vorhersage → Vergleich →
Loss → Gewichtsupdate. Gegenüberstellen:
Training ändert Gewichte; Generierung verlängert die Tokenfolge.
**Architektur-Grafik:** Separater Trainingszweig außerhalb des Modellkerns:
Zieltoken + Logits → Cross-Entropy-Loss → Gradienten → Optimizer →
Gewichtsupdate im Modell.
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
**Architektur-Grafik:** Erweitert um getrennten Validierungspfad mit festen
Gewichten (kein Optimizer-Update).
Grundlage: Kapitel 9 und 13.

## 7. Was unser Modell tatsächlich erzeugt (13:00–15:00)

Eine dokumentierte Fortsetzung mit trainiertem Checkpoint aus Kapitel 16
zeigt Wiederholungen und unverständlichen Text. Vorgegebener Prompt und neu
erzeugte Fortsetzung werden sichtbar getrennt.
Die Generierung funktioniert technisch, liefert aber noch keine gute Sprache.

Greedy und Sampling gehören zur Inferenz: Sie verändern die Tokenauswahl,
nicht die Modellgewichte.
Bei Sampling verändert Temperatur die Wahrscheinlichkeitsverteilung:
Niedrigere Temperatur konzentriert sie stärker auf hoch bewertete Tokens,
höhere Temperatur macht sie flacher. Sie bestimmt nicht direkt, ob ein Text
„sicher“ oder „kreativ“ ist. Andere Auswahlverfahren garantieren weder bessere
Texte noch korrekte Fakten.
Die zufällig initialisierte Demo aus Kapitel 11 wird nicht als gelernte
Sprachleistung präsentiert.

**Visual/Demo:** Kurzer vorbereiteter Mitschnitt einer vorab geprüften
Generierung oder die dokumentierte Ausgabe; kein Live-Training.
Illustrationen und gemessene Ergebnisse klar kennzeichnen.
**Architektur-Grafik:** Bestehender Generierungsweg hervorgehoben
(Auswahlstrategie Greedy/Sampling außerhalb des Modells; keine neuen
Modellkomponenten).
Grundlage: Kapitel 11, 12 und 16.

## 8. Der vollständige Lern- und Trainingsablauf (15:00–17:00)

Die Bausteine ergeben einen Ablauf: Daten vorbereiten, trainieren, validieren,
Gewichte speichern, Training fortsetzen und Text generieren.
Optimizer und Regularisierung beeinflussen das Lernen; letzter Trainingsstand
und bester Validierungsstand erfüllen unterschiedliche Zwecke.

Kapitel 18 verbindet diese Mechanik, verwendet aber nur acht Trainings-
und zwei Validierungsfenster. Ein vollständiger Ablauf ist damit gezeigt,
kein belastbarer Generalisierungsnachweis.

**Vier Begriffe klar getrennt:**

| Begriff | Gewichte | Braucht Zieltoken? | Zweck |
| --- | --- | --- | --- |
| **Modell** | enthält sie | nein | berechnet Logits aus Token-IDs |
| **Training** | werden geändert | ja | Fehler (Loss) senken |
| **Validierung** | fest | ja | Qualität auf getrennten Daten messen |
| **Inferenz** | fest | nein | Vorhersage anwenden (z. B. Text generieren) |

**Visual:** Ablaufgrafik mit zwei Speicherständen:
„Weitertrainieren“ und „bestes Validierungsmodell“.
**Architektur-Grafik (größer):** Erweitert um Checkpoints außerhalb des
Modells: `last/` (Modell + Optimizer zum Weitertrainieren) und `best/`
(bester Validierungsstand zum Anwenden).
Grundlage: Kapitel 10, 14, 15, 17 und 18.

## 9. Was die Experimente zeigen – und was nicht (17:00–18:00)

Das Projekt macht nachvollziehbar, wie Tokenvorhersage, Gewichtsupdates,
Validierung und Generierung zusammenarbeiten.
Bessere Messwerte auf einem kleinen Datenausschnitt garantieren keine
Faktentreue, allgemeines Sprachverständnis oder Produktionstauglichkeit.
Für eine unabhängige Schlussbewertung wäre zusätzlich ein unberührter
Testdatensatz nötig.
Aus dem Next-Token-Prinzip allein folgt weder ein Beweis für menschliches
Denken noch der Beweis, dass ein Modell keine komplexen Fähigkeiten erlernen
kann. Unsere kleinen Experimente beantworten diese allgemeinen Fragen nicht.

**Kernaussage:** Ein Sprachmodell lernt Textfortsetzung.
Ob seine Antworten für einen konkreten Einsatz verlässlich sind,
ist eine zusätzliche Bewertungsaufgabe.
**Architektur-Grafik:** Vollständige Übersicht mit klarer Abgrenzung zwischen
Modellkern (`GptModel`) und umgebender Trainings- sowie Generierungsinfrastruktur.

**Kurzer Ausblick — nächste geplante Kapitel im Buch (Kapitel 19–28):**

- **Daten & Effizienz (Kap. 19–20):** Datenqualität, unabhängiger Testdatensatz, Laufzeit und Speicher.
- **Vortrainiertes GPT-2 (Kap. 21–22):** Architektur mit GPT-2 abgleichen (u. a. finale LayerNorm) und vortrainierte Gewichte importieren.
- **Lokaler Chat (Kap. 23–24):** Terminal-Chat, lokaler HTTP-Dienst und Chatoberfläche.
- **Anwendung & Betrieb (Kap. 25–28):** RAG (Wissen zur Laufzeit ohne Gewichtsänderung), Fine-Tuning/Instruction-Tuning, Grenzen von Assistenten, Evaluation, Sicherheit und Betrieb.

(Geplanter Lernpfad aus `NächsteKapitel.md`, noch nicht umgesetzt.)

## Fragen (18:00–20:00)

Zwei Minuten für Rückfragen. Als Verständnisfrage eignet sich:
„Was verändert sich beim Training, und was verändert sich beim Generieren?“
Damit wird das Lernziel aufgegriffen, ohne weitere technische Details
einzuführen.

**Projekt & Buch (während der Fragerunde sichtbar lassen):**
Repository: `https://github.com/zuegi/rusty-mind-llm`
Buch (aktuell Kapitel 1–18): `https://zuegi.github.io/rusty-mind-llm/`

## Inspirationsquelle und Abgrenzung

[„LLMs Finally Understood: You’ll See Them Differently After This“](https://www.youtube.com/watch?v=nYiP5mZC1Zw)
von The Architecture Layer dient als didaktische Inspiration für den
schrittweisen Generierungsablauf. Der Abgleich basiert auf dem bereitgestellten
Transkript, nicht auf einer eigenen Sichtung des Videos.

Die Präsentation übernimmt keine pauschalen Aussagen über Schichtaufgaben,
Attention auf „wichtige Wörter“, Kreativität durch Temperatur oder
Generierungsgeschwindigkeit. Technische Grundlage und dokumentierte Messungen
bleiben die Kapitel des Projekts; die Verbindung zu allgemeinen Fähigkeiten
von ChatGPT wird nicht als experimentell nachgewiesen dargestellt.

Diese Präzisierung betrifft die Gliederung. Die Slidev-Folien und ihre
Sprechernotizen werden separat angepasst, nicht automatisch synchronisiert.
