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

Wir erarbeiten diesen Baustein in kleinen Schritten:

1. Ein Attention-Kopf mit festen Zahlen: vergleichen, gewichten, mischen.
2. Lernbare Query-, Key- und Value-Projektionen und die Skalierung.
3. Eine kausale Maske, damit kein Token in die Zukunft schaut.
4. Mehrere Köpfe und eine gemeinsame Ausgabeprojektion.
5. Anschluss an das Embedding und Gradienten mit Burns Autodiff.

Dieses Kapitel enthält bisher **Schritte 1 bis 3**. Die erste Demo zeigt die
Grundrechnung mit festen Vektoren. Die zweite verwendet einen trainierbaren
Attention-Kopf mit Q/K/V-Projektionen und Skalierung. Schritt 3 ergänzt
die kausale Maske, damit Tokens nicht in die Zukunft schauen. Noch fehlen
mehrere Köpfe. Training und Optimizer folgen später; wir implementieren kein
eigenes Autograd.

### Was du jetzt verstehen solltest

Für dieses Kapitel reichen zunächst drei Dinge: Zahlen paarweise multiplizieren
und addieren, eine Zahl als Anteil lesen und Zeilen von Spalten unterscheiden.
Neue Begriffe führen wir dort ein, wo sie gebraucht werden.

Nach dem Lesen solltest du erklären können, wie Query und Key die Gewichte
bestimmen und warum wir anschliessend Values mischen. Du solltest auch eine
Ausgabezeile unserer kleinen Beispiele von Hand nachrechnen können.

Burn übernimmt die Ausführung der Tensoroperationen und das automatische
Ableiten. Du musst weder Softmax noch Autograd selbst implementieren. Auch
Burns interne Parameterverwaltung musst du noch nicht kennen. Beim Rust-Code
unterscheiden wir deshalb zwischen **unserem Rechenweg** und **Framework-Code,
den wir vorerst verwenden**.

Das Kapitel ist kein Gedächtnistest. Nachschlagen gehört dazu, besonders wenn
Attention, Tensoren und Rust gleichzeitig neu sind. Die angeleiteten Übungen
stehen deshalb direkt nach den zugehörigen Erklärungen. Lies den Lösungsweg
mit und versuche dann, ihn in eigenen Worten nachzuvollziehen. Du musst weder
Begriffe noch Achsennummern auf Anhieb auswendig wissen.

## Ein Kopf, zwei Tokens

Wir verwenden zwei künstliche Token-Vektoren:

```text
Token A: [1, 0]
Token B: [0, 1]
```

Die Zahlen sind bewusst gewählt, nicht gelernt. Sie beschreiben keine
sprachlichen Eigenschaften. Die Eingabematrix `X` hat die Form `[2, 2]`:
zwei Tokens mit jeweils zwei Zahlen. Zur Vereinfachung verwenden wir noch
keinen Batch.

## Warum Query, Key und Value?

Attention muss zwei verschiedene Entscheidungen treffen:

1. **Von welchen Positionen brauche ich Informationen?**
2. **Welche Informationen übernehme ich von diesen Positionen?**

Für die erste Entscheidung braucht sie einen Vergleich zwischen der suchenden
Position und den möglichen Quellen. Für die zweite braucht sie den Inhalt,
den diese Quellen liefern. Daraus entstehen drei Rollen, nicht drei verschiedene
Arten von Tokens.

### Eine Analogie: Dokumente suchen

Stell dir eine Suche in einer Dokumentensammlung vor:

- **Query (Suchanfrage):** Du suchst eine Erklärung zur Tokenisierung.
- **Key (Suchmerkmal):** Jedes Dokument bietet Merkmale an, anhand derer du
  seine Eignung beurteilst, etwa Thema und Schlagwörter.
- **Value (Inhalt):** Aus einem passenden Dokument übernimmst du die eigentliche
  Erklärung, nicht nur seine Schlagwörter.

Suchanfrage und Suchmerkmale bestimmen also, **welches Dokument passt**.
Der Inhalt bestimmt, **was du daraus mitnimmst**.

Attention funktioniert ähnlich, aber ohne Textsuche oder fest eingetragene
Schlagwörter: Alle drei Rollen sind Zahlenvektoren. Statt ein Dokument hart
auszuwählen, berechnet Attention Gewichte und mischt die Inhalte mehrerer
Quellen.

### Jedes Token ist Suchender und Quelle

Bei **Self-Attention** stammen Query, Key und Value aus derselben Eingabesequenz.
Jede Token-Position bekommt alle drei Vektoren:

- Mit ihrer **Query** sucht sie Informationen für ihren eigenen neuen Vektor.
- Mit ihrem **Key** bietet sie anderen Positionen ein Vergleichssignal an.
- Mit ihrem **Value** liefert sie den Inhalt, den andere Positionen aufnehmen
  können.

Token A ist also nicht "die Query" und Token B nicht "der Key". A hat eine Query,
einen Key und einen Value; B ebenfalls.

Wenn A Informationen sammelt, vergleichen wir **Query A mit Key A** und
**Query A mit Key B**. Daraus entstehen As Gewichte. Anschliessend mischen wir
**Value A und Value B** mit diesen Gewichten:

```text
Skalarprodukt(Query A, Key A) -> Score für Quelle A
Skalarprodukt(Query A, Key B) -> Score für Quelle B

Scores -> Softmax -> Gewichte für A

Kontext A = Gewicht für Quelle A * Value A
          + Gewicht für Quelle B * Value B
```

Wie das Skalarprodukt berechnet wird, rechnen wir im nächsten Abschnitt nach.

Für Bs Kontext beginnt derselbe Ablauf mit **Query B**. Deshalb kann B andere
Gewichte bekommen als A. Eine spätere kausale Maske schränkt zusätzlich ein,
welche Quellen eine Position überhaupt verwenden darf.

### Warum nicht einfach einen Vektor für alles nehmen?

Das geht als vereinfachte Attention-Rechnung; genau das macht unsere erste Demo.
Aber es bindet unterschiedliche Aufgaben unnötig an dieselben Zahlen.

**Query und Key trennen Suchbedarf und Angebot.** Was eine Position braucht,
muss nicht dasselbe sein wie das, was sie anderen Positionen anbieten kann.
Mit einem gemeinsamen Vektor wäre das rohe Skalarprodukt zwischen A und B in
beiden Richtungen gleich. Unterschiedliche Query- und Key-Vektoren erlauben
bereits beim Vergleich unterschiedliche Scores: A kann B stark gewichten,
ohne dass B denselben Vergleichsscore für A erhält. Die späteren
Softmax-Gewichte können auch bei symmetrischen Scores unterschiedlich sein,
weil jede Zeile separat normalisiert wird.

**Key und Value trennen Auswahlmerkmal und Inhalt.** Merkmale, die eine Quelle
auffindbar machen, müssen nicht dieselben Informationen sein, die wir aus ihr
übernehmen möchten. In der Dokumentensuche helfen Schlagwörter beim Finden;
zum Lernen brauchen wir den Erklärungstext. Entsprechend dürfen die
Value-Vektoren andere Informationen enthalten als die Key-Vektoren.

Die drei Rollen geben dem Modell damit getrennte Möglichkeiten zu lernen:
**wonach suchen, womit vergleichen, was weitergeben**.

### Woher kommen diese Vektoren später?

Im lernbaren Baustein wird jeder Eingabevektor durch drei verschiedene
Gewichtsmatrizen umgerechnet. Eine solche Umrechnung heisst *lineare Projektion*:

```text
Query = Eingabevektor @ W_Q
Key   = Eingabevektor @ W_K
Value = Eingabevektor @ W_V
```

`@` steht für Matrixmultiplikation. Die drei Matrizen sind lernbare Parameter.
Innerhalb eines Attention-Kopfes verwenden alle Positionen dieselben
Projektionsmatrizen; es gibt nicht für jedes Token eigene Matrizen.

Wir legen keine konkreten Suchfragen, Schlagwörter oder sprachlichen Aufgaben
fest. Das Training passt die Matrizen so an, dass die daraus berechneten
Kontext-Vektoren bei der Vorhersage helfen. "Suchfrage" und "Inhalt" sind
Verständnishilfen, keine von Menschen lesbaren Texte in den Vektoren.

In unserer ersten Demo überspringen wir diese Projektionen bewusst:
**`Q = K = V = X`**. Die drei Aufgaben bleiben getrennt, obwohl ihre Vektoren
hier dieselben Zahlen enthalten. So können wir erst den Ablauf nachrechnen,
bevor wir lernbare Gewichte ergänzen.

## 1. Vergleichen: Scores

### Warum vergleichen wir?

Unser Ziel ist ein neuer Vektor für jede Position, der passende Informationen
aus dem Kontext enthält. Dafür soll eine Position nicht einfach von allen
anderen Positionen gleich viel übernehmen: Je nach Suchbedarf können
unterschiedliche Quellen hilfreich sein.

Die Query beschreibt den Suchbedarf der aktuellen Position. Die Keys liefern
Vergleichssignale der möglichen Quellen. Wir vergleichen deshalb **eine Query
mit jedem Key**. Jeder Vergleich liefert eine Zahl, den Score:

```text
Query der aktuellen Position mit Key A -> Score für Quelle A
Query der aktuellen Position mit Key B -> Score für Quelle B
```

Ohne einen queryabhängigen Vergleich könnten wir beispielsweise alle Values
gleich gewichten. Dann erhielte jede Position dieselbe Zusammenfassung des
Fensters. Mit unterschiedlichen Queries können die Positionen dagegen
unterschiedliche Informationen stärker berücksichtigen.

Ein Score ist keine objektive Messung sprachlicher Bedeutung. Im lernbaren
Modell werden die Projektionen so angepasst, dass die daraus entstehenden
Vergleiche bei der Vorhersage helfen. Unsere festen Zahlen zeigen zunächst
nur die Rechenmechanik.

### Wie vergleichen wir?

Für Token A vergleichen wir seine Query mit beiden Keys. Das Skalarprodukt
multipliziert die jeweils gleich positionierten Zahlen und addiert:

```text
Query A mit Key A: 1 * 1 + 0 * 0 = 1
Query A mit Key B: 1 * 0 + 0 * 1 = 0
```

Beim zweiten Vergleich stehen die Zahlen so gegenüber:

```text
Query A: [1, 0]
Key B:   [0, 1]

Erste Stelle:  1 * 0 = 0
Zweite Stelle: 0 * 1 = 0
Summe:             0 + 0 = 0
```

Zwei Vektoren gehen hinein, eine Vergleichszahl kommt heraus. Für Token B
führen wir denselben Ablauf mit seiner Query aus:

```text
Query B mit Key A: 0 * 1 + 1 * 0 = 0
Query B mit Key B: 0 * 0 + 1 * 1 = 1
```

Damit haben wir vier Vergleiche: Jede der zwei Queries wird mit jedem der zwei
Keys verglichen. Alle vier lassen sich gemeinsam als Matrixmultiplikation
berechnen:

```text
scores = Q @ K^T

          Key A  Key B
Query A     1      0
Query B     0      1
```

`@` bedeutet Matrixmultiplikation. `K^T` vertauscht Zeilen und Spalten, damit
jede Query gegen jeden Key gerechnet wird. Warum ist das Vertauschen nötig?
In `Q` stehen die Queries als Zeilen, in `K` zunächst auch die Keys.
Matrixmultiplikation vergleicht aber **Zeilen links mit Spalten rechts**.
Durch das Transponieren wird deshalb jede Key-Zeile zu einer Key-Spalte:

```text
Q @ K^T
|   |
|   +-- Keys als Spalten
+------ Queries als Zeilen
```

`Q @ K^T` ist somit die Kurzform für unsere vier Skalarprodukte, kein weiterer
inhaltlicher Rechenschritt. Bei unserer symmetrischen Eingabe sehen `K` und
`K^T` zufällig gleich aus; das Vertauschen ist hier optisch nicht erkennbar.

Die Score-Matrix hat die Form `[Token-Anzahl, Token-Anzahl]`. Eintrag `[i, j]`
vergleicht die Query von Token `i` mit dem Key von Token `j`. Eine Zeile gehört
also einer suchenden Position, eine Spalte einer möglichen Quelle. As Zeile
liest sich: "Mein Score für Quelle A ist 1, für Quelle B ist er 0."

Ein Score ist noch kein Anteil. Entscheidend sind seine Unterschiede zu den
anderen Scores derselben Query-Zeile, nicht seine absolute Grösse. Score `0`
bedeutet noch nicht "ignorieren", Score `1` nicht "100 Prozent übernehmen".
Erst im nächsten Schritt entstehen die Anteile.

## 2. Gewichten: Softmax

### Warum brauchen wir Gewichte?

Die Scores sagen, welche Quellen im Vergleich höher oder niedriger bewertet
werden. Sie können aber beliebige Zahlen sein, auch negative. Wir benötigen
jetzt eine Antwort auf eine andere Frage: **Wie viel übernehmen wir von jeder
Quelle?**

Softmax wandelt dafür die Scores in nichtnegative Anteile um, die pro Query
zusammen `1` ergeben. Zum Beispiel:

```text
Scores [2, 1] -> Anteile ungefähr [0.731, 0.269]
```

Damit verwenden wir rund 73 Prozent des ersten und 27 Prozent des zweiten
Value-Vektors. "Prozent" ist eine Lesehilfe für die Mischung, keine
Wahrscheinlichkeit dafür, dass ein Token sprachlich richtig ist.

Warum wählen wir nicht nur die Quelle mit dem höchsten Score? Mehrere Quellen
können gleichzeitig hilfreich sein. Eine weiche Verteilung erlaubt eine
Mischung statt einer harten Ja-/Nein-Auswahl. Ausserdem ist Softmax
differenzierbar: Burn kann Gradienten durch die Gewichte zu den Query- und
Key-Projektionen zurückrechnen. So kann das Modell später lernen, anders zu
vergleichen.

Wir verwenden Softmax, weil es aus den Scores Anteile macht, die zusammen
`1` ergeben, und weil Burn durch diese Berechnung Gradienten zurückrechnen
kann. So lässt sich später lernen, welche Quellen stärker berücksichtigt
werden sollen.

### Wie berechnet Softmax die Anteile?

Softmax wandelt jede Score-Zeile in Gewichte um:

```text
Gewicht j = exp(Score j) / Summe aller exp(Scores) dieser Zeile
```

`exp(x)` ist die Exponentialfunktion. Für die erste Zeile:

```text
exp(1) = 2.71828...
exp(0) = 1
Summe  = 3.71828...

Gewicht A = 2.71828... / 3.71828... = 0.73106...
Gewicht B = 1         / 3.71828... = 0.26894...
```

Damit ergeben sich:

```text
          Value A  Value B
Query A    0.731    0.269
Query B    0.269    0.731
```

Die Gewichte sind positiv und ergeben pro Zeile zusammen ungefähr `1`
(Rundungsfehler sind möglich). Gleiche Scores bekommen gleiche Gewichte.
Wir verwenden Burns `softmax`, statt die Exponentialfunktion selbst auszurechnen.

## 3. Mischen: Kontext

### Warum müssen wir noch mischen?

Die Attention-Gewichte sagen nur, **wie viel** wir von jeder Quelle übernehmen.
Sie enthalten noch nicht die Informationen, die wir weitergeben wollen.
Diese stecken in den Values.

Ohne Mischen hätten wir lediglich eine Liste von Anteilen für passende Quellen.
Die Inhalte dieser Quellen wären noch nicht in einen neuen Token-Vektor
eingeflossen. Deshalb multiplizieren wir jeden Value-Vektor mit seinem Anteil
und addieren die Beiträge:

```text
Value A = [2, 0]
Value B = [0, 3]
Beispielhafte Anteile = [0.75, 0.25]

Neuer Vektor = 0.75 * [2, 0] + 0.25 * [0, 3]
             = [1.5, 0] + [0, 0.75]
             = [1.5, 0.75]
```

Diese Zahlen zeigen nur das Mischen; es sind nicht die Values und Gewichte
unserer ersten Demo. Jede Komponente eines Values wird mit demselben Anteil
multipliziert. Anschliessend addieren wir die jeweils gleich positionierten
Komponenten. Das Ergebnis hat dieselbe Breite wie ein Value-Vektor.

### Unsere beiden Kontext-Vektoren

Für jede Query bilden wir eine gewichtete Summe der Values:

```text
Kontext A = 0.731... * [1, 0] + 0.269... * [0, 1]
          = [0.731..., 0.269...]

Kontext B = 0.269... * [1, 0] + 0.731... * [0, 1]
          = [0.269..., 0.731...]
```

Als Matrixmultiplikation:

```text
Kontext = weights @ V
```

Jedes Token bekommt damit einen neuen Vektor, der Informationen beider Tokens
enthält. Dass Kontext und Gewichte hier dieselben Zahlen haben, liegt an unseren
Values `[1, 0]` und `[0, 1]`. Mit anderen Values wäre das nicht so.

Attention wählt nicht zwingend einen einzelnen Nachbarn aus. Sie mischt Inhalte
aus Sicht jeder Query. Diese feste Rechnung zeigt nur die Mechanik, noch kein
Sprachverständnis.

### Die drei Schritte gehören zusammen

```text
Query und Keys         -> Vergleichen -> Scores
Scores                 -> Gewichten  -> Attention-Anteile
Attention-Anteile
und Values             -> Mischen    -> Kontext-Vektor
```

**Vergleichen bestimmt die Passung. Gewichten bestimmt die Anteile.
Mischen überträgt die Inhalte.** Für jede Query entsteht so eine eigene
Zusammenfassung der erlaubten Quellen. In dieser ersten Demo sind noch alle
Positionen erlaubt; die spätere kausale Maske schränkt das ein.

Zum Mitdenken: Könnten wir nach dem Softmax aufhören? Dann hätten wir zwar
Anteile, aber noch keinen neuen Vektor aus den Inhalten. Dafür brauchen wir
die Values und die gewichtete Summe.

## Die Rechnung mit Burn

```rust,ignore
{{#include ../../../crates/model/examples/ch05.rs:attention}}
```

`Tensor<2>` ist ein zweidimensionaler Tensor. Die Demo wählt das CPU-Backend
mit `Device::ndarray()`. `clone()` erhält einen Tensor für weitere Verwendungen, weil
Burns Rechenoperationen ihre Eingaben in Rust verbrauchen.

`matmul` führt Matrixmultiplikation aus, `transpose` vertauscht die beiden
Dimensionen. `softmax(scores.clone(), 1)` normalisiert über Dimension `1`:
die Key-Spalten jeder Query-Zeile. Über Dimension `0` würden wir stattdessen
verschiedene Queries vermischen; das wäre hier falsch.

Die Funktion gibt Scores, Gewichte und Kontext zurück, damit wir jeden
Zwischenschritt ansehen können. Sie bleibt eine Lernhilfe im Example, keine
öffentliche Modell-API. Das Modell-Crate hängt weiterhin nicht vom
Training-Crate ab.

## Demo

```bash
cargo run -p model --example ch05
```

Ausgabe:

```text
{{#include ../ausgaben/ch05.txt}}
```

Vollständiger ausführbarer Code:

```rust,ignore
{{#include ../../../crates/model/examples/ch05.rs}}
```

Die Tests im Example prüfen unsere Rechnung, die zeilenweise Normalisierung
und einen zweiten Fall: Eine Query `[0, 0]` liefert für beide Keys den Score `0`.
Die Gewichte sind dann `[0.5, 0.5]`; Values `[0, 0]` und `[0, 2]` ergeben
den Kontext `[0, 1]`. Anders als im ersten Beispiel sind Gewichte und Kontext
hier verschieden.

```bash
cargo test -p model --example ch05
```

## Schritt 2: Drei lernbare Projektionen

### Was war bisher festgelegt?

In der ersten Demo haben wir denselben Eingabevektor unverändert für alle
drei Aufgaben verwendet:

```text
Token B, Eingabe: [0, 1]

Query B: [0, 1]
Key B:   [0, 1]
Value B: [0, 1]
```

So konnten wir Vergleichen, Gewichten und Mischen einzeln kennenlernen.
Aber das Modell kann damit noch nicht unabhängig bestimmen, wie es einen
Eingabevektor zum Suchen, zum Vergleichen und zum Weitergeben aufbereitet.

Jetzt ergänzen wir deshalb **vor dem Vergleichen** drei Umrechnungen:

```text
                    -> Umrechnung für die Query -> Query
Eingabe eines Tokens -> Umrechnung für den Key   -> Key
                    -> Umrechnung für den Value -> Value
```

Alle drei erhalten dieselbe Eingabe. Jede erzeugt einen eigenen Ausgabevektor.
Die Eingabe selbst bleibt unverändert. Diese Umrechnung heisst hier
**Projektion**. Das Wort bedeutet in diesem Kapitel: Aus den Eingabezahlen
durch Multiplikation und Addition neue Zahlen berechnen. Die Ausgabe muss
dabei weder kleiner werden noch eine für Menschen lesbare Bedeutung haben.

### Erst eine Umrechnung, noch keine Matrix

Wir legen für die Query zunächst diese zwei Regeln fest:

```text
Erste Query-Zahl  = erste Eingabezahl + zweite Eingabezahl
Zweite Query-Zahl = zweite Eingabezahl
```

Wenden wir sie auf Token B an:

```text
Eingabe B: [0, 1]

Erste Query-Zahl:  0 + 1 = 1
Zweite Query-Zahl:     1 = 1

Query B: [1, 1]
```

Für Token A verwenden wir **dieselben Regeln**:

```text
Eingabe A: [1, 0]

Erste Query-Zahl:  1 + 0 = 1
Zweite Query-Zahl:     0 = 0

Query A: [1, 0]
```

Die Regeln unterscheiden sich also nicht je Token. Unterschiedliche
Eingabezahlen können trotzdem unterschiedliche Queries ergeben.

### Wo stecken die Gewichte?

Schreiben wir die Query-Regeln etwas ausführlicher:

```text
Erste Query-Zahl  = erste Eingabezahl * 1 + zweite Eingabezahl * 1
Zweite Query-Zahl = erste Eingabezahl * 0 + zweite Eingabezahl * 1
```

Die vier Faktoren `1`, `1`, `0`, `1` bestimmen die Umrechnung. Diese Faktoren
heissen **Projektionsgewichte**. Statt die Regeln als eigenen Programmcode
für jede Ausgabezahl zu schreiben, sammeln wir ihre Faktoren in einer Tabelle:

```text
                        Erste Query-Zahl   Zweite Query-Zahl
Erste Eingabezahl               1                  0
Zweite Eingabezahl              1                  1
```

Eine solche rechteckige Zahlentabelle heisst **Matrix**. Unsere Tabelle nennen
wir `W_Q`: `W` steht für die Gewichte, `Q` für die Query-Projektion.
In der Schreibweise des Codes sieht sie so aus:

```text
W_Q = [[1, 0],
       [1, 1]]
```

Jede innere Liste ist eine Zeile. Um eine Query-Zahl zu berechnen, nehmen wir
die Faktoren ihrer **Spalte**, multiplizieren sie mit den Eingabezahlen
und addieren. Für Token B `[0, 1]`:

```text
Spalte 0: [1, 1] -> erste Query-Zahl  = 0 * 1 + 1 * 1 = 1
Spalte 1: [0, 1] -> zweite Query-Zahl = 0 * 0 + 1 * 1 = 1
```

Das sind genau unsere beiden Regeln, nur als Tabelle gespeichert.
Die Kurzschreibweise **`Eingabe @ W_Q`** führt diese Multiplikationen und
Additionen für alle Ausgabespalten aus. Sie heisst Matrixmultiplikation.
Du musst dafür keine neue Rechenart lernen: Wir bündeln die bereits gezeigten
Einzelrechnungen.

### Key und Value bekommen andere Regeln

Warum andere Regeln? Nicht, weil für Keys das Tauschen und für Values das
Verdoppeln vorgeschrieben wäre. **Die konkreten Regeln wählen wir nur für
diese Demo**, damit wir unterschiedliche Rollen mit kleinen Zahlen
nachrechnen können. Das gilt auch für die Query-Regel aus dem vorherigen
Abschnitt.

Für den Key wählen wir eine andere Umrechnung: die zwei Eingabezahlen tauschen.
Für den Value verdoppeln wir die erste Zahl und verdreifachen die zweite:

```text
Key:
Erste Key-Zahl  = zweite Eingabezahl
Zweite Key-Zahl = erste Eingabezahl

Value:
Erste Value-Zahl  = erste Eingabezahl * 2
Zweite Value-Zahl = zweite Eingabezahl * 3
```

Auch diese Regeln lassen sich mit je vier Faktoren in einer Matrix speichern:

```text
W_K = [[0, 1],       W_V = [[2, 0],
       [1, 0]]              [0, 3]]
```

Rechnen wir wieder nur Token B `[0, 1]` durch:

```text
Erste Key-Zahl:    0 * 0 + 1 * 1 = 1
Zweite Key-Zahl:   0 * 1 + 1 * 0 = 0
Key B: [1, 0]

Erste Value-Zahl:  0 * 2 + 1 * 0 = 0
Zweite Value-Zahl: 0 * 0 + 1 * 3 = 3
Value B: [0, 3]
```

Aus **einem** Eingabevektor `[0, 1]` haben wir somit drei verschiedene
Vektoren erzeugt: Query `[1, 1]`, Key `[1, 0]` und Value `[0, 3]`.
Es sind weiterhin drei Rollen derselben Token-Position, keine neuen Tokens.

### Warum drei getrennte Umrechnungen?

Wir wollen drei Aufgaben unabhängig voneinander lernen lassen:

- Die Query-Projektion bestimmt, welche Merkmale eine Position beim Suchen
  verwendet.
- Die Key-Projektion bestimmt, welche Merkmale eine Position zum Vergleich
  anbietet.
- Die Value-Projektion bestimmt, welche Informationen eine Position weitergibt.

Würden alle drei dieselbe Gewichtsmatrix verwenden, würde eine Änderung dieser
Matrix gleichzeitig Suchfragen, Vergleichssignale und Inhalte verändern.
Mit drei getrennten Matrizen kann das Training beispielsweise die
Vergleichssignale anpassen, ohne dabei zwangsläufig die Values zu verändern.
Bei gleicher Eingabe bleiben die anderen Projektionen dann unverändert.

Die Regeln **müssen nicht zwingend unterschiedlich sein**. Getrennte Matrizen
geben dem Modell die Möglichkeit, unterschiedliche Umrechnungen zu lernen.
Es gibt keine Vorschrift, wie stark sie sich unterscheiden sollen. Für unsere
Demo haben wir unterschiedliche Zahlen bewusst gewählt, um diese Möglichkeit
sichtbar zu machen.

Für beide Tokens zusammen erhalten wir:

```text
             Eingabe    Query     Key       Value
Token A      [1, 0]     [1, 0]    [0, 1]    [2, 0]
Token B      [0, 1]     [1, 1]    [1, 0]    [0, 3]
```

**Bis hier haben wir noch keine Nachbarinformationen aufgenommen.** Jede
Projektion verarbeitet die Eingabe eines Tokens für sich. Erst die folgenden
Query-Key-Vergleiche verbinden verschiedene Positionen; beim Mischen werden
ihre Inhalte übernommen. Die Reihenfolge ist:

```text
Eingabe -> Q/K/V erzeugen -> vergleichen -> gewichten -> Values mischen
```

### Warum heisst das "lernbar"?

Unsere drei Umrechnungsregeln sind keine Vorgaben dafür, wie Sprache
funktioniert. Ihre Faktoren sind bewusst einfache Beispielzahlen.
Im normalen Modell initialisiert Burn diese Zahlen zufällig.

Dabei müssen wir zwei Entscheidungen auseinanderhalten:

**Die Rechenform legen wir als Entwickler fest.** In unserem Baustein soll
jede Projektion eine Matrixmultiplikation sein:

```text
Query = Eingabe @ W_Q
Key   = Eingabe @ W_K
Value = Eingabe @ W_V
```

Wir wählen auch die Grössen der Matrizen. Diese Rechenform und die Grössen
ändern sich beim Training unseres Bausteins nicht.

**Die Zahlen in den Matrizen sind die anpassbaren Parameter.** Sie bestimmen,
welche konkrete Umrechnung die festgelegte Rechenform ausführt. In der Demo
geben wir sie vor. Im normalen Modell beginnt Burn mit zufälligen Werten.
Diese Startwerte sind noch keine sinnvoll gelernten Such- oder Vergleichsregeln.

Wie werden daraus nützliche Umrechnungen? Im späteren Training läuft wiederholt
dieser Ablauf:

1. Das Modell verarbeitet Text und sagt nächste Tokens vorher. Dabei verwendet
   es auch seine aktuellen Query-, Key- und Value-Projektionen.
2. Eine Verlustfunktion bewertet, wie gut die Vorhersagen zu den tatsächlichen
   nächsten Tokens passen.
3. Burn berechnet Gradienten, auch für die Zahlen in `W_Q`, `W_K` und `W_V`.
   Sie zeigen, wie kleine Parameteränderungen den Verlust beeinflussen.
4. Ein Optimizer verändert die Parameter anhand dieser Gradienten, um den
   Verlust zu verringern. Nicht jeder einzelne Schritt muss die Vorhersagen
   verbessern.

Wir schreiben dabei keine Suchfragen wie "Finde das Verb" in die Matrizen.
Die Rückmeldung kommt aus der Vorhersageaufgabe. Über viele Trainingsschritte
können dadurch Umrechnungen entstehen, die für diese Aufgabe hilfreich sind.
Ihre einzelnen Zahlen müssen für uns keine leicht benennbare Bedeutung haben.

Im späteren Training verändert ein Optimizer die Zahlen in den Matrizen
mithilfe ihrer Gradienten. Dadurch verändert sich die Umrechnungsregel, ohne
dass wir den Programmcode ändern. Ein einzelnes Beispiel:

```text
Bisherige erste Query-Spalte: [1, 1]
Andere erste Query-Spalte:    [1, 2]

Für dieselbe Eingabe B [0, 1]:
Bisherige erste Query-Zahl = 0 * 1 + 1 * 1 = 1
Andere erste Query-Zahl    = 0 * 1 + 1 * 2 = 2
```

Eine geänderte Zahl in `W_Q` erzeugt also eine andere Query. Damit können
sich später Scores, Attention-Anteile und Kontext ändern. Entsprechend
verändert `W_K` die Keys und `W_V` die weitergegebenen Inhalte.

**Lernbar** bedeutet: Diese Zahlen sind Modellparameter, deren Gradienten
Burn berechnen kann und die ein Optimizer verändern kann. **Gelernt** sind
sie erst nach solchen Trainingsschritten. Unsere Demo verändert noch nichts
durch Training.

### Die Grössen der Tabellen benennen

Jetzt können wir auch die Namen im Code zuordnen:

- `embedding_dim`: Anzahl der Zahlen im Eingabevektor, hier `2`.
- `head_dim`: Anzahl der Zahlen im erzeugten Query-, Key- oder Value-Vektor,
  in unserem Baustein ebenfalls `2`.

Eine Projektionsmatrix hat eine Zeile pro Eingabezahl und eine Spalte pro
Ausgabezahl. Ihre Form ist deshalb `[embedding_dim, head_dim]`, hier `[2, 2]`.
Hätten wir vier Eingabezahlen und wollten zwei Ausgabezahlen, bräuchten wir
vier Zeilen und zwei Spalten. Die Rechenregel bleibt dieselbe.

Zum Mitdenken: Entstehen durch die drei Projektionen drei neue Textpositionen?
Nein. Für jede vorhandene Position entstehen drei Zahlenvektoren mit
unterschiedlichen Aufgaben. Anzahl und Reihenfolge der Tokens bleiben gleich.

### Zwei verschiedene Bedeutungen von "Gewichten"

Hier kann die Sprache verwirren. Wir verwenden zwei unterschiedliche Arten
von Gewichten:

- **Projektionsgewichte** sind die Zahlen in `W_Q`, `W_K` und `W_V`.
  Sie gehören zum Modell und bleiben bei einem normalen `forward` unverändert.
- **Attention-Gewichte** sind die Anteile, die Softmax aus den Scores berechnet.
  Sie entstehen für die jeweilige Eingabe neu und bestimmen, wie die Values
  gemischt werden.

Dass wir beide "Gewichte" nennen, macht sie nicht zum selben Gegenstand.
Projektionsgewichte erzeugen Vektoren; Attention-Gewichte verteilen Anteile
auf Quellen. Im Code heisst die Matrix einer Projektion `weight`, die
berechnete Attention-Verteilung dagegen `weights`.

Der Datenfluss hilft, die Aufgaben auseinanderzuhalten:

```text
Eingabe -> W_Q -> Query --+
                        +-> Scores -> Skalierung -> Softmax -> Attention-Gewichte
Eingabe -> W_K -> Key ----+

Eingabe -> W_V -> Values

Attention-Gewichte und Values -> gewichtete Summe -> Kontext
```

Ein Pfeil bedeutet: Der nächste Schritt verwendet das vorherige Ergebnis.
Die Pluszeichen verbinden hier nur die gezeichneten Linien, keine Additionen.
**Der Value-Weg führt nicht durch die Score-Berechnung.** Er trifft erst beim
Mischen mit den Attention-Gewichten zusammen.

### Warum die Scores skalieren?

Jetzt sind die beiden Arten von Gewichten getrennt: Die Projektionsgewichte
erzeugen Query, Key und Value; die Attention-Gewichte bestimmen später die
Mischung. Folgen wir nun dem Weg von Query und Key im Datenflussbild weiter.
Aus ihrem Vergleich entstehen Scores.

Bisher haben wir diese Scores direkt an Softmax übergeben. Jetzt ergänzen wir
den im Bild bereits eingezeichneten Zwischenschritt: die **Skalierung**.
Dabei teilen wir die Scores durch die Quadratwurzel der Query-/Key-Breite.
Es entsteht also keine weitere Art von Gewichten; wir verändern die Scores,
bevor Softmax daraus Attention-Gewichte berechnet. Warum brauchen wir das?

Ein Skalarprodukt addiert für jede Vektorkomponente ein Produkt. Bei breiteren
Query-/Key-Vektoren kommen mehr Summanden zusammen. Dadurch können die Scores
stärker streuen, obwohl die einzelnen Komponenten ähnlich gross bleiben.

Softmax reagiert auf grosse Unterschiede mit sehr ungleichen Gewichten.
Ein Beispiel:

```text
softmax([1, 0])  = ungefähr [0.731, 0.269]
softmax([10, 0]) = ungefähr [0.99995, 0.00005]
```

Wenn ein Gewicht fast `1` ist und alle anderen fast `0`, reagiert Softmax nur
noch schwach auf kleine Score-Änderungen. Das kann das Lernen erschweren.
Deshalb teilen wir **vor Softmax** durch die Quadratwurzel der Query-/Key-Breite:

```text
scaled_scores = (Q @ K^T) / sqrt(head_dim)
```

Die Quadratwurzel einer Zahl ist die positive Zahl, die mit sich selbst
multipliziert wieder die Ausgangszahl ergibt. Zum Beispiel ist `sqrt(4) = 2`,
weil `2 * 2 = 4`. Für unsere Breite `2` ist `sqrt(2)` ungefähr `1.414`.

Wir ändern dabei nicht, welcher Score grösser ist. Wir verkleinern die
Unterschiede, bevor Softmax daraus Gewichte bildet. In unserem Beispiel wird
aus `[0, 1]` ungefähr `[0, 0.707]`: Der zweite Eintrag bleibt grösser, bekommt
aber ein weniger stark dominierendes Gewicht.

**Für den Rechenweg genügt:** Die Query-/Key-Breite liefert den Teiler.
Skalierung kommt vor Softmax und verhindert, dass die Vektorbreite allein
die Gewichte unnötig zuspitzt. Sie garantiert keine kleinen Scores und keine
gleichmässigen Gewichte; das hängt auch von den Projektionen ab.

Die statistische Begründung für die Quadratwurzel ist optional:
[Warum Attention durch die Quadratwurzel teilt](../exkurse/attention-skalierung.md).

### Den neuen Kontext nachrechnen

Jetzt verbinden wir die bisherigen Schritte zu einer vollständigen
Handrechnung. Wir verwenden weiterhin dieselben beiden Tokens. Die
Projektionen haben bereits diese Vektoren erzeugt:

```text
          Query     Key       Value
Token A   [1, 0]    [0, 1]    [2, 0]
Token B   [1, 1]    [1, 0]    [0, 3]
```

Unser Ziel ist **ein neuer Kontext-Vektor für A und einer für B**.
Wir berechnen zunächst As Anteile und mischen damit die Values. Danach
wiederholen wir denselben Ablauf für B. Die Projektionen selbst ändern wir
dabei nicht; es findet kein Training statt.

#### 1. Jede Query mit beiden Keys vergleichen

Für A rechnen wir:

```text
Query A mit Key A: 1 * 0 + 0 * 1 = 0
Query A mit Key B: 1 * 1 + 0 * 0 = 1
```

As unskalierte Score-Zeile ist also `[0, 1]`. Für B:

```text
Query B mit Key A: 1 * 0 + 1 * 1 = 1
Query B mit Key B: 1 * 1 + 1 * 0 = 1
```

Bs Score-Zeile ist `[1, 1]`. Zusammen:

```text
             Key A [0, 1]   Key B [1, 0]
Query A [1, 0]     0              1
Query B [1, 1]     1              1
```

Die erste Zeile gehört zu A, die zweite zu B. Innerhalb jeder Zeile steht
zuerst der Vergleich mit Quelle A, dann der mit Quelle B.

#### 2. Alle Scores durch denselben Faktor teilen

Query und Key enthalten jeweils zwei Zahlen. Deshalb ist `head_dim = 2`,
und wir teilen jeden Score durch `sqrt(2)`, ungefähr `1.414`:

```text
Für A: [0 / sqrt(2), 1 / sqrt(2)] = ungefähr [0.000, 0.707]
Für B: [1 / sqrt(2), 1 / sqrt(2)] = ungefähr [0.707, 0.707]
```

Das ist die Skalierung, noch nicht die Umwandlung in Anteile.
A bewertet Quelle B weiterhin höher; bei B bleiben beide Scores gleich.

#### 3. Für jede Query eigene Attention-Anteile berechnen

Softmax verarbeitet As Zeile getrennt von Bs Zeile. Für A berechnen wir
zuerst die Exponentialwerte:

```text
Quelle A: exp(0)           = 1
Quelle B: exp(1 / sqrt(2)) = ungefähr 2.02811
Summe                     = ungefähr 3.02811

Anteil für Quelle A = 1       / 3.02811... = ungefähr 0.330238
Anteil für Quelle B = 2.02811... / 3.02811... = ungefähr 0.669762
```

Die Anteile ergeben zusammen `1`. A übernimmt also ungefähr 33 Prozent von
Value A und 67 Prozent von Value B.

Bei B sind beide skalierten Scores gleich. Deshalb sind auch ihre
Exponentialwerte gleich:

```text
Quelle A: exp(1 / sqrt(2)) = ungefähr 2.02811
Quelle B: exp(1 / sqrt(2)) = ungefähr 2.02811
Summe                     = ungefähr 4.05623

Anteil für Quelle A = 0.5
Anteil für Quelle B = 0.5
```

Bs Anteile sind also jeweils 50 Prozent. Die beiden Zeilen der
Attention-Gewichte lauten:

```text
           Anteil Quelle A   Anteil Quelle B
Für A          0.330238...        0.669762...
Für B          0.5                0.5
```

#### 4. Die Values mit diesen Anteilen mischen

Jetzt mischen wir **die projizierten Values**, nicht die Eingabevektoren:

```text
Value A = [2, 0]
Value B = [0, 3]
```

Für As Kontext erhält jede Zahl von Value A den Faktor `0.330238...`;
jede Zahl von Value B erhält den Faktor `0.669762...`. Anschliessend
addieren wir die Beiträge an der jeweils gleichen Stelle:

```text
Erste Kontext-Zahl für A:
0.330238... * 2 + 0.669762... * 0 = 0.660477...

Zweite Kontext-Zahl für A:
0.330238... * 0 + 0.669762... * 3 = 2.009285...

Kontext A = [0.660477..., 2.009285...]
```

Für B verwenden wir dieselben Values, aber Bs eigene Anteile:

```text
Erste Kontext-Zahl für B:  0.5 * 2 + 0.5 * 0 = 1
Zweite Kontext-Zahl für B: 0.5 * 0 + 0.5 * 3 = 1.5

Kontext B = [1, 1.5]
```

#### Was hat sich gegenüber der Eingabe verändert?

```text
          Eingabe    Neuer Kontext
Token A   [1, 0]     [0.660477..., 2.009285...]
Token B   [0, 1]     [1, 1.5]
```

Die Anzahl der Positionen bleibt zwei. Jede Position hat aber einen neuen
Vektor, der Beiträge beider Values enthält. Beide Positionsausgaben verwenden
dieselben Quellen, gewichten sie jedoch unterschiedlich.

Dabei hat keine Zahl automatisch eine benennbare sprachliche Bedeutung.
Wir haben nur gezeigt, wie getrennte Projektionen, Vergleich, Skalierung,
Softmax und Mischung rechnerisch zusammenwirken.

Die Demo zeigt die Ergebnisse auf drei Nachkommastellen gerundet. In der
Handrechnung solltest du die ungerundeten Anteile weiterverwenden: Mit
`0.670 * 3` bekämst du beispielsweise `2.010` statt der Demo-Ausgabe `2.009`.
Das ist kein anderer Rechenweg, sondern ein Rundungsunterschied.

### Angeleitete Übung: Nur den Inhalt verändern

Wir lassen Eingabe, `W_Q` und `W_K` unverändert. Nur die Value-Matrix ersetzen
wir gedanklich:

```text
Bisher: W_V = [[2, 0],     Neu: W_V = [[4, 0],
               [0, 3]]                 [0, 6]]
```

Was verändert sich? Verfolge die beiden Wege in der Zeichnung:

1. Query und Key verwenden weiterhin dieselben Eingaben und Matrizen.
   Ihre Vektoren bleiben gleich.
2. Deshalb bleiben auch Scores, skalierte Scores und Attention-Gewichte gleich.
   Für A verwenden wir weiterhin ungefähr `[0.330, 0.670]`.
3. Die neue Value-Matrix liefert dagegen Value A `[4, 0]` und Value B `[0, 6]`.
4. Mit denselben Anteilen mischen wir nun andere Inhalte:

```text
Neuer Kontext A = 0.330... * [4, 0] + 0.670... * [0, 6]
                = [1.320..., 4.018...]

Neuer Kontext B = 0.5 * [4, 0] + 0.5 * [0, 6]
                = [2, 3]
```

Hier haben wir alle Value-Zahlen verdoppelt, also verdoppelt sich auch der
Kontext. Die Auswahlanteile ändern sich nicht. Allgemein kann eine Änderung
von `W_V` den Kontext verändern; nicht jede denkbare Änderung muss bei jeder
Eingabe sichtbar werden.

**Zum Mitdenken:** Wenn du nur `W_K` änderst, welchen Weg betrifft das?
Es ist der Vergleichsweg: Keys und damit möglicherweise Scores und
Attention-Gewichte ändern sich. Die Values bleiben gleich, können aber mit
anderen Anteilen gemischt werden. Auch dadurch kann sich der Kontext ändern.

## Ein trainierbarer Burn-Baustein

Die Handrechnung ist vollständig. Jetzt betrachten wir, wie derselbe Ablauf
in Rust mit Burn umgesetzt wird. Dabei kommen keine neuen Attention-Schritte
hinzu, sondern Modellverwaltung und eine Batch-Achse.

Der Abschnitt hat fünf Stationen:

1. [Modell aufbauen](#modell-aufbauen): Welche Werte speichert der Baustein?
2. [Eingabe verstehen](#eingabe-verstehen): Wie sind Fenster und Tokens gruppiert?
3. [Berechnung verfolgen](#berechnung-verfolgen): Wie setzt `forward` die
   Handrechnung um?
4. [Demo ausführen](#demo-ausführen): Wie setzen wir unsere festen Beispielzahlen?
5. [Lernbarkeit prüfen](#lernbarkeit-prüfen): Können Gradienten zurückfliessen?

Lies zuerst den Aufbau, dann die Eingabeform und anschliessend `forward`.
Die Übungen stehen jeweils nach der vollständigen Erklärung, nicht mitten
im Rechenweg. Rust-Begriffe erklären wir getrennt von den Rechenoperationen.

### Modell aufbauen

Der Baustein liegt in `crates/model/src/attention.rs`. Er hat zwei Teile:
Die Config beschreibt die Grössen; das Modell speichert die drei Projektionen.

```rust,ignore
{{#include ../../../crates/model/src/attention.rs:config}}
```

`Linear` ist Burns lernbare lineare Projektion. Wir verwenden drei getrennte
Instanzen, jede mit eigener Gewichtsmatrix. Das `Module`-Derive macht ihre
Parameter für Burn zugänglich. Das Backend wählen wir über das Device:
`Device::ndarray()` für die Demo, mit `.autodiff()` für Gradienten.

#### Drei Schichten initialisieren

Ein `impl`-Block enthält Funktionen, die zu einem Typ gehören. Hier gehören
`new` und `init` zur Config `SelfAttentionConfig`. `new` speichert die
gewünschten Grössen; `init` erzeugt daraus die drei Schichten:

```rust,ignore
{{#include ../../../crates/model/src/attention.rs:config_impl}}
```

Beide Funktionen stehen innerhalb derselben äusseren geschweiften Klammern
von `impl SelfAttentionConfig`. `new` gibt mit `Self` eine Config zurück,
`init` dagegen ein Modell vom Typ `SelfAttention`.

`with_bias(false)` lässt einen zusätzlichen lernbaren Summanden weg. So bleibt
die Berechnung genau `X @ W` und passt zur Handrechnung. Dafür schreiben wir
keine eigene Matrixmultiplikation oder Parameterverwaltung.

#### Rust und Burn: Den Aufbau lesen

Eine `struct` bündelt zusammengehörige Werte. Unsere Config enthält nur zwei
Grössen: `embedding_dim` und `head_dim`. `usize` ist Rusts Ganzzahltyp für
Grössen und Indizes. Das Modell selbst enthält dagegen drei `Linear`-Schichten,
die jeweils eine Gewichtsmatrix und die dazugehörige Berechnung verwalten.
`pub` macht diese Felder öffentlich; so kann die Demo die Gewichte ersetzen.

`SelfAttention` ist nicht fest auf einen Backend-Typ beschränkt.
Seit Burn 0.22 gibt es dafür keinen Typparameter `B` mehr.
Die Demo übergibt ein `Device::ndarray()`: CPU-Berechnung mit standardmässig
32-Bit-Gleitkommazahlen. Die Tensoren tragen ihre Device-Zuordnung zur Laufzeit.

`#[derive(Module, Debug)]` bittet Rust und Burn, Hilfscode zu erzeugen.
`Module` ermöglicht die Verwaltung der enthaltenen Modellparameter; `Debug`
ermöglicht eine Debug-Darstellung. Diesen erzeugten Code musst du nicht selbst
schreiben oder für die Attention-Rechnung nachvollziehen.

Das Modell hat einen eigenen `impl`-Block für `SelfAttention`. Darin steht
`forward`, das die eigentliche Rechnung ausführt. Diesen Block betrachten wir
vollständig unter [Berechnung verfolgen](#berechnung-verfolgen).
Die Trennung ist wichtig: **Initialisieren erzeugt Parameter; `forward`
verwendet sie, ohne sie zu trainieren.**

In `init` ist `device: &Device` ein ausgeliehenes Gerät der gewählten
Rechenumgebung. `&` bedeutet hier, dass wir das Gerät verwenden, ohne es zu
übernehmen. Burns `LinearConfig` beschreibt eine einzelne Projektion.
Wir rufen ihr `init` dreimal auf: gleiche Grössen, aber drei getrennte
Schichten und Gewichtsmatrizen.

Für die ersten Durchgänge reicht es, `Module`, `Device` und die Initialisierung
als Burn-Bausteine einzuordnen. Den Datenfluss in `forward` solltest du dagegen
Schritt für Schritt verfolgen können. Weitere Rust-Grundlagen stehen im Exkurs
[Rust-Grundlagen für das Embedding](../exkurse/rust-grundlagen-embedding.md);
die Framework-Rollen erklärt [Burn: Die Bausteine](../exkurse/burn-grundlagen.md).

### Eingabe verstehen

#### Vom einzelnen Fenster zum Batch

Ein **Tensor** ist hier eine Sammlung von Zahlen mit einer Form. Bisher war die
Form `[2, 2]`: zwei Token-Zeilen mit je zwei Zahlen. Für mehrere unabhängige
Textfenster brauchen wir eine zusätzliche äussere Gruppierung, den **Batch**.

Unsere neue Demo verwendet genau ein Fenster. Stell es dir so vor:

```text
Batch
|
+-- Fenster 0
    |
    +-- Token A: [1, 0]
    |
    +-- Token B: [0, 1]
```

In verschachtelten Listen stehen dieselben Zahlen:

```text
[
    [             <- Fenster 0
        [1, 0],   <- Token A
        [0, 1]    <- Token B
    ]
]
```

Die Kommentare mit Pfeilen gehören nur zur Darstellung, nicht zum Rust-Code.
Die Form ist `[1, 2, 2]`: **ein Fenster, zwei Tokens pro Fenster, zwei Zahlen
pro Token**. Wir haben keine zusätzlichen Token-Eigenschaften erfunden; nur
die Gruppierung um eine Ebene erweitert.

Jede Stelle in der Form gehört zu einer **Achse**. Die Achsen zählen ab `0`:

```text
Achse 0: Welches Fenster?          Grösse 1
Achse 1: Welche Token-Position?    Grösse 2
Achse 2: Welche Zahl im Vektor?    Grösse 2
```

"Dreidimensional" bedeutet hier **drei Achsen**, nicht drei Zahlen pro Token.
Die Vektorbreite ist die Grösse der letzten Achse. Zwei Fenster mit jeweils
zwei Tokens und Vektorbreite zwei hätten die Form `[2, 2, 2]` und insgesamt
`2 * 2 * 2 = 8` Zahlen.

Das Embedding liefert genau diese Gruppierung. Unser Attention-Kopf übernimmt
deshalb die Form `[Batch, Sequenzlänge, embedding_dim]`.

#### Angeleitete Übung: Eine Form lesen

Bei einer **Eingabe** ist unsere feste Vereinbarung:

```text
Form:         [Batch, Tokens pro Fenster, Zahlen pro Token]
Beispiel:     [    2,                  3,                4]
```

Lies die Form von links nach rechts. Wir haben zwei Fenster. Öffnen wir eines,
finden wir darin drei Token-Vektoren. Öffnen wir einen Token-Vektor, finden wir
vier Zahlen:

```text
Fenster 0: Token 0 [_, _, _, _]
           Token 1 [_, _, _, _]
           Token 2 [_, _, _, _]

Fenster 1: Token 0 [_, _, _, _]
           Token 1 [_, _, _, _]
           Token 2 [_, _, _, _]
```

Die Unterstriche sind Platzhalter, keine Werte. Insgesamt sind es
`2 * 3 * 4 = 24` Zahlen. Mehr Fenster vergrössern den Batch, nicht die
Vektoren der einzelnen Tokens.

**Zum Mitdenken:** Was wäre bei `[1, 3, 4]` anders?
Nur die Zahl der Fenster: eines statt zwei. Pro Fenster bleiben drei Tokens
mit jeweils vier Zahlen.

Die Form allein nennt keine Bedeutung. `[2, 3, 4]` könnte in einem anderen
Programm etwa zwei Bilder mit drei Zeilen und vier Spalten beschreiben.
Erst die Vereinbarung unseres Modells sagt, welche Achse wofür steht.

### Berechnung verfolgen

Zuerst betrachten wir die Formen des gesamten Rechenwegs. Danach lesen wir
`forward` in derselben Reihenfolge: prüfen, projizieren, vergleichen und
skalieren, gewichten, mischen.

#### Die Formen im Überblick

Der Rechenweg hat diese Formen:

```text
Eingabe             [Batch, Sequenzlänge, embedding_dim]
Q, K, V             [Batch, Sequenzlänge, head_dim]
K mit Tausch        [Batch, head_dim, Sequenzlänge]
Scores und Gewichte [Batch, Sequenzlänge, Sequenzlänge]
Kontext             [Batch, Sequenzlänge, head_dim]
```

`head_dim` darf von `embedding_dim` abweichen. Ohne Ausgabeprojektion bleibt
die Ausgabe dann entsprechend breiter oder schmaler als die Eingabe.

```rust,ignore
{{#include ../../../crates/model/src/attention.rs:model_impl}}
```

#### Eingabe prüfen

Die Signatur `forward(&self, input: Tensor<3>) -> Tensor<3>` sagt:
Wir leihen das Modell mit `&self` aus, übernehmen einen Eingabetensor und liefern
einen Ausgabetensor zurück. Die `3` ist die Anzahl der Achsen, nicht deren
Grösse. Welche konkreten Grössen der Tensor hat, liefert `dims()`.

`let [batch_size, context_length, embedding_dim] = input.dims();` verteilt diese
drei Grössen auf verständliche Namen. Dieses Zerlegen eines Arrays heisst
*Destrukturierung*. Die folgende Zeile liest die beiden Grössen der Query-Matrix:
Eingabebreite und Head-Breite.

Die Assertions prüfen zuerst, ob die Eingabe zum Modell passt. `assert!`
verlangt eine wahre Bedingung, `assert_eq!` zwei gleiche Werte. Ist eine
Bedingung verletzt, bricht die Berechnung mit der angegebenen Meldung ab.
Diese Zeilen sind Formprüfungen, nicht Teil der Attention-Mathematik.

#### Query, Key und Value erzeugen

Die nächsten drei Zeilen berechnen unabhängig voneinander:

```text
query = Eingabe @ Query-Gewichte
key   = Eingabe @ Key-Gewichte
value = Eingabe @ Value-Gewichte
```

Hier wird Rusts Ownership sichtbar: Die Projektionsmethode übernimmt ihren
Tensor als Wert. Nach der Übergabe kann dieselbe Variable nicht nochmals
verwendet werden. Query und Key bekommen deshalb jeweils `input.clone()`;
erst Value bekommt das Original. Bei Burn teilt ein Tensor-Clone die
zugrunde liegenden Daten, statt sofort alle Zahlen zu kopieren.

#### Vergleichen, skalieren und maskieren

`key.swap_dims(1, 2)` tauscht für jedes Fenster die Token- und Vektorachse.
Aus `[Batch, Tokens, Head-Breite]` wird `[Batch, Head-Breite, Tokens]`.
So kann `matmul` jede Query gegen jeden Key vergleichen. Für unser Fenster:

```text
Query      [1, 2, 2]
Key-Tausch [1, 2, 2]
Scores     [1, 2, 2]
```

Die Grössen sehen hier zufällig identisch aus. Bei drei Tokens und Head-Breite
zwei wären sie `[1, 3, 2]`, `[1, 2, 3]` und `[1, 3, 3]`. Die zwei inneren
Grössen der Matrixmultiplikation müssen übereinstimmen: hier jeweils `2`.

`(head_dim as f64).sqrt()` liefert den Skalierungsfaktor. `as f64` wandelt die
Ganzzahlbreite in eine Gleitkommazahl um; `.sqrt()` berechnet deren Quadratwurzel.
Durch diesen Faktor teilen wir sämtliche Scores.

Danach folgt die **kausale Maskierung**:
`Tensor::ones(...).triu(1).bool().unsqueeze()` erzeugt eine obere Dreiecksmatrix,
die überall dort `true` ist, wo eine Position in die Zukunft schauen würde
(über der Hauptdiagonale). `mask_fill(mask, f32::NEG_INFINITY)` ersetzt diese
Zukunftswerte durch $-\infty$. Wenn die Sequenz nur ein einzelnes Token hat
(`context_length == 1`), gibt es keine Zukunft — die Maskierung kann übersprungen
werden.

Die Batch-Achse bleibt bei diesen Operationen unverändert. `matmul` rechnet
jedes Fenster separat; Fenster tauschen keine Informationen aus.

#### Attention-Anteile berechnen

Nach dem Vergleich hat sich die Bedeutung der letzten Achse geändert:
In `scores` steht sie für **Key-Positionen**, nicht mehr für Vektorkomponenten.
Deshalb bedeutet `softmax(scores, 2)`: Gewichte entlang der Key-Positionen
berechnen, für jede Query und jedes Fenster separat.

#### Values mischen und Kontext zurückgeben

Die letzte Zeile `weights.matmul(value)` mischt die Value-Vektoren. In Rust
wird ein letzter Ausdruck ohne Semikolon zum Rückgabewert der Funktion.
Für drei Tokens mit Head-Breite zwei ist die Formrechnung:

```text
Gewichte [1, 3, 3] @ Values [1, 3, 2] -> Kontext [1, 3, 2]
```

Die Kontextausgabe enthält wieder einen Vektor pro Token. Die Zahl der
Value-Komponenten bestimmt seine Breite.

#### Angeleitete Übung: Drei Verwendungen derselben Eingabe

Der Rechenweg ist jetzt vollständig. Betrachte zur Vertiefung zuerst Rusts
Ownership, nicht die Mathematik:

```text
Vor Query: input verfügbar
Query bekommt einen Clone -> input weiterhin verfügbar
Key bekommt einen Clone   -> input weiterhin verfügbar
Value bekommt input       -> input danach nicht mehr verwendbar
```

Diese Reihenfolge erklärt die Clones vollständig. Value erhält das Original
nicht, weil es mathematisch wichtiger wäre, sondern weil es **im Code zuletzt**
auf die Eingabe zugreift.

**Zum Mitdenken:** Würden wir zuerst Value, dann Key, zuletzt Query berechnen,
wer könnte dann das Original bekommen?
Query. Value und Key bräuchten Clones, damit Query die Eingabe noch erhält.
Die Attention-Mathematik wäre dieselbe. Entscheidend ist die letzte Verwendung,
nicht der Rollenname. Diese Frage betrifft Rust-Ownership, nicht Attention.

#### Angeleitete Übung: Für wen sollen Gewichte zusammen 1 ergeben?

Diese zweite Übung betrifft die Attention-Rechnung. Nach der Score-Berechnung
lesen wir die Achsen so:

```text
scores[Fenster, Query-Position, Key-Position]
       Achse 0   Achse 1        Achse 2
```

Fixiere gedanklich Fenster 0 und Query A. Übrig bleibt eine Liste der Scores
gegen die Quellen A und B:

```text
                    Quelle A   Quelle B
Query A, Scores       0.000      0.707
Query A, Gewichte     0.330      0.670    Summe: 1
```

Wir wollen wissen, wie **diese eine Query** ihren Anteil auf alle Quellen
verteilt. Deshalb normalisiert Softmax entlang der Quelle-Liste, also der
Key-Achse `2`. Für Query B berechnet es unabhängig eine weitere Verteilung.

Über Achse `1` würden wir stattdessen für eine feste Quelle die verschiedenen
Queries normalisieren. Das beantwortet eine andere Frage und liefert nicht
die gewünschten Anteile pro suchender Position. Über Achse `0` würden wir
verschiedene Fenster zur Normalisierung heranziehen.

**Zum Mitdenken:** Warum stand in der ersten Demo `softmax(..., 1)`?
Dort gab es keinen Batch. Die Form war `[Query-Position, Key-Position]`:
Query auf Achse `0`, Key auf Achse `1`. Normalisiert wird in beiden Demos
über **Keys**; nur ihre Achsennummer unterscheidet sich.

### Demo ausführen

#### Feste Projektionsgewichte einsetzen

Wir ersetzen die zufälligen Startwerte durch unsere drei Matrizen:

```rust,ignore
{{#include ../../../crates/model/examples/ch05_projections.rs:fixed}}
```

`Param::from_tensor` verpackt einen Tensor als Modellparameter. Die festen
Zahlen dienen nur der nachvollziehbaren Demo, nicht als sinnvolle Startwerte
für ein Sprachmodell.

Lies die erste Zuweisung von innen nach aussen: `Tensor::from_data` erstellt
aus `QUERY_WEIGHTS` eine Burn-Zahlentabelle auf unserem Gerät.
`Param::from_tensor` kennzeichnet sie als Modellparameter.
`attention.query.weight = ...` ersetzt damit die bisherige Query-Matrix.
`let mut attention` erlaubt diese Änderung am Modellwert; ohne `mut` wäre
die Zuweisung in Rust nicht erlaubt. Key und Value werden genauso ersetzt.

```bash
cargo run -p model --example ch05_projections
```

Ausgabe:

```text
{{#include ../ausgaben/ch05_projections.txt}}
```

Die erste Demo bleibt unverändert: Mit `ch05` kannst du weiter die Grundrechnung
ohne Projektion und Skalierung betrachten.

### Lernbarkeit prüfen

Wir haben jetzt das Modell aufgebaut, seine Berechnung verfolgt und die
Demo betrachtet. Zuletzt prüfen wir, was an diesem Baustein "trainierbar" ist.
Die Tests berechnen Gradienten, verändern aber noch keine Parameter.

Die Integrationstests in `crates/model/tests/attention.rs` verwenden denselben
Baustein auch mit `Device::ndarray().autodiff()`. Wie in Kapitel 4 summieren sie die
Ausgabe und rufen `backward()` auf. Sie prüfen endliche, nicht durchweg null
gesetzte Gradienten für alle drei Projektionsmatrizen und die markierte Eingabe.
Softmax und Matrixmultiplikationen bleiben Teil von Burns Rechengraph.

Weitere Tests vergleichen das Ergebnis mit der Handrechnung, prüfen getrennte
Batches, ein Fenster mit nur einem Token und den Anschluss an das vorhandene
Embedding mit unterschiedlicher Head-Breite.

```bash
cargo test -p model --test attention
```

Die Summe ist wieder nur ein Lernbeispiel, keine geeignete Verlustfunktion für
Sprachvorhersagen. Ein Optimizer verändert noch keine Gewichte. Modellberechnung
bleibt im Modell-Crate; die spätere Trainingslogik gehört in das Training-Crate.

## Die Zusammenhänge wiederfinden

Du darfst die Zeichnungen, Rechnungen und den Code danebenlegen. Versuche,
die folgenden Aussagen dort wiederzufinden, statt Antworten auswendig zu lernen:

- **Attention:** Query und Key bestimmen die Attention-Gewichte. Values liefern
  die Inhalte. Erst beim Mischen treffen beide Wege zusammen.
- **Tensorformen:** Bei der Eingabe unterscheiden wir Fenster, Token-Position
  und Vektorkomponente. Bei den Scores steht die letzte Achse dagegen für
  Key-Positionen. Die Bedeutung hängt vom Rechenschritt ab.
- **Rust:** Clones erhalten die Eingabe für weitere Projektionen. Welche Rolle
  das Original erhält, entscheidet die Reihenfolge der Verwendungen im Code.
- **Softmax:** Jede Query verteilt ihre Anteile auf die Keys ihres Fensters.
  Deshalb normalisieren wir entlang der Key-Achse, unabhängig von deren Nummer.

Wenn ein Zusammenhang noch unklar ist, gehe zu seiner angeleiteten Übung
zurück. Du musst die statistische Skalierungsbegründung und Burns erzeugten
Modulcode weiterhin nicht erklären können. Die Lernziele sind verschiedene
Fähigkeiten: Attention verstehen, Formen lesen und Rust-Code verfolgen.
Eine Unsicherheit bei Ownership bedeutet nicht, dass du Attention nicht
verstanden hast.

## Schritt 3: Kausale Maskierung (Causal Masking)

### Das Problem: Der Blick in die Zukunft

In unseren bisherigen Rechnungen berechnet Token A (`[1, 0]`) seine Attention
auch über Token B (`[0, 1]`). Bei zwei Tokens an den Positionen 0 und 1:

```text
Position 0: "Der"
Position 1: "Hund"
```

Im autoregressiven Sprachmodell (wie GPT) ist die Aufgabe: **Vorhersage des
nächsten Tokens**. Wenn Position 0 ("Der") bereits Informationen von Position 1
("Hund") aufnehmen darf, "sieht" das Modell die Zukunft. Beim echten Erzeugen
von Text existiert Position 1 aber noch gar nicht!

Das Modell würde lernen, auf zukünftige Wörter zu schauen, statt aus dem
bisherigen Kontext das nächste Wort vorherzusagen.

*(Hinweis: Modelle wie **BERT** arbeiten anders: Sie generieren keinen Text,
sondern füllen Lücken im Satz und dürfen deshalb in beide Richtungen schauen.
Den genauen Unterschied beleuchtet der Exkurs
[BERT vs. GPT: Bidirektionale vs. kausale Attention](../exkurse/bert-vs-gpt.md).)*

### Die Lösung: Vor Softmax auf $-\infty$ setzen

Wir müssen verhindern, dass Position $i$ Informationen von Positionen $j > i$
erhält. Können wir die unerlaubten Gewichte nicht einfach nach dem Softmax auf 0
setzen?

Nein: Softmax normalisiert die Zeile so, dass die Summe aller Anteile exakt 1
ergibt. Wenn wir danach Werte auf 0 setzen, stimmt die Summe nicht mehr:

```text
Gewichte vor Nullsetzen:  [0.33, 0.67]  Summe: 1.0
Nach Nullsetzen von B:    [0.33, 0.00]  Summe: nur noch 0.33!
```

Die verbleibenden Quellen müssten wieder auf 100 % hochskaliert werden.
Viel eleganter und mathematisch sauber ist es, das **vor dem Softmax** zu tun:

Erinnerung an Softmax:

```text
Gewicht j = exp(Score j) / Summe aller exp(Scores) dieser Zeile
```

Was macht die Exponentialfunktion `exp(x)` mit verschiedenen Werten?

```text
exp(0)   = 1.0
exp(1)   = ungefähr 2.718
exp(-10) = ungefähr 0.000045
exp(-inf) = 0.0
```

Wenn ein Score vor dem Softmax $-\infty$ (Minus Unendlich) ist, wird sein
Zählerterm exakt `0.0`. Dadurch erhält jede Zukunftsposition **automatisch
genau 0 % Aufmerksamkeit**, und die Summe der erlaubten Positionen bleibt
exakt 1!

### Die Maske als Matrix

Für eine Sequenz der Länge 2 sieht die Erlaubnis so aus:

```text
                 Quelle 0    Quelle 1
Suchender 0:     erlaubt     ZUKUNFT (-inf)
Suchender 1:     erlaubt     erlaubt
```

Für drei Tokens:

```text
                 Quelle 0    Quelle 1    Quelle 2
Suchender 0:     erlaubt     -inf        -inf
Suchender 1:     erlaubt     erlaubt     -inf
Suchender 2:     erlaubt     erlaubt     erlaubt
```

Die erlaubten Einträge bilden ein Dreieck unten links (untere Dreiecksmatrix).
Jede Position $i$ darf nur Quellen $j \le i$ sehen.

### Die Handrechnung mit Maske

Nehmen wir die skalierten Scores aus Schritt 2:

```text
            Quelle 0  Quelle 1
Query 0:      0.000     0.707
Query 1:      0.707     0.707
```

Jetzt wenden wir die Maske an. Score `[0, 1]` (Query 0 gegen Quelle 1) ist
Zukunft und wird durch $-\infty$ ersetzt:

```text
Maskierte Scores:
            Quelle 0  Quelle 1
Query 0:      0.000     -inf
Query 1:      0.707     0.707
```

#### Softmax für Query 0 (Position 0)

```text
Quelle 0: exp(0)    = 1.0
Quelle 1: exp(-inf) = 0.0
Summe               = 1.0 + 0.0 = 1.0

Anteil Quelle 0 = 1.0 / 1.0 = 1.0  (100 %)
Anteil Quelle 1 = 0.0 / 1.0 = 0.0  (0 %)
```

Ohne Maske waren es noch `[0.330, 0.670]`. Durch die Maske konzentriert sich
Position 0 zu 100 % auf sich selbst!

#### Softmax für Query 1 (Position 1)

Position 1 darf beide Quellen sehen. Hier ändert sich nichts:

```text
Anteil Quelle 0 = 0.5 (50 %)
Anteil Quelle 1 = 0.5 (50 %)
```

Die maskierten Attention-Gewichte lauten:

```text
           Anteil Quelle 0   Anteil Quelle 1
Query 0          1.0               0.0
Query 1          0.5               0.5
```

### Neuer Kontext mit Maske

Unsere Value-Vektoren waren:

```text
Value 0 = [2, 0]
Value 1 = [0, 3]
```

**Kontext für Position 0:**

```text
1.0 * [2, 0] + 0.0 * [0, 3] = [2, 0]
```

Position 0 gibt exakt ihren eigenen Value-Vektor weiter. Sie erfährt rein gar
nichts von Position 1.

**Kontext für Position 1:**

```text
0.5 * [2, 0] + 0.5 * [0, 3] = [1, 1.5]
```

Position 1 kombiniert Vergangenheit und Gegenwart zu gleichen Teilen.

### Kausale Maskierung in Rust mit Burn

Wie sieht diese Maskierung im echten Code von `crates/model/src/attention.rs` aus?

```rust,ignore
{{#include ../../../crates/model/src/attention.rs:causal_mask}}
```

Die Zeilen im Detail:

1. **`if context_length > 1`**: Bei einem einzelnen Token (`context_length == 1`)
   gibt es keine Zukunft. Die Maskierung kann übersprungen werden.
2. **`Tensor::<2>::ones([context_length, context_length], ...)`**: Erzeugt
   eine quadratische 2D-Matrix aus Einsen.
3. **`.triu(1)`** (*upper triangular*): Behält nur Werte **oberhalb** der
   Hauptdiagonale (Offset 1) und setzt Diagonale sowie untere Werte auf 0.
   Das sind genau die verbotenen Zukunftspositionen.
4. **`.bool()`**: Wandelt die Matrix in Wahrheitswerte (`true` für Zukunft,
   `false` für Gegenwart und Vergangenheit).
5. **`.unsqueeze()`**: Fügt die Batch-Dimension hinzu, damit die Form `[1, context_length, context_length]`
   zum 3D-Score-Tensor passt (Broadcasting über alle Batches).
6. **`scores.mask_fill(mask, f32::NEG_INFINITY)`**: Ersetzt alle `true`-Stellen
   durch $-\infty$.

### Die Ausgabe der Demo mit Maske

Wenn wir die Demo `ch05_projections` laufen lassen, sehen wir genau das
Ergebnis unserer Handrechnung:

```text
{{#include ../ausgaben/ch05_projections.txt}}
```

Position 0 liefert `[2.000, 0.000]` (nur eigener Value).
Position 1 liefert `[1.000, 1.500]` (Mischung aus Token 0 und Token 1).

### Kausalität mit einem Test beweisen

Um sicherzugehen, dass frühere Tokens nicht durch spätere Tokens beeinflusst
werden, testen wir das Verhalten in `crates/model/tests/attention.rs`:

```rust,ignore
{{#include ../../../crates/model/tests/attention.rs:test_causal}}
```

Die Idee des Tests:
Wir übergeben zwei Sequenzen an `forward`.
- Sequenz A hat 2 Tokens.
- Sequenz B hat dieselben 2 Tokens und zusätzlich ein drittes Token mit extremen
  Zahlenwerten (`[99.0, -99.0]`).

Dank der kausalen Maskierung müssen die Ausgaben für Token 0 und Token 1 in beiden
Fällen **exakt identisch** sein.

```bash
cargo test -p model --test attention future_tokens_do_not_influence_earlier_outputs
```

```bash
cargo test -p model --test attention future_tokens_do_not_influence_earlier_outputs
```

## Schritt 4: Multi-Head Attention

Bisher hat unser Modell einen einzigen Attention-Kopf (`SelfAttention`).
Dieser Kopf kann pro Token nur **einen einzigen Fokus** gleichzeitig legen.

### Das Problem: Ein einziger Fokus reicht nicht

Betrachte den Satz:
> „Die Katze fraß die Maus, weil **sie** hungrig war.“

Um das Wort **„sie“** richtig in seinen Kontext einzubetten, muss ein
Sprachmodell zwei Fragen gleichzeitig beantworten:

1. **Grammatik (Syntax):** Welches Substantiv passt zum Pronomen „sie“?
   $\to$ „Katze“ oder „Maus“.
2. **Bedeutung / Kausalität:** Wer war hungrig und fraß wen?
   $\to$ „Katze“.

Ein einzelner Attention-Kopf berechnet pro Query genau eine Zeile
Attention-Gewichte. Er muss sich entscheiden: Schaut er auf das nächste
Nomen, das Prädikat oder das Subjekt?

Mit **Multi-Head Attention** (mehreren Köpfen) lösen wir dieses Problem:
- **Kopf 1** spezialisiert sich auf grammatische Bezüge.
- **Kopf 2** spezialisiert sich auf kausale Zusammenhänge.

Beide Köpfe betrachten denselben Satz parallel, aber mit unterschiedlichen
„Brillen“.

### Das Prinzip: Aufteilen, parallel rechnen, zusammenführen

1. **Aufteilen:** Wir teilen die Modell-Dimension $d_{model}$ (z. B. 4)
   auf $h$ Köpfe auf (z. B. 2 Köpfe à Head-Breite 2). Der Rechenaufwand bleibt
   damit fast genau gleich wie bei einem einzigen großen Kopf.
2. **Parallele Attention:** Jeder Kopf rechnet unabhängig seine eigene
   kausale Attention (mit eigenen $W_Q, W_K, W_V$-Matrizen).
3. **Zusammenkleben (Concat):** Die Teilergebnisse beider Köpfe werden
   hintereinandergehängt. Aus zweimal Breite 2 wird wieder die volle Breite 4.
4. **Mischen (Output-Projektion $W_O$):** Eine abschließende lineare Schicht
   vermischt die Erkenntnisse aller Köpfe miteinander.

```text
                  Eingabe [Batch, Tokens, 4]
                 /                          \
          Kopf 1 (Breite 2)          Kopf 2 (Breite 2)
                 \                          /
                  Zusammenkleben (Concat)
                             ↓
                    Breite 4 (nebeneinander)
                             ↓
                  Output-Projektion W_O
                             ↓
                  Ausgabe [Batch, Tokens, 4]
```

### Die Handrechnung für 2 Köpfe

Betrachten wir zwei Tokens (Position 0 und 1) mit Einbettungsbreite 4:

#### 1. Unabhängige Köpfe rechnen

Jeder Kopf erzeugt für jedes Token einen Kontext-Vektor der Head-Breite 2:

**Kopf 1** liefert:
- Token 0: `[2.0, 0.0]`
- Token 1: `[1.0, 1.5]`

**Kopf 2** liefert:
- Token 0: `[0.0, 1.0]`
- Token 1: `[2.0, 0.5]`

#### 2. Zusammenkleben (Concat)

Wir hängen die Vektoren pro Token einfach nebeneinander:

```text
Token 0:  [2.0, 0.0]  +  [0.0, 1.0]  ->  [2.0, 0.0,  0.0, 1.0]
          \________/     \________/      \________/  \________/
            Kopf 1         Kopf 2          Kopf 1      Kopf 2

Token 1:  [1.0, 1.5]  +  [2.0, 0.5]  ->  [1.0, 1.5,  2.0, 0.5]
```

Form vorher je Kopf: `[1, 2, 2]`.
Form nach Concat: `[1, 2, 4]`. Die Dimension 4 ist wieder hergestellt.

#### 3. Warum Concat allein nicht reicht: Die Output-Projektion $W_O$

Nach dem Concat liegen die Zahlen nur nebeneinander im selben Array:
- Platz 1 und 2 gehören stur zu Kopf 1.
- Platz 3 und 4 gehören stur zu Kopf 2.

Die Köpfe haben noch nicht miteinander gesprochen! Wenn die nächste Schicht
des Modells eine Entscheidung treffen soll, die beide Köpfe gleichzeitig
braucht (z. B. Subjekt aus Kopf 1 und Verneinung aus Kopf 2 kombinieren),
fehlt die Verbindung.

Die lineare Schicht $W_O$ ist eine $4 \times 4$ Matrix. Sie wirkt wie ein
**Chefredakteur**: Sie liest die Berichte beider Experten, gewichtet deren
Aussagen und erzeugt die finale Zusammenfassung für die nächste Schicht.

Beispielrechnung für Token 0 mit einer festen $W_O$-Matrix:

```text
Eingabe: [2.0, 0.0, 0.0, 1.0]

         Spalte 1    Spalte 2    Spalte 3    Spalte 4
Kopf 1-A:    1           0           1           0
Kopf 1-B:    0           1           0           1
Kopf 2-A:    1           0          -1           0
Kopf 2-B:    0           1           0          -1
```

Für Ausgabekomponente 1:
```text
  2.0 * 1  (Kopf 1-A)
+ 0.0 * 0  (Kopf 1-B)
+ 0.0 * 1  (Kopf 2-A)
+ 1.0 * 0  (Kopf 2-B)
= 2.0
```

Für Ausgabekomponente 2:
```text
  2.0 * 0 + 0.0 * 1 + 0.0 * 0 + 1.0 * 1 = 1.0
```

Das Ergebnis für Token 0 ist `[2.0, 1.0, 2.0, -1.0]`. Es hat wieder
Länge 4, enthält aber jetzt Informationen aus beiden Köpfen in
gemeinsamer Repräsentation.

### Multi-Head Attention in Burn: Zwei Wege

Im Code bieten wir beide Varianten an:
1. **`NaiveMultiHeadAttention` (didaktisch):** Eine Liste von `SelfAttention`-Köpfen,
   die nacheinander rechnen, mit `Tensor::cat` verkettet und per $W_O$ gemischt werden.
   Deckungsgleich mit der Handrechnung.
2. **`MultiHeadAttention` (produktiv & optimiert):** Eine einzige gemeinsame
   Projektionsmatrix, parallele 4D-Tensoroperationen über alle Köpfe und
   Batch-Dimensionen hinweg. Das ist der Standard für GPU-Training.

Beide berechnen mathematisch exakt dasselbe Ergebnis!

#### 1. Der didaktische Weg (`NaiveMultiHeadAttention`)

Hier entspricht jede Zeile Code unserer Handrechnung:

```rust,ignore
{{#include ../../../crates/model/src/attention.rs:naive_mha}}
```

Die Schritte in `forward`:
- Jeder Kopf in `self.heads` rechnet unabhängig seine eigene Attention:
  `head.forward(input.clone())`.
- `Tensor::cat(head_outputs, 2)` hängt die Ausgaben entlang der Merkmalsachse
  aneinander (`[1, 2, 2] + [1, 2, 2] -> [1, 2, 4]`).
- `self.output_projection.forward(...)` wendet die $W_O$-Matrix an.

#### 2. Der produktive Weg (`MultiHeadAttention` mit 4D-Reshape)

Warum optimieren? Eine Schleife über Köpfe führt viele kleine Kernel-Aufrufe
auf der GPU aus. Viel schneller ist es, **alle Köpfe in einem einzigen großen
Tensor parallel** zu berechnen.

Die Struktur verwendet gemeinsame Projektionen für die volle Breite:

```rust,ignore
{{#include ../../../crates/model/src/attention.rs:mha_config}}
```

Die Vorwärtsrechnung formt die Tensoren geschickt um:

```rust,ignore
{{#include ../../../crates/model/src/attention.rs:mha_forward}}
```

Wie funktioniert der 4D-Trick?

1. **Gemeinsam projizieren:** `query`, `key` und `value` projizieren von
   `[Batch, Tokens, EmbeddingDim]` direkt in `[Batch, Tokens, EmbeddingDim]`.
   Die Kanäle aller Köpfe liegen bereits nebeneinander.
2. **Form aufteilen (`project_and_split`):**
   - `.reshape([Batch, Tokens, Heads, HeadDim])`: Trennt die flache
     Merkmalsachse in Köpfe und Kopf-Breite auf.
   - `.swap_dims(1, 2)`: Tauscht Achse 1 und 2 $\to$
     `[Batch, Heads, Tokens, HeadDim]`.
   - Jetzt verhält sich jede Kopf-Ebene wie ein eigener Batch!
3. **Paralleles Matmul:**
   - `q.matmul(k.swap_dims(2, 3))` berechnet alle Scores für alle Köpfe
     und alle Batches in einer einzigen GPU-Operation.
   - Form: `[Batch, Heads, Tokens, Tokens]`.
4. **Kausale Maske:** Die 2D-Maske `[Tokens, Tokens]` wird per `.unsqueeze()`
   auf `[1, 1, Tokens, Tokens]` erweitert. Burn broadcastet sie automatisch
   über alle Batches und Köpfe.
5. **Zurückformen:**
   - Nach `weights.matmul(v)` haben wir `[Batch, Heads, Tokens, HeadDim]`.
   - `.swap_dims(1, 2)` $\to$ `[Batch, Tokens, Heads, HeadDim]`.
   - `.reshape([Batch, Tokens, EmbeddingDim])` klebt die Köpfe wieder zusammen.
6. **Output-Projektion:** `self.output_projection.forward(merged)`.

### Der Beweis: Beide Wege sind numerisch identisch

In `crates/model/tests/attention.rs` verifizieren wir, dass die optimierte
4D-Variante und die didaktische Einzelschritt-Variante auf dieselben Zahlen
führen:

```rust,ignore
{{#include ../../../crates/model/tests/attention.rs:test_mha_equivalence}}
```

```bash
cargo test -p model --test attention naive_and_batched_multi_head_attention_are_numerically_equivalent
```

### Die Ausgabe der Demo

Das Beispiel `crates/model/examples/ch05_multi_head.rs` führt beide
Varianten aus:

```bash
cargo run -p model --example ch05_multi_head
```

```text
{{#include ../ausgaben/ch05_multi_head.txt}}
```

Beide liefern Tensoren der Form `[1, 2, 4]`. Für das weitere Training und
die Transformer-Blöcke in Kapitel 6 nutzen wir die produktive `MultiHeadAttention`.

## Zusammenfassung von Kapitel 5

In diesem Kapitel haben wir das Herzstück moderner Sprachmodelle schrittweise
aufgebaut:

1. **Einfache Attention:** Vergleichen (Scores via Skalarprodukt), Gewichten
   (Softmax) und Mischen (gewichtete Summe).
2. **Projektionen & Skalierung:** Eigene Rollen durch $W_Q, W_K, W_V$ und
   stabilisierende Division durch $\sqrt{d_k}$.
3. **Kausale Maskierung:** Verhindert den Blick in die Zukunft durch
   $-\infty$-Scores vor dem Softmax.
4. **Multi-Head Attention:** Parallele Köpfe mit Teilbreiten, zusammengeführt
   über Concat und eine Output-Projektion $W_O$.

Damit ist das Attention-Modul vollständig. Im nächsten Kapitel setzen wir die
einzelnen Schichten (Embedding, Multi-Head Attention, Feedforward-Netz und
Layer-Norm) zu einem vollständigen **Transformer-Block** zusammen.
