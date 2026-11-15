# 16. Temperatur und Sampling

## Lernziel

Kapitel 11 wählt mit `argmax` immer das wahrscheinlichste nächste Token.
Jetzt ziehen wir ein Token aus einer Wahrscheinlichkeitsverteilung.
Temperatur, Top-k und Top-p steuern diese Auswahl. Ein Zufallsseed macht
die Zufallsfolge wiederholbar; ein Stopp-Token und eine maximale Anzahl
neuer Tokens begrenzen die Generierung.

**Wir ändern keine Modellgewichte.** Sampling gehört zur Inferenz, nicht
zum Training. Es verändert Auswahl und Vielfalt, nicht die gelernte
Modellqualität. Kapitel 15 behandelt dagegen die Wiederaufnahme des
Trainings; dessen Optimizer-Zustand brauchen wir hier nicht.

### Was bedeutet Inferenz?

**Inferenz bedeutet, ein Modell anzuwenden, um Vorhersagen zu berechnen,
ohne dabei seine Gewichte zu verändern.** Training lernt die Gewichte;
Inferenz verwendet die vorhandenen Gewichte.

Bei unserem GPT läuft die Textgenerierung so ab:

1. Den Starttext in Token-IDs umwandeln.
2. Mit `forward` die Logits für das nächste Token berechnen; dafür
   verwenden wir die letzte Kontextposition.
3. Mit `argmax` oder Sampling ein Token auswählen.
4. Das Token anhängen und den Vorgang mit dem erweiterten Kontext wiederholen.

Der Unterschied zum Training:

```text
Training:
Vorhersage → mit Ziel vergleichen → Loss und Gradienten berechnen
           → Gewichte anpassen

Inferenz:
Vorhandene Gewichte → Vorhersage berechnen
                   → für die Textgenerierung ein Token auswählen
```

Temperatur und Sampling steuern die Auswahl, nicht das Lernen.
Inferenz ist dabei nicht auf Textgenerierung beschränkt: Auch das Anwenden
eines Modells zur Klassifikation eines Textes ist Inferenz.

Selbst ein zufällig initialisiertes Modell kann Vorhersagen berechnen
und damit Inferenz ausführen. Der Begriff sagt nichts darüber aus,
ob diese Vorhersagen bereits brauchbar sind.

## 1. Logits sind noch keine Wahrscheinlichkeiten

Wie in Kapitel 11 nehmen wir nur die Logits der letzten Kontextposition.
Als nachvollziehbares Beispiel verwenden wir fünf feste Werte:

```text
Token-ID:       0     1     2     3     4
Logit:         3     2     1     0    -1
```

Ein Logit darf negativ sein. Wahrscheinlichkeiten sind dagegen
nichtnegativ und ergeben zusammen eins. Softmax wandelt die Logits um:
Zuerst exponentieren wir die Werte, danach teilen wir durch ihre Summe.
Das ergibt hier ungefähr `[0.6364, 0.2341, 0.0861, 0.0317, 0.0117]`.

`argmax` wählt immer Token 0. Beim **Sampling** hat Token 0 etwa 63,6 Prozent
Chance, Token 1 etwa 23,4 Prozent. Höchster Wert bedeutet also nicht,
dass dieses Token bei jeder Ziehung gewinnt.

Ein einzelnes Ergebnis beweist diese Verteilung nicht. Erst viele
unabhängige Ziehungen aus denselben festen Logits nähern sich den
erwarteten Anteilen. Bei Textgenerierung ändern sich die Logits nach
jedem angehängten Token: Der Kontext ist dann ein anderer.

## 2. Temperatur verändert die Verteilung

Wir teilen die Logits vor Softmax durch eine positive **Temperatur**:

```text
Temperatur < 1: Unterschiede stärker → Verteilung konzentrierter
Temperatur = 1: gewöhnliche Softmax-Verteilung
Temperatur > 1: Unterschiede schwächer → Verteilung flacher
```

Die Rangfolge bleibt gleich. Eine hohe Temperatur macht unwahrscheinliche
Tokens aber wahrscheinlicher; das kann Vielfalt und Unsinn erhöhen.
Eine niedrige Temperatur ist nicht automatisch besser und verhindert
keine Halluzinationen.

Für unsere festen Logits `[3.0, 2.0, 1.0, 0.0, -1.0]` sehen wir den
Unterschied direkt:

```text
Token-ID:          0       1       2       3       4
Temperatur 0.5: 0.8647  0.1170  0.0158  0.0021  0.0003
Temperatur 1.0: 0.6364  0.2341  0.0861  0.0317  0.0117
Temperatur 1.5: 0.5046  0.2591  0.1330  0.0683  0.0351
```

Token 0 bleibt bei allen drei Temperaturen Favorit. Bei Temperatur 0.5
liegt seine Auswahlchance aber bei etwa 86 Prozent, bei Temperatur 1.5
nur noch bei etwa 50 Prozent. Die Werte sind gerundet.

Für eine numerisch stabile Berechnung ziehen wir vor Division und
Exponentiation den grössten Logit ab. Dieser wird dadurch null:
Sein exponentierter Wert ist eins, alle übrigen liegen zwischen null
und eins. So vermeiden wir überlaufende Exponentialwerte. Die
Wahrscheinlichkeiten verändern sich durch die gemeinsame Verschiebung
nicht.

```rust
{{#include ../../../crates/training/src/sampling.rs:softmax}}
```

### Was macht `normalize(&mut weights)`?

**`normalize` macht aus den exponentierten Gewichten eine
Wahrscheinlichkeitsverteilung.** Erst nach diesem Schritt können wir
die Werte als Wahrscheinlichkeiten für die einzelnen Token interpretieren.
**Gewicht** meint hier den unnormalisierten Softmax-Wert, nicht einen
Modellparameter.
Dazu teilt die Funktion jeden Wert durch die Summe aller Werte:

```text
Logits                 beliebige Zahlen, auch negativ
    ↓ exponentieren    nach Verschiebung und Temperaturskalierung
Softmax-Gewichte        nichtnegative Werte, Summe noch nicht unbedingt eins
    ↓ normalize
Wahrscheinlichkeiten   Werte zwischen null und eins, Summe eins
```

Exponentiation und Normalisierung bilden zusammen die Softmax-Berechnung.
Rohe Logits direkt zu normalisieren reicht nicht: Sie können negativ
sein. `normalize` erzeugt nur dann eine Wahrscheinlichkeitsverteilung,
wenn die Eingabewerte nichtnegativ sind und ihre Summe positiv ist.
Unsere exponentierten Gewichte erfüllen diese Bedingungen.

```rust
fn normalize(weights: &mut [f64]) {
    let sum: f64 = weights.iter().sum();
    for weight in weights {
        *weight /= sum;
    }
}
```

`weights.iter().sum()` berechnet zuerst die Summe, bevor die Schleife
die Werte verändert. Ein Zahlenbeispiel:

```text
Vorher:  [1.0, 0.5, 0.5]
Summe:   2.0

Nachher: [1.0 / 2.0, 0.5 / 2.0, 0.5 / 2.0]
       = [0.5, 0.25, 0.25]
Summe:   1.0
```

Die **Verhältnisse bleiben erhalten**: Der erste Wert ist weiterhin
doppelt so gross wie jeder der beiden anderen. Die normalisierten
Werte sind nichtnegativ und ergeben zusammen eins, abgesehen von
kleinen Rundungsabweichungen.

`&mut weights` leiht den Vektor veränderbar aus. Die Funktion nimmt
einen veränderbaren Slice `&mut [f64]` entgegen, also eine Sicht auf
die vorhandenen Werte. In der Schleife ist `weight` eine veränderbare
Referenz auf einen einzelnen Wert. `*weight /= sum` bedeutet:
Den Wert lesen, durch `sum` teilen und das Ergebnis an dieselbe Stelle
schreiben. Es entsteht kein neuer Vektor.

Hier ist die Summe positiv, weil vor der Softmax-Normalisierung
mindestens ein Gewicht eins ist. Nach Top-k oder Top-p normalisieren
wir erneut: Entfernte Kandidaten haben dann Gewicht null, mindestens
ein Kandidat mit positivem Gewicht bleibt erhalten.

### Datentypen und numerischer Unterlauf

Nachdem die Normalisierung klar ist, betrachten wir die numerischen
Details. Die Berechnung verwendet `f64`, obwohl die Modell-Logits `f32`
sind. Das erkennen wir an `logits: &[f32]` in der Funktionssignatur und an
dieser Zeile:

```rust
((f64::from(logit) - f64::from(maximum)) / temperature).exp()
```

`f64::from(...)` wandelt die beiden `f32`-Werte vor der Subtraktion in
`f64` um. Auch `temperature` ist vom Typ `f64`; Division und
Exponentiation verwenden deshalb ebenfalls `f64`.

Für den grössten Logit ist die Differenz zum Maximum null. Bei jeder
gültigen, positiven Temperatur bleibt sein exponentierter Wert eins:

```text
logit - maximum = 0
0 / temperature = 0
exp(0) = 1
```

Kleinere Logits ergeben dagegen negative Differenzen. Eine sehr kleine
Temperatur macht den Exponenten stark negativ:

```text
Logit 2, Maximum 3, Temperatur 0.001:
(2 - 3) / 0.001 = -1000
exp(-1000) → als f64 numerisch 0
```

Mathematisch ist dieser Wert positiv, aber zu klein, um als `f64`
dargestellt zu werden. Das heisst **numerischer Unterlauf**. Sehr kleine
Gewichte können dadurch auf null fallen; mindestens ein Gewicht bleibt
eins. Die anschliessende Normalisierung hat deshalb weiterhin eine
positive Summe als Divisor. Nach dem Normalisieren ist auch die
Wahrscheinlichkeit des grössten Logits normalerweise kleiner als eins.

**Temperatur 0 ist hier ungültig.** Für deterministische Auswahl verwenden
wir ausdrücklich `SamplingStrategy::Greedy`. Auch negative, unendliche
oder `NaN`-Temperaturen führen zu einem Fehler.

## 3. Top-k: eine feste Anzahl Kandidaten

Top-k behält nur die `k` wahrscheinlichsten Tokens. Alle anderen erhalten
Wahrscheinlichkeit null. Anschliessend normalisieren wir die verbleibenden
Werte erneut, damit ihre Summe wieder eins ist.

**Top-k 3 bedeutet: Nur die drei wahrscheinlichsten Tokens dürfen
ausgewählt werden.** Es bedeutet nicht, dass wir drei Tokens erzeugen.
Pro Generierungsschritt wählen wir weiterhin genau ein nächstes Token.

Unsere fünf Tokens haben zunächst diese Wahrscheinlichkeiten:

```text
Token-ID:        0       1       2       3       4
Vorher:      0.6364  0.2341  0.0861  0.0317  0.0117
```

### Schritt 1: Die drei grössten Werte behalten

Token 0, 1 und 2 sind die wahrscheinlichsten Kandidaten.
Token 3 und 4 schliessen wir aus, indem wir ihre Werte auf null setzen:

```text
Token-ID:        0       1       2       3       4
Gefiltert:   0.6364  0.2341  0.0861  0       0
```

### Schritt 2: Erneut normalisieren

Die verbliebenen Werte ergeben zusammen ungefähr `0.9566`, nicht mehr
eins. `normalize` teilt deshalb jeden Wert durch diese Summe:

```text
Token 0: 0.6364 / 0.9566 ≈ 0.6652
Token 1: 0.2341 / 0.9566 ≈ 0.2447
Token 2: 0.0861 / 0.9566 ≈ 0.0900
Token 3 und 4: weiterhin 0
```

Die Zahlen sind gerundet; der Code rechnet mit den ungerundeten Werten.
Die ausgeschlossenen Tokens geben ihre Wahrscheinlichkeit gewissermassen
an die verbleibenden Kandidaten ab. Deren relative Verhältnisse bleiben
durch die gemeinsame Division erhalten.

### Schritt 3: Genau ein Token zufällig ziehen

Aus der neuen Wahrscheinlichkeitsverteilung wählen wir ein Token:

```text
Token 0: etwa 66.5 % Chance
Token 1: etwa 24.5 % Chance
Token 2: etwa  9.0 % Chance
Token 3 und 4: ausgeschlossen
```

Token 0 bleibt Favorit, gewinnt aber nicht zwingend. Deshalb ist Top-k
nicht selbst `argmax`: Es begrenzt die Kandidaten, bevor wir zufällig
ziehen. Nach dem Anhängen des ausgewählten Tokens berechnet das Modell
neue Logits. Top-k wird für den nächsten Schritt erneut angewendet;
die drei Kandidaten können dann andere Token-IDs sein.

`k = 1` lässt nur einen Kandidaten übrig.
`k` muss zwischen eins und der Vokabulargrösse liegen.
Bei gleich wahrscheinlichen Kandidaten bevorzugt unsere Sortierung die
kleinere Token-ID, um genau `k` Kandidaten reproduzierbar auszuwählen.

### Wie wählen wir `k`?

**`k` ist eine Anzahl, keine Rate oder Prozentangabe.** `k = 3` erlaubt
drei Kandidaten, nicht drei Prozent des Vokabulars. Wir legen diese
Anzahl als Einstellung der Generierung fest; das Modell lernt oder
berechnet keinen optimalen `k`-Wert.

Es gibt keine allgemein beste Wahl:

- **Kleines `k`:** weniger Auswahlmöglichkeiten. `k = 1` ist bei
  eindeutiger höchster Wahrscheinlichkeit dieselbe Auswahl wie `argmax`.
  Kleine Werte können Wiederholungen begünstigen und sinnvolle
  Alternativen ausschliessen.
- **Grösseres `k`:** mehr Auswahlmöglichkeiten, darunter auch
  unwahrscheinliche Tokens. Ob diese tatsächlich oft gezogen werden,
  hängt von ihren Wahrscheinlichkeiten und der Temperatur ab.
  Mehr Kandidaten garantieren weder mehr Vielfalt noch bessere Texte.
- **`k` gleich Vokabulargrösse:** kein Token wird ausgeschlossen;
  die Auswahl entspricht Sampling ohne Top-k-Filter.

Für unser Fünf-Token-Beispiel vergleichen wir zunächst `k = 1`, `3`
und `5`. So sehen wir deterministische Auswahl, begrenzte Auswahl
und das gesamte Vokabular. **Top-k 3 in der Demo ist eine didaktische
Wahl, keine Empfehlung für ein brauchbares Sprachmodell.**

Bei einem ausreichend trainierten Modell mit grossem Vokabular können
beispielsweise `k = 10`, `20` und `50` als erste Vergleichswerte dienen.
Das sind Versuchswerte, kein Qualitätsversprechen. Gleichen Checkpoint,
gleiche Prompts, gleiche Temperatur und dieselben mehreren Seeds verwenden;
nur `k` verändern. Dann auf Wiederholungen, Verständlichkeit und passende
Fortsetzungen achten. Ein einzelner gelungener Text reicht nicht, um
einen Wert zu empfehlen.

Für faktische Aufgaben zusätzlich die Richtigkeit prüfen: Ein kleineres
`k` verhindert keine Halluzinationen. Bei unserem schwach trainierten
Mini-Modell kann keine Einstellung fehlendes Sprachverständnis ersetzen.

Die feste Kandidatenzahl ist auch eine Grenze von Top-k: Manchmal gibt
es nur wenige plausible Fortsetzungen, manchmal viele. Top-p im nächsten
Abschnitt passt die Anzahl anhand der aktuellen Verteilung an.

## 4. Top-p: eine variable Anzahl Kandidaten

Top-p, auch **Nucleus Sampling**, sortiert die Tokens nach absteigender
Wahrscheinlichkeit. Wir behalten das kleinste Präfix, dessen Summe
mindestens `p` erreicht. Das Token, das die Schwelle erreicht oder
überschreitet, gehört noch dazu.

Bei `p = 0.9` im Beispiel:

```text
Token 0:                    Summe 0.6364 → noch nicht genug
Token 0, 1:                 Summe 0.8705 → noch nicht genug
Token 0, 1, 2:              Summe 0.9566 → diese drei behalten
```

Nach erneutem Normalisieren ergibt sich hier dieselbe Verteilung wie
bei Top-k 3. Das ist kein allgemeiner Zusammenhang: Bei konzentrierten
Logits kann Top-p einen einzigen Kandidaten behalten, bei flachen
Logits sehr viele.

Top-p wird **nach Temperatur und Softmax** berechnet. Temperatur kann
deshalb auch die Anzahl der Top-p-Kandidaten verändern. `p` muss in
`(0, 1]` liegen; `p = 1` lässt die Verteilung unverändert.

```rust
{{#include ../../../crates/training/src/sampling.rs:filter}}
```

## 5. Die Auswahl im Rust-Code

Das Lernziel ist der Weg von Modell-Logits zum nächsten Token.
Wir erklären zuerst diesen Kernablauf, danach die Eingabeprüfungen,
die ihn absichern:

```text
Strategie festlegen
  → Wahrscheinlichkeiten berechnen
  → ein Token auswählen
  → Token anhängen und mit neuem Kontext wiederholen
```

### 1. Die Auswahlstrategie festlegen

Unsere Varianten sind ausdrücklich getrennt:

```rust
{{#include ../../../crates/training/src/sampling.rs:strategy}}
```

`Greedy` bedeutet deterministische Auswahl des höchsten Logits.
`Multinomial` bezeichnet die gewichtete Zufallsauswahl eines Tokens.
`SamplingFilter::None` verwendet das gesamte Vokabular.
In diesem Kapitel wählen wir entweder Top-k oder Top-p; eine Kombination
beider Filter ist nicht Teil dieser API.

### 2. Mit `probabilities` die Auswahlverteilung berechnen

`probabilities` ist eine Methode unserer `SamplingStrategy` in
`crates/training/src/sampling.rs`. Sie führt die bereits erklärten
mathematischen Schritte zusammen:

```rust
let strategy = SamplingStrategy::Multinomial {
    temperature: 1.0,
    filter: SamplingFilter::TopK(3),
};
let logits = [3.0, 2.0, 1.0, 0.0, -1.0];
let probabilities = strategy.probabilities(&logits)?;
```

Nach erfolgreicher Prüfung berechnet sie bei Sampling Temperatur und
Softmax, filtert gegebenenfalls mit Top-k oder Top-p und normalisiert
die verbliebenen Werte erneut. Das Ergebnis ist ein `Vec<f64>`:

```text
Token-ID:          0       1       2       3       4
Wahrscheinlichkeit: 0.6652  0.2447  0.0900  0       0
```

Die Werte sind gerundet. Die Indizes bleiben Token-IDs: Der Wert an
Index 1 gehört weiterhin zu Token-ID 1, auch wenn beim Filtern eine
separate ID-Liste nach Wahrscheinlichkeit sortiert wurde.

Bei `Greedy` erzeugt die Methode stattdessen eine Auswahlverteilung
mit `1.0` für den höchsten Logit und `0.0` für alle anderen.
**Bis hier ist noch kein Token ausgewählt.** Wir haben nur festgelegt,
welche Tokens mit welcher Wahrscheinlichkeit zur Auswahl stehen.

### 3. Mit `select` das nächste Token auswählen

**Die Generierung braucht eine Token-ID, nicht nur Wahrscheinlichkeiten.**
Diese Auswahl übernimmt die Methode `select` unserer `SamplingStrategy`
in `crates/training/src/sampling.rs`:

```rust
use rand::{SeedableRng, rngs::StdRng};

let strategy = SamplingStrategy::Multinomial {
    temperature: 1.0,
    filter: SamplingFilter::TopK(3),
};
let logits = [3.0, 2.0, 1.0, 0.0, -1.0];
let mut rng = StdRng::seed_from_u64(42);
let next_id = strategy.select(&logits, &mut rng)?;
```

`select` bekommt die Logits und einen veränderbaren Zufallsgenerator.
Sie liefert bei Erfolg genau eine Token-ID vom Typ `i32`.
Diese ID wird anschliessend an den Kontext angehängt. Das `?` gibt
einen möglichen Fehler an die aufrufende Funktion weiter.

`select` verbindet Vorbereitung und Auswahl. Deshalb reicht in der
Generierung der einzelne Aufruf `strategy.select(...)`:

1. **Auswahlverteilung vorbereiten.** Intern ruft sie `probabilities`
   auf, einschliesslich der Eingabeprüfungen.
2. **Ein Token ziehen.** `WeightedIndex` aus `rand` verwendet diese
   Wahrscheinlichkeiten und den Zufallsgenerator zur Auswahl.

```text
select
  → probabilities: [0.6652, 0.2447, 0.0900, 0.0000, 0.0000]
  → WeightedIndex: genau einen Index ziehen, etwa 1
  → Rückgabe: Token-ID 1
```

Die gezogene ID `1` ist hier nur ein mögliches Beispiel, keine Aussage
über das konkrete Ergebnis mit Seed 42. Token 0 ist wahrscheinlicher,
Token 1 und 2 können aber ebenfalls gezogen werden.

Weil der Vektor in Token-ID-Reihenfolge bleibt, ist der von
`WeightedIndex` gezogene Index direkt die gesuchte Token-ID.

Bei `Greedy` gibt `select` dagegen deterministisch die ID mit dem
höchsten Logit zurück; dafür wird keine Zufallszahl gezogen.
Die vorbereitete Auswahlverteilung lässt dabei nur einen Kandidaten zu.

**Kurz:** `probabilities` bereitet die Auswahlverteilung vor.
`select` verwendet sie und liefert das nächste Token.

### 4. Die Auswahl in der Generierung wiederholen

Die bestehende Funktion `generate` bleibt unverändert greedy.
Für Sampling verwenden wir zusätzlich `generate_with_options`:

```rust
{{#include ../../../crates/training/src/generation.rs:sampling_options}}
```

```rust
{{#include ../../../crates/training/src/generation.rs:sampled_generation}}
```

`validate_generation` prüft zuerst Startkontext, Kontextlänge, Token-IDs
und Strategie gegen das Modell. Erst danach startet die Schleife.
Pro Schritt berechnet `context_values` die Logits der letzten
Kontextposition; `select` bereitet daraus die Verteilung vor und wählt
die nächste ID. `push` hängt diese an die Folge an.

Der Zufallsgenerator wird **einmal pro Generierungsaufruf** initialisiert,
nicht bei jedem Token erneut. Die neuen Tokens erweitern wie bisher
den Kontext, und `context_window` begrenzt jeden Modellaufruf.

Gleiche Gewichte, gleicher Prompt, gleiche Optionen und gleicher Seed
liefern in derselben deterministischen Software-/Backend-Umgebung
dieselbe Tokenfolge. Ein anderer Seed kann eine andere Folge ergeben,
muss es aber nicht: Top-k 1 lässt etwa keine alternative Auswahl zu.
Verschiedene Strategien müssen selbst bei gleichem Seed nicht dieselbe
Fortsetzung liefern. Über beliebige `rand`-/Burn-Versionen oder Hardware
hinweg versprechen wir keine identischen Ergebnisse.

### 5. Den Ablauf durch Eingabeprüfungen absichern

Die Prüfungen gehören zur Implementierung, wählen aber selbst kein
Token. **Zur Laufzeit finden sie vor den jeweiligen Berechnungen statt**,
auch wenn wir sie hier erst nach dem Kernablauf erklären.
Die Methode `validate` unserer `SamplingStrategy` prüft die Einstellungen
gegen die Vokabulargrösse:

- Das Vokabular darf nicht leer sein.
- Bei Sampling muss die Temperatur endlich und grösser als null sein.
- Top-k muss zwischen eins und der Vokabulargrösse liegen.
- Top-p muss endlich sein und in `(0, 1]` liegen.

Danach prüft `validate_logits`, ob alle Logits endlich sind und die
Token-IDs im verwendeten Typ `i32` darstellbar bleiben.
Ungültige Eingaben führen zu `SamplingError`, nicht zu Ersatzwerten.

Die Methode `probabilities` ruft beide Prüfungen vor ihrer Berechnung
auf; wir müssen sie beim normalen Aufruf nicht selbst wiederholen.
`validate_generation` sichert zusätzlich den Startkontext und die
Modellgrenzen ab, bevor die Generierungsschleife beginnt.

## 6. Wann stoppen wir?

`max_new_tokens` begrenzt nur die Fortsetzung, nicht den Starttext.
Bei null neuen Tokens bleibt der Starttext unverändert.

`stop_token` ist optional. Sobald dieses Token **neu erzeugt** wird,
hängen wir es an die Ausgabe an und beenden die Schleife. Ein bereits
im Prompt enthaltenes Stopp-Token beendet die Generierung nicht.
Ein Stoppwort aus mehreren Tokens oder mehrere Stopp-IDs sind hier
nicht umgesetzt.

Der GPT-2-Tokenizer kennt `<|endoftext|>` als ID 50'256. Unsere Demo
verwendet sie als Stopp-Token. Das kleine selbst trainierte Modell hat
aber nicht zwangsläufig gelernt, sie sinnvoll zu erzeugen. Deshalb
bleibt die maximale Tokenzahl nötig. Sie ist kein hartes Zeitlimit:
Ein gerade laufender Modellaufruf wird nicht abgebrochen.

## 7. Demo ohne Training

Zuerst die festen Logits vergleichen; der Aufruf lädt kein Modell und
trainiert nichts:

```bash
cargo run -p training --example ch16
```

Die Demo macht den Temperaturvergleich aus Abschnitt 2 ausführbar und
ergänzt Greedy, Top-k und Top-p. Die tatsächlich ausgeführte Demo zeigt:

```text
Feste Logits für Token-IDs 0 bis 4: [3.0, 2.0, 1.0, 0.0, -1.0]
argmax: [1.0000, 0.0000, 0.0000, 0.0000, 0.0000]
Temperatur 0.5: [0.8647, 0.1170, 0.0158, 0.0021, 0.0003]
Temperatur 1.5: [0.5046, 0.2591, 0.1330, 0.0683, 0.0351]
Top-k 3, Temperatur 1: [0.6652, 0.2447, 0.0900, 0.0000, 0.0000]
Top-p 0.9, Temperatur 1: [0.6652, 0.2447, 0.0900, 0.0000, 0.0000]
```

Die Greedy-Zeile ist eine **Auswahlverteilung mit einem einzigen
Kandidaten**, nicht die ursprüngliche Softmax-Verteilung.
Durch Rundung müssen die angezeigten Werte nicht exakt eins ergeben.

## 8. Gleicher Checkpoint, gleicher Prompt

Die Demo braucht einen gespeicherten Checkpoint. Existiert
`checkpoints/ch13-best.bpk` noch nicht, erzeugen wir ihn zuerst mit dem
Trainingslauf aus Kapitel 13 (20 Epochen, speichert dabei auch den besten
Stand):

```bash
cargo run --release -p training --example ch13 -- checkpoints/ch13-best.bpk 20
```

Danach vergleichen wir fünf Auswahlstrategien für das nächste Token:
**argmax**, **Temperatur 0.5**, **Temperatur 1.5**,
**Top-k 3 bei Temperatur 1** und **Top-p 0.9 bei Temperatur 1**.
Mit „Varianten“ sind diese Strategien gemeint, nicht unterschiedliche
Modelle. Sie sind in `VARIANTS` in `crates/training/examples/ch16.rs`
festgelegt. Der folgende Befehl führt alle fünf nacheinander aus:

```bash
cargo run --release -p training --example ch16 -- checkpoints/ch13-best.bpk "I HAD always" 16 42
```

Die letzten Argumente bedeuten **maximal 16 neue Tokens**, **Seed 42**.

### Die Ausgabe lesen

Die Ausgabe beginnt mit:

```text
Gleiche Gewichte, gleicher Prompt; kein Training, keine Qualitätsgarantie.
Checkpoint: checkpoints/ch13-best.bpk
Dein Starttext: "I HAD always"
Seed: 42
Maximal neue Tokens je Strategie: 16
```

Das Modell wird einmal geladen. Jede Variante startet erneut mit
`I HAD always`; keine übernimmt die Fortsetzung der vorherigen Variante.
Gewichte, Starttext und Seed bleiben gleich; geändert wird nur die
Auswahlstrategie für das nächste Token.

Danach folgt pro Variante ein Block. **Schematisch**, nicht als konkrete
Ausgabe eines Trainingslaufs:

```text
=== argmax ===
Neue Tokens: <Anzahl>

Vom Modell erzeugte Fortsetzung:
<Fortsetzung>

Token-IDs (Starttext + Fortsetzung):
[<IDs des Prompts>, <neue IDs>]
```

Der gemeinsame Kopf nennt Checkpoint, Prompt, Seed und Tokenlimit einmal.
Jeder deutlich getrennte Block nennt die **Strategie** und die tatsächlich
erzeugte Anzahl neuer Tokens. Unter **„Vom Modell erzeugte Fortsetzung“**
steht nur der neu erzeugte Text, ohne deinen Starttext. Zum Lesen des
gesamten Ergebnisses gehört diese Fortsetzung direkt hinter `I HAD always`;
ein führendes Leerzeichen in der Fortsetzung gehört zum erzeugten Text.
Danach folgen die Token-IDs zur technischen Untersuchung.
`16` im Aufruf ist eine Obergrenze, keine
Wortanzahl: Ein Token kann ein Wort, ein Wortstück oder ein Satzzeichen
darstellen. Wird vorher das Stopp-Token erzeugt, endet die Fortsetzung
früher; das Stopp-Token zählt dabei als neues Token.

`Token-IDs` zeigt die gesamte Folge: zuerst den tokenisierten Prompt,
danach die angehängten Tokens. Die Zahlen sind **Vokabular-IDs**, keine
Wahrscheinlichkeiten oder Rangplätze. Anders als der Fortsetzungstext
enthält diese Liste weiterhin auch die IDs deines Starttexts.

Die nächste Überschrift, etwa `=== Temperatur 0.5 ===`, beginnt einen
neuen Versuch. Die fünf Blöcke sind
**keine aufeinanderfolgenden Absätze einer einzigen Fortsetzung**.
Für den ersten Vergleich kannst du die Token-IDs ignorieren und nur die
fünf Fortsetzungen lesen.

Eine beobachtete `argmax`-Fortsetzung ergibt zusammen mit dem Starttext:

```text
I HAD always.

""I to-- to I to I to I to I
```

Nur `I HAD always` stammt von uns. Punkt, zwei Zeilenumbrüche,
Anführungszeichen und der restliche Text wurden erzeugt. Die **16 neuen
Tokens** zählen auch Satzzeichen und Zeilenumbrüche mit; in der ID-Liste
stehen davor noch die vier Tokens des Prompts. Andere Checkpoints können
andere Fortsetzungen liefern.

Der unverständliche Text ist keine zusätzliche Ausgabeinformation,
sondern das tatsächliche Ergebnis unseres winzigen Lernmodells.
`argmax` wählt jeweils das höchstbewertete Token, nicht garantiert ein
sprachlich korrektes. So können Wiederholungsschleifen entstehen.
Technisch funktionierende Generierung bedeutet noch keine gute
Textqualität; hier untersuchen wir die Token-Auswahl.

Die fünf Blöcke lassen sich so vergleichen:

- **`argmax`** nimmt immer das Token mit dem höchsten Logit. Der angezeigte
  Seed hat hier keinen Einfluss.
- **`Temperatur 0.5`** zieht ohne Filter aus einer stärker konzentrierten
  Verteilung; wahrscheinliche Tokens werden stärker bevorzugt.
- **`Temperatur 1.5`** zieht ohne Filter aus einer flacheren Verteilung;
  weniger wahrscheinliche Tokens haben grössere Chancen.
- **`Top-k 3, Temperatur 1`** zieht bei jedem Schritt nur aus den drei
  wahrscheinlichsten Tokens.
- **`Top-p 0.9, Temperatur 1`** zieht bei jedem Schritt aus der kleinsten
  Kandidatenmenge, deren aufsummierte Wahrscheinlichkeit mindestens
  90 Prozent erreicht.

**Gleicher Seed bedeutet nicht gleiche Fortsetzung zwischen Strategien.**
Die Verteilungen unterscheiden sich. Sobald zwei Varianten unterschiedliche
Tokens wählen, unterscheiden sich auch ihre nächsten Eingaben und damit
die folgenden Logits. Jede Variante setzt ihren Zufallsgenerator erneut
auf Seed 42; Wiederholbarkeit prüfen wir deshalb zwischen zwei Aufrufen
**derselben Variante** in derselben deterministischen Umgebung.

Anders als die Demo ohne Argumente zeigt dieser Aufruf keine
Wahrscheinlichkeitsvektoren, sondern die daraus gewählten Tokenfolgen.
Wir vergleichen Auswahlverhalten, nicht Trainings-Loss oder Textqualität.
Eine höhere Temperatur garantiert weder kreativere noch bessere Texte.

### Selbst vergleichen

1. Lies zuerst nur die Fortsetzungen für **Temperatur 0.5** und
   **Temperatur 1.5**. Welche Wörter oder Satzteile wiederholen sich?
   Sind die Texte unterschiedlich? Aus einer einzelnen Fortsetzung
   lässt sich noch keine allgemeine Aussage über Vielfalt ableiten.
2. Führe denselben Befehl erneut mit **Seed 42** aus. Vergleiche jede
   Strategie mit ihrer eigenen vorherigen Ausgabe, nicht mit einer
   anderen Strategie. In derselben deterministischen Umgebung sollten
   die Tokenfolgen jeweils identisch bleiben.
3. Ändere nur den Seed auf **43**:

```bash
cargo run --release -p training --example ch16 -- checkpoints/ch13-best.bpk "I HAD always" 16 43
```

Welche Fortsetzungen ändern sich? **`argmax` muss unverändert bleiben**,
weil es keine Zufallszahlen verwendet. Die Sampling-Varianten können
andere Tokens wählen, müssen es aber nicht. Wenn beide Temperaturen
denselben Text erzeugen, widerlegt das ihren Einfluss nicht: Sie verändern
die Auswahlchancen, erzwingen aber keine unterschiedlichen Ergebnisse.
Die Verteilungstabelle aus Abschnitt 2 zeigt diesen Unterschied direkt.

### Wie entstehen die fünf Vergleiche?

`compare()` durchläuft die fünf Strategien aus `VARIANTS`. Jeder Aufruf
von `generate_with_options()` erhält dasselbe Modell und mit
`prompt.to_vec()` eine neue Kopie der ursprünglichen Prompt-Tokens.
Die erzeugte Folge wird nicht an die nächste Variante weitergegeben.
`report()` formatiert anschliessend das Ergebnis für die Ausgabe.

```rust
{{#include ../../../crates/training/examples/ch16.rs:compare}}
```

Die Demo erwartet GPT-2-Token-IDs, Kontextlänge acht, Embedding-Dimension
acht, zwei Köpfe und einen Transformer-Block. Checkpoints aus Kapitel 10
mit fünf IDs sind nicht kompatibel. Kapitel 15 speichert ein anderes
Checkpoint-Verzeichnisformat; dessen Ordner ist ebenfalls kein gültiges
Argument für diese Demo.

Ein fehlender Checkpoint führt zu einem Fehler; es gibt keinen Rückfall
auf Zufallsgewichte. Der Vergleich verändert oder ersetzt keinen Checkpoint.

Die Demo zeigt zuerst die decodierte Fortsetzung, danach Token-IDs. GPT-2-Tokens
sind Bytefolgen: Eine Fortsetzung kann mit einem unvollständigen
UTF-8-Zeichen enden oder ungültige Bytes erzeugen. Wie Kapitel 12 meldet
die Demo, dass die Fortsetzung nicht decodiert werden konnte, gibt die Token-IDs
zur Untersuchung aus und beendet den Vergleich mit einem Decodierungsfehler.
Sinnvolle Texte oder bessere Antworten sind nicht garantiert.

## 9. Was prüfen wir?

Tests mit festen Logits prüfen Temperatur, Normalisierung, Top-k,
Top-p einschliesslich des schwellenüberschreitenden Tokens,
Gleichstände und ungültige Eingaben. Viele Ziehungen mit festem Seed
prüfen die erwarteten Häufigkeiten.

Die Generierungstests vergleichen die neue Greedy-Variante mit dem
bisherigen Ablauf, prüfen Kontextbegrenzung, fortlaufenden Zufallszustand,
Tokenlimit und Stopp-Token. Ein kleiner Checkpoint-Roundtrip prüft für
alle fünf Varianten dieselben Ausgaben vor und nach dem Laden.
Diese Tests trainieren kein Sprachmodell.

```bash
cargo test -p training --lib --example ch16 --example ch11 --test ch10
```

## Das nehmen wir mit

Temperatur verändert Wahrscheinlichkeiten, Top-k und Top-p begrenzen
Kandidaten, der Seed steuert die Zufallsfolge. Gewichte und Modellaufbau
bleiben unverändert. Kapitel 17 kehrt zum Training zurück:
Regularisierung soll Overfitting begrenzen, nicht die Tokenauswahl verändern.
