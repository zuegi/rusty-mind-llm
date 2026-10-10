# Rust: Grundlagen für das Embedding

Zum Nachschlagen bekannter Begriffe hilft der Exkurs
[Rust im Vergleich zu Kotlin und Java](rust-kotlin-java.md).

## Eine eigene Crate

Eine Crate ist eine Rust-Bibliothek oder ein ausführbares Programm. Unser
Workspace enthält jetzt zwei Bibliotheken: `training` bereitet Daten auf,
`model` enthält die Modellberechnung. In `crates/model/src/lib.rs` macht
`pub mod embedding;` das Embedding-Modul von aussen erreichbar.

Die Demo importiert es mit:

```rust,ignore
use model::embedding::{GptEmbedding, GptEmbeddingConfig};
```

## Konfiguration und Modul

Eine `struct` bündelt zusammengehörige Werte. `GptEmbeddingConfig` enthält
drei Grössen vom Typ `usize`, Rusts Typ für Grössen und Indizes.

Ein `impl`-Block definiert Funktionen für einen Typ. `new` gehört zum Typ
und erstellt einen Wert. `init` wird auf diesem Wert aufgerufen und erstellt
das eigentliche Modell:

```rust,ignore
let config = GptEmbeddingConfig::new(5, 3, 4);
let embedding = config.init::<Cpu>(&device);
```

`&device` leiht das Gerät aus, statt es zu verschieben. `Cpu` ist in der Demo
ein Typalias für `NdArray<f32>`.

## Generische Typen

`GptEmbedding<B: Backend>` ist nicht auf eine bestimmte Rechenumgebung festgelegt.
`B` ist ein Platzhalter für einen Typ. `Backend` ist ein *Trait*: Er beschreibt
die Fähigkeiten, die dieser Typ mitbringen muss.

Der zugehörige Gerätetyp wird als `B::Device` geschrieben. Bei unserem CPU-Backend
erstellt `Default::default()` dessen Standardgerät.

In `Tensor<B, 2, Int>` ist `B` ein Typ, `2` dagegen ein konstanter Zahlenparameter.
Rust kennt damit bereits die Anzahl der Tensorachsen. Deren konkrete Grössen
liest das Programm mit `dims()` aus.

## Ausleihen und Verschieben

Die Methode `forward(&self, token_ids: Tensor<B, 2, Int>)` leiht das Modell aus,
übernimmt aber den Eingabetensor als Wert. Danach kann der Aufrufer dieselbe
Tensorvariable nicht erneut verwenden.

Die Demo braucht den Tensor sowohl zur Anzeige der Token-Vektoren als auch für
die gesamte Berechnung. Deshalb übergibt sie zunächst `token_ids.clone()`.
Burns Tensor-Clone teilt die zugrunde liegenden Daten; er kopiert nicht sofort
den gesamten Zahlenbestand. Danach darf der Originaltensor an die vollständige
Berechnung übergeben werden.

## Form auslesen und ändern

```rust,ignore
let [batch_size, context_length] = token_ids.dims();
```

Das nennt sich *Destrukturierung*: Die beiden Einträge des Arrays werden direkt
an zwei Variablennamen gebunden.

`reshape([1, context_length])` ändert die Form eines Positionstensors. Die Anzahl
der Einträge bleibt gleich. Aus einer Liste wird eine Tabelle mit einer Zeile.

Der Bereich `0..context_length` schliesst den Endwert aus. Bei Länge drei
entstehen also `0`, `1`, `2`. Burn erwartet für `arange` einen Ganzzahlbereich
mit `i64`; deshalb steht im Code `context_length as i64`.

## Lernbare Gewichte

Burns `Embedding` enthält seine Tabelle als `Param<Tensor<B, 2>>`. `Param`
kennzeichnet den Tensor als Modellparameter. Mit `#[derive(Module)]` erzeugt
Burn den Code, der die Parameter unseres zusammengesetzten Moduls verwaltet.

In der Demo ersetzt `Param::from_tensor(...)` die Gewichte durch feste Zahlen.
Diese Ersetzung dient ausschliesslich dem nachrechenbaren Beispiel.

## Fehler sichtbar machen

`assert!` bricht bei einer verletzten Programmierannahme mit einer Meldung ab,
etwa bei einem Fenster, das nicht in die Positions-Tabelle passt.

Die Demo gibt ein `Result` zurück. Das `?` reicht mögliche Fehler beim Auslesen
der Tensordaten an `main` weiter, statt sie zu verschlucken.
