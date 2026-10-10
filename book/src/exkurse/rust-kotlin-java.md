# Rust im Vergleich zu Kotlin und Java

Diese Übersicht hilft beim Nachschlagen. Die Begriffe sind Orientierungshilfen,
keine exakten Entsprechungen: Rust hat insbesondere ein anderes Modell für
Speicherverwaltung und keine Klassenvererbung.

## Typen und Methoden

| Rust | Kotlin / Java | Wichtiger Unterschied |
| ---- | ------------- | --------------------- |
| `struct` | Klasse mit Datenfeldern | Definiert die Daten; Methoden stehen in `impl`-Blöcken. Keine Klassenvererbung. |
| `impl Typ` | Methoden im Klassenkörper | Definiert Methoden und zugehörige Funktionen für einen bereits definierten Typ. |
| `trait` | Interface | Beschreibt Fähigkeiten und kann Standardimplementierungen enthalten. |
| `impl Trait for Typ` | `class Typ : Trait` / `implements Trait` | Implementiert ein Trait für einen Typ in einem separaten Block. |
| `enum` | Eher Kotlin `sealed class` als Java `enum` | Rust-Varianten können unterschiedliche Daten enthalten; einfache Varianten ähneln einem gewöhnlichen Enum. |
| `#[derive(...)]` | Automatisch erzeugte Implementierungen, etwa bei `data class` | Erzeugt nur die ausdrücklich genannten Trait-Implementierungen; ein `struct` ist nicht automatisch eine Datenklasse. |

### Beispiel: Eine Struktur mit Konstruktor

```rust
struct Config {
    vocab_size: usize,
}

impl Config {
    fn new(vocab_size: usize) -> Self {
        Self { vocab_size }
    }
}

let config = Config::new(5);
```

Vergleich in Kotlin:

```kotlin
class Config(val vocabSize: Int)

val config = Config(5)
```

`new` ist in Rust kein Schlüsselwort und kein besonderer Konstruktor. Es ist
eine übliche Benennung für eine zugehörige Funktion, die einen Wert erstellt.
`Self` bezeichnet innerhalb des `impl`-Blocks den Typ `Config`.

### Die Schreibweise Schritt für Schritt

Betrachten wir die Funktion noch einmal mit Zeilenumbrüchen:

```rust
impl Config {
    fn new(vocab_size: usize) -> Self {
        Self { vocab_size }
    }
}
```

**`impl Config`** eröffnet einen Block mit Funktionen für den bereits
definierten Typ `Config`. Die Felder stehen im `struct`, nicht hier.

**`fn new(vocab_size: usize)`** definiert eine Funktion namens `new`.
Sie erhält einen Parameter `vocab_size` vom Typ `usize`: eine nicht negative
Ganzzahl, die Rust häufig für Grössen und Indizes verwendet. Anders als eine
Methode mit `&self` benötigt diese Funktion keinen vorhandenen `Config`-Wert.

**`-> Self`** beschreibt den Rückgabetyp. Innerhalb von `impl Config` bedeutet
`Self` genau `Config`: Die Funktion liefert einen neuen `Config`-Wert.

**`Self { vocab_size }`** erstellt diesen Wert. Die geschweiften Klammern nach
dem Typnamen enthalten die Feldwerte. Weil Feld und Parameter gleich heissen,
erlaubt Rust eine Kurzschreibweise. Ausgeschrieben lautet derselbe Ausdruck:

```rust
Config {
    vocab_size: vocab_size,
}
```

Links vom Doppelpunkt steht der Feldname, rechts der Parameterwert. Das ist
keine Typangabe wie bei `vocab_size: usize` in der Funktionssignatur.

Der letzte Ausdruck im Funktionskörper hat **kein Semikolon**. Deshalb wird
sein Wert zurückgegeben, ohne ausdrücklich `return` zu schreiben. Ein Semikolon
würde diesen Ausdruck zu einer Anweisung machen; die Funktion würde damit
nicht mehr den verlangten `Config`-Wert zurückgeben.

Beim Aufruf `Config::new(5)` bezeichnet `::` die zum Typ gehörende Funktion.
Der Parameter erhält `5`, die Funktion erstellt `Config { vocab_size: 5 }`,
und `let config = ...` bindet das Ergebnis an die Variable `config`.

Im Kotlin-Beispiel oben übernimmt der Konstruktor diese Aufgabe. Rust verwendet
hier eine gewöhnliche zugehörige Funktion, die einen Strukturwert erstellt.

### Beispiel: Ein Trait als Fähigkeit

```rust
trait Named {
    fn name(&self) -> &str;
}

struct Token {
    text: String,
}

impl Named for Token {
    fn name(&self) -> &str {
        &self.text
    }
}
```

Vergleich in Kotlin:

```kotlin
interface Named {
    fun name(): String
}

class Token(val text: String) : Named {
    override fun name(): String = text
}
```

Rusts `&str` im Beispiel ist ein ausgeliehener Blick auf den vorhandenen Text,
kein neu erzeugter `String`.

## Generics

`B: Backend` ist eine Trait-Schranke: Der eingesetzte Typ `B` muss `Backend`
implementieren. Das ähnelt Kotlin `<B : Backend>` oder Java
`<B extends Backend>`, wenn `Backend` dort ein Interface ist.

Unser `GptEmbedding<B: Backend>` ist also eine Struktur mit einem generischen
Backend-Typ, kein Interface.

Rust erlaubt auch konstante Zahlenparameter. In `Tensor<B, 3>` ist `B` ein Typ,
`3` die Anzahl der Dimensionen. Kotlin und Java haben keine direkte Entsprechung
für solche Zahlenparameter in ihren Generics. Die konkreten Tensorgrössen
stehen weiterhin im Tensor, nicht in der `3`.

## Besitz und Referenzen

Kotlin und Java verwalten gewöhnliche Objekte über Referenzen und einen Garbage
Collector. Rust verwaltet Werte über Besitz (*Ownership*) und prüft das
Ausleihen (*Borrowing*) beim Kompilieren.

| Rust | Bedeutung | Vergleich / Grenze |
| ---- | --------- | ------------------ |
| `self` | Methode übernimmt den Wert | Kein gewöhnliches `this`: Der bisherige Besitzer kann einen nicht kopierbaren Wert danach nicht weiterverwenden. |
| `&self` | Methode leiht den Wert gemeinsam aus | Für normalen Feldzugriff nur lesend; spezielle Typen ermöglichen innere Veränderbarkeit. |
| `&mut self` | Methode leiht den Wert exklusiv und veränderbar aus | Währenddessen sind keine konkurrierenden Ausleihen desselben Werts erlaubt. |
| `let mut value` | Veränderbare Bindung | Anders als Kotlin `val` / `var` entscheidet dies bei gewöhnlichen Werten auch über veränderbaren Zugriff auf deren Felder. |
| `.clone()` | Explizite, vom Typ definierte Kopie | Nicht automatisch eine tiefe Kopie; Burn-Tensoren teilen zunächst ihre zugrunde liegenden Daten. |

Bei `forward(&self, token_ids: Tensor<B, 2, Int>)` wird das Modell ausgeliehen,
der Eingabetensor dagegen als Wert übernommen. Deshalb benötigt die Demo
`token_ids.clone()`, wenn sie ihn für eine weitere Berechnung behalten möchte.

## Fehlende Werte und Fehler

| Rust | Kotlin / Java | Wichtiger Unterschied |
| ---- | ------------- | --------------------- |
| `Option<T>` mit `Some` / `None` | Nullable Typ `T?` / etwa Java `Optional<T>` | Fehlender Wert ist eine ausdrückliche Variante; gewöhnliche Rust-Referenzen sind nicht nullable. |
| `Result<T, E>` mit `Ok` / `Err` | Erfolgs-/Fehlerwert statt Exception | Erwartbare Fehler stehen im Rückgabetyp und müssen behandelt oder weitergereicht werden. |
| `?` | Fehler weiterreichen | Bei `Result` gibt es im Fehlerfall frühzeitig `Err` zurück; es ist nicht Kotlin `?.` oder `?:`. |
| `assert!` | Prüfung mit Abbruch | Löst bei verletzter Annahme einen Panic aus; anders als Java-Assertions standardmässig auch in Release-Builds aktiv. |

Die Embedding-Demo reicht Fehler beim Auslesen der Tensorwerte mit `?` weiter.
Das Modell prüft dagegen ungültige Konfigurationsgrössen mit `assert!`, weil sie
Programmierfehler sind.

## Module und Crates

Ein Rust-Modul (`mod`) gliedert Code und Sichtbarkeit. Ein Pfad wie
`model::embedding::GptEmbedding` ähnelt einem qualifizierten Klassenpfad.
`use` bringt Namen in den aktuellen Gültigkeitsbereich, ähnlich einem Import.

Eine Crate ist eine Rust-Kompilationseinheit, etwa eine Bibliothek. Ein
Cargo-Paket besitzt eine `Cargo.toml` und kann eine Bibliotheks-Crate sowie
ausführbare Programme und Beispiele enthalten. Unser Workspace bündelt die
Pakete `training` und `model`, ähnlich einem Projekt mit mehreren Build-Modulen.

Die konkrete Verwendung im Embedding-Code erklärt der Exkurs
[Rust-Grundlagen für das Embedding](rust-grundlagen-embedding.md).
