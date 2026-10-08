# Exkurs: Rust-Grundlagen für den DataLoader

Dieser Exkurs stellt die Rust-Konzepte vor, die dir in Kapitel 2 begegnen. Er baut auf
dem Exkurs [Rust-Grundlagen für den Tokenizer](rust-grundlagen-tokenizer.md) auf.
Die Beispiele sind klein und werden mit `mdbook test` geprüft.

## Slices schneiden

```rust
let tokens: Vec<u32> = vec![10, 11, 12, 13, 14, 15];
let anfang: &[u32] = &tokens[..4];  // Index 0 bis 3
let ende: &[u32] = &tokens[4..];    // Index 4 bis zum Schluss

assert_eq!(anfang, [10, 11, 12, 13]);
assert_eq!(ende, [14, 15]);
```

- `a..b` ist ein Bereich von `a` bis **vor** `b`. Fehlt eine Grenze, geht er bis zum Rand.
- Ein Slice ist eine Sicht auf dieselben Daten, es wird nichts kopiert.
- Mehrere Sichten gleichzeitig sind erlaubt, solange niemand schreibt. Genau so teilen
  wir den Token-Strom in Training und Validierung, ohne ihn zu duplizieren.
- Ein Bereich ausserhalb der Daten bricht das Programm ab (`panic`). Das ist ein
  Programmierfehler, kein Normalfall.

## Iteratoren auf Slices

Ein Iterator liefert Elemente nacheinander. Auf Slices gibt es fertige Bausteine zum
Zerschneiden:

```rust
let tokens = [1, 2, 3, 4, 5, 6, 7];

// lückenlose Stücke fester Länge, ein Rest bleibt übrig
let stuecke: Vec<&[i32]> = tokens.chunks_exact(3).collect();
assert_eq!(stuecke, [&[1, 2, 3][..], &[4, 5, 6][..]]);

// überlappende Fenster, Schritt 1
let fenster: Vec<&[i32]> = tokens.windows(3).take(2).collect();
assert_eq!(fenster, [&[1, 2, 3][..], &[2, 3, 4][..]]);

// Elemente umwandeln und zusammenführen
let doppelt: Vec<i32> = tokens.iter().map(|x| x * 2).collect();
assert_eq!(doppelt, [2, 4, 6, 8, 10, 12, 14]);
```

- `chunks_exact(n)` schneidet lückenlos und lässt einen kürzeren Rest weg. Das ist
  unser Fenster aus Kapitel 2.
- `windows(n)` überlappt. Wir nutzen es nicht, es zeigt nur den Unterschied.
- `map` wandelt jedes Element um, `collect` sammelt das Ergebnis in einen `Vec`.
- Der Typ hinter `collect` muss bekannt sein, deshalb steht `Vec<…>` an der Variablen.
- In Kotlin entsprechen `chunked` und `windowed` diesen Bausteinen.

## Veränderbarkeit

```rust
let mut zahlen = vec![3, 1, 2];
zahlen.sort();
assert_eq!(zahlen, [1, 2, 3]);
```

- `let` ist standardmässig **unveränderlich**. Erst `let mut` erlaubt Änderungen.
- Eine Methode, die ihr Objekt verändert, nimmt `&mut self` oder `&mut` als Parameter.
  Beim Mischen übergeben wir deshalb `&mut`.
- Die Regel des Compilers: Entweder beliebig viele lesende Verweise **oder** ein
  einziger schreibender. Das verhindert Fehler durch gleichzeitige Änderung schon beim
  Kompilieren.

## Zahlen umwandeln

```rust
let laenge: usize = 1000;
let anteil: f64 = 0.9;
let train = (laenge as f64 * anteil) as usize;
assert_eq!(train, 900);

let id: u32 = 50_256;
let index = id as usize;
assert_eq!(index, 50_256);
```

- Rust wandelt Zahlentypen **nie stillschweigend** um. Man schreibt `as`.
- `usize` für Längen und Indizes, `u32` für Token-IDs, `f64` für Kommazahlen.
- `f64 as usize` schneidet die Nachkommastellen ab. Das reicht für den 90/10-Split.
- Vorsicht: `as` kann Werte abschneiden, wenn das Ziel zu klein ist. Bei unseren
  Grössen passiert das nicht.

## `derive` für eigene Typen

```rust
#[derive(Debug, Clone, PartialEq)]
struct Batch {
    eingabe: Vec<u32>,
    ziel: Vec<u32>,
}

let a = Batch { eingabe: vec![1, 2], ziel: vec![2, 3] };
let b = a.clone();
assert_eq!(a, b);
```

- `Debug` erlaubt die Ausgabe mit `{:?}`.
- `Clone` erlaubt `.clone()`, eine echte Kopie.
- `PartialEq` erlaubt `==` und damit `assert_eq!` in Tests.
- Ohne `derive` müsste man diese Eigenschaften selbst schreiben.

## `Option`, `Result` und `panic`

```rust
let leer: Vec<u32> = vec![];
assert_eq!(leer.first(), None);        // kein Element: None
assert_eq!([7, 8].first(), Some(&7));  // Element vorhanden: Some
```

- `Option<T>` heisst "kann fehlen": `Some(wert)` oder `None`. Rust hat kein `null`.
- Für einen Fehler mit Ursache gibt es `Result` (siehe Tokenizer-Exkurs).
- Faustregel: Ein **Programmierfehler** (falscher Parameter, der nie vorkommen darf)
  bricht mit `assert!` oder `panic!` ab. Ein **erwartbarer Fehler** (Datei fehlt, Text
  zu kurz) wird als `Result` zurückgegeben, damit der Aufrufer reagieren kann.

## Zufall mit festem Seed

Zufallszahlen liefert das Crate `rand`. Mit einem festen Seed ist die Folge
wiederholbar:

```rust,ignore
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

let mut rng = StdRng::seed_from_u64(42);
let mut fenster = vec![0, 1, 2, 3, 4];
fenster.shuffle(&mut rng);
```

- `seed_from_u64(42)` erzeugt einen Zufallsgenerator mit festem Startwert. Gleicher
  Seed ergibt dieselbe gemischte Reihenfolge.
- `shuffle` verändert den `Vec` an Ort und Stelle, deshalb `let mut` und `&mut rng`.
- **Häufige Falle:** `shuffle` ist keine Methode des Slices selbst, sondern gehört dem
  Trait `SliceRandom`. Ohne `use rand::seq::SliceRandom;` meldet der Compiler, die
  Methode existiere nicht. Methoden eines Traits sind nur sichtbar, wenn der Trait im
  Scope ist.
- Das Beispiel ist als `ignore` markiert, weil `mdbook test` keine externen Crates kennt.
  Der echte Code läuft im `training`-Crate.

## Grenzen

Lifetimes (`&'a`) werden hier nicht erklärt. Wo möglich geben wir Besitz zurück (`Vec`
statt Slice), dann braucht der Code keine. Ein eigener Exkurs folgt, sobald uns ein
Fall im Code zwingt.
