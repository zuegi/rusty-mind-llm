# 1. Tokenisierung

## Das Problem

Ein neuronales Netz rechnet mit Zahlen, nicht mit Text. Bevor ein Sprachmodell
irgendetwas lernen kann, brauchen wir eine feste Abbildung:

```text
Text  ⇄  Folge ganzer Zahlen
```

Die Zahlen heissen *Token-IDs*, ein einzelnes Stück Text, das eine ID bekommt, heisst
*Token*. Die Abbildung muss in beide Richtungen funktionieren: Wir wandeln Text in
IDs um (*encode*), und die Ausgabe des Modells wieder in Text (*decode*).

## Drei Ansätze

| Ansatz     | Token sind …       | Vorteil                         | Nachteil                                        |
| ---------- | ------------------ | ------------------------------- | ----------------------------------------------- |
| Zeichen    | einzelne Zeichen   | winziges Vokabular              | sehr lange Folgen, das Modell muss Wörter lernen |
| Wörter     | ganze Wörter       | kurze Folgen                    | riesiges Vokabular, unbekannte Wörter scheitern |
| Subwörter  | Wort(teil)stücke   | Kompromiss aus beiden           | Vokabular muss zuerst gebaut werden             |

GPT-2 verwendet Subwörter. Häufige Wörter wie ` the` sind ein einziges Token,
seltene Wörter werden in Stücke zerlegt, und jeder beliebige Text lässt sich
darstellen.

## Byte Pair Encoding (BPE)

BPE baut das Vokabular aus Daten auf. Die Idee ist einfach:

1. Starte mit den kleinsten Einheiten als Vokabular.
2. Suche das häufigste benachbarte Paar.
3. Verschmilz es zu einem neuen Token und nimm es ins Vokabular auf.
4. Wiederhole, bis das Vokabular die gewünschte Grösse hat.

### Beispiel von Hand

Text: `abcabcabd`, Start: jedes Zeichen ist ein Token (9 Token).

| Schritt | Häufigstes Paar     | Neues Token | Ergebnis        | Länge |
| ------- | ------------------- | ----------- | --------------- | ----- |
| 0       | –                   | –           | `a b c a b c a b d` | 9     |
| 1       | `a b` (3×)          | `X = ab`    | `X c X c X d`   | 6     |
| 2       | `X c` (2×, Gleichstand mit `c X`, wir nehmen das erste) | `Y = Xc` | `Y Y X d` | 4 |

Aus 9 Token wurden 4. Das Vokabular ist um zwei Einträge gewachsen (`X`, `Y`).
Genau dieses Prinzip, nur mit sehr viel mehr Daten, hat das GPT-2-Vokabular erzeugt.

### Byte-Level-BPE bei GPT-2

GPT-2 startet nicht bei Zeichen, sondern bei den **256 möglichen Bytes**. Dadurch
kann der Tokenizer jeden Text darstellen, auch Umlaute, Emojis oder fremde Schriften.
Es gibt keine "unbekannten" Zeichen.

Das Vokabular besteht aus:

- 256 Bytes als Grundbausteine
- 50 000 gelernten Verschmelzungen
- 1 Sonderzeichen `<|endoftext|>`

Zusammen sind das **50 257 Token**. Diese Zahl taucht später im Modell wieder auf,
als Grösse der Embedding-Tabelle und der Ausgabeschicht.

> **Hinweis:** Das Vokabular wird in diesem Buch nicht selbst trainiert. Wir verwenden
> das fertige GPT-2-Vokabular. Das Prinzip kennst du nun, das reicht für alles Weitere.

> **Rust-Grundlagen:** Für den Code in diesem Kapitel siehe den Exkurs
> [Rust: Grundlagen für den Tokenizer](../exkurse/rust-grundlagen-tokenizer.md).

## Der Tokenizer in Rust

Wir kapseln den GPT-2-Tokenizer (`r50k_base`) aus der Bibliothek `tiktoken-rs` in einem
eigenen Typ. Der Rest des Projekts kennt nur diesen Typ, die Bibliothek bleibt austauschbar.
Der Code liegt in `crates/training/src/tokenizer.rs`. Eine Erklärung Zeile für Zeile steht
im Exkurs [Rust: Der Tokenizer-Code im Detail](../exkurse/rust-tokenizer-code.md).

Der Typ besitzt nur einen Verweis auf den geladenen Tokenizer:

```rust,ignore
{{#include ../../../crates/training/src/tokenizer.rs:struct}}
```

Der Konstruktor holt die einmal geladene Instanz. Das Vokabular ist in `tiktoken-rs`
eingebettet, es wird nichts heruntergeladen:

```rust,ignore
{{#include ../../../crates/training/src/tokenizer.rs:new}}
```

`encode` wandelt Text in IDs um, `decode` zurück. Nur `decode` kann scheitern und gibt ein
`Result` zurück:

```rust,ignore
{{#include ../../../crates/training/src/tokenizer.rs:encode_decode}}
```

`pieces` liefert den Text jedes einzelnen Tokens. Das ist für die Demo und zum Verstehen
gedacht, das Modell braucht nur die IDs:

```rust,ignore
{{#include ../../../crates/training/src/tokenizer.rs:pieces}}
```

## Demo

Die Demo `ch01` tokenisiert einige Beispieltexte und zeigt IDs, Stücke und Längen:

```rust,ignore
{{#include ../../../crates/training/examples/ch01.rs}}
```

Ausführen im Projektstamm:

```bash
cargo run -p training --example ch01
```

Ausgabe:

```text
{{#include ../ausgaben/ch01.txt}}
```

### Was die Ausgabe zeigt

- **Häufige Wörter sind ein Token.** `Hello world` ergibt zwei Token. Das Leerzeichen
  gehört zum folgenden Wort (`" world"`).
- **Seltene Wörter zerfallen.** `Antidisestablishmentarianism` (28 Zeichen) wird in 5 Stücke
  zerlegt, die jeweils im Vokabular stehen.
- **Umlaute brauchen mehrere Token.** `ä` und `ö` bestehen in UTF-8 aus zwei Bytes. In
  `"Grüezi mit Umlauten: äöü"` sind das 24 Zeichen, aber 28 Bytes und 14 Token. Einzelne
  Token enthalten nur ein halbes Zeichen und erscheinen als `�`. Das ist die Folge des
  byte-basierten Vokabulars: Nichts ist unbekannt, aber deutscher Text braucht mehr
  Token als englischer.
- **Sondertoken.** `<|endoftext|>` ergibt eine einzige ID, `50256`.
- **Roundtrip.** `decode(encode(text))` ergibt immer den Originaltext.

## Test

Der wichtigste Test ist der Roundtrip, auch mit Umlauten, Emoji und leerem Text:

```rust,ignore
{{#include ../../../crates/training/src/tokenizer.rs:test_roundtrip}}
```

Weitere Tests prüfen bekannte GPT-2-IDs, das Sondertoken, den Wertebereich der IDs und
den Fehlerfall bei halbem UTF-8-Zeichen. Ausführen:

```bash
cargo test -p training
```

## Grenzen

- Der Tokenizer wird nicht trainiert, wir verwenden das fertige GPT-2-Vokabular.
- Die Ausgabe in `ausgaben/ch01.txt` ist ein Schnappschuss der Demo. Ändert sich die Demo,
  wird sie neu erzeugt mit `cargo run -q -p training --example ch01 > book/src/ausgaben/ch01.txt`.
- Deutscher Text wird mit diesem englisch geprägten Vokabular weniger effizient zerlegt.
  Für unseren Prototyp ist das ohne Bedeutung.

## Ausblick

Wir können nun Text in IDs umwandeln und zurück. Im nächsten Kapitel schneiden wir die
Token-Folge in Trainingsbeispiele: Eingabe und die jeweils um eins verschobene Zielfolge.
