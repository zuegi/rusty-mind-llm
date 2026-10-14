# Burn: Die Bausteine

## Was ist Burn?

Burn ist ein Framework für neuronale Netze in Rust. Es liefert Tensoren,
Rechenoperationen und Modellschichten wie `Embedding`. Wir müssen das
Nachschlagen von Tabellenzeilen oder das Addieren von Tensoren deshalb nicht
selbst programmieren.

Burn nimmt uns die Tensor- und Ableitungsrechnung ab, nicht den Entwurf des
Modells. Welche Tabellen wir brauchen, welche Formen zusammenpassen und wie
Daten ins Modell gelangen, legen wir weiterhin selbst fest.

## Tensoren: Zahlen mit Form

Ein Tensor speichert Zahlen und ihre Form. Unser Eingabetensor enthält
Token-IDs mit der Form `[batch_size, context_length]`. Nach dem Embedding
enthält er Gleitkommazahlen mit der Form
`[batch_size, context_length, embedding_dim]`.

Operationen wie `reshape` oder `+` arbeiten auf diesen Tensoren. Burn übernimmt
die Ausführung, aber wir müssen passende Formen und Datentypen vorgeben.

## Module: Berechnung und Parameter

Ein Modul bündelt einen Teil des Modells. Burns `Embedding` enthält eine
lernbare Tabelle und eine Methode `forward`, die Tabellenzeilen nachschlägt.
Unser `GptEmbedding` setzt zwei solche Module zusammen.

Die Tabellen sind als `Param` gekennzeichnet. Dadurch kann Burn sie als
Modellparameter verwalten. `#[derive(Module)]` erzeugt die dafür nötige
Implementierung für unsere zusammengesetzte Rust-Struktur.

## Backends: Wo gerechnet wird

Das Backend führt die Tensoroperationen aus, etwa auf der CPU oder einer GPU.
Es ist die Rechenumgebung, nicht das Sprachmodell selbst.

Seit Burn 0.22 wählen wir das Backend zur Laufzeit über `Device`.
`Device::ndarray()` wählt unser CPU-Backend mit standardmässig
32-Bit-Gleitkommazahlen. `Tensor<3>` trägt nur Rang und Zahlenart im Rust-Typ;
sein Device bestimmt, wo gerechnet wird. Modelle benötigen keinen Backend-Typparameter.

Die Demo braucht keine GPU. NdArray ist in Burn 0.22 deprecated; für neue
reine Rust-CPU-Projekte empfiehlt Burn `Device::flex()` mit Feature `flex`.
Wir behalten NdArray hier bei, um API-Migration und Backend-Wechsel zu trennen.
Andere Backends brauchen passende Cargo-Features und Geräte.

## Autodiff: Grundlage für späteres Training

Mit `Device::ndarray().autodiff()` aktivieren wir automatische Ableitungen,
bevor wir Modell und Eingaben auf diesem Device erzeugen. Ein Gradient
beschreibt, wie sich ein Fehlerwert bei Änderungen der Modellparameter verändert.
Ein Optimierer kann diese Gradienten später verwenden, um Gewichte anzupassen.

Die Schritte sind verschieden:

1. Die Vorwärtsberechnung erzeugt eine Ausgabe.
2. Eine Verlustfunktion berechnet daraus und aus dem Ziel einen Fehlerwert.
3. Die Rückwärtsberechnung ermittelt Gradienten.
4. Ein Optimierer aktualisiert die Gewichte.

`forward` allein verändert keine Gewichte und ist noch kein Training. In
Kapitel 3 verwenden wir Autodiff nur in einem Test, um die Lernbarkeit beider
Embedding-Tabellen zu prüfen. Eine Trainingsschleife bleibt Aufgabe des
separaten Trainingsbereichs.

## Unsere Aufgabenteilung

Der Tokenizer und der DataLoader bereiten Text und Token-IDs auf.
`crates/model` enthält die Burn-Modellberechnung. Der spätere Trainingsbereich
verbindet Daten, Modell, Verlustfunktion und Optimierer.

Rust-Syntax zu den Typen und Methoden aus Kapitel 3 erklärt der Exkurs
[Rust-Grundlagen für das Embedding](rust-grundlagen-embedding.md).
