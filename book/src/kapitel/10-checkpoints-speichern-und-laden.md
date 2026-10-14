# 10. Checkpoints speichern und laden

## Lernziel

In Kapitel 9 haben wir ein Modell trainiert. Dabei hat es seine Gewichte
verändert: die Zahlen, in denen das Gelernte steckt. Bisher bleiben diese
Zahlen nur im Arbeitsspeicher. Sobald das Programm endet, gehen sie verloren.
Beim nächsten Start müssten wir wieder von vorn trainieren.

In diesem Kapitel lernen wir deshalb, die **Modellgewichte in einer Datei
zu speichern und später wieder zu laden**. Einen solchen gespeicherten
Zwischenstand nennen wir **Checkpoint**. Beim Laden bauen wir zunächst ein
Modell mit derselben Struktur auf und setzen seine Gewichte auf die
gespeicherten Werte.

## Warum wir speichern und wieder laden

Unser Modell besteht aus zwei Teilen: Der **Programmcode** beschreibt seinen
Aufbau und seine Berechnungen. Die **Gewichte** bestimmen, was es durch
Training gelernt hat. Wenn wir nur den Code behalten, bleibt zwar der Aufbau
erhalten, nicht aber das Gelernte. Ein neu initialisiertes Modell beginnt
wieder mit zufälligen Gewichten.

Eine Datei bewahrt die Gewichte über das Ende des Programms hinaus auf.
Das Laden macht sie in einem später gestarteten Programm wieder nutzbar.
So müssen wir ein Modell nicht jedes Mal neu trainieren, wenn wir es
verwenden möchten. Gerade bei grossen Modellen kann das Training sehr lange
dauern.

Darauf bauen wir im nächsten Kapitel zur **Textgenerierung** auf:

1. Wir trainieren ein Modell, damit es lernt, nächste Tokens vorherzusagen.
2. Wir speichern seine gelernten Gewichte und können das Trainingsprogramm beenden.
3. Später laden wir diese Gewichte in ein Modell mit demselben Aufbau.
4. Wir verwenden das geladene Modell, um Schritt für Schritt neue Tokens zu erzeugen.

Speichern und Laden verbinden also das **Lernen** mit der **späteren Nutzung**.
Die Textgenerierung selbst gehört noch nicht zu diesem Kapitel. Zunächst
stellen wir sicher, dass die Gewichte beim Speichern und Laden erhalten
bleiben.

## Was wir hier überprüfen

Am Ende kannst du Gewichte speichern, in ein neues Modell laden und prüfen,
dass beide Modelle für dieselbe Eingabe dieselben Ausgaben liefern. Unsere
kleine Demo verwendet dafür ein zufällig initialisiertes, noch untrainiertes
Modell. Damit prüfen wir das Speichern und Laden unabhängig vom Training.
Mit trainierten Gewichten funktioniert der Ablauf genauso; erst diese
enthalten jedoch das durch Training Gelernte.

Wir sichern hier nur die Gewichte, nicht den gesamten Trainingsfortschritt.
Dazu gehören beispielsweise die zuletzt abgeschlossene Epoche und der
interne Zustand des Optimizers, also des Verfahrens, das die Gewichte
aktualisiert. Ein geladenes Modell hat damit zwar seine gelernten Gewichte
wieder, das Training wird aber nicht automatisch an genau derselben Stelle
fortgesetzt. Diesen Unterschied erklären wir im weiteren Verlauf.

## 1. Was ist ein Checkpoint?

Ein Checkpoint ist ein gespeicherter Zwischenstand. Bei unserem Modell sind
das vor allem die lernbaren Tensoren:

```text
Training → Modellgewichte → Datei
                         ↓
                 neues Modell + Konfiguration
                         ↓
                   geladene Gewichte
```

Die Token-IDs werden nicht gespeichert. Ebenso wenig wird die Architektur aus
den Gewichten erraten. Beim Laden muss dieselbe `GptModelConfig` bekannt sein:

```rust
let config = GptModelConfig::new(5, 3, 8, 2, 1);
let model = config.init::<Cpu>(&device);
```

Ändert sich etwa `vocab_size` oder `embedding_dim`, passen Record und Modell
nicht mehr zusammen. Diese Konfiguration gehört deshalb in ein echtes
Trainingsprojekt in eine versionierte Konfigurationsdatei oder in Metadaten
neben den Checkpoint.

## 2. Burns Recorder

Burn stellt `Module::save_file` und `Module::load_file` bereit. Der Recorder
legt die Datei mit seiner passenden Endung an. Im Beispiel verwenden wir
`DefaultRecorder`, der die Gewichte mit voller Genauigkeit aufzeichnet:

```rust
use burn::record::DefaultRecorder;

let recorder = DefaultRecorder::new();
model.clone().save_file("checkpoints/model", &recorder)?;
let loaded = model.load_file("checkpoints/model", &recorder, &device)?;
```

`save_file` nimmt das Modell als Besitz. `clone()` lässt das ursprüngliche
Modell weiterleben. `DefaultRecorder` erzeugt dabei eine `.mpk`-Datei.
`load_file` lädt in ein bereits initialisiertes Modell;
dadurch liefert die Konfiguration die Struktur und der Record die Werte.

## 3. Kleine Checkpoint-Hilfsfunktionen

Die Demo kapselt beide Aufrufe in `training::checkpoint`:

```rust
use std::path::Path;

use burn::{
    module::Module,
    record::{DefaultRecorder, RecorderError},
    tensor::backend::Backend,
};
use model::gpt::{GptModel, GptModelConfig};

pub fn save_model<B: Backend>(
    model: &GptModel<B>,
    path: impl AsRef<Path>,
) -> Result<(), RecorderError> {
    model.clone().save_file(path.as_ref(), &DefaultRecorder::new())
}

pub fn load_model<B: Backend>(
    config: &GptModelConfig,
    path: impl AsRef<Path>,
    device: &B::Device,
) -> Result<GptModel<B>, RecorderError> {
    config
        .init(device)
        .load_file(path.as_ref(), &DefaultRecorder::new(), device)
}
```

Die Funktionen speichern keine Eingabedaten und keinen Optimizer. Der
Rückgabewert bleibt ein `Result`: Ein fehlender oder inkompatibler Checkpoint
wird nicht stillschweigend als Erfolg behandelt.

## 4. Ausführbare Demo

Die vollständige Demo in `crates/training/examples/ch10.rs` erzeugt ein
kleines Modell, berechnet Logits, speichert die Gewichte und lädt sie wieder:

```rust
let model = config.init::<Cpu>(&device);
let inputs = Tensor::<Cpu, 2, Int>::from_ints([[0, 1, 2]], &device);
let before = model.forward(inputs.clone()).into_data();

save_model(&model, &path)?;
let loaded = load_model::<Cpu>(&config, &path, &device)?;
let after = loaded.forward(inputs).into_data();

assert_eq!(before, after);
```

Der Vergleich prüft genau das Lernziel: Für dieselben Eingaben erzeugen
Original und geladenes Modell dieselben Logits. Die Demo räumt ihre temporäre `.mpk`-Datei danach wieder auf:

```bash
cargo run -p training --example ch10
```

## 5. Test und Grenzen

Der Integrationstest wiederholt denselben Vergleich:

```bash
cargo test -p training --test ch10
```

Damit ist noch kein vollständiger Trainings-Resume implementiert. Nach dem
Laden kann man mit den Gewichten weiterrechnen oder sie erneut trainieren,
aber ein neuer Optimizer startet ohne den früheren internen Zustand. Das ist
bei einfachem SGD ohne Momentum weniger sichtbar; bei Momentum oder Adam kann
es den weiteren Verlauf deutlich verändern.

Für einen vollständigen Resume müssten zusätzlich gespeichert und geladen
werden:

```text
Modell-Record
Optimizer-Record
Modell- und Optimizer-Konfiguration
Epoche und Schrittzähler
Daten-/Shuffle-Konfiguration
```

Diese Erweiterung gehört in ein späteres Trainings-Checkpoint-Format. Das
aktuelle Kapitel bleibt absichtlich bei sicher nachweisbarem Speichern und
Laden der Modellgewichte.

## Zusammenfassung

- `DefaultRecorder` speichert Burn-Modellgewichte.
- Laden braucht ein frisch initialisiertes Modell mit passender Konfiguration.
- Ein geladener Record liefert dieselben Ausgaben wie das Original.
- Der Optimizer-Zustand ist in dieser Demo nicht Bestandteil des Checkpoints.
