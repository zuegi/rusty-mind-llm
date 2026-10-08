# Trainingstext

Hier liegt der Text für die Demos und das Training. Textdateien in diesem Ordner werden
nicht eingecheckt (siehe `.gitignore`).

## Quelle

Vorgesehen ist "The Verdict" (Edith Wharton, 1908), bereitgestellt als Kaggle-Datensatz:
<https://www.kaggle.com/datasets/golammostofas/the-verdict>

Laut Kaggle-Metadaten steht der Datensatz unter Apache 2.0 (Autor: Md. Golam Mostofa).

Der Text selbst ist gemeinfrei: Edith Wharton starb 1937, "The Verdict" erschien 1908.
Er ist auch bei Project Gutenberg (<https://www.gutenberg.org/ebooks/306>) und Wikisource
(<https://en.wikisource.org/wiki/The_Verdict>) verfügbar. Der Projektbesitzer kennt ihn
zudem aus Sebastian Raschkas Buch "Build a Large Language Model (From Scratch)". Die
Datei im Datensatz kann vom Original abweichen; die Rechtslage wurde nicht juristisch
geprüft. Der Text liegt vorerst nicht im Repository.

## Verwendung

1. Datensatz von Kaggle herunterladen.
2. Die Textdatei als `data/the-verdict.txt` ablegen.
3. Pfad an die Demos übergeben:

```bash
cargo run -p training --example ch02 -- data/the-verdict.txt
```
