# Trainingstext

Hier liegt der Text für die Demos und das Training: `the-verdict.txt`.

## Quelle

Vorgesehen ist "The Verdict" (Edith Wharton, 1908), bereitgestellt als Kaggle-Datensatz:
<https://www.kaggle.com/datasets/golammostofas/the-verdict>

Laut Kaggle-Metadaten steht der Datensatz unter Apache 2.0 (Autor: Md. Golam Mostofa).

Der Text selbst ist gemeinfrei: Edith Wharton starb 1937, "The Verdict" erschien 1908.
Er ist auch bei Project Gutenberg (<https://www.gutenberg.org/ebooks/306>) und Wikisource
(<https://en.wikisource.org/wiki/The_Verdict>) verfügbar. Der Projektbesitzer kennt ihn
zudem aus Sebastian Raschkas Buch "Build a Large Language Model (From Scratch)". Die
Datei im Datensatz kann vom Original abweichen; die Rechtslage wurde nicht juristisch
geprüft. Der Text ist wegen der Gemeinfreiheit im Repository enthalten.

## Verwendung

```bash
cargo run -p training --example ch02 -- data/the-verdict.txt
```
