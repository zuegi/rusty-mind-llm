# Trainingstext

Hier liegt der Text für die Demos und das Training. Textdateien in diesem Ordner werden
nicht eingecheckt (siehe `.gitignore`).

## Quelle

Vorgesehen ist "The Verdict" (Edith Wharton, 1908), bereitgestellt als Kaggle-Datensatz:
<https://www.kaggle.com/datasets/golammostofas/the-verdict>

Laut Kaggle-Metadaten steht der Datensatz unter Apache 2.0 (Autor: Md. Golam Mostofa).

Der Projektbesitzer hat den Text aus dem Buch "Build a Large Language Model (From
Scratch)" von Sebastian Raschka, wo er als lizenzfrei bezeichnet wird. Diese Angabe ist
hier nicht unabhängig geprüft. Solange das so bleibt, liegt der Text nicht im Repository.

## Verwendung

1. Datensatz von Kaggle herunterladen.
2. Die Textdatei als `data/the-verdict.txt` ablegen.
3. Pfad an die Demos übergeben:

```bash
cargo run -p training --example ch02 -- data/the-verdict.txt
```
