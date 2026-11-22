# Browser-Präsentation mit Slidev

Eine gemeinsame Slidev-Installation versorgt mehrere unabhängige Präsentationen.
Jedes Deck bekommt einen eigenen Ordner und einen eigenen Build-Ausgabeordner.
Dieses Node.js-Projekt ist unabhängig vom Cargo-Workspace und von der
mdBook-Veröffentlichung.

## Struktur

```text
presentation/
  package.json
  package-lock.json
  templates/
    slides.md
  decks/
    llm-grundlagen/
      slides.md
      public/
        README.md
  dist/                    # generiert, nicht versioniert
    llm-grundlagen/
```

Erstes Deck: [LLM-Grundlagen](decks/llm-grundlagen/slides.md).
Die ausführliche Gliederung bleibt in
[`../Praesentation.md`](../Praesentation.md).

## Voraussetzungen und Start

Node.js **ab 22.22.0**, empfohlen Node.js 24 LTS, und npm.
Mit vorhandenem nvm kann die Version über `nvm use` ausgewählt werden.

```bash
cd presentation
npm ci
npm run dev -- decks/llm-grundlagen/slides.md
```

Die Folien sind unter <http://localhost:3030> erreichbar.
Die Referentenansicht mit Sprechernotizen öffnet sich unter
<http://localhost:3030/presenter/1>.
Der Server bleibt lokal; für einen Online-Vortrag kann das Browserfenster
über die Videokonferenz geteilt werden.

## Folien bearbeiten

`---` trennt Folien; ein YAML-Block am Anfang konfiguriert das Deck.
HTML-Kommentare am Ende jeder Folie enthalten die Sprechernotizen.
Der Entwicklungsserver aktualisiert die Vorschau bei Änderungen.
Bilder und andere statische Assets gehören in den `public/`-Ordner des
jeweiligen Decks, nicht in einen gemeinsamen Asset-Ordner.

Die Basis enthält neun Inhaltsfolien und eine Fragenfolie:
18 Minuten Vortrag plus 2 Minuten Fragen. Das Next-Token-Beispiel ist
ausdrücklich illustrativ; die Loss-Werte stammen aus dem dokumentierten
Lauf in Kapitel 13. Eine eigene Generierungsdemo ist noch nicht aufgezeichnet.

## Statische HTML-Website bauen

```bash
npm run build -- decks/llm-grundlagen/slides.md --out ../../dist/llm-grundlagen
npm run preview -- --outDir dist/llm-grundlagen
```

`dist/llm-grundlagen/` enthält die komplette Website, nicht nur eine einzelne
HTML-Datei. Die lokale Build-Vorschau ist unter
<http://localhost:4173> erreichbar.
Dieser Build enthält die Sprechernotizen; sie sind kein geschützter Inhalt.

Für eine Veröffentlichung ohne Sprechernotizen:

```bash
npm run build:public -- decks/llm-grundlagen/slides.md --out ../../dist/llm-grundlagen
```

Bei Hosting unter einem Unterpfad muss der Basispfad passen, zum Beispiel:

```bash
npm run build:public -- decks/llm-grundlagen/slides.md \
  --out ../../dist/llm-grundlagen \
  --base /rusty-mind-llm/presentation/llm-grundlagen/
```

Der Basispfad muss mit `/` beginnen und enden. Der Webserver muss bei
direkten Aufrufen von Folienrouten die `index.html` des Decks ausliefern
(SPA-Fallback).

Noch keine automatische Veröffentlichung eingerichtet. Insbesondere darf
ein separater Pages-Deploy nicht die bestehende Buchseite überschreiben:
Für gemeinsames GitHub-Pages-Hosting müssten Buch und Folien in einem
gemeinsamen Deployment veröffentlicht werden.

## Weitere Präsentation hinzufügen

Kurzen, eindeutigen Ordnernamen wählen und die Vorlage kopieren:

```bash
mkdir -p decks/mein-thema/public
cp templates/slides.md decks/mein-thema/slides.md
npm run dev -- decks/mein-thema/slides.md
```

Titel, Inhalte und Sprechernotizen in der neuen Datei anpassen.
Das neue Deck nutzt dieselben Abhängigkeiten; keine weitere Installation nötig.

```bash
npm run build:public -- decks/mein-thema/slides.md --out ../../dist/mein-thema
```

Für Hosting unter einem Unterpfad zusätzlich dessen `--base` angeben.
Slidev löst `--out` relativ zum Ordner der jeweiligen `slides.md` auf;
`../../dist/mein-thema` liegt deshalb im gemeinsamen `presentation/dist/`.
Decks werden einzeln gebaut; ein Build überschreibt nicht die Ausgabe anderer
Decks, solange jedes seinen eigenen `--out`-Ordner verwendet.
