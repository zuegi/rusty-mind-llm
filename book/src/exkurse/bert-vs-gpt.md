# BERT vs. GPT: Bidirektionale vs. kausale Attention

In Kapitel 5 haben wir gelernt, dass unser Attention-Baustein eine **kausale Maske**
nutzt: Jedes Token darf nur sich selbst und vorherige Tokens sehen. Warum ist
das so, und warum machen andere bekannte Modelle wie **BERT** das ganz anders?

## Der grundlegende Unterschied: Die Aufgabe

Die Architektur von Transformern (Attention) ist bei beiden Modellen im Kern
dieselbe. Der entscheidende Unterschied liegt im **Ziel des Trainings**:

| Eigenschaft | GPT (Generative Pretrained Transformer) | BERT (Bidirectional Encoder Representations from Transformers) |
|---|---|---|
| **Aufgabe** | Nächstes Token vorhersagen (*Autoregression*) | Fehlende/maskierte Wörter im Satz erraten (*Masked Language Model*) |
| **Richtung** | Streng von links nach rechts (kausal) | In beide Richtungen gleichzeitig (bidirektional) |
| **Maske** | Ja ($-\infty$ für Zukunft) | Keine kausale Maske (jedes Token sieht alle) |
| **Stärke** | Text generieren, Geschichten schreiben, Code vervollständigen | Text verstehen, klassifizieren, durchsuchen, Fragen beantworten |

## Wie BERT trainiert wird (Masked Language Modeling)

BERT generiert keinen zusammenhängenden Text Wort für Wort. Stattdessen wird
ein ganzer Satz genommen und zufällig etwa 15 % der Wörter mit einem Platzhalter
ersetzt (z. B. `[MASK]`):

```text
Eingabe: "Die [MASK] am Fluss war nass."
Ziel:    Errate das fehlende Wort ("Bank").
```

Um zu erraten, was `[MASK]` bedeutet, ist Information von **beiden Seiten**
hilfreich:
- Von links: "Die"
- Von rechts: "am Fluss war nass"

Hätte BERT eine kausale Maske, könnte das Modell bei `[MASK]` nicht sehen, was
danach kommt ("am Fluss war nass"). Die Information, die für die richtige
Bedeutung ("Flussufer" statt "Geldinstitut") entscheidend ist, würde fehlen!

Deshalb nutzt BERT **bidirektionale Self-Attention**:
In der Attention-Matrix gibt es kein $-\infty$. Jede Position darf mit allen
Positionen der gesamten Sequenz interagieren.

## Warum GPT nicht bidirektional sein darf

GPT ist ein Sprachmodell zur **Textgenerierung**:

```text
Eingabe:  "Es war einmal ein"
Ausgabe:  "König"
Nächste:  "der"
Nächste:  "in"
...
```

Wenn GPT trainiert wird, lernt es: *Gegeben die bisherigen Wörter, was ist das
nächste?*

Dürfte GPT dabei in die Zukunft schauen, würde das Training zusammenbrechen:
Das Modell müsste nicht mühsam lernen, Zusammenhänge und Grammatik zu verstehen,
sondern würde einfach das bereits vorhandene nächste Token direkt aus der Zukunft
kopieren ("spicken"). Beim späteren freien Generieren von Text existiert diese
Zukunft aber noch nicht — das Modell wäre völlig unbrauchbar.

Deshalb ist die kausale Maske für GPT **zwingend erforderlich**.

## Zusammenfassung

- **Kausale Attention (GPT):** Blick nur zurück $\rightarrow$ ermöglicht das Erzeugen von neuem Text Wort für Wort.
- **Bidirektionale Attention (BERT):** Blick in beide Richtungen $\rightarrow$ maximales Kontextverständnis für Analyse, aber ungeeignet für freie Textgenerierung.

Zurück zu [5.7 Zukunft ausblenden](../kapitel/05-07-zukunft-ausblenden.md).
