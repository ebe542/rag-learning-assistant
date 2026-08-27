# Architecture diagrams

The class model is split into bounded views so each diagram remains readable.
Open the SVG links for browser navigation or the PlantUML links to edit sources:

| View | Diagram | Source |
|---|---|---|
| Architecture overview | [SVG](../class-overview.svg) | [PlantUML](../class-overview.puml) |
| Ingestion, retrieval, and library | [SVG](ingestion-retrieval-library.svg) | [PlantUML](ingestion-retrieval-library.puml) |
| Grounded generation and summaries | [SVG](generation-summaries.svg) | [PlantUML](generation-summaries.puml) |
| Question banks and learning packages | [SVG](question-banks-packages.svg) | [PlantUML](question-banks-packages.puml) |
| Study sessions, review, and progress | [SVG](study-progress.svg) | [PlantUML](study-progress.puml) |
| Evaluation and CLI boundary | [SVG](evaluation-interfaces.svg) | [PlantUML](evaluation-interfaces.puml) |

All diagrams include the shared `_theme.puml`. Cross-boundary classes may occur
in more than one view when necessary for a complete runtime relationship.
Protocol implementations use dashed arrows because Python relies on structural
typing rather than explicit inheritance.

Regenerate all published SVG files after changing a source diagram:

```bash
python scripts/render_architecture_diagrams.py
```

The script uses `plantuml` from `PATH` or the PlantUML jar installed by the VS
Code PlantUML extension. A different jar can be selected with `--plantuml-jar`
or the `PLANTUML_JAR` environment variable. `_theme.puml` is an include file
and is not rendered independently.
