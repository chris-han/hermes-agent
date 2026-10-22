---
name: nihaixia
description: |
  Ni Haixia classical Chinese medicine knowledge skill. Use for Chinese-language
  questions that explicitly ask for 倪海厦, 倪师, 海厦视角, 经方思维, 六经辨证,
  伤寒论, 金匮要略, 黄帝内经, 针灸, 神农本草经, or related source-backed
  reference from the bundled corpus.
metadata:
  semantier:
    route: procedural_only
    source_plugin: nihaixia
---
# Ni Haixia Knowledge Skill

Use this skill when the user asks for Ni Haixia-style classical Chinese medicine
reference, source-backed corpus lookup, or a summary of the bundled 倪海厦
materials.

This plugin is knowledge-only. It does not register deterministic runtime tools.
Use the bundled Markdown corpus under `source/`:

- `source/UPSTREAM_SKILL.md` for the original activation guide, keyword index,
  quick answers, and high-level retrieval map.
- `source/modules/` for major topic modules such as 伤寒论, 金匮要略, 医案,
  黄帝内经, 针灸, 本草, 梁冬对话, and closed-course material.
- `source/cases/` for disease-grouped clinical case references.
- `source/references/research/` for source, timeline, expression, and
  methodology notes.
- `source/distilled_cases.md` and `source/expression_style.md` for distilled
  case and response-style material.

For corpus lookup, first identify the likely source file from
`source/UPSTREAM_SKILL.md`, then inspect the relevant source file before
answering. Do not answer from memory when the corpus can be checked. If the
bundled corpus does not cover the topic, say that the Ni Haixia corpus included
in this plugin does not cover it.

Health and safety rules:

- Present content as historical, educational, or corpus-derived reference,
  not as a diagnosis, prescription, or substitute for care from a qualified clinician.
- Do not provide personalized dosing, treatment instructions, pregnancy advice,
  emergency care instructions, or medication-herb interaction decisions as final
  medical guidance.
- For severe symptoms, cancer, cardiovascular symptoms, neurologic symptoms,
  pregnancy, children, poisoning, medication interactions, or worsening illness,
  tell the user to seek qualified medical care promptly.
- Keep source attribution clear when summarizing specific formulas, cases, or
  claims from the bundled corpus.

Do not use ad hoc internet searches, shell scripts, generated Python, prompt
memory, or user self-claims as substitutes for the bundled corpus when the user
asks for the Ni Haixia skill content. If this plugin is not installed or the
corpus files are unavailable, stop and report that the `nihaixia` plugin must be
installed or repaired in the active workspace.
