---
name: tender_sensitive_term_detection_modeling
description: Use governed Semantier core services to compile sensitive lexicon DOCX sources, review candidates, label tender documents, generate lineage reports, generate labeled DOCX derivatives, and verify replay evidence.
plugin: tender_sensitive_term_detection_modeling
---

# Tender Sensitive Term Detection Modeling

Use the registered `tender_sensitive_term_detection_modeling` tools for tender-sensitive labeling workflows:

- `compile_sensitive_lexicon`
- `review_sensitive_lexicon_candidates`
- `label_tender_document`
- `generate_sensitive_labeling_report`
- `generate_labeled_docx`
- `verify_sensitive_labeling_replay`

The plugin relies on authenticated Semantier runtime context and governed core services for identity, tenancy, knowledge-source authority, candidate lifecycle, rule activation, snapshot resolution, replay, audit, and artifacts. It must not use user claims, prompt memory, unmanaged host paths, local plugin state, raw database access, or marketplace trust as authority.

Missing governed context or unsupported plugin/core contract versions must fail closed.
