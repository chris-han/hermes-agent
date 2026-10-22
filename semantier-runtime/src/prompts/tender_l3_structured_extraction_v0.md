Extract only facts explicitly stated in the supplied tender document into the provided JSON Schema.

Rules:
- This is source observation, not legal judgment or compliance qualification.
- Do not infer unstated facts.
- Do not convert absence into a positive assertion.
- Preserve the document's business wording.
- For a requested qualification/requirement field, return the minimal operative eligibility predicate only. Exclude subsequent proof/document instructions introduced by terms such as 提供, 需提供, 须提供, 应提供, unless the schema field explicitly asks for evidence requirements.
- Use determination="FOUND" only for explicitly supported requirements.
- Use determination="UNKNOWN" only when the document explicitly presents an unresolved/indeterminate state relevant to the field; do not use UNKNOWN as a synonym for missing.
- Do not invent project identity, requirements, evidence requirements, conditions, or values.
- Return only schema-valid JSON.
