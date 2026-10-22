def validate_ontology(J_t, ontology):
    entity = J_t.get("entity")
    if entity and entity not in ontology.get("entities", []):
        return {"status": "REJECT", "reason": "unknown_entity"}
    return {"status": "ALLOW"}
