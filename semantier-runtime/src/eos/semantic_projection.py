def semantic_projection(ontology, semantic_projection_input):
    allowed_entities = semantic_projection_input.get("allowed_entities", [])
    constraints = semantic_projection_input.get("constraints", {})

    scope_entities = []
    for e in allowed_entities:
        if e not in ontology.get("entities", []):
            return {"status": "REJECT", "reason": "projection_outside_ontology"}
        scope_entities.append(e)

    return {
        "status": "ALLOW",
        "semantic_scope": {"entities": scope_entities},
        "bindings": semantic_projection_input,
        "constraint_layer": {
            "rules": [
                {"entity": e, "constraint": constraints}
                for e in scope_entities
            ]
        }
    }
