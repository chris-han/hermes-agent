from eos.ontology_validator import validate_ontology
from eos.policy_projection import validate_policy_against_ontology


def enforce_semantic_authority(J_t, ontology, policy):
    v_ont = validate_ontology(J_t, ontology)
    if v_ont["status"] != "ALLOW":
        return v_ont

    v_pol = validate_policy_against_ontology(policy, ontology)
    if v_pol["status"] != "ALLOW":
        return v_pol

    return {"status": "ALLOW"}
