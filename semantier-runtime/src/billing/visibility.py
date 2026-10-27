"""Read-only customer projection; full replay evidence remains management-only."""

def customer_projection(value):
    if isinstance(value, list):
        return [customer_projection(row) for row in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, child in value.items():
        if key.startswith("supplier_") or key in {"native_cost", "native_currency", "postings", "fx_quotes"}:
            continue
        if key == "prices":
            child = [row for row in child if row.get("kind") == "customer"]
        result[key] = customer_projection(child)
    if result.get("schema_version") == "semantier.metering-bundle.v1":
        result.pop("sha256", None)
        result["schema_version"] = "semantier.customer-metering-projection.v1"
        result["completeness"] = "CUSTOMER_PROJECTION_NOT_REPLAY_BUNDLE"
    return result
