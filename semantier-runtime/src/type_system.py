import yaml

def load_type(path):
    with open(path) as f:
        return yaml.safe_load(f)

def type_check(context, justification, semantic_type):
    rules = semantic_type.get("rules", [])
    for rule in rules:
        if rule["type"] == "balance_check":
            if not context.get("balanced", False):
                return False
    return True