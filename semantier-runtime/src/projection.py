def project(intent):
    return {
        "event_type": "SEMANTIER_EXECUTION",
        "payload": {
            "action": intent.action,
            "justification_hash": intent.justification_hash,
        },
    }