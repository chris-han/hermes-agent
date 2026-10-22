from agents.gateway import execute


class SemantierAdapter:
    def execute(self, payload: dict):
        return execute(payload)
